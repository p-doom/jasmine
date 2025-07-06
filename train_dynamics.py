from dataclasses import dataclass, field
import os

import einops
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.mesh_utils import create_device_mesh
import optax
import orbax
from orbax.checkpoint import PyTreeCheckpointer
import numpy as np
import jax
import jax.numpy as jnp
import tyro
import wandb

from genie import Genie, restore_genie_components
from models.tokenizer import TokenizerVQVAE
from models.lam import LatentActionModel
from utils.dataloader import get_dataloader
from utils.parameter_utils import count_parameters_by_component


@dataclass
class Args:
    # Experiment
    num_steps: int = 200_000
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_height: int = 90
    image_width: int = 160
    data_dir: str = "data_tfrecords/coinrun"
    # Optimization
    batch_size: int = 36
    min_lr: float = 3e-6
    max_lr: float = 3e-5
    warmup_steps: int = 5000
    # Tokenizer
    tokenizer_dim: int = 512
    latent_patch_dim: int = 32
    num_patch_latents: int = 1024
    patch_size: int = 4
    tokenizer_num_blocks: int = 8
    tokenizer_num_heads: int = 8
    tokenizer_checkpoint: str = ""
    # LAM
    lam_dim: int = 512
    latent_action_dim: int = 32
    num_latent_actions: int = 6
    lam_patch_size: int = 16
    lam_num_blocks: int = 8
    lam_num_heads: int = 8
    lam_checkpoint: str = ""
    # Dynamics
    dyna_dim: int = 512
    dyna_num_blocks: int = 12
    dyna_num_heads: int = 8
    dropout: float = 0.0
    mask_limit: float = 0.5
    # Logging
    log: bool = False
    entity: str = ""
    project: str = ""
    name: str = "train_dynamics"
    tags: list[str] = field(default_factory=lambda: ["dynamics"])
    log_interval: int = 5
    log_image_interval: int = 250
    ckpt_dir: str = ""
    log_checkpoint_interval: int = 25000
    log_gradients: bool = False


args = tyro.cli(Args)


def dynamics_loss_fn(params, state, inputs):
    """Compute masked dynamics loss"""
    outputs = state.apply_fn(
        params,
        inputs,
        training=True,
        rngs={"params": inputs["rng"], "dropout": inputs["dropout_rng"]},
    )
    mask = outputs["mask"]
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
        outputs["token_logits"], outputs["video_tokens"]
    )
    ce_loss = (mask * ce_loss).sum() / mask.sum()
    acc = outputs["token_logits"].argmax(-1) == outputs["video_tokens"]
    acc = (mask * acc).sum() / mask.sum()
    select_probs = jax.nn.softmax(outputs["token_logits"])
    metrics = dict(
        cross_entropy_loss=ce_loss,
        masked_token_accuracy=acc,
        select_logit=outputs["token_logits"].max(-1).mean(),
        select_p=select_probs.max(-1).mean(),
        entropy=jax.scipy.special.entr(select_probs).sum(-1).mean(),
    )
    return ce_loss, (outputs["recon"], metrics)


@jax.jit
def train_step(state, inputs):
    """Update state and compute metrics"""
    grad_fn = jax.value_and_grad(dynamics_loss_fn, has_aux=True, allow_int=True)
    (loss, (recon, metrics)), grads = grad_fn(state.params, state, inputs)
    state = state.apply_gradients(grads=grads)
    if args.log_gradients:
        metrics["gradients_std/"] = jax.tree.map(
            lambda x: x.std(), grads["params"]["dynamics"]
        )
    return state, loss, recon, metrics


if __name__ == "__main__":
    jax.distributed.initialize()
    num_devices = jax.device_count()
    if num_devices == 0:
        raise ValueError("No JAX devices found.")
    print(f"Running on {num_devices} devices.")

    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by "
            f"number of devices {num_devices}."
        )

    per_device_batch_size_for_init = args.batch_size // num_devices

    rng = jax.random.PRNGKey(args.seed)

    # --- Initialize model ---
    genie = Genie(
        # Tokenizer
        in_dim=args.image_channels,
        tokenizer_dim=args.tokenizer_dim,
        latent_patch_dim=args.latent_patch_dim,
        num_patch_latents=args.num_patch_latents,
        patch_size=args.patch_size,
        tokenizer_num_blocks=args.tokenizer_num_blocks,
        tokenizer_num_heads=args.tokenizer_num_heads,
        # LAM
        lam_dim=args.lam_dim,
        latent_action_dim=args.latent_action_dim,
        num_latent_actions=args.num_latent_actions,
        lam_patch_size=args.lam_patch_size,
        lam_num_blocks=args.lam_num_blocks,
        lam_num_heads=args.lam_num_heads,
        lam_co_train=not args.lam_checkpoint,
        # Dynamics
        dyna_dim=args.dyna_dim,
        dyna_num_blocks=args.dyna_num_blocks,
        dyna_num_heads=args.dyna_num_heads,
        dropout=args.dropout,
        mask_limit=args.mask_limit,
    )
    rng, _rng = jax.random.split(rng)
    image_shape = (args.image_height, args.image_width, args.image_channels)
    dummy_inputs = dict(
        videos=jnp.zeros(
            (per_device_batch_size_for_init, args.seq_len, *image_shape),
            dtype=jnp.float32,
        ),
        action=jnp.zeros(
            (per_device_batch_size_for_init, args.seq_len), dtype=jnp.float32
        ),
        mask_rng=_rng,
    )
    rng, _rng = jax.random.split(rng)
    init_params = genie.init(_rng, dummy_inputs)

    param_counts = count_parameters_by_component(init_params)

    if args.log and jax.process_index() == 0:
        wandb.init(
            entity=args.entity,
            project=args.project,
            name=args.name,
            tags=args.tags,
            group="debug",
            config=args,
        )
        wandb.config.update({"model_param_count": param_counts})

    print("Parameter counts:")
    print(param_counts)

    # --- Initialize optimizer ---
    lr_schedule = optax.warmup_cosine_decay_schedule(
        args.min_lr, args.max_lr, args.warmup_steps, args.num_steps
    )
    tx = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.9, weight_decay=1e-4)
    train_state = TrainState.create(apply_fn=genie.apply, params=init_params, tx=tx)

    device_mesh_arr = create_device_mesh((num_devices,))
    mesh = Mesh(devices=device_mesh_arr, axis_names=("data",))

    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    videos_sharding = NamedSharding(
        mesh, PartitionSpec("data", None, None, None, None)
    )
    train_state = jax.device_put(train_state, replicated_sharding)

    # --- Restore checkpoint ---
    train_state = restore_genie_components(
        train_state, replicated_sharding, dummy_inputs, rng, args
    )

    # --- TRAIN LOOP ---
    tfrecord_files = [
        os.path.join(args.data_dir, x)
        for x in os.listdir(args.data_dir)
        if x.endswith(".tfrecord")
    ]
    dataloader = get_dataloader(
        # NOTE: We deliberately pass the global batch size
        # The dataloader shards the dataset across all processes
        tfrecord_files,
        args.seq_len,
        args.batch_size,
        *image_shape,
    )
    dataloader = (jax.make_array_from_process_local_data(videos_sharding, elem) for elem in dataloader) # type: ignore
    step = 0
    while step < args.num_steps:
        for videos in dataloader:
            # --- Train step ---
            rng, _rng, _rng_dropout, _rng_mask = jax.random.split(rng, 4)

            inputs = dict(
                videos=videos,
                rng=_rng,
                dropout_rng=_rng_dropout,
                mask_rng=_rng_mask,
            )
            train_state, loss, recon, metrics = train_step(train_state, inputs)
            print(f"Step {step}, loss: {loss}")
            step += 1

            # --- Logging ---
            if args.log:
                if step % args.log_interval == 0 and jax.process_index() == 0:
                    wandb.log(
                        {
                            "loss": loss,
                            "step": step,
                            **metrics,
                        }
                    )
                if step % args.log_image_interval == 0:
                    gt_seq = inputs["videos"][0]
                    recon_seq = recon[0].clip(0, 1)
                    comparison_seq = jnp.concatenate((gt_seq, recon_seq), axis=1)
                    comparison_seq = einops.rearrange(
                        comparison_seq * 255, "t h w c -> h (t w) c"
                    )
                    if jax.process_index() == 0:
                        log_images = dict(
                            image=wandb.Image(np.asarray(gt_seq[args.seq_len - 1])),
                            recon=wandb.Image(np.asarray(recon_seq[args.seq_len - 1])),
                            true_vs_recon=wandb.Image(
                                np.asarray(comparison_seq.astype(np.uint8))
                            ),
                        )
                        wandb.log(log_images)
            if step % args.log_checkpoint_interval == 0:
                ckpt = {"model": train_state}
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(
                    os.path.join(os.getcwd(), args.ckpt_dir, f"genie_{step}"),
                    ckpt,
                    save_args=save_args,
                )
            if step >= args.num_steps:
                break
