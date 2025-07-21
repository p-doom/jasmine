from dataclasses import dataclass, field
import os

import einops
from flax.training.train_state import TrainState
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.mesh_utils import create_device_mesh
import optax
import orbax.checkpoint as ocp
import numpy as np
import dm_pix as pix
import jax
import jax.numpy as jnp
import tyro
import wandb
import grain

from genie import Genie, restore_genie_components
from utils.dataloader import get_dataloader
from utils.lr_utils import get_lr_schedule
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
    data_dir: str = ""
    save_ckpt: bool = False
    restore_ckpt: bool = False
    # Optimization
    batch_size: int = 36
    init_lr: float = 0.0
    max_lr: float = 3e-5
    decay_end: float = 0.0
    wsd_decay_steps: int = 10000 # NOTE: wsd_decay_steps will only be used when using a wsd-schedule
    warmup_steps: int = 5000
    lr_schedule : str = "wsd" # supported options: wsd, cos
    # Tokenizer
    tokenizer_dim: int = 512
    tokenizer_ffn_dim: int = 2048
    latent_patch_dim: int = 32
    num_patch_latents: int = 1024
    patch_size: int = 4
    tokenizer_num_blocks: int = 4
    tokenizer_num_heads: int = 8
    tokenizer_checkpoint: str = ""
    # LAM
    lam_dim: int = 512
    lam_ffn_dim: int = 2048
    latent_action_dim: int = 32
    num_actions: int = 6
    lam_patch_size: int = 16
    lam_num_blocks: int = 4
    lam_num_heads: int = 8
    lam_checkpoint: str = ""
    # Dynamics
    dyna_dim: int = 512
    dyna_ffn_dim: int = 2048
    dyna_num_blocks: int = 6
    dyna_num_heads: int = 8
    dropout: float = 0.0
    mask_limit: float = 0.5
    param_dtype: jnp.dtype = jnp.float32
    dtype: jnp.dtype = jnp.bfloat16
    use_flash_attention: bool = True
    use_gt_actions: bool = False
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
    log_checkpoint_keep_period: int = 20000
    log_gradients: bool = False
    wandb_id: str = ""


args = tyro.cli(Args)


def dynamics_loss_fn(params, state, inputs):
    """Compute masked dynamics loss"""
    inputs["videos"] = inputs["videos"].astype(args.dtype) / 255.0
    outputs = state.apply_fn(
        params,
        inputs,
        training=True,
        rngs={"params": inputs["rng"], "dropout": inputs["dropout_rng"]},
    )
    mask = outputs["mask"]
    outputs["token_logits"] = outputs["token_logits"].astype(jnp.float32)
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
        outputs["token_logits"], outputs["video_tokens"]
    )
    ce_loss = (mask * ce_loss).sum() / mask.sum()
    acc = outputs["token_logits"].argmax(-1) == outputs["video_tokens"]
    acc = (mask * acc).sum() / mask.sum()
    select_probs = jax.nn.softmax(outputs["token_logits"])
    gt = inputs["videos"].clip(0, 1).reshape(-1, *inputs["videos"].shape[2:])
    recon = outputs["recon"].clip(0, 1).reshape(-1, *outputs["recon"].shape[2:])
    psnr = pix.psnr(gt, recon).mean() # type: ignore
    ssim = pix.ssim(gt, recon).mean() # type: ignore
    _, index_counts_tokenizer = jnp.unique_counts(
        jnp.ravel(outputs["video_tokens"]), size=args.num_patch_latents, fill_value=0
    )
    codebook_usage_tokenizer = (index_counts_tokenizer != 0).mean()
    # FIXME (f.srambical): check whether the if-else breaks jitting
    if not args.use_gt_actions:
        _, index_counts_lam = jnp.unique_counts(
            jnp.ravel(outputs["lam_indices"]), size=args.num_actions, fill_value=0
        )
        codebook_usage_lam = (index_counts_lam != 0).mean()
    else:
        codebook_usage_lam = None
    metrics = dict(
        cross_entropy_loss=ce_loss,
        masked_token_accuracy=acc,
        select_logit=outputs["token_logits"].max(-1).mean(),
        select_p=select_probs.max(-1).mean(),
        entropy=jax.scipy.special.entr(select_probs).sum(-1).mean(),
        psnr=psnr,
        ssim=ssim,
        codebook_usage_lam=codebook_usage_lam,
        codebook_usage_tokenizer=codebook_usage_tokenizer,
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
        tokenizer_ffn_dim=args.tokenizer_ffn_dim,
        latent_patch_dim=args.latent_patch_dim,
        num_patch_latents=args.num_patch_latents,
        patch_size=args.patch_size,
        tokenizer_num_blocks=args.tokenizer_num_blocks,
        tokenizer_num_heads=args.tokenizer_num_heads,
        # LAM
        lam_dim=args.lam_dim,
        lam_ffn_dim=args.lam_ffn_dim,
        latent_action_dim=args.latent_action_dim,
        num_actions=args.num_actions,
        lam_patch_size=args.lam_patch_size,
        lam_num_blocks=args.lam_num_blocks,
        lam_num_heads=args.lam_num_heads,
        lam_co_train=not args.lam_checkpoint,
        # Dynamics
        dyna_dim=args.dyna_dim,
        dyna_ffn_dim=args.dyna_ffn_dim,
        dyna_num_blocks=args.dyna_num_blocks,
        dyna_num_heads=args.dyna_num_heads,
        dropout=args.dropout,
        mask_limit=args.mask_limit,
        param_dtype=args.param_dtype,
        dtype=args.dtype,
        use_flash_attention=args.use_flash_attention,
        use_gt_actions=args.use_gt_actions,
    )
    rng, _rng = jax.random.split(rng)
    image_shape = (args.image_height, args.image_width, args.image_channels)
    dummy_inputs = dict(
        videos=jnp.zeros(
            (per_device_batch_size_for_init, args.seq_len, *image_shape),
            dtype=args.dtype,
        ),
        action=jnp.zeros(
            (per_device_batch_size_for_init, args.seq_len), dtype=args.dtype
        ),
        mask_rng=_rng,
    )
    rng, _rng = jax.random.split(rng)
    init_params = genie.init(_rng, dummy_inputs)

    param_counts = count_parameters_by_component(init_params)

    if args.log and jax.process_index() == 0:
        wandb_init_kwargs = {
            "entity": args.entity,
            "project": args.project,
            "name": args.name,
            "tags": args.tags,
            "group": "debug",
            "config": args,
        }

        if args.wandb_id:
            wandb_init_kwargs.update(
                {
                    "id": args.wandb_id,
                    "resume": "allow",
                }
            )
        wandb.init(**wandb_init_kwargs)

        wandb.config.update({"model_param_count": param_counts})

    print("Parameter counts:")
    print(param_counts)

    # --- Initialize optimizer ---
    lr_schedule = get_lr_schedule(args.lr_schedule, 
                                  args.init_lr, 
                                  args.max_lr, 
                                  args.decay_end, 
                                  args.num_steps, 
                                  args.warmup_steps, 
                                  args.wsd_decay_steps)
    tx = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.9, weight_decay=1e-4, mu_dtype=args.dtype)
    train_state = TrainState.create(apply_fn=genie.apply, params=init_params, tx=tx)

    device_mesh_arr = create_device_mesh((num_devices,))
    mesh = Mesh(devices=device_mesh_arr, axis_names=("data",))

    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    videos_sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None, None))
    train_state = jax.device_put(train_state, replicated_sharding)

    # --- Initialize checkpoint manager ---
    step = 0
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add(
        "model_state", ocp.args.StandardSave, ocp.handlers.StandardCheckpointHandler
    )
    handler_registry.add(
        "model_state", ocp.args.StandardRestore, ocp.handlers.StandardCheckpointHandler
    )
    handler_registry.add("dataloader_state", grain.checkpoint.CheckpointSave, grain.checkpoint.CheckpointHandler)  # type: ignore
    handler_registry.add("dataloader_state", grain.checkpoint.CheckpointRestore, grain.checkpoint.CheckpointHandler)  # type: ignore

    checkpoint_options = ocp.CheckpointManagerOptions(
        save_interval_steps=args.log_checkpoint_interval,
        max_to_keep=3,
        keep_period=args.log_checkpoint_keep_period,
        step_format_fixed_length=6,
        cleanup_tmp_directories=True,
    )

    checkpoint_manager = ocp.CheckpointManager(
        args.ckpt_dir,
        options=checkpoint_options,
        handler_registry=handler_registry,
    )

    # --- Create DataLoaderIterator from dataloader ---
    array_record_files = [
        os.path.join(args.data_dir, x)
        for x in os.listdir(args.data_dir)
        if x.endswith(".array_record")
    ]
    grain_dataloader = get_dataloader(
        array_record_files,
        args.seq_len,
        # NOTE: We deliberately pass the global batch size
        # The dataloader shards the dataset across all processes
        args.batch_size,
        *image_shape,
        num_workers=8,
        prefetch_buffer_size=1,
        seed=args.seed,
    )
    initial_state = grain_dataloader._create_initial_state()
    grain_iterator = grain.DataLoaderIterator(grain_dataloader, initial_state)

    # --- Restore checkpoint ---
    if args.restore_ckpt:
        # Restore full dynamics model
        abstract_train_state = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct, train_state
        )
        restored = checkpoint_manager.restore(
            checkpoint_manager.latest_step(),
            args=ocp.args.Composite(
                model_state=ocp.args.StandardRestore(abstract_train_state),
                dataloader_state=grain.checkpoint.CheckpointRestore(grain_iterator),
            ),
        )
        train_state = restored["model_state"]
        grain_iterator = restored["dataloader_state"]
        step = checkpoint_manager.latest_step() or 0
        print(f"Restored dataloader and model state from step {step}")
    else:
        # Restore from pre-trained tokenizer (and LAM)
        train_state = restore_genie_components(
            train_state, replicated_sharding, grain_iterator, dummy_inputs, rng, args
        )

    # --- TRAIN LOOP ---
    dataloader = (jax.make_array_from_process_local_data(videos_sharding, elem) for elem in grain_iterator)  # type: ignore
    while step < args.num_steps:
        for videos in dataloader:
            # --- Train step ---
            rng, _rng, _rng_dropout, _rng_mask = jax.random.split(rng, 4)

            if args.use_gt_actions:
                # FIXME (f.srambical): use real actions instead of mock actions
                actions = jax.random.randint(
                    _rng, 
                    shape=(videos.shape[0], videos.shape[1]), 
                    minval=0, 
                    maxval=args.num_actions,
                    dtype=jnp.int32
                )
            else:
                actions = None

            inputs = dict(
                videos=videos,
                actions=actions,
                rng=_rng,
                dropout_rng=_rng_dropout,
                mask_rng=_rng_mask,
            )
            train_state, loss, recon, metrics = train_step(train_state, inputs)
            metrics["lr"] = lr_schedule(step)
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
                    gt_seq = inputs["videos"][0].astype(jnp.float32) / 255.0
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
            # --- Checkpointing ---
            if args.save_ckpt and step % args.log_checkpoint_interval == 0:
                checkpoint_manager.save(
                    step,
                    args=ocp.args.Composite(
                        model_state=ocp.args.StandardSave(train_state),
                        dataloader_state=grain.checkpoint.CheckpointSave(
                            grain_iterator
                        ),
                    ),
                )
                print(f"Saved checkpoint at step {step}")
            if step >= args.num_steps:
                break

    checkpoint_manager.close()
