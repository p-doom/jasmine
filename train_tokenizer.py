from dataclasses import dataclass, field
import os

import einops
from flax.training import orbax_utils
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

from models.tokenizer import TokenizerVQVAE
from utils.dataloader import get_dataloader
from utils.lr_utils import get_lr_schedule
from utils.parameter_utils import count_parameters_by_component


@dataclass
class Args:
    # Experiment
    num_steps: int = 300_000
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_height: int = 90
    image_width: int = 160
    data_dir: str = ""
    save_ckpt: bool = False
    restore_ckpt: bool = False
    # Optimization
    vq_beta: float = 0.25
    batch_size: int = 48
    init_lr: float = 0.0
    max_lr: float = 3e-4
    decay_end: float = 0.0
    wsd_decay_steps: int = 20000 # NOTE: wsd_decay_steps will only be used when using a wsd-schedule
    lr_schedule: str = "wsd" # supported options: wsd, cos 
    warmup_steps: int = 10000
    # Tokenizer
    model_dim: int = 512
    ffn_dim: int = 2048
    latent_dim: int = 32
    num_latents: int = 1024
    patch_size: int = 4
    num_blocks: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    codebook_dropout: float = 0.01
    param_dtype: jnp.dtype = jnp.float32
    dtype: jnp.dtype = jnp.bfloat16
    # Logging
    log: bool = False
    entity: str = ""
    project: str = ""
    name: str = "train_tokenizer"
    tags: list[str] = field(default_factory=lambda: ["tokenizer"])
    log_interval: int = 5
    log_image_interval: int = 250
    ckpt_dir: str = ""
    log_checkpoint_interval: int = 10000
    log_checkpoint_keep_period: int = 20000
    log_gradients: bool = False
    wandb_id: str = ""
    use_flash_attention: bool = True


args = tyro.cli(Args)


def tokenizer_loss_fn(params, state, inputs):
    # --- Compute loss ---
    # FIXME (f.srambical): Can we even do native int8 training without casting the video at all?
    # FIXME (f.srambical): If the tokenizer is the reason for the dynamics model being memory-bound,
    # should we at least train the tokenizer natively in int8?
    inputs["videos"] = inputs["videos"].astype(args.dtype) / 255.0
    outputs = state.apply_fn(
        params,
        inputs,
        training=True,
        rngs={"params": inputs["rng"], "dropout": inputs["dropout_rng"]},
    )
    mse = jnp.square(inputs["videos"] - outputs["recon"]).mean()
    q_loss = jnp.square(jax.lax.stop_gradient(outputs["emb"]) - outputs["z"]).mean()
    commitment_loss = jnp.square(
        outputs["emb"] - jax.lax.stop_gradient(outputs["z"])
    ).mean()
    loss = mse + q_loss + args.vq_beta * commitment_loss

    # --- Compute validation metrics ---
    gt = inputs["videos"].clip(0, 1).reshape(-1, *inputs["videos"].shape[2:])
    recon = outputs["recon"].clip(0, 1).reshape(-1, *outputs["recon"].shape[2:])
    psnr = pix.psnr(gt, recon).mean()  # type: ignore
    ssim = pix.ssim(gt, recon).mean()  # type: ignore
    _, index_counts = jnp.unique_counts(
        jnp.ravel(outputs["indices"]), size=args.num_latents, fill_value=0
    )
    codebook_usage = (index_counts != 0).mean()
    metrics = dict(
        loss=loss,
        mse=mse,
        q_loss=q_loss,
        commitment_loss=commitment_loss,
        psnr=psnr,
        ssim=ssim,
        codebook_usage=codebook_usage,
    )
    return loss, (outputs["recon"], metrics)


@jax.jit
def train_step(state, inputs):
    grad_fn = jax.value_and_grad(tokenizer_loss_fn, has_aux=True, allow_int=True)
    (loss, (recon, metrics)), grads = grad_fn(state.params, state, inputs)
    state = state.apply_gradients(grads=grads)
    if args.log_gradients:
        metrics["encoder_gradients_std/"] = jax.tree.map(
            lambda x: x.std(), grads["params"]["encoder"]
        )
        metrics["vq_gradients_std/"] = jax.tree.map(
            lambda x: x.std(), grads["params"]["vq"]
        )
        metrics["decoder_gradients_std/"] = jax.tree.map(
            lambda x: x.std(), grads["params"]["decoder"]
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
    tokenizer = TokenizerVQVAE(
        in_dim=args.image_channels,
        model_dim=args.model_dim,
        ffn_dim=args.ffn_dim,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        patch_size=args.patch_size,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        codebook_dropout=args.codebook_dropout,
        param_dtype=args.param_dtype,
        dtype=args.dtype,
        use_flash_attention=args.use_flash_attention,
    )
    rng, _rng = jax.random.split(rng)
    image_shape = (args.image_height, args.image_width, args.image_channels)
    inputs = dict(
        videos=jnp.zeros(
            (per_device_batch_size_for_init, args.seq_len, *image_shape),
            dtype=args.dtype,
        ),
    )
    init_params = tokenizer.init(_rng, inputs)

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
    train_state = TrainState.create(apply_fn=tokenizer.apply, params=init_params, tx=tx)

    # FIXME: switch to create_hybrid_device_mesh for runs spanning multiple nodes
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

    # --- TRAIN LOOP ---
    dataloader = (jax.make_array_from_process_local_data(videos_sharding, elem) for elem in grain_iterator)  # type: ignore
    print(f"Starting training from step {step}...")
    while step < args.num_steps:
        for videos in dataloader:
            # --- Train step ---
            rng, _rng, _rng_dropout = jax.random.split(rng, 3)

            inputs = dict(videos=videos, rng=_rng, dropout_rng=_rng_dropout)
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
                    # NOTE: Process-dependent control flow deliberately happens
                    # after indexing operation since it must not contain code
                    # sections that lead to cross-accelerator communication.
                    if jax.process_index() == 0:
                        log_images = dict(
                            image=wandb.Image(np.asarray(gt_seq[0])),
                            recon=wandb.Image(np.asarray(recon_seq[0])),
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
