from dataclasses import dataclass, field
import os
from typing import cast

import einops
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
import flax.nnx as nnx

from models.lam import LatentActionModel
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
    vq_beta: float = 0.25
    init_lr: float = 0.0
    max_lr: float = 3e-5
    decay_end: float = 0.0
    wsd_decay_steps: int = (
        10000  # NOTE: wsd_decay_steps will only be used when using a wsd-schedule
    )
    warmup_steps: int = 5000
    lr_schedule: str = "wsd"  # supported options: wsd, cos
    vq_reset_thresh: int = 50
    # LAM
    model_dim: int = 512
    ffn_dim: int = 2048
    latent_dim: int = 32
    num_latents: int = 6
    patch_size: int = 16
    num_blocks: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    codebook_dropout: float = 0.0
    param_dtype = jnp.float32
    dtype = jnp.bfloat16
    # Logging
    log: bool = False
    entity: str = ""
    project: str = ""
    name: str = "train_lam"
    tags: list[str] = field(default_factory=lambda: ["lam"])
    log_interval: int = 5
    log_image_interval: int = 250
    ckpt_dir: str = ""
    log_checkpoint_interval: int = 10000
    log_checkpoint_keep_period: int = 20000
    wandb_id: str = ""
    use_flash_attention: bool = True


def lam_loss_fn(
    model: LatentActionModel, inputs: dict, args: Args
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, dict]]:
    # --- Compute loss ---
    gt = jnp.asarray(inputs["videos"], dtype=jnp.float32) / 255.0
    inputs["videos"] = gt.astype(args.dtype)
    model.train()
    outputs = model(inputs, training=True)
    outputs["recon"] = outputs["recon"].astype(jnp.float32)
    gt_future_frames = gt[:, 1:]
    mse = jnp.square(gt_future_frames - outputs["recon"]).mean()
    q_loss = jnp.square(jax.lax.stop_gradient(outputs["emb"]) - outputs["z"]).mean()
    commitment_loss = jnp.square(
        outputs["emb"] - jax.lax.stop_gradient(outputs["z"])
    ).mean()
    loss = mse + q_loss + args.vq_beta * commitment_loss

    # --- Compute validation metrics ---
    gt = gt_future_frames.clip(0, 1).reshape(-1, *gt_future_frames.shape[2:])
    recon = outputs["recon"].clip(0, 1).reshape(-1, *outputs["recon"].shape[2:])
    psnr = jnp.asarray(pix.psnr(gt, recon)).mean()
    ssim = jnp.asarray(pix.ssim(gt, recon)).mean()
    count_fn = jax.vmap(lambda i: (outputs["indices"] == i).sum())
    index_counts = count_fn(jnp.arange(args.num_latents))
    metrics = dict(
        loss=loss,
        mse=mse,
        q_loss=q_loss,
        commitment_loss=commitment_loss,
        psnr=psnr,
        ssim=ssim,
        codebook_usage=(index_counts != 0).mean(),
    )
    return loss, (outputs["recon"], index_counts, metrics)


@nnx.jit
def train_step(
    lam: LatentActionModel,
    optimizer: nnx.Optimizer,
    inputs: dict,
    action_last_active: jax.Array,
    rng: jax.Array,
    args: Args,
) -> tuple[jax.Array, jax.Array, jax.Array, dict]:
    def loss_fn(
        model: LatentActionModel,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, dict]]:
        return lam_loss_fn(model, inputs, args)

    # --- Update model ---
    (loss, (recon, idx_counts, metrics)), grads = nnx.value_and_grad(
        loss_fn, has_aux=True
    )(lam)
    optimizer.update(grads)

    # --- Reset inactive latent actions ---
    codebook = lam.vq.codebook
    num_codes = len(codebook)
    active_codes = idx_counts != 0.0
    action_last_active = jnp.where(active_codes, 0, action_last_active + 1)
    p_code = active_codes / active_codes.sum()
    reset_idxs = jax.random.choice(rng, num_codes, shape=(num_codes,), p=p_code)
    do_reset = action_last_active >= args.vq_reset_thresh
    new_codebook = jnp.where(
        jnp.expand_dims(do_reset, -1), codebook[reset_idxs], codebook.value
    )
    lam.vq.codebook.value = new_codebook
    action_last_active = jnp.where(do_reset, 0, action_last_active)
    return loss, recon, action_last_active, metrics


def build_model(a: Args, rng: jax.Array) -> tuple[LatentActionModel, jax.Array]:
    rng, _rng = jax.random.split(rng)
    rngs = nnx.Rngs(_rng)
    return LatentActionModel(
        in_dim=a.image_channels,
        model_dim=a.model_dim,
        ffn_dim=a.ffn_dim,
        latent_dim=a.latent_dim,
        num_latents=a.num_latents,
        patch_size=a.patch_size,
        num_blocks=a.num_blocks,
        num_heads=a.num_heads,
        dropout=a.dropout,
        codebook_dropout=a.codebook_dropout,
        param_dtype=a.param_dtype,
        dtype=a.dtype,
        use_flash_attention=a.use_flash_attention,
        rngs=rngs,
    ), rng


def build_optimizer(model: LatentActionModel, a: Args):
    lr_schedule = get_lr_schedule(
        a.lr_schedule,
        a.init_lr,
        a.max_lr,
        a.decay_end,
        a.num_steps,
        a.warmup_steps,
        a.wsd_decay_steps,
    )
    tx = optax.adamw(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=0.9,
        weight_decay=1e-4,
        mu_dtype=a.param_dtype, # moments in full precision
    )
    optimizer = nnx.Optimizer(model, tx)
    return optimizer, lr_schedule


def build_mesh_and_sharding(num_devices: int):
    device_mesh_arr = create_device_mesh((num_devices,))
    mesh = Mesh(devices=device_mesh_arr, axis_names=("data",))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    videos_sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None, None))
    return mesh, replicated_sharding, videos_sharding


def shard_optimizer_states(optimizer: nnx.Optimizer, replicated_sharding: NamedSharding) -> None:
    model_state = nnx.state(optimizer.model)
    model_sharded_state = jax.lax.with_sharding_constraint(model_state, replicated_sharding)
    nnx.update(optimizer.model, model_sharded_state)
    optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
    optimizer_sharded_state = jax.lax.with_sharding_constraint(optimizer_state, replicated_sharding)
    nnx.update(optimizer, optimizer_sharded_state)


def build_dataloader(a: Args):
    image_shape = (a.image_height, a.image_width, a.image_channels)
    array_record_files = [
        os.path.join(a.data_dir, x)
        for x in os.listdir(a.data_dir)
        if x.endswith(".array_record")
    ]
    grain_dataloader = get_dataloader(
        array_record_files,
        a.seq_len,
        # NOTE: We deliberately pass the global batch size
        # The dataloader shards the dataset across all processes
        a.batch_size,
        *image_shape,
        num_workers=8,
        prefetch_buffer_size=1,
        seed=a.seed,
    )
    initial_state = grain_dataloader._create_initial_state()
    grain_iterator = grain.DataLoaderIterator(grain_dataloader, initial_state)
    return grain_dataloader, grain_iterator


def build_checkpoint_manager(a: Args):
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add("model_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
    handler_registry.add("model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    handler_registry.add(
        "dataloader_state",
        grain.checkpoint.CheckpointSave,
        cast(ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler),
    )
    handler_registry.add(
        "dataloader_state",
        grain.checkpoint.CheckpointRestore,
        cast(ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler),
    )
    checkpoint_options = ocp.CheckpointManagerOptions(
        save_interval_steps=a.log_checkpoint_interval,
        max_to_keep=3,
        keep_period=a.log_checkpoint_keep_period,
        step_format_fixed_length=6,
        cleanup_tmp_directories=True,
    )
    checkpoint_manager = ocp.CheckpointManager(
        a.ckpt_dir,
        options=checkpoint_options,
        handler_registry=handler_registry,
    )
    return checkpoint_manager


def restore_checkpoint_if_needed(a: Args, checkpoint_manager, optimizer, grain_iterator):
    step = 0
    if a.restore_ckpt:
        abstract_optimizer = nnx.eval_shape(lambda: optimizer)
        abstract_optimizer_state = nnx.state(abstract_optimizer)
        restored = checkpoint_manager.restore(
            checkpoint_manager.latest_step(),
            args=ocp.args.Composite(
                model_state=ocp.args.PyTreeRestore(abstract_optimizer_state),  # type: ignore
                dataloader_state=grain.checkpoint.CheckpointRestore(grain_iterator),  # type: ignore
            ),
        )
        restored_optimizer_state = restored["model_state"]
        nnx.update(optimizer, restored_optimizer_state)
        grain_iterator = restored["dataloader_state"]
        step = checkpoint_manager.latest_step() or 0
        print(f"Restored dataloader and model state from step {step}")
    return step, optimizer, grain_iterator


def enable_sowing(lam: LatentActionModel) -> None:
    for model in [lam.encoder, lam.decoder]:
        setattr(model, "sow_logits", True)
        for blk in getattr(model, "blocks", []):
            setattr(blk, "sow_weights", True)
            setattr(blk, "sow_activations", True)


if __name__ == "__main__":
    jax.distributed.initialize()
    args = tyro.cli(Args)
    num_devices = jax.device_count()
    if num_devices == 0:
        raise ValueError("No JAX devices found.")
    print(f"Running on {num_devices} devices.")

    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by "
            f"number of devices {num_devices}."
        )

    rng = jax.random.key(args.seed)

    # --- Initialize model ---
    lam, rng = build_model(args, rng)

    # Count parameters
    _, params, _ = nnx.split(lam, nnx.Param, ...)
    param_counts = count_parameters_by_component(params)

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
    optimizer, lr_schedule = build_optimizer(lam, args)

    # FIXME: switch to create_hybrid_device_mesh for runs spanning multiple nodes
    mesh, replicated_sharding, videos_sharding = build_mesh_and_sharding(num_devices)

    shard_optimizer_states(optimizer, replicated_sharding)

    # --- Initialize checkpoint manager ---
    step = 0
    checkpoint_manager = build_checkpoint_manager(args)

    # --- Create DataLoaderIterator from dataloader ---
    grain_dataloader, grain_iterator = build_dataloader(args)

    # --- Restore checkpoint ---
    step, optimizer, grain_iterator = restore_checkpoint_if_needed(
        args, checkpoint_manager, optimizer, grain_iterator
    )

    # --- TRAIN LOOP ---
    dataloader = (
        jax.make_array_from_process_local_data(videos_sharding, elem)
        for elem in grain_iterator
    )
    print(f"Starting training from step {step}...")
    action_last_active = jnp.zeros(args.num_latents, dtype=jnp.int32)
    while step < args.num_steps:
        for videos in dataloader:
            # --- Train step ---
            rng, _rng = jax.random.split(rng)

            inputs = dict(videos=videos, rng=_rng)
            rng, _rng = jax.random.split(rng)
            loss, recon, action_last_active, metrics = train_step(
                lam, optimizer, inputs, action_last_active, _rng, args
            )
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
                    gt_seq = inputs["videos"][0, 1:].astype(jnp.float32) / 255.0
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
            if (args.save_ckpt and step % args.log_checkpoint_interval == 0):
                optimizer_state = nnx.state(optimizer)
                checkpoint_manager.save(
                    step,
                    args=ocp.args.Composite(
                        model_state=ocp.args.PyTreeSave(optimizer_state),  # type: ignore
                        dataloader_state=grain.checkpoint.CheckpointSave(  # type: ignore
                            grain_iterator  # type: ignore
                        ),
                    ),
                )
                print(f"Saved checkpoint at step {step}")
            if step >= args.num_steps:
                break

    checkpoint_manager.close()
