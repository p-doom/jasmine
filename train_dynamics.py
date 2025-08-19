from dataclasses import dataclass, field
import os
from typing import cast, Optional

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
    wsd_decay_steps: int = (
        10000  # NOTE: wsd_decay_steps will only be used when using a wsd-schedule
    )
    warmup_steps: int = 5000
    lr_schedule: str = "wsd"  # supported options: wsd, cos
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
    num_latent_actions: int = 6
    lam_patch_size: int = 16
    lam_num_blocks: int = 4
    lam_num_heads: int = 8
    lam_checkpoint: str = ""
    # Dynamics
    dyna_type: str = "maskgit"  # supported options: maskgit, causal
    dyna_dim: int = 512
    dyna_ffn_dim: int = 2048
    dyna_num_blocks: int = 6
    dyna_num_heads: int = 8
    dropout: float = 0.0
    mask_limit: float = 0.5
    param_dtype = jnp.float32
    dtype = jnp.bfloat16
    use_flash_attention: bool = True
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


def dynamics_loss_fn(
    model: Genie, inputs: dict, args: Args
) -> tuple[jax.Array, tuple[jax.Array, dict]]:
    """Compute masked dynamics loss"""
    gt = jnp.asarray(inputs["videos"], dtype=jnp.float32) / 255.0
    inputs["videos"] = gt.astype(args.dtype)
    model.train()
    outputs = model(inputs, training=True)
    mask = outputs["mask"]
    outputs["token_logits"] = outputs["token_logits"].astype(jnp.float32)
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
        outputs["token_logits"], outputs["video_tokens"]
    )
    ce_loss = (mask * ce_loss).sum() / mask.sum()
    acc = outputs["token_logits"].argmax(-1) == outputs["video_tokens"]
    acc = (mask * acc).sum() / mask.sum()
    select_probs = jax.nn.softmax(outputs["token_logits"])
    gt = gt.clip(0, 1).reshape(-1, *gt.shape[2:])
    recon = outputs["recon"].clip(0, 1).reshape(-1, *outputs["recon"].shape[2:])
    psnr = jnp.asarray(pix.psnr(gt, recon)).mean()
    ssim = jnp.asarray(pix.ssim(gt, recon)).mean()
    _, index_counts_lam = jnp.unique_counts(
        jnp.ravel(outputs["lam_indices"]), size=args.num_latent_actions, fill_value=0
    )
    _, index_counts_tokenizer = jnp.unique_counts(
        jnp.ravel(outputs["video_tokens"]), size=args.num_patch_latents, fill_value=0
    )
    codebook_usage_lam = (index_counts_lam != 0).mean()
    codebook_usage_tokenizer = (index_counts_tokenizer != 0).mean()
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


@nnx.jit
def train_step(
    model: Genie, optimizer: nnx.Optimizer, inputs: dict, args: Args
) -> tuple[jax.Array, jax.Array, dict]:
    """Update state and compute metrics"""

    def loss_fn(model: Genie) -> tuple[jax.Array, tuple[jax.Array, dict]]:
        return dynamics_loss_fn(model, inputs, args)

    (loss, (recon, metrics)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)
    if args.log_gradients:
        metrics["gradients_std/"] = jax.tree.map(
            lambda x: x.std(), grads["params"]["dynamics"]
        )
    return loss, recon, metrics


def build_model(a: Args, rng: jax.Array) -> tuple[Genie, jax.Array]:
    rng, _rng = jax.random.split(rng)
    rngs = nnx.Rngs(_rng)
    genie = Genie(
        # Tokenizer
        in_dim=a.image_channels,
        tokenizer_dim=a.tokenizer_dim,
        tokenizer_ffn_dim=a.tokenizer_ffn_dim,
        latent_patch_dim=a.latent_patch_dim,
        num_patch_latents=a.num_patch_latents,
        patch_size=a.patch_size,
        tokenizer_num_blocks=a.tokenizer_num_blocks,
        tokenizer_num_heads=a.tokenizer_num_heads,
        # LAM
        lam_dim=a.lam_dim,
        lam_ffn_dim=a.lam_ffn_dim,
        latent_action_dim=a.latent_action_dim,
        num_latent_actions=a.num_latent_actions,
        lam_patch_size=a.lam_patch_size,
        lam_num_blocks=a.lam_num_blocks,
        lam_num_heads=a.lam_num_heads,
        lam_co_train=not a.lam_checkpoint,
        # Dynamics
        dyna_type=a.dyna_type,
        dyna_dim=a.dyna_dim,
        dyna_ffn_dim=a.dyna_ffn_dim,
        dyna_num_blocks=a.dyna_num_blocks,
        dyna_num_heads=a.dyna_num_heads,
        dropout=a.dropout,
        mask_limit=a.mask_limit,
        param_dtype=a.param_dtype,
        dtype=a.dtype,
        use_flash_attention=a.use_flash_attention,
        decode=False,
        rngs=rngs,
    )
    del genie.lam.decoder
    return genie, rng


def build_optimizer(genie: Genie, a: Args) -> tuple[nnx.Optimizer, optax.Schedule]:
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
        mu_dtype=a.param_dtype,  # moments in full precision
    )
    optimizer = nnx.Optimizer(genie, tx)
    return optimizer, lr_schedule


def build_mesh_and_sharding(
    num_devices: int,
) -> tuple[Mesh, NamedSharding, NamedSharding]:
    device_mesh_arr = create_device_mesh((num_devices,))
    mesh = Mesh(devices=device_mesh_arr, axis_names=("data",))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    videos_sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None, None))
    return mesh, replicated_sharding, videos_sharding


def shard_optimizer_states(
    optimizer: nnx.Optimizer, replicated_sharding: NamedSharding
) -> None:
    model_state = nnx.state(optimizer.model)
    model_sharded_state = jax.lax.with_sharding_constraint(
        model_state, replicated_sharding
    )
    nnx.update(optimizer.model, model_sharded_state)
    optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
    optimizer_sharded_state = jax.lax.with_sharding_constraint(
        optimizer_state, replicated_sharding
    )
    nnx.update(optimizer, optimizer_sharded_state)


def build_dataloader(a: Args) -> tuple[grain.DataLoader, grain.DataLoaderIterator]:
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


def build_checkpoint_manager(a: Args) -> ocp.CheckpointManager:
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add(
        "model_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler
    )
    handler_registry.add(
        "model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
    )
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


def restore_or_initialize_components(
    a: Args,
    checkpoint_manager: ocp.CheckpointManager,
    optimizer: nnx.Optimizer,
    grain_iterator: grain.DataLoaderIterator,
    rng: jax.Array,
    replicated_sharding: NamedSharding,
    restore_step: Optional[int] = None,
) -> tuple[int, nnx.Optimizer, grain.DataLoaderIterator, jax.Array]:
    step = 0
    if restore_step is None:
        restore_step = checkpoint_manager.latest_step()
    if a.restore_ckpt:
        abstract_optimizer = nnx.eval_shape(lambda: optimizer)
        abstract_optimizer_state = nnx.state(abstract_optimizer)
        restored = checkpoint_manager.restore(
            restore_step,
            args=ocp.args.Composite(
                model_state=ocp.args.PyTreeRestore(abstract_optimizer_state),  # type: ignore
                dataloader_state=grain.checkpoint.CheckpointRestore(grain_iterator),  # type: ignore
            ),
        )
        restored_optimizer_state = restored["model_state"]
        nnx.update(optimizer, restored_optimizer_state)
        grain_iterator = restored["dataloader_state"]
        step = restore_step or 0
        print(f"Restored dataloader and model state from step {step}")
    else:
        # Restore from pre-trained tokenizer (and LAM)
        rng, _rng = jax.random.split(rng)
        optimizer = restore_genie_components(optimizer, replicated_sharding, _rng, a)
        # NOTE: We have to remove the (unused) tokenizer vq dropout due flax.nnx lazily initializing modules.
        # Specifically, the first dynamics model checkpoint will contain the vq dropout module,
        # but the first full restore will fail due to nnx not initializing the module when
        # dropout is set to 0.0.
        del optimizer.model.tokenizer.vq.drop
    return step, optimizer, grain_iterator, rng


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
    genie, rng = build_model(args, rng)
    _, params, _ = nnx.split(genie, nnx.Param, ...)
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
    optimizer, lr_schedule = build_optimizer(genie, args)
    del genie

    # FIXME: switch to create_hybrid_device_mesh for runs spanning multiple nodes
    mesh, replicated_sharding, videos_sharding = build_mesh_and_sharding(num_devices)

    shard_optimizer_states(optimizer, replicated_sharding)

    # --- Initialize checkpoint manager ---
    checkpoint_manager = build_checkpoint_manager(args)

    # --- Create DataLoaderIterator from dataloader ---
    _, grain_iterator = build_dataloader(args)

    # --- Restore checkpoint ---
    step, optimizer, grain_iterator, rng = restore_or_initialize_components(
        args, checkpoint_manager, optimizer, grain_iterator, rng, replicated_sharding
    )

    # --- TRAIN LOOP ---
    dataloader = (
        jax.make_array_from_process_local_data(videos_sharding, elem)
        for elem in grain_iterator
    )
    print(f"Starting training from step {step}...")
    while step < args.num_steps:
        for videos in dataloader:
            # --- Train step ---
            rng, _rng_mask = jax.random.split(rng, 2)
            inputs = dict(videos=videos, mask_rng=_rng_mask)
            loss, recon, metrics = train_step(optimizer.model, optimizer, inputs, args)
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
