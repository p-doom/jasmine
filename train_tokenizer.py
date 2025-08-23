from dataclasses import dataclass, field
import os
from typing import cast, Optional

import einops
import itertools
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

from models.tokenizer import TokenizerVQVAE
from utils.dataloader import get_dataloader
from utils.train_utils import (
    get_lr_schedule,
    count_parameters_by_component,
    print_compiled_memory_stats,
    print_compiled_cost_analysis,
)


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
    wsd_decay_steps: int = (
        20000  # NOTE: wsd_decay_steps will only be used when using a wsd-schedule
    )
    lr_schedule: str = "wsd"  # supported options: wsd, cos
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
    param_dtype = jnp.float32
    dtype = jnp.bfloat16
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


def build_model(args: Args, rng: jax.Array) -> tuple[TokenizerVQVAE, jax.Array]:
    rng, _rng = jax.random.split(rng)
    rngs = nnx.Rngs(_rng)
    return (
        TokenizerVQVAE(
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
            rngs=rngs,
        ),
        rng,
    )


def build_optimizer(
    model: TokenizerVQVAE, args: Args
) -> tuple[nnx.Optimizer, optax.Schedule]:
    lr_schedule = get_lr_schedule(
        args.lr_schedule,
        args.init_lr,
        args.max_lr,
        args.decay_end,
        args.num_steps,
        args.warmup_steps,
        args.wsd_decay_steps,
    )
    tx = optax.adamw(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=0.9,
        weight_decay=1e-4,
        mu_dtype=args.param_dtype,  # moments in full precision
    )
    optimizer = nnx.Optimizer(model, tx)
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


def build_dataloader(args: Args) -> grain.DataLoaderIterator:
    image_shape = (args.image_height, args.image_width, args.image_channels)
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
    return grain_iterator


def build_checkpoint_manager(args: Args) -> ocp.CheckpointManager:
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
    return checkpoint_manager


def restore_checkpoint_if_needed(
    args: Args,
    checkpoint_manager: ocp.CheckpointManager,
    optimizer: nnx.Optimizer,
    grain_iterator: grain.DataLoaderIterator,
    restore_step: Optional[int] = None,
) -> tuple[int, nnx.Optimizer, grain.DataLoaderIterator]:
    step = 0
    if restore_step is None:
        restore_step = checkpoint_manager.latest_step()
    if args.restore_ckpt:
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
    return step, optimizer, grain_iterator


def main(args: Args) -> None:
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

    rng = jax.random.key(args.seed)

    # --- Initialize model ---
    tokenizer, rng = build_model(args, rng)

    _, params, _ = nnx.split(tokenizer, nnx.Param, ...)
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
    optimizer, lr_schedule = build_optimizer(tokenizer, args)
    del tokenizer

    # FIXME: switch to create_hybrid_device_mesh for runs spanning multiple nodes
    mesh, replicated_sharding, videos_sharding = build_mesh_and_sharding(num_devices)

    shard_optimizer_states(optimizer, replicated_sharding)

    # --- Initialize checkpoint manager ---
    checkpoint_manager = build_checkpoint_manager(args)

    # --- Create DataLoaderIterator from dataloader ---
    grain_iterator = build_dataloader(args)

    # --- Restore checkpoint ---
    step, optimizer, grain_iterator = restore_checkpoint_if_needed(
        args, checkpoint_manager, optimizer, grain_iterator
    )

    # --- Define loss and train step (close over args) ---
    def tokenizer_loss_fn(
        model: TokenizerVQVAE, inputs: dict
    ) -> tuple[jax.Array, tuple[jax.Array, dict]]:
        gt = jnp.asarray(inputs["videos"], dtype=jnp.float32) / 255.0
        inputs["videos"] = gt.astype(args.dtype)
        model.train()
        outputs = model(inputs, training=True)
        outputs["recon"] = outputs["recon"].astype(jnp.float32)
        mse = jnp.square(gt - outputs["recon"]).mean()
        q_loss = jnp.square(jax.lax.stop_gradient(outputs["emb"]) - outputs["z"]).mean()
        commitment_loss = jnp.square(
            outputs["emb"] - jax.lax.stop_gradient(outputs["z"])
        ).mean()
        loss = mse + q_loss + args.vq_beta * commitment_loss

        gt_clipped = gt.clip(0, 1).reshape(-1, *gt.shape[2:])
        recon = outputs["recon"].clip(0, 1).reshape(-1, *outputs["recon"].shape[2:])
        psnr = jnp.asarray(pix.psnr(gt_clipped, recon)).mean()
        ssim = jnp.asarray(pix.ssim(gt_clipped, recon)).mean()
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

    @nnx.jit
    def train_step(
        optimizer: nnx.Optimizer, inputs: dict
    ) -> tuple[jax.Array, jax.Array, dict]:
        def loss_fn(model: TokenizerVQVAE) -> tuple[jax.Array, tuple[jax.Array, dict]]:
            return tokenizer_loss_fn(model, inputs)

        (loss, (recon, metrics)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            optimizer.model
        )
        optimizer.update(grads)
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
        return loss, recon, metrics

    # --- TRAIN LOOP ---
    dataloader = (
        jax.make_array_from_process_local_data(videos_sharding, elem)
        for elem in grain_iterator
    )
    if jax.process_index() == 0:
        first_videos = next(dataloader)
        sample_inputs = dict(videos=first_videos)
        compiled = train_step.lower(optimizer, sample_inputs).compile()
        print_compiled_memory_stats(compiled.memory_analysis())
        print_compiled_cost_analysis(compiled.cost_analysis())
        # Do not skip the first batch during training
        dataloader = itertools.chain([first_videos], dataloader)
    print(f"Starting training from step {step}...")
    while step < args.num_steps:
        for videos in dataloader:
            # --- Train step ---
            inputs = dict(videos=videos)
            loss, recon, metrics = train_step(optimizer, inputs)
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


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
