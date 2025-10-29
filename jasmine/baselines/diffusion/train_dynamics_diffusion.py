"""
Parts of the diffusion training, sampling, and DiT implementation are adapted from:
https://github.com/kvfrans/shortcut-models

For diffusion-forcing training, we integrate several elements inspired by Dreamer 4
(https://arxiv.org/abs/2509.24527).
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

from dataclasses import dataclass, field
import itertools
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

from jasmine.models.genie import GenieDiffusion, restore_genie_components
from jasmine.utils.dataloader import get_dataloader
from jasmine.utils.train_utils import (
    get_lr_schedule,
    count_parameters_by_component,
    print_mem_stats,
    print_compiled_memory_stats,
    print_compiled_cost_analysis,
)


@dataclass
class Args:
    # Experiment
    num_steps: int = 200_000
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_height: int = 64
    image_width: int = 64
    data_dir: str = ""
    save_ckpt: bool = False
    restore_ckpt: bool = False
    # Optimization
    batch_size: int = 36
    init_lr: float = 0.0
    max_lr: float = 1e-4
    decay_end: float = 0.0
    wsd_decay_steps: int = (
        20_000  # NOTE: wsd_decay_steps will only be used when using a wsd-schedule
    )
    warmup_steps: int = 5000
    lr_schedule: str = "wsd"  # supported options: wsd, cos
    # Tokenizer
    tokenizer_dim: int = 512
    tokenizer_ffn_dim: int = 2048
    latent_patch_dim: int = 32
    num_patch_latents: int = 1024
    patch_size: int = 16
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
    diffusion_denoise_steps: int = 0
    diffusion_use_ramp_weight: bool = True
    param_dtype = jnp.float32
    dtype = jnp.bfloat16
    use_flash_attention: bool = True
    use_gt_actions: bool = False
    # Logging
    log: bool = True
    entity: str = ""
    project: str = ""
    name: str = "train_dynamics_diffusion"
    tags: list[str] = field(default_factory=lambda: ["dynamics", "diffusion"])
    log_interval: int = 50
    log_image_interval: int = 1000
    ckpt_dir: str = ""
    log_checkpoint_interval: int = 5000
    log_checkpoint_keep_period: int = 20_000
    log_gradients: bool = False
    val_data_dir: str = ""
    val_interval: int = 20_000
    val_steps: int = 50
    eval_full_frame: bool = True
    val_temperature: float = 1
    val_sample_argmax: bool = False
    wandb_id: str = ""


def build_model(args: Args, rng: jax.Array) -> tuple[GenieDiffusion, jax.Array]:
    rng, _rng = jax.random.split(rng)
    rngs = nnx.Rngs(_rng)
    genie = GenieDiffusion(
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
        use_gt_actions=args.use_gt_actions,
        # Dynamics
        dyna_dim=args.dyna_dim,
        dyna_ffn_dim=args.dyna_ffn_dim,
        dyna_num_blocks=args.dyna_num_blocks,
        dyna_num_heads=args.dyna_num_heads,
        dropout=args.dropout,
        diffusion_denoise_steps=args.diffusion_denoise_steps,
        param_dtype=args.param_dtype,
        dtype=args.dtype,
        use_flash_attention=args.use_flash_attention,
        decode=False,
        rngs=rngs,
    )
    if args.use_gt_actions:
        assert (
            not args.lam_checkpoint
        ), "Cannot use LAM when using ground-truth actions."
    else:
        assert genie.lam is not None
        del genie.lam.decoder
    return genie, rng


def build_optimizer(genie: GenieDiffusion, args: Args) -> nnx.ModelAndOptimizer:
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
    optimizer = nnx.ModelAndOptimizer(genie, tx)
    return optimizer


def build_mesh_and_sharding(
    num_devices: int,
) -> tuple[Mesh, NamedSharding, NamedSharding, NamedSharding]:
    device_mesh_arr = create_device_mesh((num_devices,))
    mesh = Mesh(devices=device_mesh_arr, axis_names=("data",))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    videos_sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None, None))
    actions_sharding = NamedSharding(mesh, PartitionSpec("data", None))
    return mesh, replicated_sharding, videos_sharding, actions_sharding


def shard_optimizer_states(
    optimizer: nnx.ModelAndOptimizer, replicated_sharding: NamedSharding
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


def build_dataloader(args: Args, data_dir: str) -> grain.DataLoaderIterator:
    image_shape = (args.image_height, args.image_width, args.image_channels)
    array_record_files = [
        os.path.join(data_dir, x)
        for x in os.listdir(data_dir)
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


def build_checkpoint_manager(args: Args) -> Optional[ocp.CheckpointManager]:
    if args.restore_ckpt or args.save_ckpt:
        handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
        handler_registry.add(
            "model_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler
        )
        handler_registry.add(
            "model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
        )
        handler_registry.add(
            "train_dataloader_state",
            grain.checkpoint.CheckpointSave,
            cast(ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler),
        )
        handler_registry.add(
            "train_dataloader_state",
            grain.checkpoint.CheckpointRestore,
            cast(ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler),
        )
        if args.val_data_dir:
            handler_registry.add(
                "val_dataloader_state",
                grain.checkpoint.CheckpointSave,
                cast(
                    ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler
                ),
            )
            handler_registry.add(
                "val_dataloader_state",
                grain.checkpoint.CheckpointRestore,
                cast(
                    ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler
                ),
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
    else:
        return None


def restore_or_initialize_components(
    args: Args,
    checkpoint_manager: Optional[ocp.CheckpointManager],
    optimizer: nnx.ModelAndOptimizer,
    train_iterator: grain.DataLoaderIterator,
    rng: jax.Array,
    replicated_sharding: NamedSharding,
    val_iterator: Optional[grain.DataLoaderIterator],
    restore_step: Optional[int] = None,
) -> tuple[
    int,
    nnx.ModelAndOptimizer,
    grain.DataLoaderIterator,
    grain.DataLoaderIterator,
    jax.Array,
]:
    step = 0
    if checkpoint_manager and restore_step is None:
        restore_step = checkpoint_manager.latest_step()
    if args.restore_ckpt:
        assert checkpoint_manager is not None
        abstract_optimizer = nnx.eval_shape(lambda: optimizer)
        abstract_optimizer_state = nnx.state(abstract_optimizer)
        if val_iterator:
            restore_args = ocp.args.Composite(
                model_state=ocp.args.PyTreeRestore(abstract_optimizer_state),  # type: ignore
                train_dataloader_state=grain.checkpoint.CheckpointRestore(train_iterator),  # type: ignore
                val_dataloader_state=grain.checkpoint.CheckpointRestore(val_iterator),  # type: ignore
            )
        else:
            restore_args = ocp.args.Composite(
                model_state=ocp.args.PyTreeRestore(abstract_optimizer_state),  # type: ignore
                train_dataloader_state=grain.checkpoint.CheckpointRestore(train_iterator),  # type: ignore
            )
        restored = checkpoint_manager.restore(restore_step, args=restore_args)
        restored_optimizer_state = restored["model_state"]
        nnx.update(optimizer, restored_optimizer_state)
        train_iterator = restored["train_dataloader_state"]
        if val_iterator:
            val_iterator = restored["val_dataloader_state"]
        step = restore_step or 0
        print(f"Restored dataloader and model state from step {step}")
    else:
        # Restore from pre-trained tokenizer (and LAM)
        rng, _rng = jax.random.split(rng)
        optimizer = restore_genie_components(
            optimizer, replicated_sharding, _rng, "mae", args
        )
    return step, optimizer, train_iterator, val_iterator, rng


def _calculate_step_metrics(
    outputs: dict[str, jax.Array],
    gt: jax.Array,
    num_actions: int,
) -> tuple[jax.Array, dict]:

    gt_val = gt.clip(0, 1).reshape(-1, *gt.shape[2:])
    recon = outputs["recon"].clip(0, 1).reshape(-1, *outputs["recon"].shape[2:])
    psnr = jnp.asarray(pix.psnr(gt_val, recon)).mean()
    ssim = jnp.asarray(pix.ssim(gt_val, recon)).mean()
    metrics = dict(
        psnr=psnr,
        ssim=ssim,
    )

    loss = jnp.asarray(0.0)
    if "x_pred" in outputs.keys():
        # x-pred instead of v-pred as per Dreamer 4 section 3.2
        mse_BTNL = (outputs["x_pred"] - outputs["x_gt"]) ** 2
        mse_BT = jnp.mean(mse_BTNL, axis=(2, 3))
        mse = jnp.mean(mse_BT)
        metrics["mse"] = mse
        if args.diffusion_use_ramp_weight:
            # ramp weight as per Dreamer 4 section 3.2
            ramp_weight = 0.9 * outputs["signal_level"] + 0.1
            loss = jnp.mean(mse_BT * ramp_weight)
        else:
            loss = mse

    if "lam_indices" in outputs.keys():
        _, index_counts_lam = jnp.unique_counts(
            jnp.ravel(outputs["lam_indices"]),
            size=num_actions,
            fill_value=0,
        )
        codebook_usage_lam = (index_counts_lam != 0).mean()
        metrics["codebook_usage_lam"] = codebook_usage_lam
    return loss, metrics


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
    optimizer = build_optimizer(genie, args)
    del genie

    # FIXME: switch to create_hybrid_device_mesh for runs spanning multiple nodes
    _, replicated_sharding, videos_sharding, actions_sharding = build_mesh_and_sharding(
        num_devices
    )

    shard_optimizer_states(optimizer, replicated_sharding)

    # --- Initialize checkpoint manager ---
    checkpoint_manager = build_checkpoint_manager(args)

    # --- Create DataLoaderIterator from dataloader ---
    train_iterator = build_dataloader(args, args.data_dir)
    val_iterator = None
    if args.val_data_dir:
        val_iterator = build_dataloader(args, args.val_data_dir)

    # --- Restore checkpoint ---
    step, optimizer, train_iterator, val_iterator, rng = (
        restore_or_initialize_components(
            args,
            checkpoint_manager,
            optimizer,
            train_iterator,
            rng,
            replicated_sharding,
            val_iterator,
        )
    )

    # --- Define loss and train step (close over args) ---
    def dynamics_loss_fn(
        model: GenieDiffusion,
        inputs: dict,
    ) -> tuple[jax.Array, tuple[jax.Array, dict]]:
        gt = jnp.asarray(inputs["videos"], dtype=jnp.float32) / 255.0
        inputs["videos"] = gt.astype(args.dtype)
        outputs = model(inputs)
        loss, metrics = _calculate_step_metrics(outputs, gt, args.num_actions)
        metrics["total_loss"] = loss
        return loss, (outputs["recon"], metrics)

    @nnx.jit(donate_argnums=0)
    def train_step(
        optimizer: nnx.ModelAndOptimizer, inputs: dict
    ) -> tuple[jax.Array, jax.Array, dict]:
        def loss_fn(model: GenieDiffusion) -> tuple[jax.Array, tuple[jax.Array, dict]]:
            model.train()
            return dynamics_loss_fn(model, inputs)

        (loss, (recon, metrics)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            optimizer.model
        )
        optimizer.update(grads)
        if args.log_gradients:
            metrics["gradients_std/"] = jax.tree.map(
                lambda x: x.std(), grads["params"]["dynamics"]
            )
        return loss, recon, metrics

    @nnx.jit
    def val_step(genie: GenieDiffusion, inputs: dict) -> dict:
        """Evaluate model and compute metrics"""
        genie.eval()
        gt = jnp.asarray(inputs["videos"], dtype=jnp.float32) / 255.0
        (loss, (recon, metrics)) = dynamics_loss_fn(genie, inputs)
        val_output = {"loss": loss, "recon": recon, "metrics": metrics}

        # --- Evaluate full frame prediction (sampling) ---
        if args.eval_full_frame:
            inputs["videos"] = gt.astype(args.dtype)
            lam_indices_E = None
            if not args.use_gt_actions:
                lam_indices_E = genie.vq_encode(inputs, training=False)
                inputs["latent_actions"] = lam_indices_E
            inputs["videos"] = inputs["videos"][
                :, :-1
            ]  # remove last frame for generation
            recon_full_frame = genie.sample(
                inputs,
                args.seq_len,
                args.diffusion_denoise_steps,
            )

            # Calculate metrics for the last frame only
            step_outputs = {
                "recon": recon_full_frame[:, -1],
            }
            if lam_indices_E is not None:
                lam_indices_B = lam_indices_E.reshape((-1, args.seq_len - 1))[:, -1]
                step_outputs["lam_indices"] = lam_indices_B

            loss_full_frame, metrics_full_frame = _calculate_step_metrics(
                step_outputs, gt[:, -1], args.num_actions
            )
            val_output.update(
                {
                    "loss_full_frame": loss_full_frame,
                    "recon_full_frame": recon_full_frame,
                    "metrics_full_frame": metrics_full_frame,
                }
            )
        return val_output

    def calculate_validation_metrics(val_dataloader, genie, rng):
        step = 0
        loss_per_step = []
        metrics_per_step = []
        loss_full_frame_per_step = []
        metrics_full_frame_per_step = []
        batch = None
        recon = None
        recon_full_frame = None
        for batch in val_dataloader:
            rng, _rng_mask = jax.random.split(rng, 2)
            batch["rng"] = _rng_mask
            val_outputs = val_step(genie, batch)
            loss_per_step.append(val_outputs["loss"])
            metrics_per_step.append(val_outputs["metrics"])
            recon = val_outputs["recon"]
            if args.eval_full_frame:
                loss_full_frame_per_step.append(val_outputs["loss_full_frame"])
                metrics_full_frame_per_step.append(val_outputs["metrics_full_frame"])
                recon_full_frame = val_outputs["recon_full_frame"]
            step += 1
            if step > args.val_steps:
                break

        if step < args.val_steps:
            print(
                f"Warning: Your validation dataset is too small to make val_steps many steps. Made {step} steps, expected {args.val_steps}"
            )

        val_metrics = {
            f"val_{key}": np.mean([float(m[key]) for m in metrics_per_step])
            for key in metrics_per_step[0].keys()
        }
        val_metrics["val_loss"] = np.mean(loss_per_step)
        if args.eval_full_frame:
            val_metrics_full_frame = {
                f"val_full_frame_{key}": np.mean(
                    [float(m[key]) for m in metrics_full_frame_per_step]
                )
                for key in metrics_full_frame_per_step[0].keys()
            }
            val_metrics.update(val_metrics_full_frame)
            val_metrics["val_full_frame_loss"] = np.mean(loss_full_frame_per_step)
        return val_metrics, batch, recon, recon_full_frame

    # --- TRAIN LOOP ---
    dataloader_train = (
        {
            "videos": jax.make_array_from_process_local_data(
                videos_sharding, local_data=elem["videos"]
            ),
            "actions": (
                jax.make_array_from_process_local_data(
                    actions_sharding, elem["actions"]
                )
                if args.use_gt_actions
                else None
            ),
        }
        for elem in train_iterator
    )
    dataloader_val = None
    if val_iterator:
        dataloader_val = (
            {
                "videos": jax.make_array_from_process_local_data(
                    videos_sharding, elem["videos"]
                ),
                "actions": (
                    jax.make_array_from_process_local_data(
                        actions_sharding, elem["actions"]
                    )
                    if args.use_gt_actions
                    else None
                ),
            }
            for elem in val_iterator
        )
    if jax.process_index() == 0:
        first_batch = next(dataloader_train)
        first_batch["rng"] = rng  # type: ignore
        compiled = train_step.lower(optimizer, first_batch).compile()
        print_compiled_memory_stats(compiled.memory_analysis())
        print_compiled_cost_analysis(compiled.cost_analysis())
        # Do not skip the first batch during training
        dataloader_train = itertools.chain([first_batch], dataloader_train)
    print(f"Starting training from step {step}...")
    first_step = step
    while step < args.num_steps:
        for batch in dataloader_train:
            # --- Train step ---
            rng, _rng_mask = jax.random.split(rng, 2)
            batch["rng"] = _rng_mask
            loss, recon, metrics = train_step(optimizer, batch)
            if step == first_step:
                print_mem_stats("After params initialized")
            step += 1

            # --- Validation loss ---
            val_results = {}
            if dataloader_val and step % args.val_interval == 0:
                rng, _rng_mask_val = jax.random.split(rng, 2)
                print("Calculating validation metrics...")
                val_metrics, val_gt_batch, val_recon, val_recon_full_frame = (
                    calculate_validation_metrics(
                        dataloader_val, optimizer.model, _rng_mask_val
                    )
                )
                print(f"Step {step}, validation loss: {val_metrics['val_loss']}")
                val_results = {
                    "metrics": val_metrics,
                    "gt_batch": val_gt_batch,
                    "recon": val_recon,
                    "full_frame": val_recon_full_frame,
                }

            # --- Logging ---
            if args.log:
                if step % args.log_interval == 0 and jax.process_index() == 0:
                    log_dict = {"loss": loss, "step": step, **metrics}
                    if val_results:
                        log_dict.update(val_results["metrics"])
                    wandb.log(log_dict)
                if step % args.log_image_interval == 0:
                    gt_seq = batch["videos"][0].astype(jnp.float32) / 255.0
                    recon_seq = recon[0].clip(0, 1)
                    comparison_seq = jnp.concatenate((gt_seq, recon_seq), axis=1)
                    comparison_seq = einops.rearrange(
                        comparison_seq * 255, "t h w c -> h (t w) c"
                    )
                    if val_results:
                        val_results["gt_seq_val"] = (
                            val_results["gt_batch"]["videos"][0].astype(jnp.float32)
                            / 255.0
                        )
                        val_results["recon_seq_val"] = val_results["recon"][0].clip(
                            0, 1
                        )
                        val_comparison_seq = jnp.concatenate(
                            (val_results["gt_seq_val"], val_results["recon_seq_val"]),
                            axis=1,
                        )
                        val_results["val_comparison_seq"] = einops.rearrange(
                            val_comparison_seq * 255, "t h w c -> h (t w) c"
                        )
                        if args.eval_full_frame:
                            val_results["full_frame_seq_val"] = val_results[
                                "full_frame"
                            ][0].clip(0, 1)
                            val_results["val_full_frame_comparison_seq"] = (
                                jnp.concatenate(
                                    (
                                        val_results["gt_seq_val"],
                                        val_results["full_frame_seq_val"],
                                    ),
                                    axis=1,
                                )
                            )
                            val_results["val_full_frame_comparison_seq"] = (
                                einops.rearrange(
                                    val_results["val_full_frame_comparison_seq"] * 255,
                                    "t h w c -> h (t w) c",
                                )
                            )
                    # NOTE: Process-dependent control flow deliberately happens
                    # after indexing operation since it must not contain code
                    # sections that lead to cross-accelerator communication.
                    if jax.process_index() == 0:
                        log_images = dict(
                            image=wandb.Image(np.asarray(gt_seq[args.seq_len - 1])),
                            recon=wandb.Image(np.asarray(recon_seq[args.seq_len - 1])),
                            true_vs_recon=wandb.Image(
                                np.asarray(comparison_seq.astype(np.uint8))
                            ),
                        )
                        if val_results:
                            log_images.update(
                                dict(
                                    val_image=wandb.Image(
                                        np.asarray(
                                            val_results["gt_seq_val"][args.seq_len - 1]
                                        )
                                    ),
                                    val_recon=wandb.Image(
                                        np.asarray(
                                            val_results["recon_seq_val"][
                                                args.seq_len - 1
                                            ]
                                        )
                                    ),
                                    val_true_vs_recon=wandb.Image(
                                        np.asarray(
                                            val_results["val_comparison_seq"].astype(
                                                np.uint8
                                            )
                                        )
                                    ),
                                )
                            )
                            if args.eval_full_frame:
                                log_images.update(
                                    dict(
                                        val_full_frame=wandb.Image(
                                            np.asarray(
                                                val_results["full_frame_seq_val"][
                                                    args.seq_len - 1
                                                ]
                                            )
                                        ),
                                        val_true_vs_full_frame=wandb.Image(
                                            np.asarray(
                                                val_results[
                                                    "val_full_frame_comparison_seq"
                                                ].astype(np.uint8)
                                            )
                                        ),
                                    )
                                )
                        wandb.log(log_images)
            # --- Checkpointing ---
            if args.save_ckpt and step % args.log_checkpoint_interval == 0:
                assert checkpoint_manager is not None
                optimizer_state = nnx.state(optimizer)
                if val_iterator:
                    ckpt_manager_args = ocp.args.Composite(
                        model_state=ocp.args.PyTreeSave(optimizer_state),  # type: ignore
                        train_dataloader_state=grain.checkpoint.CheckpointSave(  # type: ignore
                            train_iterator  # type: ignore
                        ),
                        val_dataloader_state=grain.checkpoint.CheckpointSave(  # type: ignore
                            val_iterator  # type: ignore
                        ),
                    )
                else:
                    ckpt_manager_args = ocp.args.Composite(
                        model_state=ocp.args.PyTreeSave(optimizer_state),  # type: ignore
                        train_dataloader_state=grain.checkpoint.CheckpointSave(  # type: ignore
                            train_iterator  # type: ignore
                        ),
                    )
                checkpoint_manager.save(step, args=ckpt_manager_args)
                print(f"Saved checkpoint at step {step}")
            if step >= args.num_steps:
                break

    if checkpoint_manager:
        checkpoint_manager.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
