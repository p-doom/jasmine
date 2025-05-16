from dataclasses import dataclass
import os
import time

import einops
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import optax
import orbax
from orbax.checkpoint import PyTreeCheckpointer
import numpy as np
import dm_pix as pix
import jax
import jax.numpy as jnp
import tyro
import wandb

from models.tokenizer import TokenizerVQVAE
from utils.dataloader import get_dataloader

ts = int(time.time())


@dataclass
class Args:
    # Experiment
    num_steps: int = 300_000
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_resolution: int = 64
    data_dir: str = "data/coinrun_episodes"
    checkpoint: str = ""
    # Optimization
    vq_beta: float = 0.25
    batch_size: int = 48
    min_lr: float = 3e-4
    max_lr: float = 3e-4
    warmup_steps: int = 10000
    # Tokenizer
    model_dim: int = 512
    latent_dim: int = 32
    num_latents: int = 1024
    patch_size: int = 4
    num_blocks: int = 8
    num_heads: int = 8
    dropout: float = 0.0
    codebook_dropout: float = 0.01
    # Logging
    log: bool = False
    entity: str = ""
    project: str = ""
    log_interval: int = 5
    log_image_interval: int = 250
    ckpt_dir: str = ""
    log_checkpoint_interval: int = 10000
    log_gradients: bool = False


args = tyro.cli(Args)


def tokenizer_loss_fn(params, state, inputs):
    # --- Compute loss ---
    outputs = state.apply_fn(
        params, inputs, training=True, rngs={"dropout": inputs["rng"]}
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
    psnr = pix.psnr(gt, recon).mean()
    ssim = pix.ssim(gt, recon).mean()
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


def train_step(state, inputs):
    grad_fn = jax.value_and_grad(tokenizer_loss_fn, has_aux=True, allow_int=True)
    (loss, (recon, metrics)), grads = grad_fn(state.params, state, inputs)

    grads = jax.lax.pmean(grads, axis_name='devices')
    loss = jax.lax.pmean(loss, axis_name='devices')
    metrics = jax.lax.pmean(metrics, axis_name='devices')

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
    num_devices = jax.device_count()
    if num_devices == 0:
        raise ValueError("No JAX devices found.")
    print(f"Running on {num_devices} devices.")

    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by "
            f"number of devices {num_devices}."
        )
    per_device_batch_size = args.batch_size // num_devices

    rng = jax.random.PRNGKey(args.seed)
    if args.log:
        wandb.init(entity=args.entity, project=args.project, group="debug", config=args)

    # --- Initialize model ---
    tokenizer = TokenizerVQVAE(
        in_dim=args.image_channels,
        model_dim=args.model_dim,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        patch_size=args.patch_size,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        codebook_dropout=args.codebook_dropout,
    )
    rng, _rng = jax.random.split(rng)
    image_shape = (args.image_resolution, args.image_resolution, args.image_channels)
    inputs = dict(
        videos=jnp.zeros(
            (args.batch_size, args.seq_len, *image_shape), dtype=jnp.float32
        ),
    )
    init_params = tokenizer.init(_rng, inputs)

    # --- Load checkpoint ---
    step = 0
    if args.checkpoint:
        init_params["params"].update(
            PyTreeCheckpointer().restore(args.checkpoint)["model"]["params"]["params"]
        )
        # Assume checkpoint is of the form tokenizer_<timestamp>_<step>
        step += int(args.checkpoint.split("_")[-1])

    # --- Initialize optimizer ---
    lr_schedule = optax.warmup_cosine_decay_schedule(
        args.min_lr, args.max_lr, args.warmup_steps, args.num_steps
    )
    tx = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.9, weight_decay=1e-4)
    train_state = TrainState.create(apply_fn=tokenizer.apply, params=init_params, tx=tx)
    train_state = jax.device_put_replicated(train_state, jax.local_devices())

    pmapped_train_step = jax.pmap(train_step, axis_name='devices')

    # --- TRAIN LOOP ---
    dataloader = get_dataloader(args.data_dir, args.seq_len, args.batch_size)
    while step < args.num_steps:
        for videos in dataloader:
            # --- Train step ---
            rng, *_rngs = jax.random.split(rng, num_devices + 1)
            _rngs = jnp.stack(_rngs)
            
            videos = einops.rearrange(
                videos, '(d b) t h w c -> d b t h w c', d=num_devices, b=per_device_batch_size
            )
            
            inputs = dict(videos=videos, rng=_rngs)

            train_state, loss, recon, metrics = pmapped_train_step(
                train_state, inputs
            )
            
            print(f"Step {step}, loss: {loss[0].item()}")
            step += 1

            if args.log:
                if step % args.log_interval == 0:
                    log_data = {}
                    for k, v_arr in metrics.items():
                        log_data[k] = v_arr[0].item()
                    log_data = {"loss": loss[0].item(), "step": step, **log_data}

                if step % args.log_image_interval == 0:
                    gt_seq = videos[0, 0]
                    recon_seq = recon[0, 0].clip(0, 1)
                    comparison_seq = jnp.concatenate((gt_seq, recon_seq), axis=1)
                    comparison_seq = einops.rearrange(
                        comparison_seq * 255, "t h w c -> h (t w) c"
                    )
                    log_images = dict(
                        image=wandb.Image(np.asarray(gt_seq[0])),
                        recon=wandb.Image(np.asarray(recon_seq[0])),
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
                        os.path.join(
                            os.getcwd(), args.ckpt_dir, f"tokenizer_{ts}_{step}"
                        ),
                        ckpt,
                        save_args=save_args,
                    )
            if step >= args.num_steps:
                break
