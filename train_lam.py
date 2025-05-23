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

from models.lam import LatentActionModel
from utils.dataloader import get_dataloader

ts = int(time.time())


@dataclass
class Args:
    # Experiment
    num_steps: int = 200_000
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_resolution: int = 64
    data_dir: str = "data/coinrun_episodes"
    checkpoint: str = ""
    # Optimization
    batch_size: int = 36
    vq_beta: float = 0.25
    min_lr: float = 3e-6
    max_lr: float = 3e-5
    warmup_steps: int = 5000
    vq_reset_thresh: int = 50
    # LAM
    model_dim: int = 512
    latent_dim: int = 32
    num_latents: int = 6
    patch_size: int = 16
    num_blocks: int = 8
    num_heads: int = 8
    dropout: float = 0.0
    codebook_dropout: float = 0.0
    # Logging
    log: bool = False
    entity: str = ""
    project: str = ""
    log_interval: int = 5
    log_image_interval: int = 250
    ckpt_dir: str = ""
    log_checkpoint_interval: int = 10000


args = tyro.cli(Args)


def lam_loss_fn(params, state, inputs):
    # --- Compute loss ---
    outputs = state.apply_fn(
        params, inputs, training=True, rngs={"dropout": inputs["rng"]}
    )
    gt_future_frames = inputs["videos"][:, 1:]
    mse = jnp.square(gt_future_frames - outputs["recon"]).mean()
    q_loss = jnp.square(jax.lax.stop_gradient(outputs["emb"]) - outputs["z"]).mean()
    commitment_loss = jnp.square(
        outputs["emb"] - jax.lax.stop_gradient(outputs["z"])
    ).mean()
    loss = mse + q_loss + args.vq_beta * commitment_loss

    # --- Compute validation metrics ---
    gt = gt_future_frames.clip(0, 1).reshape(-1, *gt_future_frames.shape[2:])
    recon = outputs["recon"].clip(0, 1).reshape(-1, *outputs["recon"].shape[2:])
    psnr = pix.psnr(gt, recon).mean()
    ssim = pix.ssim(gt, recon).mean()
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


@jax.jit
def train_step(state, inputs, action_last_active):
    # --- Update model ---
    rng, inputs["rng"] = jax.random.split(inputs["rng"])
    grad_fn = jax.value_and_grad(lam_loss_fn, has_aux=True, allow_int=True)
    (loss, (recon, idx_counts, metrics)), grads = grad_fn(state.params, state, inputs)
    state = state.apply_gradients(grads=grads)

    # --- Reset inactive latent actions ---
    codebook = state.params["params"]["vq"]["codebook"]
    num_codes = len(codebook)
    active_codes = idx_counts != 0.0
    action_last_active = jnp.where(active_codes, 0, action_last_active + 1)
    p_code = active_codes / active_codes.sum()
    reset_idxs = jax.random.choice(rng, num_codes, shape=(num_codes,), p=p_code)
    do_reset = action_last_active >= args.vq_reset_thresh
    new_codebook = jnp.where(
        jnp.expand_dims(do_reset, -1), codebook[reset_idxs], codebook
    )
    state.params["params"]["vq"]["codebook"] = new_codebook
    action_last_active = jnp.where(do_reset, 0, action_last_active)
    return state, loss, recon, action_last_active, metrics


if __name__ == "__main__":
    rng = jax.random.PRNGKey(args.seed)
    if args.log:
        wandb.init(entity=args.entity, project=args.project, group="debug", config=args)

    # --- Initialize model ---
    lam = LatentActionModel(
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
    # Track when each action was last sampled
    action_last_active = jnp.zeros(args.num_latents)
    image_shape = (args.image_resolution, args.image_resolution, args.image_channels)
    rng, _rng = jax.random.split(rng)
    inputs = dict(
        videos=jnp.zeros(
            (args.batch_size, args.seq_len, *image_shape), dtype=jnp.float32
        ),
        rng=_rng,
    )
    rng, _rng = jax.random.split(rng)
    init_params = lam.init(_rng, inputs)

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
    train_state = TrainState.create(apply_fn=lam.apply, params=init_params, tx=tx)

    # --- TRAIN LOOP ---
    tfrecord_files = [
        os.path.join(args.data_dir, x)
        for x in os.listdir(args.data_dir)
        if x.endswith(".tfrecord")
    ]
    dataloader = get_dataloader(
        tfrecord_files, args.seq_len, args.batch_size, *image_shape
    )
    print(f"Starting training from step {step}...")
    while step < args.num_steps:
        for videos in dataloader:
            # --- Train step ---
            rng, _rng = jax.random.split(rng)
            inputs = dict(videos=videos, rng=_rng)
            train_state, loss, recon, action_last_active, metrics = train_step(
                train_state, inputs, action_last_active
            )
            print(f"Step {step}, loss: {loss}")
            step += 1

            # --- Logging ---
            if args.log:
                if step % args.log_interval == 0:
                    wandb.log({"loss": loss, "step": step, **metrics})
                if step % args.log_image_interval == 0:
                    gt_seq = inputs["videos"][0][1:]
                    recon_seq = recon[0].clip(0, 1)
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
                        os.path.join(os.getcwd(), args.ckpt_dir, f"lam_{ts}_{step}"),
                        ckpt,
                        save_args=save_args,
                    )
            if step >= args.num_steps:
                break
