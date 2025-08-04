from dataclasses import dataclass
import time
import os
import optax

import dm_pix as pix
import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image, ImageDraw
import tyro
from flax import nnx

from genie import Genie
from utils.dataloader import get_dataloader


@dataclass
class Args:
    # Experiment
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_height: int = 90
    image_width: int = 160
    data_dir: str = "data/coinrun_episodes"
    checkpoint: str = ""
    # Sampling
    batch_size: int = 1
    maskgit_steps: int = 25
    temperature: float = 1.0
    sample_argmax: bool = True
    start_frame: int = 0
    # Tokenizer checkpoint
    tokenizer_dim: int = 512
    tokenizer_ffn_dim: int = 2048
    latent_patch_dim: int = 32
    num_patch_latents: int = 1024
    patch_size: int = 4
    tokenizer_num_blocks: int = 4
    tokenizer_num_heads: int = 8
    # LAM checkpoint
    lam_dim: int = 512
    lam_ffn_dim: int = 2048
    latent_action_dim: int = 32
    num_latent_actions: int = 6
    lam_patch_size: int = 16
    lam_num_blocks: int = 4
    lam_num_heads: int = 8
    # Dynamics checkpoint
    dyna_dim: int = 512
    dyna_ffn_dim: int = 2048
    dyna_num_blocks: int = 6
    dyna_num_heads: int = 8
    param_dtype = jnp.float32
    dtype = jnp.bfloat16
    use_flash_attention: bool = True


args = tyro.cli(Args)

if __name__ == "__main__":
    """
    Dimension keys:
        B: batch size
        T: number of input (conditioning) frames
        N: number of patches per frame
        S: sequence length
        H: height
        W: width
        E: B * (S - 1)
    """
    jax.distributed.initialize()

    rng = jax.random.key(args.seed)

    # --- Load Genie checkpoint ---
    rngs = nnx.Rngs(rng)
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
        num_latent_actions=args.num_latent_actions,
        lam_patch_size=args.lam_patch_size,
        lam_num_blocks=args.lam_num_blocks,
        lam_num_heads=args.lam_num_heads,
        lam_co_train=False,
        # Dynamics
        dyna_dim=args.dyna_dim,
        dyna_ffn_dim=args.dyna_ffn_dim,
        dyna_num_blocks=args.dyna_num_blocks,
        dyna_num_heads=args.dyna_num_heads,
        param_dtype=args.param_dtype,
        dtype=args.dtype,
        use_flash_attention=args.use_flash_attention,
        rngs=rngs,
    )

    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add(
        "model_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler
    )
    handler_registry.add(
        "model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
    )
    checkpoint_options = ocp.CheckpointManagerOptions(
        step_format_fixed_length=6,
    )
    checkpoint_manager = ocp.CheckpointManager(
        args.checkpoint,
        options=checkpoint_options,
        handler_registry=handler_registry,
    )

    dummy_tx = optax.adamw(
        learning_rate=optax.linear_schedule(0.0001, 0.0001, 10000),
        b1=0.9,
        b2=0.9,
        weight_decay=1e-4,
        mu_dtype=args.dtype,
    )
    dummy_optimizer = nnx.Optimizer(genie, dummy_tx)

    abstract_optimizer = nnx.eval_shape(lambda: dummy_optimizer)
    abstract_optimizer_state = nnx.state(abstract_optimizer)
    restored = checkpoint_manager.restore(
        checkpoint_manager.latest_step(),
        args=ocp.args.Composite(
            model_state=ocp.args.PyTreeRestore(abstract_optimizer_state),  # type: ignore
        ),
    )
    restored_optimizer_state = restored["model_state"]
    nnx.update(dummy_optimizer, restored_optimizer_state)

    # --- Define sampling function ---
    def _sampling_fn(model: Genie, batch: dict) -> jax.Array:
        """Runs Genie.sample with pre-defined generation hyper-parameters."""
        return model.sample(
            batch,
            args.seq_len,
            args.maskgit_steps,
            args.temperature,
            args.sample_argmax,
        )

    # --- Define autoregressive sampling loop ---
    @nnx.jit
    def _autoreg_sample(rng, video_batch_BSHWC, action_batch_E):
        input_video_BTHWC = video_batch_BSHWC[:, :args.start_frame]
        rng, _rng = jax.random.split(rng)
        batch = dict(videos=input_video_BTHWC, latent_actions=action_batch_E, rng=_rng)
        generated_vid_BSHWC = _sampling_fn(genie, batch)
        return generated_vid_BSHWC

    # --- Get video + latent actions ---
    array_record_files = [
        os.path.join(args.data_dir, x)
        for x in os.listdir(args.data_dir)
        if x.endswith(".array_record")
    ]
    dataloader = get_dataloader(
        array_record_files,
        args.seq_len,
        args.batch_size,
        args.image_height,
        args.image_width,
        args.image_channels,
        # We don't use workers in order to avoid grain shutdown issues (https://github.com/google/grain/issues/398)
        num_workers=0,
        prefetch_buffer_size=1,
        seed=args.seed,
    )
    dataloader = iter(dataloader)
    video_batch_BSHWC = next(dataloader)
    gt_video = jnp.asarray(video_batch_BSHWC, dtype=jnp.float32) / 255.0
    video_batch_BSHWC = gt_video.astype(args.dtype)
    # Get latent actions for all videos in the batch
    batch = dict(videos=video_batch_BSHWC)
    action_batch_E = genie.vq_encode(batch, training=False)

    # --- Sample + evaluate video ---
    recon_video_BSHWC = _autoreg_sample(rng, video_batch_BSHWC, action_batch_E)
    recon_video_BSHWC = recon_video_BSHWC.astype(jnp.float32)
    gt = gt_video[:, : recon_video_BSHWC.shape[1]].clip(0, 1).reshape(-1, *gt_video.shape[2:])
    recon = recon_video_BSHWC.clip(0, 1).reshape(-1, *recon_video_BSHWC.shape[2:])
    ssim = jnp.asarray(
        pix.ssim(gt[:, args.start_frame:], recon[:, args.start_frame:])
    ).mean()
    print(f"SSIM: {ssim}")

    # --- Construct video ---
    true_videos = (gt_video * 255).astype(np.uint8)
    pred_videos = (recon_video_BSHWC * 255).astype(np.uint8)
    video_comparison = np.zeros((2, *recon_video_BSHWC.shape), dtype=np.uint8)
    video_comparison[0] = true_videos[:, : args.seq_len]
    video_comparison[1] = pred_videos
    frames = einops.rearrange(video_comparison, "n b t h w c -> t (b h) (n w) c")

    # --- Save video ---
    imgs = [Image.fromarray(img) for img in frames]
    # Write actions on each frame, on each row (i.e., for each video in the batch, on the GT row)
    B, S, _, _, _ = video_batch_BSHWC.shape
    action_batch_BSm11 = jnp.reshape(action_batch_E, (B, S-1, 1))
    for t, img in enumerate(imgs[1:]):
        d = ImageDraw.Draw(img)
        for row in range(action_batch_BSm11.shape[0]):
            action = action_batch_BSm11[row, t, 0]
            y_offset = row * video_batch_BSHWC.shape[2] + 2
            d.text((2, y_offset), f"{action}", fill=255)
    imgs[0].save(
        f"generation_{time.time()}.gif",
        save_all=True,
        append_images=imgs[1:],
        duration=250,
        loop=0,
    )
