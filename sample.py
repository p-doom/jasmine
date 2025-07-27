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

from jasmine import Jasmine
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
    lam_co_train: bool = False
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
    dynamics_type: str = "maskgit"


args = tyro.cli(Args)

if __name__ == "__main__":
    jax.distributed.initialize()

    rng = jax.random.PRNGKey(args.seed)

    # --- Load Dynamics model checkpoint ---
    rngs = nnx.Rngs(rng)
    jasmine = Jasmine(
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
        lam_co_train=args.lam_co_train,
        # Dynamics
        dynamics_type=args.dynamics_type,
        dyna_dim=args.dyna_dim,
        dyna_ffn_dim=args.dyna_ffn_dim,
        dyna_num_blocks=args.dyna_num_blocks,
        dyna_num_heads=args.dyna_num_heads,
        param_dtype=args.param_dtype,
        dtype=args.dtype,
        use_flash_attention=args.use_flash_attention,
        decode=True,
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
    dummy_optimizer = nnx.Optimizer(jasmine, dummy_tx)

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
    def _sampling_fn(model: Jasmine, batch: dict) -> jax.Array:
        """Runs Jasmine.sample with pre-defined generation hyper-parameters."""
        if args.dynamics_type == "maskgit":
            return model.sample_maskgit(
                batch,
                args.seq_len,
                args.maskgit_steps,
                args.temperature,
                args.sample_argmax,
            )
        else:
            return model.sample_causal(
                batch,
                args.seq_len,
                args.temperature,
                args.sample_argmax,
            )

    # --- Define autoregressive sampling loop ---
    @nnx.jit
    def _autoreg_sample(rng, video_batch, action_batch):
        vid = video_batch[:, : args.start_frame + 1]
        rng, _rng = jax.random.split(rng)
        batch = dict(videos=vid, latent_actions=action_batch, rng=_rng)
        generated_vid = _sampling_fn(jasmine, batch)
        return generated_vid

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
    video_batch = next(dataloader)
    video_batch = video_batch.astype(args.dtype) / 255.0
    # Get latent actions for all videos in the batch
    batch = dict(videos=video_batch)
    action_batch = jasmine.vq_encode(batch, training=False)
    action_batch = jnp.asarray(action_batch).reshape(
        video_batch.shape[0], args.seq_len - 1, 1
    )

    # --- Sample + evaluate video ---
    # The autoregressive cache needs to be initialized with the shape of the tokenized inputs, not the raw video.
    # The number of spatial tokens is derived from the image dimensions and patch size.
    # It appears the 90x160 image is padded to 92x160, and a CLS token is added.
    # (92 // args.patch_size) * (160 // args.patch_size) + 1 = 23 * 40 + 1 = 921
    num_patches = ((args.image_height + 3) // 4 * 4 // args.patch_size) * (
        args.image_width // args.patch_size
    ) + 1
    # Shape for spatial attention: (batch, time, patches, num_heads, head_dim)
    spatial_token_shape = (
        args.batch_size,
        1,
        num_patches,
        args.dyna_dim,
    )
    # Shape for temporal attention: (batch, patches, time, num_heads, head_dim)
    temporal_token_shape = (
        args.batch_size,
        num_patches,
        1,
        args.dyna_dim,
    )
    if args.dynamics_type == "causal":
        transformer_blocks = jasmine.dynamics.transformer.blocks
        for block in transformer_blocks:
            block.spatial_attention.init_cache(spatial_token_shape, dtype=args.dtype)
            block.temporal_attention.init_cache(temporal_token_shape, dtype=args.dtype)
    vid = _autoreg_sample(rng, video_batch, action_batch)
    gt = video_batch[:, : vid.shape[1]].clip(0, 1).reshape(-1, *video_batch.shape[2:])
    recon = vid.clip(0, 1).reshape(-1, *vid.shape[2:])
    ssim = jnp.asarray(
        pix.ssim(gt[:, args.start_frame + 1 :], recon[:, args.start_frame + 1 :])
    ).mean()
    print(f"SSIM: {ssim}")

    # --- Construct video ---
    true_videos = (video_batch * 255).astype(np.uint8)
    pred_videos = (vid * 255).astype(np.uint8)
    video_comparison = np.zeros((2, *vid.shape), dtype=np.uint8)
    video_comparison[0] = true_videos[:, : args.seq_len]
    video_comparison[1] = pred_videos
    frames = einops.rearrange(video_comparison, "n b t h w c -> t (b h) (n w) c")

    # --- Save video ---
    imgs = [Image.fromarray(img) for img in frames]
    # Write actions on each frame, on each row (i.e., for each video in the batch, on the GT row)
    for t, img in enumerate(imgs[1:]):
        d = ImageDraw.Draw(img)
        for row in range(action_batch.shape[0]):
            action = action_batch[row, t, 0]
            y_offset = row * video_batch.shape[2] + 2
            d.text((2, y_offset), f"{action}", fill=255)
    imgs[0].save(
        f"generation_{time.time()}.gif",
        save_all=True,
        append_images=imgs[1:],
        duration=250,
        loop=0,
    )
