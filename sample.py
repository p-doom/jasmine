from dataclasses import dataclass
import time
import os

import dm_pix as pix
import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from orbax.checkpoint import PyTreeCheckpointer
from PIL import Image, ImageDraw
import tyro

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
    num_actions: int = 6
    lam_patch_size: int = 16
    lam_num_blocks: int = 4
    lam_num_heads: int = 8
    # Dynamics checkpoint
    dyna_dim: int = 512
    dyna_ffn_dim: int = 2048
    dyna_num_blocks: int = 6
    dyna_num_heads: int = 8
    param_dtype: jnp.dtype = jnp.float32
    dtype: jnp.dtype = jnp.bfloat16
    use_flash_attention: bool = True
    use_gt_actions: bool = False


args = tyro.cli(Args)
rng = jax.random.PRNGKey(args.seed)

# --- Load Genie checkpoint ---
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
    lam_co_train=False,
    # Dynamics
    dyna_dim=args.dyna_dim,
    dyna_ffn_dim=args.dyna_ffn_dim,
    dyna_num_blocks=args.dyna_num_blocks,
    dyna_num_heads=args.dyna_num_heads,
    param_dtype=args.param_dtype,
    dtype=args.dtype,
    use_flash_attention=args.use_flash_attention,
    use_gt_actions=args.use_gt_actions,
)
rng, _rng = jax.random.split(rng)
image_shape = (args.image_height, args.image_width, args.image_channels)
dummy_inputs = dict(
    videos=jnp.zeros((args.batch_size, args.seq_len, *image_shape), dtype=jnp.float32),
    mask_rng=_rng,
)
rng, _rng = jax.random.split(rng)
params = genie.init(_rng, dummy_inputs)
ckpt = PyTreeCheckpointer().restore(args.checkpoint)["model"]["params"]["params"]
params["params"].update(ckpt)


def _sampling_wrapper(module, batch):
    return module.sample(batch, args.seq_len, args.maskgit_steps, args.temperature, args.sample_argmax)

# --- Define autoregressive sampling loop ---
def _autoreg_sample(rng, video_batch, action_batch):
    vid = video_batch[:, : args.start_frame + 1]
    sampling_fn = jax.jit(nn.apply(_sampling_wrapper, genie)) 
    rng, _rng = jax.random.split(rng)
    batch = dict(videos=vid, actions=action_batch, rng=_rng)
    generated_vid = sampling_fn(
        params,
        batch
    )
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
    num_workers=8,
    prefetch_buffer_size=1,
    seed=args.seed,
)
video_batch = next(iter(dataloader))
# Get actions for all videos in the batch
if args.use_gt_actions:
    # FIXME (f.srambical): use the actions from the dataset annotations instead of dummy actions
    rng, _rng = jax.random.split(rng)
    action_batch = jax.random.randint(
        _rng, 
        shape=(video_batch.shape[0], args.seq_len - 1, 1), 
        minval=0, 
        maxval=args.num_actions,
        dtype=jnp.int32
    )
else:
    batch = dict(videos=video_batch)
    action_batch = genie.apply(params, batch, False, method=Genie.vq_encode)
    action_batch = action_batch.reshape(video_batch.shape[0], args.seq_len - 1, 1)

# --- Sample + evaluate video ---
vid = _autoreg_sample(rng, video_batch, action_batch)
gt = video_batch[:, : vid.shape[1]].clip(0, 1).reshape(-1, *video_batch.shape[2:])
recon = vid.clip(0, 1).reshape(-1, *vid.shape[2:])
ssim = pix.ssim(gt[:, args.start_frame + 1 :], recon[:, args.start_frame + 1 :]).mean()
print(f"SSIM: {ssim}")

# --- Construct video ---
true_videos = (video_batch * 255).astype(np.uint8)
pred_videos = (vid * 255).astype(np.uint8)
video_comparison = np.zeros((2, *vid.shape), dtype=np.uint8)
video_comparison[0] = true_videos[:, :args.seq_len]
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
