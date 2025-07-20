from typing import Dict, Any

import jax.numpy as jnp
import flax.linen as nn

from utils.preprocess import patchify, unpatchify
from utils.nn import STTransformer, VectorQuantizer


class LatentActionModel(nn.Module):
    """Latent Action ST-ViVit VQ-VAE"""

    in_dim: int
    model_dim: int
    ffn_dim: int
    latent_dim: int
    num_latents: int
    patch_size: int
    num_blocks: int
    num_heads: int
    dropout: float
    codebook_dropout: float
    param_dtype: jnp.dtype
    dtype: jnp.dtype
    use_flash_attention: bool

    def setup(self):
        self.patch_token_dim = self.in_dim * self.patch_size**2
        self.encoder = STTransformer(
            self.model_dim,
            self.ffn_dim,
            self.latent_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            use_flash_attention=self.use_flash_attention,
        )
        self.action_in = self.param(
            "action_in",
            nn.initializers.lecun_uniform(),
            (1, 1, 1, self.patch_token_dim),
        )
        self.vq = VectorQuantizer(
            self.latent_dim,
            self.num_latents,
            self.codebook_dropout,
        )
        self.patch_up = nn.Dense(
            self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.action_up = nn.Dense(
            self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.decoder = STTransformer(
            self.model_dim,
            self.ffn_dim,
            self.patch_token_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            use_flash_attention=self.use_flash_attention,
        )

    def __call__(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        # --- Encode + VQ ---
        H, W = batch["videos"].shape[2:4]
        outputs = self.vq_encode(batch["videos"], training)
        video_action_patches = self.action_up(outputs["z_q"]) + self.patch_up(
            outputs["patches"][:, :-1]
        )
        del outputs["patches"]

        # --- Decode ---
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon.astype(jnp.float32)
        video_recon = nn.sigmoid(video_recon)
        video_recon = video_recon.astype(self.dtype)
        outputs["recon"] = unpatchify(video_recon, self.patch_size, H, W)
        return outputs

    def vq_encode(self, videos: Any, training: bool = True) -> Dict[str, Any]:
        # --- Preprocess videos ---
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        action_pad = jnp.broadcast_to(self.action_in, (B, T, 1, self.patch_token_dim))
        padded_patches = jnp.concatenate((action_pad, patches), axis=2)

        # --- Encode ---
        z = self.encoder(padded_patches)  # (B, T, N, E)
        # Get latent action for all future frames
        z = z[:, 1:, 0]  # (B, T-1, E)

        # --- Vector quantize ---
        z = z.reshape(B * (T - 1), self.latent_dim)
        z_q, z, emb, indices = self.vq(z, training)
        z_q = z_q.reshape(B, T - 1, 1, self.latent_dim)
        return dict(patches=patches, z_q=z_q, z=z, emb=emb, indices=indices)
