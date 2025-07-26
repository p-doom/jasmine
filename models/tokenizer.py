from typing import Dict, Tuple

import flax.nnx as nnx
import jax.numpy as jnp
import jax

from utils.preprocess import patchify, unpatchify
from utils.nn import STTransformer, VectorQuantizer


class TokenizerVQVAE(nnx.Module):
    """ST-ViVit VQ-VAE"""

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        ffn_dim: int,
        latent_dim: int,
        num_latents: int,
        patch_size: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        codebook_dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
    ):
        self.in_dim = in_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.codebook_dropout = codebook_dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

        self.encoder = STTransformer(
            self.in_dim * self.patch_size**2,
            self.model_dim,
            self.ffn_dim,
            self.latent_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            decode=False,
            rngs=rngs,
        )
        self.vq = VectorQuantizer(
            self.latent_dim,
            self.num_latents,
            self.codebook_dropout,
            rngs=rngs,
        )
        self.out_dim = self.in_dim * self.patch_size**2
        self.decoder = STTransformer(
            self.latent_dim,
            self.model_dim,
            self.ffn_dim,
            self.out_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            decode=False,
            rngs=rngs,
        )

    def __call__(
        self, batch: Dict[str, jax.Array], training: bool = True
    ) -> Dict[str, jax.Array]:
        H, W = batch["videos"].shape[2:4]
        outputs = self.vq_encode(batch["videos"], training)
        recon = self.decoder(outputs["z_q"])  # (B, T, H_down * W_down, C)
        recon = recon.astype(jnp.float32)
        recon = nnx.sigmoid(recon)
        recon = recon.astype(self.dtype)
        outputs["recon"] = unpatchify(recon, self.patch_size, H, W)
        return outputs

    def vq_encode(
        self, videos: jax.Array, training: bool = True
    ) -> Dict[str, jax.Array]:
        # --- Preprocess + encode ---
        B, T = videos.shape[:2]
        x = patchify(videos, self.patch_size)
        N = x.shape[2]
        x = self.encoder(x)  # (B, T, N, E)

        # --- Vector quantize ---
        x = x.reshape(B * T * N, self.latent_dim)
        z_q, z, emb, indices = self.vq(x, training)
        z_q = z_q.reshape(B, T, N, self.latent_dim)
        indices = indices.reshape(B, T, N)
        return dict(z_q=z_q, z=z, emb=emb, indices=indices)

    def decode(self, indices: jax.Array, video_hw: Tuple[int, int]) -> jax.Array:
        z = self.vq.codebook[indices]
        recon = self.decoder(z)
        recon = recon.astype(jnp.float32)
        recon = nnx.sigmoid(recon)
        recon = recon.astype(self.dtype)
        return unpatchify(recon, self.patch_size, *video_hw)
