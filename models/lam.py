from typing import Dict

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from utils.preprocess import patchify, unpatchify
from utils.nn import STTransformer, VectorQuantizer


class LatentActionModel(nnx.Module):
    """Latent Action ST-ViVit VQ-VAE"""

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

        self.patch_token_dim = self.in_dim * self.patch_size**2
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
        self.action_in = nnx.Param(
            nnx.initializers.lecun_uniform()(
                rngs.params(), (1, 1, 1, self.patch_token_dim)
            )
        )
        self.vq = VectorQuantizer(
            self.latent_dim,
            self.num_latents,
            self.codebook_dropout,
            rngs=rngs,
        )
        self.patch_up = nnx.Linear(
            self.patch_token_dim,
            self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.action_up = nnx.Linear(
            self.latent_dim,
            self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.decoder = STTransformer(
            self.model_dim,
            self.model_dim,
            self.ffn_dim,
            self.patch_token_dim,
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
        video_recon = nnx.sigmoid(video_recon)
        video_recon = video_recon.astype(self.dtype)
        outputs["recon"] = unpatchify(video_recon, self.patch_size, H, W)
        return outputs

    def vq_encode(
        self, videos: jax.Array, training: bool = True
    ) -> Dict[str, jax.Array]:
        # --- Preprocess videos ---
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        action_pad = jnp.broadcast_to(
            self.action_in.value, (B, T, 1, self.patch_token_dim)
        )
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
