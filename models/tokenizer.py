from typing import Dict, Tuple

import flax.nnx as nnx
import jax.numpy as jnp
import jax

from utils.preprocess import patchify, unpatchify
from utils.nn import STTransformer, VectorQuantizer


class TokenizerVQVAE(nnx.Module):
    """
    ST-ViVit VQ-VAE

    Dimension keys:
        B: batch size
        T: sequence length
        N: number of patches per frame
        L: latent dimension
        D: B * T * N
        H: height
        W: width
        C: number of channels
        P: patch token dimension (patch_size^2 * C)
    """

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
            rngs=rngs,
        )

    def __call__(
        self, batch: Dict[str, jax.Array], training: bool = True
    ) -> Dict[str, jax.Array]:
        H, W = batch["videos"].shape[2:4]
        videos_BTHWC = batch["videos"]
        outputs = self.vq_encode(videos_BTHWC, training)
        z_q_BTNL = outputs["z_q"]
        recon_BTHWC = self.decoder(z_q_BTNL)
        recon_BTHWC = recon_BTHWC.astype(jnp.float32)
        recon_BTHWC = nnx.sigmoid(recon_BTHWC)
        recon_BTHWC = recon_BTHWC.astype(self.dtype)
        recon_BTHWC = unpatchify(recon_BTHWC, self.patch_size, H, W)
        outputs["recon"] = recon_BTHWC
        return outputs

    def vq_encode(
        self, videos: jax.Array, training: bool = True
    ) -> Dict[str, jax.Array]:
        # --- Preprocess + encode ---
        B, T = videos.shape[:2]
        patch_BTNP = patchify(videos, self.patch_size)
        N = patch_BTNP.shape[2]
        x_BTNL = self.encoder(patch_BTNP)

        # --- Vector quantize ---
        x_DL = x_BTNL.reshape(B * T * N, self.latent_dim)
        z_q_DL, z_DL, emb_DL, indices_D = self.vq(x_DL, training)
        z_q_BTNL = z_q_DL.reshape(B, T, N, self.latent_dim)
        indices_BTN = indices_D.reshape(B, T, N)
        return dict(z_q=z_q_BTNL, z=z_DL, emb=emb_DL, indices=indices_BTN)

    def decode(self, indices_BTN: jax.Array, video_hw: Tuple[int, int]) -> jax.Array:
        z_BTNL = self.vq.codebook[indices_BTN]
        recon_BTNP = self.decoder(z_BTNL)
        recon_BTNP = recon_BTNP.astype(jnp.float32)
        recon_BTNP = nnx.sigmoid(recon_BTNP)
        recon_BTNP = recon_BTNP.astype(self.dtype)
        return unpatchify(recon_BTNP, self.patch_size, *video_hw)
