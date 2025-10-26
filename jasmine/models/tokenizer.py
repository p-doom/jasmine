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
            self.dtype,
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


class TokenizerMAE(nnx.Module):
    """
    ST-ViVit MAE.
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
        max_mask_ratio: float,
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
        self.max_mask_ratio = max_mask_ratio
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.encoder = STTransformer(
            in_dim * patch_size**2,
            model_dim,
            ffn_dim,
            latent_dim,
            num_blocks,
            num_heads,
            dropout,
            param_dtype,
            dtype,
            use_flash_attention,
            rngs,
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
        self.mask_patch = nnx.Param(
            nnx.initializers.lecun_uniform()(
                rngs.params(), (1, 1, 1, self.in_dim * self.patch_size**2)
            )
        )

    def __call__(
        self, batch: Dict[str, jax.Array], training: bool = True
    ) -> Dict[str, jax.Array]:
        H, W = batch["videos"].shape[2:4]
        videos_BTHWC = batch["videos"]
        outputs = self.mask_and_encode(videos_BTHWC, batch["rng"])
        z_BTNL = outputs["z"]
        recon_BTHWC = self.decoder(z_BTNL)
        recon_BTHWC = recon_BTHWC.astype(jnp.float32)
        recon_BTHWC = nnx.sigmoid(recon_BTHWC)
        recon_BTHWC = recon_BTHWC.astype(self.dtype)
        recon_BTHWC = unpatchify(recon_BTHWC, self.patch_size, H, W)
        outputs["recon"] = recon_BTHWC
        return outputs

    def mask_and_encode(self, videos: jax.Array, rng: nnx.Rngs) -> Dict[str, jax.Array]:
        # --- Mask and encode ---
        B, T = videos.shape[:2]
        patch_BTNP = patchify(videos, self.patch_size)
        N = patch_BTNP.shape[2]

        # randomly mask patches
        _rng_prob, *_rngs_mask = jax.random.split(rng, B + 1)

        mask_prob = jax.random.uniform(
            _rng_prob, shape=(B,), minval=0, maxval=self.max_mask_ratio
        )
        mask_BTN = jax.vmap(
            lambda rng, prob: jax.random.bernoulli(rng, prob, (T, N)),
            in_axes=(0, 0),
        )(jnp.asarray(_rngs_mask), mask_prob)

        masked_patch_BTNP = jnp.where(
            mask_BTN[..., None], self.mask_patch.value, patch_BTNP
        )
        z_BTNL = self.encoder(masked_patch_BTNP)
        z_BTNL = nnx.tanh(z_BTNL)
        outputs = dict(z=z_BTNL, mask=mask_BTN)
        return outputs

    def decode(self, z_BTNL: jax.Array, video_hw: Tuple[int, int]) -> jax.Array:
        recon_BTNP = self.decoder(z_BTNL)
        recon_BTNP = recon_BTNP.astype(jnp.float32)
        recon_BTNP = nnx.sigmoid(recon_BTNP)
        recon_BTNP = recon_BTNP.astype(self.dtype)
        return unpatchify(recon_BTNP, self.patch_size, *video_hw)
