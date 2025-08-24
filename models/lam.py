from typing import Dict

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from utils.preprocess import patchify, unpatchify
from utils.nn import STTransformer, VectorQuantizer


class LatentActionModel(nnx.Module):
    """Latent Action ST-ViVit VQ-VAE

    Dimension keys:
        B: batch size
        T: sequence length
        N: number of patches per frame
        M: model dimension
        L: latent dimension
        E: B * (T - 1)
        H: height
        W: width
        C: number of channels (n_dim)
        P: patch token dimension (patch_size^2 * C)

        Tm1: T - 1
        Np1: N + 1
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

        self.patch_token_dim = self.in_dim * self.patch_size**2
        self.encoder = STTransformer(
            self.patch_token_dim,
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
        self.action_in = nnx.Param(
            nnx.initializers.lecun_uniform()(
                rngs.params(), (1, 1, 1, self.patch_token_dim)
            )
        )
        self.vq = VectorQuantizer(
            self.latent_dim,
            self.num_latents,
            self.codebook_dropout,
            self.dtype,
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
            rngs=rngs,
        )

    def __call__(
        self, batch: Dict[str, jax.Array], training: bool = True
    ) -> Dict[str, jax.Array]:
        # --- Encode + VQ ---
        H, W = batch["videos"].shape[2:4]
        videos_BTHWC = batch["videos"]
        outputs = self.vq_encode(videos_BTHWC, training)
        patch_BTNP = outputs["patches"]
        z_q_BTm11L = outputs["z_q"]
        action_BTm11M = self.action_up(z_q_BTm11L)
        patch_BTm1NM = self.patch_up(patch_BTNP[:, :-1])
        action_BTm1NM = jnp.broadcast_to(action_BTm11M, patch_BTm1NM.shape)
        video_action_patches_BTm1NM = action_BTm1NM + patch_BTm1NM
        del outputs["patches"], patch_BTNP, patch_BTm1NM

        # --- Decode ---
        video_recon_BTm1P = self.decoder(video_action_patches_BTm1NM)
        video_recon_BTm1P = video_recon_BTm1P.astype(jnp.float32)
        video_recon_BTm1P = nnx.sigmoid(video_recon_BTm1P)
        video_recon_BTm1P = video_recon_BTm1P.astype(self.dtype)
        video_recon_BTHWC = unpatchify(video_recon_BTm1P, self.patch_size, H, W)
        outputs["recon"] = video_recon_BTHWC
        return outputs

    def vq_encode(
        self, videos_BTHWC: jax.Array, training: bool = True
    ) -> Dict[str, jax.Array]:
        # --- Preprocess videos ---
        B, T = videos_BTHWC.shape[:2]
        patch_BTNP = patchify(videos_BTHWC, self.patch_size)
        action_pad_BT1P = jnp.broadcast_to(
            self.action_in.value, (B, T, 1, self.patch_token_dim)
        )
        padded_patch_BTNp1P = jnp.concatenate((action_pad_BT1P, patch_BTNP), axis=2)

        # --- Encode ---
        z_BTNp1L = self.encoder(padded_patch_BTNp1P)
        # Get latent action for all future frames
        z_BTm1L = z_BTNp1L[:, 1:, 0]

        # --- Vector quantize ---
        z_EL = z_BTm1L.reshape(B * (T - 1), self.latent_dim)
        z_q_EL, z_EL, emb_EL, indices_E = self.vq(z_EL, training)
        z_q_BTm11L = z_q_EL.reshape(B, T - 1, 1, self.latent_dim)
        return dict(
            patches=patch_BTNP, z_q=z_q_BTm11L, z=z_EL, emb=emb_EL, indices=indices_E
        )
