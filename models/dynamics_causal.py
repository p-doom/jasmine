from typing import Dict

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from utils.nn import STTransformer


class DynamicsCausal(nnx.Module):
    """Causal dynamics model"""

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_latents: int,
        latent_action_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        decode: bool,
        rngs: nnx.Rngs,
    ):
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.num_latents = num_latents
        self.latent_action_dim = latent_action_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.decode = decode

        self.transformer = STTransformer(
            self.model_dim,
            self.model_dim,
            self.ffn_dim,
            self.num_latents,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=True,
            decode=self.decode,
            rngs=rngs,
        )
        self.patch_embed = nnx.Embed(self.num_latents, self.model_dim, rngs=rngs)
        self.action_up = nnx.Linear(
            self.latent_action_dim,
            self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

    def __call__(
        self, batch: Dict[str, jax.Array], training: bool = True
    ) -> tuple[jax.Array, jax.Array | None]:
        vid_embed = self.patch_embed(batch["video_tokens"])
        act_embed = self.action_up(batch["latent_actions"])
        vid_embed += jnp.pad(act_embed, ((0, 0), (1, 0), (0, 0), (0, 0)))
        vid_embed_padded = jnp.pad(vid_embed, ((0, 0), (0, 0), (1, 0), (0, 0)))

        logits = self.transformer(vid_embed_padded)[:, :, :-1]

        mask = jnp.ones(vid_embed.shape[:-1])
        return logits, mask
