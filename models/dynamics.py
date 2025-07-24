from typing import Dict, Any

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from utils.nn import STTransformer


class DynamicsMaskGIT(nnx.Module):
    """MaskGIT dynamics model"""

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_latents: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        mask_limit: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
    ):
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.num_latents = num_latents
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.mask_limit = mask_limit
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

        self.dynamics = STTransformer(
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
            rngs=rngs,
        )
        self.patch_embed = nnx.Embed(self.num_latents, self.model_dim, rngs=rngs)
        self.mask_token = nnx.Param(
            nnx.initializers.lecun_uniform()(rngs.params(), (1, 1, 1, self.model_dim))
        )
        self.action_up = nnx.Linear(
            self.num_latents,
            self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

    def __call__(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        # --- Mask videos ---
        vid_embed = self.patch_embed(batch["video_tokens"])
        if training:
            rng1, rng2 = jax.random.split(batch["mask_rng"])
            mask_prob = jax.random.uniform(rng1, minval=self.mask_limit)
            mask = jax.random.bernoulli(rng2, mask_prob, vid_embed.shape[:-1])
            mask = mask.at[:, 0].set(False)
            vid_embed = jnp.where(
                jnp.expand_dims(mask, -1), self.mask_token.value, vid_embed
            )
        else:
            mask = None

        # --- Predict transition ---
        act_embed = self.action_up(batch["latent_actions"])
        vid_embed += jnp.pad(act_embed, ((0, 0), (1, 0), (0, 0), (0, 0)))
        logits = self.dynamics(vid_embed)
        return dict(token_logits=logits, mask=mask)
