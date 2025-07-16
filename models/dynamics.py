from typing import Dict, Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import einops

from utils.nn import STTransformer


class DynamicsMaskGIT(nn.Module):
    """MaskGIT dynamics model"""

    model_dim: int
    num_latents: int
    num_blocks: int
    num_heads: int
    dropout: float
    mask_limit: float
    param_dtype: jnp.dtype
    dtype: jnp.dtype

    def setup(self):
        self.dynamics = STTransformer(
            self.model_dim,
            self.num_latents,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            spacial_bert=True,
        )
        self.patch_embed = nn.Embed(self.num_latents, self.model_dim)
        self.mask_token = self.param(
            "mask_token",
            nn.initializers.lecun_uniform(),
            (1, 1, 1, self.model_dim),
        )
        self.action_up = nn.Dense(
            self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        # --- Mask videos ---
        vid_embed = self.patch_embed(batch["video_tokens"])
        if training:
            rng1, rng2 = jax.random.split(batch["mask_rng"])
            mask_prob = jax.random.uniform(rng1, minval=self.mask_limit)
            mask = jax.random.bernoulli(rng2, mask_prob, vid_embed.shape[:-1])
            mask = mask.at[:, 0].set(False)
            vid_embed = jnp.where(jnp.expand_dims(mask, -1), self.mask_token, vid_embed)
        else:
            mask = None

        # --- Predict transition ---
        act_embed = self.action_up(batch["latent_actions"])
        vid_embed += jnp.pad(act_embed, ((0, 0), (1, 0), (0, 0), (0, 0)))
        logits = self.dynamics(vid_embed)
        return dict(token_logits=logits, mask=mask)


class DynamicsAutoregressive(nn.Module):
    """Autoregressive (causal) dynamics model"""

    model_dim: int
    num_latents: int
    num_blocks: int
    num_heads: int
    dropout: float
    param_dtype: jnp.dtype
    dtype: jnp.dtype

    def setup(self):
        self.dynamics = STTransformer(
            self.model_dim,
            self.num_latents,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            spacial_bert=False,
        )
        self.patch_embed = nn.Embed(self.num_latents, self.model_dim)
        self.action_up = nn.Dense(
            self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        vid_embed = self.patch_embed(batch["video_tokens"])
        act_embed = self.action_up(batch["latent_actions"])
        vid_embed += jnp.pad(act_embed, ((0, 0), (1, 0), (0, 0), (0, 0)))
        logits = self.dynamics(vid_embed)
        mask = jnp.ones(vid_embed.shape[:-1])
        return dict(token_logits=logits, mask=mask)