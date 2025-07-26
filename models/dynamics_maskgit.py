from typing import Dict

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
        latent_action_dim: int,
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
        self.latent_action_dim = latent_action_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.mask_limit = mask_limit
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

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
            spatial_causal=False,
            decode=False,
            rngs=rngs,
        )
        self.patch_embed = nnx.Embed(self.num_latents, self.model_dim, rngs=rngs)
        self.mask_token = nnx.Param(
            nnx.initializers.lecun_uniform()(rngs.params(), (1, 1, 1, self.model_dim))
        )
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
        # --- Mask videos ---
        vid_embed = self.patch_embed(batch["video_tokens"])
        if training:
            batch_size = vid_embed.shape[0]
            _rng_prob, *_rngs_mask = jax.random.split(batch["mask_rng"], batch_size + 1)
            mask_prob = jax.random.uniform(
                _rng_prob, shape=(batch_size,), minval=self.mask_limit
            )
            per_sample_shape = vid_embed.shape[1:-1]
            mask = jax.vmap(
                lambda rng, prob: jax.random.bernoulli(rng, prob, per_sample_shape),
                in_axes=(0, 0),
            )(jnp.asarray(_rngs_mask), mask_prob)
            mask = mask.at[:, 0].set(False)
            vid_embed = jnp.where(
                jnp.expand_dims(mask, -1), self.mask_token.value, vid_embed
            )
        else:
            mask = None

        # --- Predict transition ---
        act_embed = self.action_up(batch["latent_actions"])
        vid_embed += jnp.pad(act_embed, ((0, 0), (1, 0), (0, 0), (0, 0)))
        logits = self.transformer(vid_embed)
        return logits, mask
