from typing import Dict

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from utils.nn import STTransformer, Transformer


class DynamicsMaskGIT(nnx.Module):
    """
    MaskGIT dynamics model

    Dimension keys:
        B: batch size
        T: sequence length
        N: number of patches per frame
        L: latent dimension
        V: vocabulary size (number of latents)
    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_latents: int,
        latent_action_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        max_noise_level: float,
        noise_buckets: int,
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
        self.max_noise_level = max_noise_level
        self.noise_buckets = noise_buckets
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
        self.noise_level_embed = nnx.Embed(
            self.noise_buckets, self.model_dim, rngs=rngs
        )

    def __call__(
        self,
        batch: Dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        # --- Mask videos ---
        video_tokens_BTN = batch["video_tokens"]
        latent_actions_BTm11L = batch["latent_actions"]
        vid_embed_BTNM = self.patch_embed(video_tokens_BTN)

        B, T, N, M = vid_embed_BTNM.shape
        rng, _rng_prob, *_rngs_mask = jax.random.split(batch["mask_rng"], B + 2)
        mask_prob = jax.random.uniform(_rng_prob, shape=(B,), minval=self.mask_limit)
        per_sample_shape = vid_embed_BTNM.shape[1:-1]
        mask = jax.vmap(
            lambda rng, prob: jax.random.bernoulli(rng, prob, per_sample_shape),
            in_axes=(0, 0),
        )(jnp.asarray(_rngs_mask), mask_prob)
        mask = mask.at[:, 0].set(False)
        vid_embed_BTNM = jnp.where(
            jnp.expand_dims(mask, -1), self.mask_token.value, vid_embed_BTNM
        )

        # --- Sample noise ---
        rng, _rng_noise_lvl, _rng_noise = jax.random.split(rng, 3)
        noise_level_B = jax.random.uniform(
            _rng_noise_lvl, shape=(B,), minval=0.0, maxval=self.max_noise_level
        )
        noise_BTNM = jax.random.normal(_rng_noise, shape=(B, T, N, M))
        # We calculate `(noise_level * noise_buckets) / max_noise_level` instead of
        # `(noise_level_B / max_noise_level) * noise_buckets` for numerical stability.
        noise_bucket_idx_B = jnp.floor(
            (noise_level_B * self.noise_buckets) / self.max_noise_level
        ).astype(jnp.int32)
        noise_bucket_idx_B11 = noise_bucket_idx_B.reshape(B, 1, 1)
        noise_level_embed_B11M = self.noise_level_embed(noise_bucket_idx_B11)
        noise_level_embed_BT1M = jnp.tile(noise_level_embed_B11M, (1, T, 1, 1))
        noise_level_B111 = noise_level_B.reshape(B, 1, 1, 1)

        # safe sqrt: clip argument to >= 0
        one_minus_noise = jnp.clip(1.0 - noise_level_B111, a_min=0.0)
        sqrt_one_minus = jnp.sqrt(one_minus_noise)
        sqrt_noise = jnp.sqrt(jnp.clip(noise_level_B111, a_min=0.0))

        vid_embed_BTNM = sqrt_one_minus * vid_embed_BTNM + sqrt_noise * noise_BTNM
        # --- Predict transition ---
        act_embed_BTm11M = self.action_up(latent_actions_BTm11L)
        padded_act_embed_BT1M = jnp.pad(
            act_embed_BTm11M, ((0, 0), (1, 0), (0, 0), (0, 0))
        )
        vid_embed_BTNp2M = jnp.concatenate(
            [padded_act_embed_BT1M, noise_level_embed_BT1M, vid_embed_BTNM], axis=2
        )
        logits_BTNp2V = self.transformer(vid_embed_BTNp2M)
        logits_BTNV = logits_BTNp2V[:, :, 2:]
        return logits_BTNV, mask


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

        self.transformer = Transformer(
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
        self,
        batch: Dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        video_tokens_BTN = batch["video_tokens"]
        latent_actions_BTm11L = batch["latent_actions"]
        vid_embed_BTNM = self.patch_embed(video_tokens_BTN)
        act_embed_BTm11M = self.action_up(latent_actions_BTm11L)
        padded_act_embed_BT1M = jnp.pad(
            act_embed_BTm11M, ((0, 0), (1, 0), (0, 0), (0, 0))
        )
        vid_embed_BTNp1M = jnp.concatenate(
            [padded_act_embed_BT1M, vid_embed_BTNM], axis=2
        )
        logits_BTNp1V = self.transformer(vid_embed_BTNp1M)
        logits_BTNV = logits_BTNp1V[:, :, :-1]
        return logits_BTNV, jnp.ones_like(video_tokens_BTN)
