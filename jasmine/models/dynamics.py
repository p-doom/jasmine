from typing import Dict

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from utils.nn import STTransformer, Transformer, DiffusionTransformer


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
        self,
        batch: Dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        # --- Mask videos ---
        video_tokens_BTN = batch["video_tokens"]
        latent_actions_BTm11L = batch["latent_actions"]
        vid_embed_BTNM = self.patch_embed(video_tokens_BTN)

        batch_size = vid_embed_BTNM.shape[0]
        _rng_prob, *_rngs_mask = jax.random.split(batch["mask_rng"], batch_size + 1)
        mask_prob = jax.random.uniform(
            _rng_prob, shape=(batch_size,), minval=self.mask_limit
        )
        per_sample_shape = vid_embed_BTNM.shape[1:-1]
        mask = jax.vmap(
            lambda rng, prob: jax.random.bernoulli(rng, prob, per_sample_shape),
            in_axes=(0, 0),
        )(jnp.asarray(_rngs_mask), mask_prob)
        mask = mask.at[:, 0].set(False)
        vid_embed_BTNM = jnp.where(
            jnp.expand_dims(mask, -1), self.mask_token.value, vid_embed_BTNM
        )

        # --- Predict transition ---
        act_embed_BTm11M = self.action_up(latent_actions_BTm11L)
        padded_act_embed_BT1M = jnp.pad(
            act_embed_BTm11M, ((0, 0), (1, 0), (0, 0), (0, 0))
        )
        vid_embed_BTNp1M = jnp.concatenate(
            [padded_act_embed_BT1M, vid_embed_BTNM], axis=2
        )
        logits_BTNp1V = self.transformer(vid_embed_BTNp1M)
        logits_BTNV = logits_BTNp1V[:, :, 1:]
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


class DynamicsDiffusion(nnx.Module):
    """Diffusion transformer dynamics model"""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        ffn_dim: int,
        out_dim: int,
        num_latents: int,
        latent_action_dim: int,
        num_blocks: int,
        num_heads: int,
        denoise_steps: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        decode: bool,
        rngs: nnx.Rngs,
    ):
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.out_dim = out_dim
        self.num_latents = num_latents
        self.latent_action_dim = latent_action_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.decode = decode
        self.denoise_steps = denoise_steps
        self.diffusion_transformer = DiffusionTransformer(
            input_dim=self.input_dim,
            model_dim=self.model_dim,
            ffn_dim=self.ffn_dim,
            out_dim=self.out_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            decode=self.decode,
            rngs=rngs,
        )
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
        # For now we only denoise the last frame
        rng, time_rng, noise_rng = jax.random.split(batch["rng"], 3)
        token_latents_BTNL = batch["token_latents"]
        latent_actions_BTm11L = batch["latent_actions"]
        B, T, N, L = token_latents_BTNL.shape
        # Sample t.
        t = jax.random.randint(time_rng, (B, T), minval=0, maxval=self.denoise_steps)
        t /= self.denoise_steps
        t_full = t[:, :, None, None]  # [B, T, 1, 1]
        x_1 = token_latents_BTNL
        x_0 = jax.random.normal(noise_rng, (B, T, N, L))
        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        dt_flow = jnp.log2(self.denoise_steps).astype(jnp.int32)
        dt_base = jnp.ones((B, T), dtype=jnp.int32) * dt_flow  # [B, T]
        token_latents_BTNL = x_t

        act_embed_BTm11M = self.action_up(latent_actions_BTm11L)
        padded_act_embed_BTM = jnp.pad(
            act_embed_BTm11M, ((0, 0), (1, 0), (0, 0), (0, 0))
        ).reshape(B, T, self.model_dim)

        # call the diffusion transformer
        # TODO add action conditioning
        x_pred = self.diffusion_transformer(
            token_latents_BTNL, t, dt_base, padded_act_embed_BTM
        )
        return x_pred, t
