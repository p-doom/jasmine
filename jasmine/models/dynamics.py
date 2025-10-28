from jax._src.basearray import Array
from typing import Dict

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from jasmine.utils.nn import STTransformer, Transformer, DiffusionTransformer


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
        decode: bool,
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
            decode=self.decode,
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
        rngs: nnx.Rngs,
        decode: bool,
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
            rngs=rngs,
            decode=self.decode,
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
        model_dim: int,
        ffn_dim: int,
        latent_patch_dim: int,
        latent_action_dim: int,
        num_blocks: int,
        num_heads: int,
        denoise_steps: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
        decode: bool,
    ):
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.latent_patch_dim = latent_patch_dim
        self.latent_action_dim = latent_action_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.denoise_steps = denoise_steps
        self.decode = decode
        self.diffusion_transformer = DiffusionTransformer(
            self.latent_patch_dim,
            self.model_dim,
            self.ffn_dim,
            self.latent_patch_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            use_flash_attention=self.use_flash_attention,
            rngs=rngs,
            decode=self.decode,
        )
        self.action_up = nnx.Linear(
            self.latent_action_dim,
            self.latent_patch_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.timestep_embed = nnx.Embed(
            num_embeddings=self.denoise_steps,
            features=self.latent_patch_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        batch: Dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        # Code adapted from https://github.com/kvfrans/shortcut-models/blob/main/baselines/targets_naive.py
        _rng_time, _rng_noise = jax.random.split(batch["rng"])
        latents_BTNL = batch["token_latents"]
        latent_actions_BTm11L = batch["latent_actions"]
        B, T, N, L = latents_BTNL.shape

        # --- Add noise to latents ---
        denoise_step_BT = jax.random.randint(
            _rng_time, (B, T), minval=0, maxval=self.denoise_steps
        )
        denoise_step_embed_BT1L = self.timestep_embed(denoise_step_BT).reshape(
            B, T, 1, self.latent_patch_dim
        )
        denoise_t_BT = denoise_step_BT / self.denoise_steps
        denoise_t_BT11 = denoise_t_BT[:, :, None, None]
        noise_BTNL = jax.random.normal(_rng_noise, (B, T, N, L))
        noised_latents_BTNL = (
            1 - (1 - 1e-5) * denoise_t_BT11
        ) * noise_BTNL + denoise_t_BT11 * latents_BTNL

        # --- Process actions ---
        act_embed_BTm11L = self.action_up(latent_actions_BTm11L)
        padded_act_embed_BT1L: Array = jnp.pad(
            act_embed_BTm11L, ((0, 0), (1, 0), (0, 0), (0, 0))
        )

        # --- Call the diffusion transformer ---
        inputs_BTNp2L = jnp.concatenate(
            [padded_act_embed_BT1L, denoise_step_embed_BT1L, noised_latents_BTNL],
            axis=2,
        )
        outputs_BTNp2L = self.diffusion_transformer(
            inputs_BTNp2L,
        )
        pred_latents_BTNL = outputs_BTNp2L[:, :, 2:]
        return pred_latents_BTNL, denoise_t_BT
