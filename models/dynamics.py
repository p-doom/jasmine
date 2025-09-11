from typing import Dict

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import einops

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
        self, batch: Dict[str, jax.Array], training: bool = True, eval_full_frame_pred: bool = False,
    ) -> tuple[jax.Array, jax.Array | None]:
        assert not (training and eval_full_frame_pred), "Cannot evaluate full frame prediction during training."
        # --- Mask videos ---
        video_tokens_BTN = batch["video_tokens"]
        latent_actions_BTm11L = batch["latent_actions"]
        vid_embed_BTNM = self.patch_embed(video_tokens_BTN)
        if training:
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
        elif eval_full_frame_pred:
            mask = jnp.zeros_like(vid_embed_BTNM)
            mask = mask.at[: -1].set(True)
            vid_embed_BTNM = jnp.where(
                jnp.expand_dims(mask, -1), self.mask_token.value, vid_embed_BTNM
            )
        else:
            mask = jnp.ones_like(video_tokens_BTN)

        # --- Predict transition ---
        act_embed_BTm11M = self.action_up(latent_actions_BTm11L)
        padded_act_embed_BT1M = jnp.pad(
            act_embed_BTm11M, ((0, 0), (1, 0), (0, 0), (0, 0))
        )
        padded_act_embed_BTNM = jnp.broadcast_to(
            padded_act_embed_BT1M, vid_embed_BTNM.shape
        )
        vid_embed_BTNM += padded_act_embed_BTNM
        logits_BTNV = self.transformer(vid_embed_BTNM)
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
            decode=decode,
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
        self, batch: Dict[str, jax.Array], training: bool = True, eval_full_frame_pred: bool = False,
    ) -> tuple[jax.Array, jax.Array | None]:
        assert not (training and eval_full_frame_pred), "Cannot evaluate full frame prediction during training."
        video_tokens_BTN = batch["video_tokens"]
        latent_actions_BTm11L = batch["latent_actions"]
        if eval_full_frame_pred:
            def _eval_full_frame_pred(carry, step_n):
                video_tokens_BTN, final_logits_BTNV = carry
                vid_embed_BTNM = self.patch_embed(video_tokens_BTN)
                act_embed_BTm11M = self.action_up(latent_actions_BTm11L)
                padded_act_embed_BT1M = jnp.pad(
                    act_embed_BTm11M, ((0, 0), (1, 0), (0, 0), (0, 0))
                )
                vid_embed_BTNp1M = jnp.concatenate(
                    [padded_act_embed_BT1M, vid_embed_BTNM], axis=2
                )
                step_logits_BTNp1V = self.transformer(vid_embed_BTNp1M)
                step_logits_BV = step_logits_BTNp1V[:, -1, step_n, :]
                final_logits_BTNV = final_logits_BTNV.at[:, -1, step_n].set(step_logits_BV)
                sampled_token_idxs_B = jnp.argmax(step_logits_BV, axis=-1)
                video_tokens_BTN = video_tokens_BTN.at[:, -1, step_n].set(
                    sampled_token_idxs_B
                )
                return (video_tokens_BTN, final_logits_BTNV), None
            final_logits_BTNV = jax.lax.scan(
                _eval_full_frame_pred, 
                (video_tokens_BTN, jnp.zeros(video_tokens_BTN.shape + [self.num_latents])), 
                jnp.arange(video_tokens_BTN.shape[1])
            )
            mask_out = jnp.zeros_like(video_tokens_BTN)
            mask_out = mask_out.at[:, -1].set(True)
            return final_logits_BTNV, mask_out
        else:
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
