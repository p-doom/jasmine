from typing import Dict

import einops
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint as ocp

from models.dynamics import DynamicsMaskGIT
from models.lam import LatentActionModel
from models.tokenizer import TokenizerVQVAE


class Genie(nnx.Module):
    """Genie model"""

    def __init__(
        self,
        in_dim: int,
        tokenizer_dim: int,
        tokenizer_ffn_dim: int,
        latent_patch_dim: int,
        num_patch_latents: int,
        patch_size: int,
        tokenizer_num_blocks: int,
        tokenizer_num_heads: int,
        lam_dim: int,
        lam_ffn_dim: int,
        latent_action_dim: int,
        num_latent_actions: int,
        lam_patch_size: int,
        lam_num_blocks: int,
        lam_num_heads: int,
        lam_co_train: bool,
        dyna_dim: int,
        dyna_ffn_dim: int,
        dyna_num_blocks: int,
        dyna_num_heads: int,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
        dropout: float = 0.0,
        mask_limit: float = 0.0,
    ):
        # --- Tokenizer ---
        self.in_dim = in_dim
        self.tokenizer_dim = tokenizer_dim
        self.tokenizer_ffn_dim = tokenizer_ffn_dim
        self.latent_patch_dim = latent_patch_dim
        self.num_patch_latents = num_patch_latents
        self.patch_size = patch_size
        self.tokenizer_num_blocks = tokenizer_num_blocks
        self.tokenizer_num_heads = tokenizer_num_heads
        # --- LAM ---
        self.lam_dim = lam_dim
        self.lam_ffn_dim = lam_ffn_dim
        self.latent_action_dim = latent_action_dim
        self.num_latent_actions = num_latent_actions
        self.lam_patch_size = lam_patch_size
        self.lam_num_blocks = lam_num_blocks
        self.lam_num_heads = lam_num_heads
        self.lam_co_train = lam_co_train
        # --- Dynamics ---
        self.dyna_dim = dyna_dim
        self.dyna_ffn_dim = dyna_ffn_dim
        self.dyna_num_blocks = dyna_num_blocks
        self.dyna_num_heads = dyna_num_heads
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.dropout = dropout
        self.mask_limit = mask_limit

        self.tokenizer = TokenizerVQVAE(
            in_dim=self.in_dim,
            model_dim=self.tokenizer_dim,
            ffn_dim=self.tokenizer_ffn_dim,
            latent_dim=self.latent_patch_dim,
            num_latents=self.num_patch_latents,
            patch_size=self.patch_size,
            num_blocks=self.tokenizer_num_blocks,
            num_heads=self.tokenizer_num_heads,
            dropout=0.0,
            codebook_dropout=0.0,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            rngs=rngs,
        )
        self.lam = LatentActionModel(
            in_dim=self.in_dim,
            model_dim=self.lam_dim,
            ffn_dim=self.lam_ffn_dim,
            latent_dim=self.latent_patch_dim,
            num_latents=self.num_latent_actions,
            patch_size=self.lam_patch_size,
            num_blocks=self.lam_num_blocks,
            num_heads=self.lam_num_heads,
            dropout=0.0,
            codebook_dropout=0.0,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            rngs=rngs,
        )
        self.dynamics = DynamicsMaskGIT(
            model_dim=self.dyna_dim,
            ffn_dim=self.dyna_ffn_dim,
            num_latents=self.num_patch_latents,
            latent_action_dim=self.latent_action_dim,
            num_blocks=self.dyna_num_blocks,
            num_heads=self.dyna_num_heads,
            dropout=self.dropout,
            mask_limit=self.mask_limit,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            rngs=rngs,
        )

    def __call__(
        self, batch: Dict[str, jax.Array], training: bool = True
    ) -> Dict[str, jax.Array]:
        videos_BTHWC = batch["videos"]
        tokenizer_outputs = self.tokenizer.vq_encode(videos_BTHWC, training=False)
        token_indices_BTN = tokenizer_outputs["indices"]
        lam_outputs = self.lam.vq_encode(videos_BTHWC, training=False)
        z_q_BTm11L = lam_outputs["z_q"]
        action_indices_E = lam_outputs["indices"]
        latent_actions_BTm11L = jax.lax.cond(
            self.lam_co_train,
            lambda: z_q_BTm11L,
            lambda: jax.lax.stop_gradient(z_q_BTm11L),
        )
        outputs = dict(
            video_tokens=jax.lax.stop_gradient(token_indices_BTN),
            latent_actions=latent_actions_BTm11L,
        )
        outputs["mask_rng"] = batch["mask_rng"]
        dyna_logits_BTNV, dyna_mask = self.dynamics(outputs, training)
        outputs["token_logits"] = dyna_logits_BTNV
        if dyna_mask is not None:
            outputs["mask"] = dyna_mask
        mle_indices_BTN = jnp.argmax(outputs["token_logits"], axis=-1)
        H, W = batch["videos"].shape[2:4]
        outputs["recon"] = self.tokenizer.decode(mle_indices_BTN, (H, W))
        outputs["lam_indices"] = action_indices_E
        return outputs

    def sample(
        self,
        batch: Dict[str, jax.Array],
        seq_len: int,
        steps: int = 25,
        temperature: float = 1,
        sample_argmax: bool = False,
    ) -> jax.Array:
        """
        Autoregressively samples up to `seq_len` future frames, following Figure 8 of the paper.

        - Input frames are tokenized once.
        - Future frames are generated autoregressively in token space.
        - All frames are detokenized in a single pass.

        Note:
        - For interactive or step-wise sampling, detokenization should occur after each action.
        - To maintain consistent tensor shapes across timesteps, all current and future frames are decoded at every step.
        - Temporal causal structure is preserved by
            a) reapplying the mask before each decoding step.
            b) a temporal causal mask is applied within each ST-transformer block.

        Dimension keys:
            B: batch size
            T: number of input (conditioning) frames
            N: number of patches per frame
            M: model dimension
            S: sequence length
            H: height
            W: width
            E: B * (S - 1)
            F: S * N
        """
        # --- Encode videos and actions ---
        videos_BTHWC = batch["videos"]
        latent_actions_E = batch["latent_actions"]
        tokenizer_out = self.tokenizer.vq_encode(videos_BTHWC, training=False)
        token_idxs_BTN = tokenizer_out["indices"]
        B, T, N = token_idxs_BTN.shape
        pad_shape = (B, seq_len - T, N)
        pad = jnp.zeros(pad_shape, dtype=token_idxs_BTN.dtype)
        token_idxs_BSN = jnp.concatenate([token_idxs_BTN, pad], axis=1)
        action_tokens_EL = self.lam.vq.get_codes(latent_actions_E)

        def maskgit_step_fn(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], step: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
            rng, token_idxs_BSN, mask_BSN, action_tokens_EL = carry
            S, N = token_idxs_BSN.shape[1:]
            L = action_tokens_EL.shape[-1]

            # --- Construct + encode video ---
            vid_embed_BSNM = self.dynamics.patch_embed(token_idxs_BSN)
            mask_token_111M = self.dynamics.mask_token.value
            mask_expanded_BSN1 = mask_BSN[..., None]
            vid_embed_BSNM = jnp.where(mask_expanded_BSN1, mask_token_111M, vid_embed_BSNM)

            # --- Predict transition ---
            action_tokens_BSm1L = jnp.reshape(action_tokens_EL, (B, S - 1, L))
            act_embed_BSm1M = self.dynamics.action_up(action_tokens_BSm1L)
            act_embed_BSM = jnp.pad(act_embed_BSm1M, ((0, 0), (1, 0), (0, 0)))
            act_embed_BS1M = jnp.reshape(act_embed_BSM, (B, S, 1, act_embed_BSM.shape[-1]))
            vid_embed_BSNM += act_embed_BS1M
            unmasked_ratio = jnp.cos(jnp.pi * (step + 1) / (steps * 2))
            step_temp = temperature * (1.0 - unmasked_ratio)
            final_logits_BSNV = self.dynamics.transformer(vid_embed_BSNM) / step_temp

            # --- Sample new tokens for final frame ---
            if sample_argmax:
                sampled_token_idxs_BSN = jnp.argmax(final_logits_BSNV, axis=-1)
            else:
                rng, _rng = jax.random.split(rng)
                sampled_token_idxs_BSN = jax.random.categorical(_rng, final_logits_BSNV)
            gather_fn = jax.vmap(jax.vmap(jax.vmap(lambda x, y: x[y])))
            final_token_probs_BSN = gather_fn(
                jax.nn.softmax(final_logits_BSNV), sampled_token_idxs_BSN
            )
            final_token_probs_BSN += ~mask_BSN
            # Update masked tokens only
            token_idxs_BSN = jnp.where(mask_BSN, sampled_token_idxs_BSN, token_idxs_BSN)

            # --- Update mask ---
            num_unmasked_tokens = jnp.round(N * (1.0 - unmasked_ratio)).astype(int)
            idx_mask_N = jnp.arange(final_token_probs_BSN.shape[-1]) <= N - num_unmasked_tokens
            final_token_probs_flat_BF = einops.rearrange(final_token_probs_BSN, "b s n -> b (s n)")
            sorted_idxs_BF = jnp.argsort(final_token_probs_flat_BF, axis=-1)
            mask_update_fn = jax.vmap(lambda msk, ids: msk.at[ids].set(idx_mask_N))
            mask_flat_BF = einops.rearrange(mask_BSN, "b s n -> b (s n)")
            new_mask_flat_BF = mask_update_fn(mask_flat_BF, sorted_idxs_BF)
            new_mask_BSN = einops.rearrange(new_mask_flat_BF, "b (s n) -> b s n", n=N)

            new_carry = (rng, token_idxs_BSN, new_mask_BSN, action_tokens_EL)
            return new_carry, None

        def generation_step_fn(
            carry: tuple[jax.Array, jax.Array], step_t: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            rng, current_token_idxs_BSN = carry
            rng, step_rng = jax.random.split(rng)

            # Mask current frame (i.e., t == step_t)
            mask_S = jnp.arange(seq_len) == step_t
            mask_BSN = jnp.broadcast_to(mask_S[None, :, None], (B, seq_len, N)).astype(
                bool
            )
            masked_token_idxs_BSN = current_token_idxs_BSN * ~mask_BSN

            # --- Initialize and run MaskGIT loop ---
            init_carry_maskgit = (
                step_rng,
                masked_token_idxs_BSN,
                mask_BSN,
                action_tokens_EL,
            )
            final_carry_maskgit, _ = jax.lax.scan(
                maskgit_step_fn, init_carry_maskgit, jnp.arange(steps)
            )
            updated_token_idxs = final_carry_maskgit[1]
            new_carry = (rng, updated_token_idxs)
            return new_carry, None

        # --- Run the autoregressive generation using jax.lax.scan ---
        initial_carry = (batch["rng"], token_idxs_BSN)
        timesteps_to_scan = jnp.arange(T, seq_len)
        final_carry, _ = jax.lax.scan(
            generation_step_fn, initial_carry, timesteps_to_scan
        )
        final_token_idxs = final_carry[1]

        # --- Decode all tokens at once at the end ---
        H, W = batch["videos"].shape[2:4]
        final_frames = self.tokenizer.decode(
            final_token_idxs,
            video_hw=(H, W),
        )
        return final_frames

    def vq_encode(self, batch: Dict[str, jax.Array], training: bool) -> jax.Array:
        # --- Preprocess videos ---
        video_BTHWC = batch["videos"]
        lam_output = self.lam.vq_encode(video_BTHWC, training=training)
        lam_indices_E = lam_output["indices"]
        return lam_indices_E


# FIXME (f.srambical): add conversion script for old checkpoints
def restore_genie_components(
    optimizer: nnx.Optimizer,
    sharding: jax.sharding.NamedSharding,
    rng: jax.Array,
    args,
) -> nnx.Optimizer:
    """Restore pre-trained Genie components"""
    rngs = nnx.Rngs(rng)

    tx = optimizer.tx
    model = optimizer.model
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add(
        "model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
    )

    checkpoint_options = ocp.CheckpointManagerOptions(
        step_format_fixed_length=6,
    )
    tokenizer_checkpoint_manager = ocp.CheckpointManager(
        directory=args.tokenizer_checkpoint,
        options=checkpoint_options,
        handler_registry=handler_registry,
    )
    dummy_tokenizer = TokenizerVQVAE(
        in_dim=args.image_channels,
        model_dim=args.tokenizer_dim,
        ffn_dim=args.tokenizer_ffn_dim,
        latent_dim=args.latent_patch_dim,
        num_latents=args.num_patch_latents,
        patch_size=args.patch_size,
        num_blocks=args.tokenizer_num_blocks,
        num_heads=args.tokenizer_num_heads,
        dropout=args.dropout,
        codebook_dropout=args.dropout,
        param_dtype=args.param_dtype,
        dtype=args.dtype,
        use_flash_attention=args.use_flash_attention,
        rngs=rngs,
    )
    dummy_tokenizer_optimizer = nnx.Optimizer(dummy_tokenizer, tx)
    dummy_tokenizer_optimizer_state = nnx.state(dummy_tokenizer_optimizer)
    abstract_sharded_tokenizer_optimizer_state = _create_abstract_sharded_pytree(
        dummy_tokenizer_optimizer_state, sharding
    )
    restored_tokenizer = tokenizer_checkpoint_manager.restore(
        step=tokenizer_checkpoint_manager.latest_step(),
        args=ocp.args.Composite(
            model_state=ocp.args.PyTreeRestore(  # type: ignore
                abstract_sharded_tokenizer_optimizer_state  # type: ignore
            ),
        ),
    )["model_state"]
    nnx.update(dummy_tokenizer_optimizer.model, restored_tokenizer.model)
    model.tokenizer = dummy_tokenizer_optimizer.model
    tokenizer_checkpoint_manager.close()

    if args.lam_checkpoint:
        lam_checkpoint_manager = ocp.CheckpointManager(
            directory=args.lam_checkpoint,
            options=checkpoint_options,
            handler_registry=handler_registry,
        )
        dummy_lam = LatentActionModel(
            in_dim=args.image_channels,
            model_dim=args.lam_dim,
            ffn_dim=args.lam_ffn_dim,
            latent_dim=args.latent_patch_dim,
            num_latents=args.num_latent_actions,
            patch_size=args.lam_patch_size,
            num_blocks=args.lam_num_blocks,
            num_heads=args.lam_num_heads,
            dropout=args.dropout,
            codebook_dropout=args.dropout,
            param_dtype=args.param_dtype,
            dtype=args.dtype,
            use_flash_attention=args.use_flash_attention,
            rngs=rngs,
        )
        dummy_lam_optimizer = nnx.Optimizer(dummy_lam, tx)
        dummy_lam_optimizer_state = nnx.state(dummy_lam_optimizer)
        abstract_sharded_lam_optimizer_state = _create_abstract_sharded_pytree(
            dummy_lam_optimizer_state, sharding
        )
        restored_lam_optimizer = lam_checkpoint_manager.restore(
            step=lam_checkpoint_manager.latest_step(),
            args=ocp.args.Composite(
                model_state=ocp.args.PyTreeRestore(  # type: ignore
                    abstract_sharded_lam_optimizer_state  # type: ignore
                ),
            ),
        )["model_state"]
        nnx.update(dummy_lam_optimizer.model, restored_lam_optimizer.model)
        model.lam = dummy_lam_optimizer.model
        # Remove the LAM decoder to save memory and avoid unnecessary computation.
        del model.lam.decoder
        lam_checkpoint_manager.close()
    
    # Reinitialize the optimizer states
    optimizer = nnx.Optimizer(model, tx)
    return optimizer


def _create_abstract_sharded_pytree(
    pytree_template: nnx.GraphState, sharding_spec: jax.sharding.NamedSharding
) -> jax.Array:
    """Replaces arrays in a pytree with ShapeDtypeStructs having the given sharding."""

    def map_fn(leaf_template):
        if hasattr(leaf_template, "shape") and hasattr(leaf_template, "dtype"):
            return jax.ShapeDtypeStruct(
                leaf_template.shape, leaf_template.dtype, sharding=sharding_spec
            )
        return leaf_template

    return jax.tree_util.tree_map(map_fn, pytree_template)
