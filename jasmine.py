from typing import Dict

import optax
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint as ocp

from models.dynamics_causal import DynamicsCausal
from models.dynamics_maskgit import DynamicsMaskGIT
from models.lam import LatentActionModel
from models.tokenizer import TokenizerVQVAE


class Jasmine(nnx.Module):
    """World model with three components: a tokenizer, a latent action model (LAM), and a dynamics model for predicting future tokens."""

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
        dynamics_type: str,
        dyna_dim: int,
        dyna_ffn_dim: int,
        dyna_num_blocks: int,
        dyna_num_heads: int,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        decode: bool,
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
        self.dynamics_type = dynamics_type
        self.dyna_dim = dyna_dim
        self.dyna_ffn_dim = dyna_ffn_dim
        self.dyna_num_blocks = dyna_num_blocks
        self.dyna_num_heads = dyna_num_heads
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.decode = decode
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
        if self.dynamics_type == "maskgit":
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
        elif self.dynamics_type == "causal":
            self.dynamics = DynamicsCausal(
                model_dim=self.dyna_dim,
                ffn_dim=self.dyna_ffn_dim,
                num_latents=self.num_patch_latents,
                latent_action_dim=self.latent_action_dim,
                num_blocks=self.dyna_num_blocks,
                num_heads=self.dyna_num_heads,
                dropout=self.dropout,
                param_dtype=self.param_dtype,
                dtype=self.dtype,
                use_flash_attention=self.use_flash_attention,
                decode=self.decode,
                rngs=rngs,
            )
        else:
            raise ValueError(f"Invalid dynamics type: {self.dynamics_type}")

    def __call__(
        self, batch: Dict[str, jax.Array], training: bool = True
    ) -> Dict[str, jax.Array]:
        tokenizer_outputs = self.tokenizer.vq_encode(batch["videos"], training=False)
        lam_outputs = self.lam.vq_encode(batch["videos"], training=False)
        latent_actions = jax.lax.cond(
            self.lam_co_train,
            lambda: lam_outputs["z_q"],
            lambda: jax.lax.stop_gradient(lam_outputs["z_q"]),
        )
        outputs = dict(
            video_tokens=jax.lax.stop_gradient(tokenizer_outputs["indices"]),
            latent_actions=latent_actions,
        )
        outputs["mask_rng"] = batch["mask_rng"]
        dyna_logits, dyna_mask = self.dynamics(outputs, training)
        outputs["token_logits"] = dyna_logits
        if dyna_mask is not None:
            outputs["mask"] = dyna_mask
        mle_indices = jnp.argmax(outputs["token_logits"], axis=-1)
        H, W = batch["videos"].shape[2:4]
        outputs["recon"] = self.tokenizer.decode(mle_indices, (H, W))
        outputs["lam_indices"] = lam_outputs["indices"]
        return outputs

    def sample_maskgit(
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
            N: patches per frame
            S: sequence length
            A: action space
            D: model latent dimension
        """
        assert self.dynamics_type == "maskgit"
        # --- Encode videos and actions ---
        tokenizer_out = self.tokenizer.vq_encode(batch["videos"], training=False)
        token_idxs = tokenizer_out["indices"]  # (B, T, N)
        B, T, N = token_idxs.shape
        pad_shape = (B, seq_len - T, N)
        pad = jnp.zeros(pad_shape, dtype=token_idxs.dtype)
        token_idxs = jnp.concatenate([token_idxs, pad], axis=1)  # (B, S, N)
        action_tokens = self.lam.vq.get_codes(batch["latent_actions"])

        def maskgit_step_fn(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], step: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
            rng, token_idxs, mask, action_tokens = carry
            N = token_idxs.shape[2]

            # --- Construct + encode video ---
            vid_embed = self.dynamics.patch_embed(token_idxs)  # (B, S, N, D)
            if not isinstance(self.dynamics, DynamicsMaskGIT):
                raise TypeError("`sample_maskgit` requires `DynamicsMaskGIT`.")
            mask_token = self.dynamics.mask_token.value  # (1, 1, 1, D,)
            mask_expanded = mask[..., None]  # (B, S, N, 1)
            vid_embed = jnp.where(mask_expanded, mask_token, vid_embed)

            # --- Predict transition ---
            act_embed = self.dynamics.action_up(action_tokens)
            vid_embed += jnp.pad(act_embed, ((0, 0), (1, 0), (0, 0), (0, 0)))
            unmasked_ratio = jnp.cos(jnp.pi * (step + 1) / (steps * 2))
            step_temp = temperature * (1.0 - unmasked_ratio)
            final_logits = self.dynamics.transformer(vid_embed) / step_temp

            # --- Sample new tokens for final frame ---
            if sample_argmax:
                sampled_token_idxs = jnp.argmax(final_logits, axis=-1)
            else:
                rng, _rng = jax.random.split(rng)
                sampled_token_idxs = jax.random.categorical(_rng, final_logits)
            gather_fn = jax.vmap(jax.vmap(jax.vmap(lambda x, y: x[y])))
            final_token_probs = gather_fn(
                jax.nn.softmax(final_logits), sampled_token_idxs
            )
            final_token_probs += ~mask
            # Update masked tokens only
            token_idxs = jnp.where(mask, sampled_token_idxs, token_idxs)

            # --- Update mask ---
            num_unmasked_tokens = jnp.round(N * (1.0 - unmasked_ratio)).astype(int)
            idx_mask = jnp.arange(final_token_probs.shape[-1]) > num_unmasked_tokens
            sorted_idxs = jnp.argsort(final_token_probs, axis=-1, descending=True)
            mask_update_fn = jax.vmap(lambda msk, ids: msk.at[ids].set(idx_mask))
            new_mask = mask_update_fn(mask, sorted_idxs)

            new_carry = (rng, token_idxs, new_mask, action_tokens)
            return new_carry, None

        def generation_step_fn(
            carry: tuple[jax.Array, jax.Array], step_t: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            rng, current_token_idxs = carry
            rng, step_rng = jax.random.split(rng)

            # Mask current and future frames (i.e., t >= step_t)
            mask = jnp.arange(seq_len) >= step_t  # (S,)
            mask = jnp.broadcast_to(mask[None, :, None], (B, seq_len, N)).astype(
                bool
            )  # (B, S, N)
            masked_token_idxs = current_token_idxs * ~mask

            # --- Initialize and run MaskGIT loop ---
            init_carry_maskgit = (
                step_rng,
                masked_token_idxs,
                mask,
                action_tokens,
            )
            final_carry_maskgit, _ = jax.lax.scan(
                maskgit_step_fn, init_carry_maskgit, jnp.arange(steps)
            )
            updated_token_idxs = final_carry_maskgit[1]
            new_carry = (rng, updated_token_idxs)
            return new_carry, None

        # --- Run the autoregressive generation using jax.lax.scan ---
        initial_carry = (batch["rng"], token_idxs)
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

    def sample_causal(
        self,
        batch: Dict[str, jax.Array],
        seq_len: int,
        temperature: float = 1,
        sample_argmax: bool = False,
    ) -> jax.Array:
        """
        Autoregressively samples up to `seq_len` future frames using the causal transformer backend.
        - Input frames are tokenized once.
        - Future frames are generated one at a time, each conditioned on all previous frames.
        - All frames are detokenized in a single pass at the end.
        Args:
            batch: Dict with at least "videos" (B, T, H, W, C)
            seq_len: total number of frames to generate (including context)
            temperature: sampling temperature
            sample_argmax: if True, use argmax instead of sampling
        Returns:
            Generated video frames (B, seq_len, H, W, C)
        """
        # --- Encode context frames ---
        tokenizer_out = self.tokenizer.vq_encode(batch["videos"], training=False)
        token_idxs = tokenizer_out["indices"]  # (B, T, N)
        B, T, N = token_idxs.shape

        # --- Prepare initial token sequence ---
        # Pad with zeros for future frames
        pad_shape = (B, seq_len - T, N)
        token_idxs_full = jnp.concatenate(
            [token_idxs, jnp.zeros(pad_shape, dtype=token_idxs.dtype)], axis=1
        )  # (B, seq_len, N)

        # --- Prepare latent actions ---
        action_tokens = self.lam.vq.get_codes(batch["latent_actions"])  # (B, S-1, )

        def token_step_fn(
            carry: tuple[jax.Array, jax.Array, jax.Array], token_idx: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
            rng, token_idxs_full, action_tokens = carry
            t = token_idx // N
            n = token_idx % N

            # For autoregressive decoding, we only need to pass the token from the previous step.
            # The model internally uses a KV cache to remember previous tokens.
            current_token_sequence = jax.lax.dynamic_slice(
                token_idxs_full, (0, t, 0), (B, 1, N)
            )

            dyna_inputs = {
                "video_tokens": current_token_sequence,
                "latent_actions": action_tokens,
            }
            # The model will output logits for all patches in the sequence (which is just one frame).
            next_token_logits, _ = self.dynamics(dyna_inputs, training=False)
            # We select the logits for the specific patch `n` we are currently generating.
            next_token_logits = next_token_logits[:, 0, n, :].astype(
                jnp.float32
            )  # (B, vocab_size)

            if sample_argmax:
                next_token = jnp.argmax(next_token_logits, axis=-1)  # (B,)
            else:
                rng, step_rng = jax.random.split(rng)
                next_token = jax.random.categorical(
                    step_rng, next_token_logits / temperature, axis=-1
                )  # (B,)

            # Insert the generated token into the full sequence.
            token_idxs_full = token_idxs_full.at[:, t, n].set(next_token)

            new_carry = (rng, token_idxs_full, action_tokens)
            return new_carry, None

        # --- Autoregressive generation ---
        future_frames = seq_len - T
        total_future_tokens = future_frames * N
        start_token_idx = T * N
        step_indices = jnp.arange(start_token_idx, start_token_idx + total_future_tokens)

        initial_carry = (batch["rng"], token_idxs_full, action_tokens)
        final_carry, _ = jax.lax.scan(
            token_step_fn, initial_carry, step_indices
        )
        final_token_idxs = final_carry[1]

        # --- Decode all tokens at once at the end ---
        H, W = batch["videos"].shape[2:4]
        final_frames = self.tokenizer.decode(final_token_idxs, video_hw=(H, W))
        return final_frames

    def vq_encode(self, batch: Dict[str, jax.Array], training: bool) -> jax.Array:
        # --- Preprocess videos ---
        lam_output = self.lam.vq_encode(batch["videos"], training=training)
        return lam_output["indices"]


# FIXME (f.srambical): add conversion script for old checkpoints
def restore_components(
    optimizer: nnx.Optimizer,
    sharding: jax.sharding.NamedSharding,
    rng: jax.Array,
    args,
) -> nnx.Optimizer:
    """Restore pre-trained Genie components"""
    rngs = nnx.Rngs(rng)

    # dummy values since we only use tx to initialize the dummy train states
    dummy_tx = optax.adamw(
        learning_rate=optax.constant_schedule(args.max_lr),
        b1=0.9,
        b2=0.9,
        weight_decay=1e-4,
    )
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
    dummy_tokenizer_optimizer = nnx.Optimizer(dummy_tokenizer, dummy_tx)
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
    optimizer.model.tokenizer = dummy_tokenizer_optimizer.model
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
        dummy_lam_optimizer = nnx.Optimizer(dummy_lam, dummy_tx)
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
        optimizer.model.lam = dummy_lam_optimizer.model
        # Remove the LAM decoder to save memory and avoid unnecessary computation.
        del optimizer.model.lam.decoder
        lam_checkpoint_manager.close()

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
