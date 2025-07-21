from typing import Dict, Any

import optax
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

from models.dynamics import DynamicsMaskGIT
from models.lam import LatentActionModel
from models.tokenizer import TokenizerVQVAE

import grain


class Genie(nn.Module):
    """Genie model"""

    # --- Tokenizer ---
    in_dim: int
    tokenizer_dim: int
    tokenizer_ffn_dim: int
    latent_patch_dim: int
    num_patch_latents: int
    patch_size: int
    tokenizer_num_blocks: int
    tokenizer_num_heads: int
    # --- LAM ---
    lam_dim: int
    lam_ffn_dim: int
    latent_action_dim: int
    num_actions: int
    lam_patch_size: int
    lam_num_blocks: int
    lam_num_heads: int
    lam_co_train: bool
    # --- Dynamics ---
    dyna_dim: int
    dyna_ffn_dim: int
    dyna_num_blocks: int
    dyna_num_heads: int
    param_dtype: jnp.dtype
    dtype: jnp.dtype
    use_flash_attention: bool
    use_gt_actions: bool = False
    dropout: float = 0.0
    mask_limit: float = 0.0

    def setup(self):
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
        )
        if not self.use_gt_actions:
            self.lam = LatentActionModel(
                in_dim=self.in_dim,
                model_dim=self.lam_dim,
                ffn_dim=self.lam_ffn_dim,
                latent_dim=self.latent_patch_dim,
                num_latents=self.num_actions,
                patch_size=self.lam_patch_size,
                num_blocks=self.lam_num_blocks,
                num_heads=self.lam_num_heads,
                dropout=0.0,
                codebook_dropout=0.0,
                param_dtype=self.param_dtype,
                dtype=self.dtype,
                use_flash_attention=self.use_flash_attention,
            )
        self.dynamics = DynamicsMaskGIT(
            model_dim=self.dyna_dim,
            ffn_dim=self.dyna_ffn_dim,
            num_latents=self.num_patch_latents,
            num_blocks=self.dyna_num_blocks,
            num_heads=self.dyna_num_heads,
            dropout=self.dropout,
            mask_limit=self.mask_limit,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            use_gt_actions=self.use_gt_actions,
            num_actions=self.num_actions,
        )

    def __call__(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        tokenizer_outputs = self.tokenizer.vq_encode(batch["videos"], training=False)
        if self.use_gt_actions:
            actions = batch["actions"]
        else:
            lam_outputs = self.lam.vq_encode(batch["videos"], training=False)
            actions = jax.lax.cond(
                self.lam_co_train,
                lambda: lam_outputs["z_q"],
                lambda: jax.lax.stop_gradient(lam_outputs["z_q"])
            )
        outputs = dict(
            video_tokens=jax.lax.stop_gradient(tokenizer_outputs["indices"]),
            actions=actions,
        )
        outputs["mask_rng"] = batch["mask_rng"]
        dyna_outputs = self.dynamics(outputs, training)
        outputs.update(dyna_outputs)
        mle_indices = jnp.argmax(outputs["token_logits"], axis=-1)
        outputs["recon"] = self.tokenizer.decode(
            mle_indices, batch["videos"].shape[2:4]
        )
        if not self.use_gt_actions:
            outputs["lam_indices"] = lam_outputs["indices"] # type: ignore[unbound]
        return outputs

    @nn.compact
    def sample(
        self,
        batch: Dict[str, Any],
        seq_len: int,
        steps: int = 25,
        temperature: float = 1,
        sample_argmax: bool = False,
    ) -> Any:
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
        # --- Encode videos and actions ---
        tokenizer_out = self.tokenizer.vq_encode(batch["videos"], training=False)
        token_idxs = tokenizer_out["indices"] # (B, T, N)
        B, T, N = token_idxs.shape
        pad_shape = (B, seq_len - T, N)
        pad = jnp.zeros(pad_shape, dtype=token_idxs.dtype)
        token_idxs = jnp.concatenate([token_idxs, pad], axis=1) # (B, S, N)
        if self.use_gt_actions:
            action_tokens = batch["actions"]
        else:
            action_tokens = self.lam.vq.get_codes(batch["actions"])

        MaskGITLoop = nn.scan(
            MaskGITStep,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            length=steps,
        )
    
        loop_fn = MaskGITLoop(
            dynamics=self.dynamics,
            tokenizer=self.tokenizer,
            temperature=temperature,
            sample_argmax=sample_argmax,
            steps=steps,
        )

        def generation_step_fn(carry, step_t):
            rng, current_token_idxs = carry
            rng, step_rng = jax.random.split(rng)

            # Mask current and future frames (i.e., t >= step_t)
            mask = jnp.arange(seq_len) >= step_t # (S,)
            mask = jnp.broadcast_to(mask[None, :, None], (B, seq_len, N)) # (B, S, N)
            mask = mask.astype(bool)
            masked_token_idxs = current_token_idxs * ~mask

            # --- Initialize and run MaskGIT loop ---
            init_carry_maskgit = (
                step_rng,
                masked_token_idxs,
                mask,
                action_tokens,
            )
            final_carry_maskgit, _ = loop_fn(init_carry_maskgit, jnp.arange(steps))
            updated_token_idxs = final_carry_maskgit[1]
            new_carry = (rng, updated_token_idxs)
            return new_carry, None

        # --- Run the autoregressive generation using scan ---
        initial_carry = (batch["rng"], token_idxs)
        timesteps_to_scan = jnp.arange(T, seq_len)
        final_carry, _ = jax.lax.scan(
            generation_step_fn,
            initial_carry,
            timesteps_to_scan
        )
        final_token_idxs = final_carry[1]

        # --- Decode all tokens at once at the end ---
        final_frames = self.tokenizer.decode(
            final_token_idxs,
            video_hw=batch["videos"].shape[2:4],
        )
        return final_frames

    def vq_encode(self, batch, training) -> Dict[str, Any]:
        # --- Preprocess videos ---
        lam_output = self.lam.vq_encode(batch["videos"], training=training)
        return lam_output["indices"]


class MaskGITStep(nn.Module):
    dynamics: nn.Module
    tokenizer: nn.Module
    temperature: float
    sample_argmax: bool
    steps: int

    @nn.compact
    def __call__(self, carry, x):
        rng, token_idxs, mask, action_tokens = carry
        step = x
        N = token_idxs.shape[2]

        # --- Construct + encode video ---
        vid_embed = self.dynamics.patch_embed(token_idxs) # (B, S, N, D)
        mask_token = self.dynamics.mask_token  # (1, 1, 1, D,)
        mask_expanded = mask[..., None] # (B, S, N, 1) 
        vid_embed = jnp.where(mask_expanded, mask_token, vid_embed)

        # --- Predict transition ---
        act_embed = self.dynamics.action_up(action_tokens)
        vid_embed += jnp.pad(act_embed, ((0, 0), (1, 0), (0, 0), (0, 0)))
        unmasked_ratio = jnp.cos(jnp.pi * (step + 1) / (self.steps * 2))
        step_temp = self.temperature * (1.0 - unmasked_ratio)
        final_logits = self.dynamics.dynamics(vid_embed) / step_temp

        # --- Sample new tokens for final frame ---
        if self.sample_argmax:
            sampled_token_idxs = jnp.argmax(final_logits, axis=-1)
        else:
            rng, _rng = jax.random.split(rng)
            sampled_token_idxs = jax.random.categorical(_rng, final_logits)
        gather_fn = jax.vmap(jax.vmap(jax.vmap(lambda x, y: x[y])))
        final_token_probs = gather_fn(jax.nn.softmax(final_logits), sampled_token_idxs)
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

def restore_genie_components(
    train_state: TrainState,
    sharding: jax.sharding.NamedSharding,
    grain_iterator: grain.DataLoaderIterator,
    inputs: Dict[str, jax.Array],
    rng: jax.Array,
    args,
):
    """Restore pre-trained Genie components"""
    rng, _rng = jax.random.split(rng)

    # dummy values since we only use tx to initialize the dummy train states
    dummy_tx = optax.adamw(
        learning_rate=optax.constant_schedule(args.max_lr),
        b1=0.9,
        b2=0.9,
        weight_decay=1e-4,
    )
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add('model_state', ocp.args.StandardRestore, ocp.handlers.StandardCheckpointHandler)
    

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
    )
    tokenizer_init_params = dummy_tokenizer.init(_rng, inputs)
    dummy_tokenizer_train_state = TrainState.create(
        apply_fn=dummy_tokenizer.apply, params=tokenizer_init_params, tx=dummy_tx
    )
    abstract_sharded_tokenizer_state = _create_abstract_sharded_pytree(
        dummy_tokenizer_train_state, sharding
    )
    restored_tokenizer = tokenizer_checkpoint_manager.restore(
        step=tokenizer_checkpoint_manager.latest_step(),
        args=ocp.args.Composite(
            model_state=ocp.args.StandardRestore(abstract_sharded_tokenizer_state),
        ),
    )["model_state"]
    restored_tokenizer_params = restored_tokenizer.params["params"]
    train_state.params["params"]["tokenizer"].update(restored_tokenizer_params)
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
            num_latents=args.num_actions,
            patch_size=args.lam_patch_size,
            num_blocks=args.lam_num_blocks,
            num_heads=args.lam_num_heads,
            dropout=args.dropout,
            codebook_dropout=args.dropout,
            param_dtype=args.param_dtype,
            dtype=args.dtype,
            use_flash_attention=args.use_flash_attention,
        )
        lam_init_params = dummy_lam.init(_rng, inputs)
        dummy_lam_train_state = TrainState.create(
            apply_fn=dummy_lam.apply, params=lam_init_params, tx=dummy_tx
        )
        abstract_sharded_lam_state = _create_abstract_sharded_pytree(
            dummy_lam_train_state, sharding
        )
        restored_lam = lam_checkpoint_manager.restore(
            step=lam_checkpoint_manager.latest_step(),
            args=ocp.args.Composite(
                model_state=ocp.args.StandardRestore(abstract_sharded_lam_state),
            ),
        )["model_state"]
        restored_lam_params = restored_lam.params["params"]
        # Genie does not initialize all LAM modules, thus we omit those extra modules during restoration
        # (f.srambical) FIXME: Currently, this is a small HBM memory crunch since the LAM's decoder is loaded into HBM and immediately dicarded.
        # A workaround would be to restore to host memory first, and only move the weights to HBM after pruning the decoder
        restored_lam_params = {
            k: v
            for k, v in restored_lam_params.items()
            if k in train_state.params["params"]["lam"]
        }
        train_state.params["params"]["lam"].update(restored_lam_params)
        lam_checkpoint_manager.close()

    return train_state

def _create_abstract_sharded_pytree(pytree_template, sharding_spec):
    """Replaces arrays in a pytree with ShapeDtypeStructs having the given sharding."""

    def map_fn(leaf_template):
        if hasattr(leaf_template, "shape") and hasattr(leaf_template, "dtype"):
            return jax.ShapeDtypeStruct(
                leaf_template.shape, leaf_template.dtype, sharding=sharding_spec
            )
        return leaf_template

    return jax.tree_util.tree_map(map_fn, pytree_template)