from typing import Dict, Any

import optax
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import NamedSharding
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from orbax.checkpoint import PyTreeCheckpointer

from models.dynamics import DynamicsMaskGIT
from models.lam import LatentActionModel
from models.tokenizer import TokenizerVQVAE


class Genie(nn.Module):
    """Genie model"""

    # --- Tokenizer ---
    in_dim: int
    tokenizer_dim: int
    latent_patch_dim: int
    num_patch_latents: int
    patch_size: int
    tokenizer_num_blocks: int
    tokenizer_num_heads: int
    # --- LAM ---
    lam_dim: int
    latent_action_dim: int
    num_latent_actions: int
    lam_patch_size: int
    lam_num_blocks: int
    lam_num_heads: int
    lam_co_train: bool
    # --- Dynamics ---
    dyna_dim: int
    dyna_num_blocks: int
    dyna_num_heads: int
    dropout: float = 0.0
    mask_limit: float = 0.0

    def setup(self):
        self.tokenizer = TokenizerVQVAE(
            in_dim=self.in_dim,
            model_dim=self.tokenizer_dim,
            latent_dim=self.latent_patch_dim,
            num_latents=self.num_patch_latents,
            patch_size=self.patch_size,
            num_blocks=self.tokenizer_num_blocks,
            num_heads=self.tokenizer_num_heads,
            dropout=0.0,
            codebook_dropout=0.0,
        )
        self.lam = LatentActionModel(
            in_dim=self.in_dim,
            model_dim=self.lam_dim,
            latent_dim=self.latent_patch_dim,
            num_latents=self.num_latent_actions,
            patch_size=self.lam_patch_size,
            num_blocks=self.lam_num_blocks,
            num_heads=self.lam_num_heads,
            dropout=0.0,
            codebook_dropout=0.0,
        )
        self.dynamics = DynamicsMaskGIT(
            model_dim=self.dyna_dim,
            num_latents=self.num_patch_latents,
            num_blocks=self.dyna_num_blocks,
            num_heads=self.dyna_num_heads,
            dropout=self.dropout,
            mask_limit=self.mask_limit,
        )

    def __call__(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        tokenizer_outputs = self.tokenizer.vq_encode(batch["videos"], training=False)
        lam_outputs = self.lam.vq_encode(batch["videos"], training=False)
        outputs = dict(
            video_tokens=jax.lax.stop_gradient(tokenizer_outputs["indices"]),
            latent_actions=lam_outputs["z_q"] if self.lam_co_train else jax.lax.stop_gradient(lam_outputs["z_q"]),
        )
        outputs["mask_rng"] = batch["mask_rng"]
        dyna_outputs = self.dynamics(outputs, training)
        outputs.update(dyna_outputs)
        mle_indices = jnp.argmax(outputs["token_logits"], axis=-1)
        outputs["recon"] = self.tokenizer.decode(
            mle_indices, batch["videos"].shape[2:4]
        )
        return outputs

    @nn.compact
    def sample(
        self,
        batch: Dict[str, Any],
        steps: int = 25,
        temperature: int = 1,
        sample_argmax: bool = False,
    ) -> Any:
        # --- Encode videos and actions ---
        tokenizer_out = self.tokenizer.vq_encode(batch["videos"], training=False)
        token_idxs = tokenizer_out["indices"]
        new_frame_idxs = jnp.zeros_like(token_idxs)[:, 0]
        action_tokens = self.lam.vq.get_codes(batch["latent_actions"])

        # --- Initialize MaskGIT ---
        init_mask = jnp.ones_like(token_idxs, dtype=bool)[:, 0]
        init_carry = (
            batch["rng"],
            new_frame_idxs,
            init_mask,
            token_idxs,
            action_tokens,
        )
        MaskGITLoop = nn.scan(
            MaskGITStep,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            length=steps,
        )

        # --- Run MaskGIT loop ---
        loop_fn = MaskGITLoop(
            dynamics=self.dynamics,
            tokenizer=self.tokenizer,
            temperature=temperature,
            sample_argmax=sample_argmax,
            steps=steps,
        )
        final_carry, _ = loop_fn(init_carry, jnp.arange(steps))
        new_frame_idxs = final_carry[1]
        new_frame_pixels = self.tokenizer.decode(
            jnp.expand_dims(new_frame_idxs, 1),
            video_hw=batch["videos"].shape[2:4],
        )
        return new_frame_pixels

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
        rng, final_token_idxs, mask, token_idxs, action_tokens = carry
        step = x
        B, T, N = token_idxs.shape[:3]

        # --- Construct + encode video ---
        vid_token_idxs = jnp.concatenate(
            (token_idxs, jnp.expand_dims(final_token_idxs, 1)), axis=1
        )
        vid_embed = self.dynamics.patch_embed(vid_token_idxs)
        curr_masked_frame = jnp.where(
            jnp.expand_dims(mask, -1),
            self.dynamics.mask_token[0],
            vid_embed[:, -1],
        )
        vid_embed = vid_embed.at[:, -1].set(curr_masked_frame)

        # --- Predict transition ---
        act_embed = self.dynamics.action_up(action_tokens)
        vid_embed += jnp.pad(act_embed, ((0, 0), (1, 0), (0, 0), (0, 0)))
        unmasked_ratio = jnp.cos(jnp.pi * (step + 1) / (self.steps * 2))
        step_temp = self.temperature * (1.0 - unmasked_ratio)
        final_logits = self.dynamics.dynamics(vid_embed)[:, -1] / step_temp

        # --- Sample new tokens for final frame ---
        if self.sample_argmax:
            sampled_token_idxs = jnp.argmax(final_logits, axis=-1)
        else:
            rng, _rng = jax.random.split(rng)
            sampled_token_idxs = jnp.where(
                step == self.steps - 1,
                jnp.argmax(final_logits, axis=-1),
                jax.random.categorical(_rng, final_logits),
            )
        gather_fn = jax.vmap(jax.vmap(lambda x, y: x[y]))
        final_token_probs = gather_fn(jax.nn.softmax(final_logits), sampled_token_idxs)
        final_token_probs += ~mask
        # Update masked tokens only
        new_token_idxs = jnp.where(mask, sampled_token_idxs, final_token_idxs)

        # --- Update mask ---
        num_unmasked_tokens = jnp.round(N * (1.0 - unmasked_ratio)).astype(int)
        idx_mask = jnp.arange(final_token_probs.shape[-1]) > num_unmasked_tokens
        sorted_idxs = jnp.argsort(final_token_probs, axis=-1, descending=True)
        mask_update_fn = jax.vmap(lambda msk, ids: msk.at[ids].set(idx_mask))
        new_mask = mask_update_fn(mask, sorted_idxs)

        new_carry = (rng, new_token_idxs, new_mask, token_idxs, action_tokens)
        return new_carry, None


def restore_genie_components(
    train_state: TrainState,
    sharding: NamedSharding,
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

    dummy_tokenizer = TokenizerVQVAE(
        in_dim=args.image_channels,
        model_dim=args.tokenizer_dim,
        latent_dim=args.latent_patch_dim,
        num_latents=args.num_patch_latents,
        patch_size=args.patch_size,
        num_blocks=args.tokenizer_num_blocks,
        num_heads=args.tokenizer_num_heads,
        dropout=args.dropout,
        codebook_dropout=args.dropout,
    )
    tokenizer_init_params = dummy_tokenizer.init(_rng, inputs)
    dummy_tokenizer_train_state = TrainState.create(
        apply_fn=dummy_tokenizer.apply, params=tokenizer_init_params, tx=dummy_tx
    )
    abstract_sharded_tokenizer_state = _create_abstract_sharded_pytree(
        dummy_tokenizer_train_state, sharding
    )
    tokenizer_restore_target = {"model": abstract_sharded_tokenizer_state}
    tokenizer_restore_args = orbax_utils.restore_args_from_target(
        tokenizer_restore_target
    )
    restored_tokenizer_params = (
        PyTreeCheckpointer()
        .restore(
            args.tokenizer_checkpoint,
            item=tokenizer_restore_target,
            restore_args=tokenizer_restore_args,
        )["model"]
        .params["params"]
    )
    train_state.params["params"]["tokenizer"].update(restored_tokenizer_params)

    if args.lam_checkpoint:
        dummy_lam = LatentActionModel(
            in_dim=args.image_channels,
            model_dim=args.lam_dim,
            latent_dim=args.latent_patch_dim,
            num_latents=args.num_latent_actions,
            patch_size=args.lam_patch_size,
            num_blocks=args.lam_num_blocks,
            num_heads=args.lam_num_heads,
            dropout=args.dropout,
            codebook_dropout=args.dropout,
        )
        lam_init_params = dummy_lam.init(_rng, inputs)
        dummy_lam_train_state = TrainState.create(
            apply_fn=dummy_lam.apply, params=lam_init_params, tx=dummy_tx
        )
        abstract_sharded_lam_state = _create_abstract_sharded_pytree(
            dummy_lam_train_state, sharding
        )
        lam_restore_target = {"model": abstract_sharded_lam_state}
        lam_restore_args = orbax_utils.restore_args_from_target(lam_restore_target)
        restored_lam_params = (
            PyTreeCheckpointer()
            .restore(
                args.lam_checkpoint, item=lam_restore_target, restore_args=lam_restore_args
            )["model"]
            .params["params"]
        )
        # Genie does not initialize all LAM modules, thus we omit those extra modules during restoration
        # (f.srambical) FIXME: Currently, this is a small HBM memory crunch since the LAM's decoder is loaded into HBM and immediately dicarded.
        # A workaround would be to restore to host memory first, and only move the weights to HBM after pruning the decoder
        restored_lam_params = {
            k: v
            for k, v in restored_lam_params.items()
            if k in train_state.params["params"]["lam"]
        }
        train_state.params["params"]["lam"].update(restored_lam_params)

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