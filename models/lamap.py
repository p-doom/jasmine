from typing import Dict, Any

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from models.lam import LatentActionModel


class LatentActionMapper(nnx.Module):
    """Latent Action Mapper"""


    # --- Mapper ---

    def __init__(
        self, 
        in_dim: int,
        model_dim: int,
        ffn_dim: int,
        latent_dim: int,
        num_latents: int,
        patch_size: int,
        num_blocks: int,
        num_heads: int,
        action_dim: int,
        use_flash_attention: bool,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs
    ):
        self.in_dim = in_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

        self.lam = LatentActionModel(
            in_dim=self.in_dim,
            model_dim=self.model_dim,
            ffn_dim=self.ffn_dim,
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            patch_size=self.patch_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=0.0,
            codebook_dropout=0.0,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            rngs=rngs,
        )
        self.action_map = nnx.Linear(
            self.latent_dim,
            self.action_dim,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        lam_outputs = self.lam.vq_encode(batch["videos"], training=False)
        latent_actions = jax.lax.stop_gradient(lam_outputs["z_q"])
        action_predictions = self.action_map(latent_actions)
        outputs = dict(
            action_predictions=action_predictions,
        )
        return outputs