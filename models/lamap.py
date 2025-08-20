from typing import Dict, Any

import jax
import jax.numpy as jnp
import flax.linen as nn

from models.lam import LatentActionModel


class LatentActionMapper(nn.Module):
    """Latent Action Mapper"""

    param_dtype: jnp.dtype
    dtype: jnp.dtype

    # --- LAM ---
    in_dim: int
    model_dim: int
    latent_dim: int
    num_latents: int
    patch_size: int
    num_blocks: int
    num_heads: int
    dropout: float
    codebook_dropout: float
    use_flash_attention: bool

    # --- Mapper ---
    action_dim: int

    def setup(self):
        self.lam = LatentActionModel(
            in_dim=self.in_dim,
            model_dim=self.lam_dim,
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            patch_size=self.lam_patch_size,
            num_blocks=self.lam_num_blocks,
            num_heads=self.lam_num_heads,
            dropout=self.lam_dropout,
            codebook_dropout=self.lam_codebook_dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
        )
        self.action_map = nn.Dense(
            self.action_dim,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            )

    def __call__(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        lam_outputs = self.lam.vq_encode(batch["videos"], training=False)
        latent_actions = jax.lax.stop_gradient(lam_outputs["z_q"])
        outputs = dict(
            action_predictions=self.action_map(latent_actions),
        )
        return outputs