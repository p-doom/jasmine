import math
from typing import Dict, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp


class PositionalEncoding(nn.Module):
    """https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html"""

    d_model: int  # Hidden dimensionality of the input.
    max_len: int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        self.pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
        )
        self.pe = self.pe.at[:, 0::2].set(jnp.sin(position * div_term))
        self.pe = self.pe.at[:, 1::2].set(jnp.cos(position * div_term))

    def __call__(self, x):
        x = x + self.pe[: x.shape[2]]
        return x


class STBlock(nn.Module):
    dim: int
    num_heads: int
    dropout: float

    @nn.remat
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # --- Spatial attention ---
        z = PositionalEncoding(self.dim)(x)
        z = nn.LayerNorm()(z)
        z = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
        )(z)
        x = x + z

        # --- Temporal attention ---
        x = x.swapaxes(1, 2)
        z = PositionalEncoding(self.dim)(x)
        z = nn.LayerNorm()(z)
        causal_mask = jnp.tri(z.shape[-2])
        z = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
        )(z, mask=causal_mask)
        x = x + z
        x = x.swapaxes(1, 2)

        # --- Feedforward ---
        z = nn.LayerNorm()(x)
        z = nn.Dense(self.dim)(z)
        z = nn.gelu(z)
        x = x + z

        return x


class STTransformer(nn.Module):
    model_dim: int
    out_dim: int
    num_blocks: int
    num_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Sequential(
            [
                nn.LayerNorm(),
                nn.Dense(self.model_dim),
                nn.LayerNorm(),
            ]
        )(x)
        for _ in range(self.num_blocks):
            x = STBlock(
                dim=self.model_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x)
        x = nn.Dense(self.out_dim)(x)
        return x  # (B, T, E)


def normalize(x):
    return x / (jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-8)


class VectorQuantizer(nn.Module):
    latent_dim: int
    num_latents: int
    dropout: float

    def setup(self):
        self.codebook = normalize(
            self.param(
                "codebook",
                nn.initializers.lecun_uniform(),
                (self.num_latents, self.latent_dim),
            )
        )
        self.drop = nn.Dropout(self.dropout, deterministic=False)

    def __call__(
        self, x: jax.Array, training: bool
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # --- Compute distances ---
        x = normalize(x)
        codebook = normalize(self.codebook)
        distance = -jnp.matmul(x, codebook.T)
        if training:
            distance = self.drop(distance)

        # --- Get indices and embeddings ---
        indices = jnp.argmin(distance, axis=-1)
        z = self.codebook[indices]

        # --- Straight through estimator ---
        z_q = x + jax.lax.stop_gradient(z - x)
        return z_q, z, x, indices

    def get_codes(self, indices: jax.Array):
        return self.codebook[indices]
