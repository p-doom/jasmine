import math
from typing import Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
import einops


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
    param_dtype: jnp.dtype
    dtype: jnp.dtype
    use_flash_attention: bool

    @nn.remat
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # --- Spatial attention ---
        z = PositionalEncoding(self.dim)(x)
        z = nn.LayerNorm(
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )(z)
        z = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(self.use_flash_attention, is_causal=False),
        )(z)
        x = x + z

        # --- Temporal attention ---
        x = x.swapaxes(1, 2)
        z = PositionalEncoding(self.dim)(x)
        z = nn.LayerNorm(
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )(z)
        causal_mask = jnp.tri(z.shape[-2])
        z = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(self.use_flash_attention, is_causal=True),
        # FIXME (f.srambical): check whether we should still pass the mask if we set is_causal=True
        )(z, mask=causal_mask)
        x = x + z
        x = x.swapaxes(1, 2)

        # --- Feedforward ---
        z = nn.LayerNorm(
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )(x)
        # FIXME (f.srambical): Here, the attention hidden dimension is the same as the FFN's. Usually, FFN hidden dimension is 4x model_dim
        z = nn.Dense(
            self.dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )(z)
        z = nn.gelu(z)
        x = x + z

        return x


class STTransformer(nn.Module):
    model_dim: int
    out_dim: int
    num_blocks: int
    num_heads: int
    dropout: float
    param_dtype: jnp.dtype
    dtype: jnp.dtype
    use_flash_attention: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Sequential(
            [
                nn.LayerNorm(
                    param_dtype=self.param_dtype,
                    dtype=self.dtype,
                ),
                nn.Dense(self.model_dim,
                param_dtype=self.param_dtype,
                dtype=self.dtype,
                ),
                nn.LayerNorm(
                    param_dtype=self.param_dtype,
                    dtype=self.dtype,
                ),
            ]
        )(x)
        for _ in range(self.num_blocks):
            x = STBlock(
                dim=self.model_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                param_dtype=self.param_dtype,
                dtype=self.dtype,
                use_flash_attention=self.use_flash_attention,
            )(x)
        x = nn.Dense(
            self.out_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )(x)
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
            dropout_key = self.make_rng("dropout")
            distance = self.drop(distance, rng=dropout_key)

        # --- Get indices and embeddings ---
        indices = jnp.argmin(distance, axis=-1)
        z = self.codebook[indices]

        # --- Straight through estimator ---
        z_q = x + jax.lax.stop_gradient(z - x)
        return z_q, z, x, indices

    def get_codes(self, indices: jax.Array):
        return self.codebook[indices]


def _create_flash_attention_fn(use_flash_attention: bool, is_causal: bool):
    """
    Create an attention function that uses flash attention if enabled.

    Flax MultiHeadAttention provides tensors with shape (batch..., length, num_heads, head_dim)
    jax.nn.dot_product_attention expects (batch, length, num_heads, head_dim) after _ensure_4d
    We need to reshape to ensure compatibility
    """
        
    def attention_fn(query, key, value, bias=None, mask=None, **kwargs):
        implementation = 'cudnn' if use_flash_attention else None
        
        original_shape = query.shape
        
        query_4d = einops.rearrange(query, '... l h d -> (...) l h d')
        key_4d = einops.rearrange(key, '... l h d -> (...) l h d')
        value_4d = einops.rearrange(value, '... l h d -> (...) l h d')
        
        bias_4d = None
        if bias is not None:
            bias_4d = einops.rearrange(bias, '... l h d -> (...) l h d')
        
        mask_4d = None
        if mask is not None:
            mask_4d = mask[jnp.newaxis, jnp.newaxis, :, :]  # (1, 1, seq_len, seq_len)
            
            mask_4d = mask_4d.astype(jnp.bool_)
        
        output_4d = jax.nn.dot_product_attention(
            query=query_4d,
            key=key_4d,
            value=value_4d,
            bias=bias_4d,
            mask=mask_4d,
            implementation=implementation,
            is_causal=is_causal,
            **kwargs
        )
        return output_4d.reshape(original_shape)
    
    return attention_fn

