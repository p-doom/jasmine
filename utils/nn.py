import math
from typing import Tuple, Callable

from flax import nnx
import jax
import jax.numpy as jnp
import einops


class SpatioTemporalPositionalEncoding(nnx.Module):
    """
    Applies separate sinusoidal positional encodings to the temporal and spatial dimensions.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len

        pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
        )
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = nnx.Variable(pe)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: The input tensor of shape (Batch, Time, Space, Dimension).

        Returns:
            The input tensor with positional encodings added.
        """
        assert x.ndim == 4, f"Input must be 4-dimensional, but got shape {x.shape}"

        num_timesteps = x.shape[1]
        num_spatial_patches = x.shape[2]

        # Temporal positional encoding: (1, T, 1, D)
        temporal_pe = self.pe.value[None, :num_timesteps, None, :]
        x = x + temporal_pe

        # Spatial positional encoding: (1, 1, S, D)
        spatial_pe = self.pe.value[None, None, :num_spatial_patches, :]
        x = x + spatial_pe

        return x


class STBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

        self.spatial_norm = nnx.LayerNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.spatial_attention = nnx.MultiHeadAttention(
            num_heads=self.num_heads,
            in_features=self.dim,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(
                self.use_flash_attention, is_causal=False
            ),
            rngs=rngs,
            decode=False,
        )

        self.temporal_norm = nnx.LayerNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.temporal_attention = nnx.MultiHeadAttention(
            num_heads=self.num_heads,
            in_features=self.dim,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(
                self.use_flash_attention, is_causal=True
            ),
            rngs=rngs,
            decode=False,
        )

        self.ffn_norm = nnx.LayerNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.ffn_dense1 = nnx.Linear(
            in_features=self.dim,
            out_features=self.ffn_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.ffn_dense2 = nnx.Linear(
            in_features=self.ffn_dim,
            out_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

    @nnx.remat
    def __call__(self, x: jax.Array) -> jax.Array:
        # --- Spatial attention ---
        z = self.spatial_norm(x)
        z = self.spatial_attention(z)
        x = x + z

        # --- Temporal attention ---
        x = x.swapaxes(1, 2)
        z = self.temporal_norm(x)
        z = self.temporal_attention(z)
        x = x + z
        x = x.swapaxes(1, 2)

        # --- Feedforward ---
        z = self.ffn_norm(x)
        z = self.ffn_dense1(z)
        z = jax.nn.gelu(z)
        z = self.ffn_dense2(z)
        x = x + z

        return x


class STTransformer(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        ffn_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
        max_len: int = 5000,
    ):
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.out_dim = out_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

        self.input_norm1 = nnx.LayerNorm(
            num_features=self.input_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.input_dense = nnx.Linear(
            in_features=self.input_dim,
            out_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.input_norm2 = nnx.LayerNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

        self.pos_enc = SpatioTemporalPositionalEncoding(self.model_dim, max_len=max_len)

        self.blocks = []
        for _ in range(self.num_blocks):
            self.blocks.append(
                STBlock(
                    dim=self.model_dim,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    param_dtype=self.param_dtype,
                    dtype=self.dtype,
                    use_flash_attention=self.use_flash_attention,
                    rngs=rngs,
                )
            )

        self.output_dense = nnx.Linear(
            in_features=self.model_dim,
            out_features=self.out_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.input_norm1(x)
        x = self.input_dense(x)
        x = self.input_norm2(x)
        x = self.pos_enc(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_dense(x)
        return x  # (B, T, E)


def normalize(x: jax.Array) -> jax.Array:
    return x / (jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-8)


class VectorQuantizer(nnx.Module):
    """
    Dimension keys:
        D: B * T * N
        K: number of latents
        L: latent dimension
    """
    def __init__(
        self, latent_dim: int, num_latents: int, dropout: float, rngs: nnx.Rngs
    ):
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.dropout = dropout

        self.codebook = nnx.Param(
            normalize(
                nnx.initializers.lecun_uniform()(
                    rngs.params(), (self.num_latents, self.latent_dim)
                )
            )
        )
        self.drop = nnx.Dropout(self.dropout, rngs=rngs)

    def __call__(
        self, x_DL: jax.Array, training: bool
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # --- Compute distances ---
        x_DL = normalize(x_DL)
        normalized_codebook_KL = normalize(self.codebook.value)
        distance_DK = -jnp.matmul(x_DL, normalized_codebook_KL.T)
        if training:
            distance_DK = self.drop(distance_DK)

        # --- Get indices and embeddings ---
        indices_D = jnp.argmin(distance_DK, axis=-1)
        z_DL = self.codebook[indices_D]

        # --- Straight through estimator ---
        z_q_DL = x_DL + jax.lax.stop_gradient(z_DL - x_DL)
        return z_q_DL, z_DL, x_DL, indices_D

    def get_codes(self, indices_E: jax.Array) -> jax.Array:
        return self.codebook[indices_E]


def _create_flash_attention_fn(use_flash_attention: bool, is_causal: bool) -> Callable:
    """
    Create an attention function that uses flash attention if enabled.

    Flax MultiHeadAttention provides tensors with shape (batch..., length, num_heads, head_dim)
    jax.nn.dot_product_attention expects (batch, length, num_heads, head_dim).

    We need to reshape to ensure compatibility. cuDNN's flash attention additionally
    requires a sequence length that is a multiple of 4. We pad the sequence length to the nearest
    multiple of 4 and mask accordingly.
    """

    def attention_fn(query, key, value, bias=None, mask=None, **kwargs):
        implementation = "cudnn" if use_flash_attention else None

        def _rearrange(x):
            return einops.rearrange(x, "... l h d -> (...) l h d")

        def _pad(x):
            return jnp.pad(x, ((0, 0), (0, pad_size), (0, 0), (0, 0)))

        def _fuse_masks(mask: jax.Array, attention_mask: jax.Array) -> jax.Array:
            mask_bool = mask.astype(jnp.bool_)
            expanded_mask = jnp.pad(
                mask_bool, ((0, pad_size), (0, pad_size)), constant_values=False
            )
            return jnp.logical_and(attention_mask, expanded_mask)

        original_shape = query.shape
        original_seq_len = query.shape[-3]

        # Pad to nearest multiple of 4
        target_seq_len = ((original_seq_len + 3) // 4) * 4
        pad_size = target_seq_len - original_seq_len

        query_4d = _pad(_rearrange(query))
        key_4d = _pad(_rearrange(key))
        value_4d = _pad(_rearrange(value))

        attention_mask = jnp.ones((target_seq_len, target_seq_len), dtype=jnp.bool_)
        attention_mask = attention_mask.at[original_seq_len:, :].set(False)
        attention_mask = attention_mask.at[:, original_seq_len:].set(False)

        mask_4d = (
            _fuse_masks(mask, attention_mask) if mask is not None else attention_mask
        )
        mask_4d = mask_4d[jnp.newaxis, jnp.newaxis, :, :]  # (1, 1, seq_len, seq_len)

        bias_4d = _pad(_rearrange(bias)) if bias is not None else None

        # NOTE: jax.nn.dot_product_attention does not support dropout
        output_4d = jax.nn.dot_product_attention(
            query=query_4d,
            key=key_4d,
            value=value_4d,
            bias=bias_4d,
            mask=mask_4d,
            implementation=implementation,
            is_causal=is_causal,
        )
        return output_4d[..., :original_seq_len, :, :].reshape(original_shape)

    return attention_fn
