import math
from typing import Tuple, Callable

from flax import nnx
import jax
import jax.numpy as jnp
import einops


class PositionalEncoding(nnx.Module):
    """https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html"""

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
        x = x + self.pe[: x.shape[2]]
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
        spatial_causal: bool,
        decode: bool,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.spatial_causal = spatial_causal
        self.decode = decode

        self.spatial_pos_enc = PositionalEncoding(self.dim)
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
                self.use_flash_attention,
                is_causal=self.spatial_causal,
            ),
            rngs=rngs,
            # decode=self.decode,
            decode=False,
        )

        self.temporal_pos_enc = PositionalEncoding(self.dim)
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
            decode=self.decode,
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
        z = self.spatial_pos_enc(x)
        z = self.spatial_norm(z)
        z = self.spatial_attention(z)
        x = x + z

        # --- Temporal attention ---
        x = x.swapaxes(1, 2)
        z = self.temporal_pos_enc(x)
        z = self.temporal_norm(z)
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
        spatial_causal: bool,
        decode: bool,
        rngs: nnx.Rngs,
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
        self.spatial_causal = spatial_causal
        self.decode = decode

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

        self.blocks: list[STBlock] = []
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
                    spatial_causal=self.spatial_causal,
                    decode=self.decode,
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
        # x.shape (1, 1, 921, 512)
        x = self.input_norm1(x)
        x = self.input_dense(x)
        x = self.input_norm2(x)

        for block in self.blocks:
            # x.shape (1, 1, 921, 512)
            x = block(x)

        x = self.output_dense(x)
        return x  # (B, T, E)


def normalize(x: jax.Array) -> jax.Array:
    return x / (jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-8)


class VectorQuantizer(nnx.Module):
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
        self, x: jax.Array, training: bool
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # --- Compute distances ---
        x = normalize(x)
        normalized_codebook = normalize(self.codebook.value)
        distance = -jnp.matmul(x, normalized_codebook.T)
        if training:
            distance = self.drop(distance)

        # --- Get indices and embeddings ---
        indices = jnp.argmin(distance, axis=-1)
        z = self.codebook[indices]

        # --- Straight through estimator ---
        z_q = x + jax.lax.stop_gradient(z - x)
        return z_q, z, x, indices

    def get_codes(self, indices: jax.Array) -> jax.Array:
        return self.codebook[indices]


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

        # Handle causal mask for cached decoder self-attention (from nnx.MultiHeadAttention)
        if mask is not None:
            mask_4d = _rearrange(mask)
            # NOTE: We need to broadcast T and S dimensions to target_seq_len since cudnn attention strictly checks the mask shape
            # https://github.com/jax-ml/jax/issues/28974
            # https://github.com/jax-ml/jax/blob/08c7677393672ccb85c10f1ed0bd506905c3c994/jax/_src/cudnn/fused_attention_stablehlo.py#L1830
            # https://github.com/jax-ml/jax/blob/08c7677393672ccb85c10f1ed0bd506905c3c994/jax/_src/cudnn/fused_attention_stablehlo.py#L337
            mask_4d = einops.repeat(mask_4d, "... 1 1 -> ... t s", t=target_seq_len, s=target_seq_len)
            mask_4d = mask_4d.astype(jnp.bool)
        else:
            mask_4d = attention_mask[jnp.newaxis, jnp.newaxis, :, :]  # (1, 1, seq_len, seq_len)

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
