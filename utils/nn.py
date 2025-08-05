import math
from typing import Tuple, Callable, List

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
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

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
                self.use_flash_attention, is_causal=False
            ),
            rngs=rngs,
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
    def __call__(self, x_BTNM: jax.Array) -> jax.Array:
        # --- Spatial attention ---
        z_BTNM = self.spatial_pos_enc(x_BTNM)
        z_BTNM = self.spatial_norm(z_BTNM)
        z_BTNM = self.spatial_attention(z_BTNM)
        x_BTNM = x_BTNM + z_BTNM

        # --- Temporal attention ---
        x_BNTM = x_BTNM.swapaxes(1, 2)
        z_BNTM = self.temporal_pos_enc(x_BNTM)
        z_BNTM = self.temporal_norm(z_BNTM)
        z_BNTM = self.temporal_attention(z_BNTM)
        x_BNTM = x_BNTM + z_BNTM
        x_BTNM = x_BNTM.swapaxes(1, 2)

        # --- Feedforward ---
        z_BTNM = self.ffn_norm(x_BTNM)
        z_BTND = self.ffn_dense1(z_BTNM)
        z_BTND = jax.nn.gelu(z_BTND)
        z_BTNM = self.ffn_dense2(z_BTND)
        x_BTNM = x_BTNM + z_BTNM

        return x_BTNM


class STTransformer(nnx.Module):
    """
    Dimension keys:
        B: batch size
        T: number of frames
        N: number of patches per frame
        I: number of input features
        M: model dimension
        D: FFN dimension
        O: number of output features
    """
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

    def __call__(self, x_BTNI: jax.Array) -> jax.Array:
        x_BTNI = self.input_norm1(x_BTNI)
        x_BTNM = self.input_dense(x_BTNI)
        x_BTNM = self.input_norm2(x_BTNM)

        for block in self.blocks:
            x_BTNM = block(x_BTNM)

        x_BTNO = self.output_dense(x_BTNM)
        return x_BTNO

class TransformerBlock(nnx.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        decode: bool,
        rngs: nnx.Rngs,
    ):
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.decode = decode

        self.temporal_pos_enc = PositionalEncoding(self.model_dim)
        self.spatial_pos_enc = PositionalEncoding(self.model_dim)
        self.temporal_norm = nnx.LayerNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.spatial_norm = nnx.LayerNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.ffn_norm = nnx.LayerNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.temporal_attention = nnx.MultiHeadAttention(
            num_heads=self.num_heads,
            in_features=self.model_dim,
            qkv_features=self.model_dim,
            dropout_rate=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(
                self.use_flash_attention, is_causal=True
            ),
            rngs=rngs,
            decode=self.decode,
        )
        self.spatial_attention = nnx.MultiHeadAttention(
            num_heads=self.num_heads,
            in_features=self.model_dim,
            qkv_features=self.model_dim,
            dropout_rate=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(
                self.use_flash_attention, is_causal=True
            ),
            rngs=rngs,
            decode=self.decode,
        )
        self.ffn_dense1 = nnx.Linear(
            in_features=self.model_dim,
            out_features=self.ffn_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.ffn_dense2 = nnx.Linear(
            in_features=self.ffn_dim,
            out_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

    @nnx.remat
    def __call__(self, x_BTNM: jax.Array, pos_index: Tuple[jax.Array, jax.Array] | None = None) -> jax.Array:
        # --- Spatial attention ---
        B, T, N, M = x_BTNM.shape
        z_FNM = einops.rearrange(x_BTNM, "b t n m -> (b t) n m")
        z_FNM = self.spatial_norm(z_FNM)
        z_FNM = self.spatial_attention(z_FNM)
        z_BTNM = einops.rearrange(z_FNM, "(b t) n m -> b t n m", t=T)
        x_BTNM = x_BTNM + z_BTNM
        # --- Temporal attention ---
        z_PTM = einops.rearrange(x_BTNM, "b t n m -> (b n) t m")
        z_PTM = self.temporal_norm(z_PTM)
        z_PTM = self.temporal_attention(z_PTM)
        z_BTNM = einops.rearrange(z_PTM, "(b n) t m -> b t n m", n=N)
        x_BTNM = x_BTNM + z_BTNM
        # --- Feedforward ---
        z_BTNM = self.ffn_norm(x_BTNM)
        z_BTND = self.ffn_dense1(z_BTNM)
        z_BTND = jax.nn.gelu(z_BTND)
        z_BTNM = self.ffn_dense2(z_BTND)
        x_BTNM = x_BTNM + z_BTNM

        return x_BTNM

class Transformer(nnx.Module):
    """
    Dimension keys:
        B: batch size
        T: number of frames
        N: number of patches per frame
        I: number of input features
        M: model dimension
        D: FFN dimension
        O: number of output features
        F: number of frames in batch
        P: number of patch positions in batch
    """
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

        self.blocks: List[TransformerBlock] = []
        for _ in range(self.num_blocks):
            self.blocks.append(
                TransformerBlock(
                    model_dim=self.model_dim,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    param_dtype=self.param_dtype,
                    dtype=self.dtype,
                    use_flash_attention=self.use_flash_attention,
                    decode=decode,
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

    def __call__(self, x_BTNI: jax.Array, pos_index: Tuple[jax.Array, jax.Array] | None = None) -> jax.Array:
        x_BTNI = self.input_norm1(x_BTNI)
        x_BTNM = self.input_dense(x_BTNI)
        x_BTNM = self.input_norm2(x_BTNM)

        for block in self.blocks:
            x_BTNM = block(x_BTNM, pos_index)

        x_BTNV = self.output_dense(x_BTNM)
        return x_BTNV

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

    flax.nnx.MultiHeadAttention provides tensors with shape (batch..., length, num_heads, head_dim),
    but jax.nn.dot_product_attention expects (batch, length, num_heads, head_dim). We reshape to
    ensure compatibility. cuDNN's flash attention additionally requires a sequence length that
    is a multiple of 4. We pad the sequence length to the nearest multiple of 4 and mask
    accordingly. Note that cuDNN requires the mask to be broadcast before calling the attention
    function due to strict shape checking.
    """

    def attention_fn(query_BTHD, key_BSHD, value_BSHD, bias=None, mask_B111=None, **kwargs):
        implementation = "cudnn" if use_flash_attention else None

        def _merge_batch_dims(x):
            return einops.rearrange(x, "... l h k -> (...) l h k")

        def _pad(x, pad_size):
            return jnp.pad(x, ((0, 0), (0, pad_size), (0, 0), (0, 0)))

        original_shape = query_BTHD.shape
        T = query_BTHD.shape[-3]
        S = key_BSHD.shape[-3]

        # Pad to nearest multiple of 4
        Q = ((T + 3) // 4) * 4
        pad_size_Q = Q - T
        K = ((S + 3) // 4) * 4
        pad_size_K = K - S

        query_BQHD = _pad(_merge_batch_dims(query_BTHD), pad_size_Q)
        key_BKHD = _pad(_merge_batch_dims(key_BSHD), pad_size_K)
        value_BKHD = _pad(_merge_batch_dims(value_BSHD), pad_size_K)
        B = query_BQHD.shape[0]

        attention_mask = jnp.ones((Q, K), dtype=jnp.bool_)
        attention_mask = attention_mask.at[Q:, :].set(False)
        attention_mask = attention_mask.at[:, K:].set(False)

        # Handle causal mask for cached decoder self-attention (from nnx.MultiHeadAttention)
        if mask_B111 is not None:
            # FIXME (f.srambical): Why do we need this?
            mask_B111 = _merge_batch_dims(mask_B111)
            # We need to broadcast T and S dimensions to target_seq_len since cudnn attention strictly checks the mask shape
            # https://github.com/jax-ml/jax/issues/28974
            # https://github.com/jax-ml/jax/blob/08c7677393672ccb85c10f1ed0bd506905c3c994/jax/_src/cudnn/fused_attention_stablehlo.py#L1830
            # https://github.com/jax-ml/jax/blob/08c7677393672ccb85c10f1ed0bd506905c3c994/jax/_src/cudnn/fused_attention_stablehlo.py#L337
            mask_B1TS = einops.repeat(mask_B111, "... 1 1 -> ... t s", t=Q, s=K)
            mask_B1TS = mask_B111.astype(jnp.bool)
        else:
            mask_11TS = attention_mask[jnp.newaxis, jnp.newaxis, :, :]
            mask_B1TS = jnp.broadcast_to(mask_11TS, (B, 1, Q, K))

        bias_4d = _merge_batch_dims(bias) if bias is not None else None

        # NOTE: jax.nn.dot_product_attention does not support dropout
        output_4d = jax.nn.dot_product_attention(
            query=query_BQHD,
            key=key_BKHD,
            value=value_BKHD,
            bias=bias_4d,
            mask=mask_B1TS,
            implementation=implementation,
            is_causal=is_causal,
        )
        return output_4d[..., :T, :, :].reshape(original_shape)

    return attention_fn
