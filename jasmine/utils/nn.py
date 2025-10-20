import math
from typing import Tuple, Callable, List

from flax import nnx
import jax
import jax.numpy as jnp
import einops
from utils.preprocess import patchify, unpatchify


def _get_spatiotemporal_positional_encoding(d_model: int, max_len: int = 5000):
    """
    Creates a function that applies separate sinusoidal positional encodings to the temporal and spatial dimensions.
    """
    pe = jnp.zeros((max_len, d_model))
    position = jnp.arange(0, max_len, dtype=jnp.float32)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

    def _encode(x: jax.Array) -> jax.Array:
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
        temporal_pe = pe[None, :num_timesteps, None, :]
        x = x + temporal_pe

        # Spatial positional encoding: (1, 1, S, D)
        spatial_pe = pe[None, None, :num_spatial_patches, :]
        x = x + spatial_pe

        return x

    return _encode


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
        sow_weights: bool,
        sow_activations: bool,
    ):
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.sow_weights = sow_weights
        self.sow_activations = sow_activations

        self.spatial_norm = nnx.LayerNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
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
            dtype=self.param_dtype,  # layer norm in full precision
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
            dtype=self.param_dtype,  # layer norm in full precision
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
        z_BTNM = self.spatial_norm(x_BTNM)
        z_BTNM = self.spatial_attention(z_BTNM, sow_weights=self.sow_weights)
        x_BTNM = x_BTNM + z_BTNM

        # --- Temporal attention ---
        x_BNTM = x_BTNM.swapaxes(1, 2)
        z_BNTM = self.temporal_norm(x_BNTM)
        z_BNTM = self.temporal_attention(z_BNTM, sow_weights=self.sow_weights)
        x_BNTM = x_BNTM + z_BNTM
        x_BTNM = x_BNTM.swapaxes(1, 2)

        # --- Feedforward ---
        z_BTNM = self.ffn_norm(x_BTNM)
        z_BTND = self.ffn_dense1(z_BTNM)
        z_BTND = jax.nn.gelu(z_BTND)
        z_BTNM = self.ffn_dense2(z_BTND)
        x_BTNM = x_BTNM + z_BTNM
        if self.sow_activations:
            self.sow(nnx.Intermediate, "activations", x_BTNM)
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
        V: vocabulary size
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
        sow_weights: bool = False,
        sow_activations: bool = False,
        sow_logits: bool = False,
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
        self.sow_logits = sow_logits
        self.sow_weights = sow_weights
        self.sow_activations = sow_activations

        self.input_norm1 = nnx.LayerNorm(
            num_features=self.input_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
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
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )

        self.pos_enc = _get_spatiotemporal_positional_encoding(
            self.model_dim, max_len=max_len
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
                    sow_weights=self.sow_weights,
                    sow_activations=self.sow_activations,
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
        x_BTNM = self.pos_enc(x_BTNM)
        for block in self.blocks:
            x_BTNM = block(x_BTNM)

        x_BTNV = self.output_dense(x_BTNM)
        if self.sow_logits:
            self.sow(nnx.Intermediate, "logits", x_BTNV)
        return x_BTNV


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
        sow_weights: bool,
        sow_activations: bool,
    ):
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.decode = decode
        self.sow_weights = sow_weights
        self.sow_activations = sow_activations

        self.temporal_norm = nnx.LayerNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )
        self.spatial_norm = nnx.LayerNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )
        self.ffn_norm = nnx.LayerNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
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
    def __call__(
        self, x_BTNM: jax.Array, pos_index: Tuple[jax.Array, jax.Array] | None = None
    ) -> jax.Array:
        # --- Spatial attention ---
        B, T, N, M = x_BTNM.shape
        z_FNM = einops.rearrange(x_BTNM, "b t n m -> (b t) n m")
        z_FNM = self.spatial_norm(z_FNM)
        z_FNM = self.spatial_attention(z_FNM, sow_weights=self.sow_weights)
        z_BTNM = einops.rearrange(z_FNM, "(b t) n m -> b t n m", t=T)
        x_BTNM = x_BTNM + z_BTNM
        # --- Temporal attention ---
        z_PTM = einops.rearrange(x_BTNM, "b t n m -> (b n) t m")
        z_PTM = self.temporal_norm(z_PTM)
        z_PTM = self.temporal_attention(z_PTM, sow_weights=self.sow_weights)
        z_BTNM = einops.rearrange(z_PTM, "(b n) t m -> b t n m", n=N)
        x_BTNM = x_BTNM + z_BTNM
        # --- Feedforward ---
        z_BTNM = self.ffn_norm(x_BTNM)
        z_BTND = self.ffn_dense1(z_BTNM)
        z_BTND = jax.nn.gelu(z_BTND)
        z_BTNM = self.ffn_dense2(z_BTND)
        x_BTNM = x_BTNM + z_BTNM
        if self.sow_activations:
            self.sow(nnx.Intermediate, "activations", x_BTNM)

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
        V: vocabulary size
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
        sow_logits: bool = False,
        sow_weights: bool = False,
        sow_activations: bool = False,
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
        self.sow_logits = sow_logits
        self.sow_weights = sow_weights
        self.sow_activations = sow_activations

        self.input_norm1 = nnx.LayerNorm(
            num_features=self.input_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
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
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )

        self.pos_enc = _get_spatiotemporal_positional_encoding(
            self.model_dim, max_len=max_len
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
                    sow_weights=self.sow_weights,
                    sow_activations=self.sow_activations,
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

    def __call__(
        self, x_BTNI: jax.Array, pos_index: Tuple[jax.Array, jax.Array] | None = None
    ) -> jax.Array:
        x_BTNI = self.input_norm1(x_BTNI)
        x_BTNM = self.input_dense(x_BTNI)
        x_BTNM = self.input_norm2(x_BTNM)
        x_BTNM = self.pos_enc(x_BTNM)
        for block in self.blocks:
            x_BTNM = block(x_BTNM, pos_index)

        x_BTNV = self.output_dense(x_BTNM)
        if self.sow_logits:
            self.sow(nnx.Intermediate, "logits", x_BTNV)
        return x_BTNV

def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]

# TODO mihir clean this up
class TimestepEmbedder(nnx.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size: int,
        rngs,
        bias: bool = True
    ):
        self.hidden_size = hidden_size
        self.bias = bias
        self.dense1 = nnx.Linear(self.hidden_size, self.hidden_size, kernel_init=nnx.initializers.normal(0.02), bias_init=nnx.initializers.normal(0.02), rngs=rngs)
        self.dense2 = nnx.Linear(self.hidden_size, self.hidden_size, kernel_init=nnx.initializers.normal(0.02), bias_init=nnx.initializers.normal(0.02), rngs=rngs)

    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = self.dense1(x)
        x = nnx.silu(x)
        x = self.dense2(x)
        return x
    
    # t is between [0, 1].
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                            These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        print(t)
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.hidden_size
        half = dim // 2
        freqs = jnp.exp( -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding

# TODO mihir remove this
class PatchEmbed(nnx.Module):
    """ 2D Image to Patch Embedding """

    def __init__(
        self,
        patch_size: int,
        hidden_size: int,
        rngs: nnx.Rngs,
        bias: bool = True, 
    ):
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.bias = bias
        patch_tuple = (self.patch_size, self.patch_size)
        self.conv = nnx.Conv(self.hidden_size, self.hidden_size, patch_tuple, patch_tuple, padding="VALID", rngs=rngs)

    def __call__(self, x):
        B, T, H, W, C = x.shape
        num_patches = (H // self.patch_size)
        x = einops.rearrange(x, "b t h w c -> (b t) h w c")
        x = self.conv(x) # (B, P, P, hidden_size)
        x = einops.rearrange(x, '(b t) h w c -> b t (h w) c', t=T, h=num_patches, w=num_patches)
        return x

class DiTBlock(nnx.Module):
    """DiT block"""

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        use_flash_attention: bool = False,
        decode: bool = False,
        sow_weights: bool = False,
        sow_activations: bool = False,
    ):
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.decode = decode
        self.sow_weights = sow_weights
        self.sow_activations = sow_activations
        self.condition_up = nnx.Linear(self.model_dim, 9 * self.model_dim, kernel_init=nnx.initializers.normal(0.02), bias_init=nnx.initializers.normal(0.02), rngs=rngs)
        
        self.temporal_norm = nnx.LayerNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )
        self.spatial_norm = nnx.LayerNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )
        self.ffn_norm = nnx.LayerNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
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



    def __call__(self, x_BTNM, c):
        B, T, N, M = x_BTNM.shape
        
        # Calculate adaLn modulation parameters.
        c = nnx.silu(c)
        c = self.condition_up(c)
        shift_spatial, scale_spatial, gate_spatial, shift_temporal, scale_temporal, gate_temporal, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 9, axis=-1)

        # TODO find a better way to do this
        shift_spatial = jnp.tile(shift_spatial, (T, 1))
        scale_spatial = jnp.tile(scale_spatial, (T, 1))
        gate_spatial = jnp.tile(gate_spatial, (T, 1))
        shift_temporal = jnp.tile(shift_temporal, (N, 1))
        scale_temporal = jnp.tile(scale_temporal, (N, 1))
        gate_temporal = jnp.tile(gate_temporal, (N, 1))

        # --- Spatial attention ---
        z_FNM = einops.rearrange(x_BTNM, "b t n m -> (b t) n m")
        z_FNM = self.spatial_norm(z_FNM)
        z_FNM = modulate(z_FNM, shift_spatial, scale_spatial)
        z_FNM = self.spatial_attention(z_FNM, sow_weights=self.sow_weights)
        z_FNM = z_FNM * gate_spatial[:, None]
        z_BTNM = einops.rearrange(z_FNM, "(b t) n m -> b t n m", t=T)
        x_BTNM = x_BTNM + z_BTNM
        # --- Temporal attention ---
        z_PTM = einops.rearrange(x_BTNM, "b t n m -> (b n) t m")
        z_PTM = self.temporal_norm(z_PTM)
        z_PTM = modulate(z_PTM, shift_temporal, scale_temporal)
        z_PTM = self.temporal_attention(z_PTM, sow_weights=self.sow_weights)
        z_PTM = z_PTM * gate_temporal[:, None]
        z_BTNM = einops.rearrange(z_PTM, "(b n) t m -> b t n m", n=N)
        x_BTNM = x_BTNM + z_BTNM
        # --- Feedforward ---
        z_BTNM = self.ffn_norm(x_BTNM)
        z_BTNM = modulate(z_BTNM, shift_mlp[:, None], scale_mlp[:, None])
        z_BTND = self.ffn_dense1(z_BTNM)
        z_BTND = jax.nn.gelu(z_BTND)
        z_BTNM = self.ffn_dense2(z_BTND)
        z_BTNM = z_BTNM * gate_mlp[:, None, None]
        x_BTNM = x_BTNM + z_BTNM
        if self.sow_activations:
            self.sow(nnx.Intermediate, "activations", x_BTNM)

        return x_BTNM

class DiffusionTransformer(nnx.Module):
    """Diffusion transformer"""

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.pos_enc = _get_spatiotemporal_positional_encoding(
            self.model_dim
        )
        self.blocks: nnx.List[DiTBlock] = nnx.List([])
        for _ in range(self.num_blocks):
            self.blocks.append(
                DiTBlock(
                    model_dim=self.model_dim,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    param_dtype=self.param_dtype,
                    dtype=self.dtype,
                    rngs=rngs,
                )
            )
        self.patch_size = 4
        self.time_step_embedder = TimestepEmbedder(self.model_dim, rngs=rngs)
        self.step_size_embedder = TimestepEmbedder(self.model_dim, rngs=rngs)

        self.patch_dim = self.patch_size * self.patch_size * 3
        self.patch_up = nnx.Linear(self.patch_dim, self.model_dim, kernel_init=nnx.initializers.normal(0.02), bias_init=nnx.initializers.normal(0.02), rngs=rngs)
        self.condition_up = nnx.Linear(self.model_dim, 2*self.model_dim, kernel_init=nnx.initializers.normal(0.02), bias_init=nnx.initializers.normal(0.02), rngs=rngs)
        self.layer_norm = nnx.LayerNorm(self.ffn_dim, use_bias=False, use_scale=False, rngs=rngs)
        self.dense2 = nnx.Linear(self.model_dim, self.patch_dim, kernel_init=nnx.initializers.normal(0.02), bias_init=nnx.initializers.normal(0.02), rngs=rngs)

    def __call__(self, x_BTHWC: jax.Array, t: jax.Array, dt: jax.Array) -> jax.Array:
        B,T,H,W,C = x_BTHWC.shape
        x_BTNP = patchify(x_BTHWC, self.patch_size)
        x_BTNM = self.patch_up(x_BTNP)
        x_BTNM = self.pos_enc(x_BTNM)
        te = self.time_step_embedder(t) # (B, hidden_size)
        dte = self.step_size_embedder(dt) # (B, hidden_size)
        c = te + dte # TODO change this to prepending later
        for block in self.blocks:
            x_BTNM = block(x_BTNM, c)

        c = nnx.silu(c)
        c = self.condition_up(c)
        shift, scale = jnp.split(c, 2, axis=-1)
        x_BTNM = self.layer_norm(x_BTNM)
        x_BTNM = modulate(x_BTNM, shift[:, None], scale[:, None])
        x_BTNP = self.dense2(x_BTNM)
        x_BTHWC = unpatchify(x_BTNP, self.patch_size, H, W)
        print(x_BTHWC.shape)

        return x_BTHWC


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
        self,
        latent_dim: int,
        num_latents: int,
        dropout: float,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.dropout = dropout
        self.dtype = dtype

        self.codebook = nnx.Param(
            normalize(
                nnx.initializers.normal(stddev=1)(
                    rngs.params(), (self.num_latents, self.latent_dim)
                )
            )
        )
        self.drop = nnx.Dropout(self.dropout, rngs=rngs)

    def __call__(
        self, x_DL: jax.Array, training: bool
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # --- Compute distances ---
        x_DL = x_DL.astype(self.dtype)
        codebook = self.codebook.value.astype(self.dtype)

        x_DL = normalize(x_DL)
        normalized_codebook_KL = normalize(codebook)
        distance_DK = -jnp.matmul(x_DL, normalized_codebook_KL.T)
        if training:
            distance_DK = self.drop(distance_DK)

        # --- Get indices and embeddings ---
        indices_D = jnp.argmin(distance_DK, axis=-1)
        z_DL = codebook[indices_D]

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

    def attention_fn(
        query_BTHD, key_BSHD, value_BSHD, bias=None, mask_B111=None, **kwargs
    ):
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

        attention_mask = jnp.ones((Q, K), dtype=jnp.bool_)
        attention_mask = attention_mask.at[T:, :].set(False)
        attention_mask = attention_mask.at[:, S:].set(False)

        mask_11TS = attention_mask[jnp.newaxis, jnp.newaxis, :, :]

        bias_4d = (
            jnp.pad(
                _merge_batch_dims(bias),
                ((0, 0), (0, 0), (0, pad_size_Q), (0, pad_size_K)),
            )
            if bias is not None
            else None
        )

        # NOTE: jax.nn.dot_product_attention does not support dropout
        output_4d = jax.nn.dot_product_attention(
            query=query_BQHD,
            key=key_BKHD,
            value=value_BKHD,
            bias=bias_4d,
            mask=mask_11TS,
            implementation=implementation,
            is_causal=is_causal,
        )
        return output_4d[..., :T, :, :].reshape(original_shape)

    return attention_fn
