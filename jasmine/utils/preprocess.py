import einops
import jax
import jax.numpy as jnp


def patchify(videos: jax.Array, size: int) -> jax.Array:
    B, T, H, W, C = videos.shape
    x = jnp.pad(videos, ((0, 0), (0, 0), (0, -H % size), (0, -W % size), (0, 0)))
    return einops.rearrange(
        x, "b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)", hp=size, wp=size
    )


def unpatchify(patches: jax.Array, size: int, h_out: int, w_out: int) -> jax.Array:
    h_pad = -h_out % size
    hn = (h_out + h_pad) // size
    x = einops.rearrange(
        patches,
        "b t (hn wn) (hp wp c) -> b t (hn hp) (wn wp) c",
        hp=size,
        wp=size,
        hn=hn,
    )
    return x[:, :, :h_out, :w_out]
