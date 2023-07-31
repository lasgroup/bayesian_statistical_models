from jax import numpy as jnp


def create_windowed_array(arr: jnp.array, window_size: int = 10) -> jnp.array:
    """Sliding window over an array along the first axis."""
    arr_strided = jnp.stack([arr[i:(-window_size + i)] for i in range(window_size)], axis=-2)
    assert arr_strided.shape == (arr.shape[0] - window_size, window_size, arr.shape[-1])
    return jnp.array(arr_strided)
