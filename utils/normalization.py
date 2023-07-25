from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree


class Stats(NamedTuple):
    mean: chex.Array
    std: chex.Array


class DataStats(NamedTuple):
    inputs: Stats
    outputs: Stats


class Normalizer:
    def __init__(self, num_correction=1e-6):
        self.num_correction = num_correction

    def compute_stats(self, data: PyTree) -> PyTree[Stats]:
        return jtu.tree_map(self.get_stats, data)

    @partial(jax.jit, static_argnums=0)
    def get_stats(self, data: chex.Array) -> Stats:
        assert data.ndim == 2
        mean = jnp.mean(data, axis=0)
        std = jnp.std(data, axis=0) + self.num_correction
        return Stats(mean, std)

    @partial(jax.jit, static_argnums=0)
    def normalize(self, datum: chex.Array, stats: Stats) -> chex.Array:
        assert datum.ndim == 1
        return (datum - stats.mean) / stats.std

    @partial(jax.jit, static_argnums=0)
    def normalize_std(self, datum: chex.Array, stats: Stats) -> chex.Array:
        assert datum.ndim == 1
        return datum / stats.std

    @partial(jax.jit, static_argnums=0)
    def denormalize(self, datum: chex.Array, stats: Stats) -> chex.Array:
        assert datum.ndim == 1
        return datum * stats.std + stats.mean

    @partial(jax.jit, static_argnums=0)
    def denormalize_std(self, datum: chex.Array, stats: Stats) -> chex.Array:
        assert datum.ndim == 1
        return datum * stats.std
