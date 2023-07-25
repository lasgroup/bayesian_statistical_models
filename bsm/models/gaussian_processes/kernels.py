from abc import ABC, abstractmethod

import chex
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softplus
from jaxtyping import PyTree


class Kernel(ABC):
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    def apply(self, x1: chex.Array, x2: chex.Array, kernel_params: PyTree) -> chex.Array:
        assert x1.shape == x2.shape == (self.input_dim,)
        return self._apply(x1, x2, kernel_params)

    @abstractmethod
    def _apply(self, x1: chex.Array, x2: chex.Array, kernel_params: PyTree) -> chex.Array:
        raise NotImplementedError

    @abstractmethod
    def init(self, key: chex.PRNGKey) -> PyTree:
        raise NotImplementedError


class RBF(Kernel):
    def __init__(self, input_dim: int):
        super().__init__(input_dim)

    def _apply(self, x1: chex.Array, x2: chex.Array, kernel_params: PyTree) -> chex.Array:
        pseudo_length_scale = kernel_params['pseudo_length_scale']
        length_scale = softplus(pseudo_length_scale)
        return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2 / length_scale ** 2))

    def init(self, key: chex.PRNGKey) -> PyTree:
        return {'pseudo_length_scale': jr.normal(key, shape=())}


if __name__ == "__main__":
    x = 10

