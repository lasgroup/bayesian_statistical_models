import chex
import jax.numpy as jnp
from distrax import Normal


class ExtendedNormal(Normal):
    def __init__(self, loc: chex.Array, scale: chex.Array, aleatoric_std: chex.Array | None = None):
        super().__init__(loc=loc, scale=scale)

        if aleatoric_std is None:
            aleatoric_std = jnp.zeros_like(scale)
        self._aleatoric_std = aleatoric_std

    def aleatoric_std(self):
        return self._aleatoric_std
