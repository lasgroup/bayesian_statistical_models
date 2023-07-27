from typing import Sequence, Optional, Callable

import chex
import jax
from flax import linen as nn
from typing import Any
import jax.numpy as jnp

Carry = Any


class MLP(nn.Module):
    features: Sequence[int]
    output_dim: Optional[int]
    activation: Callable = nn.swish

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(features=feat)(x)
            x = self.activation(x)
        if self.output_dim is not None:
            x = nn.Dense(features=self.output_dim)(x)
        return x


class GRUModel(nn.Module):
    features: Sequence[int]
    hidden_state_size: int
    num_cells: int
    output_dim: Optional[int]
    activation: Callable = nn.swish

    @nn.compact
    def __call__(self, inputs: jax.Array, initial_h: Optional[Carry] = None):
        x = inputs
        if initial_h is None:
            initial_h = [None] * self.num_cells
        else:
            initial_h = jnp.split(initial_h, self.num_cells, -1)
        rnn_hidden_state = jnp.empty((self.hidden_state_size * self.num_cells))
        for i in range(self.num_cells):
            final_h, full_h = nn.RNN(nn.GRUCell(features=self.hidden_state_size), return_carry=True,
                                     name='rnn')(inputs=x,
                                                 initial_carry=initial_h[i])
            x = full_h
            rnn_hidden_state = rnn_hidden_state.at[..., i * self.hidden_state_size:
                                                        (i + 1) * self.hidden_state_size].set(final_h)

        def mlp(x: chex.Array):
            for feat in self.features:
                x = nn.Dense(features=feat)(x)
                x = self.activation(x)
            if self.output_dim is not None:
                x = nn.Dense(features=self.output_dim)(x)
            return x
        x = jax.vmap(mlp)(x)
        return rnn_hidden_state, x
