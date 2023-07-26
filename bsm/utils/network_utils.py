from typing import Sequence, Optional, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn


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
    output_dim: Optional[int]
    activation: Callable = nn.swish
    hidden_size: int

    @nn.compact
    def __call__(self, x: jax.Array, hidden_state: jax.Array, train: bool = False):
        if train:
            hidden_state = jnp.zeros((x.shape[0], self.hidden_size))
        hidden_state = nn.GRUCell(hidden_state, x)
        x = hidden_state
        for feat in self.features:
            x = nn.Dense(features=feat)(x)
            x = self.activation(x)
        if self.output_dim is not None:
            x = nn.Dense(features=self.output_dim)(x)
        return hidden_state, x


class LSTMModel(nn.Module):
    features: Sequence[int]
    output_dim: Optional[int]
    activation: Callable = nn.swish
    hidden_size: int

    @nn.compact
    def __call__(self, x: jax.Array, hidden_state: jax.Array, cell_state: jax.Array, train: bool = False):
        if train:
            batch_size = x.shape[0]
            hidden_state = jnp.zeros((batch_size, self.hidden_size))
            cell_state = jnp.zeros((batch_size, self.hidden_size))
        rnn_state, hidden_state = nn.LSTMCell(cell_state, hidden_state, x)
        cell_state = rnn_state[0]
        x = hidden_state
        for feat in self.features:
            x = nn.Dense(features=feat)(x)
            x = self.activation(x)
        if self.output_dim is not None:
            x = nn.Dense(features=self.output_dim)(x)
        return cell_state, hidden_state, x
