from abc import ABC, abstractmethod
from typing import Generic

import chex

from bsm.utils.type_aliases import ModelState, StatisticalModelOutput


class StatisticalModel(ABC, Generic[ModelState]):
    def __init__(self, input_dim: int, output_dim: int, ):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def apply(self, input: chex.Array, model_state: ModelState) -> StatisticalModelOutput:
        assert input.shape == (self.input_dim,)
        outs = self._apply(input, model_state)
        assert outs.mean.shape == outs.beta.shape == (self.output_dim,)
        assert outs.epistemic_std.shape == outs.aleatoric_std.shape == (self.output_dim,)
        return outs

    @abstractmethod
    def _apply(self, input: chex.Array, model_state: ModelState) -> StatisticalModelOutput:
        raise NotImplementedError

    @abstractmethod
    def update(self, model_state: ModelState, data) -> ModelState:
        pass
