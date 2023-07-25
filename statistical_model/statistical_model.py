from abc import ABC, abstractmethod
from typing import TypeVar, Generic

import chex

EpistemicStd = chex.Array
AleatoricStd = chex.Array
Mean = chex.Array
Beta = chex.Array


@chex.dataclass
class StatisticalModelOutput(Generic[ModelState]):
    mean: Mean
    epistemic_std: EpistemicStd
    aleatoric_std: AleatoricStd
    beta: Beta
    model_state: ModelState


class StatisticalModel(ABC, Generic[ModelState]):
    def __init__(self, input_dim: int, output_dim: int, ):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def apply(self, input, model_state: ModelState) -> StatisticalModelOutput:
        assert input.shape == (self.input_dim,)
        outs = self._apply(input, model_state)
        assert outs[0].shape == outs[1].shape == outs[2].shape == outs[3].shape == (self.output_dim,)
        return outs

    @abstractmethod
    def _apply(self, input, model_state: ModelState) -> StatisticalModelOutput:
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs) -> ModelState:
        pass
