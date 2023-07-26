import chex

from abstract_statistical_model import StatisticalModel
from bsm.models.gaussian_processes.gaussian_processes import GPModelState, GaussianProcess
from bsm.utils.type_aliases import ModelState, StatisticalModelOutput


class GPStatisticalModel(StatisticalModel[GPModelState]):
    def __init__(self, input_dim: int, output_dim: int, ):
        super().__init__(input_dim, output_dim)
        self.model = GaussianProcess

    def _apply(self, input: chex.Array, model_state: GPModelState) -> StatisticalModelOutput:
        raise NotImplementedError

    def update(self, model_state: GPModelState, data) -> ModelState:
        pass
