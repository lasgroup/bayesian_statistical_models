from abc import ABC, abstractmethod
from typing import Generic

import chex

from bsm.bayesian_regression.bayesian_regression_model import BayesianRegressionModel
from bsm.utils.type_aliases import ModelState, StatisticalModelOutput, StatisticalModelState


class StatisticalModel(ABC, Generic[ModelState]):
    def __init__(self, input_dim: int, output_dim: int, model: BayesianRegressionModel[ModelState]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = model

    def __call__(self,
                input: chex.Array,
                statistical_model_state: StatisticalModelState[ModelState]) -> StatisticalModelOutput[ModelState]:
        assert input.shape == (self.input_dim,)
        outs = self._predict(input, statistical_model_state)
        assert outs.mean.shape == outs.statistical_model_state.beta.shape == (self.output_dim,)
        assert outs.epistemic_std.shape == outs.aleatoric_std.shape == (self.output_dim,)
        return outs

    def _predict(self,
                 input: chex.Array,
                 statistical_model_state: StatisticalModelState[ModelState]) -> StatisticalModelOutput[ModelState]:
        dist_f, dist_y = self.model.posterior(input, statistical_model_state.model_state)
        statistical_model = StatisticalModelOutput(mean=dist_f.mean(), epistemic_std=dist_f.stddev(),
                                                   aleatoric_std=dist_y.aleatoric_std(),
                                                   statistical_model_state=statistical_model_state)
        return statistical_model

    @abstractmethod
    def update(self,
               statistical_model_state: StatisticalModelState[ModelState],
               data) -> StatisticalModelState[ModelState]:
        pass

    @abstractmethod
    def init(self, key: chex.PRNGKey) -> ModelState:
        pass
