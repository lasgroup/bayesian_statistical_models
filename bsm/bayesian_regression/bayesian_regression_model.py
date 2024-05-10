from abc import ABC, abstractmethod
from typing import Generic, Tuple

import chex
from distrax import Distribution

from bsm.utils.normalization import Normalizer, Data
from bsm.utils.type_aliases import ModelState


class BayesianRegressionModel(ABC, Generic[ModelState]):
    def __init__(self, input_dim: int, output_dim: int, logging_frequency: int = 1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalizer = Normalizer()
        self.logging_frequency = logging_frequency


    @abstractmethod
    def posterior(self, input: chex.Array, model_state: ModelState) -> Tuple[Distribution, Distribution]:
        """

        :param input: input to the function f()
        :param model_state: state of the function f()
        :return: Tuple of distributions first on  describes f(input) and second describes f(input) + noise
        """
        pass

    @abstractmethod
    def fit_model(self, data: Data, num_training_steps: int, model_state: ModelState) -> ModelState:
        pass

    def init(self, key: chex.Array) -> ModelState:
        pass
