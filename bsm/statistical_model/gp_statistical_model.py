import chex
import optax

from abstract_statistical_model import StatisticalModel
from bsm.models.gaussian_processes.gaussian_processes import GPModelState, GaussianProcess
from bsm.models.gaussian_processes.kernels import Kernel
from bsm.utils.type_aliases import StatisticalModelState


class GPStatisticalModel(StatisticalModel[GPModelState]):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 output_stds: chex.Array,
                 kernel: Kernel | None = None,
                 weight_decay: float = 0.0,
                 lr_rate: optax.Schedule | float = optax.constant_schedule(1e-2),
                 seed: int = 0):
        super().__init__(input_dim, output_dim)
        self.model = GaussianProcess(input_dim=input_dim, output_dim=output_dim, output_stds=output_stds,
                                     kernel=kernel, weight_decay=weight_decay, lr_rate=lr_rate, seed=seed)

    def update(self, model_state: GPModelState, data) -> StatisticalModelState[GPModelState]:
        pass
