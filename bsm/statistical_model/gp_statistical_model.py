import chex
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import vmap

from bsm.bayesian_regression.gaussian_processes.gaussian_processes import GPModelState, GaussianProcess
from bsm.statistical_model.abstract_statistical_model import StatisticalModel
from bsm.utils.normalization import Data, DataStats
from bsm.utils.type_aliases import StatisticalModelState
from typing import Union


class GPStatisticalModel(StatisticalModel[GPModelState]):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 f_norm_bound: float | chex.Array = 1.0,
                 delta: float = 0.1,
                 num_training_steps: Union[int, optax.Schedule] = 1000,
                 beta: chex.Array | optax.Schedule | None = None,
                 normalize: bool = True,
                 fixed_kernel_params: bool = False,
                 normalization_stats: DataStats | None = None,
                 *args, **kwargs
                 ):
        self.normalize = normalize
        model = GaussianProcess(input_dim=input_dim, output_dim=output_dim, normalize=normalize, *args, **kwargs)
        super().__init__(input_dim, output_dim, model)
        self.fixed_kernel_params = fixed_kernel_params
        self.normalization_stats = normalization_stats
        self.model = model
        if f_norm_bound is float:
            f_norm_bound = jnp.ones(output_dim) * f_norm_bound
        self.f_norm_bound = f_norm_bound
        self.delta = delta
        self.num_training_steps = num_training_steps
        if isinstance(beta, chex.Array):
            beta = optax.constant_schedule(beta)
        self._potential_beta = beta
        if isinstance(num_training_steps, int):
            self.num_training_steps = optax.constant_schedule(num_training_steps)
        else:
            self.num_training_steps = num_training_steps

    def update(self, stats_model_state: StatisticalModelState, data: Data) -> StatisticalModelState[GPModelState]:
        size = len(data.inputs)
        num_training_steps = int(self.num_training_steps(size))
        if self.fixed_kernel_params:
            new_model_state = GPModelState(history=data, data_stats=self.normalization_stats,
                                           params=stats_model_state.model_state.params)
        else:
            new_model_state = self.model.fit_model(data, num_training_steps, stats_model_state.model_state)

        if self._potential_beta is None:
            beta = self.compute_beta(new_model_state, data)
            return StatisticalModelState(model_state=new_model_state, beta=beta)
        else:
            beta = self._potential_beta(data.inputs.shape[0])
            assert beta.shape == (self.output_dim,)
            return StatisticalModelState(model_state=new_model_state, beta=beta)

    def compute_beta(self, model_state: GPModelState, data: Data):
        if self.normalize:
            inputs_norm = vmap(self.model.normalizer.normalize, in_axes=(0, None))(data.inputs,
                                                                                   model_state.data_stats.inputs)
        else:
            inputs_norm = data.inputs
        covariance_matrix = self.model.m_kernel_multiple_output(inputs_norm, inputs_norm, model_state.params)
        covariance_matrix = covariance_matrix / (self.model.output_stds ** 2)[:, None, None]
        covariance_matrix = covariance_matrix + jnp.eye(covariance_matrix.shape[-1])[None, :, :]
        sign, logdet = vmap(jnp.linalg.slogdet)(covariance_matrix)
        info_gains = 0.5 * logdet
        assert info_gains.shape == (self.output_dim,)
        betas = self.model.output_stds * jnp.sqrt(2 * (info_gains + jnp.log(self.output_dim / self.delta)))
        betas += self.f_norm_bound
        return betas


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    key = jr.PRNGKey(0)
    input_dim = 1
    output_dim = 2

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 32).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(3 * xs)], axis=1)
    ys = ys + noise_level * jr.normal(key=jr.PRNGKey(0), shape=ys.shape)
    data_std = noise_level * jnp.ones(shape=(output_dim,))
    data = Data(inputs=xs, outputs=ys)

    model = GPStatisticalModel(input_dim=input_dim,
                               output_dim=output_dim,
                               output_stds=data_std,
                               logging_wandb=False,
                               f_norm_bound=3,
                               beta=None)
    init_statistical_model_state = model.init(key=jr.PRNGKey(0))
    statistical_model_state = model.update(stats_model_state=init_statistical_model_state, data=data)

    # Test on new data
    test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(3 * test_xs)], axis=1)

    preds = model.predict_batch(test_xs, statistical_model_state)
    for j in range(output_dim):
        plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        plt.plot(test_xs, preds.mean[:, j], label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (preds.mean[:, j] - preds.statistical_model_state.beta[j] * preds.epistemic_std[:,
                                                                                     j]).reshape(-1),
                         (preds.mean[:, j] + preds.statistical_model_state.beta[j] * preds.epistemic_std[:, j]).reshape(
                             -1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(f'gp_{j}.pdf')
        plt.show()
