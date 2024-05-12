import chex
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import vmap

from bsm.bayesian_regression.bayesian_recurrent_neural_networks.rnn_ensembles import RNNState
from bsm.bayesian_regression.bayesian_neural_networks.bnn import BayesianNeuralNet
from bsm.bayesian_regression.bayesian_recurrent_neural_networks.rnn_ensembles import DeterministicGRUEnsemble, \
    ProbabilisticGRUEnsemble
from bsm.utils.general_utils import create_windowed_array
import jax.random as random
from bsm.statistical_model.abstract_statistical_model import StatisticalModel
from bsm.utils.normalization import Data
from bsm.utils.type_aliases import StatisticalModelState, StatisticalModelOutput
from typing import Union


class BRNNStatisticalModel(StatisticalModel[RNNState]):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_training_steps: Union[int, optax.Schedule] = 1000,
                 beta: chex.Array | optax.Schedule | None = None,
                 bnn_type: BayesianNeuralNet = DeterministicGRUEnsemble,
                 *model_args, **model_kwargs):
        self.bnn_type = bnn_type
        if self.bnn_type == DeterministicGRUEnsemble:
            model = DeterministicGRUEnsemble(input_dim=input_dim, output_dim=output_dim, *model_args, **model_kwargs)
        elif self.bnn_type == ProbabilisticGRUEnsemble:
            model = ProbabilisticGRUEnsemble(input_dim=input_dim, output_dim=output_dim, *model_args, **model_kwargs)
        else:
            raise NotImplementedError(f"Unknown BNN type: {self.bnn_type}")
        super().__init__(input_dim, output_dim, model)
        self.model = model
        if isinstance(num_training_steps, int):
            self.num_training_steps = optax.constant_schedule(num_training_steps)
        else:
            self.num_training_steps = num_training_steps
        if beta is None:
            beta = jnp.ones(shape=(output_dim,))
        if isinstance(beta, chex.Array):
            beta = optax.constant_schedule(beta)
        self._potential_beta = beta

    @staticmethod
    def vmap_input_axis(data_axis: int = 0) -> StatisticalModelState:
        return StatisticalModelState(
            beta=None,
            model_state=RNNState(
                vmapped_params=None,
                data_stats=None,
                calibration_alpha=None,
                hidden_state=data_axis))

    @staticmethod
    def vmap_output_axis(data_axis: int = 0) -> StatisticalModelOutput:
        return StatisticalModelOutput(mean=data_axis, epistemic_std=data_axis, aleatoric_std=data_axis,
                                      statistical_model_state=
                                      StatisticalModelState(
                                          beta=None,
                                          model_state=RNNState(
                                              vmapped_params=None,
                                              data_stats=None,
                                              calibration_alpha=None,
                                              hidden_state=data_axis))
                                      )

    def update(self, stats_model_state: StatisticalModelState[RNNState], data: Data) -> StatisticalModelState[RNNState]:
        size = len(data.inputs)
        num_training_steps = self.num_training_steps(size)
        new_model_state = self.model.fit_model(data, num_training_steps, stats_model_state.model_state)
        beta = self._potential_beta(data.inputs.shape[0])
        assert beta.shape == (self.output_dim,)
        return StatisticalModelState(model_state=new_model_state, beta=beta)

    def _predict(self,
                 input: chex.Array,
                 statistical_model_state: StatisticalModelState[RNNState]) -> StatisticalModelOutput[RNNState]:
        dist_f, dist_y = self.model.posterior(input, statistical_model_state.model_state)
        new_model_state = statistical_model_state.model_state
        new_model_state = new_model_state.replace(hidden_state=dist_f.particle_hidden_states)
        new_state = statistical_model_state.replace(model_state=new_model_state)
        statistical_model_output = StatisticalModelOutput(mean=dist_f.mean(), epistemic_std=dist_f.stddev(),
                                                          aleatoric_std=dist_y.aleatoric_std(),
                                                          statistical_model_state=new_state)
        return statistical_model_output


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    key = jr.PRNGKey(0)
    input_dim = 1
    output_dim = 2
    window_size = 10
    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 256).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=1)
    weights = jnp.asarray([1.0, 0.5, 0.25, 0.15, 0.05])
    weights = jnp.repeat(weights[..., None], repeats=2, axis=-1)
    ys = vmap(lambda x, y: jnp.convolve(y, x, 'same'), in_axes=(-1, -1))(ys, weights)
    ys = ys.transpose()
    ys = ys * (1 + noise_level * random.normal(key=random.PRNGKey(0), shape=ys.shape))
    x_train = create_windowed_array(xs, window_size=window_size)
    y_train = create_windowed_array(ys, window_size=window_size)
    data_std = noise_level * jnp.ones(shape=(output_dim,))
    data = Data(inputs=x_train, outputs=y_train)

    model = BRNNStatisticalModel(input_dim=input_dim, output_dim=output_dim, output_stds=data_std, logging_wandb=False,
                                 beta=jnp.array([1.0, 1.0]), num_particles=10, features=[64, 64, 64],
                                 bnn_type=ProbabilisticGRUEnsemble, num_training_steps=5000,
                                 train_sequence_length=window_size,
                                 weight_decay=1e-4, hidden_state_size=20, num_cells=1)

    init_stats_model_state = model.init(key=jr.PRNGKey(0))
    statistical_model_state = model.update(stats_model_state=init_stats_model_state, data=data)

    # Test on new data
    test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1)
    test_ys = vmap(lambda x, y: jnp.convolve(y, x, 'same'), in_axes=(-1, -1))(test_ys, weights)
    test_ys = test_ys.transpose()

    preds = model.predict_batch(test_xs, statistical_model_state)
    eps_std = preds.epistemic_std
    al_std = preds.aleatoric_std
    total_std = jnp.sqrt(jnp.square(eps_std) + jnp.square(al_std))

    for j in range(output_dim):
        plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        plt.plot(test_xs, preds.mean[:, j], label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (preds.mean[:, j] - preds.statistical_model_state.beta[j] * total_std[:, j]).reshape(-1),
                         (preds.mean[:, j] + preds.statistical_model_state.beta[j] * total_std[:, j]).reshape(
                             -1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig('RNN_stats_model_1_' + str(j) + '.pdf')
        plt.show()

    num_test_points = 1000
    in_domain_test_xs = jnp.linspace(d_l, d_u, num_test_points).reshape(-1, 1)
    in_domain_test_ys = jnp.concatenate([jnp.sin(in_domain_test_xs), jnp.cos(in_domain_test_xs)], axis=1)
    in_domain_test_ys = vmap(lambda x, y: jnp.convolve(y, x, 'same'), in_axes=(-1, -1))(in_domain_test_ys, weights)
    in_domain_test_ys = in_domain_test_ys.transpose()
    in_domain_preds = model.predict_batch(in_domain_test_xs, statistical_model_state)
    for j in range(output_dim):
        plt.plot(in_domain_test_xs, in_domain_preds.mean[:, j], label='Mean', color='blue')
        plt.plot(in_domain_test_xs, in_domain_test_ys[:, j], label='Fun', color='Green')
        plt.legend()
        plt.savefig('RNN_stats_model_2.pdf')
        plt.show()
