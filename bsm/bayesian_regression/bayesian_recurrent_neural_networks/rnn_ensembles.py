import time
from collections import OrderedDict
from functools import partial
from typing import Sequence, Dict, Tuple, Optional, Callable

import chex
import flax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap, jit
from jax.scipy.stats import norm
from jaxtyping import PyTree
import jax.random as jr
import jax.tree_util as jtu
from brax.training.replay_buffers import UniformSamplingQueue
from bsm.bayesian_regression.bayesian_neural_networks.bnn import BNNState

import wandb

from bsm.utils.general_utils import create_windowed_array
from bsm.utils.network_utils import GRUModel
from bsm.utils.normalization import Normalizer, DataStats, Data
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
import flax.linen as nn
from bsm.utils.particle_distribution import GRUParticleDistribution
from bsm.bayesian_regression.bayesian_neural_networks.fsvgd_ensemble import prepare_stein_kernel

NO_EVAL_VALUE = 123213142134.645954392592


@chex.dataclass
class RNNState(BNNState):
    hidden_state: Optional[chex.Array] = None


class DeterministicGRUEnsemble(DeterministicEnsemble):
    def __init__(self,
                 features: Sequence[int],
                 hidden_state_size: int,
                 num_cells: int,
                 train_sequence_length: int = 1,
                 activation: Callable = nn.swish,
                 *args,
                 **kwargs,
                 ):
        self.train_sequence_length = train_sequence_length
        super().__init__(features=features, *args, **kwargs)
        self.hidden_state_size = hidden_state_size
        self.model = GRUModel(features=features, num_cells=num_cells,
                              output_dim=self.output_dim, hidden_state_size=hidden_state_size, activation=activation)
        self.normalizer = Normalizer()
        self.hidden_size = self.hidden_state_size * num_cells

    def set_up_data_buffers(self):
        dummy_data_sample = Data(inputs=jnp.zeros((self.train_sequence_length, self.input_dim)),
                                 outputs=jnp.zeros((self.train_sequence_length, self.output_dim)))
        self.train_buffer = UniformSamplingQueue(max_replay_size=self.max_buffer_size,
                                                 dummy_data_sample=dummy_data_sample,
                                                 sample_batch_size=self.batch_size)

        self.eval_buffer = UniformSamplingQueue(max_replay_size=self.max_buffer_size,
                                                dummy_data_sample=dummy_data_sample,
                                                sample_batch_size=self.eval_batch_size)

    def _apply(self,
               params: PyTree,
               x: chex.Array,
               data_stats: DataStats,
               hidden_state: Optional[chex.Array] = None,
               ) -> [chex.Array, chex.Array, chex.Array]:
        assert x.shape[-1] == self.input_dim
        ndim = x.ndim
        x = x.reshape(-1, self.input_dim)
        x = jax.vmap(self.normalizer.normalize, in_axes=(0, None))(x, data_stats.inputs)
        new_hidden_state, out = self.model.apply({'params': params}, x, hidden_state)
        out = out.reshape(self.output_dim) if ndim == 1 else out
        assert out.shape[-1] == self.output_dim
        assert new_hidden_state.shape[-1] == self.hidden_size
        al_std_norm = self.normalizer.normalize_std(self.output_stds, data_stats.outputs)
        if ndim > 1:
            al_std_norm = jnp.repeat(al_std_norm[None, ...], repeats=x.shape[0], axis=0)
        return new_hidden_state, out, al_std_norm

    def _apply_train(self,
                     params: PyTree,
                     x: chex.Array,
                     data_stats: DataStats) -> [chex.Array, chex.Array]:
        assert x.shape[-1] == self.input_dim and x.ndim == 2
        _, out, al_std_norm = self._apply(params, x, data_stats)
        return out, al_std_norm

    def apply_eval(self,
                   params: PyTree,
                   x: chex.Array,
                   data_stats: DataStats
                   ) -> [chex.Array, chex.Array]:
        assert x.shape[-1] == self.input_dim and x.ndim == 2
        mean, std = self._apply_train(params, x, data_stats)
        mean = vmap(self.normalizer.denormalize, in_axes=(0, None))(mean, data_stats.outputs)
        std = vmap(self.normalizer.denormalize_std, in_axes=(0, None))(std, data_stats.outputs)
        return mean, std

    def loss(self,
             vmapped_params: PyTree,
             inputs: chex.Array,
             outputs: chex.Array,
             data_stats: DataStats) -> [jax.Array, Dict]:
        apply_ensemble_one = vmap(self._apply_train, in_axes=(0, None, None), out_axes=0)
        apply_ensemble = vmap(apply_ensemble_one, in_axes=(None, 0, None), out_axes=1, axis_name='batch')
        predicted_outputs, predicted_stds = apply_ensemble(vmapped_params, inputs, data_stats)

        target_outputs_norm = vmap(vmap(self.normalizer.normalize, in_axes=(0, None)), in_axes=(0, None)) \
            (outputs, data_stats.outputs)
        loss = jax.vmap(self.per_step_loss, in_axes=(-2, -2, -2)) \
            (predicted_outputs, predicted_stds, target_outputs_norm)
        loss = loss.mean()
        mse = jnp.mean((predicted_outputs - target_outputs_norm[None, ...]) ** 2)
        return loss, mse

    def per_step_loss(self,
                      predicted_outputs: chex.Array,
                      predicted_stds: chex.Array,
                      target_outputs: jax.Array
                      ):
        return self._neg_log_posterior(predicted_outputs, predicted_stds, target_outputs)

    @partial(jit, static_argnums=0)
    def eval_ll(self,
                vmapped_params: chex.Array,
                inputs: chex.Array,
                outputs: chex.Array,
                data_stats: DataStats) -> chex.Array:

        apply_ensemble_one = vmap(self.apply_eval, in_axes=(0, None, None), out_axes=0)
        predicted_targets, predicted_stds = vmap(apply_ensemble_one, in_axes=(None, 0, None), out_axes=1,
                                                 axis_name='batch') \
            (vmapped_params, inputs, data_stats)
        nll = self._nll(predicted_targets, predicted_stds, outputs)
        mse = jnp.mean((predicted_targets - outputs) ** 2)
        statistics = OrderedDict(nll=nll, mse=mse)
        return statistics

    def _init(self, key):
        variables = self.model.init(key, jnp.ones(shape=(1, self.input_dim)))
        if 'params' in variables:
            stats, params = flax.core.pop(variables, 'params')
        else:
            stats, params = variables
        del variables  # Delete variables to avoid wasting resources
        return params

    def init(self, key: chex.PRNGKey) -> RNNState:
        inputs = jnp.zeros(shape=(1, self.input_dim))
        outputs = jnp.zeros(shape=(1, self.output_dim))
        data = Data(inputs=inputs, outputs=outputs)
        data_stats = self.normalizer.compute_stats(data)
        keys = jr.split(key, self.num_particles)
        vmapped_params = vmap(self._init)(keys)
        calibration_alpha = jnp.ones(shape=(self.output_dim,))
        return RNNState(vmapped_params=vmapped_params, data_stats=data_stats, calibration_alpha=calibration_alpha)

    def _calibration_errors(self,
                            vmapped_params: PyTree,
                            inputs: chex.Array,
                            outputs: chex.Array,
                            ps: chex.Array,
                            data_stats: DataStats,
                            alpha: chex.Array) -> chex.Array:
        ps_hat = self._calculate_calibration_score(vmapped_params, inputs, outputs, ps, data_stats, alpha)
        ps = jnp.repeat(ps[..., jnp.newaxis], repeats=self.output_dim, axis=1)

        def loss(ps_est):
            return jnp.mean((ps - ps_est) ** 2, axis=0)

        return jnp.mean(vmap(loss, in_axes=1)(ps_hat), axis=0)

    def _calculate_calibration_score(self,
                                     vmapped_params: PyTree,
                                     inputs: chex.Array,
                                     outputs: chex.Array,
                                     ps: chex.Array,
                                     data_stats: DataStats,
                                     alpha: chex.Array) -> chex.Array:
        chex.assert_shape(alpha, (self.output_dim,))

        def calculate_score(x: chex.Array, y: chex.Array) -> chex.Array:
            apply_ensemble_one = vmap(self.apply_eval, in_axes=(0, None, None), out_axes=0)
            predicted_outputs, predicted_stds = apply_ensemble_one(vmapped_params, x, data_stats)
            means, epistemic_stds = predicted_outputs.mean(axis=0), predicted_outputs.std(axis=0)
            aleatoric_var = jnp.square(predicted_stds).mean(axis=0)
            if self.include_aleatoric_std_for_calibration:
                std = jnp.sqrt(epistemic_stds ** 2 + aleatoric_var) * alpha
            else:
                std = epistemic_stds * alpha
            cdfs = vmap(norm.cdf)(y, means, std)

            def check_cdf(cdf):
                chex.assert_shape(cdf, ())
                return cdf <= ps

            return vmap(vmap(check_cdf, out_axes=1), out_axes=1)(cdfs)

        cdfs = vmap(calculate_score)(inputs, outputs)
        return jnp.mean(cdfs, axis=0)

    def _prepare_data_for_training(self, data: Data) -> [DataStats, Data, Data]:

        flattened_data = Data(inputs=data.inputs.reshape(-1, self.input_dim),
                              outputs=data.outputs.reshape(-1, self.output_dim))
        data_stats = self.normalizer.compute_stats(flattened_data)
        num_points = data.inputs.shape[0]
        # Prepare data
        self.key, key = jr.split(self.key)
        permuted_data = jtu.tree_map(lambda x: jr.permutation(key, x), data)

        if self.train_share < 1.0:
            # Taking self.train_share number of points for training
            train_data = jtu.tree_map(lambda x: x[:int(self.train_share * num_points)], permuted_data)
            # Taking the rest for calibration
            eval_data = jtu.tree_map(lambda x: x[int(self.train_share * num_points):], permuted_data)
        else:
            train_data = permuted_data
            eval_data = permuted_data
        return data_stats, train_data, eval_data

    def fit_model(self, data: Data, num_epochs: int, model_state: RNNState) -> RNNState:
        bnn_state = super().fit_model(data, num_epochs, model_state)
        return RNNState(
            vmapped_params=bnn_state.vmapped_params,
            data_stats=bnn_state.data_stats,
            calibration_alpha=bnn_state.calibration_alpha,
        )

    @partial(jit, static_argnums=(0,))
    def posterior(self, input: chex.Array, rnn_state: RNNState) -> \
            Tuple[GRUParticleDistribution, GRUParticleDistribution]:
        """Computes the posterior distribution of the ensemble given the input and the data statistics."""
        assert input.shape[-1] == self.input_dim
        data_stats = rnn_state.data_stats
        # vmap over parameters and hidden state
        v_apply = vmap(self._apply, in_axes=(0, None, None, 0), out_axes=0)
        new_hidden_state, means, aleatoric_stds = v_apply(rnn_state.vmapped_params, input, rnn_state.data_stats,
                                                          rnn_state.hidden_state)
        ndim = input.ndim
        if ndim == 1:
            means = vmap(self.normalizer.denormalize, in_axes=(0, None))(means, data_stats.outputs)
            aleatoric_stds = vmap(self.normalizer.denormalize_std, in_axes=(0, None))(aleatoric_stds,
                                                                                      data_stats.outputs)
        else:
            means = vmap(vmap(self.normalizer.denormalize, in_axes=(0, None)), in_axes=(0, None)) \
                (means, data_stats.outputs)
            aleatoric_stds = vmap(vmap(self.normalizer.denormalize_std, in_axes=(0, None)), in_axes=(0, None)) \
                (aleatoric_stds, data_stats.outputs)

        # assert means.shape == aleatoric_stds.shape == (self.num_particles, self.output_dim)
        f_dist = GRUParticleDistribution(
            particle_hidden_states=new_hidden_state,
            particle_means=means,
            calibration_alpha=rnn_state.calibration_alpha
        )
        y_dist = GRUParticleDistribution(
            particle_hidden_states=new_hidden_state,
            particle_means=means,
            aleatoric_stds=aleatoric_stds,
            calibration_alpha=rnn_state.calibration_alpha)
        return f_dist, y_dist


class ProbabilisticGRUEnsemble(DeterministicGRUEnsemble):
    def __init__(self,
                 features: Sequence[int],
                 hidden_state_size: int,
                 num_cells: int,
                 sig_min: float = 1e-3,
                 sig_max: float = 1e3,
                 *args,
                 **kwargs,
                 ):
        super().__init__(features=features, hidden_state_size=hidden_state_size, num_cells=num_cells, *args, **kwargs)
        self.model = GRUModel(features=features, num_cells=num_cells,
                              output_dim=2 * self.output_dim, hidden_state_size=hidden_state_size)
        self.sig_min = sig_min
        self.sig_max = sig_max

    def _apply(self,
               params: PyTree,
               x: chex.Array,
               data_stats: DataStats,
               hidden_state: Optional[chex.Array] = None,
               ) -> [chex.Array, chex.Array, chex.Array]:
        ndim = x.ndim
        x = x.reshape(-1, self.input_dim)
        x = jax.vmap(self.normalizer.normalize, in_axes=(0, None))(x, data_stats.inputs)
        new_hidden_state, out = self.model.apply({'params': params}, x, hidden_state)
        out = out.reshape(2 * self.output_dim) if ndim == 1 else out
        mean, sig = jnp.split(out, 2, axis=-1)
        sig = nn.softplus(sig)
        sig = jnp.clip(sig, 0, self.sig_max) + self.sig_min
        assert mean.shape[-1] == self.output_dim and sig.shape[-1] == self.output_dim
        assert new_hidden_state.shape[-1] == self.hidden_size
        return new_hidden_state, mean, sig


if __name__ == '__main__':
    key = random.PRNGKey(0)
    log_training = False
    window_size = 10

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 256).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=1)
    weights = jnp.asarray([1.0, 0.5, 0.25, 0.15, 0.05])
    weights = jnp.repeat(weights[..., None], repeats=2, axis=-1)
    ys = jax.vmap(lambda x, y: jnp.convolve(y, x, 'same'), in_axes=(-1, -1))(ys, weights)
    ys = ys.transpose()
    ys = ys * (1 + noise_level * random.normal(key=random.PRNGKey(0), shape=ys.shape))
    input_dim = xs.shape[-1]
    output_dim = ys.shape[-1]
    x_train = create_windowed_array(xs, window_size=window_size)
    y_train = create_windowed_array(ys, window_size=window_size)
    data_std = noise_level * jnp.ones(shape=(output_dim,))

    data = Data(inputs=x_train, outputs=y_train)
    num_particles = 10
    model = ProbabilisticGRUEnsemble(input_dim=input_dim, output_dim=output_dim, features=[64, 64, 64],
                                     hidden_state_size=20,
                                     num_cells=1, num_particles=num_particles, output_stds=data_std,
                                     train_sequence_length=window_size,
                                     logging_wandb=log_training, eval_frequency=5, return_best_model=True)
    init_model_state = model.init(model.key)
    start_time = time.time()
    print('Starting with training')
    if log_training:
        wandb.init(
            project='Pendulum',
            group='test group',
        )

    model_params = model.fit_model(data, num_epochs=2000, model_state=init_model_state)
    print(f'Training time: {time.time() - start_time:.2f} seconds')

    test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1)
    test_ys = jax.vmap(lambda x, y: jnp.convolve(y, x, 'same'), in_axes=(-1, -1))(test_ys, weights)
    test_ys = test_ys.transpose()
    test_ys = test_ys * (1 + noise_level * random.normal(key=random.PRNGKey(0), shape=test_ys.shape))
    f_dist, y_dist = vmap(model.posterior, in_axes=(0, None))(test_xs, model_params)
    pred_mean = f_dist.mean()
    eps_std = f_dist.stddev()
    al_std = jnp.mean(y_dist.aleatoric_stds, axis=1)
    total_std = jnp.sqrt(jnp.square(eps_std) + jnp.square(al_std))

    for j in range(output_dim):
        plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        for i in range(num_particles):
            plt.plot(test_xs, f_dist.particle_means[:, i, j], label='NN prediction', color='black', alpha=0.3)
        plt.plot(test_xs, f_dist.mean()[..., j], label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (pred_mean[..., j] - 2 * total_std[..., j]).reshape(-1),
                         (pred_mean[..., j] + 2 * total_std[..., j]).reshape(-1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()

    for j in range(output_dim):
        for i in range(num_particles):
            plt.plot(test_xs, f_dist.particle_means[:, i, j], label='NN prediction', color='black', alpha=0.3)
        plt.plot(test_xs, f_dist.mean()[..., j], label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (pred_mean[..., j] - 2 * total_std[..., j]).reshape(-1),
                         (pred_mean[..., j] + 2 * total_std[..., j]).reshape(-1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()
