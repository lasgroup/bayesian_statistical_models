import time
from collections import OrderedDict
from functools import partial
from typing import Sequence, Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap, jit
from jax.scipy.stats import norm
from jaxtyping import PyTree
import jax.random as jr
import jax.tree_util as jtu
from brax.training.replay_buffers import UniformSamplingQueue
from bsm.models.bayesian_neural_networks.bnn import BNNState

import wandb
from bsm.utils.network_utils import GRUModel
from bsm.utils.normalization import Normalizer, DataStats, Data
from bsm.models.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
import flax.linen as nn
from bsm.utils.particle_distribution import GRUParticleDistribution


def create_windowed_array(arr: jnp.array, window_size: int = 10) -> jnp.array:
    """Sliding window over an array along the first axis."""
    arr_strided = jnp.stack([arr[i:(-window_size + i)] for i in range(window_size)], axis=-2)
    assert arr_strided.shape == (arr.shape[0] - window_size, window_size, arr.shape[-1])
    return jnp.array(arr_strided)


@chex.dataclass
class RNNState(BNNState):
    hidden_state: Optional[chex.Array] = None


class DeterministicGRUEnsemble(DeterministicEnsemble):
    def __init__(self,
                 features: Sequence[int],
                 hidden_state_size: int,
                 num_cells: int,
                 *args,
                 **kwargs,
                 ):
        super().__init__(features=features, *args, **kwargs)
        self.hidden_state_size = hidden_state_size
        self.model = GRUModel(features=features, num_cells=num_cells,
                              output_dim=self.output_dim, hidden_state_size=hidden_state_size)
        self.normalizer = Normalizer()
        self.hidden_size = self.hidden_state_size * num_cells

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
        negative_log_likelihood = self._neg_log_posterior(predicted_outputs, predicted_stds, target_outputs_norm)
        mse = jnp.mean((predicted_outputs - target_outputs_norm[None, ...]) ** 2)
        return negative_log_likelihood, mse

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
            stats, params = variables.pop('params')
        else:
            stats, params = variables
        del variables  # Delete variables to avoid wasting resources
        return params

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
            std = jnp.sqrt((epistemic_stds * alpha) ** 2 + aleatoric_var)
            cdfs = vmap(vmap(norm.cdf))(y, means, std)

            def check_cdf(cdf):
                chex.assert_shape(cdf, ())
                return cdf <= ps

            return vmap(vmap(check_cdf, out_axes=1), out_axes=1)(cdfs)

        cdfs = vmap(calculate_score)(inputs, outputs)
        return jnp.mean(cdfs, axis=0)

    def fit_model(self, data: Data, num_epochs: int) -> RNNState:
        self.key, key = jr.split(self.key)
        vmapped_params = self.init(key)
        opt_state = self.tx.init(vmapped_params)
        flattened_data = Data(inputs=data.inputs.reshape(-1, self.input_dim),
                              outputs=data.outputs.reshape(-1, self.output_dim))
        data_stats = self.normalizer.compute_stats(flattened_data)

        num_points = data.inputs.shape[0]
        dummy_data_sample = jtu.tree_map(lambda x: x[0], data)
        buffer = UniformSamplingQueue(max_replay_size=num_points, dummy_data_sample=dummy_data_sample,
                                      sample_batch_size=self.batch_size)

        self.key, key = jr.split(self.key)
        buffer_state = buffer.init(key)
        buffer_state = buffer.insert(buffer_state, data)

        def f(carry, _):
            opt_state, vmapped_params, buffer_state = carry
            new_buffer_state, data_batch = buffer.sample(buffer_state)
            opt_state, vmapped_params, statistics = self.step_jit(opt_state, vmapped_params, data_batch.inputs,
                                                                  data_batch.outputs, data_stats)
            return (opt_state, vmapped_params, new_buffer_state), statistics

        init_carry = (opt_state, vmapped_params, buffer_state)
        (opt_state, vmapped_params, buffer_state), statistics = jax.lax.scan(f, init_carry, None, length=num_epochs)
        if self.logging_wandb:
            for i in range(num_epochs):
                wandb.log(jtu.tree_map(lambda x: x[i], statistics))
        calibrate_alpha = self.calibrate(vmapped_params, data.inputs, data.outputs, data_stats)
        model_state = RNNState(data_stats=data_stats, vmapped_params=vmapped_params,
                               calibration_alpha=calibrate_alpha)
        return model_state

    @partial(jit, static_argnums=(0,))
    def posterior(self, input: chex.Array, rnn_state: RNNState) -> \
            Tuple[GRUParticleDistribution, GRUParticleDistribution]:
        """Computes the posterior distribution of the ensemble given the input and the data statistics."""
        assert input.shape[-1] == self.input_dim
        data_stats = rnn_state.data_stats
        v_apply = vmap(self._apply, in_axes=(0, None, None, None), out_axes=0)
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

    normalizer = Normalizer()
    data_stats = normalizer.compute_stats(Data(inputs=xs, outputs=ys))
    data = Data(inputs=x_train, outputs=y_train)
    num_particles = 10
    model = ProbabilisticGRUEnsemble(input_dim=input_dim, output_dim=output_dim, features=[64, 64, 64],
                                     hidden_state_size=20,
                                     num_cells=1, num_particles=num_particles, output_stds=data_std,
                                     logging_wandb=log_training)
    start_time = time.time()
    print('Starting with training')
    if log_training:
        wandb.init(
            project='Pendulum',
            group='test group',
        )

    model_params = model.fit_model(data, num_epochs=2000)
    print(f'Training time: {time.time() - start_time:.2f} seconds')

    test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1)
    test_ys = jax.vmap(lambda x, y: jnp.convolve(y, x, 'same'), in_axes=(-1, -1))(test_ys, weights)
    test_ys = test_ys.transpose()
    test_ys = test_ys * (1 + noise_level * random.normal(key=random.PRNGKey(0), shape=test_ys.shape))
    x_test = create_windowed_array(test_xs, window_size=window_size)
    y_test = create_windowed_array(test_ys, window_size=window_size)
    alpha_best = model.calibrate(model_params.vmapped_params, x_test, y_test, data_stats)
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
