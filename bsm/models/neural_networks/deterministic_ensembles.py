import time
from collections import OrderedDict
from functools import partial
from typing import Sequence, Dict

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import random, vmap, jit
from jax.scipy.stats import norm
from jaxtyping import PyTree


import wandb
from bsm.utils.mlp import MLP
from bsm.utils.normalization import Normalizer, DataStats


class DeterministicEnsemble:
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 features: Sequence[int],
                 num_particles: int,
                 output_stds: chex.Array,
                 weight_decay: float = 1.0,
                 lr_rate: optax.Schedule | float = optax.constant_schedule(1e-3),
                 num_calibration_ps: int = 10,
                 num_test_alphas: int = 100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_particles = num_particles
        assert output_stds.shape == (output_dim,)
        self.output_stds = output_stds
        self.model = MLP(features=features, output_dim=self.output_dim)
        self.tx = optax.adamw(learning_rate=lr_rate, weight_decay=weight_decay)
        self.normalizer = Normalizer()
        self.num_calibration_ps = num_calibration_ps
        self.num_test_alphas = num_test_alphas

    def _apply_train(self,
                     params: PyTree,
                     x: chex.Array,
                     data_stats: DataStats) -> [chex.Array, chex.Array]:
        chex.assert_shape(x, (self.input_dim,))
        x = self.normalizer.normalize(x, data_stats.inputs)
        return self.model.apply({'params': params}, x), self.normalizer.normalize_std(self.output_stds,
                                                                                      data_stats.outputs)

    def apply_eval(self,
                   params: PyTree,
                   x: chex.Array,
                   data_stats: DataStats) -> [chex.Array, chex.Array]:
        chex.assert_shape(x, (self.input_dim,))
        x = self.normalizer.normalize(x, data_stats.inputs)
        out = self.model.apply({'params': params}, x)
        return self.normalizer.denormalize(out, data_stats.outputs), self.output_stds

    def _nll(self,
             predicted_outputs: chex.Array,
             predicted_stds: chex.Array,
             target_outputs: chex.Array) -> chex.Array:
        chex.assert_equal_shape([target_outputs, predicted_stds[0, ...], predicted_outputs[0, ...]])
        log_prob = norm.logpdf(target_outputs[jnp.newaxis, :], loc=predicted_outputs, scale=predicted_stds)
        return - jnp.mean(log_prob)

    def _neg_log_posterior(self,
                           predicted_outputs: chex.Array,
                           predicted_stds: chex.Array,
                           target_outputs: jax.Array) -> chex.Array:
        nll = self._nll(predicted_outputs, predicted_stds, target_outputs)
        neg_log_post = nll
        return neg_log_post

    def loss(self,
             vmapped_params: PyTree,
             inputs: chex.Array,
             outputs: chex.Array,
             data_stats: DataStats) -> [jax.Array, Dict]:

        # combine the training data batch with a batch of sampled measurement points
        # likelihood
        apply_ensemble_one = vmap(self._apply_train, in_axes=(0, None, None), out_axes=0)
        apply_ensemble = vmap(apply_ensemble_one, in_axes=(None, 0, None), out_axes=1, axis_name='batch')
        predicted_outputs, predicted_stds = apply_ensemble(vmapped_params, inputs, data_stats)

        target_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(outputs, data_stats.outputs)
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
        apply_ensemble = vmap(apply_ensemble_one, in_axes=(None, 0, None), out_axes=1, axis_name='batch')
        predicted_targets, predicted_stds = apply_ensemble(vmapped_params, inputs, data_stats)
        nll = self._nll(predicted_targets, predicted_stds, outputs)
        mse = jnp.mean((predicted_targets - outputs) ** 2)
        statistics = OrderedDict(nll=nll, mse=mse)
        return statistics

    @partial(jit, static_argnums=0)
    def step_jit(self,
                 opt_state: optax.OptState,
                 vmapped_params: chex.PRNGKey,
                 inputs: chex.Array,
                 outputs: chex.Array,
                 data_stats: DataStats) -> (optax.OptState, PyTree, OrderedDict):
        (loss, mse), grads = jax.value_and_grad(self.loss, has_aux=True)(vmapped_params, inputs, outputs, data_stats)
        updates, opt_state = self.tx.update(grads, opt_state, vmapped_params)
        vmapped_params = optax.apply_updates(vmapped_params, updates)
        statiscs = OrderedDict(nll=loss, mse=mse)
        return opt_state, vmapped_params, statiscs

    def _init(self, key):
        variables = self.model.init(key, jnp.ones(shape=(self.input_dim,)))
        if 'params' in variables:
            stats, params = variables.pop('params')
        else:
            stats, params = variables
        del variables  # Delete variables to avoid wasting resources
        return params

    def init(self, key):
        keys = random.split(key, self.num_particles)
        return vmap(self._init)(keys)

    def calibration(self,
                    vmapped_params: PyTree,
                    inputs: chex.Array,
                    outputs: chex.Array,
                    data_stats: DataStats) -> chex.Array:
        ps = jnp.linspace(0, 1, self.num_calibration_ps + 1)[1:]
        return self._calculate_calibration_alpha(vmapped_params, inputs, outputs, ps, data_stats)

    def _calculate_calibration_alpha(self,
                                     vmapped_params: PyTree,
                                     inputs: chex.Array,
                                     outputs: chex.Array,
                                     ps: chex.Array,
                                     data_stats: DataStats) -> chex.Array:
        # We flip so that we rather take more uncertainty model than less
        test_alpha = jnp.flip(jnp.linspace(0, 10, self.num_test_alphas)[1:])
        test_alphas = jnp.repeat(test_alpha[..., jnp.newaxis], repeats=self.output_dim, axis=1)
        errors = vmap(self._calibration_errors, in_axes=(None, None, None, None, None, 0))(
            vmapped_params, inputs, outputs, ps, data_stats, test_alphas)
        indices = jnp.argmin(errors, axis=0)
        best_alpha = test_alpha[indices]
        chex.assert_shape(best_alpha, (self.output_dim,))
        return best_alpha

    def _calibration_errors(self,
                            vmapped_params: PyTree,
                            inputs: chex.Array,
                            outputs: chex.Array,
                            ps: chex.Array,
                            data_stats: DataStats,
                            alpha: chex.Array) -> chex.Array:
        ps_hat = self._calculate_calibration_score(vmapped_params, inputs, outputs, ps, data_stats, alpha)
        ps = jnp.repeat(ps[..., jnp.newaxis], repeats=self.output_dim, axis=1)
        return jnp.mean((ps - ps_hat) ** 2, axis=0)

    def _calculate_calibration_score(self,
                                     vmapped_params: PyTree,
                                     inputs: chex.Array,
                                     outputs: chex.Array,
                                     ps: chex.Array,
                                     data_stats: DataStats,
                                     alpha: chex.Array) -> chex.Array:
        chex.assert_shape(alpha, (self.output_dim,))

        def calculate_score(x: chex.Array, y: chex.Array) -> chex.Array:
            assert x.shape == (self.input_dim,) and y.shape == (self.output_dim,)
            predicted_outputs, predicted_stds = vmap(self.apply_eval, in_axes=(0, None, None), out_axes=0)(
                vmapped_params, x, data_stats)
            means, epistemic_stds = predicted_outputs.mean(axis=0), predicted_outputs.std(axis=0)
            aleatoric_stds = predicted_stds.mean(axis=0)
            std = jnp.sqrt((epistemic_stds * alpha) ** 2 + aleatoric_stds ** 2)
            chex.assert_shape(std, (self.output_dim,))
            cdfs = vmap(norm.cdf)(y, means, std)

            def check_cdf(cdf):
                chex.assert_shape(cdf, ())
                return cdf <= ps

            return vmap(check_cdf, out_axes=1)(cdfs)

        cdfs = vmap(calculate_score)(inputs, outputs)
        return jnp.mean(cdfs, axis=0)


def dataset(key: jax.random.PRNGKey, x: jax.Array, y: jax.Array, batch_size: int):
    ids = jnp.arange(len(x))
    while True:
        sample_key, key = jax.random.split(key, 2)
        ids = jax.random.choice(sample_key, ids, shape=(batch_size,), replace=False)
        yield x[ids], y[ids]


def fit_model(model: DeterministicEnsemble,
              inputs: chex.Array,
              outputs: chex.Array,
              num_epochs: int,
              data_stats: DataStats,
              batch_size: int,
              key: jax.random.PRNGKey,
              log_training: bool = False):
    key, init_key = random.split(key)
    vmapped_params = model.init(key)
    opt_state = model.tx.init(vmapped_params)
    key, shuffle_key = random.split(key)
    data = iter(dataset(key, inputs, outputs, batch_size))
    num_train_steps = len(inputs) * num_epochs
    for step in range(num_train_steps):
        key, subkey = random.split(key)
        inputs_batch, target_outputs_batch = next(data)
        opt_state, vmapped_params, statistics = model.step_jit(opt_state, vmapped_params, inputs_batch,
                                                               target_outputs_batch, data_stats)
        if log_training:
            wandb.log(statistics)
        if step >= num_epochs:
            break

        if step % 100 == 0 or step == 1:
            statistics = model.eval_ll(vmapped_params, inputs_batch, target_outputs_batch, data_stats)
            print(f"Step {step}: {statistics}")

    return vmapped_params


if __name__ == '__main__':
    key = random.PRNGKey(0)
    log_training = False
    input_dim = 1
    output_dim = 2

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 256).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=1)
    ys = ys + noise_level * random.normal(key=random.PRNGKey(0), shape=ys.shape)
    data_std = noise_level * jnp.ones(shape=(output_dim,))

    normalizer = Normalizer()
    data = DataStats(inputs=xs, outputs=ys)
    data_stats = normalizer.compute_stats(data)

    num_particles = 10
    model = DeterministicEnsemble(input_dim=input_dim, output_dim=output_dim, features=[64, 64, 64],
                                  num_particles=num_particles, output_stds=data_std)
    start_time = time.time()
    print('Starting with training')
    if log_training:
        wandb.init(
            project='Pendulum',
            group='test group',
        )

    model_params = fit_model(model=model, inputs=xs, outputs=ys, num_epochs=1000, data_stats=data_stats,
                             batch_size=32, key=key, log_training=log_training)
    print(f'Training time: {time.time() - start_time:.2f} seconds')

    test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1)

    test_ys_noisy = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1) + noise_level * random.normal(
        key=random.PRNGKey(0), shape=test_ys.shape)

    test_stds = noise_level * jnp.ones(shape=test_ys.shape)

    alpha_best = model.calibration(model_params, test_xs, test_ys_noisy, data_stats)
    apply_ens = vmap(model.apply_eval, in_axes=(None, 0, None))
    preds, aleatoric_stds = vmap(apply_ens, in_axes=(0, None, None))(model_params, test_xs, data_stats)
    pred_mean = jnp.mean(preds, axis=0)
    eps_std = jnp.std(preds, axis=0)
    al_std = jnp.mean(aleatoric_stds, axis=0)
    total_std = jnp.sqrt(jnp.square(eps_std) + jnp.square(al_std))
    total_calibrated_std = jax.vmap(lambda x, y, z: jnp.sqrt(jnp.square(x * z) + jnp.square(y)), in_axes=(-1, -1, -1),
                                    out_axes=-1)(eps_std, al_std, alpha_best)
    for j in range(output_dim):
        plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        for i in range(num_particles):
            plt.plot(test_xs, preds[i, :, j], label='NN prediction', color='black', alpha=0.3)
        plt.plot(test_xs, jnp.mean(preds[..., j], axis=0), label='Mean', color='blue')
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
        # plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        for i in range(num_particles):
            plt.plot(test_xs, preds[i, :, j], label='NN prediction', color='black', alpha=0.3)
        plt.plot(test_xs, jnp.mean(preds[..., j], axis=0), label='Mean', color='blue')
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
            plt.plot(test_xs, preds[i, :, j], label='NN prediction', color='black', alpha=0.3)
        plt.plot(test_xs, jnp.mean(preds[..., j], axis=0), label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (pred_mean[..., j] - 2 * total_calibrated_std[..., j]).reshape(-1),
                         (pred_mean[..., j] + 2 * total_calibrated_std[..., j]).reshape(-1),
                         label=r'$2\sigma}$', alpha=0.3, color='yellow')
        plt.fill_between(test_xs.reshape(-1),
                         (pred_mean[..., j] - 2 * alpha_best[j] * eps_std[..., j]).reshape(-1),
                         (pred_mean[..., j] + 2 * alpha_best[j] * eps_std[..., j]).reshape(-1),
                         label=r'$2\sigma_{eps}$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()
