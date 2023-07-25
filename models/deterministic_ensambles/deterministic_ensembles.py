import time
from collections import OrderedDict
from functools import partial
from typing import Sequence, Dict

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import random, vmap, jit
from jax.scipy.stats import norm
from jaxtyping import PyTree

import wandb
from utils.mlp import MLP
from utils.normalization import Normalizer, DataStats


class DeterministicEnsemble:
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 features: Sequence[int],
                 num_particles: int,
                 normalizer: Normalizer,
                 stds: chex.Array,
                 weight_decay: float = 1.0,
                 lr_rate: optax.Schedule | float = optax.constant_schedule(1e-3),
                 num_calibration_ps: int = 10,
                 num_test_alphas: int = 100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_particles = num_particles
        assert stds.shape == (output_dim,)
        self.stds = stds
        self.model = MLP(features=features, output_dim=self.output_dim)
        self.key = random.PRNGKey(0)
        self.tx = optax.adamw(learning_rate=lr_rate, weight_decay=weight_decay)
        self.normalizer = normalizer
        self.num_calibration_ps = num_calibration_ps
        self.num_test_alphas = num_test_alphas

    def _apply_train(self,
                     params: PyTree,
                     x: chex.Array,
                     data_stats: DataStats) -> [chex.Array, chex.Array]:
        chex.assert_shape(x, (self.input_dim,))
        x = self.normalizer.normalize(x, data_stats.inputs)
        return self.model.apply({'params': params}, x), self.normalizer.normalize_std(self.stds,
                                                                                      data_stats.outputs)

    def apply_eval(self,
                   params: PyTree,
                   x: chex.Array,
                   data_stats: DataStats) -> [chex.Array, chex.Array]:
        chex.assert_shape(x, (self.input_dim,))
        x = self.normalizer.normalize(x, data_stats.inputs)
        out = self.model.apply({'params': params}, x)
        return self.normalizer.denormalize(out, data_stats.outputs), self.stds

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
             target_outputs: chex.Array,
             data_stats: DataStats) -> [jax.Array, Dict]:

        # combine the training data batch with a batch of sampled measurement points
        # likelihood
        apply_ensemble_one = vmap(self._apply_train, in_axes=(0, None, None), out_axes=0)
        apply_ensemble = vmap(apply_ensemble_one, in_axes=(None, 0, None), out_axes=1, axis_name='batch')
        predicted_outputs, predicted_stds = apply_ensemble(vmapped_params, inputs, data_stats)

        target_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(target_outputs,
                                                                                 data_stats.outputs)
        negative_log_likelihood = self._neg_log_posterior(predicted_outputs, predicted_stds, target_outputs_norm)
        mse = jnp.mean((predicted_outputs - target_outputs[None, ...]) ** 2)
        return negative_log_likelihood, mse

    @partial(jit, static_argnums=0)
    def eval_ll(self,
                vmapped_params: chex.Array,
                inputs: chex.Array,
                target_outputs: chex.Array,
                data_stats: DataStats) -> chex.Array:
        apply_ensemble_one = vmap(self.apply_eval, in_axes=(0, None, None), out_axes=0)
        apply_ensemble = vmap(apply_ensemble_one, in_axes=(None, 0, None), out_axes=1, axis_name='batch')
        predicted_targets, predicted_stds = apply_ensemble(vmapped_params, inputs, data_stats)
        nll = self._nll(predicted_targets, predicted_stds, target_outputs)
        mse = jnp.mean((predicted_targets - target_outputs) ** 2)
        statistics = OrderedDict(nll=nll, mse=mse)
        return statistics

    @partial(jit, static_argnums=0)
    def _step_jit(self,
                  opt_state: optax.OptState,
                  vmapped_params: chex.PRNGKey,
                  inputs: chex.Array,
                  target_outputs: chex.Array,
                  data_stats: DataStats) -> (optax.OptState, PyTree, OrderedDict):
        (loss, mse), grads = jax.value_and_grad(self.loss, has_aux=True)(vmapped_params, inputs, target_outputs,
                                                                         data_stats)
        updates, opt_state = self.tx.update(grads, opt_state, vmapped_params)
        vmapped_params = optax.apply_updates(vmapped_params, updates)
        statiscs = OrderedDict(nll=loss, mse=mse)
        return opt_state, vmapped_params, statiscs

    def init_params(self, key):
        variables = self.model.init(key, jnp.ones(shape=(self.input_dim,)))
        if 'params' in variables:
            stats, params = variables.pop('params')
        else:
            stats, params = variables
        del variables  # Delete variables to avoid wasting resources
        return params

    def fit_model(self,
                  inputs: chex.Array,
                  target_outputs: chex.Array,
                  num_epochs: int,
                  data_stats: DataStats,
                  batch_size):
        self.key, key, *subkeys = random.split(self.key, self.num_particles + 2)
        vmapped_params = vmap(self.init_params)(jnp.stack(subkeys))
        opt_state = self.tx.init(vmapped_params)

        train_loader = self._create_data_loader((inputs, target_outputs), batch_size=batch_size)

        for step, (inputs_batch, target_outputs_batch) in enumerate(train_loader, 1):
            key, subkey = random.split(key)
            opt_state, vmapped_params, statistics = self._step_jit(opt_state, vmapped_params, inputs_batch,
                                                                   target_outputs_batch, data_stats)

            wandb.log(statistics)
            if step >= num_epochs:
                break

            if step % 100 == 0 or step == 1:
                statistics = self.eval_ll(vmapped_params, inputs_batch, target_outputs_batch, data_stats)
                print(f"Step {step}: {statistics}")

        return vmapped_params

    def _create_data_loader(self, vec_data: PyTree,
                            batch_size: int = 64, shuffle: bool = True,
                            infinite: bool = True) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(vec_data)
        if shuffle:
            seed = int(jax.random.randint(self.key, (1,), 0, 10 ** 8))
            ds = ds.shuffle(batch_size * 4, seed=seed, reshuffle_each_iteration=True)
        if infinite:
            ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = tfds.as_numpy(ds)
        return ds

    def calibration(self,
                    vmapped_params: PyTree,
                    inputs: chex.Array,
                    target_outputs: chex.Array,
                    data_stats: DataStats) -> chex.Array:
        ps = jnp.linspace(0, 1, self.num_calibration_ps + 1)[1:]
        return self._calculate_calibration_alpha(vmapped_params, inputs, target_outputs, ps, data_stats)

    def _calculate_calibration_alpha(self,
                                     vmapped_params: PyTree,
                                     inputs: chex.Array,
                                     target_outputs: chex.Array,
                                     ps: chex.Array,
                                     data_stats: DataStats) -> chex.Array:
        # We flip so that we rather take more uncertainty model than less
        test_alpha = jnp.flip(jnp.linspace(0, 10, self.num_test_alphas)[1:])
        test_alphas = jnp.repeat(test_alpha[..., jnp.newaxis], repeats=self.output_dim, axis=1)
        errors = vmap(self._calibration_errors, in_axes=(None, None, None, None, None, 0))(
            vmapped_params, inputs, target_outputs, ps, data_stats, test_alphas)
        indices = jnp.argmin(errors, axis=0)
        best_alpha = test_alpha[indices]
        chex.assert_shape(best_alpha, (self.output_dim,))
        return best_alpha

    def _calibration_errors(self,
                            vmapped_params: PyTree,
                            inputs: chex.Array,
                            target_outputs: chex.Array,
                            ps: chex.Array,
                            data_stats: DataStats,
                            alpha: chex.Array) -> chex.Array:
        ps_hat = self._calculate_calibration_score(vmapped_params, inputs, target_outputs, ps, data_stats, alpha)
        ps = jnp.repeat(ps[..., jnp.newaxis], repeats=self.output_dim, axis=1)
        return jnp.mean((ps - ps_hat) ** 2, axis=0)

    def _calculate_calibration_score(self,
                                     vmapped_params: PyTree,
                                     inputs: chex.Array,
                                     target_outputs: chex.Array,
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

        cdfs = vmap(calculate_score)(inputs, target_outputs)
        return jnp.mean(cdfs, axis=0)


if __name__ == '__main__':
    key = random.PRNGKey(0)
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
                                  num_particles=num_particles, normalizer=normalizer, stds=data_std)
    start_time = time.time()
    print('Starting with training')
    wandb.init(
        project='Pendulum',
        group='test group',
    )

    model_params = model.fit_model(inputs=xs, target_outputs=ys, num_epochs=1000, data_stats=data_stats,
                                   batch_size=32)
    print(f'Training time: {time.time() - start_time:.2f} seconds')

    test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1)

    test_ys_noisy = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1) + noise_level * random.normal(
        key=random.PRNGKey(0), shape=test_ys.shape)

    test_stds = noise_level * jnp.ones(shape=test_ys.shape)

    alpha_best = model.calibration(model_params, test_xs, test_ys_noisy, data_stats)
    apply_ens = vmap(model.apply_eval, in_axes=(None, 0, None))
    preds, aleatoric_stds = vmap(apply_ens, in_axes=(0, None, None))(model_params, test_xs, data_stats)

    for j in range(output_dim):
        plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        for i in range(num_particles):
            plt.plot(test_xs, preds[i, :, j], label='NN prediction', color='black', alpha=0.3)
        plt.plot(test_xs, jnp.mean(preds[..., j], axis=0), label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (jnp.mean(preds[..., j], axis=0) - 2 * jnp.std(preds[..., j], axis=0)).reshape(-1),
                         (jnp.mean(preds[..., j], axis=0) + 2 * jnp.std(preds[..., j], axis=0)).reshape(-1),
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
                         (jnp.mean(preds[..., j], axis=0) - 2 * jnp.std(preds[..., j], axis=0)).reshape(-1),
                         (jnp.mean(preds[..., j], axis=0) + 2 * jnp.std(preds[..., j], axis=0)).reshape(-1),
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
                         (jnp.mean(preds[..., j], axis=0) - 2 * alpha_best[j] * jnp.std(preds[..., j], axis=0)).reshape(
                             -1),
                         (jnp.mean(preds[..., j], axis=0) + 2 * alpha_best[j] * jnp.std(preds[..., j], axis=0)).reshape(
                             -1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()
