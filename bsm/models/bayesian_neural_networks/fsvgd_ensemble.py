import time
from typing import Dict

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from jaxtyping import PyTree

from bsm.models.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
from bsm.models.bayesian_neural_networks.probabilistic_ensembles import ProbabilisticEnsemble
from bsm.utils.normalization import DataStats, Data
import wandb


def prepare_stein_kernel(h=0.2 ** 2):
    def k(x, y):
        return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * h))

    v_k = vmap(k, in_axes=(0, None), out_axes=0)
    m_k = vmap(v_k, in_axes=(None, 0), out_axes=1)

    def kernel(fs):
        kernel_matrix = m_k(fs, fs)
        return kernel_matrix

    k_x = jax.grad(k, argnums=0)

    v_k_der = vmap(k_x, in_axes=(0, None), out_axes=0)
    m_k_der = vmap(v_k_der, in_axes=(None, 0), out_axes=1)

    def kernel_derivative(fs):
        return m_k_der(fs, fs)

    return kernel, kernel_derivative


class DeterministicFSVGDEnsemble(DeterministicEnsemble):
    def __init__(self, prior_h: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_h = prior_h
        self.stein_kernel, self.stein_kernel_derivative = prepare_stein_kernel()

    def loss(self,
             vmapped_params: PyTree,
             inputs: chex.Array,
             outputs: chex.Array,
             data_stats: DataStats) -> [jax.Array, Dict]:
        apply_ensemble_one = vmap(self._apply_train, in_axes=(0, None, None), out_axes=0)
        apply_ensemble = vmap(apply_ensemble_one, in_axes=(None, 0, None), out_axes=1, axis_name='batch')
        predicted_mean, predicted_stds = apply_ensemble(vmapped_params, inputs, data_stats)
        target_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(outputs, data_stats.outputs)

        negative_log_likelihood, grad_post = jax.value_and_grad(self._neg_log_posterior)(predicted_mean, predicted_stds,
                                                                                         target_outputs_norm)
        mse = jnp.mean((predicted_mean - target_outputs_norm[None, ...]) ** 2)

        # kernel
        k = self.stein_kernel(predicted_mean)
        k_x = self.stein_kernel_derivative(predicted_mean)
        grad_k = jnp.mean(k_x, axis=0)

        surrogate_loss = jnp.sum(predicted_mean * jax.lax.stop_gradient(
            jnp.einsum('ij,jkm', k, grad_post) - grad_k))
        return surrogate_loss, mse


class ProbabilisticFSVGDEnsemble(ProbabilisticEnsemble):
    def __init__(self, prior_h: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_h = prior_h
        self.stein_kernel, self.stein_kernel_derivative = prepare_stein_kernel()

    def loss(self,
             vmapped_params: PyTree,
             inputs: chex.Array,
             outputs: chex.Array,
             data_stats: DataStats) -> [jax.Array, Dict]:
        # likelihood
        def apply_fn(params: PyTree, x: jax.Array) -> jax.Array:
            x = self.normalizer.normalize(x, data_stats.inputs)
            out = self.model.apply({'params': params}, x)
            return out

        apply_ensemble_one = vmap(apply_fn, in_axes=(0, None), out_axes=0)
        apply_ensemble = vmap(apply_ensemble_one, in_axes=(None, 0), out_axes=1, axis_name='batch')
        f_raw = apply_ensemble(vmapped_params, inputs)

        target_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(outputs, data_stats.outputs)

        def neg_log_likelihood(predictions, output):
            mu, sig = jnp.split(predictions, 2, axis=-1)
            sig = nn.softplus(sig)
            sig = jnp.clip(sig, 0, self.sig_max) + self.sig_min
            return self._neg_log_posterior(mu, sig, output), mu

        (negative_log_likelihood, mu), grad_post = jax.value_and_grad(neg_log_likelihood, has_aux=True) \
            (f_raw, target_outputs_norm)
        mse = jnp.mean((mu - target_outputs_norm[None, ...]) ** 2)

        # kernel
        k = self.stein_kernel(f_raw)
        k_x = self.stein_kernel_derivative(f_raw)
        grad_k = jnp.mean(k_x, axis=0)

        surrogate_loss = jnp.sum(f_raw * jax.lax.stop_gradient(
            jnp.einsum('ij,jkm', k, grad_post) - grad_k))
        return surrogate_loss, mse


if __name__ == '__main__':
    key = random.PRNGKey(0)
    logging_wandb = False
    input_dim = 1
    output_dim = 2

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 256).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=1)
    ys = ys * (1 + noise_level * random.normal(key=random.PRNGKey(0), shape=ys.shape))
    data_std = noise_level * jnp.ones(shape=(output_dim,))

    data = Data(inputs=xs, outputs=ys)

    num_particles = 10
    model = ProbabilisticFSVGDEnsemble(input_dim=input_dim, output_dim=output_dim, features=[64, 64, 64],
                                       num_particles=num_particles, output_stds=data_std, logging_wandb=logging_wandb)

    start_time = time.time()
    print('Starting with training')
    if logging_wandb:
        wandb.init(
            project='Pendulum',
            group='test group',
        )

    model_state = model.fit_model(data=data, num_epochs=1000)
    print(f'Training time: {time.time() - start_time:.2f} seconds')

    test_xs = jnp.linspace(-3, 13, 1000).reshape(-1, 1)
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1)

    test_ys_noisy = test_ys * (1 + noise_level * random.normal(
        key=random.PRNGKey(0), shape=test_ys.shape))

    test_stds = noise_level * jnp.ones(shape=test_ys.shape)

    f_dist, y_dist = vmap(model.posterior, in_axes=(0, None))(test_xs, model_state)

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
