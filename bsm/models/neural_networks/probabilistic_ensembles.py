import time
from typing import Sequence

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from jaxtyping import PyTree
import flax.linen as nn

import wandb
from bsm.utils.mlp import MLP
from bsm.utils.normalization import Normalizer, DataStats
from bsm.models.neural_networks.deterministic_ensembles import DeterministicEnsemble, fit_model


class ProbabilisticEnsemble(DeterministicEnsemble):
    def __init__(self, features: Sequence[int], sig_min: float = 1e-3, sig_max: float = 1e3, *args, **kwargs):
        super().__init__(features=features, *args, **kwargs)
        self.model = MLP(features=features, output_dim=2 * self.output_dim)
        self.sig_min = sig_min
        self.sig_max = sig_max

    def _apply_train(self,
                     params: PyTree,
                     x: chex.Array,
                     data_stats: DataStats) -> [chex.Array, chex.Array]:
        chex.assert_shape(x, (self.input_dim,))
        x = self.normalizer.normalize(x, data_stats.inputs)
        out = self.model.apply({'params': params}, x)
        mu, sig = jnp.split(out, 2, axis=-1)
        sig = nn.softplus(sig)
        sig = jnp.clip(sig, 0, self.sig_max) + self.sig_min
        return mu, sig

    def apply_eval(self,
                   params: PyTree,
                   x: chex.Array,
                   data_stats: DataStats) -> [chex.Array, chex.Array]:
        chex.assert_shape(x, (self.input_dim,))
        x = self.normalizer.normalize(x, data_stats.inputs)
        out = self.model.apply({'params': params}, x)
        mu, sig = jnp.split(out, 2, axis=-1)
        sig = nn.softplus(sig)
        sig = jnp.clip(sig, 0, self.sig_max) + self.sig_min
        mean = self.normalizer.denormalize(mu, data_stats.outputs)
        std = self.normalizer.denormalize_std(sig, data_stats.outputs)
        return mean, std


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
    model = ProbabilisticEnsemble(input_dim=input_dim, output_dim=output_dim, features=[64, 64, 64],
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

    test_ys_noisy = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1) * (1 + noise_level * random.normal(
        key=random.PRNGKey(0), shape=test_ys.shape))

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
