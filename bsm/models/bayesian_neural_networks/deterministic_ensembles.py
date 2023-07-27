import time
from typing import Sequence

import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import random, vmap
from jaxtyping import PyTree

import wandb
from bsm.models.bayesian_neural_networks.bnn import BayesianNeuralNet
from bsm.utils.normalization import Normalizer, DataStats, Data


class DeterministicEnsemble(BayesianNeuralNet):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 features: Sequence[int],
                 num_particles: int,
                 output_stds: chex.Array,
                 weight_decay: float = 1.0,
                 lr_rate: optax.Schedule | float = optax.constant_schedule(1e-3),
                 num_calibration_ps: int = 10,
                 num_test_alphas: int = 100,
                 batch_size: int = 64,
                 seed: int = 0,
                 logging_wandb: bool = True, ):
        super().__init__(input_dim=input_dim, output_dim=output_dim, features=features,
                         num_particles=num_particles, seed=seed, weight_decay=weight_decay, lr_rate=lr_rate,
                         batch_size=batch_size, num_calibration_ps=num_calibration_ps, num_test_alphas=num_test_alphas,
                         logging_wandb=logging_wandb)
        assert output_stds.shape == (output_dim,)
        self.output_stds = output_stds

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


if __name__ == '__main__':
    key = random.PRNGKey(0)
    logging_wandb = False
    input_dim = 1
    output_dim = 2

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 256).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=1)
    ys = ys + noise_level * random.normal(key=random.PRNGKey(0), shape=ys.shape)
    data_std = noise_level * jnp.ones(shape=(output_dim,))

    normalizer = Normalizer()
    data = Data(inputs=xs, outputs=ys)
    data_stats = normalizer.compute_stats(data)

    num_particles = 10
    model = DeterministicEnsemble(input_dim=input_dim, output_dim=output_dim, features=[64, 64, 64],
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

    test_ys_noisy = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1) + noise_level * random.normal(
        key=random.PRNGKey(0), shape=test_ys.shape)

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
