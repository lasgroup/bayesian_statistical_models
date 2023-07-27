from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
from distrax import Distribution, Normal


class ParticleDistribution(Distribution):
    def __init__(self,
                 particle_means: chex.Array,
                 aleatoric_stds: chex.Array | None = None,
                 calibration_alpha: chex.Array | None = None):
        self._particle_means = particle_means
        assert self._particle_means.ndim == 2
        self._num_particles, self._dim = self._particle_means.shape

        if calibration_alpha is None:
            calibration_alpha = jnp.ones(shape=(self._dim,))
        self._calibration_alpha = calibration_alpha

        if aleatoric_stds is None:
            aleatoric_stds = jnp.zeros(shape=(self._num_particles, self._dim))

        self._aleatoric_stds = aleatoric_stds
        self._normal_approx = Normal(loc=self.mean(), scale=self.stddev())

    def _sample_n(self, key: chex.PRNGKey, n: int) -> chex.Array:
        """Sample n times from the distribution.
            We sample f from the normal approximation and then add average aleatoric noise to it.
        """
        key_f, key_noise = jr.split(key)
        samples = self._normal_approx._sample_n(key=key_f, n=n)
        assert samples.shape[-1] == self._dim and samples.shape[0] == n
        return samples

    def mean(self) -> chex.Array:
        return jnp.mean(self._particle_means, axis=-2)

    def stddev(self) -> chex.Array:
        # Total std is sqrt of variance of particles and mean of aleatoric stds
        eps_var = (jnp.std(self._particle_means, axis=-2) * self._calibration_alpha) ** 2
        ale_var = jnp.mean(self._aleatoric_stds ** 2, axis=-2)
        total_std = jnp.sqrt(eps_var + ale_var)
        return total_std

    def median(self) -> chex.Array:
        return jnp.median(self._particle_means, axis=-2)

    def log_prob(self, value: chex.Array) -> chex.Array:
        return self._normal_approx.log_prob(value)

    def event_shape(self) -> Tuple[int, ...]:
        particles_shape = list(self._particle_means.shape)
        del particles_shape[-2]
        return tuple(particles_shape)

    def sample_particle(self, seed: chex.PRNGKey) -> chex.Array:
        key_idx, key_noise = jr.split(seed)
        particle_idx = jr.randint(key_idx, shape=(), minval=0, maxval=self._num_particles)
        f_sample = self._particle_means[..., particle_idx, :]
        noise = jr.normal(key_noise, shape=f_sample.shape) * self._aleatoric_stds[..., particle_idx, :]
        return f_sample + noise

    @property
    def particle_means(self) -> chex.Array:
        return self._particle_means

    @property
    def aleatoric_stds(self) -> chex.Array:
        return self._aleatoric_stds


class GRUParticleDistribution(ParticleDistribution):

    def __init__(self, particle_hidden_states: chex.Array, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._particle_hidden_states = particle_hidden_states

    @property
    def particle_hidden_states(self) -> chex.Array:
        return self._particle_hidden_states


if __name__ == '__main__':
    particles = jnp.array([1.0, 2.0, 3.0]).reshape(3, 1)
    pd = ParticleDistribution(particles)
    key = jr.PRNGKey(0)

    print('Sample: ', pd.sample(seed=key, sample_shape=(4,)))
    print('Mean: ', pd.mean())
    print('Std: ', pd.stddev())
    print('Median: ', pd.median())
    print('Log_prob: ', pd.log_prob(jnp.array([2.0])))
    print('Event shape: ', pd.event_shape())
    print('Sample Particle: ', pd.sample_particle(seed=key))
