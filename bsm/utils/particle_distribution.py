from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
from distrax import Distribution, Normal


class ParticleDistribution(Distribution):
    def __init__(self,
                 particles: chex.Array,
                 aleatoric_stds: chex.Array | None = None,
                 calibration_alpha: chex.Array | None = None):
        self._particles = particles
        assert self._particles.ndim == 2
        self._num_particles, self._dim = self._particles.shape

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
        return jnp.mean(self._particles, axis=-2)

    def stddev(self) -> chex.Array:
        # Total std is sqrt of variance of particles and mean of aleatoric stds
        total_std = jnp.sqrt(jnp.std(self._particles, axis=-2) ** 2 + jnp.mean(self._aleatoric_stds, axis=-2) ** 2)
        return total_std * self._calibration_alpha

    def median(self) -> chex.Array:
        return jnp.median(self._particles, axis=-2)

    def log_prob(self, value: chex.Array) -> chex.Array:
        return self._normal_approx.log_prob(value)

    def event_shape(self) -> Tuple[int, ...]:
        particles_shape = list(self._particles.shape)
        del particles_shape[-2]
        return tuple(particles_shape)

    def sample_particle(self, seed: chex.PRNGKey) -> chex.Array:
        key_idx, key_noise = jr.split(seed)
        particle_idx = jr.randint(key_idx, shape=(), minval=0, maxval=self._num_particles)
        f_sample = self._particles[..., particle_idx, :]
        noise = jr.normal(key_noise, shape=f_sample.shape) * self._aleatoric_stds[..., particle_idx, :]
        return f_sample + noise

    def particles(self) -> chex.Array:
        return self._particles

    def aleatoric_stds(self) -> chex.Array:
        return self._aleatoric_stds


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
