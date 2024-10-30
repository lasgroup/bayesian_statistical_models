from typing import Tuple

import cvxpy as cp
import jax.numpy as jnp
from cvxpy import Problem
from jaxtyping import Float, Array, Scalar


def alpha_minimize_norm(kernel_matrix: Float[Array, 'n_obs n_obs'],
                        sigma: Scalar,
                        alpha_mu: Float[Array, 'n_obs'],
                        beta: Scalar = 2) -> Tuple[Float[Array, 'n_obs'], Problem]:
    numerical_correction = 0.0
    n_obs = kernel_matrix.shape[0]
    alpha = cp.Variable(n_obs)
    alpha_diff = alpha - alpha_mu
    K = kernel_matrix
    I = jnp.eye(n_obs)

    constraints = [
        alpha_diff @ cp.psd_wrap(K + 1 / sigma ** 2 * K @ K.T + numerical_correction * I) @ alpha_diff <= 4 * beta ** 2]
    objective = cp.Minimize(alpha @ cp.psd_wrap(K + numerical_correction * I) @ alpha)
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    print(f'Minimal prior RKHS norm is {result ** 2}')
    return alpha.value, prob


def alpha_minimize_distance(kernel_matrix: Float[Array, 'n_obs n_obs'],
                            sigma: Scalar,
                            alpha_mu: Float[Array, 'n_obs'],
                            norm_bound: Scalar = 3) -> Tuple[Float[Array, 'n_obs'], Problem]:
    numerical_correction = 0.0
    n_obs = kernel_matrix.shape[0]
    alpha = cp.Variable(n_obs)
    alpha_diff = alpha - alpha_mu
    K = kernel_matrix

    I = jnp.eye(n_obs)

    constraints = [alpha @ cp.psd_wrap(K + numerical_correction * I) @ alpha <= norm_bound]
    objective = cp.Minimize(
        alpha_diff @ cp.psd_wrap(K + 1 / sigma ** 2 * K @ K.T + numerical_correction * I) @ alpha_diff)

    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    print(f'Minimal distance in the posterior kernel metric is {result ** 0.5}')
    return alpha.value, prob


if __name__ == '__main__':
    import jax.random as jr
    from jax import vmap
    import jax
    import matplotlib.pyplot as plt


    def k(x, y):
        return jnp.exp(-(x - y) ** 2 / 0.6 ** 2)


    sigma = 0.3
    num_obs = 50
    num_plot_points = 100

    xs = jnp.linspace(-5, 5, num_obs)
    ys = jnp.sin(xs) + sigma * jr.normal(key=jr.PRNGKey(0), shape=(num_obs,))
    K = vmap(vmap(k, in_axes=(0, None)), in_axes=(None, 0))(xs, xs)

    print(K.shape)
    K_noisy = (K + sigma ** 2 * jnp.eye(num_obs))


    def sigma_n(x):
        k_x_X = vmap(k, in_axes=(None, 0))(x, xs)
        cholesky_tuples = jax.scipy.linalg.cho_factor(K_noisy)
        alpha_mu = jax.scipy.linalg.cho_solve(cholesky_tuples, k_x_X)
        var = k(x, x) - jnp.dot(k_x_X.reshape(-1), alpha_mu)
        return jnp.sqrt(var)


    cholesky_tuples = jax.scipy.linalg.cho_factor(K_noisy)
    alpha_mu = jax.scipy.linalg.cho_solve(cholesky_tuples, ys)

    print(alpha_mu.shape)

    experiments = {
        'Minimize norm': alpha_minimize_norm,
        'Minimize distance': alpha_minimize_distance,
    }

    for title_name, function in experiments.items():
        alphas, prob = function(kernel_matrix=K_noisy, sigma=sigma, alpha_mu=alpha_mu)
        print(alphas)

        test_xs = jnp.linspace(-5, 5, num_plot_points)
        k_x_X = vmap(vmap(k, in_axes=(None, 0)), in_axes=(0, None))(test_xs, xs)
        mean = k_x_X @ alpha_mu
        optimized_fn = k_x_X @ alphas

        plt.plot(test_xs, jnp.sin(test_xs), label='True function', color='black')
        plt.scatter(xs, ys, color='red', label='Observations')
        plt.plot(test_xs, mean, label='Mean function', color='blue')

        posterior_std = vmap(sigma_n)(test_xs)
        plt.fill_between(test_xs, mean - 2 * posterior_std, mean + 2 * posterior_std, alpha=0.2, color='blue')
        plt.plot(test_xs, optimized_fn, label='Function with small RKHS norm', color='green')
        plt.legend()
        plt.title(title_name)
        plt.show()
