from collections import OrderedDict
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import vmap, jit
from jax.scipy.stats import multivariate_normal
from jaxtyping import PyTree, Float, Array, Scalar

import wandb
from bsm.bayesian_regression.bayesian_regression_model import BayesianRegressionModel
from bsm.bayesian_regression.gaussian_processes.kernels import Kernel, RBF
from bsm.utils.normal_with_aleatoric import ExtendedNormal
from bsm.utils.normalization import Normalizer, DataStats, Data
from bsm.bayesian_regression.gaussian_processes.rkhs_optimization import alpha_minimize_distance, alpha_minimize_norm


@chex.dataclass
class GPModelState:
    history: Data
    data_stats: DataStats
    params: PyTree
    alphas: Float[Array, 'output_dim num_data'] | None = None


class GaussianProcess(BayesianRegressionModel[GPModelState]):
    def __init__(self,
                 output_stds: chex.Array,
                 kernel: Kernel | None = None,
                 weight_decay: float = 0.0,
                 lr_rate: optax.Schedule | float = optax.constant_schedule(1e-2),
                 seed: int = 0,
                 logging_wandb: bool = True,
                 normalize: bool = True,
                 predict_regularized_mean: bool = False,
                 regularized_mean_strategy: str = 'minimize_norm',
                 # Should be in ['minimize_norm', 'minimize_distance']
                 f_norm_bound_RKHS_optimization: Float[Array, 'output_dim'] | Scalar = jnp.array(100.0),
                 beta_RKHS_optimization: Float[Array, 'output_dim'] | Scalar = jnp.array(3.0),
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        if kernel is None:
            kernel = RBF(self.input_dim)
        self.kernel = kernel
        assert output_stds.shape == (self.output_dim,)
        self.output_stds = output_stds
        self.normalizer = Normalizer()
        self.tx = optax.adamw(learning_rate=lr_rate, weight_decay=weight_decay)
        self.key = jr.PRNGKey(seed)
        self.logging_wandb = logging_wandb
        self.predict_regularized_mean = predict_regularized_mean
        self.regularized_mean_strategy = regularized_mean_strategy

        if f_norm_bound_RKHS_optimization.shape == ():
            f_norm_bound_RKHS_optimization = f_norm_bound_RKHS_optimization * jnp.ones(shape=(self.output_dim,))
        self.f_norm_bound_RKHS_optimization = f_norm_bound_RKHS_optimization

        if beta_RKHS_optimization.shape == ():
            beta_RKHS_optimization = beta_RKHS_optimization * jnp.ones(
                shape=(self.output_dim,))
        self.beta_RKHS_optimization = beta_RKHS_optimization

        self.v_kernel = vmap(self.kernel.apply, in_axes=(0, None, None), out_axes=0)
        self.m_kernel = vmap(self.v_kernel, in_axes=(None, 0, None), out_axes=1)
        self.m_kernel_multiple_output = vmap(self.m_kernel, in_axes=(None, None, 0), out_axes=0)
        self.v_kernel_multiple_output = vmap(self.v_kernel, in_axes=(None, None, 0), out_axes=0)
        self.kernel_multiple_output = vmap(self.kernel.apply, in_axes=(None, None, 0), out_axes=0)

    def init(self, key: chex.PRNGKey) -> GPModelState:
        inputs = jnp.zeros(shape=(1, self.input_dim))
        outputs = jnp.zeros(shape=(1, self.output_dim))
        data = Data(inputs=inputs, outputs=outputs)
        if self.normalize:
            data_stats = self.normalizer.compute_stats(data)
        else:
            data_stats = self.normalizer.init_stats(data)
        keys = jr.split(key, self.output_dim)
        params = vmap(self.kernel.init)(keys)
        return GPModelState(params=params, data_stats=data_stats, history=data, alphas=jnp.ones_like(outputs))

    def loss(self, vmapped_params, inputs, outputs, data_stats: DataStats):
        assert inputs.shape[0] == outputs.shape[0]

        # Normalize inputs and outputs
        inputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(inputs, data_stats.inputs)
        outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(outputs, data_stats.outputs)
        outputs_stds_norm = self.normalizer.normalize_std(self.output_stds, data_stats.outputs)

        # Compute covariance matrix
        covariance_matrix = self.m_kernel_multiple_output(inputs_norm, inputs_norm, vmapped_params)

        # Add noise term
        extended_eye = jnp.repeat(jnp.eye(covariance_matrix.shape[-1])[None, ...], repeats=self.output_dim, axis=0)
        noise_term = extended_eye * outputs_stds_norm[:, None, None] ** 2
        noisy_covariance_matrix = covariance_matrix + noise_term

        # Compute log pdf
        log_pdf = vmap(multivariate_normal.logpdf, in_axes=(1, None, 0))(outputs_norm, jnp.zeros(inputs.shape[0]),
                                                                         noisy_covariance_matrix)
        return -jnp.mean(log_pdf) / inputs.shape[0]

    @partial(jit, static_argnums=0)
    def _step_jit(self,
                  opt_state: optax.OptState,
                  vmapped_params: chex.PRNGKey,
                  inputs: chex.Array,
                  outputs: chex.Array,
                  data_stats: DataStats) -> (optax.OptState, PyTree, OrderedDict):
        loss, grads = jax.value_and_grad(self.loss)(vmapped_params, inputs, outputs,
                                                    data_stats)
        updates, opt_state = self.tx.update(grads, opt_state, vmapped_params)
        vmapped_params = optax.apply_updates(vmapped_params, updates)
        statiscs = OrderedDict(nll=loss)
        return opt_state, vmapped_params, statiscs

    def _train_model(self, num_training_steps: int, model_state: GPModelState, data_stats: DataStats, data: Data) \
            -> GPModelState:
        vmapped_params = model_state.params
        opt_state = self.tx.init(vmapped_params)

        for train_step in range(num_training_steps):
            opt_state, vmapped_params, statistics = self._step_jit(opt_state, vmapped_params, data.inputs,
                                                                   data.outputs, data_stats)
            if self.logging_wandb and train_step % self.logging_frequency == 0:
                wandb.log(statistics)

        new_model_state = GPModelState(history=data, data_stats=data_stats, params=vmapped_params)
        return new_model_state

    def compute_alphas_for_regularized_mean(self, gp_model: GPModelState) -> Float[Array, 'output_dim num_data']:
        # Compute covariance matrix
        num_data = gp_model.history.inputs.shape[0]
        history_inputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(gp_model.history.inputs,
                                                                                 gp_model.data_stats.inputs)
        history_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(gp_model.history.outputs,
                                                                                  gp_model.data_stats.outputs)
        covariance_matrix = self.m_kernel_multiple_output(history_inputs_norm, history_inputs_norm, gp_model.params)
        assert covariance_matrix.shape == (self.output_dim, num_data, num_data)
        # Add noise term
        extended_eye = jnp.repeat(jnp.eye(covariance_matrix.shape[-1])[None, ...], repeats=self.output_dim, axis=0)
        outputs_stds_norm = self.normalizer.normalize_std(self.output_stds, gp_model.data_stats.outputs)
        noise_term = extended_eye * outputs_stds_norm[:, None, None] ** 2
        noisy_covariance_matrix = covariance_matrix + noise_term
        cholesky_tuples = vmap(jax.scipy.linalg.cho_factor)(noisy_covariance_matrix)

        # Compute posterior mean
        denoised_mean = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 1))((cholesky_tuples[0], False),
                                                                                 history_outputs_norm)

        # We have
        new_alphas = []
        for i in range(self.output_dim):
            # alpha_minimize_norm, alpha_minimize_distance
            if self.regularized_mean_strategy == 'minimize_distance':
                alpha_value, prob = alpha_minimize_distance(kernel_matrix=covariance_matrix[i],
                                                            sigma=outputs_stds_norm[i],
                                                            alpha_mu=denoised_mean[i],
                                                            norm_bound=self.f_norm_bound_RKHS_optimization[i])
            elif self.regularized_mean_strategy == 'minimize_norm':
                alpha_value, prob = alpha_minimize_norm(kernel_matrix=covariance_matrix[i],
                                                        sigma=outputs_stds_norm[i],
                                                        alpha_mu=denoised_mean[i],
                                                        beta=self.beta_RKHS_optimization[i])
            new_alphas.append(alpha_value)
        new_alphas = jnp.stack(new_alphas, axis=0)
        return new_alphas

    def fit_model(self,
                  data: Data,
                  num_training_steps: int,
                  model_state: GPModelState) -> GPModelState:

        if self.normalize:
            data_stats = self.normalizer.compute_stats(data)
        else:
            data_stats = self.normalizer.init_stats(data)

        new_model_state = self._train_model(num_training_steps, model_state, data_stats, data)
        if self.predict_regularized_mean:
            alphas = self.compute_alphas_for_regularized_mean(new_model_state)
            new_model_state = new_model_state.replace(alphas=alphas)
        return new_model_state

    @partial(jit, static_argnums=0)
    def posterior(self, input, gp_model: GPModelState) -> Tuple[ExtendedNormal, ExtendedNormal]:
        assert input.ndim == 1
        input_norm = self.normalizer.normalize(input, gp_model.data_stats.inputs)

        history_inputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(gp_model.history.inputs,
                                                                                 gp_model.data_stats.inputs)
        history_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(gp_model.history.outputs,
                                                                                  gp_model.data_stats.outputs)

        outputs_stds_norm = self.normalizer.normalize_std(self.output_stds, gp_model.data_stats.outputs)

        # Compute covariance matrix
        covariance_matrix = self.m_kernel_multiple_output(history_inputs_norm, history_inputs_norm, gp_model.params)

        # Add noise term
        extended_eye = jnp.repeat(jnp.eye(covariance_matrix.shape[-1])[None, ...], repeats=self.output_dim, axis=0)
        noise_term = extended_eye * outputs_stds_norm[:, None, None] ** 2
        noisy_covariance_matrix = covariance_matrix + noise_term

        k_x_X = vmap(self.v_kernel,
                     in_axes=(None, None, 0),
                     out_axes=0)(history_inputs_norm, input_norm, gp_model.params)
        cholesky_tuples = vmap(jax.scipy.linalg.cho_factor)(noisy_covariance_matrix)

        # Compute posterior std
        denoised_var = vmap(jax.scipy.linalg.cho_solve,
                            in_axes=((0, None), 0))((cholesky_tuples[0], False), k_x_X)
        var = vmap(self.kernel.apply,
                   in_axes=(None, None, 0))(input_norm, input_norm, gp_model.params) - vmap(jnp.dot)(k_x_X,
                                                                                                     denoised_var)
        std = jnp.sqrt(var)

        # Compute posterior mean
        denoised_mean = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 1))((cholesky_tuples[0], False),
                                                                                 history_outputs_norm)
        if self.predict_regularized_mean:
            mean = vmap(jnp.dot)(k_x_X, model_state.alphas)
        else:
            mean = vmap(jnp.dot)(k_x_X, denoised_mean)

        # Denormalize
        mean = self.normalizer.denormalize(mean, gp_model.data_stats.outputs)
        std = self.normalizer.denormalize_std(std, gp_model.data_stats.outputs)

        # Distribution of f(x)
        dist_f = ExtendedNormal(loc=mean, scale=std)

        # Distribution of y(x) = f(x) + \epsilon
        std_with_noise = jnp.sqrt(std ** 2 + self.output_stds ** 2)
        # dist_y = distrax.Normal(loc=mean, scale=std_with_noise)
        dist_y = ExtendedNormal(loc=mean, scale=std_with_noise, aleatoric_std=self.output_stds)
        return dist_f, dist_y


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    # jax.config.update('jax_log_compiles', True)
    jax.config.update("jax_enable_x64", True)

    key = jr.PRNGKey(0)
    input_dim = 1
    output_dim = 2

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 64).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(3 * xs)], axis=1)
    ys = ys + noise_level * jr.normal(key=jr.PRNGKey(0), shape=ys.shape)
    data_std = noise_level * jnp.ones(shape=(output_dim,))
    data = DataStats(inputs=xs, outputs=ys)

    logging = False
    num_particles = 10
    model = GaussianProcess(input_dim=input_dim, output_dim=output_dim, output_stds=data_std, logging_wandb=False,
                            predict_regularized_mean=True, regularized_mean_strategy='minimize_distance')
    model_state = model.init(model.key)
    start_time = time.time()
    print('Starting with training')
    if logging:
        wandb.init(
            project='Pendulum',
            group='test group',
        )

    model_state = model.fit_model(data=data, num_training_steps=1000, model_state=model_state)
    print(f'Training time: {time.time() - start_time:.2f} seconds')
    model_state = model.fit_model(data=data, num_training_steps=1000, model_state=model_state)

    test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
    preds = vmap(model.posterior, in_axes=(0, None))(test_xs, model_state)[0]
    pred_means = preds.mean()
    epistemic_stds = preds.scale
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(3 * test_xs)], axis=1)

    for j in range(output_dim):
        plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        plt.plot(test_xs, pred_means[:, j], label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (pred_means[:, j] - 2 * epistemic_stds[:, j]).reshape(-1),
                         (pred_means[:, j] + 2 * epistemic_stds[:, j]).reshape(-1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()
