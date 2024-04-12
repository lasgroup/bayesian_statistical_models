from abc import abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Dict, Tuple, Callable
from typing import Sequence

import chex
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from jax import jit
from jax import vmap
from jax.lax import scan
from jax.scipy.stats import norm
from jaxtyping import PyTree

import wandb
from bsm.bayesian_regression.bayesian_regression_model import BayesianRegressionModel
from bsm.utils.network_utils import MLP
from bsm.utils.normalization import Normalizer, DataStats, Data
from bsm.utils.particle_distribution import ParticleDistribution

NO_EVAL_VALUE = 123213142134.645954392592


@chex.dataclass
class BNNState:
    vmapped_params: PyTree
    data_stats: DataStats
    calibration_alpha: chex.Array


class BayesianNeuralNet(BayesianRegressionModel[BNNState]):
    def __init__(self,
                 features: Sequence[int],
                 num_particles: int,
                 weight_decay: float = 1e-4,
                 lr_rate: optax.Schedule | float = optax.constant_schedule(1e-3),
                 activation: Callable = nn.swish,
                 num_calibration_ps: int = 10,
                 num_test_alphas: int = 100,
                 logging_wandb: bool = True,
                 batch_size: int = 32,
                 seed: int = 0,
                 train_share: bool = 0.8,
                 eval_batch_size: int = 2048,
                 eval_frequency: int | None = None,
                 return_best_model: bool = False,
                 max_buffer_size: int = 1_000_000,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_particles = num_particles
        self.model = MLP(features=features, output_dim=self.output_dim, activation=activation)
        self.tx = optax.adamw(learning_rate=lr_rate, weight_decay=weight_decay)
        self.normalizer = Normalizer()
        self.num_calibration_ps = num_calibration_ps
        self.num_test_alphas = num_test_alphas
        self.logging_wandb = logging_wandb
        self.batch_size = batch_size
        self.key = jr.PRNGKey(seed)
        self.train_share = train_share
        self.eval_batch_size = eval_batch_size
        self.return_best_model = return_best_model
        self.max_buffer_size = max_buffer_size
        assert not (eval_frequency is None and self.return_best_model), "cannot return best model if not " \
                                                                        "evaluating"
        self._eval_frequency = eval_frequency
        self.set_up_data_buffers()

    def set_up_data_buffers(self):
        dummy_data_sample = Data(inputs=jnp.zeros(self.input_dim), outputs=jnp.zeros(self.output_dim))
        self.train_buffer = UniformSamplingQueue(max_replay_size=self.max_buffer_size,
                                                 dummy_data_sample=dummy_data_sample,
                                                 sample_batch_size=self.batch_size)

        self.eval_buffer = UniformSamplingQueue(max_replay_size=self.max_buffer_size,
                                                dummy_data_sample=dummy_data_sample,
                                                sample_batch_size=self.eval_batch_size)

    @property
    def evaluation_frequency(self):
        if self._eval_frequency:
            eval_frequency = self._eval_frequency
        else:
            eval_frequency = -1
        return eval_frequency

    @partial(jit, static_argnums=(0,))
    def posterior(self, input: chex.Array, bnn_state: BNNState) -> Tuple[ParticleDistribution, ParticleDistribution]:
        """Computes the posterior distribution of the ensemble given the input and the data statistics."""
        chex.assert_shape(input, (self.input_dim,))
        v_apply = vmap(self.apply_eval, in_axes=(0, None, None), out_axes=0)
        means, aleatoric_stds = v_apply(bnn_state.vmapped_params, input, bnn_state.data_stats)
        assert means.shape == aleatoric_stds.shape == (self.num_particles, self.output_dim)
        f_dist = ParticleDistribution(means, calibration_alpha=bnn_state.calibration_alpha)
        y_dist = ParticleDistribution(means, aleatoric_stds, calibration_alpha=bnn_state.calibration_alpha)
        return f_dist, y_dist

    @abstractmethod
    def _apply_train(self,
                     params: PyTree,
                     x: chex.Array,
                     data_stats: DataStats) -> [chex.Array, chex.Array]:
        pass

    @abstractmethod
    def apply_eval(self,
                   params: PyTree,
                   x: chex.Array,
                   data_stats: DataStats) -> [chex.Array, chex.Array]:
        pass

    def _nll(self,
             predicted_outputs: chex.Array,
             predicted_stds: chex.Array,
             target_outputs: chex.Array) -> chex.Array:
        # chex.assert_equal_shape([target_outputs, predicted_stds[0, ...], predicted_outputs[0, ...]])
        chex.assert_shape(target_outputs, (self.output_dim,))
        chex.assert_shape(predicted_stds, (self.output_dim,))
        chex.assert_shape(predicted_outputs, (self.output_dim,))
        log_prob = norm.logpdf(target_outputs[jnp.newaxis, :], loc=predicted_outputs, scale=predicted_stds)
        return - jnp.mean(log_prob)

    def _neg_log_posterior(self,
                           predicted_outputs: chex.Array,
                           predicted_stds: chex.Array,
                           target_outputs: jax.Array) -> chex.Array:
        nll = jax.vmap(jax.vmap(self._nll), in_axes=(0, 0, None))(predicted_outputs, predicted_stds, target_outputs)
        neg_log_post = nll.mean()
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
            stats, params = flax.core.pop(variables, 'params')
        else:
            stats, params = variables
        del variables  # Delete variables to avoid wasting resources
        return params

    def init(self, key) -> BNNState:
        inputs = jnp.zeros(shape=(1, self.input_dim))
        outputs = jnp.zeros(shape=(1, self.output_dim))
        data = Data(inputs=inputs, outputs=outputs)
        data_stats = self.normalizer.compute_stats(data)
        keys = jr.split(key, self.num_particles)
        vmapped_params = vmap(self._init)(keys)
        calibration_alpha = jnp.ones(shape=(self.output_dim,))
        return BNNState(vmapped_params=vmapped_params,
                        data_stats=data_stats,
                        calibration_alpha=calibration_alpha,
                        )

    def calibrate(self,
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
            aleatoric_var = (predicted_stds ** 2).mean(axis=0)
            std = jnp.sqrt((epistemic_stds * alpha) ** 2 + aleatoric_var)
            chex.assert_shape(std, (self.output_dim,))
            cdfs = vmap(norm.cdf)(y, means, std)

            def check_cdf(cdf):
                chex.assert_shape(cdf, ())
                return cdf <= ps

            return vmap(check_cdf, out_axes=1)(cdfs)

        cdfs = vmap(calculate_score)(inputs, outputs)
        return jnp.mean(cdfs, axis=0)

    @partial(jax.jit, static_argnums=(0, 1))
    def _train_model(self,
                     num_epochs: int,
                     model_state: BNNState,
                     data_stats: DataStats,
                     train_buffer_state: ReplayBufferState,
                     eval_buffer_state: ReplayBufferState) -> [BNNState, OrderedDict, OrderedDict]:

        vmapped_params = model_state.vmapped_params
        opt_state = self.tx.init(vmapped_params)

        eval_frequency = self.evaluation_frequency

        best_nll = 1e12

        def evaluate_model(vmapped_params, eval_data, stats, best_params, best_nll):
            eval_nll, eval_mse = self.loss(vmapped_params, eval_data.inputs,
                                           eval_data.outputs, data_stats)
            stats = OrderedDict(eval_nll=eval_nll, eval_mse=eval_mse, **stats)
            test_nll, test_mse = self.loss(best_params, eval_data.inputs,
                                           eval_data.outputs, data_stats)

            new_best_params, new_best_nll = jax.lax.cond(
                eval_nll < test_nll,
                lambda: (vmapped_params, eval_nll),
                lambda: (best_params, test_nll)
            )
            return stats, new_best_params, new_best_nll

        def skip_evaluation(vmapped_params, eval_data, stats, best_params, best_nll):
            stats = OrderedDict(eval_nll=NO_EVAL_VALUE, eval_mse=NO_EVAL_VALUE, **stats)
            return stats, best_params, best_nll

        def f(carry, ins):
            opt_state, vmapped_params, buffer_state, best_params, best_nll, eval_buffer_state = carry
            new_buffer_state, data_batch = self.train_buffer.sample(buffer_state)
            opt_state, vmapped_params, statistics = self.step_jit(opt_state, vmapped_params, data_batch.inputs,
                                                                  data_batch.outputs, data_stats)

            new_eval_buffer_state, eval_data_batch = self.eval_buffer.sample(eval_buffer_state)

            statistics, best_params, best_nll = jax.lax.cond(
                ins % eval_frequency,
                skip_evaluation,
                evaluate_model,
                vmapped_params,
                eval_data_batch,
                statistics,
                best_params,
                best_nll
            )

            return (opt_state, vmapped_params, new_buffer_state, best_params, best_nll, new_eval_buffer_state), \
                statistics

        init_carry = (opt_state, vmapped_params, train_buffer_state, vmapped_params, best_nll, eval_buffer_state)
        iterations = jnp.arange(start=0, step=1, stop=num_epochs, dtype=jnp.int32)
        (opt_state, vmapped_params, buffer_state, best_params, best_nll, eval_buffer_state), statistics = \
            scan(f, init_carry, iterations, length=num_epochs)

        train_statistics = OrderedDict(nll=statistics['nll'], mse=statistics['mse'])
        eval_statistics = OrderedDict(eval_nll=statistics['eval_nll'], eval_mse=statistics['eval_mse'])
        desired_params = vmapped_params
        if self.return_best_model:
            desired_params = best_params
        calibrate_alpha = jnp.ones(self.output_dim)
        new_model_state = BNNState(data_stats=data_stats, vmapped_params=desired_params,
                                   calibration_alpha=calibrate_alpha)
        return new_model_state, train_statistics, eval_statistics

    def _prepare_data_for_training(self, data: Data) -> [DataStats, Data, Data]:
        data_stats = self.normalizer.compute_stats(data)
        num_points = data.inputs.shape[0]
        # Prepare data
        self.key, key = jr.split(self.key)
        permuted_data = jtu.tree_map(lambda x: jr.permutation(key, x), data)

        if self.train_share < 1.0:
            # Taking self.train_share number of points for training
            train_data = jtu.tree_map(lambda x: x[:int(self.train_share * num_points)], permuted_data)
            # Taking the rest for calibration
            eval_data = jtu.tree_map(lambda x: x[int(self.train_share * num_points):], permuted_data)
        else:
            train_data = permuted_data
            eval_data = permuted_data
        return data_stats, train_data, eval_data

    def fit_model(self, data: Data, num_epochs: int, model_state: BNNState) -> BNNState:

        self.key, key_buffer, key_eval_buffer = jr.split(self.key, 3)
        buffer_state = self.train_buffer.init(key_buffer)
        eval_buffer_state = self.eval_buffer.init(key_eval_buffer)

        data_stats, train_data, eval_data = self._prepare_data_for_training(data)

        train_buffer_state = self.train_buffer.insert(buffer_state, train_data)
        eval_buffer_state = self.eval_buffer.insert(eval_buffer_state, eval_data)
        eval_frequency = self.evaluation_frequency
        new_model_state, train_statistics, eval_statistics = self._train_model(num_epochs,
                                                                               model_state,
                                                                               data_stats,
                                                                               train_buffer_state,
                                                                               eval_buffer_state
                                                                               )

        if self.logging_wandb:
            for i in range(num_epochs):
                wandb.log(jtu.tree_map(lambda x: x[i], train_statistics))
                if i % eval_frequency == 0:
                    wandb.log(jtu.tree_map(lambda x: x[i], eval_statistics))
        if self.train_share > 0:
            if eval_data.inputs.shape[0] > self.eval_batch_size:
                new_eval_buffer_state, data_batch = self.eval_buffer.sample(eval_buffer_state)
                calibrate_alpha = self.calibrate(new_model_state.vmapped_params, data_batch.inputs, data_batch.outputs,
                                                 data_stats)
            else:
                calibrate_alpha = self.calibrate(new_model_state.vmapped_params, eval_data.inputs, eval_data.outputs,
                                                 data_stats)
        else:
            calibrate_alpha = jnp.ones(self.output_dim)

        new_model_state = new_model_state.replace(calibration_alpha=calibrate_alpha)
        return new_model_state
