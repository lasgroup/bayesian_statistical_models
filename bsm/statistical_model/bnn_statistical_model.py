import chex
import jax.numpy as jnp
import jax.random as jr
import optax

from bsm.bayesian_regression.bayesian_neural_networks.bnn import BNNState
from bsm.bayesian_regression.bayesian_neural_networks.bnn import BayesianNeuralNet
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.fsvgd_ensemble import DeterministicFSVGDEnsemble, \
    ProbabilisticFSVGDEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.probabilistic_ensembles import ProbabilisticEnsemble
from bsm.statistical_model.abstract_statistical_model import StatisticalModel
from bsm.utils.normalization import Data
from bsm.utils.type_aliases import StatisticalModelState


class BNNStatisticalModel(StatisticalModel[BNNState]):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_training_steps: int = 1000,
                 beta: chex.Array | optax.Schedule | None = None,
                 bnn_type: BayesianNeuralNet = DeterministicEnsemble,
                 *args, **kwargs):
        self.bnn_type = bnn_type
        if self.bnn_type == DeterministicEnsemble:
            model = DeterministicEnsemble(input_dim=input_dim, output_dim=output_dim, *args, **kwargs)
        elif self.bnn_type == ProbabilisticEnsemble:
            model = ProbabilisticEnsemble(input_dim=input_dim, output_dim=output_dim, *args, **kwargs)
        elif self.bnn_type == DeterministicFSVGDEnsemble:
            model = DeterministicFSVGDEnsemble(input_dim=input_dim, output_dim=output_dim, *args, **kwargs)
        elif self.bnn_type == ProbabilisticFSVGDEnsemble:
            model = ProbabilisticFSVGDEnsemble(input_dim=input_dim, output_dim=output_dim, *args, **kwargs)
        else:
            raise NotImplementedError(f"Unknown BNN type: {self.bnn_type}")
        super().__init__(input_dim, output_dim, model)
        self.model = model
        self.num_training_steps = num_training_steps
        if beta is None:
            beta = jnp.ones(shape=(output_dim,))
        if isinstance(beta, chex.Array):
            beta = optax.constant_schedule(beta)
        self._potential_beta = beta

    def update(self,
               stats_model_state: StatisticalModelState[BNNState],
               data: Data) -> StatisticalModelState[BNNState]:
        """
        stats_model_state: statistical model state
        data: Data on which we train the statistical model
        """
        new_model_state = self.model.fit_model(data, self.num_training_steps, stats_model_state.model_state)
        beta = self._potential_beta(data.inputs.shape[0])
        assert beta.shape == (self.output_dim,)
        return StatisticalModelState(model_state=new_model_state, beta=beta)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    key = jr.PRNGKey(0)
    input_dim = 1
    output_dim = 2

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 64).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=1)
    ys = ys + noise_level * jr.normal(key=jr.PRNGKey(0), shape=ys.shape)
    data_std = noise_level * jnp.ones(shape=(output_dim,))
    data = Data(inputs=xs, outputs=ys)

    model = BNNStatisticalModel(input_dim=input_dim, output_dim=output_dim, output_stds=data_std, logging_wandb=False,
                                beta=jnp.array([1.0, 1.0]), num_particles=10, features=[64, 64, 64],
                                bnn_type=DeterministicEnsemble, train_share=0.6, num_training_steps=2000,
                                weight_decay=1e-4, )

    init_model_state = model.init(key=jr.PRNGKey(0))
    statistical_model_state = model.update(stats_model_state=init_model_state, data=data)

    # Test on new data
    test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1)

    preds = model.predict_batch(test_xs, statistical_model_state)

    for j in range(output_dim):
        plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        plt.plot(test_xs, preds.mean[:, j], label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (preds.mean[:, j] - preds.statistical_model_state.beta[j] * preds.epistemic_std[:,
                                                                                     j]).reshape(-1),
                         (preds.mean[:, j] + preds.statistical_model_state.beta[j] * preds.epistemic_std[:, j]).reshape(
                             -1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(f'bnn_{j}.pdf')
        plt.show()

    num_test_points = 1000
    in_domain_test_xs = jnp.linspace(d_l, d_u, num_test_points).reshape(-1, 1)
    in_domain_test_ys = jnp.concatenate([jnp.sin(in_domain_test_xs), jnp.cos(in_domain_test_xs)], axis=1)

    in_domain_preds = model.predict_batch(in_domain_test_xs, statistical_model_state)
    for j in range(output_dim):
        plt.plot(in_domain_test_xs, in_domain_preds.mean[:, j], label='Mean', color='blue')
        plt.plot(in_domain_test_xs, in_domain_test_ys[:, j], label='Fun', color='Green')
        plt.legend()
        plt.show()
