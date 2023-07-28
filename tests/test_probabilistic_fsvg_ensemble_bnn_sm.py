import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from bsm.models.bayesian_neural_networks.fsvgd_ensemble import ProbabilisticFSVGDEnsemble
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from bsm.utils.normalization import Data
from bsm.utils.type_aliases import StatisticalModelOutput

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
                            bnn_type=ProbabilisticFSVGDEnsemble, train_share=0.6, num_training_steps=2000,
                            weight_decay=1e-1)

init_model_state = model.init(key=jr.PRNGKey(0))
statistical_model_state = model.update(model_state=init_model_state, data=data)

# Test on new data
num_test_points = 1000
test_xs = jnp.linspace(-5, 15, num_test_points).reshape(-1, 1)
test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1)

preds = vmap(model.predict, in_axes=(0, None),
             out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                             statistical_model_state=None))(test_xs, statistical_model_state)


def test_prediction_dimension():
    assert preds.mean.shape == (num_test_points, output_dim)
    assert preds.epistemic_std.shape == (num_test_points, output_dim)
    assert preds.aleatoric_std.shape == (num_test_points, output_dim)


def test_statistical_model_state_of_prediction():
    assert preds.statistical_model_state.beta.shape == (output_dim,)
    assert preds.statistical_model_state.model_state.calibration_alpha.shape == (output_dim,)


in_domain_test_xs = jnp.linspace(0, 10, num_test_points).reshape(-1, 1)
in_domain_test_ys = jnp.concatenate([jnp.sin(in_domain_test_xs), jnp.cos(in_domain_test_xs)], axis=1)

in_domain_preds = vmap(model.predict, in_axes=(0, None),
                       out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                       statistical_model_state=None))(in_domain_test_xs,
                                                                                      statistical_model_state)


def test_good_deterministic_ensemble_fit():
    assert jnp.mean((in_domain_preds.mean - in_domain_test_ys) ** 2) <= 1e-2
