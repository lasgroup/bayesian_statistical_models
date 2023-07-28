import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from bsm.statistical_model.gp_statistical_model import GPStatisticalModel
from bsm.utils.normalization import Data
from bsm.utils.type_aliases import StatisticalModelOutput

key = jr.PRNGKey(0)
input_dim = 1
output_dim = 2

noise_level = 0.1
d_l, d_u = 0, 10
num_train_points = 32
xs = jnp.linspace(d_l, d_u, num_train_points).reshape(-1, 1)
ys = jnp.concatenate([jnp.sin(xs), jnp.cos(3 * xs)], axis=1)
ys = ys + noise_level * jr.normal(key=jr.PRNGKey(0), shape=ys.shape)
data_std = noise_level * jnp.ones(shape=(output_dim,))
data = Data(inputs=xs, outputs=ys)

model = GPStatisticalModel(input_dim=input_dim, output_dim=output_dim, output_stds=data_std, logging_wandb=False,
                           f_norm_bound=3, beta=None)
model_state = model.init(key=jr.PRNGKey(0))
statistical_model_state = model.update(model_state=model_state, data=data)

# Test on new data
num_test_points = 1000
test_xs = jnp.linspace(-5, 15, num_test_points).reshape(-1, 1)
test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(3 * test_xs)], axis=1)

preds = vmap(model, in_axes=(0, None),
             out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                             statistical_model_state=None))(test_xs, statistical_model_state)


def test_prediction_dimension():
    assert preds.mean.shape == (num_test_points, output_dim)
    assert preds.epistemic_std.shape == (num_test_points, output_dim)
    assert preds.aleatoric_std.shape == (num_test_points, output_dim)


def test_statistical_model_state_of_prediction():
    assert preds.statistical_model_state.beta.shape == (output_dim,)
    assert preds.statistical_model_state.model_state.history.inputs.shape == (num_train_points, input_dim)
    assert preds.statistical_model_state.model_state.history.outputs.shape == (num_train_points, output_dim)


in_domain_test_xs = jnp.linspace(0, 10, num_test_points).reshape(-1, 1)
in_domain_test_ys = jnp.concatenate([jnp.sin(in_domain_test_xs), jnp.cos(3 * in_domain_test_xs)], axis=1)

in_domain_preds = vmap(model, in_axes=(0, None),
                       out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                       statistical_model_state=None))(in_domain_test_xs,
                                                                                      statistical_model_state)


def test_good_gp_fit():
    assert jnp.mean((in_domain_preds.mean - in_domain_test_ys) ** 2) <= 1e-2
