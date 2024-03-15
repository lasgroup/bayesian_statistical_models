import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from bsm.bayesian_regression import ProbabilisticGRUEnsemble
from bsm.statistical_model import BRNNStatisticalModel
from bsm.utils.general_utils import create_windowed_array
from bsm.utils.normalization import Data

key = jr.PRNGKey(0)
input_dim = 1
output_dim = 2
window_size = 10
noise_level = 0.1
d_l, d_u = 0, 10
xs = jnp.linspace(d_l, d_u, 1024).reshape(-1, 1)
ys = jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=1)
weights = jnp.asarray([1.0, 0.5, 0.25, 0.15, 0.05])
weights = jnp.repeat(weights[..., None], repeats=2, axis=-1)
ys = vmap(lambda x, y: jnp.convolve(y, x, 'same'), in_axes=(-1, -1))(ys, weights)
ys = ys.transpose()
ys = ys * (1 + noise_level * jr.normal(key=jr.PRNGKey(0), shape=ys.shape))
x_train = create_windowed_array(xs, window_size=window_size)
y_train = create_windowed_array(ys, window_size=window_size)
data_std = noise_level * jnp.ones(shape=(output_dim,))
data = Data(inputs=x_train, outputs=y_train)

model = BRNNStatisticalModel(input_dim=input_dim, output_dim=output_dim, output_stds=data_std, logging_wandb=False,
                             beta=jnp.array([1.0, 1.0]), num_particles=10, features=[64, 64, 64],
                             bnn_type=ProbabilisticGRUEnsemble, num_training_steps=2000,
                             weight_decay=1e-5, hidden_state_size=20, num_cells=1, train_sequence_length=window_size)

init_stats_model_state = model.init(key=jr.PRNGKey(0))
statistical_model_state = model.update(stats_model_state=init_stats_model_state, data=data)

# Test on new data
num_test_points = 1000
test_xs = jnp.linspace(-5, 15, num_test_points).reshape(-1, 1)
test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1)
test_ys = vmap(lambda x, y: jnp.convolve(y, x, 'same'), in_axes=(-1, -1))(test_ys, weights)
test_ys = test_ys.transpose()

preds = model.predict_batch(test_xs, statistical_model_state)


def test_prediction_dimension():
    assert preds.mean.shape == (num_test_points, output_dim)
    assert preds.epistemic_std.shape == (num_test_points, output_dim)
    assert preds.aleatoric_std.shape == (num_test_points, output_dim)


def test_statistical_model_state_of_prediction():
    assert preds.statistical_model_state.beta.shape == (output_dim,)
    assert preds.statistical_model_state.model_state.calibration_alpha.shape == (output_dim,)


in_domain_test_xs = jnp.linspace(0, 10, num_test_points).reshape(-1, 1)
in_domain_test_ys = jnp.concatenate([jnp.sin(in_domain_test_xs), jnp.cos(in_domain_test_xs)], axis=1)
in_domain_test_ys = vmap(lambda x, y: jnp.convolve(y, x, 'same'), in_axes=(-1, -1))(in_domain_test_ys, weights)
in_domain_test_ys = in_domain_test_ys.transpose()

in_domain_preds = model.predict_batch(in_domain_test_xs, statistical_model_state)


def test_good_probabilistic_ensemble_fit():
    assert jnp.mean((in_domain_preds.mean - in_domain_test_ys) ** 2) <= 5e-2
