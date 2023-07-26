from typing import TypeVar, Generic

import chex

ModelState = TypeVar('ModelState')

EpistemicStd = chex.Array
AleatoricStd = chex.Array
Mean = chex.Array
Beta = chex.Array


@chex.dataclass
class StatisticalModelOutput(Generic[ModelState]):
    mean: Mean
    epistemic_std: EpistemicStd
    aleatoric_std: AleatoricStd
    beta: Beta
    model_state: ModelState
