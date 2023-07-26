from typing import TypeVar, Generic

import chex

ModelState = TypeVar('ModelState')

EpistemicStd = chex.Array
AleatoricStd = chex.Array
Mean = chex.Array
Beta = chex.Array

@chex.dataclass
class StatisticalModelState(Generic[ModelState]):
    model_state: ModelState
    beta: Beta

@chex.dataclass
class StatisticalModelOutput(Generic[ModelState]):
    mean: Mean
    epistemic_std: EpistemicStd
    aleatoric_std: AleatoricStd
    statistical_model_state: StatisticalModelState[ModelState]
