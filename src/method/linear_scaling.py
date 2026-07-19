from .core.bias_corrector import BiasCorrector
from .core.data_class import VARIABLE_REGISTRY, CorrectionMode
import numpy as np


class LinearScaling(BiasCorrector):

    def __init__(self, mode: CorrectionMode) -> None:
        self.mode = mode
        self.factor_: float|None = None

    @classmethod
    def variable(cls, variable: str) -> "LinearScaling":
        """VD: LinearScaling.variable('pr')"""
        return cls(mode=VARIABLE_REGISTRY[variable].mode)

    def fit(self, obs: np.ndarray, sim_hist: np.ndarray) -> "LinearScaling":
        self.factor_ = (np.mean(obs) - np.mean(sim_hist) if self.mode is CorrectionMode.ADDITIVE
                        else np.mean(obs) / np.mean(sim_hist))
        return self

    def correct(self, sim: np.ndarray):
        if self.factor_ is None:
            raise ValueError("fit must be called before correct")
        return sim + self.factor_ if self.mode is CorrectionMode.ADDITIVE else sim * self.factor_
