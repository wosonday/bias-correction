from .core.bias_corrector import BiasCorrector

import numpy as np
from scipy import stats


class ParametricQuantileMapping(BiasCorrector):
    """
    Parametric quantile mapping bias correction method for ### Temperature ###
    - Suitable for Temperature fix the mean and standard deviation
    """

    def __init__(self):
        self.mu_ref_: float|None = None
        self.sigma_ref_: float|None = None
        self.mu_target_: float|None = None
        self.sigma_target_: float|None = None

    def fit(self, sim_hist: np.ndarray, obs_hist: np.ndarray) -> "ParametricQuantileMapping":
        self.mu_ref_, self.sigma_ref_ = np.mean(sim_hist), np.std(sim_hist)
        self.mu_target_, self.sigma_target_ = np.mean(obs_hist), np.std(obs_hist)
        return self

    def correct(self, x: np.ndarray) -> np.ndarray:
        if self.mu_ref_ is None or self.sigma_ref_ is None or self.mu_target_ is None or self.sigma_target_ is None:
            raise ValueError("Call fit() before correct()")

        z = (x - self.mu_ref_) / self.sigma_ref_
        return self.mu_target_ + self.sigma_target_ * z