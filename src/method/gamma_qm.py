from .core.bias_corrector import BiasCorrector

import numpy as np
from scipy import stats



class GammaQuantileMapping(BiasCorrector):
    """
    Gamma quantile mapping assumes a Gamma distribution for the rainfall intensity on wet days.
    Untreated frequency of wet days.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.shape_ref_: float | None = None
        self.scale_ref_: float | None = None
        self.shape_target_: float | None = None
        self.scale_target_: float | None = None

    def fit(self, sim_hist: np.ndarray, obs_hist: np.ndarray) -> "GammaQuantileMapping":
        sim_wet = sim_hist[sim_hist >= self.threshold]
        obs_wet = obs_hist[obs_hist >= self.threshold]
        self.shape_ref_, _, self.scale_ref_ = stats.gamma.fit(sim_wet, floc=0)
        self.shape_target_, _, self.scale_target_ = stats.gamma.fit(obs_wet, floc=0)
        return self


    def correct(self, x: np.ndarray) -> np.ndarray:
        if self.shape_ref_ is None or self.scale_ref_ is None or self.shape_target_ is None or self.scale_target_ is None:
            raise ValueError("Call fit() before correct()")
        return np.where(
            x >= self.threshold,
            stats.gamma.ppf(
                stats.gamma.cdf(x, self.shape_ref_, scale=self.scale_ref_),
                self.shape_target_, scale=self.scale_target_,
            ),
            x,
        )
