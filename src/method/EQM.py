from .core.bias_corrector import BiasCorrector
import numpy as np


class EmpiricalQuantileMapping(BiasCorrector):
    """
    Empirical Quantile Mapping (EQM) bias correction method.
    - Mapping bases on empirical cumulative distribution functions (CDFs)
    - Correct the entire distribution shape, not just the mean like linear scaling
    """

    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles    = n_quantiles
        self.quantiles_     : np.ndarray | None = None
        self.ref_sorted_    : np.ndarray | None = None
        self.target_sorted_ : np.ndarray | None = None

    def fit(self, sim_hist: np.ndarray, obs_hist: np.ndarray) -> "EmpiricalQuantileMapping":
        self.quantiles_ = np.linspace(0, 1, self.n_quantiles)
        self.ref_sorted_= np.quantile(sim_hist, self.quantiles_)
        self.target_sorted_ = np.quantile(obs_hist, self.quantiles_)
        return self

    def correct(self, x: np.ndarray) -> np.ndarray:

        if self.ref_sorted_ is None or self.quantiles_ is None or self.target_sorted_ is None:
            raise RuntimeError("Call fit() before correct()")

        p = np.interp(x, self.ref_sorted_, self.quantiles_)        # quantile of x
        return np.interp(p, self.quantiles_, self.target_sorted_)   # target values of "p" quantiles
