from core.data_class import CorrectionMode
from .core.bias_corrector import BiasCorrector
import numpy as np


class QuantileDeltaMapping(BiasCorrector):
    """
    Quantile Delta Mapping — Cannon, Sobie, & Murdock (2015).
    ---
    Quantile TAU compute the quantile in Sim_future (Not compute in Sim_hist),
    so this method can keep the delta change in GCMs projection. Even if
    Even when future variability differs significantly from history.
    """

    def __init__(self, n_quantiles: int = 1000, mode: CorrectionMode = CorrectionMode.ADDITIVE):
        self.n_quantiles = n_quantiles
        self.mode = mode
        self.quantiles_: np.ndarray | None = None
        self.sim_hist_sorted_: np.ndarray | None = None
        self.obs_hist_sorted_: np.ndarray | None = None


    def fit(self, sim_hist: np.ndarray, obs_hist: np.ndarray) -> "QuantileDeltaMapping":
        self.quantiles_ = np.linspace(0, 1, num=self.n_quantiles)
        self.sim_hist_sorted_ = np.quantile(sim_hist, self.quantiles_)
        self.obs_hist_sorted_ = np.quantile(obs_hist, self.quantiles_)

        return self

    def correct(self, x: np.ndarray) -> np.ndarray:
        """
        X: always is the future simulation of GCMs
        QDM not adjust each point independently, but require all entire array to caculate
        its own TAU quantile.
        """
        if self.quantiles_ or self.mode or self.sim_hist_sorted_ or self.obs_hist_sorted_ is None:
            raise RuntimeError("Call fit() before correct()")

        future_sorted = np.quantile(x, self.quantiles_)
        tau = np.interp(x, future_sorted, self.quantiles_)
        obs_tau = np.interp(tau, self.quantiles_, self.obs_hist_sorted_)
        sim_hist_tau = np.interp(tau, self.quantiles_, self.sim_hist_sorted_)

        if self.mode == CorrectionMode.ADDITIVE:
            return obs_tau + (x - sim_hist_tau)
        return obs_tau * (x / sim_hist_tau)

    