from core.data_class import CorrectionMode
from .core.bias_corrector import BiasCorrector
from ._processing import break_ties, apply_trace_floor
import numpy as np


class QuantileDeltaMapping(BiasCorrector):
    """
    Quantile Delta Mapping — Cannon, Sobie, & Murdock (2015).
    ---
    Quantile TAU compute the quantile in Sim_future (Not compute in Sim_hist),
    so this method can keep the delta change in GCMs projection. Even if
    Even when future variability differs significantly from history.
    """

    def __init__(self, n_quantiles: int = 1000,
                 mode: CorrectionMode = CorrectionMode.ADDITIVE,
                 trace: float = 0.0,
                 trace_calc: float = None,
                 random_state: int | None = 42,
                 ):
        self.n_quantiles = n_quantiles
        self.mode = mode
        self.trace = trace
        self.trace_calc = trace_calc if trace_calc is not None else 0.5 * trace
        self._rng = np.random.default_rng(random_state)
        self.quantiles_: np.ndarray | None = None
        self.sim_hist_sorted_: np.ndarray | None = None
        self.obs_hist_sorted_: np.ndarray | None = None


    def fit(self, sim_hist: np.ndarray, obs_hist: np.ndarray) -> "QuantileDeltaMapping":
        self.quantiles_ = np.linspace(0, 1, self.n_quantiles)
        sim_hist = break_ties(sim_hist, self.trace, self._rng)
        obs_hist = break_ties(obs_hist, self.trace, self._rng)
        self.sim_hist_sorted_ = np.quantile(sim_hist, self.quantiles_)
        self.obs_hist_sorted_ = np.quantile(obs_hist, self.quantiles_)
        return self

    def correct(self, x: np.ndarray) -> np.ndarray:
        """
        X: always is the future simulation of GCMs
        QDM not adjust each point independently, but require all entire array to caculate
        its own TAU quantile.
        """
        if self.quantiles_ is None:
            raise RuntimeError("Call fit() before correct()")
        x = break_ties(x, self.trace, self._rng)
        x_sorted = np.quantile(x, self.quantiles_)
        tau = np.interp(x, x_sorted, self.quantiles_)
        obs_at_tau = np.interp(tau, self.quantiles_, self.obs_hist_sorted_)
        sim_hist_at_tau = np.interp(tau, self.quantiles_, self.sim_hist_sorted_)

        corrected = (
            obs_at_tau + (x - sim_hist_at_tau) if self.mode is CorrectionMode.ADDITIVE
            else obs_at_tau * (x / sim_hist_at_tau)
        )
        return apply_trace_floor(corrected, self.trace_calc)

    def correct_windowed(
        self,
        sim_future: np.ndarray,
        future_years: np.ndarray,
        window_years: int = 30,
        step_years: int | None = None,
    ) -> np.ndarray:
        """Gọi lại self.correct() theo từng cửa sổ năm RIÊNG, thay vì
        một lần cho toàn bộ giai đoạn tương lai. Xem giải thích đầy đủ
        ở lượt trao đổi trước — chỉ QDM cần hàm này vì tau được tính
        trên chính chuỗi đưa vào correct().
        """
        if self.quantiles_ is None:
            raise RuntimeError("Call fit() before correct_windowed()")
        if step_years is None:
            step_years = window_years

        result = np.full_like(sim_future, np.nan, dtype=float)
        year_start = int(future_years.min())
        year_end = int(future_years.max())

        for w_start in range(year_start, year_end + 1, step_years):
            w_end = w_start + window_years - 1
            mask = (future_years >= w_start) & (future_years <= w_end)
            if not np.any(mask):
                continue
            result[mask] = self.correct(sim_future[mask])

        return result