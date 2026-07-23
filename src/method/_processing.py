import numpy as np

"""
Using example: 
        from method.preprocessing import frequency_adapt, apply_wet_day_threshold
        from method.gamma_qm import GammaQuantileMapping

        sim_hist_adapted, sim_threshold = frequency_adapt(sim_hist, obs_hist, threshold=1.0)
        sim_future_adapted = apply_wet_day_threshold(sim_future, sim_threshold)

        gqm = GammaQuantileMapping(threshold=1.0)
        gqm.fit(sim_hist_adapted, obs_hist)
        pr_corrected = gqm.correct(sim_future_adapted)

### verify 
sim_hist_adapted, thr = frequency_adapt(sim_hist, obs_hist, threshold=1.0)
sim_future_adapted = apply_wet_day_threshold(sim_future, thr)

assert np.mean(sim_hist_adapted > 0) == pytest.approx(np.mean(obs_hist >= 1.0), abs=1e-6)

"""

def apply_wet_day_threshold(sim: np.ndarray, sim_threshold: float) -> np.ndarray:
    """Apply a wet-day threshold calibrated by frequency_adapt() to another simulated
    series (typically sim_future). Values below the threshold are set to zero.

    The threshold is not recalculated here: it is assumed time-invariant between the
    historical and future period, per Themeßl, Gobiet, & Heinrich (2012).
    """
    return np.where(sim >= sim_threshold, sim, 0.0)


def frequency_adapt(sim_hist: np.ndarray, obs_hist: np.ndarray, threshold: float = 1.0) -> tuple[np.ndarray, float]:
    """Adjust the sim's rainy day frequency to match the obs
    Abefore applying any intensity correction algorithms (Gamma QM, etc.).
    ### Source: Themeßl, Gobiet, & Leuprecht (2011); Themeßl, Gobiet, & Heinrich (2012). ###
    """
    p_obs_wet = np.mean(obs_hist >= threshold)
    sim_threshold = np.quantile(sim_hist, 1 - p_obs_wet)
    sim_adapted = apply_wet_day_threshold(sim_hist, sim_threshold)
    return sim_adapted, sim_threshold

def break_ties(arr: np.ndarray, trace: float, rng: np.random.Generator) -> np.ndarray:
    """
    Jitter các giá trị dưới `trace` (vd. ngày mưa = 0) bằng nhiễu đều trong [0, trace),
    giúp ECDF/quantile tăng chặt, tránh ánh xạ nghịch (np.interp) nhập nhằng khi có
    nhiều giá trị trùng nhau. No-op nếu trace <= 0.
    ### Source: Cannon (2018), MBC::QDM — tham số trace / jitter.factor. ###
    """
    if trace <= 0:
        return arr
    arr = arr.copy()
    below = arr < trace
    n_below = np.count_nonzero(below)
    if n_below > 0:
        arr[below] = rng.uniform(0.0, trace, size=n_below)
    return arr


def apply_trace_floor(corrected: np.ndarray, trace_calc: float) -> np.ndarray:
    """
    Đặt lại về 0 các giá trị hiệu chỉnh nằm dưới `trace_calc`, loại bỏ nhiễu jitter
    (do break_ties tạo ra ở đầu vào) khỏi kết quả đầu ra cuối cùng.
    ### Source: Cannon (2018), MBC::QDM — tham số trace.calc (mặc định 0.5*trace). ###
    """
    if trace_calc <= 0:
        return corrected
    return np.where(corrected < trace_calc, 0.0, corrected)
