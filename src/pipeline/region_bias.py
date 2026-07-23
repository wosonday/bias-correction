import numpy as np
import xarray as xr


def bias_correct_region_dask(
    sim_hist: xr.DataArray,
    obs_hist: xr.DataArray,
    sim_future: xr.DataArray,
    corrector_factory,
    chunks: dict | None = None,
    compute: bool = True,
    scheduler: str | None = None,
    time_name: str = "time",
) -> xr.DataArray:
    """
    Bias correction by region using dask
    """

    if chunks is not None:
        sim_hist = sim_hist.chunk(chunks)
        obs_hist = obs_hist.chunk(chunks)
        sim_future = sim_future.chunk(chunks)

    def _fit_correct_1d(sh: np.ndarray, oh: np.ndarray, sf: np.ndarray) -> np.ndarray:
        if np.all(np.isnan(sh)) or np.all(np.isnan(oh)):
            return np.full_like(sf, np.nan, dtype=float)
        corrector = corrector_factory()
        corrector.fit(sh, oh)
        return corrector.correct(sf)

    result = xr.apply_ufunc(
        _fit_correct_1d,
        sim_hist, obs_hist, sim_future,
        input_core_dims=[[time_name], [time_name], [time_name]],
        output_core_dims=[[time_name]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    if not compute:
        return result
    return result.compute(scheduler=scheduler) if scheduler else result.compute()