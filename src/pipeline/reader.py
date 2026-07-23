import xarray as xr


def load_gcm(path: str, variables: str) -> xr.DataArray:
    """
    Read raw GCM file, return xr.DataArray of variables
    ---
    path: path to GCM file
    variables: variable names (ex: pr, tas,...)
    """
    ds = xr.open_dataset(path)

    if variables not in ds.data_vars:
        raise KeyError(f"Variable {variables} not found. Available variables: {list(ds.data_vars)}")

    return ds[variables]