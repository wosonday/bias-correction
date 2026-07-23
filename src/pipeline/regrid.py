from typing import Literal
import xarray as xr


Literal = Literal["linear", "nearest", "slinear", "cubic", "quintic", "pchip"]

def regrid(ref: xr.DataArray, target: xr.DataArray, lat_name: str = "lat",
           lon_name: str = "lon", method: Literal = 'nearest') -> xr.DataArray:
    """
    regrid GCMs to obs grid
    """
    return ref.interp(
        {lat_name: target[lat_name], lon_name: target[lon_name]},
        method=method,
    )

