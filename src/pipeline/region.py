from dataclasses import dataclass
from typing import Optional

import xarray as xr

@dataclass
class Region:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    name: Optional[str] = 'pipeline'

    @classmethod
    def point(cls, lat: float, lon: float,
              buffer: float = 0.25, name: str = "point") -> "Region":
        return cls(lat - buffer, lat + buffer, lon - buffer, lon + buffer, name)


def extract_region(da: xr.DataArray, region: Region,
                   lat_name: str = "lat", lon_name: str = "lon") -> xr.DataArray:
    """
    Clip DataArray by pipeline. Handle the case of decreasing latitude.
    (Many GCMs store from 90 to -90 degress, not -90 to 90 degress)
    """

    lat_ascending = bool(da[lat_name][0] < da[lat_name][-1])
    lat_slice = (
            slice(region.lat_min, region.lat_max) if lat_ascending
                    else slice(region.lat_max, region.lat_min)
    )
    return da.sel({
        lat_name: lat_slice,
        lon_name: slice(region.lon_min, region.lon_max)
        })
