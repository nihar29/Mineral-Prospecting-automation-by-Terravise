"""Raster I/O helpers shared by all stages.

Every layer is aligned to ONE master grid (same CRS, transform, shape), which
removes the 'all input arrays must have the same shape' class of errors.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


def make_master_profile(aoi_path: str, crs: str, res: float, nodata: float) -> dict:
    """Build the master grid profile from the AOI bounds, snapped to the pixel size."""
    aoi = gpd.read_file(aoi_path).to_crs(crs)
    minx, miny, maxx, maxy = aoi.total_bounds
    minx, miny = np.floor(minx / res) * res, np.floor(miny / res) * res
    maxx, maxy = np.ceil(maxx / res) * res, np.ceil(maxy / res) * res
    width, height = int((maxx - minx) / res), int((maxy - miny) / res)
    return {
        "driver": "GTiff",
        "crs": rasterio.crs.CRS.from_string(crs),
        "transform": from_origin(minx, maxy, res, res),
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }


def align_to_master(path: str | Path, master: dict, categorical: bool = False) -> np.ndarray:
    """Reproject/resample a raster onto the master grid. Returns float32 with NaN for nodata."""
    resampling = Resampling.nearest if categorical else Resampling.bilinear
    out = np.full((master["height"], master["width"]), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        src_arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            src_arr[src_arr == src.nodata] = np.nan
        reproject(
            source=src_arr,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=master["transform"],
            dst_crs=master["crs"],
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return out


def write_raster(path: str | Path, arr: np.ndarray, master: dict, descriptions=None) -> None:
    """Write a 2-D or 3-D (bands, rows, cols) array using the master profile."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    profile = master.copy()
    profile.update(count=arr.shape[0], dtype="float32")
    nodata = profile["nodata"]
    data = np.where(np.isnan(arr), nodata, arr)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if descriptions:
            for i, name in enumerate(descriptions, start=1):
                dst.set_band_description(i, name)


def read_stack(path: str | Path) -> tuple[np.ndarray, list[str], dict]:
    """Read a multi-band stack -> (array with NaN, band names, profile)."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        names = [d or f"band_{i + 1}" for i, d in enumerate(src.descriptions)]
        profile = src.profile.copy()
    return arr, names, profile
