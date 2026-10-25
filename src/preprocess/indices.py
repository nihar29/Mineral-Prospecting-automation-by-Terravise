"""Spectral indices and terrain derivatives (pure NumPy, easy to unit-test).

Band ratios follow the project tracker (data.xlsx):
  Landsat 8/9 : B4/B2 iron oxide, (B6+B7)/B5 clay+carbonate, B6/B5 ferrous minerals,
                NDVI = (B5-B4)/(B5+B4)
  ASTER       : B2/B1 ferric iron, B4/B3 ferrous iron, (B5+B7)/B6 clay,
                B13/B10 carbonate, B9/B6 silica alteration
"""
from __future__ import annotations

import numpy as np

LANDSAT_SCALE, LANDSAT_OFFSET = 0.0000275, -0.2   # Collection 2 Level-2 surface reflectance


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def landsat_reflectance(dn: np.ndarray) -> np.ndarray:
    """Convert Landsat C2 L2 DN to surface reflectance (0-1)."""
    refl = dn.astype(np.float32) * LANDSAT_SCALE + LANDSAT_OFFSET
    return np.clip(refl, 0, 1)


# ---- Landsat 8/9 (blue=B2, green=B3, red=B4, nir08=B5, swir16=B6, swir22=B7) ----
def ndvi(red, nir):
    return safe_ratio(nir - red, nir + red)


def iron_oxide(red, blue):
    return safe_ratio(red, blue)


def clay_carbonate(swir16, swir22, nir):
    return safe_ratio(swir16 + swir22, nir)


def ferrous(swir16, nir):
    return safe_ratio(swir16, nir)


# ---- ASTER (b1..b14) ----
ASTER_RATIOS = {
    "aster_ferric_iron": lambda b: safe_ratio(b[2], b[1]),
    "aster_ferrous_iron": lambda b: safe_ratio(b[4], b[3]),
    "aster_clay": lambda b: safe_ratio(b[5] + b[7], b[6]),
    "aster_carbonate": lambda b: safe_ratio(b[13], b[10]),
    "aster_silica": lambda b: safe_ratio(b[9], b[6]),
}


# ---- Terrain ----
def slope_aspect(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (degrees) and aspect (degrees clockwise from north)."""
    dzdy, dzdx = np.gradient(dem.astype(np.float32), res)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360) % 360
    return slope, aspect


def curvature(dem: np.ndarray, res: float) -> np.ndarray:
    """Laplacian curvature (positive = convex)."""
    dzdy, dzdx = np.gradient(dem.astype(np.float32), res)
    d2y, _ = np.gradient(dzdy, res)
    _, d2x = np.gradient(dzdx, res)
    return -(d2x + d2y)
