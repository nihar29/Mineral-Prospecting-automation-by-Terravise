import numpy as np

from src.preprocess import indices as ix


def test_safe_ratio_handles_zero():
    out = ix.safe_ratio(np.array([1.0, 2.0]), np.array([0.0, 4.0]))
    assert np.isnan(out[0]) and out[1] == 0.5


def test_ndvi_range():
    red, nir = np.array([0.1, 0.3]), np.array([0.5, 0.3])
    v = ix.ndvi(red, nir)
    assert np.all((v >= -1) & (v <= 1))
    assert np.isclose(v[0], 0.4 / 0.6) and v[1] == 0


def test_landsat_reflectance_scaling():
    assert np.isclose(ix.landsat_reflectance(np.array([10000]))[0], 10000 * 0.0000275 - 0.2)


def test_slope_flat_and_tilted():
    flat = np.ones((10, 10))
    s, _ = ix.slope_aspect(flat, 30)
    assert np.allclose(s, 0)
    ramp = np.tile(np.arange(10) * 30.0, (10, 1))  # rises 30 m per 30 m pixel -> 45 degrees
    s, _ = ix.slope_aspect(ramp, 30)
    assert np.allclose(s[:, 1:-1], 45)
