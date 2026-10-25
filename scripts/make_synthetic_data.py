"""Create a small synthetic dataset and config/params_demo.yaml so the whole pipeline can be
run end to end without downloads:

    python scripts/make_synthetic_data.py
    snakemake --cores 4 --config cfg=config/params_demo.yaml

The fake 'deposits' are drawn where a hidden combination of gravity, magnetics, slope and
one geology unit is favourable, so a working pipeline should recover a clear signal.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.transform import from_origin, xy
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
RES, CRS, SEED = 100, "EPSG:32735", 7


def smooth_field(rng, shape, sigma):
    f = gaussian_filter(rng.normal(size=shape), sigma)
    return (f - f.mean()) / f.std()


def write(path: Path, arr: np.ndarray, transform, nodata=-9999.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=CRS, transform=transform, nodata=nodata,
                       compress="deflate") as dst:
        dst.write(arr.astype(np.float32), 1)


def main() -> None:
    rng = np.random.default_rng(SEED)
    base = yaml.safe_load((ROOT / "config" / "params.yaml").read_text())
    aoi = gpd.read_file(ROOT / base["aoi"]).to_crs(CRS)
    minx, miny, maxx, maxy = aoi.total_bounds
    minx, maxy = np.floor(minx / RES) * RES, np.ceil(maxy / RES) * RES
    w, h = int(np.ceil((maxx - minx) / RES)), int(np.ceil((maxy - miny) / RES))
    tr = from_origin(minx, maxy, RES, RES)

    gravity = smooth_field(rng, (h, w), 25) * 8 + 20
    magnetic = smooth_field(rng, (h, w), 12) * 150 + 30000
    dem = smooth_field(rng, (h, w), 30) * 60 + 1250
    geology = np.digitize(smooth_field(rng, (h, w), 20), [-0.8, 0.0, 0.7]).astype(np.float32) + 1

    slope = np.degrees(np.arctan(np.hypot(*np.gradient(dem, RES))))
    score = (0.9 * (gravity - 20) / 8 + 0.7 * (magnetic - 30000) / 150
             - 0.4 * (slope - slope.mean()) / slope.std() + 1.5 * (geology == 3) - 3.5)
    prob = 1 / (1 + np.exp(-score))
    idx = rng.choice(prob.size, size=60, replace=False, p=prob.ravel() / prob.sum())
    r, c = np.unravel_index(idx, prob.shape)
    xs, ys = xy(tr, r, c)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=CRS).to_crs(4326)
    labels = pd.DataFrame({"Name": [f"demo_{i}" for i in range(len(pts))],
                           "Latitude": pts.geometry.y, "Longitude": pts.geometry.x})

    raw = DEMO / "raw"
    write(raw / "geophysics" / "gravity.tif", gravity, tr)
    write(raw / "geophysics" / "magnetic.tif", magnetic, tr)
    write(raw / "dem" / "dem.tif", dem, tr)
    write(raw / "geology" / "geology.tif", geology, tr)
    (raw / "labels").mkdir(parents=True, exist_ok=True)
    labels.to_csv(raw / "labels" / "copper_occurrences.csv", index=False)

    demo = dict(base)
    demo["grid"] = {**base["grid"], "res": RES}
    demo["paths"] = {"raw": "data/demo/raw", "interim": "data/demo/interim",
                     "processed": "data/demo/processed", "artifacts": "data/demo/artifacts",
                     "outputs": "data/demo/outputs", "reports": "data/demo/reports"}
    demo["sources"] = {k: {**v, "enabled": False} for k, v in base["sources"].items()}
    demo["manual_layers"] = {
        "continuous": {"gravity": "data/demo/raw/geophysics/gravity.tif",
                       "magnetic": "data/demo/raw/geophysics/magnetic.tif"},
        "categorical": {"geology": "data/demo/raw/geology/geology.tif"},
    }
    demo["labels"] = {**base["labels"], "positives": "data/demo/raw/labels/copper_occurrences.csv",
                      "neg_buffer_m": 1000}
    demo["cv"] = {**base["cv"], "block_size_m": 5000}
    demo["mlflow"] = {**base["mlflow"], "tracking_uri": "sqlite:///data/demo/mlflow.db",
                      "experiment": "demo"}
    (ROOT / "config" / "params_demo.yaml").write_text(yaml.safe_dump(demo, sort_keys=False))
    print(f"Synthetic data in {DEMO} ({w}x{h} px, {len(labels)} deposits); config/params_demo.yaml written")


if __name__ == "__main__":
    main()
