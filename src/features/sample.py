"""Stage 3a: build the point-based training table.

Positives = known deposits (+ optional pixel buffer). Negatives = random valid pixels at
least `neg_buffer_m` from any deposit. Each row gets a spatial `block_id` used for
leakage-free cross-validation.

Training on every pixel of a large "ore" polygon (as in the original notebook) makes the
model learn the polygon's footprint and inflates accuracy to ~1.0; this avoids that.

Usage:  python -m src.features.sample --config config/params.yaml
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import rowcol, xy
from scipy.ndimage import binary_dilation, distance_transform_edt

from src.utils.config import cli_config, get_logger
from src.utils.io import read_stack

log = get_logger("stage3.sample")


def load_positives(path: str, crs) -> gpd.GeoDataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                               crs=4326)
    else:
        gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"No deposits in {path}; add known occurrences before training.")
    return gdf.to_crs(crs)


def assign_blocks(x: np.ndarray, y: np.ndarray, block_size: float) -> np.ndarray:
    bx = np.floor(x / block_size).astype(np.int64)
    by = np.floor(y / block_size).astype(np.int64)
    _, block_id = np.unique(np.stack([bx, by], axis=1), axis=0, return_inverse=True)
    return block_id.ravel()


def build_training_table(stack, names, profile, positives, cfg) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["project"]["seed"])
    h, w = stack.shape[1:]
    valid = np.isfinite(stack).mean(axis=0) >= 0.5
    res = cfg["grid"]["res"]
    lab = cfg["labels"]

    pos_mask = np.zeros((h, w), dtype=bool)
    rows, cols = rowcol(profile["transform"], positives.geometry.x.values, positives.geometry.y.values)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    log.info("Deposits inside AOI: %d / %d", inside.sum(), len(positives))
    pos_mask[rows[inside], cols[inside]] = True
    if lab["pos_buffer_px"] > 0:
        pos_mask = binary_dilation(pos_mask, iterations=lab["pos_buffer_px"])
    pos_mask &= valid
    if pos_mask.sum() == 0:
        raise ValueError("No deposit falls on a valid pixel of the stack.")

    dist_m = distance_transform_edt(~pos_mask) * res
    cand = np.flatnonzero(valid & (dist_m >= lab["neg_buffer_m"]))
    n_neg = min(len(cand), int(pos_mask.sum() * lab["neg_ratio"]))
    neg_idx = rng.choice(cand, size=n_neg, replace=False)

    pos_idx = np.flatnonzero(pos_mask)
    idx = np.concatenate([pos_idx, neg_idx])
    label = np.concatenate([np.ones(len(pos_idx), int), np.zeros(len(neg_idx), int)])
    r, c = np.unravel_index(idx, (h, w))
    xs, ys = xy(profile["transform"], r, c)

    df = pd.DataFrame(stack[:, r, c].T, columns=names)
    df["x"], df["y"], df["label"] = np.asarray(xs), np.asarray(ys), label
    df["block_id"] = assign_blocks(df["x"].values, df["y"].values, cfg["cv"]["block_size_m"])
    log.info("Training table: %d positives, %d negatives, %d spatial blocks",
             len(pos_idx), len(neg_idx), df["block_id"].nunique())
    return df


def main() -> None:
    cfg = cli_config(__doc__)
    proc = Path(cfg["paths"]["processed"])
    stack, names, profile = read_stack(proc / "stack.tif")
    positives = load_positives(cfg["labels"]["positives"], profile["crs"])
    df = build_training_table(stack, names, profile, positives, cfg)
    df.to_parquet(proc / "train_table.parquet", index=False)


if __name__ == "__main__":
    main()
