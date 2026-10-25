"""Stage 2: align every layer to the master grid, compute indices, write one multi-band stack.

Output: data/processed/stack.tif (band descriptions = feature names)
        data/processed/valid_mask.tif
        reports/qa/band_stats.csv

Usage:  python -m src.preprocess.build_stack --config config/params.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.preprocess import indices as ix
from src.preprocess.qa import band_stats
from src.utils.config import cli_config, get_logger
from src.utils.io import align_to_master, make_master_profile, write_raster

log = get_logger("stage2.preprocess")


def landsat_layers(cfg, master) -> dict[str, np.ndarray]:
    ldir = Path(cfg["paths"]["raw"]) / "landsat"
    needed = ["blue", "red", "nir08", "swir16", "swir22"]
    if not all((ldir / f"{b}.tif").exists() for b in needed):
        log.warning("Landsat bands missing in %s; skipping spectral indices", ldir)
        return {}
    b = {name: ix.landsat_reflectance(align_to_master(ldir / f"{name}.tif", master))
         for name in needed}
    out = {}
    wanted = cfg["features"]["ratios"]
    v_ndvi = ix.ndvi(b["red"], b["nir08"])
    if "ndvi" in wanted:
        out["ndvi"] = v_ndvi
    if "iron_oxide" in wanted:
        out["iron_oxide"] = ix.iron_oxide(b["red"], b["blue"])
    if "clay_carbonate" in wanted:
        out["clay_carbonate"] = ix.clay_carbonate(b["swir16"], b["swir22"], b["nir08"])
    if "ferrous" in wanted:
        out["ferrous"] = ix.ferrous(b["swir16"], b["nir08"])

    # Vegetation mask: spectral ratios are unreliable under dense vegetation.
    veg = v_ndvi > cfg["features"]["ndvi_mask_threshold"]
    for k in ("iron_oxide", "clay_carbonate", "ferrous"):
        if k in out:
            out[k] = np.where(veg, np.nan, out[k])
    log.info("Vegetation-masked pixels: %.1f%%", 100 * np.nanmean(veg))
    if (ldir / "lwir11.tif").exists():
        out["thermal_b10"] = align_to_master(ldir / "lwir11.tif", master)
    return out


def dem_layers(cfg, master) -> dict[str, np.ndarray]:
    dem_dir = Path(cfg["paths"]["raw"]) / "dem"
    files = sorted(dem_dir.glob("*.tif"))
    if not files:
        log.warning("No DEM found in %s", dem_dir)
        return {}
    dem = align_to_master(files[0], master)
    out = {"elevation": dem}
    slope, aspect = ix.slope_aspect(dem, cfg["grid"]["res"])
    terrain = cfg["features"]["terrain"]
    if "slope" in terrain:
        out["slope"] = slope
    if "aspect" in terrain:
        out["aspect_sin"] = np.sin(np.radians(aspect))   # circular -> sin/cos
        out["aspect_cos"] = np.cos(np.radians(aspect))
    if "curvature" in terrain:
        out["curvature"] = ix.curvature(dem, cfg["grid"]["res"])
    return out


def sar_layers(cfg, master) -> dict[str, np.ndarray]:
    sdir = Path(cfg["paths"]["raw"]) / "sentinel1"
    out = {}
    for pol in ("vv", "vh"):
        f = sdir / f"{pol}.tif"
        if f.exists():
            lin = align_to_master(f, master)
            out[f"s1_{pol}_db"] = 10 * np.log10(np.where(lin > 0, lin, np.nan))
    return out


def manual_layers(cfg, master) -> tuple[dict[str, np.ndarray], list[str]]:
    out, categorical = {}, []
    for name, path in cfg["manual_layers"].get("continuous", {}).items():
        if Path(path).exists():
            out[name] = align_to_master(path, master, categorical=False)
    for name, path in cfg["manual_layers"].get("categorical", {}).items():
        if Path(path).exists():
            out[name] = align_to_master(path, master, categorical=True)
            categorical.append(name)
    return out, categorical


def main() -> None:
    cfg = cli_config(__doc__)
    g = cfg["grid"]
    master = make_master_profile(cfg["aoi"], g["crs"], g["res"], g["nodata"])
    log.info("Master grid: %s, %d x %d px @ %s m", g["crs"], master["width"], master["height"], g["res"])

    layers: dict[str, np.ndarray] = {}
    layers.update(landsat_layers(cfg, master))
    layers.update(dem_layers(cfg, master))
    layers.update(sar_layers(cfg, master))
    manual, categorical = manual_layers(cfg, master)
    layers.update(manual)
    if not layers:
        raise RuntimeError("No input layers found. Run Stage 1 or add manual layers.")

    names = list(layers)
    stack = np.stack([layers[n] for n in names]).astype(np.float32)
    # A pixel is usable if at least half its layers have data; remaining gaps
    # (e.g. vegetation-masked ratios) are imputed inside the model pipeline.
    valid = np.isfinite(stack).mean(axis=0) >= 0.5

    proc = Path(cfg["paths"]["processed"])
    write_raster(proc / "stack.tif", stack, master, descriptions=names)
    write_raster(proc / "valid_mask.tif", valid.astype(np.float32), master)
    with open(proc / "feature_types.json", "w") as f:
        json.dump({"features": names, "categorical": categorical}, f, indent=2)

    qa_dir = Path(cfg["paths"]["reports"]) / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    band_stats(stack, names).to_csv(qa_dir / "band_stats.csv", index=False)
    log.info("Stack written: %d bands, %.1f%% valid pixels", len(names), 100 * valid.mean())


if __name__ == "__main__":
    main()
