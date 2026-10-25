"""Stage 7: batch prediction with the trained model, window by window.

Outputs (in outputs/):
  probability.tif   copper prospectivity probability (0-1)
  uncertainty.tif   std of tree predictions (tree models only)
  classes.tif       1 Low, 2 Moderate, 3 High, 4 Very high (probability quantiles)
  success_rate.csv  % deposits captured vs % area flagged
  map.html          interactive web map

Usage:
  python -m src.predict --config config/params.yaml
  python -m src.predict --config config/params.yaml --stack path/to/new_area_stack.tif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from src.models.evaluate import area_for_capture, success_rate_curve
from src.utils.config import get_logger, load_config

log = get_logger("stage7.predict")


def tree_std(pipeline, X: pd.DataFrame) -> np.ndarray | None:
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "estimators_"):
        return None
    Xt = pipeline.named_steps["pre"].transform(X)
    preds = np.stack([t.predict_proba(Xt)[:, 1] for t in clf.estimators_])
    return preds.std(axis=0)


def predict_stack(stack_path: Path, bundle: dict, out_dir: Path, window: int) -> tuple[Path, dict]:
    pipe, features = bundle["pipeline"], bundle["features"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(stack_path) as src:
        names = list(src.descriptions)
        missing = [f for f in features if f not in names]
        if missing:
            raise ValueError(f"Stack lacks features used in training: {missing}")
        band_idx = [names.index(f) + 1 for f in features]
        prof = src.profile.copy()
        prof.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")
        prob_p, unc_p = out_dir / "probability.tif", out_dir / "uncertainty.tif"
        with rasterio.open(prob_p, "w", **prof) as dp, rasterio.open(unc_p, "w", **prof) as du:
            for r0 in range(0, src.height, window):
                for c0 in range(0, src.width, window):
                    win = Window(c0, r0, min(window, src.width - c0), min(window, src.height - r0))
                    arr = src.read(band_idx, window=win).astype(np.float32)
                    if src.nodata is not None:
                        arr[arr == src.nodata] = np.nan
                    h, w = arr.shape[1:]
                    flat = arr.reshape(len(features), -1).T
                    ok = np.isfinite(flat).mean(axis=1) >= 0.5
                    prob = np.full(h * w, -9999.0, np.float32)
                    unc = np.full(h * w, -9999.0, np.float32)
                    if ok.any():
                        X = pd.DataFrame(flat[ok], columns=features)
                        prob[ok] = pipe.predict_proba(X)[:, 1]
                        s = tree_std(pipe, X)
                        if s is not None:
                            unc[ok] = s
                    dp.write(prob.reshape(h, w), 1, window=win)
                    du.write(unc.reshape(h, w), 1, window=win)
    return prob_p, prof


def classify(prob_path: Path, out_dir: Path, cuts: dict) -> np.ndarray:
    with rasterio.open(prob_path) as src:
        p = src.read(1).astype(np.float32)
        prof = src.profile.copy()
    p[p == -9999.0] = np.nan
    q = np.nanquantile(p, [cuts["low"], cuts["moderate"], cuts["high"]])
    cls = np.digitize(p, q).astype(np.float32) + 1
    cls[np.isnan(p)] = -9999.0
    with rasterio.open(out_dir / "classes.tif", "w", **prof) as dst:
        dst.write(cls, 1)
    return p


def web_map(prob: np.ndarray, prob_path: Path, out_html: Path, deposits=None) -> None:
    import folium
    import matplotlib
    from rasterio.warp import transform_bounds

    with rasterio.open(prob_path) as src:
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    rgba = matplotlib.colormaps["YlOrRd"](np.nan_to_num(prob, nan=0.0))
    rgba[..., 3] = np.isfinite(prob).astype(float) * 0.75
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=10, tiles="OpenStreetMap")
    folium.raster_layers.ImageOverlay((rgba * 255).astype(np.uint8), bounds=[[s, w], [n, e]],
                                      name="Copper probability").add_to(m)
    if deposits is not None:
        fg = folium.FeatureGroup(name="Known deposits")
        for _, r in deposits.iterrows():
            folium.CircleMarker([r.geometry.y, r.geometry.x], radius=4, color="black",
                                fill=True, tooltip=str(r.get("name", ""))).add_to(fg)
        fg.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(out_html)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/params.yaml")
    ap.add_argument("--stack", default=None, help="stack to predict (default: training stack)")
    ap.add_argument("--out", default=None, help="output folder (default: paths.outputs)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    bundle = joblib.load(Path(cfg["paths"]["artifacts"]) / "model.joblib")
    stack = Path(args.stack or Path(cfg["paths"]["processed"]) / "stack.tif")
    out_dir = Path(args.out or cfg["paths"]["outputs"])
    log.info("Predicting %s with %s", stack, bundle["model"])

    prob_path, _ = predict_stack(stack, bundle, out_dir, cfg["predict"]["window"])
    prob = classify(prob_path, out_dir, cfg["predict"]["classes"])

    deposits = None
    summary = {"model": bundle["model"]}
    lab = Path(cfg["labels"]["positives"])
    if lab.exists():
        from rasterio.transform import rowcol
        from src.features.sample import load_positives
        with rasterio.open(prob_path) as src:
            deposits = load_positives(str(lab), src.crs)
            rr, cc = rowcol(src.transform, deposits.geometry.x.values, deposits.geometry.y.values)
        rr, cc = np.asarray(rr), np.asarray(cc)
        mask = np.zeros(prob.shape, bool)
        inside = (rr >= 0) & (rr < prob.shape[0]) & (cc >= 0) & (cc < prob.shape[1])
        mask[rr[inside], cc[inside]] = True
        if mask.any():
            area, capture = success_rate_curve(prob, mask)
            pd.DataFrame({"area_fraction": area, "capture_fraction": capture}) \
                .to_csv(out_dir / "success_rate.csv", index=False)
            summary["area_for_80pct_capture"] = area_for_capture(area, capture, 0.8)
        deposits = deposits.to_crs(4326)

    (out_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2))
    web_map(prob, prob_path, out_dir / "map.html", deposits)
    log.info("Prediction complete: %s", summary)


if __name__ == "__main__":
    main()
