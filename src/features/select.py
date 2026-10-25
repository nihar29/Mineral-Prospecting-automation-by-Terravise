"""Stage 3b: feature selection.

Steps: drop sparse/constant features -> correlation filter (keep the more informative of a
pair) -> VIF filter on continuous features -> optional RFECV with SPATIAL folds.

Output: artifacts/selected_features.json, artifacts/feature_ranking.csv

Usage:  python -m src.features.select --config config/params.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

from src.utils.config import cli_config, get_logger

log = get_logger("stage3.select")


def drop_sparse_constant(df: pd.DataFrame, cols: list[str], max_nodata: float) -> list[str]:
    keep = [c for c in cols if df[c].isna().mean() <= max_nodata and df[c].nunique(dropna=True) > 1]
    log.info("Sparse/constant filter: %d -> %d", len(cols), len(keep))
    return keep


def mutual_info(df: pd.DataFrame, cols: list[str], y: np.ndarray, seed: int) -> pd.Series:
    X = SimpleImputer(strategy="median").fit_transform(df[cols])
    return pd.Series(mutual_info_classif(X, y, random_state=seed), index=cols)


def correlation_filter(df: pd.DataFrame, cols: list[str], mi: pd.Series, thr: float) -> list[str]:
    corr = df[cols].corr().abs()
    order = mi[cols].sort_values(ascending=False).index.tolist()
    keep: list[str] = []
    for c in order:
        if all(corr.loc[c, k] <= thr for k in keep):
            keep.append(c)
    log.info("Correlation filter (|r| > %.2f): %d -> %d", thr, len(cols), len(keep))
    return keep


def vif_filter(df: pd.DataFrame, cols: list[str], vif_max: float) -> list[str]:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm

    cols = list(cols)
    X = df[cols].fillna(df[cols].median())
    while len(cols) > 2:
        Xc = sm.add_constant(X[cols], has_constant="add").values
        vifs = pd.Series([variance_inflation_factor(Xc, i + 1) for i in range(len(cols))], index=cols)
        worst = vifs.idxmax()
        if vifs[worst] <= vif_max:
            break
        log.info("  VIF drop %s (%.1f)", worst, vifs[worst])
        cols.remove(worst)
    return cols


def rfecv_select(df, cols, y, groups, n_folds, seed) -> list[str]:
    X = SimpleImputer(strategy="median").fit_transform(df[cols])
    n_splits = min(n_folds, len(np.unique(groups)))
    sel = RFECV(
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1),
        step=1, cv=GroupKFold(n_splits), scoring="average_precision", min_features_to_select=2,
    ).fit(X, y, groups=groups)
    keep = [c for c, s in zip(cols, sel.support_) if s]
    log.info("RFECV (spatial folds): %d -> %d", len(cols), len(keep))
    return keep


def main() -> None:
    cfg = cli_config(__doc__)
    fcfg, seed = cfg["features"], cfg["project"]["seed"]
    proc, art = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["artifacts"])
    df = pd.read_parquet(proc / "train_table.parquet")
    types = json.loads((proc / "feature_types.json").read_text())
    categorical = [c for c in types["categorical"] if c in df.columns]
    continuous = [c for c in types["features"] if c not in categorical]
    y, groups = df["label"].values, df["block_id"].values

    cont = drop_sparse_constant(df, continuous, fcfg["max_nodata_frac"])
    mi = mutual_info(df, cont, y, seed)
    cont = correlation_filter(df, cont, mi, fcfg["corr_threshold"])
    cont = vif_filter(df, cont, fcfg["vif_max"])
    if fcfg.get("rfecv") and len(cont) > 2:
        cont = rfecv_select(df, cont, y, groups, cfg["cv"]["n_folds"], seed)

    selected = {"continuous": cont, "categorical": categorical}
    (art / "selected_features.json").write_text(json.dumps(selected, indent=2))
    mi.sort_values(ascending=False).rename("mutual_info").to_csv(art / "feature_ranking.csv")
    log.info("Selected: %s", selected)


if __name__ == "__main__":
    main()
