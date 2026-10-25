"""Data-quality summaries for each band of the stack."""
from __future__ import annotations

import numpy as np
import pandas as pd


def band_stats(stack: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = []
    for name, band in zip(names, stack):
        finite = band[np.isfinite(band)]
        rows.append({
            "band": name,
            "pct_nodata": round(100 * (1 - finite.size / band.size), 2),
            "min": float(finite.min()) if finite.size else np.nan,
            "max": float(finite.max()) if finite.size else np.nan,
            "mean": float(finite.mean()) if finite.size else np.nan,
            "std": float(finite.std()) if finite.size else np.nan,
        })
    return pd.DataFrame(rows)
