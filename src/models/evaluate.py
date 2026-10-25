"""Metrics and plots for prospectivity models."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             roc_auc_score, roc_curve)


def youden_threshold(y_true, y_prob) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def classification_metrics(y_true, y_prob) -> dict[str, float]:
    thr = youden_threshold(y_true, y_prob)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "f1_youden": f1_score(y_true, (y_prob >= thr).astype(int)),
        "threshold_youden": thr,
    }


def success_rate_curve(prob_map: np.ndarray, deposit_mask: np.ndarray, n: int = 100):
    """% of deposits captured vs % of area flagged, ranking pixels by probability.

    Returns (area_fraction, capture_fraction) arrays of length n+1.
    """
    valid = np.isfinite(prob_map)
    p = prob_map[valid]
    d = deposit_mask[valid].astype(bool)
    order = np.argsort(-p)
    captured = np.cumsum(d[order]) / max(d.sum(), 1)
    area = np.arange(1, p.size + 1) / p.size
    grid = np.linspace(0, 1, n + 1)
    return grid, np.interp(grid, np.concatenate([[0], area]), np.concatenate([[0], captured]))


def area_for_capture(area, capture, target: float = 0.8) -> float:
    """Fraction of the AOI that must be flagged to capture `target` of known deposits."""
    idx = np.searchsorted(capture, target)
    return float(area[min(idx, len(area) - 1)])
