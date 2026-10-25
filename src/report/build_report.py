"""Stage 8: build a self-contained HTML report from the run's artifacts.

Every number, table and figure is read from files produced by earlier stages, so the
report is regenerated on each run and never edited by hand.

Usage:  python -m src.report.build_report --config config/params.yaml
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from jinja2 import Template  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from src.utils.config import cli_config, get_logger  # noqa: E402

log = get_logger("stage8.report")


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "n/a"


def curves_figure(oof: pd.DataFrame, models: list[str]) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    y = oof["label"].values
    for m in models:
        ok = oof[m].notna()
        fpr, tpr, _ = roc_curve(y[ok], oof.loc[ok, m])
        prec, rec, _ = precision_recall_curve(y[ok], oof.loc[ok, m])
        a1.plot(fpr, tpr, label=m)
        a2.plot(rec, prec, label=m)
    a1.plot([0, 1], [0, 1], "--", color="grey")
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (spatial CV)")
    a2.axhline(y.mean(), ls="--", color="grey")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (spatial CV)")
    a1.legend(); a2.legend()
    return fig_to_b64(fig)


def raster_figure(path: Path, title: str, cmap: str) -> str | None:
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        a = src.read(1).astype(float)
        a[a == src.nodata] = np.nan
    if np.all(np.isnan(a)):
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(a, cmap=cmap)
    ax.set_title(title); ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig_to_b64(fig)


def success_figure(path: Path) -> str | None:
    if not path.exists():
        return None
    sr = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(100 * sr.area_fraction, 100 * sr.capture_fraction, lw=2, label="model")
    ax.plot([0, 100], [0, 100], "--", color="grey", label="random")
    ax.set(xlabel="% of area flagged", ylabel="% of known deposits captured", title="Success-rate curve")
    ax.legend()
    return fig_to_b64(fig)


TEMPLATE = Template("""<!doctype html><html><head><meta charset="utf-8">
<title>Mineral Prospecting Report</title>
<style>body{font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:auto;padding:24px;color:#222}
h1{margin-bottom:0}table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}
th{background:#f2f2f2}img{max-width:100%}.muted{color:#666}</style></head><body>
<h1>Copper Prospectivity Report</h1><p class="muted">Mineral Prospecting Automation by Terravise ·
generated {{ now }} · commit {{ commit }} · MLflow run {{ run_id }}</p>
<h2>1. Data inventory and quality</h2>{{ band_stats }}
<h2>2. Feature selection</h2><p>Selected continuous: {{ sel.continuous|join(', ') }}<br>
Selected categorical: {{ sel.categorical|join(', ') or 'none' }}</p>{{ ranking }}
<h2>3. Model comparison (spatial block CV)</h2>{{ cv_table }}
<p>Best model by PR-AUC: <b>{{ summary.model }}</b>.
{% if summary.area_for_80pct_capture %}Flagging {{ '%.1f'|format(100*summary.area_for_80pct_capture) }}% of the area captures 80% of known deposits.{% endif %}</p>
<img src="data:image/png;base64,{{ curves }}">
{% if success %}<h2>4. Success-rate curve</h2><img src="data:image/png;base64,{{ success }}">{% endif %}
<h2>5. Maps</h2>
{% if prob_img %}<img src="data:image/png;base64,{{ prob_img }}">{% endif %}
{% if unc_img %}<img src="data:image/png;base64,{{ unc_img }}">{% endif %}
<p>Interactive map: <code>outputs/map.html</code>. GeoTIFFs: <code>outputs/probability.tif</code>,
<code>outputs/classes.tif</code>, <code>outputs/uncertainty.tif</code>.</p>
<h2>6. Limitations</h2><ul><li>Scores depend on the completeness of the deposit database.</li>
<li>Spectral ratios are masked under dense vegetation (NDVI above threshold).</li>
<li>Predictions outside the training feature range are less reliable.</li></ul>
</body></html>""")


def main() -> None:
    cfg = cli_config(__doc__)
    art, out = Path(cfg["paths"]["artifacts"]), Path(cfg["paths"]["outputs"])
    rep = Path(cfg["paths"]["reports"])

    cv = pd.read_csv(art / "cv_results.csv")
    cv_table = cv.drop(columns="fold").groupby("model").agg(["mean", "std"]).round(3)
    oof = pd.read_csv(art / "oof_predictions.csv")
    models = [m for m in cfg["models"] if m in oof.columns]
    summary = json.loads((out / "prediction_summary.json").read_text())
    bs = rep / "qa" / "band_stats.csv"

    html = TEMPLATE.render(
        now=datetime.now().strftime("%Y-%m-%d %H:%M"), commit=git_commit(),
        run_id=(art / "run_id.txt").read_text().strip() if (art / "run_id.txt").exists() else "n/a",
        band_stats=pd.read_csv(bs).round(3).to_html(index=False) if bs.exists() else "",
        sel=json.loads((art / "selected_features.json").read_text()),
        ranking=pd.read_csv(art / "feature_ranking.csv").round(4).to_html(index=False),
        cv_table=cv_table.to_html(), summary=summary, curves=curves_figure(oof, models),
        success=success_figure(out / "success_rate.csv"),
        prob_img=raster_figure(out / "probability.tif", "Copper probability", "YlOrRd"),
        unc_img=raster_figure(out / "uncertainty.tif", "Uncertainty (tree std)", "viridis"),
    )
    (rep / "report.html").write_text(html, encoding="utf-8")
    log.info("Report written: %s", rep / "report.html")


if __name__ == "__main__":
    main()
