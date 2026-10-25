# Mineral Prospecting Automation by Terravise

An automated, reproducible machine-learning pipeline for **copper prospectivity mapping** from
multi-sensor remote sensing, geophysical and geological data. Developed for the Zambian
Copperbelt, but every setting (AOI, grid, data sources, models) lives in one config file, so it
transfers to other areas and commodities.

One command takes raw data to a versioned model, probability maps and a finished report:

```bash
snakemake --cores 4
```

---

## Pipeline

```
config/params.yaml ──► 1 Download ─► 2 Preprocess ─► 3 Features ─► 4 Train + tune
                                                                        │
          outputs/ ◄── 8 Report ◄── 7 Predict ◄── 6 Evaluate ◄──────────┘
                     5 Version control underneath: Git (code) · DVC (data) · MLflow (runs, models)
```

| Stage | Module | What it does | Main output |
| --- | --- | --- | --- |
| 1 Download | `src/download/` | Landsat 8/9, Copernicus DEM, Sentinel-1 (Planetary Computer STAC), ASTER (NASA Earthdata); logs URL + SHA-256 to a manifest | `data/raw/` |
| 2 Preprocess | `src/preprocess/` | Aligns every layer to one master grid (CRS, extent, pixel), Landsat reflectance, band ratios, NDVI vegetation mask, terrain, SAR dB, QA stats | `data/processed/stack.tif` |
| 3 Features | `src/features/` | Point-based training table (deposits + buffered background), spatial blocks; sparse/constant filter, correlation filter, VIF, RFECV with spatial folds | `artifacts/selected_features.json` |
| 4 Train | `src/models/train.py` | Logistic regression, Random Forest, LightGBM (+ XGBoost, SVM) under **spatial block CV**; optional Optuna tuning; best model by PR-AUC | `artifacts/model.joblib` |
| 5 Version | Git, DVC, MLflow | Code history, data hashes, experiment tracking, model registry (`@challenger` / `@champion`) | `mlflow.db` |
| 6 Evaluate | `src/models/evaluate.py` | ROC-AUC, PR-AUC, Brier, Youden threshold, success-rate curve, area needed for 80% deposit capture | `artifacts/cv_results.csv` |
| 7 Predict | `src/predict.py` | Windowed batch prediction on any stack; probability, uncertainty, 4-class map, interactive web map | `outputs/*.tif`, `outputs/map.html` |
| 8 Report | `src/report/build_report.py` | Self-contained HTML report built entirely from run artifacts | `reports/report.html` |

### Spectral indices implemented

| Sensor | Index | Formula |
| --- | --- | --- |
| Landsat 8/9 | Iron oxide | B4 / B2 |
| Landsat 8/9 | Clay + carbonate | (B6 + B7) / B5 |
| Landsat 8/9 | Ferrous minerals | B6 / B5 |
| Landsat 8/9 | NDVI (vegetation mask) | (B5 − B4) / (B5 + B4) |
| ASTER | Ferric iron, ferrous iron, clay, carbonate, silica | B2/B1, B4/B3, (B5+B7)/B6, B13/B10, B9/B6 (`indices.ASTER_RATIOS`) |

---

## Why spatial cross-validation matters

Random pixel-level train/test splits put neighbouring pixels of the same deposit in both sets,
which inflates accuracy and AUC to ~1.000 without any real predictive skill. This pipeline
instead:

1. trains on **deposit points plus sampled background** (not every pixel of an ore polygon), and
2. validates on **spatially separate blocks** (`GroupKFold` on `block_id`).

Expect lower, but honest, scores — they estimate performance in unexplored ground.

---

## Repository structure

```
├── config/
│   ├── params.yaml              # every tunable setting
│   └── aoi_example.geojson      # replace with your study area
├── data/                        # tracked by DVC, not Git
│   ├── raw/                     # landsat/, dem/, sentinel1/, aster/, geophysics/, geology/, labels/
│   ├── interim/
│   └── processed/               # stack.tif, train_table.parquet
├── src/
│   ├── download/                # fetch_stac.py, fetch_aster.py, run_all.py
│   ├── preprocess/              # indices.py, build_stack.py, qa.py
│   ├── features/                # sample.py, select.py
│   ├── models/                  # pipeline.py, train.py, evaluate.py
│   ├── report/                  # build_report.py
│   ├── utils/                   # config.py, io.py
│   └── predict.py
├── scripts/make_synthetic_data.py   # demo data for a full test run
├── tests/                       # pytest unit tests
├── artifacts/  outputs/  reports/   # generated
├── notebooks/                   # exploration only
├── Snakefile                    # pipeline DAG
├── environment.yml  requirements.txt  pyproject.toml
├── Dockerfile
└── .github/workflows/ci.yml     # lint + tests + smoke run on every push
```

---

## Getting started

### 1. Environment

```bash
conda env create -f environment.yml      # recommended (GDAL on Windows)
conda activate terravise-prospecting
# or: pip install -r requirements.txt
```

### 2. Try the demo (no downloads needed)

```bash
python scripts/make_synthetic_data.py
snakemake --cores 4 --config cfg=config/params_demo.yaml
# open data/demo/reports/report.html and data/demo/outputs/map.html
```

### 3. Run on real data

1. Put your AOI polygon in `config/` and set `aoi:` and `grid.crs` in `config/params.yaml`.
2. Add known deposits to `data/raw/labels/copper_occurrences.csv`
   (columns `Name, Latitude, Longitude`; see the template file).
3. Copy your own layers to the paths under `manual_layers` (gravity, magnetic, geology, ...).
4. Enable the remote-sensing sources you want under `sources:`
   (ASTER needs a free [NASA Earthdata](https://urs.earthdata.nasa.gov/) login).
5. Run:

```bash
snakemake --cores 4
```

Snakemake re-runs only the stages whose inputs changed — edit the feature settings and only
Stages 3–8 run again.

### 4. Predict a new area

```bash
python -m src.predict --stack path/to/new_area/stack.tif --out outputs/new_area
```

The new stack must contain the same feature bands as the training stack (build it with Stage 2
and a different AOI).

---

## Version control workflow

```bash
git init && dvc init                       # once
dvc remote add -d storage gdrive://<folder-id>   # or s3://, a NAS path, ...
dvc add data/raw && git add data/raw.dvc

git checkout -b exp/add-aster-ratios       # one branch per experiment
snakemake --cores 4
git commit -am "Add ASTER clay and silica ratios" && dvc push && git push
mlflow ui --backend-store-uri sqlite:///mlflow.db   # compare runs at http://127.0.0.1:5000
```

**Promotion rule:** after review, move the alias `@champion` to a new model version only if its
spatial-CV PR-AUC beats the current champion and its map passes a geological sanity check.

---

## Technology stack

| Purpose | Libraries |
| --- | --- |
| Data access | pystac-client, planetary-computer, earthaccess |
| Raster / vector | rasterio, rioxarray, xarray, geopandas, shapely, scipy |
| Machine learning | scikit-learn, LightGBM, XGBoost, statsmodels, Optuna, SHAP |
| Tracking and versioning | Git, DVC, MLflow |
| Orchestration | Snakemake (Prefect for scheduling, optional) |
| Visualisation and reports | matplotlib, folium, Jinja2 |
| Quality | pytest, ruff, GitHub Actions |
| Deployment | Docker |

---

## Configuration reference

All parameters are documented inline in [`config/params.yaml`](config/params.yaml). The most
important ones:

| Key | Meaning |
| --- | --- |
| `grid.crs`, `grid.res` | Master grid every layer is aligned to |
| `features.ndvi_mask_threshold` | NDVI above which spectral ratios are masked |
| `features.corr_threshold`, `features.vif_max` | Redundancy filters |
| `labels.neg_ratio`, `labels.neg_buffer_m` | Background sampling |
| `cv.block_size_m` | Spatial CV block size; set larger than the autocorrelation range |
| `models` | Models to compare |
| `optuna_trials` | `> 0` enables hyperparameter tuning |

---

## Roadmap

- [x] Config-driven pipeline, spatial CV, MLflow tracking, HTML report
- [ ] Sentinel-1 terrain correction with SNAP / pyroSAR, ALOS PALSAR mosaics
- [ ] Structural features: lineament density, distance to faults, magnetic derivatives
- [ ] SHAP plots and calibration in the report
- [ ] FastAPI service and scheduled re-runs

---

## Author

**Nihar Ranjan Sahoo** — Terravise · Department of Geology, Ravenshaw University, Cuttack

## License

Proprietary — all rights reserved. See [LICENSE](LICENSE). Use, copying or distribution requires
written permission from the author.
