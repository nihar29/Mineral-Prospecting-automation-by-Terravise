"""Stage 4: compare models under SPATIAL block cross-validation, log everything to MLflow,
refit the best model (by PR-AUC) on all data and register it as `@champion`.

Outputs: artifacts/model.joblib, artifacts/cv_results.csv, artifacts/oof_predictions.csv,
         artifacts/run_id.txt

Usage:  python -m src.models.train --config config/params.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.models.evaluate import classification_metrics
from src.models.pipeline import SEARCH_SPACES, make_pipeline
from src.utils.config import cli_config, get_logger

log = get_logger("stage4.train")


def spatial_cv(name, X, y, groups, cont, cat, n_folds, seed, params=None):
    """Return per-fold metrics and out-of-fold probabilities."""
    n_splits = min(n_folds, len(np.unique(groups)))
    oof = np.full(len(y), np.nan)
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits).split(X, y, groups)):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            log.warning("%s fold %d has a single class; skipped", name, k)
            continue
        pipe = make_pipeline(name, cont, cat, seed, dict(params or {}))
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
        folds.append({"model": name, "fold": k, **classification_metrics(y[te], oof[te])})
    return pd.DataFrame(folds), oof


def tune(name, X, y, groups, cont, cat, cfg) -> dict:
    import optuna

    def objective(trial):
        params = SEARCH_SPACES[name](trial)
        res, _ = spatial_cv(name, X, y, groups, cont, cat, cfg["cv"]["n_folds"],
                            cfg["project"]["seed"], params)
        return res["pr_auc"].mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=cfg["project"]["seed"]))
    study.optimize(objective, n_trials=cfg["optuna_trials"])
    log.info("%s best params: %s (PR-AUC %.3f)", name, study.best_params, study.best_value)
    return study.best_params


def main() -> None:
    import mlflow
    import mlflow.sklearn

    cfg = cli_config(__doc__)
    seed = cfg["project"]["seed"]
    proc, art = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["artifacts"])
    df = pd.read_parquet(proc / "train_table.parquet")
    sel = json.loads((art / "selected_features.json").read_text())
    cont, cat = sel["continuous"], sel["categorical"]
    X, y, groups = df[cont + cat], df["label"].values, df["block_id"].values

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    all_folds, oof_cols, best_params = [], {}, {}
    with mlflow.start_run(run_name="model_comparison") as parent:
        mlflow.log_params({"features": ",".join(cont + cat), "n_pos": int(y.sum()),
                           "n_neg": int((1 - y).sum()), "n_blocks": int(len(np.unique(groups))),
                           "block_size_m": cfg["cv"]["block_size_m"]})
        for name in cfg["models"]:
            params = tune(name, X, y, groups, cont, cat, cfg) \
                if cfg.get("optuna_trials", 0) > 0 and name in SEARCH_SPACES else {}
            best_params[name] = params
            with mlflow.start_run(run_name=name, nested=True):
                folds, oof = spatial_cv(name, X, y, groups, cont, cat, cfg["cv"]["n_folds"], seed, params)
                summary = folds.drop(columns=["model", "fold"]).agg(["mean", "std"])
                mlflow.log_params({f"p_{k}": v for k, v in params.items()})
                for metric in summary.columns:
                    mlflow.log_metric(f"{metric}_mean", summary.loc["mean", metric])
                    mlflow.log_metric(f"{metric}_std", summary.loc["std", metric])
                log.info("%-9s PR-AUC %.3f ± %.3f | ROC-AUC %.3f ± %.3f", name,
                         summary.loc["mean", "pr_auc"], summary.loc["std", "pr_auc"],
                         summary.loc["mean", "roc_auc"], summary.loc["std", "roc_auc"])
                all_folds.append(folds)
                oof_cols[name] = oof

        cv = pd.concat(all_folds, ignore_index=True)
        cv.to_csv(art / "cv_results.csv", index=False)
        pd.DataFrame({"x": df["x"], "y": df["y"], "label": y, "block_id": groups, **oof_cols}) \
            .to_csv(art / "oof_predictions.csv", index=False)

        best = cv.groupby("model")["pr_auc"].mean().idxmax()
        log.info("Best model by spatial-CV PR-AUC: %s", best)
        final = make_pipeline(best, cont, cat, seed, dict(best_params[best])).fit(X, y)
        joblib.dump({"pipeline": final, "features": cont + cat, "model": best,
                     "threshold": float(cv[cv.model == best]["threshold_youden"].median())},
                    art / "model.joblib")

        mlflow.log_param("best_model", best)
        mlflow.log_artifact(str(art / "cv_results.csv"))
        mlflow.log_artifact(str(art / "selected_features.json"))
        kwargs = dict(input_example=X.head(5).astype("float64"),
                      registered_model_name=cfg["mlflow"]["model_name"],
                      serialization_format="cloudpickle")
        try:
            info = mlflow.sklearn.log_model(final, name="model", **kwargs)          # MLflow 3.x
        except TypeError:
            info = mlflow.sklearn.log_model(final, artifact_path="model", **kwargs)  # MLflow 2.x
        try:
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(cfg["mlflow"]["model_name"], "challenger",
                                              info.registered_model_version)
            log.info("Registered version %s as @challenger; promote to @champion after review",
                     info.registered_model_version)
        except Exception as exc:  # registry not available on some backends
            log.warning("Model registry alias not set: %s", exc)

        (art / "run_id.txt").write_text(parent.info.run_id)


if __name__ == "__main__":
    main()
