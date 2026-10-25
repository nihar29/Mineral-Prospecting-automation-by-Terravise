"""Model factory: every model shares the same preprocessing inside one sklearn Pipeline,
so scaling / one-hot encoding are fitted on training folds only and saved with the model."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def make_preprocessor(continuous: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformers = [("num", num, continuous)]
    if categorical:
        transformers.append(("cat", cat, categorical))
    return ColumnTransformer(transformers, remainder="drop")


def make_estimator(name: str, seed: int, params: dict | None = None):
    params = params or {}
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", **params)
    if name == "rf":
        return RandomForestClassifier(n_estimators=params.pop("n_estimators", 300),
                                      class_weight="balanced", n_jobs=-1, random_state=seed, **params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=params.pop("n_estimators", 400), learning_rate=0.05,
                              class_weight="balanced", random_state=seed, verbose=-1, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.05, eval_metric="logloss",
                             random_state=seed, **params)
    if name == "svm":
        return SVC(probability=True, class_weight="balanced", random_state=seed, **params)
    raise ValueError(f"Unknown model '{name}'")


def make_pipeline(name, continuous, categorical, seed, params=None) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(continuous, categorical)),
                     ("clf", make_estimator(name, seed, params))])


SEARCH_SPACES = {
    "rf": lambda t: {"max_depth": t.suggest_int("max_depth", 3, 20),
                     "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20)},
    "lightgbm": lambda t: {"num_leaves": t.suggest_int("num_leaves", 8, 64),
                           "min_child_samples": t.suggest_int("min_child_samples", 5, 50),
                           "subsample": t.suggest_float("subsample", 0.5, 1.0)},
    "logreg": lambda t: {"C": t.suggest_float("C", 1e-3, 10, log=True)},
    "svm": lambda t: {"C": t.suggest_float("C", 1e-2, 100, log=True),
                      "gamma": t.suggest_float("gamma", 1e-4, 1, log=True)},
}
