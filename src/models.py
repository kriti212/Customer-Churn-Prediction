"""Candidate models, pipeline factory and Optuna search spaces."""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.preprocess import FeatureEngineer, build_preprocessor

# Only the linear model needs scaled inputs; trees are scale-invariant.
NEEDS_SCALING = {"logreg": True, "random_forest": False, "xgboost": False, "lightgbm": False}
TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}


def make_estimator(name: str, params: dict | None = None, pos_weight: float = 4.0):
    """Create an estimator with imbalance handling built in.

    pos_weight = (#negatives / #positives), used by XGBoost, which has no class_weight.
    """
    params = dict(params or {})
    rs = config.RANDOM_STATE
    if name == "logreg":
        return LogisticRegression(class_weight="balanced", max_iter=5000, random_state=rs, **params)
    if name == "random_forest":
        defaults = dict(n_estimators=300, min_samples_leaf=5)
        return RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=rs, **{**defaults, **params})
    if name == "xgboost":
        defaults = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8)
        return XGBClassifier(
            scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=-1, random_state=rs, **{**defaults, **params}
        )
    if name == "lightgbm":
        defaults = dict(n_estimators=400, learning_rate=0.05, num_leaves=31)
        return LGBMClassifier(class_weight="balanced", random_state=rs, verbose=-1, n_jobs=-1, **{**defaults, **params})
    raise ValueError(f"Unknown model: {name}")


def make_pipeline(name: str, engineered: bool, params: dict | None = None, pos_weight: float = 4.0) -> Pipeline:
    """Feature engineering -> preprocessing -> model, as one object.

    Keeping everything in a single Pipeline guarantees the app applies exactly the
    same transformations as training and prevents leakage across CV folds.
    """
    prep = build_preprocessor(scale=NEEDS_SCALING[name], engineered=engineered).set_output(transform="pandas")
    return Pipeline(
        [
            ("features", FeatureEngineer(enabled=engineered)),
            ("prep", prep),
            ("model", make_estimator(name, params, pos_weight)),
        ]
    )


def suggest_params(name: str, trial) -> dict:
    """Optuna search space for each model."""
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 100, log=True)}
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    raise ValueError(f"Unknown model: {name}")
