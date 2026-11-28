"""SHAP explanations, grouped back to the original (human-readable) features.

SHAP runs on the *uncalibrated* twin of the production model. Calibration is a
monotonic rescaling of the score, so the ranking of reasons is the same; the SHAP
values are in the model's raw score units (log-odds for boosting / logistic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src import config
from src.models import TREE_MODELS

# Transformed column -> group shown to the user.
ENGINEERED_LABELS = {
    "ZeroBalance": "Balance",
    "BalanceToSalary": "Balance-to-salary ratio",
    "AgeXActive": "Age × active member",
    "AgeXInactive": "Age × inactive member",
}


def group_name(col: str) -> str:
    if col in ENGINEERED_LABELS:
        return ENGINEERED_LABELS[col]
    if col.startswith("ProductsCat_"):
        return "NumOfProducts"
    for cat in config.CATEGORICAL_COLS:
        if col.startswith(cat + "_"):
            return cat
    return col


def make_explainer(pipe, model_name: str, background: pd.DataFrame):
    """background: raw feature rows (training sample) used by the linear explainer."""
    model = pipe.named_steps["model"]
    if model_name in TREE_MODELS:
        return shap.TreeExplainer(model)
    bg = pipe[:-1].transform(background)
    return shap.LinearExplainer(model, bg)


def shap_values(explainer, pipe, X_raw: pd.DataFrame) -> pd.DataFrame:
    """SHAP values grouped to original features. Rows align with X_raw."""
    Xt = pipe[:-1].transform(X_raw)
    vals = explainer.shap_values(Xt)
    if isinstance(vals, list):  # older API: one array per class
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:  # (rows, features, classes)
        vals = vals[:, :, 1]
    sv = pd.DataFrame(vals, columns=Xt.columns, index=X_raw.index)
    return sv.T.groupby(sv.columns.map(group_name)).sum().T


def top_reasons(grouped_row: pd.Series, raw_row: pd.Series, n: int = 5) -> list[dict]:
    """Largest absolute contributions for one customer."""
    order = grouped_row.abs().sort_values(ascending=False).index[:n]
    out = []
    for feat in order:
        value = raw_row.get(feat, None)
        out.append({
            "feature": feat,
            "value": None if value is None else (value.item() if hasattr(value, "item") else value),
            "impact": float(grouped_row[feat]),
            "direction": "raises risk" if grouped_row[feat] > 0 else "lowers risk",
        })
    return out
