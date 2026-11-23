"""Data loading, input validation, feature engineering and the preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


# --- Loading ---------------------------------------------------------------
def load_raw(path=config.RAW_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model inputs (raw feature columns only) and the target."""
    X = df[config.FEATURE_COLS].copy()
    y = df[config.TARGET].astype(int)
    return X, y


# --- Validation ------------------------------------------------------------
class InputValidationError(ValueError):
    """Raised when incoming data does not match the expected schema."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns, types and ranges. Returns a clean copy with only feature columns.

    Extra columns (e.g. CustomerId, Surname, Exited, Complain) are allowed and ignored,
    so users can upload the original file format. Unknown category values are
    reported as errors rather than silently scored.
    """
    errors: list[str] = []

    missing = [c for c in config.FEATURE_COLS if c not in df.columns]
    if missing:
        raise InputValidationError([f"Missing required column(s): {', '.join(missing)}"])
    if len(df) == 0:
        raise InputValidationError(["The file has no rows."])

    out = df[config.FEATURE_COLS].copy()

    for col in config.NUMERIC_COLS:
        converted = pd.to_numeric(out[col], errors="coerce")
        bad = converted.isna()
        if bad.any():
            rows = (out.index[bad][:5] + 1).tolist()
            errors.append(f"'{col}' has missing or non-numeric values (rows {rows}{'...' if bad.sum() > 5 else ''})")
            continue
        lo, hi = config.NUMERIC_RANGES[col]
        out_of_range = (converted < lo) | (converted > hi)
        if out_of_range.any():
            rows = (out.index[out_of_range][:5] + 1).tolist()
            errors.append(f"'{col}' must be between {lo} and {hi} (rows {rows}{'...' if out_of_range.sum() > 5 else ''})")
        out[col] = converted

    for col, allowed in config.CATEGORY_VALUES.items():
        values = out[col].astype(str).str.strip()
        bad = ~values.isin(allowed)
        if bad.any():
            examples = sorted(values[bad].unique().tolist())[:5]
            errors.append(f"'{col}' has unknown value(s) {examples}; allowed: {allowed}")
        out[col] = values

    if errors:
        raise InputValidationError(errors)
    return out


# --- Feature engineering ---------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds derived features. Stateless, so it is safe inside cross-validation.

    enabled=False passes data through unchanged, which lets experiments compare
    "raw features" against "engineered features" with the same pipeline.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if not self.enabled:
            return X
        X["ZeroBalance"] = (X["Balance"] == 0).astype(int)
        X["BalanceToSalary"] = X["Balance"] / (X["EstimatedSalary"] + 1.0)
        X["AgeXActive"] = X["Age"] * X["IsActiveMember"]
        # Churn jumps from 8% (2 products) to 83% (3) and 100% (4): a categorical
        # view lets the linear model capture that non-monotonic effect.
        X["ProductsCat"] = X["NumOfProducts"].clip(upper=3).astype(str)
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(config.FEATURE_COLS)
        if self.enabled:
            names += ["ZeroBalance", "BalanceToSalary", "AgeXActive", "ProductsCat"]
        return np.array(names, dtype=object)


ENGINEERED_NUMERIC = ["ZeroBalance", "BalanceToSalary", "AgeXActive"]
ENGINEERED_CATEGORICAL = ["ProductsCat"]


def build_preprocessor(scale: bool, engineered: bool) -> ColumnTransformer:
    """One-hot encode categoricals (unknown values ignored) and optionally scale numerics."""
    numeric = config.NUMERIC_COLS + (ENGINEERED_NUMERIC if engineered else [])
    categorical = config.CATEGORICAL_COLS + (ENGINEERED_CATEGORICAL if engineered else [])
    numeric_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric_step, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


def add_age_band(df: pd.DataFrame) -> pd.Series:
    """Age groups used only for the fairness audit and EDA, not as a model feature."""
    return pd.cut(df["Age"], bins=[0, 29, 39, 49, 59, 200], labels=["18-29", "30-39", "40-49", "50-59", "60+"])
