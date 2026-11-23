"""Run from the project root:  pytest -q"""
import json

import pandas as pd
import pytest

from src import config
from src.models import make_pipeline
from src.preprocess import FeatureEngineer, InputValidationError, load_raw, validate_input

GOOD_ROW = {
    "CreditScore": 650, "Geography": "Germany", "Gender": "Female", "Age": 45, "Tenure": 3,
    "Balance": 120000.0, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 0,
    "EstimatedSalary": 90000.0, "Satisfaction Score": 3, "Card Type": "GOLD", "Point Earned": 500,
}

model_exists = pytest.mark.skipif(not config.MODEL_FILE.exists(), reason="model not trained yet")


# --- Leakage guards ----------------------------------------------------------
def test_complain_is_never_a_feature():
    assert "Complain" not in config.FEATURE_COLS
    assert config.TARGET not in config.FEATURE_COLS
    assert not set(config.ID_COLS) & set(config.FEATURE_COLS)


@model_exists
def test_saved_model_does_not_use_leaky_columns():
    meta = json.loads(config.METADATA_FILE.read_text())
    assert "Complain" not in meta["feature_columns"]
    assert "Complain" in meta["dropped_columns"]


def test_pipeline_ignores_extra_columns_like_complain():
    raw = load_raw().head(300)
    pipe = make_pipeline("logreg", engineered=True)
    X = validate_input(raw)  # raw file still contains Complain / Exited
    assert "Complain" not in X.columns and "Exited" not in X.columns
    pipe.fit(X, raw[config.TARGET])
    assert not any("Complain" in c for c in pipe[:-1].transform(X).columns)


# --- Validation ----------------------------------------------------------------
def test_valid_row_passes():
    out = validate_input(pd.DataFrame([GOOD_ROW]))
    assert list(out.columns) == config.FEATURE_COLS


def test_missing_column_rejected():
    df = pd.DataFrame([GOOD_ROW]).drop(columns=["Age"])
    with pytest.raises(InputValidationError, match="Age"):
        validate_input(df)


def test_unknown_category_rejected():
    df = pd.DataFrame([{**GOOD_ROW, "Geography": "Italy"}])
    with pytest.raises(InputValidationError, match="Italy"):
        validate_input(df)


def test_non_numeric_and_out_of_range_rejected():
    df = pd.DataFrame([{**GOOD_ROW, "Age": "abc"}, {**GOOD_ROW, "CreditScore": 5000}])
    with pytest.raises(InputValidationError) as exc:
        validate_input(df)
    assert len(exc.value.errors) == 2


def test_empty_file_rejected():
    with pytest.raises(InputValidationError):
        validate_input(pd.DataFrame(columns=config.FEATURE_COLS))


# --- Feature engineering -------------------------------------------------------
def test_feature_engineer():
    out = FeatureEngineer().transform(pd.DataFrame([{**GOOD_ROW, "Balance": 0.0, "NumOfProducts": 4}]))
    assert out.loc[0, "ZeroBalance"] == 1
    assert out.loc[0, "ProductsCat"] == "3"  # 3 and 4 grouped together
    assert out.loc[0, "AgeXActive"] == 0
    assert FeatureEngineer(enabled=False).transform(pd.DataFrame([GOOD_ROW])).shape[1] == len(config.FEATURE_COLS)


def test_encoder_tolerates_unseen_category_inside_pipeline():
    raw = load_raw().head(300)
    pipe = make_pipeline("lightgbm", engineered=True).fit(raw[config.FEATURE_COLS], raw[config.TARGET])
    odd = pd.DataFrame([{**GOOD_ROW, "Geography": "Italy"}])  # bypasses validation on purpose
    assert pipe.predict_proba(odd).shape == (1, 2)


# --- Saved model -----------------------------------------------------------------
@model_exists
def test_predictions_are_probabilities():
    from src.predict import load_artifacts, score
    art = load_artifacts()
    test = load_raw(config.TEST_FILE).head(200)
    out = score(test, art)
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["risk_level"]) <= {"High", "Medium", "Low"}
    assert len(out) == 200


@model_exists
def test_explanations_returned():
    from src.predict import explain, load_artifacts, suggest_actions
    art = load_artifacts()
    reasons = explain(pd.DataFrame([GOOD_ROW]), art, top_n=5)[0]
    assert len(reasons) == 5
    assert all("feature" in r and "impact" in r for r in reasons)
    assert suggest_actions(reasons)
