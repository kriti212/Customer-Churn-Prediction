"""Load the saved model, validate input, score customers and explain predictions."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from importlib.metadata import version

import joblib
import pandas as pd

from src import config
from src.explain import make_explainer, shap_values, top_reasons
from src.preprocess import load_raw, validate_input


@dataclass
class Artifacts:
    model: object            # calibrated pipeline -> probabilities
    explainer_model: object  # uncalibrated pipeline -> SHAP
    explainer: object
    metadata: dict

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])


def load_artifacts() -> Artifacts:
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError("No trained model found. Run:  python -m src.train")
    metadata = json.loads(config.METADATA_FILE.read_text())

    # A pickled pipeline is only reliable under the library versions it was saved with.
    for pkg in ["scikit-learn", "xgboost", "lightgbm"]:
        saved, installed = metadata["versions"].get(pkg), version(pkg)
        if saved and saved != installed:
            warnings.warn(f"{pkg} version mismatch: model saved with {saved}, running {installed}. "
                          f"Reinstall from requirements.txt or retrain.")
    if metadata["feature_columns"] != config.FEATURE_COLS:
        raise RuntimeError("Saved model expects different columns than src/config.py. Retrain the model.")

    model = joblib.load(config.MODEL_FILE)
    explainer_model = joblib.load(config.EXPLAINER_MODEL_FILE)
    background = load_raw(config.TRAIN_FILE)[config.FEATURE_COLS].sample(200, random_state=config.RANDOM_STATE)
    explainer = make_explainer(explainer_model, metadata["model_name"], background)
    return Artifacts(model, explainer_model, explainer, metadata)


def risk_level(prob: float, threshold: float) -> str:
    """High = at or above the cost-optimal threshold (act now); Medium = half-way there."""
    if prob >= threshold:
        return "High"
    if prob >= threshold / 2:
        return "Medium"
    return "Low"


def score(df: pd.DataFrame, art: Artifacts) -> pd.DataFrame:
    """Validate and score. Returns the input rows plus probability, flag and risk level.

    Raises InputValidationError with a list of readable problems if the data is invalid.
    """
    X = validate_input(df)
    probs = art.model.predict_proba(X)[:, 1]
    out = df.copy()
    out["churn_probability"] = probs.round(4)
    out["predicted_churn"] = (probs >= art.threshold).astype(int)
    out["risk_level"] = [risk_level(p, art.threshold) for p in probs]
    return out


def explain(df: pd.DataFrame, art: Artifacts, top_n: int = 5) -> list[list[dict]]:
    """Top SHAP reasons for each row of df (already valid)."""
    X = validate_input(df)
    grouped = shap_values(art.explainer, art.explainer_model, X)
    return [top_reasons(grouped.loc[i], X.loc[i], top_n) for i in X.index]


# Simple playbook: which retention action fits which risk driver.
ACTIONS = {
    "IsActiveMember": "Re-engage: personalised offers or a check-in call to bring the customer back to active use.",
    "Age × active member": "Re-engage: personalised offers or a check-in call to bring the customer back to active use.",
    "NumOfProducts": "Review product fit: customers holding 3-4 products churn far more often. Check for mis-selling or bundle issues.",
    "Age": "Offer age-relevant products (e.g. retirement or wealth-management advice).",
    "Geography": "Apply the region-specific retention programme (churn is about twice as high in Germany).",
    "Balance": "Contact about balance movements: offer better savings rates or a relationship-manager review.",
    "Balance-to-salary ratio": "Contact about balance movements: offer better savings rates or a relationship-manager review.",
    "Gender": None,  # never base an action on gender
    "CreditScore": "Review credit options: a pre-approved limit increase or a better loan rate.",
    "Tenure": "Loyalty recognition: tenure-based reward or fee waiver.",
}


def suggest_actions(reasons: list[dict], max_actions: int = 3) -> list[str]:
    actions = []
    for r in reasons:
        if r["impact"] <= 0:
            continue
        act = ACTIONS.get(r["feature"])
        if act and act not in actions:
            actions.append(act)
        if len(actions) == max_actions:
            break
    return actions or ["No specific driver stands out. Standard relationship check-in."]
