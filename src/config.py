"""Single source of truth for paths, columns and business assumptions.

Every other module (training, evaluation, app, tests) reads column lists from
here, so the app and the training code can never disagree about the schema.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "Customer-Churn-Records.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
MODELS_DIR = ROOT / "models"
MODEL_FILE = MODELS_DIR / "model.joblib"            # calibrated pipeline -> probabilities
EXPLAINER_MODEL_FILE = MODELS_DIR / "explainer_model.joblib"  # uncalibrated pipeline -> SHAP
METADATA_FILE = MODELS_DIR / "metadata.json"
METRICS_FILE = MODELS_DIR / "metrics.json"
EXPERIMENT_LOG = ROOT / "experiments" / "log.csv"
FIGURES_DIR = ROOT / "reports" / "figures"

# --- Columns ---------------------------------------------------------------
TARGET = "Exited"

# Identifiers carry no predictive meaning (Surname is also personal data).
ID_COLS = ["RowNumber", "CustomerId", "Surname"]

# LEAKAGE: Complain matches Exited in 99.86% of rows (only 14 of 10,000 disagree).
# It is almost certainly recorded at/after the exit event, so it would not be
# available when we need the prediction. Never use it as a feature.
LEAKY_COLS = ["Complain"]

DROP_COLS = ID_COLS + LEAKY_COLS

CATEGORICAL_COLS = ["Geography", "Gender", "Card Type"]
NUMERIC_COLS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Satisfaction Score",
    "Point Earned",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS  # raw inputs the model expects

# Allowed values / ranges, used for input validation in the app and batch scoring.
CATEGORY_VALUES = {
    "Geography": ["France", "Germany", "Spain"],
    "Gender": ["Female", "Male"],
    "Card Type": ["DIAMOND", "GOLD", "PLATINUM", "SILVER"],
}
NUMERIC_RANGES = {  # (min, max) accepted; generous bounds around the training data
    "CreditScore": (300, 900),
    "Age": (18, 100),
    "Tenure": (0, 10),
    "Balance": (0, 300_000),
    "NumOfProducts": (1, 4),
    "HasCrCard": (0, 1),
    "IsActiveMember": (0, 1),
    "EstimatedSalary": (0, 250_000),
    "Satisfaction Score": (1, 5),
    "Point Earned": (0, 1_500),
}

# Sensitive attributes audited for fairness (kept as features, but checked).
FAIRNESS_COLS = ["Gender", "Geography", "AgeBand"]

# --- Business assumptions ---------------------------------------------------
# A missed churner (false negative) costs 3x a wasted retention offer (false positive).
COST_FN = 3.0
COST_FP = 1.0

# Success criteria agreed before modelling.
TARGET_ROC_AUC = 0.85
TARGET_RECALL = 0.70

# --- Experiment settings ---------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
OPTUNA_TRIALS = 50
GO_NO_GO_MIN_AUC = 0.75  # logistic baseline must reach this to continue
