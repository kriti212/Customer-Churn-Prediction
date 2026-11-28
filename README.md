# Bank Customer Churn Predictor

Predicts which bank customers are likely to leave (`Exited = 1`). Each prediction comes with a calibrated
probability, its main reasons and a suggested retention action, all shown in a Streamlit app.

## Results (held-out test set, 2,000 customers, evaluated once)

| Metric | Value | Target |
|---|---|---|
| ROC-AUC | **0.879** | ≥ 0.85 ✅ |
| Recall (churners caught) | **70.8%** | ≥ 70% ✅ |
| Precision | 56.7% | |
| PR-AUC | 0.726 | random = 0.204 |
| Brier score (calibration) | 0.097 | |
| Expected cost vs. no model | **−53%** | |

- **Model:** XGBoost tuned with Optuna (50 trials), engineered features, isotonic calibration.
- **Decision threshold:** 0.27, chosen to minimise cost when a missed churner costs **3×** a wasted retention offer.
- **Model comparison** (5-fold CV ROC-AUC on the training set):
  - Default settings: logistic regression 0.766 (raw) / 0.835 (engineered), Random Forest 0.858, XGBoost 0.858, LightGBM 0.847.
  - Tuned with Optuna: XGBoost **0.864**, Random Forest 0.860.
  - Every run is logged in `experiments/log.csv`.
- **Main drivers (SHAP):** number of products, age × inactive member, age × active member, gender, geography.

## Important decisions

- **`Complain` is excluded (leakage).** It matches the churn label for 9,986 of 10,000 customers. A model
  that uses it scores ROC-AUC 0.999, because it is effectively reading the answer. A complaint is most likely
  recorded at the time of exit, so it would not be available early enough to act on.
- **Identifiers dropped:** `RowNumber`, `CustomerId`, `Surname`.
- **Kept on request despite almost no signal:** `Satisfaction Score`, `Card Type`, `Point Earned`.
- **Calibrated probabilities.** Class weighting inflates raw scores: before calibration the model predicted
  37% churn on average against an actual 20%. After isotonic calibration the average is 20.2%.
- **The test set is used once.** `src/evaluate.py` refuses to run again unless you pass `--force`.

## Fairness check (test set)

| Group | Recall | False-positive rate |
|---|---|---|
| Female / Male | 0.76 / 0.64 | 0.20 / 0.09 |
| Germany / France / Spain | 0.83 / 0.64 / 0.63 | 0.28 / 0.10 / 0.10 |
| Age 18-29 / 40-49 / 50-59 | 0.19 / 0.84 / 0.92 | 0.03 / 0.34 / 0.48 |

The model mostly reflects real differences in churn rates between groups. For example, German customers
actually churn at 30% against 17% in France. Even so, women and German customers who stay are flagged about
twice as often as other customers who stay, and young churners are mostly missed. Before using this to target
offers, decide whether these gaps are acceptable. Options include removing `Gender`, or setting a separate
threshold for each group. The app never suggests an action based on gender.

## Project structure

```
data/raw/                 original CSV
data/processed/           train/test split (created by training)
notebooks/01_eda.ipynb    exploratory analysis (imports from src/)
src/config.py             columns, leakage list, costs, targets: single source of truth
src/preprocess.py         validation, feature engineering, preprocessing
src/models.py             candidate models and Optuna search spaces
src/train.py              go/no-go -> compare -> Optuna -> calibrate -> threshold -> save
src/evaluate.py           one-time test evaluation, fairness, SHAP importance
src/predict.py            load model, score, explain, retention actions
src/explain.py            SHAP grouped back to original features
app/streamlit_app.py      Streamlit front end
models/                   model.joblib, explainer_model.joblib, metadata.json, metrics.json
experiments/log.csv       every experiment run
reports/figures/          calibration, confusion matrix, SHAP importance
tests/                    pytest suite (validation, leakage guards, predictions)
```

## How to run (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.train          # ~12 minutes (Optuna tuning)
python -m src.evaluate       # once: writes models/metrics.json
pytest -q                    # 19 tests
streamlit run app/streamlit_app.py
```

The trained model is already included, so to only use the app, install the requirements and run the last command.

## App

The app is built for **business users**. Its main pages use plain language only; technical detail is kept on
a separate page. Pages, using Streamlit multipage navigation (`st.navigation`):

| Section | Page | What it shows |
|---|---|---|
| Predict | **Check a customer** | Form, likelihood of leaving, risk level with guidance, reasons pushing risk up or down, recommended actions |
| Predict | **Score a customer list** | CSV upload with validation; risk counts, a sortable table with main reason and suggested action, download, detail view |
| Understand | **Customer insights** | Key findings and churn rate by segment, computed from historical data |
| Understand | **About the model** | Plain summary and reliability, then expandable technical sections: metrics, curves, SHAP, fairness, model comparison, data decisions, versions |

Risk levels: High = likelihood ≥ 27% (contact soon), Medium = 13.5–27% (monitor), Low = below 13.5%.

Code layout: `app/streamlit_app.py` (navigation), `app/common.py` (shared loaders and plain-language
wording), `app/views/*.py` (one file per page). `tests/test_app.py` checks that every page renders and that the
business pages contain no ML jargon.

## Limitations

- About 10,000 customers from a single snapshot, with no time dimension. We can't check how performance
  changes over time.
- The 3:1 cost ratio is an assumption. Change `COST_FN` and `COST_FP` in `src/config.py` and retrain.
- The model file must be loaded with the library versions in `requirements.txt`.
- No monitoring or retraining yet (planned for after deployment).
