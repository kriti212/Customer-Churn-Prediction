"""Train the churn model end to end.

Run from the project root:  python -m src.train

Stages
  1. Split            stratified 80/20; the test set is saved and NOT used here.
  2. Go/no-go         logistic baseline must reach GO_NO_GO_MIN_AUC.
  3. Compare          4 models x {raw, engineered} features, 5-fold CV.
  4. Tune (Optuna)    the 2 best model types, OPTUNA_TRIALS each.
  5. Calibrate        isotonic calibration, so probabilities mean what they say.
  6. Threshold        minimise expected cost (FN costs COST_FN, FP costs COST_FP)
                      on out-of-fold calibrated probabilities.
  7. Save             calibrated model, explainer model and metadata.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime
from importlib.metadata import version

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from src import config
from src.metrics import best_threshold, classification_metrics, log_experiment
from src.models import make_pipeline, suggest_params
from src.preprocess import load_raw, split_xy

MODEL_NAMES = ["logreg", "random_forest", "xgboost", "lightgbm"]
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall", "f1": "f1"}


def cv_splitter():
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def cv_scores(pipe, X, y) -> dict:
    res = cross_validate(pipe, X, y, cv=cv_splitter(), scoring=SCORING, n_jobs=1)
    return {
        "roc_auc": res["test_roc_auc"].mean(),
        "roc_auc_std": res["test_roc_auc"].std(),
        "pr_auc": res["test_pr_auc"].mean(),
        "recall": res["test_recall"].mean(),
        "f1": res["test_f1"].mean(),
    }


def feat_label(engineered: bool) -> str:
    return "engineered" if engineered else "raw"


def main() -> None:
    t0 = time.time()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Split ----------------------------------------------------------
    df = load_raw()
    assert not set(config.LEAKY_COLS) & set(config.FEATURE_COLS), "Leaky column found in FEATURE_COLS"
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET], random_state=config.RANDOM_STATE
    )
    train_df.to_csv(config.TRAIN_FILE, index=False)
    test_df.to_csv(config.TEST_FILE, index=False)
    X, y = split_xy(train_df)
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    print(f"[split] train={len(train_df)} test={len(test_df)} churn rate={y.mean():.3f} pos_weight={pos_weight:.2f}")

    # ---- 2. Go/no-go -------------------------------------------------------
    base = cv_scores(make_pipeline("logreg", engineered=False), X, y)
    log_experiment("go_no_go", "logreg", "raw", base, f"threshold={config.GO_NO_GO_MIN_AUC}")
    print(f"[go/no-go] logistic baseline CV ROC-AUC = {base['roc_auc']:.4f} (need >= {config.GO_NO_GO_MIN_AUC})")
    if base["roc_auc"] < config.GO_NO_GO_MIN_AUC:
        raise SystemExit("STOP: baseline below go/no-go threshold. The data does not support modelling.")

    # ---- 3. Compare models -------------------------------------------------
    results = []  # (cv_auc, name, engineered, params, scores)
    for name in MODEL_NAMES:
        for engineered in (False, True):
            scores = cv_scores(make_pipeline(name, engineered, pos_weight=pos_weight), X, y)
            log_experiment("compare", name, feat_label(engineered), scores, "default params")
            results.append((scores["roc_auc"], name, engineered, {}, scores))
            print(f"[compare] {name:<14} {feat_label(engineered):<10} ROC-AUC={scores['roc_auc']:.4f} "
                  f"(+/-{scores['roc_auc_std']:.4f})  PR-AUC={scores['pr_auc']:.4f}")

    # Best feature set per model type, then the two strongest model types.
    best_per_model = {}
    for r in sorted(results, key=lambda r: r[0], reverse=True):
        best_per_model.setdefault(r[1], r)
    top2 = sorted(best_per_model.values(), key=lambda r: r[0], reverse=True)[:2]
    print(f"[compare] tuning: {[(r[1], feat_label(r[2])) for r in top2]}")

    # ---- 4. Optuna tuning --------------------------------------------------
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    candidates = list(results)
    for _, name, engineered, _, _ in top2:
        def objective(trial, name=name, engineered=engineered):
            params = suggest_params(name, trial)
            pipe = make_pipeline(name, engineered, params, pos_weight)
            return cross_validate(pipe, X, y, cv=cv_splitter(), scoring="roc_auc")["test_score"].mean()

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
            study_name=f"{name}_{feat_label(engineered)}",
        )
        ts = time.time()
        study.optimize(objective, n_trials=config.OPTUNA_TRIALS)
        params = {k: v for k, v in study.best_params.items()}
        if name == "lightgbm":
            params["subsample_freq"] = 1
        scores = cv_scores(make_pipeline(name, engineered, params, pos_weight), X, y)
        log_experiment("optuna", name, feat_label(engineered), scores,
                       {"trials": config.OPTUNA_TRIALS, "best_params": params})
        candidates.append((scores["roc_auc"], name, engineered, params, scores))
        print(f"[optuna] {name:<14} {feat_label(engineered):<10} best ROC-AUC={scores['roc_auc']:.4f} "
              f"({time.time() - ts:.0f}s) params={params}")

    best_auc, best_name, best_eng, best_params, best_scores = max(candidates, key=lambda r: r[0])
    print(f"[select] {best_name} / {feat_label(best_eng)} / "
          f"{'tuned' if best_params else 'default'} params  CV ROC-AUC={best_auc:.4f}")

    # ---- 5 & 6. Calibration + cost-based threshold (out-of-fold) ------------
    best_pipe = make_pipeline(best_name, best_eng, best_params, pos_weight)
    calibrated = CalibratedClassifierCV(best_pipe, method="isotonic", cv=config.CV_FOLDS)

    oof_raw = cross_val_predict(best_pipe, X, y, cv=cv_splitter(), method="predict_proba")[:, 1]
    oof_cal = cross_val_predict(calibrated, X, y, cv=cv_splitter(), method="predict_proba")[:, 1]
    brier_raw, brier_cal = brier_score_loss(y, oof_raw), brier_score_loss(y, oof_cal)
    print(f"[calibrate] Brier score (lower is better): uncalibrated={brier_raw:.4f} calibrated={brier_cal:.4f}")
    print(f"[calibrate] mean predicted churn: uncalibrated={oof_raw.mean():.3f} "
          f"calibrated={oof_cal.mean():.3f} actual={y.mean():.3f}")

    threshold, cost = best_threshold(y, oof_cal)
    oof_metrics = classification_metrics(y, oof_cal, threshold)
    print(f"[threshold] cost-optimal threshold={threshold:.2f} (FN cost {config.COST_FN}, FP cost {config.COST_FP}) "
          f"-> OOF recall={oof_metrics['recall']:.3f} precision={oof_metrics['precision']:.3f}")
    log_experiment("calibration", best_name, feat_label(best_eng),
                   {"roc_auc": roc_auc_score(y, oof_cal), "pr_auc": oof_metrics["pr_auc"]},
                   {"brier_uncalibrated": round(brier_raw, 4), "brier_calibrated": round(brier_cal, 4),
                    "threshold": threshold, "oof_recall": round(oof_metrics["recall"], 4)})
    _plot_calibration(y, oof_raw, oof_cal)

    # ---- 7. Fit on full training set and save ------------------------------
    calibrated.fit(X, y)
    best_pipe.fit(X, y)  # uncalibrated twin used only for SHAP explanations
    joblib.dump(calibrated, config.MODEL_FILE)
    joblib.dump(best_pipe, config.EXPLAINER_MODEL_FILE)

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": best_name,
        "feature_set": feat_label(best_eng),
        "params": best_params,
        "pos_weight": pos_weight,
        "calibration": "isotonic",
        "threshold": threshold,
        "cost_fn": config.COST_FN,
        "cost_fp": config.COST_FP,
        "feature_columns": config.FEATURE_COLS,
        "dropped_columns": config.DROP_COLS,
        "train_rows": int(len(train_df)),
        "train_churn_rate": float(y.mean()),
        "cv": {k: float(v) for k, v in best_scores.items()},
        "oof_calibrated": {k: v for k, v in oof_metrics.items()},
        "brier_uncalibrated": float(brier_raw),
        "comparison": [
            {"model": n, "features": feat_label(e), "tuned": bool(p), "cv_roc_auc": float(a)}
            for a, n, e, p, _ in sorted(candidates, key=lambda r: r[0], reverse=True)
        ],
        "versions": {
            "python": platform.python_version(),
            **{pkg: version(pkg) for pkg in ["scikit-learn", "pandas", "numpy", "xgboost", "lightgbm", "optuna", "shap"]},
        },
    }
    config.METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    print(f"[save] {config.MODEL_FILE.name}, {config.EXPLAINER_MODEL_FILE.name}, {config.METADATA_FILE.name} "
          f"written. Total time {time.time() - t0:.0f}s")
    print("Next: python -m src.evaluate  (uses the held-out test set ONCE)")


def _plot_calibration(y, raw, cal) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="Perfect calibration")
    for probs, label in [(raw, "Uncalibrated"), (cal, "Calibrated (isotonic)")]:
        frac, mean = calibration_curve(y, probs, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=label)
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Actual churn rate")
    ax.set_title("Calibration (out-of-fold, training set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "calibration_oof.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    np.random.seed(config.RANDOM_STATE)
    main()
