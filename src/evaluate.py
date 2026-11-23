"""Final, one-time evaluation on the held-out test set.

Run from the project root:  python -m src.evaluate

The test set is the only unbiased estimate of real-world performance. Looking at
it repeatedly while changing the model turns it into another validation set, so
this script refuses to run twice unless you pass --force (and the run is logged).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config
from src.explain import shap_values
from src.metrics import classification_metrics, log_experiment
from src.predict import load_artifacts
from src.preprocess import add_age_band, load_raw, split_xy


def fairness_table(df: pd.DataFrame, y, probs, threshold) -> list[dict]:
    pred = (probs >= threshold).astype(int)
    d = pd.DataFrame({"y": y.values, "p": probs, "pred": pred}, index=df.index)
    d["Gender"], d["Geography"], d["AgeBand"] = df["Gender"], df["Geography"], add_age_band(df).astype(str)
    rows = []
    for col in config.FAIRNESS_COLS:
        for grp, g in d.groupby(col, observed=True):
            pos, neg = g[g.y == 1], g[g.y == 0]
            rows.append({
                "attribute": col,
                "group": str(grp),
                "n": int(len(g)),
                "actual_churn_rate": float(g.y.mean()),
                "mean_predicted_prob": float(g.p.mean()),
                "flagged_rate": float(g.pred.mean()),
                "recall": float(pos.pred.mean()) if len(pos) else None,
                "false_positive_rate": float(neg.pred.mean()) if len(neg) else None,
            })
    return rows


def _downsample(xs, ys, n=200):
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return [float(v) for v in np.asarray(xs)[idx]], [float(v) for v in np.asarray(ys)[idx]]


def main(force: bool = False) -> None:
    if config.METRICS_FILE.exists() and not force:
        raise SystemExit("Test set already evaluated (models/metrics.json exists). Re-running the test set after "
                         "changing the model biases the result. Use --force only if you know why.")

    art = load_artifacts()
    test_df = load_raw(config.TEST_FILE)
    X, y = split_xy(test_df)
    probs = art.model.predict_proba(X)[:, 1]
    m = classification_metrics(y, probs, art.threshold)

    # Curves (stored as data so the app can draw interactive charts).
    fpr, tpr, _ = roc_curve(y, probs)
    prec, rec, _ = precision_recall_curve(y, probs)
    frac, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
    roc_x, roc_y = _downsample(fpr, tpr)
    pr_x, pr_y = _downsample(rec, prec)

    # Global SHAP importance (mean |SHAP| per original feature) on the test set.
    grouped = shap_values(art.explainer, art.explainer_model, X)
    importance = grouped.abs().mean().sort_values(ascending=False)

    fairness = fairness_table(test_df, y, probs, art.threshold)
    criteria = {
        "roc_auc": {"target": config.TARGET_ROC_AUC, "actual": m["roc_auc"], "met": m["roc_auc"] >= config.TARGET_ROC_AUC},
        "recall": {"target": config.TARGET_RECALL, "actual": m["recall"], "met": m["recall"] >= config.TARGET_RECALL},
    }
    baseline_cost = config.COST_FN * int(y.sum())  # cost of doing nothing (miss every churner)

    out = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": art.metadata["model_name"],
        "test_rows": int(len(y)),
        "test_churn_rate": float(y.mean()),
        "metrics": m,
        "baseline_cost_no_model": float(baseline_cost),
        "cost_saving_vs_no_model": float(1 - m["expected_cost"] / baseline_cost),
        "success_criteria": criteria,
        "curves": {
            "roc": {"fpr": roc_x, "tpr": roc_y},
            "pr": {"recall": pr_x, "precision": pr_y},
            "calibration": {"mean_predicted": mean_pred.tolist(), "actual_rate": frac.tolist()},
        },
        "shap_importance": {k: float(v) for k, v in importance.items()},
        "fairness": fairness,
    }
    config.METRICS_FILE.write_text(json.dumps(out, indent=2))
    log_experiment("test_eval", art.metadata["model_name"], art.metadata["feature_set"],
                   {"roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"]},
                   {"threshold": art.threshold, "recall": round(m["recall"], 4), "forced": force})

    _save_figures(y, probs, m, grouped, X, art)

    print(f"Test set: {len(y)} customers, churn rate {y.mean():.3f}")
    print(f"ROC-AUC {m['roc_auc']:.4f} | PR-AUC {m['pr_auc']:.4f} | Brier {m['brier']:.4f}")
    print(f"At threshold {art.threshold:.2f}: recall {m['recall']:.3f}, precision {m['precision']:.3f}, "
          f"F1 {m['f1']:.3f}, FPR {m['false_positive_rate']:.3f}")
    print(f"Confusion matrix: {m['confusion_matrix']}")
    print(f"Expected cost {m['expected_cost']:.0f} vs {baseline_cost:.0f} with no model "
          f"({out['cost_saving_vs_no_model']:.1%} saving)")
    for k, c in criteria.items():
        print(f"Success criterion {k}: target {c['target']} actual {c['actual']:.3f} -> {'MET' if c['met'] else 'NOT MET'}")
    print("Top drivers (mean |SHAP|):", ", ".join(f"{k} {v:.3f}" for k, v in importance.head(6).items()))
    print("\nFairness check (test set):")
    print(pd.DataFrame(fairness).round(3).to_string(index=False))


def _save_figures(y, probs, m, grouped, X, art) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = m["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center", color="black", fontsize=13)
    ax.set_xticks([0, 1], ["Predicted stay", "Predicted churn"])
    ax.set_yticks([0, 1], ["Actual stay", "Actual churn"])
    ax.set_title(f"Confusion matrix (threshold {art.threshold:.2f})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    imp = grouped.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp.index, imp.values, color="#4C78A8")
    ax.set_xlabel("Mean |SHAP value| (impact on churn score)")
    ax.set_title("What drives the model (test set)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shap_importance.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-evaluate even though the test set was already used")
    main(parser.parse_args().force)
