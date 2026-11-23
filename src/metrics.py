"""Cost-based threshold selection, metric helpers and the experiment log."""
from __future__ import annotations

import csv
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config


def expected_cost(y_true, y_prob, threshold: float) -> float:
    """Total business cost = COST_FN * missed churners + COST_FP * wasted offers."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return config.COST_FN * fn + config.COST_FP * fp


def best_threshold(y_true, y_prob) -> tuple[float, float]:
    """Threshold (0.01 grid) that minimises expected cost. Returns (threshold, cost)."""
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    costs = np.array([expected_cost(y_true, y_prob, t) for t in grid])
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def classification_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "expected_cost": float(config.COST_FN * fn + config.COST_FP * fp),
    }


LOG_FIELDS = [
    "timestamp", "stage", "model", "features",
    "cv_roc_auc", "cv_roc_auc_std", "cv_pr_auc", "cv_recall@0.5", "cv_f1@0.5", "notes",
]


def log_experiment(stage: str, model: str, features: str, scores: dict | None = None, notes: dict | str = "") -> None:
    """Append one row to experiments/log.csv so every run is recorded and comparable."""
    config.EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not config.EXPERIMENT_LOG.exists()
    scores = scores or {}
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "model": model,
        "features": features,
        "cv_roc_auc": _fmt(scores.get("roc_auc")),
        "cv_roc_auc_std": _fmt(scores.get("roc_auc_std")),
        "cv_pr_auc": _fmt(scores.get("pr_auc")),
        "cv_recall@0.5": _fmt(scores.get("recall")),
        "cv_f1@0.5": _fmt(scores.get("f1")),
        "notes": notes if isinstance(notes, str) else json.dumps(notes),
    }
    with open(config.EXPERIMENT_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _fmt(v):
    return "" if v is None else f"{v:.4f}"
