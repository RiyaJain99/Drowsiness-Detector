"""Metrics. Everything here is computed from real predictions; nothing has a default value."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_NAMES = ["alert", "drowsy"]


def binary_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> Dict:
    pred = (proba >= threshold).astype(int)
    both_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "n_drowsy": int(y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if both_classes else None,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y, pred, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0, output_dict=True
        ),
    }


def best_f1_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    grid = np.round(np.arange(0.2, 0.81, 0.01), 2)
    scores = [f1_score(y, proba >= g, zero_division=0) for g in grid]
    return float(grid[int(np.argmax(scores))])


def threshold_for_false_alarm_rate(y: np.ndarray, proba: np.ndarray, max_fpr: float) -> float:
    """Lowest cut-off (so highest recall) that flags at most ``max_fpr`` of alert windows.

    A warning system tuned for F1 alarms on a large share of alert driving and
    gets switched off; capping false alarms is the requirement that matters.
    """
    alert = np.sort(proba[y == 0])
    if len(alert) == 0:
        return 0.5
    # Values strictly above this alert-score quantile are flagged.
    k = int(np.floor(len(alert) * (1 - max_fpr)))
    cut = alert[min(k, len(alert) - 1)]
    return float(np.nextafter(cut, np.inf)) if k < len(alert) else float(alert[-1])


def video_level(frame: pd.DataFrame, proba: np.ndarray, threshold: float) -> Dict:
    """Mean probability per video: RLDD labels whole videos, so this is the like-for-like score."""
    grouped = frame.assign(proba=proba).groupby("video_key").agg(y=("y", "first"), proba=("proba", "mean"))
    return binary_metrics(grouped.y.to_numpy(), grouped.proba.to_numpy(), threshold)


def per_subject(frame: pd.DataFrame, proba: np.ndarray, threshold: float) -> List[Dict]:
    pred = proba >= threshold
    out = []
    for subject, rows in frame.assign(correct=pred == frame.y.to_numpy()).groupby("subject_key"):
        out.append({"subject": subject, "windows": int(len(rows)), "accuracy": float(rows.correct.mean())})
    return sorted(out, key=lambda r: r["accuracy"])


def summarise(values: List[Optional[float]]) -> Dict:
    kept = [v for v in values if v is not None]
    return {"mean": float(np.mean(kept)), "std": float(np.std(kept))} if kept else {"mean": None, "std": None}
