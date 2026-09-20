"""Metrics, with MASE as the headline.

MASE is the mean absolute error scaled by the baseline's: 1.00 ties the
baseline, below 1.00 beats it. Reporting raw MAE lets a comparison be
dropped silently; reporting MASE does not.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def mase(y_true, y_pred, y_baseline) -> float:
    """Mean absolute scaled error against an explicit baseline."""
    baseline_mae = mean_absolute_error(y_true, y_baseline)
    if baseline_mae == 0:
        raise ValueError(
            "baseline has zero error; MASE is undefined. Check the baseline "
            "is not the target itself."
        )
    return float(mean_absolute_error(y_true, y_pred) / baseline_mae)


def classification_report_at_threshold(
    y_true, y_proba, threshold: float
) -> dict[str, float]:
    """Precision/recall at an operating point, plus threshold-free PR-AUC.

    Accuracy is deliberately absent: at a 3% base rate, predicting "never"
    scores 97% and means nothing.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    flagged = (y_proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "threshold": float(threshold),
        "n_positive": int(y_true.sum()),
        "n_flagged": int(flagged.sum()),
        "base_rate": float(y_true.mean()),
    }


def interval_coverage(y_true, lower, upper) -> float:
    """Fraction of outcomes inside the predicted interval."""
    y_true = np.asarray(y_true)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    return float(inside.mean())
