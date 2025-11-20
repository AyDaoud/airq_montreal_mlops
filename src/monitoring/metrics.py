from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def regression_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}

def simple_drift_score(series_baseline, series_current):
    """Return absolute difference of medians as a cheap drift proxy."""
    b = np.median(series_baseline)
    c = np.median(series_current)
    return float(abs(c - b))
