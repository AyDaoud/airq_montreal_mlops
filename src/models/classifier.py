"""Exceedance classifier: will IQA exceed 50 tomorrow?

Base rate is about 3% of station-days, so accuracy is meaningless - a model
predicting "never" scores 97%. Everything here is framed around precision,
recall and an explicit cost trade-off instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# A missed exceedance costs this many times a false alarm. Public-health
# framing: failing to warn a sensitive resident is worse than warning
# unnecessarily. Stated here and in the README, never left implicit.
COST_RATIO = 5.0


def build_classifier(**kwargs) -> RandomForestClassifier:
    """The exceedance classifier, balanced for a rare positive class."""
    params = {
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }
    params.update(kwargs)
    return RandomForestClassifier(**params)


def expected_cost(y_true, y_pred, cost_ratio: float = COST_RATIO) -> float:
    """Total cost where a miss costs ``cost_ratio`` and a false alarm 1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    misses = int(((y_true == 1) & (y_pred == 0)).sum())
    false_alarms = int(((y_true == 0) & (y_pred == 1)).sum())
    return float(cost_ratio * misses + false_alarms)


def choose_threshold(
    y_true, y_proba, cost_ratio: float = COST_RATIO, n_steps: int = 200
) -> float:
    """The probability threshold minimising expected cost on this data."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    grid = np.linspace(0.01, 0.99, n_steps)
    costs = [
        expected_cost(y_true, (y_proba >= t).astype(int), cost_ratio) for t in grid
    ]
    return float(grid[int(np.argmin(costs))])
