"""Candidate models, each declaring its target framing.

Target framing and model family interact. A delta target with a
RandomForest is worse than the level (MAE 8.511 vs 8.324); the same delta
target with a robust linear model is the best thing measured (mean MASE
0.899). Neither choice can be evaluated alone, so both live here together.

Why tree ensembles lose on this problem: they average over leaves and
shrink toward the training mean, which is the wrong inductive bias for a
near-random-walk series. A linear model on the difference starts from
today's value and learns only the correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VALUE_COLUMN = "iqa"
TARGET_COLUMN = "target"


@dataclass(frozen=True)
class Candidate:
    """One model under test.

    ``target`` is ``"level"`` (predict tomorrow directly) or ``"delta"``
    (predict tomorrow minus today, then add today back).
    """

    name: str
    target: str
    build: Callable[[], object]


CANDIDATES: dict[str, Candidate] = {
    "huber_delta": Candidate(
        name="huber_delta",
        target="delta",
        build=lambda: make_pipeline(
            StandardScaler(), HuberRegressor(max_iter=800, epsilon=1.35)
        ),
    ),
    "ridge_delta": Candidate(
        name="ridge_delta",
        target="delta",
        build=lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    ),
    "ridge_level": Candidate(
        name="ridge_level",
        target="level",
        build=lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    ),
    "hgb_delta": Candidate(
        name="hgb_delta",
        target="delta",
        build=lambda: HistGradientBoostingRegressor(random_state=42, max_iter=300),
    ),
    "rf_level": Candidate(
        name="rf_level",
        target="level",
        build=lambda: RandomForestRegressor(
            n_estimators=300, max_depth=12, n_jobs=-1, random_state=42
        ),
    ),
}


def fit_predict(
    candidate: Candidate,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    """Fit on ``train`` and predict the LEVEL for ``test``.

    Delta candidates are trained on ``target - iqa`` and have today's value
    added back, so every candidate returns a comparable level prediction.
    """
    model = candidate.build()
    if candidate.target == "delta":
        model.fit(train[features], train[TARGET_COLUMN] - train[VALUE_COLUMN])
        return np.asarray(test[VALUE_COLUMN] + model.predict(test[features]))
    model.fit(train[features], train[TARGET_COLUMN])
    return np.asarray(model.predict(test[features]))
