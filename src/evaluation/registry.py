"""Candidate models, each declaring its target framing.

Two factors were varied: model family and target framing (predict
tomorrow's level, or predict the change from today). A 2x2 backtest
isolated their effects:

                      level    delta    effect of framing
    huber            0.8990   0.8990        +0.0000
    ridge            0.9701   0.9701        +0.0000
    effect of loss   -0.0711  -0.0711

**The delta framing is a no-op; the entire win is the robust loss.**

Why the framing does nothing here: ``iqa`` is itself a feature, so for a
LINEAR model the two parametrisations span the same hypothesis space --
``y = iqa + w.x`` is reachable either way. They differ only through the L2
penalty, by about 3e-05 in MASE.

Why Huber wins: squared error is dominated by the heavy-tailed exceedance
days (persistence has RMSE 14.7 against MAE 7.4, a 2x ratio). Huber loss
is linear beyond its threshold, so those days stop dragging the fit.

Why tree ensembles lose: they average over leaves and shrink toward the
training mean, the wrong inductive bias for a near-random-walk series.
Trees cannot represent "start from today" internally, which is why the
delta framing is NOT a no-op for them -- there it actively hurts
(MAE 8.511 vs 8.324).
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
    "huber_level": Candidate(
        # Identical in performance to huber_delta. Kept so the scoreboard
        # itself demonstrates that the delta reframe is a no-op here.
        name="huber_level",
        target="level",
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
