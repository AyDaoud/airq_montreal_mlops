import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import add_horizon_target, run_backtest
from src.evaluation.registry import Candidate


class _AlwaysZero:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _frame(n_days=500, n_stations=2):
    rng = np.random.RandomState(0)
    rows = []
    for station_id in range(n_stations):
        value = 25.0
        for day in range(n_days):
            value = max(1.0, value + rng.randn() * 3)
            rows.append(
                {
                    "station_id": station_id,
                    "date_local": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "iqa": value,
                    "lag_1": value,
                }
            )
    return pd.DataFrame(rows)


def test_add_horizon_target_shifts_by_the_requested_days():
    frame = add_horizon_target(_frame(n_days=10, n_stations=1), horizon=2)
    station = frame.sort_values("date_local").reset_index(drop=True)
    assert station["target"].iloc[0] == pytest.approx(station["iqa"].iloc[2])


def test_add_horizon_target_never_crosses_a_station_boundary():
    frame = add_horizon_target(_frame(n_days=5, n_stations=2), horizon=1)
    last = frame[frame["station_id"] == 0].sort_values("date_local").iloc[-1]
    assert pd.isna(last["target"])


def test_backtest_returns_one_row_per_fold_candidate_horizon():
    candidates = {
        "always_today": Candidate("always_today", "delta", lambda: _AlwaysZero())
    }
    out = run_backtest(
        _frame(),
        features=["lag_1"],
        candidates=candidates,
        n_folds=3,
        test_days=60,
        horizons=(1, 2),
    )
    # 3 folds x 2 horizons x (1 candidate + 3 baselines) = 24, but
    # seasonal_naive may drop rows; assert the candidate's own count.
    own = out[out["candidate"] == "always_today"]
    assert len(own) == 3 * 2
    assert set(out.columns) >= {"fold", "candidate", "horizon", "mae", "mase"}


def test_a_model_that_predicts_today_ties_persistence():
    """A delta model predicting zero change IS persistence, so MASE == 1."""
    candidates = {
        "always_today": Candidate("always_today", "delta", lambda: _AlwaysZero())
    }
    out = run_backtest(
        _frame(),
        features=["lag_1"],
        candidates=candidates,
        n_folds=2,
        test_days=60,
        horizons=(1,),
    )
    own = out[out["candidate"] == "always_today"]
    assert own["mase"].round(6).eq(1.0).all(), own[["fold", "mase"]]


def test_backtest_includes_the_baselines():
    out = run_backtest(
        _frame(),
        features=["lag_1"],
        candidates={},
        n_folds=2,
        test_days=60,
        horizons=(1,),
    )
    assert set(out["candidate"]) >= {"persistence", "seasonal_naive", "climatology"}


def test_persistence_always_scores_exactly_one():
    out = run_backtest(
        _frame(),
        features=["lag_1"],
        candidates={},
        n_folds=3,
        test_days=60,
        horizons=(1, 2),
    )
    persistence_rows = out[out["candidate"] == "persistence"]
    assert persistence_rows["mase"].round(9).eq(1.0).all()
