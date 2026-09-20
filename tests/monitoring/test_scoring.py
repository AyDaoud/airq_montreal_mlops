from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.monitoring.scoring import score_window
from src.monitoring.store import PredictionRecord, Store


def _truth(start="2026-01-01", days=30, station_id=3):
    """A gold-shaped truth frame: station_id, date_local, iqa."""
    return pd.DataFrame(
        {
            "station_id": station_id,
            "date_local": pd.date_range(start, periods=days, freq="D"),
            "iqa": [20 + (d % 7) for d in range(days)],
        }
    )


def _logged(store, target, prediction, station_id=3, served_offset=0):
    store.log_predictions(
        [
            PredictionRecord(
                request_id=f"r-{target}-{prediction}",
                served_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
                + timedelta(days=served_offset),
                station_id=station_id,
                target_date=target,
                prediction=prediction,
                model_name="huber_level",
                model_version="test",
                features_json="{}",
            )
        ]
    )


def test_perfect_predictions_score_zero_mae():
    store = Store("sqlite:///:memory:", create=True)
    truth = _truth()
    for offset in range(5):
        day = date(2026, 1, 2) + timedelta(days=offset)
        actual = float(truth.loc[truth.date_local == pd.Timestamp(day), "iqa"].iloc[0])
        _logged(store, day, actual, served_offset=offset)
    result = score_window(store, truth, window_days=30, now=datetime(2026, 1, 20))
    assert result is not None
    assert result.mae_model == pytest.approx(0.0, abs=1e-9)


def test_mase_below_one_when_the_model_beats_persistence():
    store = Store("sqlite:///:memory:", create=True)
    truth = _truth()
    for offset in range(10):
        day = date(2026, 1, 3) + timedelta(days=offset)
        actual = float(truth.loc[truth.date_local == pd.Timestamp(day), "iqa"].iloc[0])
        _logged(store, day, actual, served_offset=offset)  # perfect model
    result = score_window(store, truth, window_days=30, now=datetime(2026, 1, 20))
    assert result.mase < 1.0


def test_persistence_uses_the_day_before_the_target():
    """The baseline must be yesterday's actual, per station, not a global mean."""
    store = Store("sqlite:///:memory:", create=True)
    truth = _truth()
    day = date(2026, 1, 5)
    _logged(store, day, 999.0)
    result = score_window(store, truth, window_days=30, now=datetime(2026, 1, 20))
    previous = float(
        truth.loc[
            truth.date_local == pd.Timestamp(day) - pd.Timedelta(days=1), "iqa"
        ].iloc[0]
    )
    actual = float(truth.loc[truth.date_local == pd.Timestamp(day), "iqa"].iloc[0])
    assert result.mae_persistence == pytest.approx(abs(actual - previous))


def test_no_matured_predictions_yields_no_score():
    """Do not divide by zero, and do not write a meaningless row."""
    store = Store("sqlite:///:memory:", create=True)
    _logged(store, date(2027, 6, 1), 25.0)  # target far in the future
    assert (
        score_window(store, _truth(), window_days=7, now=datetime(2026, 1, 20)) is None
    )


def test_predictions_outside_the_window_are_excluded():
    store = Store("sqlite:///:memory:", create=True)
    truth = _truth(days=60)
    _logged(store, date(2026, 1, 2), 25.0)  # old
    _logged(store, date(2026, 2, 20), 25.0)  # recent
    narrow = score_window(store, truth, window_days=7, now=datetime(2026, 2, 21))
    wide = score_window(store, truth, window_days=90, now=datetime(2026, 2, 21))
    assert narrow.n == 1
    assert wide.n == 2


def test_predictions_without_a_target_date_are_ignored():
    store = Store("sqlite:///:memory:", create=True)
    store.log_predictions(
        [
            PredictionRecord(
                "no-date",
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                3,
                None,
                25.0,
                "huber_level",
                "test",
                "{}",
            )
        ]
    )
    assert (
        score_window(store, _truth(), window_days=30, now=datetime(2026, 1, 20)) is None
    )
