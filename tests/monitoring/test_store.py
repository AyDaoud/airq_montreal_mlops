from datetime import date, datetime, timezone

import pytest

from src.monitoring.store import (
    PredictionRecord,
    ScoreRecord,
    Store,
    resolve_database_url,
)


@pytest.fixture
def store():
    return Store("sqlite:///:memory:", create=True)


def _record(**overrides):
    base = dict(
        request_id="req-1",
        served_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        station_id=3,
        target_date=date(2026, 1, 2),
        prediction=25.5,
        model_name="huber_level",
        model_version="abc1234",
        features_json='{"iqa": 24.0}',
    )
    base.update(overrides)
    return PredictionRecord(**base)


def test_round_trip_a_prediction(store):
    store.log_predictions([_record()])
    rows = store.recent_predictions(limit=10)
    assert len(rows) == 1
    assert rows[0].prediction == pytest.approx(25.5)
    assert rows[0].model_name == "huber_level"


def test_logging_many_predictions_in_one_call(store):
    store.log_predictions([_record(request_id=f"req-{i}") for i in range(25)])
    assert len(store.recent_predictions(limit=100)) == 25


def test_nullable_station_and_target_date_are_allowed(store):
    """A caller may post feature rows without saying which station or day."""
    store.log_predictions([_record(station_id=None, target_date=None)])
    assert store.recent_predictions(limit=1)[0].station_id is None


def test_scores_round_trip(store):
    store.write_score(
        ScoreRecord(
            scored_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
            window_days=7,
            n=120,
            mae_model=6.3,
            mae_persistence=7.4,
            mase=0.851,
        )
    )
    latest = store.latest_scores()
    assert latest[7].mase == pytest.approx(0.851)


def test_latest_scores_returns_the_most_recent_per_window(store):
    for day, mase in ((1, 1.2), (2, 0.9)):
        store.write_score(
            ScoreRecord(
                scored_at=datetime(2026, 1, day, tzinfo=timezone.utc),
                window_days=7,
                n=10,
                mae_model=1.0,
                mae_persistence=1.0,
                mase=mase,
            )
        )
    assert store.latest_scores()[7].mase == pytest.approx(0.9)


def test_latest_scores_is_empty_before_any_scoring_run(store):
    assert store.latest_scores() == {}


def test_resolve_database_url_defaults_to_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert resolve_database_url().startswith("sqlite:///")


def test_resolve_database_url_honours_the_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user@host/db")
    assert resolve_database_url() == "postgresql://user@host/db"


def test_empty_batch_is_a_no_op(store):
    store.log_predictions([])
    assert store.recent_predictions(limit=5) == []
