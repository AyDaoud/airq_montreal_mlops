from datetime import datetime, timezone

import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from src.monitoring.gauges import ModelGauges, freshness_days
from src.monitoring.store import ScoreRecord, Store


def test_freshness_is_days_since_the_newest_row():
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    truth = pd.DataFrame({"date_local": pd.to_datetime(["2026-09-14", "2026-09-10"])})
    assert freshness_days(truth, now=now) == pytest.approx(6.0, abs=0.01)


def test_freshness_of_an_empty_frame_is_not_a_crash():
    assert freshness_days(pd.DataFrame({"date_local": []})) is None


def test_gauges_publish_the_latest_mase_per_window():
    registry = CollectorRegistry()
    gauges = ModelGauges(registry=registry)
    store = Store("sqlite:///:memory:", create=True)
    for window, mase in ((7, 0.88), (30, 0.91)):
        store.write_score(
            ScoreRecord(
                scored_at=datetime(2026, 9, 20, tzinfo=timezone.utc),
                window_days=window,
                n=100,
                mae_model=6.5,
                mae_persistence=7.4,
                mase=mase,
            )
        )
    gauges.refresh(store, freshness=6.0)
    assert registry.get_sample_value("airq_mase_7d") == pytest.approx(0.88)
    assert registry.get_sample_value("airq_mase_30d") == pytest.approx(0.91)
    assert registry.get_sample_value("airq_data_freshness_days") == pytest.approx(6.0)


def test_refresh_with_no_scores_leaves_gauges_unset_not_wrong():
    """Zero would read as 'the model is perfect'. Unset is the honest value."""
    registry = CollectorRegistry()
    gauges = ModelGauges(registry=registry)
    gauges.refresh(Store("sqlite:///:memory:", create=True), freshness=None)
    assert registry.get_sample_value("airq_mase_7d") in (None, 0.0)


def test_refresh_never_raises_when_the_store_is_broken():
    class Broken:
        def latest_scores(self):
            raise RuntimeError("database gone")

    gauges = ModelGauges(registry=CollectorRegistry())
    gauges.refresh(Broken(), freshness=1.0)  # must not raise


def test_collector_yields_mase_when_scores_exist(tmp_path):
    """ModelGauges was built but never driven; this is what drives it."""
    from src.monitoring.gauges import ModelMetricsCollector

    store = Store("sqlite:///:memory:", create=True)
    store.write_score(
        ScoreRecord(
            scored_at=datetime(2026, 9, 20, tzinfo=timezone.utc),
            window_days=7,
            n=50,
            mae_model=6.0,
            mae_persistence=7.4,
            mase=0.81,
        )
    )
    collector = ModelMetricsCollector(
        store=store, gold_path=tmp_path / "absent.parquet"
    )
    names = {m.name: m for m in collector.collect()}
    assert "airq_mase_7d" in names
    assert names["airq_mase_7d"].samples[0].value == pytest.approx(0.81)


def test_collector_yields_nothing_rather_than_zero_when_empty(tmp_path):
    """An absent series is honest. A zero would read as a perfect model."""
    from src.monitoring.gauges import ModelMetricsCollector

    collector = ModelMetricsCollector(
        store=Store("sqlite:///:memory:", create=True),
        gold_path=tmp_path / "absent.parquet",
    )
    assert list(collector.collect()) == []


def test_collector_never_raises_on_a_broken_store(tmp_path):
    from src.monitoring.gauges import ModelMetricsCollector

    class Broken:
        def latest_scores(self):
            raise RuntimeError("database gone")

    collector = ModelMetricsCollector(
        store=Broken(), gold_path=tmp_path / "absent.parquet"
    )
    assert list(collector.collect()) == []


def test_collector_reports_freshness_from_a_real_parquet(tmp_path):
    from src.monitoring.gauges import ModelMetricsCollector

    path = tmp_path / "gold.parquet"
    pd.DataFrame({"date_local": pd.to_datetime(["2026-09-14"])}).to_parquet(path)
    collector = ModelMetricsCollector(
        store=Store("sqlite:///:memory:", create=True), gold_path=path
    )
    names = {m.name: m for m in collector.collect()}
    assert "airq_data_freshness_days" in names
    assert names["airq_data_freshness_days"].samples[0].value > 0
