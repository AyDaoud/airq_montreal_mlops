"""Model-quality gauges exposed on /metrics.

Operational metrics (latency, request rate) come from the instrumentator.
These are the business metrics: is the model still beating persistence,
and is the data fresh enough to trust the answer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from prometheus_client import REGISTRY, CollectorRegistry, Gauge


def freshness_days(truth: pd.DataFrame, now: datetime | None = None) -> float | None:
    """Days since the newest row of ground truth, or None if there is none."""
    if truth.empty or "date_local" not in truth.columns:
        return None
    newest = pd.to_datetime(truth["date_local"]).max()
    if pd.isna(newest):
        return None
    reference = pd.Timestamp(now or datetime.now(timezone.utc))
    if reference.tzinfo is not None:
        reference = reference.tz_localize(None)
    newest = pd.Timestamp(newest)
    if newest.tzinfo is not None:
        newest = newest.tz_localize(None)
    return float((reference - newest).total_seconds() / 86400)


class ModelGauges:
    """Prometheus gauges refreshed from the score table."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        registry = registry if registry is not None else REGISTRY
        self.mase_7d = Gauge(
            "airq_mase_7d",
            "Rolling 7-day MASE against persistence. 1.0 ties; below 1.0 beats it.",
            registry=registry,
        )
        self.mase_30d = Gauge(
            "airq_mase_30d",
            "Rolling 30-day MASE against persistence.",
            registry=registry,
        )
        self.freshness = Gauge(
            "airq_data_freshness_days",
            "Days since the newest ground-truth row.",
            registry=registry,
        )

    def refresh(self, store, freshness: float | None) -> None:
        """Update the gauges. Never raises: a metrics failure is not an outage."""
        try:
            scores = store.latest_scores()
            if 7 in scores:
                self.mase_7d.set(scores[7].mase)
            if 30 in scores:
                self.mase_30d.set(scores[30].mase)
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            print(f"[warn] could not refresh MASE gauges: {exc}")
        if freshness is not None:
            self.freshness.set(freshness)


class ModelMetricsCollector:
    """Compute model-quality metrics at scrape time.

    ``ModelGauges`` needs something to call ``refresh()``. Rather than a
    background task that can silently die and leave stale values on the
    dashboard, this computes on scrape: if the collector runs, the numbers
    are current; if it fails, the series simply disappear rather than
    freezing at their last value.

    The gold table is read at most once per ``ttl_seconds`` so a 15-second
    scrape does not re-read a 16,000-row parquet every time.
    """

    def __init__(self, store=None, gold_path=None, ttl_seconds: int = 60) -> None:
        self._store = store
        self._gold_path = gold_path
        self._ttl = ttl_seconds
        self._cached_at: float | None = None
        self._freshness: float | None = None

    def _resolve_store(self):
        if self._store is not None:
            return self._store
        from src.monitoring.store import Store

        self._store = Store()
        return self._store

    def _resolve_freshness(self) -> float | None:
        import time

        now = time.monotonic()
        if self._cached_at is not None and now - self._cached_at < self._ttl:
            return self._freshness
        self._cached_at = now
        self._freshness = None
        try:
            from src.models.training_daily import GOLD_PATH

            path = self._gold_path or GOLD_PATH
            if Path(path).exists():
                self._freshness = freshness_days(
                    pd.read_parquet(path, columns=["date_local"])
                )
        except Exception as exc:  # noqa: BLE001 - a metric is not worth an error
            print(f"[warn] freshness unavailable: {exc}")
        return self._freshness

    def collect(self):
        """Yield the current metrics. Never raises."""
        from prometheus_client.core import GaugeMetricFamily

        try:
            scores = self._resolve_store().latest_scores()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] scores unavailable: {exc}")
            scores = {}

        for window in (7, 30):
            if window in scores:
                yield GaugeMetricFamily(
                    f"airq_mase_{window}d",
                    f"Rolling {window}-day MASE against persistence. "
                    "1.0 ties; below 1.0 beats it.",
                    value=float(scores[window].mase),
                )

        freshness = self._resolve_freshness()
        if freshness is not None:
            yield GaugeMetricFamily(
                "airq_data_freshness_days",
                "Days since the newest ground-truth row.",
                value=float(freshness),
            )
