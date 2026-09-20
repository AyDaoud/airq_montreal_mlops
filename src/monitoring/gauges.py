"""Model-quality gauges exposed on /metrics.

Operational metrics (latency, request rate) come from the instrumentator.
These are the business metrics: is the model still beating persistence,
and is the data fresh enough to trust the answer.
"""

from __future__ import annotations

from datetime import datetime, timezone

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
