"""Score served predictions against ground truth as it arrives.

This is the difference between measuring a model in backtest and measuring
the one that actually answered requests. The metric is MASE against
persistence, matching Spec B: 1.00 ties "tomorrow = today", below beats it.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.monitoring.store import ScoreRecord, Store

WINDOWS = (7, 30)


def score_window(
    store: Store,
    truth: pd.DataFrame,
    window_days: int,
    now: datetime | None = None,
) -> ScoreRecord | None:
    """Score predictions whose target day falls inside the window.

    Returns ``None`` when nothing has matured, rather than writing a row
    computed from no data.
    """
    now = now or datetime.utcnow()
    cutoff = pd.Timestamp(now) - pd.Timedelta(days=window_days)

    rows = [
        p
        for p in store.matured_predictions()
        if p.target_date is not None
        and cutoff <= pd.Timestamp(p.target_date) <= pd.Timestamp(now)
    ]
    if not rows:
        return None

    frame = pd.DataFrame(
        {
            "station_id": [p.station_id for p in rows],
            "target_date": [pd.Timestamp(p.target_date) for p in rows],
            "prediction": [p.prediction for p in rows],
        }
    )

    reference = truth[["station_id", "date_local", "iqa"]].copy()
    reference["date_local"] = pd.to_datetime(reference["date_local"])

    merged = frame.merge(
        reference,
        left_on=["station_id", "target_date"],
        right_on=["station_id", "date_local"],
        how="inner",
    )
    if merged.empty:
        return None

    # Persistence for a target day is that station's value the day before.
    previous = reference.copy()
    previous["date_local"] = previous["date_local"] + pd.Timedelta(days=1)
    previous = previous.rename(columns={"iqa": "persistence"})
    merged = merged.merge(
        previous[["station_id", "date_local", "persistence"]],
        left_on=["station_id", "target_date"],
        right_on=["station_id", "date_local"],
        how="inner",
        suffixes=("", "_prev"),
    )
    if merged.empty:
        return None

    mae_model = float((merged["prediction"] - merged["iqa"]).abs().mean())
    mae_persistence = float((merged["persistence"] - merged["iqa"]).abs().mean())
    mase = mae_model / mae_persistence if mae_persistence else float("nan")

    return ScoreRecord(
        scored_at=pd.Timestamp(now).to_pydatetime(),
        window_days=window_days,
        n=len(merged),
        mae_model=mae_model,
        mae_persistence=mae_persistence,
        mase=mase,
    )


def run_scoring(
    store: Store | None = None, gold_path: str | None = None
) -> list[ScoreRecord]:
    """Score every window and persist the results."""
    from src.models.training_daily import GOLD_PATH

    store = store or Store()
    truth = pd.read_parquet(gold_path or GOLD_PATH)

    written: list[ScoreRecord] = []
    for window in WINDOWS:
        record = score_window(store, truth, window)
        if record is None:
            print(f"[skip] window={window}d: no matured predictions yet")
            continue
        store.write_score(record)
        written.append(record)
        print(
            f"[ok]   window={window}d  n={record.n}  "
            f"MASE={record.mase:.3f}  (model {record.mae_model:.3f} "
            f"vs persistence {record.mae_persistence:.3f})"
        )
    return written


if __name__ == "__main__":
    run_scoring()
