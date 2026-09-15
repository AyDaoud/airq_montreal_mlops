"""bronze -> silver -> gold transformations.

Silver normalizes and reconciles; gold produces one row per
(station_id, date_local) with weather, geography and targets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.contracts import SILVER_IQA_HOURLY, validate

SILVER_COLUMNS = ["station_id", "ts_utc", "ts_local", "pollutant", "value", "source"]

# Historical first: the annual dump is the corrected official record and
# must win over any realtime row covering the same hour.
_SOURCE_PRIORITY = {"historical": 0, "realtime": 1}


def build_silver_iqa(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate bronze frames, reconcile overlaps, validate."""
    if not frames:
        return pd.DataFrame(columns=SILVER_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined["_priority"] = combined["source"].map(_SOURCE_PRIORITY).fillna(9)
    combined = combined.sort_values(["station_id", "ts_utc", "pollutant", "_priority"])
    combined = combined.drop_duplicates(
        subset=["station_id", "ts_utc", "pollutant"], keep="first"
    )
    combined = combined.drop(columns="_priority").reset_index(drop=True)
    combined = combined.sort_values(["ts_utc", "station_id", "pollutant"]).reset_index(
        drop=True
    )
    validate(combined, SILVER_IQA_HOURLY)
    return combined[SILVER_COLUMNS]


def load_bronze_frames(bronze_root: Path) -> list[pd.DataFrame]:
    """Read every ``part.parquet`` under ``bronze_root``."""
    root = Path(bronze_root)
    if not root.exists():
        return []
    return [pd.read_parquet(p) for p in sorted(root.rglob("part.parquet"))]
