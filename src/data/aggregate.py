"""bronze -> silver -> gold transformations.

Silver normalizes and reconciles; gold produces one row per
(station_id, date_local) with weather, geography and targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.contracts import SILVER_IQA_HOURLY, validate
from src.data.sources import IQA_POOR_THRESHOLD, POLLUTANTS

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


def hourly_to_daily(silver: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly sub-indices into one row per (station_id, date_local).

    Daily IQA is the maximum over every (hour, pollutant) observation, which
    is identical to the max over hours of the hourly IQA because max is
    associative.
    """
    if silver.empty:
        return pd.DataFrame(columns=["station_id", "date_local", "iqa"])

    df = silver.copy()
    df["date_local"] = pd.to_datetime(df["ts_local"].dt.date)

    grouped = df.groupby(["station_id", "date_local"], sort=True)
    daily = grouped.agg(
        iqa=("value", "max"),
        n_hours_observed=("ts_utc", "nunique"),
    ).reset_index()

    # The pollutant attaining the daily maximum.
    idx = df.groupby(["station_id", "date_local"])["value"].idxmax()
    driving = df.loc[idx, ["station_id", "date_local", "pollutant"]].rename(
        columns={"pollutant": "driving_pollutant"}
    )
    daily = daily.merge(driving, on=["station_id", "date_local"], how="left")

    # Per-pollutant daily maxima as sub_<POLLUTANT> columns.
    wide = (
        df.pivot_table(
            index=["station_id", "date_local"],
            columns="pollutant",
            values="value",
            aggfunc="max",
        )
        .reindex(columns=list(POLLUTANTS))
        .add_prefix("sub_")
        .reset_index()
    )
    daily = daily.merge(wide, on=["station_id", "date_local"], how="left")

    daily["iqa"] = daily["iqa"].astype("int16")
    daily["n_hours_observed"] = daily["n_hours_observed"].clip(upper=24).astype("int8")
    return daily


def daily_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly weather to one row per (cell_id, date_local).

    Wind direction is reduced to a speed-weighted resultant unit vector, so
    that 359 degrees and 1 degree are neighbours rather than opposites.
    """
    if weather.empty:
        return pd.DataFrame(columns=["cell_id", "date_local"])

    df = weather.copy()
    # Open-Meteo is fetched with timezone="UTC" (see src.data.weather), so the
    # UTC calendar date already lines up with the archive's own bucketing.
    df["date_local"] = pd.to_datetime(df["ts_utc"].dt.date)

    radians = np.radians(df["wind_direction_10m"].astype(float))
    speed = df["wind_speed_10m"].astype(float)
    df["_u"] = speed * np.sin(radians)
    df["_v"] = speed * np.cos(radians)

    out = (
        df.groupby(["cell_id", "date_local"])
        .agg(
            temp_mean=("temperature_2m", "mean"),
            temp_min=("temperature_2m", "min"),
            temp_max=("temperature_2m", "max"),
            humidity_mean=("relative_humidity_2m", "mean"),
            precip_sum=("precipitation", "sum"),
            wind_speed_mean=("wind_speed_10m", "mean"),
            wind_speed_max=("wind_speed_10m", "max"),
            pressure_mean=("surface_pressure", "mean"),
            _u=("_u", "mean"),
            _v=("_v", "mean"),
        )
        .reset_index()
    )

    magnitude = np.hypot(out["_u"], out["_v"])
    safe = magnitude.replace(0, np.nan)
    out["wind_dir_sin"] = (out["_u"] / safe).fillna(0.0)
    out["wind_dir_cos"] = (out["_v"] / safe).fillna(0.0)
    return out.drop(columns=["_u", "_v"])


def add_targets(daily: pd.DataFrame) -> pd.DataFrame:
    """Attach h24/h48 regression and exceedance targets.

    Each station is reindexed onto a complete daily calendar before shifting,
    so a gap in the record never makes "tomorrow" mean several days later,
    and a shift never reaches across a station boundary.
    """
    if daily.empty:
        return daily.assign(
            target_iqa_h24=pd.Series(dtype="Int16"),
            target_iqa_h48=pd.Series(dtype="Int16"),
            target_exceed_h24=pd.Series(dtype="boolean"),
            target_exceed_h48=pd.Series(dtype="boolean"),
        )

    pieces = []
    for station_id, group in daily.groupby("station_id", sort=True):
        group = group.sort_values("date_local").set_index("date_local")
        calendar = pd.date_range(group.index.min(), group.index.max(), freq="D")
        filled = group.reindex(calendar)

        for horizon, shift in (("h24", -1), ("h48", -2)):
            future = filled["iqa"].shift(shift)
            filled[f"target_iqa_{horizon}"] = future.astype("Int16")
            exceed = future > IQA_POOR_THRESHOLD
            filled[f"target_exceed_{horizon}"] = exceed.where(future.notna()).astype(
                "boolean"
            )

        filled = filled.loc[group.index]
        filled["station_id"] = station_id
        filled.index.name = "date_local"
        pieces.append(filled.reset_index())

    return pd.concat(pieces, ignore_index=True)


def build_gold(
    silver_iqa: pd.DataFrame,
    stations: pd.DataFrame,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the gold table from silver inputs."""
    daily = hourly_to_daily(silver_iqa)
    daily = daily.merge(
        stations[["station_id", "name", "borough", "latitude", "longitude", "cell_id"]],
        on="station_id",
        how="left",
    )

    if weather is not None and not weather.empty:
        daily = daily.merge(
            daily_weather(weather), on=["cell_id", "date_local"], how="left"
        )

    daily = add_targets(daily)
    return daily.sort_values(["station_id", "date_local"]).reset_index(drop=True)
