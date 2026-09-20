"""pandera contracts for every external data boundary.

A contract failure is a loud, actionable error naming the offending rows.
This deliberately replaces column-name guessing, which fails silently.
"""

from __future__ import annotations

import pandas as pd

try:  # pandera >= 0.23
    from pandera.pandas import Check, Column, DataFrameSchema
    from pandera.errors import SchemaError, SchemaErrors
except ImportError:  # pandera < 0.23
    from pandera import Check, Column, DataFrameSchema
    from pandera.errors import SchemaError, SchemaErrors

# Montreal island bounding box, generous by ~0.1 degrees.
LAT_MIN, LAT_MAX = 45.2, 45.8
LON_MIN, LON_MAX = -74.1, -73.4

VALID_POLLUTANTS = ["PM", "O3", "NO2", "SO2", "CO"]


class ContractError(ValueError):
    """Raised when a frame violates its contract."""


RAW_IQA_HISTORICAL = DataFrameSchema(
    {
        "stationId": Column(int, Check.ge(0), nullable=False),
        "polluant": Column(str, Check.isin(VALID_POLLUTANTS), nullable=False),
        "valeur": Column(int, [Check.ge(0), Check.le(1000)], nullable=False),
        "date": Column(str, nullable=False),
        "heure": Column(int, [Check.ge(0), Check.le(23)], nullable=False),
    },
    strict=False,
    coerce=True,
    name="RAW_IQA_HISTORICAL",
)

RAW_IQA_REALTIME = DataFrameSchema(
    {
        "stationId": Column(int, Check.ge(0), nullable=False),
        "pollutant": Column(str, Check.isin(VALID_POLLUTANTS), nullable=False),
        "valeur": Column(int, [Check.ge(0), Check.le(1000)], nullable=False),
        "date": Column(str, nullable=False),
        "heure": Column(int, [Check.ge(0), Check.le(23)], nullable=False),
    },
    strict=False,
    coerce=True,
    name="RAW_IQA_REALTIME",
)

RAW_STATIONS = DataFrameSchema(
    {
        "numero_station": Column("Int64", Check.ge(0), nullable=False),
        "nom": Column(str, nullable=True),
        "latitude": Column(float, Check.in_range(LAT_MIN, LAT_MAX), nullable=False),
        "longitude": Column(float, Check.in_range(LON_MIN, LON_MAX), nullable=False),
    },
    strict=False,
    coerce=True,
    name="RAW_STATIONS",
)

SILVER_IQA_HOURLY = DataFrameSchema(
    {
        "station_id": Column("int16", nullable=False),
        "ts_utc": Column("datetime64[ns, UTC]", nullable=False),
        "pollutant": Column(str, Check.isin(VALID_POLLUTANTS), nullable=False),
        "value": Column("int16", Check.ge(0), nullable=False),
    },
    strict=False,
    coerce=True,
    unique=["station_id", "ts_utc", "pollutant"],
    name="SILVER_IQA_HOURLY",
)

SILVER_WEATHER_HOURLY = DataFrameSchema(
    {
        "cell_id": Column(str, nullable=False),
        "ts_utc": Column("datetime64[ns, UTC]", nullable=False),
        "wind_direction_10m": Column(float, Check.in_range(0, 360), nullable=True),
        "precipitation": Column(float, Check.ge(0), nullable=True),
    },
    strict=False,
    coerce=True,
    unique=["cell_id", "ts_utc"],
    name="SILVER_WEATHER_HOURLY",
)

GOLD_DAILY_STATION_IQA = DataFrameSchema(
    {
        "station_id": Column("int16", nullable=False),
        "date_local": Column("datetime64[ns]", nullable=False),
        "iqa": Column("int16", Check.ge(0), nullable=False),
        "driving_pollutant": Column(str, Check.isin(VALID_POLLUTANTS), nullable=False),
        "n_hours_observed": Column("int8", [Check.ge(1), Check.le(24)], nullable=False),
    },
    strict=False,
    coerce=True,
    unique=["station_id", "date_local"],
    name="GOLD_DAILY_STATION_IQA",
)


def validate(df: pd.DataFrame, schema: DataFrameSchema) -> pd.DataFrame:
    """Validate ``df`` against ``schema``, raising :class:`ContractError`.

    Wrapping pandera's exceptions keeps callers free of pandera imports and
    guarantees the schema name appears in the message.
    """
    try:
        return schema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise ContractError(f"{schema.name} contract violated:\n{exc}") from exc


def check_freshness(newest: pd.Timestamp, max_age_days: int = 2) -> str | None:
    """Return a warning string when data is staler than ``max_age_days``.

    This is the check that surfaces the 8-month staleness of the historical
    dump described in the spec, section 2.2.
    """
    now = pd.Timestamp.now(tz="UTC").normalize()
    newest = pd.Timestamp(newest)
    if newest.tzinfo is None:
        newest = newest.tz_localize("UTC")
    age = (now - newest.normalize()).days
    if age > max_age_days:
        return (
            f"Data is {age} days old (newest={newest.date()}, allowed={max_age_days})"
        )
    return None
