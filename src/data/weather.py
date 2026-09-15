"""Open-Meteo weather ingestion.

Training uses the archive endpoint; inference uses the forecast endpoint.
``boundary_layer_height`` is deliberately absent: the archive returns null
for it, so training on it would create train/serve skew.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.data.sources import KNOWN_STATION_IDS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
)

EMPTY_COLUMNS = ["cell_id", "ts_utc", *HOURLY_VARS]


def parse_openmeteo(payload: dict, cell_id: str) -> pd.DataFrame:
    """Convert an Open-Meteo JSON response into a tidy hourly frame."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    data = {"cell_id": cell_id, "ts_utc": pd.to_datetime(times, utc=True)}
    for var in HOURLY_VARS:
        data[var] = pd.to_numeric(pd.Series(hourly.get(var, [None] * len(times))))
    return pd.DataFrame(data)


def _cell_to_latlon(cell_id: str) -> tuple[float, float]:
    lat_str, lon_str = cell_id.split("_")
    return float(lat_str), float(lon_str)


def fetch_archive(
    cell_id: str, start: str, end: str, timeout: int = 120
) -> pd.DataFrame:
    """Fetch historical hourly weather for one grid cell."""
    lat, lon = _cell_to_latlon(cell_id)
    response = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "UTC",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_openmeteo(response.json(), cell_id)


def fetch_forecast(cell_id: str, days: int = 3, timeout: int = 60) -> pd.DataFrame:
    """Fetch forecast hourly weather for one grid cell (used at inference)."""
    lat, lon = _cell_to_latlon(cell_id)
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(HOURLY_VARS),
            "forecast_days": days,
            "timezone": "UTC",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_openmeteo(response.json(), cell_id)


def ingest_weather_archive(
    stations_path: Path = Path("data/silver/stations.parquet"),
    out_path: Path = Path("data/silver/weather_hourly.parquet"),
    start: str = "2021-12-01",
    end: str | None = None,
) -> Path:
    """Fetch archive weather for every cell used by an IQA-reporting station.

    Restricted to KNOWN_STATION_IDS: the station dimension contains stations
    that report no IQA, and their cells would never join to anything.

    ``start`` precedes the IQA history by a month so lag features at the
    beginning of 2022 have weather to reference.
    """
    stations = pd.read_parquet(stations_path)
    reporting = stations[stations["station_id"].isin(KNOWN_STATION_IDS)]
    cells = sorted(reporting["cell_id"].unique())
    end = end or (pd.Timestamp.utcnow() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    frames = []
    for cell_id in cells:
        frame = fetch_archive(cell_id, start, end)
        frames.append(frame)
        print(f"[ok]   weather {cell_id}: {len(frame):,} hours")

    weather = pd.concat(frames, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(out_path, index=False)
    print(
        f"[ok]   weather total {len(weather):,} rows, {len(cells)} cells -> {out_path}"
    )
    return out_path
