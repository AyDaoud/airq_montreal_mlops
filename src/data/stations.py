"""Station dimension: geography, names, and weather-cell assignment.

The upstream file is dirty: UTF-8 BOM, a duplicated header fragment, five
trailing all-NaN rows, and station 62 carrying latitude 4.504576e+07.
Cleaning happens here, once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import sources
from src.data.ckan import resolve
from src.data.contracts import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    RAW_STATIONS,
    validate,
)
from src.data.http import build_session, download

CELL_RESOLUTION = 0.1  # ERA5-Land grid; finer rounding would duplicate fetches.


def assign_cell(latitude: float, longitude: float) -> str:
    """Map a coordinate to its Open-Meteo grid cell identifier."""
    lat = round(round(latitude / CELL_RESOLUTION) * CELL_RESOLUTION, 1)
    lon = round(round(longitude / CELL_RESOLUTION) * CELL_RESOLUTION, 1)
    return f"{lat:.1f}_{lon:.1f}"


def load_stations(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the raw stations frame.

    Note: ``statut`` is deliberately NOT used as a filter. Stations 28, 50
    and 66 are marked ``ferme`` yet report data through 2026-01-18.
    """
    df = raw.dropna(subset=["numero_station"]).copy()
    df = df[
        df["latitude"].between(LAT_MIN, LAT_MAX)
        & df["longitude"].between(LON_MIN, LON_MAX)
    ]
    validate(df, RAW_STATIONS)

    out = pd.DataFrame(
        {
            "station_id": df["numero_station"].astype("int16"),
            "name": df["nom"].astype(str).str.strip(),
            "borough": df["arrondissement_ville"].astype(str).str.strip(),
            "latitude": df["latitude"].astype("float32"),
            "longitude": df["longitude"].astype("float32"),
        }
    )
    out["cell_id"] = [
        assign_cell(lat, lon) for lat, lon in zip(df["latitude"], df["longitude"])
    ]
    return out.drop_duplicates(subset=["station_id"]).reset_index(drop=True)


def ingest_stations(
    silver_path: Path = Path("data/silver/stations.parquet"),
    raw_dir: Path = Path("data/_raw"),
    session=None,
) -> Path:
    """Download, clean and persist the station dimension."""
    session = session or build_session()
    resolved = resolve(sources.STATIONS, session=session)
    raw_path = download(resolved.url, Path(raw_dir) / "stations.csv", session=session)

    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    stations = load_stations(raw)

    silver_path = Path(silver_path)
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    stations.to_parquet(silver_path, index=False)
    print(
        f"[ok]   stations: {len(stations)} rows, "
        f"{stations['cell_id'].nunique()} weather cells -> {silver_path}"
    )
    return silver_path
