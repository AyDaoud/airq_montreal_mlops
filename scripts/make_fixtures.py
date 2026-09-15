"""Regenerate test fixtures from the live sources.

Run manually:  .venv/bin/python -m scripts.make_fixtures
The generated files are committed; tests never hit the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from src.data import sources
from src.data.ckan import resolve
from src.data.http import build_session, download

OUT = Path("tests/fixtures")
RAW = Path("data/_fixture_raw")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    session = build_session()

    # Historical IQA: keep two stations across a DST transition so the
    # timezone tests have something real to assert against.
    hist = resolve(sources.IQA_HISTORICAL[0], session=session)
    hist_path = download(hist.url, RAW / "iqa_hist.csv", session=session)
    df = pd.read_csv(hist_path)
    sample = df[
        df["stationId"].isin([3, 6])
        & df["date"].isin(["2024-03-09", "2024-03-10", "2024-03-11", "2024-11-03"])
    ]
    sample.to_csv(OUT / "iqa_historical_sample.csv", index=False)
    print("historical sample rows:", len(sample))

    # Realtime IQA: whole file, it is tiny.
    rt = resolve(sources.IQA_REALTIME, session=session)
    rt_path = download(rt.url, RAW / "iqa_rt.csv", session=session)
    rt_df = pd.read_csv(rt_path)
    rt_df.head(200).to_csv(OUT / "iqa_realtime_sample.csv", index=False)
    print("realtime sample rows:", min(len(rt_df), 200))

    # Stations: keep the file verbatim, including the corrupt row and the
    # trailing all-NaN rows. The contract test depends on them being present.
    st = resolve(sources.STATIONS, session=session)
    st_path = download(st.url, RAW / "stations.csv", session=session)
    (OUT / "stations_sample.csv").write_bytes(st_path.read_bytes())
    print("stations bytes:", st_path.stat().st_size)

    # Open-Meteo archive response for one cell, two days.
    params = {
        "latitude": 45.5,
        "longitude": -73.6,
        "start_date": "2024-06-01",
        "end_date": "2024-06-02",
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "surface_pressure",
            ]
        ),
        "timezone": "UTC",
    }
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive", params=params, timeout=90
    )
    resp.raise_for_status()
    (OUT / "weather_archive_sample.json").write_text(json.dumps(resp.json(), indent=1))
    print("weather hours:", len(resp.json()["hourly"]["time"]))


if __name__ == "__main__":
    main()
