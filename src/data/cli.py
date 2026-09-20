"""Command line entry point for the data pipeline.

python -m src.data.cli ingest    # download sources into bronze/silver
python -m src.data.cli build     # bronze/silver -> gold
python -m src.data.cli all       # both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.aggregate import build_gold, build_silver_iqa, load_bronze_frames
from src.data.contracts import GOLD_DAILY_STATION_IQA, check_freshness, validate
from src.data.rsqa_ingest import ingest_historical, ingest_realtime
from src.data.stations import ingest_stations
from src.data.weather import ingest_weather_archive

GOLD_PATH = Path("data/gold/daily_station_iqa.parquet")
SILVER_IQA_PATH = Path("data/silver/iqa_hourly.parquet")
STATIONS_PATH = Path("data/silver/stations.parquet")
WEATHER_PATH = Path("data/silver/weather_hourly.parquet")
BRONZE_ROOT = Path("data/bronze/rsqa_iqa")


def cmd_ingest(args: argparse.Namespace) -> None:
    ingest_stations(silver_path=STATIONS_PATH)
    ingest_historical(force=getattr(args, "force", False))
    try:
        ingest_realtime()
    except Exception as exc:  # the realtime feed is known to 503 occasionally
        print(f"[warn] realtime ingest failed, continuing with history: {exc}")
    if not getattr(args, "skip_weather", False):
        ingest_weather_archive(stations_path=STATIONS_PATH, out_path=WEATHER_PATH)


def cmd_build(args: argparse.Namespace) -> None:
    frames = load_bronze_frames(BRONZE_ROOT)
    if not frames:
        raise SystemExit("No bronze data found. Run 'python -m src.data.cli ingest'.")

    silver = build_silver_iqa(frames)
    SILVER_IQA_PATH.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(SILVER_IQA_PATH, index=False)
    print(f"[ok]   silver iqa: {len(silver):,} rows -> {SILVER_IQA_PATH}")

    stations = pd.read_parquet(STATIONS_PATH)
    weather = pd.read_parquet(WEATHER_PATH) if WEATHER_PATH.exists() else None

    gold = build_gold(silver, stations, weather)
    validate(gold, GOLD_DAILY_STATION_IQA)
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(GOLD_PATH, index=False)
    print(f"[ok]   gold: {len(gold):,} rows -> {GOLD_PATH}")

    warning = check_freshness(gold["date_local"].max())
    if warning:
        print(f"[warn] FRESHNESS: {warning}")
    else:
        print("[ok]   freshness within tolerance")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_ingest(args)
    cmd_build(args)


def main() -> None:
    parser = argparse.ArgumentParser("airq data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="download sources into bronze/silver")
    ingest.add_argument("--force", action="store_true", help="ignore watermarks")
    ingest.add_argument("--skip-weather", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    build = sub.add_parser("build", help="bronze/silver -> gold")
    build.set_defaults(func=cmd_build)

    every = sub.add_parser("all", help="ingest then build")
    every.add_argument("--force", action="store_true")
    every.add_argument("--skip-weather", action="store_true")
    every.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
