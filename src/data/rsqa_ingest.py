"""Ingest RSQA air-quality index data into the bronze layer.

Two sources, one schema:

* historical annual dumps  (column ``polluant``,  updated ~yearly)
* the realtime feed        (column ``pollutant``, updated daily)

Timestamps are UTC-5 year-round, NOT local-with-DST. Hour 02 exists on
spring-forward days, so ``tz_localize("America/Montreal")`` would raise
NonExistentTimeError. Localize to the fixed offset instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import sources
from src.data.ckan import resolve
from src.data.contracts import RAW_IQA_HISTORICAL, RAW_IQA_REALTIME, validate
from src.data.http import build_session, download
from src.data.watermark import should_skip, write_watermark

SOURCE_TZ = "Etc/GMT+5"  # POSIX sign inversion: this is UTC-5.
CIVIL_TZ = "America/Montreal"

CANONICAL_COLUMNS = [
    "station_id",
    "ts_utc",
    "ts_local",
    "pollutant",
    "value",
    "source",
]

# The realtime feed spells particulate matter "PM2.5"; the historical dumps
# (and the contracts.RAW_IQA_* schemas, and sources.POLLUTANTS) use "PM".
# Canonicalize before validation so both sources satisfy the same contract.
POLLUTANT_ALIASES = {"PM2.5": "PM", "PM2_5": "PM"}


def normalize_iqa(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize either raw IQA schema to the canonical silver shape."""
    if "polluant" in df.columns:
        pollutant_col = "polluant"
        schema = RAW_IQA_HISTORICAL
    elif "pollutant" in df.columns:
        pollutant_col = "pollutant"
        schema = RAW_IQA_REALTIME
    else:
        raise ValueError(
            "Frame has neither 'polluant' nor 'pollutant'; "
            f"got columns {list(df.columns)}"
        )

    df = df.copy()
    df[pollutant_col] = (
        df[pollutant_col].astype(str).str.strip().str.upper().replace(POLLUTANT_ALIASES)
    )

    df = validate(df, schema)

    ts_naive = pd.to_datetime(df["date"], format="%Y-%m-%d") + pd.to_timedelta(
        df["heure"].astype(int), unit="h"
    )
    ts_utc = ts_naive.dt.tz_localize(SOURCE_TZ).dt.tz_convert("UTC")

    out = pd.DataFrame(
        {
            "station_id": df["stationId"].astype("int16"),
            "ts_utc": ts_utc,
            "ts_local": ts_utc.dt.tz_convert(CIVIL_TZ),
            "pollutant": df[pollutant_col].astype(str).str.strip().str.upper(),
            "value": df["valeur"].astype("int16"),
            "source": source,
        }
    )
    return out[CANONICAL_COLUMNS]


def to_bronze(df: pd.DataFrame, bronze_dir: Path, partition: str) -> Path:
    """Write ``df`` to ``bronze_dir/partition/part.parquet``, overwriting.

    Overwrite rather than append is what makes re-running a day idempotent.
    """
    out_dir = Path(bronze_dir) / partition
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "part.parquet"
    df.to_parquet(path, index=False)
    return path


def ingest_historical(
    bronze_root: Path = Path("data/bronze/rsqa_iqa/historical"),
    watermark_path: Path = Path("data/_watermarks.json"),
    raw_dir: Path = Path("data/_raw"),
    session=None,
    force: bool = False,
) -> list[Path]:
    """Download and normalize the annual historical dumps."""
    session = session or build_session()
    written: list[Path] = []

    for resource in sources.IQA_HISTORICAL:
        resolved = resolve(resource, session=session)
        if not force and should_skip(
            watermark_path, resource.name, resolved.last_modified
        ):
            print(f"[skip] {resource.name} unchanged ({resolved.last_modified})")
            continue

        raw_path = download(
            resolved.url, Path(raw_dir) / f"{resource.name}.csv", session=session
        )
        raw = pd.read_csv(raw_path)
        normalized = normalize_iqa(raw, source="historical")
        path = to_bronze(normalized, bronze_root, partition=f"resource={resource.name}")
        written.append(path)
        print(f"[ok]   {resource.name}: {len(normalized):,} rows -> {path}")

        if resolved.last_modified:
            write_watermark(watermark_path, resource.name, resolved.last_modified)

    return written


def ingest_realtime(
    bronze_root: Path = Path("data/bronze/rsqa_iqa/realtime"),
    watermark_path: Path = Path("data/_watermarks.json"),
    raw_dir: Path = Path("data/_raw"),
    session=None,
) -> Path | None:
    """Fetch today's realtime slice. Safe to run many times per day."""
    session = session or build_session()
    resolved = resolve(sources.IQA_REALTIME, session=session)

    raw_path = download(
        resolved.url, Path(raw_dir) / "iqa_realtime.csv", session=session
    )
    raw = pd.read_csv(raw_path)
    if raw.empty:
        print("[warn] realtime feed returned no rows")
        return None

    normalized = normalize_iqa(raw, source="realtime")
    day = normalized["ts_local"].dt.date.max()
    path = to_bronze(normalized, bronze_root, partition=f"date={day}")
    write_watermark(watermark_path, "iqa_realtime", str(day))
    print(f"[ok]   realtime {day}: {len(normalized):,} rows -> {path}")
    return path
