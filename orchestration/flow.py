# orchestration/flow.py
"""
Prefect workflows for the AirQ Montréal project.

We keep *all* orchestration logic in this single file to avoid
duplication:

- hourly pipeline (optional):
    Ingest hourly RSQA CSVs → build features → (you can add training later).

- daily_pipeline:
    Orchestrate the full daily pipeline by calling the existing
    CLIs:
        scripts.train_daily_iqa      (training + MLflow + registry)
        src.models.forecast          (batch forecast)
        src.monitoring.check_iqa     (monitoring)

Both flows can be built into Prefect deployments and scheduled.
"""

from __future__ import annotations

import sys
from pathlib import Path
import subprocess

import pandas as pd
from prefect import flow, task

from src.data.rsqa_ingest import ingest_rsqa_csv
from src.features.build_features import build_features


# ---------------------------------------------------------------------
# Hourly ingest + featurize (optional)
# ---------------------------------------------------------------------


@task
def ingest_hourly() -> list[str]:
    """
    Download / ingest hourly RSQA CSVs and convert them to parquet.

    Returns:
        List of paths to the parquet files.
    """
    urls = {
        "2025_2027": "https://donnees.montreal.ca/dataset/547b8052-1710-4d69-8760-beaa3aa35ec6/resource/6cf08815-49d2-4d2f-a400-ce36ee52b0fc/download/rsqa-indice-qualite-air-2025-2027.csv",
        "2022_2024": "https://donnees.montreal.ca/dataset/547b8052-1710-4d69-8760-beaa3aa35ec6/resource/0c325562-e742-4e8e-8c36-971f3c9e58cd/download/rsqa-indice-qualite-air-2022-2024.csv",
    }

    paths: list[str] = []
    for suffix, url in urls.items():
        pq_path = ingest_rsqa_csv(
            url,
            f"data/interim/rsqa_{suffix}.parquet",
            f"data/raw/rsqa_{suffix}.csv",
        )
        paths.append(str(pq_path))
    return paths


@task
def featurize_hourly(
    paths: list[str],
    station: str = "SAINT-DENIS",
    pollutant: str = "NO2",
):
    """
    Concatenate yearly parquet files, filter by station/pollutant,
    and call the existing feature builder.
    """
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    if "station_name" in df.columns:
        df = df[df["station_name"].str.contains(station, case=False, na=False)]
    if "pollutant" in df.columns:
        df = df[df["pollutant"] == pollutant]

    df = df.sort_values("datetime")

    feat_df, feats = build_features(
        df_pollut=df.rename(columns={"value": "value"}),  # ensure 'value' column
        df_weather=pd.DataFrame({"datetime": df["datetime"]}),  # dummy join for now
        pollutant=pollutant,
        lags=(1, 2, 3, 6, 12, 24),
    )

    Path("data/features").mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet("data/features/hourly_features.parquet", index=False)
    return feat_df, feats


# ---------------------------------------------------------------------
# Daily pipeline (train -> forecast -> monitor) using CLIs
# ---------------------------------------------------------------------


@task
def train_daily_model(model: str = "rf") -> None:
    """
    Call the existing daily training CLI.

    This:
      - trains the model,
      - logs metrics to MLflow,
      - and (for RF) registers the model in the MLflow Model Registry.
    """
    subprocess.run(
        [sys.executable, "-m", "scripts.train_daily_iqa", "--model", model],
        check=True,
    )


@task
def forecast_daily_model(model: str = "rf", freq: str = "D", horizon: int = 30) -> None:
    """
    Call the forecast CLI to produce batch predictions.
    """
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.models.forecast",
            "--model",
            model,
            "--freq",
            freq,
            "--horizon",
            str(horizon),
        ],
        check=True,
    )


@task
def monitor_daily_model(model: str = "rf") -> None:
    """
    Call the monitoring CLI to evaluate the latest predictions
    and log metrics / alerts to MLflow.
    """
    subprocess.run(
        [sys.executable, "-m", "src.monitoring.check_iqa", "--model", model],
        check=True,
    )


@flow(name="airq-daily-pipeline")
def daily_pipeline(
    model: str = "rf",
    freq: str = "D",
    horizon: int = 30,
) -> None:
    """
    Full daily pipeline for a given model:
        train → forecast → monitor

    You can change `model` to 'prophet' or 'lstm' if you want to orchestrate
    those instead of the default RF.
    """
    train_daily_model(model=model)
    forecast_daily_model(model=model, freq=freq, horizon=horizon)
    monitor_daily_model(model=model)


if __name__ == "__main__":
    # Allow: python -m orchestration.flow
    daily_pipeline()
