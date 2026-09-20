# orchestration/flow.py
"""
Prefect workflows for the AirQ Montréal project.

We keep *all* orchestration logic in this single file to avoid
duplication:

- daily_pipeline:
    Ingest today's data (stations, RSQA history + realtime), rebuild the
    silver/gold tables, then orchestrate training/forecasting/monitoring
    by calling the existing CLIs:
        scripts.train_daily_iqa      (training + MLflow + registry)
        src.models.forecast          (batch forecast)
        src.monitoring.check_iqa     (monitoring)

This flow can be built into a Prefect deployment and scheduled.
"""

from __future__ import annotations

import sys
import subprocess

from prefect import flow, task

from src.data.cli import cmd_build
from src.data.rsqa_ingest import ingest_historical, ingest_realtime
from src.data.stations import ingest_stations

# ---------------------------------------------------------------------
# Daily ingest + gold build
# ---------------------------------------------------------------------


@task
def ingest_daily() -> None:
    """Refresh the station dimension, backfill history, pull today's slice."""
    ingest_stations()
    ingest_historical()
    ingest_realtime()


@task
def build_gold_table() -> None:
    """Rebuild silver and gold from bronze."""
    import argparse

    cmd_build(argparse.Namespace())


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
        ingest → build → train → forecast → monitor

    You can change `model` to 'prophet' or 'lstm' if you want to orchestrate
    those instead of the default RF.
    """
    ingest_daily()
    build_gold_table()
    train_daily_model(model=model)
    forecast_daily_model(model=model, freq=freq, horizon=horizon)
    monitor_daily_model(model=model)


if __name__ == "__main__":
    # Allow: python -m orchestration.flow
    daily_pipeline()
