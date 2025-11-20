# src/monitoring/check_iqa.py
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd

from evidently import ColumnMapping
from evidently.metric_preset import RegressionPreset, DataDriftPreset
from evidently.report import Report

from src.models.training_daily import _daily_df
from src.monitoring.metrics import regression_metrics, simple_drift_score


# ---------- helpers to load data ----------

def _load_truth() -> pd.DataFrame:
    """
    Load the daily IQA truth series from the same source used for training.
    Returns columns: [datetime, value].
    """
    df = _daily_df()[["datetime", "value"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["datetime", "value"]).sort_values("datetime")


def _default_pred_path(model: str) -> Path:
    if model == "rf":
        return Path("data/predictions/rf_forecast.parquet")
    if model == "prophet":
        return Path("data/predictions/prophet_forecast.parquet")
    if model == "lstm":
        return Path("data/predictions/lstm_forecast.parquet")
    raise ValueError(f"Unknown model: {model}")


def _load_predictions(model: str, path: Path | None) -> pd.DataFrame:
    """
    Load batch predictions for monitoring.
    Expects columns with datetime + forecast value (we normalize names).
    """
    path = path or _default_pred_path(model)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_parquet(path)

    rename = {}
    for c in df.columns:
        lc = c.lower()
        if "date" in lc or "time" in lc:
            rename[c] = "datetime"
        if lc in ("forecast", "pred", "prediction", "yhat"):
            rename[c] = "forecast"
    df = df.rename(columns=rename)

    if "datetime" not in df.columns or "forecast" not in df.columns:
        raise ValueError(
            f"Prediction file {path} must contain datetime and forecast columns. "
            f"Got columns: {list(df.columns)}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")

    return df.dropna(subset=["datetime", "forecast"]).sort_values("datetime")


def _align_truth_and_pred(truth: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join on datetime and keep only aligned rows.
    Returns columns: [datetime, value, forecast].
    """
    merged = truth.merge(preds, on="datetime", how="inner")
    merged = merged.dropna(subset=["value", "forecast"]).sort_values("datetime")
    return merged


# ---------- Evidently report ----------

def _build_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_dir: Path,
) -> Tuple[Path, bool]:
    """
    Build an Evidently regression + data-drift report and save it as HTML.
    Returns: (html_path, dataset_drift_detected).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evidently needs a column mapping
    column_mapping = ColumnMapping(
        target="value",
        prediction="forecast",
        numerical_features=["value", "forecast"],
        datetime_features=["datetime"],
    )

    report = Report(metrics=[RegressionPreset(), DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    html_path = out_dir / f"iqa_monitoring_{ts}.html"
    report.save_html(html_path)

    # Try to extract a global "dataset drift" flag from the report
    drift_detected = False
    try:
        result = report.as_dict()
    except AttributeError:
        result = report.dict()  # older/newer Evidently versions

    for m in result.get("metrics", []):
        # For DataDriftPreset, there's usually a DatasetDriftMetric / DataDriftTable-like metric
        name = m.get("metric", "").lower()
        if "datasetdrift" in name or "datadrifttable" in name:
            val = m.get("result") or m.get("value") or {}
            drift_detected = bool(val.get("dataset_drift", False))
            break

    return html_path, drift_detected


# ---------- main monitoring logic ----------

def run_monitoring(
    model: str,
    mlflow_uri: str | None = None,
    experiment: str = "AirQ-Monitoring",
    mae_threshold: float = 15.0,
    drift_threshold: float = 10.0,
    pred_path: str | None = None,
) -> tuple[dict, float, bool, Path | None]:
    """
    Core monitoring function (usable from CLI or Prefect later).

    Returns:
        metrics_dict, drift_score, alert_flag, html_report_path
    """
    tracking_uri = mlflow_uri or "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    truth = _load_truth()
    preds = _load_predictions(model, Path(pred_path) if pred_path else None)
    merged = _align_truth_and_pred(truth, preds)

    if merged.empty:
        print("⚠️  No overlapping timestamps between truth and predictions → nothing to monitor.")
        return {}, 0.0, False, None

    # Use ALL aligned rows for performance metrics
    y_true = merged["value"].to_numpy()
    y_pred = merged["forecast"].to_numpy()
    mets = regression_metrics(y_true, y_pred)

    # For drift, follow Zoomcamp idea: baseline vs recent window
    # reference = older part, current = last 30 points (or half if fewer)
    if len(merged) <= 60:
        k = len(merged) // 2
        reference = merged.iloc[:k]
        current = merged.iloc[k:]
    else:
        reference = merged.iloc[:-30]
        current = merged.iloc[-30:]

    drift = simple_drift_score(reference["value"].to_numpy(), current["value"].to_numpy())

    # Build Evidently report (drift + regression) from the same reference/current
    dashboards_dir = Path("dashboards")
    html_path, drift_flag_evidently = _build_evidently_report(reference, current, dashboards_dir)

    # Decide alert
    alert = (mets["mae"] > mae_threshold) or (drift > drift_threshold) or drift_flag_evidently

    # Log everything to MLflow
    with mlflow.start_run(run_name=f"monitor_{model}"):
        mlflow.log_param("model", model)
        mlflow.log_param("mae_threshold", mae_threshold)
        mlflow.log_param("drift_threshold", drift_threshold)
        mlflow.log_param("n_points", len(merged))

        for k, v in mets.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_metric("drift_median_abs", float(drift))
        mlflow.log_metric("drift_dataset_flag", float(drift_flag_evidently))
        mlflow.log_metric("alert_flag", float(alert))

        if html_path is not None:
            mlflow.log_artifact(str(html_path), artifact_path="evidently")

    return mets, drift, alert, html_path


def main() -> None:
    parser = argparse.ArgumentParser("Monitor IQA model predictions (Zoomcamp-style)")
    parser.add_argument("--model", choices=["rf", "prophet", "lstm"], default="rf")
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="AirQ-Monitoring")
    parser.add_argument("--mae-threshold", type=float, default=15.0)
    parser.add_argument("--drift-threshold", type=float, default=10.0)
    parser.add_argument("--pred-path", type=str, default=None)
    args = parser.parse_args()

    mets, drift, alert, html_path = run_monitoring(
        model=args.model,
        mlflow_uri=args.mlflow_uri,
        experiment=args.experiment,
        mae_threshold=args.mae_threshold,
        drift_threshold=args.drift_threshold,
        pred_path=args.pred_path,
    )

    if not mets:
        # No overlapping data, message already printed
        return

    print("=== Monitoring summary ===")
    print(f"Model: {args.model}")
    print(f"MAE : {mets['mae']:.3f}")
    print(f"RMSE: {mets['rmse']:.3f}")
    print(f"Drift score (median diff): {drift:.3f}")
    if html_path is not None:
        print(f"Evidently HTML report: {html_path}")

    if alert:
        print("⚠️  Alert: thresholds violated (MAE, drift or Evidently). Consider retraining.")
    else:
        print("✅  Within thresholds. No action required.")


if __name__ == "__main__":
    main()
