# scripts/train_daily_iqa.py
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow as mlf
import mlflow.sklearn as mlf_sklearn  # for sklearn model logging

from src.models.training_daily import train_rf, train_prophet, train_lstm
from src.models.model_factory import save_sklearn, save_prophet, save_lstm


def main() -> None:
    parser = argparse.ArgumentParser(
        "Train Daily IQA with RF / Prophet / LSTM and log to MLflow + Model Registry"
    )
    parser.add_argument("--model", choices=["rf", "prophet", "lstm"], default="rf")
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="AirQ-Train-Daily")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    # --------- MLflow tracking setup ---------
    tracking_uri = (
        args.mlflow_uri
        or "sqlite:///C:/Users/AU51870/Downloads/airq_montreal_mlops/mlflow.db"
    )
    mlf.set_tracking_uri(tracking_uri)
    mlf.set_experiment(args.experiment)

    art_dir = Path("artifacts") / args.model
    art_dir.mkdir(parents=True, exist_ok=True)

    with mlf.start_run(run_name=f"train_{args.model}") as run:
        mlf.log_param("model", args.model)
        mlf.log_param("epochs", args.epochs)
        mlf.log_param("seq_len", args.seq_len)

        registry_name = None  # default: don't register

        if args.model == "rf":
            # ------------- Random Forest -------------
            model, feats, mets = train_rf()
            save_sklearn(model, feats, art_dir)

            # Log as MLflow sklearn model & mark for registry
            mlf_sklearn.log_model(model, artifact_path="model")
            registry_name = "airq-daily-rf"

        elif args.model == "prophet":
            # ------------- Prophet -------------
            model, feats, mets = train_prophet()
            save_prophet(model, art_dir)

            # OPTIONAL: log/register Prophet (requires mlflow[prophet])
            try:
                import mlflow.prophet as mlf_prophet  # type: ignore

                mlf_prophet.log_model(model, artifact_path="model")
                registry_name = "airq-daily-prophet"
            except Exception as e:
                print(f"⚠️ Could not log Prophet model as MLflow model: {e}")
                registry_name = None

        else:
            # ------------- LSTM -------------
            state, info, mets = train_lstm(
                seq_len=args.seq_len,
                epochs=args.epochs,
            )
            save_lstm(state, info["seq_len"], info["mean"], info["std"], art_dir)
            # (we’re not registering the LSTM for now)
            registry_name = None

        # Log metrics & artifacts
        for k, v in mets.items():
            mlf.log_metric(k, float(v))
        for p in art_dir.glob("*"):
            mlf.log_artifact(str(p))

        # --------- Register in MLflow Model Registry ---------
        if registry_name is not None:
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                mv = mlf.register_model(model_uri, registry_name)
                print(
                    f"✅ Registered model '{registry_name}' as version {mv.version} "
                    f"from {model_uri}"
                )
            except Exception as e:
                print(f"⚠️ Could not register model in MLflow Model Registry: {e}")

        print(f"✅ Trained {args.model}. Metrics: {mets}")
        print(f"Artifacts in: {art_dir}")


if __name__ == "__main__":
    main()
