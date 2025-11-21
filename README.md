# AirQ Montréal – End-to-End MLOps Project

![CI](https://github.com/AyDaoud/airq_montreal_mlops/actions/workflows/ci.yml/badge.svg)

End-to-end MLOps pipeline for **air-quality forecasting in Montréal**.

The project takes hourly/daily air-quality data (IQA / pollutants), trains forecasting models (Random Forest, Prophet, LSTM), tracks experiments in **MLflow**, orchestrates training/forecast/monitoring with **Prefect**, and serves predictions via a **FastAPI** service that’s **containerized with Docker** and validated by **tests + GitHub Actions CI + pre-commit hooks**.

---

## 1. Problem & Goals

**Problem.**
Given historical air-quality measurements (daily IQA / pollutant values for Montréal), we want to:

- Forecast IQA/pollutant for the next *N* days/hours
- Track all model experiments and artifacts
- Run a **daily pipeline**: train → forecast → monitor
- Expose a **web API** that serves model predictions
- Package the service into a **Docker image**, ready for deployment to any cloud/container platform

This aligns with the MLOps Zoomcamp project rubric: experiment tracking, model registry, workflow orchestration, deployment, monitoring, tests, linting, and CI.

---

## 2. Tech Stack

- **Language:** Python 3.12
- **Data / ML:** pandas, scikit-learn, Prophet, PyTorch (LSTM)
- **Experiment Tracking & Registry:** MLflow (SQLite backend)
- **Orchestration:** Prefect flows (`orchestration/flow.py`)
- **Serving:** FastAPI + Uvicorn (`src/serving/app.py`)
- **Containerization:** Docker (see `Dockerfile`)
- **Testing:** pytest (`tests/`)
- **Code Quality:** black, flake8, pre-commit
- **CI:** GitHub Actions (`.github/workflows/ci.yml`)

---

## 3. Repository Structure

```text
.
├── orchestration/
│   └── flow.py                  # Prefect flows: daily pipeline (train → forecast → monitor)
├── scripts/
│   └── train_daily_iqa.py       # Train RF / Prophet / LSTM, log to MLflow + Model Registry
├── src/
│   ├── features/
│   │   └── build_features.py    # Feature engineering / daily IQA features
│   ├── models/
│   │   ├── forecast.py          # Batch forecasting CLI
│   │   ├── model_factory.py     # Saving / loading models and artifacts
│   │   └── training_daily.py    # Helper functions for daily training
│   ├── monitoring/
│   │   ├── check_iqa.py         # Monitoring script: metrics + drift + alert flag
│   │   └── metrics.py           # Regression metrics, drift score
│   └── serving/
│       └── app.py               # FastAPI app (`/health`, `/predict`)
├── tests/
│   ├── conftest.py
│   ├── test_api_basic.py        # Unit test for FastAPI predict endpoint
│   └── test_build_features.py   # Unit test for feature engineering
├── Dockerfile                   # Containerized API
├── makefile                     # Common dev commands (lint, test, docker, flow, …)
├── requirements.txt
├── setup.cfg                    # flake8 / pytest configuration
├── .pre-commit-config.yaml      # black, flake8, basic formatting hooks
└── .github/
    └── workflows/
        └── ci.yml               # GitHub Actions: run tests on push/PR
