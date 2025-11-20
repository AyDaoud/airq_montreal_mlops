# AirQ Montréal MLOps (Scaffold)

A minimal, **testable** end-to-end scaffold for an air-quality forecasting pipeline (PM2.5) in Montréal.

## What’s included
- Synthetic RSQA-like and weather data (72 hours) under `data/raw/`
- Ingestion → Features → Training (MLflow logging) → Serving (FastAPI)
- Prefect flow (`orchestration/flow.py`)
- **Pytest** covering ingestion, feature building, training, and API prediction

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
# Run the Prefect flow once:
python orchestration/flow.py
# Serve the API:
export MODEL_PATH=artifacts/model.pkl  # Windows PowerShell: $env:MODEL_PATH="artifacts/model.pkl"
uvicorn src.serving.app:app --reload --port 8000
# Test:
curl -X GET http://127.0.0.1:8000/health
```

> During tests, a model is trained if missing and saved to `artifacts/model.pkl`.
