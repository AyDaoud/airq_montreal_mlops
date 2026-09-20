"""FastAPI service for daily IQA prediction.

The model is loaded once at startup. A request whose columns do not match
the model's feature list is rejected with 422 naming the missing columns:
silently substituting a different feature set produces wrong numbers that
look right.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from joblib import load
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.monitoring.store import PredictionRecord, Store

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/rf/model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "artifacts/rf/feature_names.json")

_MODEL_CACHE: tuple[object, list[str]] | None = None


def _load_model() -> tuple[object, list[str]]:
    """Load the model and its feature list from disk."""
    model = load(MODEL_PATH)
    features_file = Path(FEATURES_PATH)
    if features_file.exists():
        feature_names = json.loads(features_file.read_text())
    elif hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        raise RuntimeError(
            f"No feature list available: {FEATURES_PATH} is absent and the "
            "model exposes no feature_names_in_."
        )
    return model, list(feature_names)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup.

    A load failure is logged rather than raised so /health stays reachable
    for diagnostics; /predict then fails loudly on first use.
    """
    global _MODEL_CACHE
    try:
        _MODEL_CACHE = _load_model()
    except Exception as exc:
        print(f"[warn] model not loaded at startup: {exc}")
        _MODEL_CACHE = None
    yield
    _MODEL_CACHE = None


app = FastAPI(title="AirQ Montreal", version="1.0.0", lifespan=lifespan)

if not getattr(app.state, "_instrumented", False):
    Instrumentator().instrument(app).expose(app, include_in_schema=False)
    app.state._instrumented = True

MODEL_NAME = os.getenv("MODEL_NAME", "rf")
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")

_STORE: Store | None = None


def _get_store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE


def _store_predictions(records: list[PredictionRecord]) -> None:
    """Separated so tests can make persistence fail without touching the store."""
    _get_store().log_predictions(records)


def log_predictions_safely(records: list[PredictionRecord]) -> None:
    """Persist predictions. A failure here must never fail the request."""
    try:
        _store_predictions(records)
    except Exception as exc:  # noqa: BLE001 - deliberately broad
        print(f"[warn] prediction logging failed: {exc}")


def _coerce_station_id(value: object) -> int | None:
    """Best-effort coercion of a caller-supplied station id.

    The request body is a free-form dict; ``station_id`` may be absent, a
    string, or otherwise not directly int-able. A bad value must not fail
    the request, so anything that does not cleanly convert becomes None.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_model() -> tuple[object, list[str]]:
    """Dependency returning the cached model. Tests override this."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _load_model()
    return _MODEL_CACHE


class PredictRequest(BaseModel):
    rows: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest, model_bundle=Depends(get_model)):
    if not req.rows:
        return {"n": 0, "preds": []}

    model, feature_names = model_bundle
    frame = pd.DataFrame(req.rows)

    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Request is missing required feature columns: {missing}",
        )

    preds = model.predict(frame[feature_names])

    request_id = str(uuid.uuid4())
    served_at = datetime.now(timezone.utc)
    log_predictions_safely(
        [
            PredictionRecord(
                request_id=request_id,
                served_at=served_at,
                station_id=_coerce_station_id(row.get("station_id")),
                target_date=None,
                prediction=float(value),
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                features_json=json.dumps(row, default=str),
            )
            for row, value in zip(req.rows, preds)
        ]
    )

    return {"n": len(preds), "preds": preds.tolist()}
