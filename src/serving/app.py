"""FastAPI service for daily IQA prediction.

The model is loaded once at startup. A request whose columns do not match
the model's feature list is rejected with 422 naming the missing columns:
silently substituting a different feature set produces wrong numbers that
look right.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel

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
    return {"n": len(preds), "preds": preds.tolist()}
