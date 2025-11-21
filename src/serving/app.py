# src/serving/app.py
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

# Default paths; can be overridden with env vars in Docker
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/rf/model.pkl")
FEATURE_PATH = os.getenv("FEATURE_PATH", "artifacts/rf/feature_names.json")


class PredictRequest(BaseModel):
    rows: list[dict]


app = FastAPI(title="AirQ Montréal API")


def _load_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Load feature names from FEATURE_PATH if possible.

    If the saved feature names are not all present in the incoming DataFrame
    (e.g. in unit tests with a minimal payload), fall back to using all
    non-datetime columns.
    """
    p = Path(FEATURE_PATH)

    if p.exists():
        try:
            names = json.loads(p.read_text())
            # Use them only if they actually exist in df
            if all(name in df.columns for name in names):
                return names
        except Exception:
            # On any JSON/IO error, fall back below
            pass

    # Fallback: everything except datetime
    return [c for c in df.columns if c != "datetime"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    # 1) Load model
    if not os.path.exists(MODEL_PATH):
        return {"error": f"Model not found at {MODEL_PATH}"}

    model = load(MODEL_PATH)

    # 2) Build dataframe from incoming rows
    df = pd.DataFrame(req.rows)

    # 3) Choose feature columns (robust to test data)
    feature_cols = _load_feature_cols(df)
    X = df[feature_cols]

    # 4) Predict
    preds = model.predict(X)
    try:
        preds_list = preds.tolist()
    except AttributeError:
        preds_list = list(preds)

    # 5) JSON response
    return {
        "n": len(preds_list),
        "preds": preds_list,
    }
