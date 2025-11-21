# src/serving/app.py
import os
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/rf/model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "artifacts/rf/feature_names.json")


class PredictRequest(BaseModel):
    rows: list[dict]


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


def _load_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Load feature column names from FEATURES_PATH if present and compatible
    with the incoming dataframe. If they don't match, fall back to a simple
    heuristic based on the current request.
    """
    path = Path(FEATURES_PATH)
    cols_from_meta: list[str] | None = None

    if path.exists():
        try:
            cols_from_meta = json.loads(path.read_text())
        except Exception:
            cols_from_meta = None

    # If we loaded meta AND all those columns exist in df, use them
    if cols_from_meta:
        missing = [c for c in cols_from_meta if c not in df.columns]
        if not missing:
            return cols_from_meta

    # Fallback: use non-datetime, non-string pollutant columns from the request
    ignore = {"datetime", "pollutant"}
    return [c for c in df.columns if c not in ignore]


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame(req.rows)

    # Empty request: return empty result but valid shape
    if df.empty:
        return {"n": 0, "preds": [], "predictions": []}

    if not os.path.exists(MODEL_PATH):
        return {"error": f"Model not found at {MODEL_PATH}"}

    model = load(MODEL_PATH)
    feature_cols = _load_feature_cols(df)
    X = df[feature_cols]

    preds = model.predict(X)
    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    # include BOTH keys: "preds" (for tests) and "predictions" (for humans)
    return {
        "n": len(preds),
        "preds": preds,
        "predictions": preds,
    }
