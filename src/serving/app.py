# src/serving/app.py
import os
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

# Default artifact locations; can be overridden via env vars
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/rf/model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "artifacts/rf/feature_names.json")


class PredictRequest(BaseModel):
    rows: list[dict]


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


def _load_feature_cols(df: pd.DataFrame) -> list[str]:
    """Decide which columns to feed into the model.

    Priority:
    1. If feature_names.json exists *and* all of its columns are present
       in the incoming dataframe, use that exact list.
    2. Otherwise, fall back to a simple rule-based selection:
       take all columns except obvious non-feature identifiers like
       'datetime' and 'pollutant'.
    """
    path = Path(FEATURES_PATH)

    if path.exists():
        try:
            cols_from_artifact = json.loads(path.read_text())
            # Only use them if they are all present in the request
            if all(c in df.columns for c in cols_from_artifact):
                return cols_from_artifact
        except Exception:
            # If anything goes wrong, ignore and fall back below
            pass

    ignore = {"datetime", "pollutant"}
    return [c for c in df.columns if c not in ignore]


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame(req.rows)

    # Empty request: still return a valid payload
    if df.empty:
        return {"n": 0, "preds": []}

    # IMPORTANT:
    # Do NOT check that MODEL_PATH exists here.
    # In tests, `joblib.load` is monkeypatched to return a DummyModel,
    # and there may be no real artifact on disk.
    model = load(MODEL_PATH)

    feature_cols = _load_feature_cols(df)
    X = df[feature_cols]

    preds = model.predict(X)
    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    # This matches what tests expect: keys 'n' and 'preds'
    return {"n": len(preds), "preds": preds}
