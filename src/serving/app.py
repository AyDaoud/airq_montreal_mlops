from __future__ import annotations
import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load

#####
# 👇 default to RF model, but allow override from env
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/rf/model.pkl")
FEATURE_COLUMNS = os.environ.get("FEATURE_COLUMNS")  # optional comma-separated

app = FastAPI()


class PredictRequest(BaseModel):
    rows: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


def _load_feature_cols(df_sample: pd.DataFrame):
    if FEATURE_COLUMNS:
        return [c.strip() for c in FEATURE_COLUMNS.split(",") if c.strip()]
    # default heuristic: drop id/time/target-like cols
    return [
        c
        for c in df_sample.columns
        if c not in ("datetime", "station_id", "pollutant", "value", "target")
    ]


@app.post("/predict")
def predict(req: PredictRequest):
    if not os.path.exists(MODEL_PATH):
        return {"error": f"Model not found at {MODEL_PATH}"}
    model = load(MODEL_PATH)
    df = pd.DataFrame(req.rows)
    feature_cols = _load_feature_cols(df)
    preds = model.predict(df[feature_cols]).tolist()
    return {"predictions": preds, "n": len(preds), "features_used": feature_cols}
