# tests/test_api_basic.py
import numpy as np
from fastapi.testclient import TestClient

import src.serving.app as serving_module
from src.serving.app import app


class DummyModel:
    def predict(self, X):
        # return something that behaves like a NumPy array
        return np.array([42] * len(X))


def test_predict_with_dummy_model(monkeypatch):
    # Patch joblib.load inside serving module to return dummy model
    def fake_load(path):
        return DummyModel()

    monkeypatch.setattr(serving_module, "load", fake_load, raising=True)

    client = TestClient(app)

    payload = {
        "rows": [
            {
                "datetime": "2024-01-01T00:00:00",
                "station_id": 1,
                "pollutant": "NO2",
                "value": 20,
            }
        ]
    }

    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    body = resp.json()

    # API contract we enforce
    assert body["n"] == 1
    assert body["preds"] == [42]
