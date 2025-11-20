from fastapi.testclient import TestClient
import numpy as np  # 👈 add this
import src.serving.app as serving_module
from src.serving.app import app


def test_health_ok():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


class DummyModel:
    def predict(self, X):
        # Return a NumPy array, like sklearn does
        return np.array([0.0] * len(X))


def test_predict_with_dummy_model(monkeypatch):
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
    assert body["n"] == 1
    assert body["predictions"] == [0.0]
