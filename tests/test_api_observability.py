import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.serving.app as serving
from src.serving.app import app, get_model

FEATURES = ["lag_1", "lag_2", "temp_mean"]


class DummyModel:
    feature_names_in_ = np.array(FEATURES)

    def predict(self, X):
        return np.array([42.0] * len(X))


@pytest.fixture
def client():
    app.dependency_overrides[get_model] = lambda: (DummyModel(), list(FEATURES))
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_metrics_endpoint_returns_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "# HELP" in response.text


def test_predict_logs_one_row_per_prediction(client, monkeypatch):
    logged = []
    monkeypatch.setattr(
        serving, "log_predictions_safely", lambda records: logged.extend(records)
    )
    rows = [{"lag_1": 1, "lag_2": 2, "temp_mean": 3.0} for _ in range(3)]
    client.post("/predict", json={"rows": rows})
    assert len(logged) == 3


def test_predict_still_succeeds_when_logging_fails(client, monkeypatch):
    """Observability must never take down the endpoint."""

    def explode(records):
        raise RuntimeError("disk full")

    monkeypatch.setattr(serving, "_store_predictions", explode)
    response = client.post(
        "/predict", json={"rows": [{"lag_1": 1, "lag_2": 2, "temp_mean": 3.0}]}
    )
    assert response.status_code == 200
    assert response.json()["n"] == 1


def test_predict_response_is_unchanged_by_instrumentation(client):
    body = client.post(
        "/predict", json={"rows": [{"lag_1": 1, "lag_2": 2, "temp_mean": 3.0}]}
    ).json()
    assert body == {"n": 1, "preds": [42.0]}
