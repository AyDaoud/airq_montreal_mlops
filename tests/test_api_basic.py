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


def test_health_is_ok(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_expected_contract(client):
    payload = {"rows": [{"lag_1": 10, "lag_2": 12, "temp_mean": 5.0}]}
    body = client.post("/predict", json=payload).json()
    assert body["n"] == 1
    assert body["preds"] == [42.0]


def test_predict_handles_several_rows(client):
    rows = [{"lag_1": 1, "lag_2": 2, "temp_mean": 3.0} for _ in range(5)]
    body = client.post("/predict", json={"rows": rows}).json()
    assert body["n"] == 5
    assert len(body["preds"]) == 5


def test_missing_feature_returns_422_naming_the_column(client):
    """The old code silently substituted a different feature set."""
    response = client.post("/predict", json={"rows": [{"lag_1": 10}]})
    assert response.status_code == 422
    assert "lag_2" in response.text
    assert "temp_mean" in response.text


def test_extra_columns_are_ignored(client):
    payload = {"rows": [{"lag_1": 1, "lag_2": 2, "temp_mean": 3.0, "junk": "x"}]}
    assert client.post("/predict", json=payload).status_code == 200


def test_empty_rows_returns_empty_predictions(client):
    body = client.post("/predict", json={"rows": []}).json()
    assert body == {"n": 0, "preds": []}


def test_model_is_loaded_once_not_per_request(monkeypatch):
    """The old implementation called joblib.load inside the handler."""
    calls = []

    def fake_load(path):
        calls.append(path)
        return DummyModel()

    monkeypatch.setattr(serving, "load", fake_load)
    monkeypatch.setattr(serving, "_MODEL_CACHE", None)
    app.dependency_overrides.clear()

    row = {"lag_1": 1, "lag_2": 2, "temp_mean": 3.0}
    with TestClient(app) as client:
        client.post("/predict", json={"rows": [row]})
        client.post("/predict", json={"rows": [row]})
        client.post("/predict", json={"rows": [row]})

    assert len(calls) == 1, f"model loaded {len(calls)} times, expected 1"
