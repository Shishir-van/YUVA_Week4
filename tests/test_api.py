import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app import app

client = TestClient(app)


def make_valid_payload():
    dataset = load_breast_cancer(as_frame=True)
    row = dataset.data.iloc[0]
    return {"features": {key: float(value) for key, value in row.items()}}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_has_30_features():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["number_of_features"] == 30


def test_prediction_endpoint_returns_probabilities():
    response = client.post("/predict", json=make_valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in {"malignant", "benign"}
    assert set(body["probabilities"].keys()) == {"malignant", "benign"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.00001


def test_prediction_rejects_missing_features():
    response = client.post("/predict", json={"features": {"mean radius": 14.0}})
    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]
