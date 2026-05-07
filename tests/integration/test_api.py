"""
Integration tests for the FastAPI serving layer.
Requires a running PostgreSQL (provided by GitHub Actions service containers).
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Patch model loading so we don't need MLflow in tests
    import unittest.mock as mock
    with mock.patch("src.serving.api._load_model"):
        from src.serving.api import app
        yield TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "database" in data


def test_predict_returns_valid_shape(client):
    import unittest.mock as mock
    import numpy as np
    # Mock the model prediction and feature building
    with mock.patch("src.serving.api._model") as mock_model, \
         mock.patch("src.serving.api._build_feature_df") as mock_features:
        import pandas as pd
        mock_features.return_value = pd.DataFrame(
            [{"pickup_location_id": i, "pickup_hour": "2024-01-01T00:00:00Z"}
             for i in [1, 132, 161]]
        )
        mock_model.predict.return_value = np.array([10.0, 25.0, 8.0])

        resp = client.post("/predict", json={"location_ids": [1, 132, 161]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 3
        assert all("predicted_rides" in p for p in data["predictions"])
        assert all(p["predicted_rides"] >= 0 for p in data["predictions"])
