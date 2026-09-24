"""
Unit tests for FastAPI REST API endpoints.
Tests:
- GET / (Static frontend delivery)
- GET /api/health
- GET /api/predict (Horizons 1, 7, 14 and invalid horizon handling)
- GET /api/history
- GET /api/metrics
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_api_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["market"] == "Ujjain APMC, Madhya Pradesh"
    assert "latest_dataset_date" in data
    assert "total_daily_records" in data


@pytest.mark.parametrize("horizon", [1, 7, 14])
def test_api_predict_valid_horizons(horizon):
    response = client.get(f"/api/predict?horizon={horizon}")
    assert response.status_code == 200
    data = response.json()
    assert data["horizon_days"] == horizon
    assert "latest_actual_price" in data
    assert len(data["daily_forecasts"]) == horizon


def test_api_predict_invalid_horizon():
    response = client.get("/api/predict?horizon=5")
    assert response.status_code == 400
    data = response.json()
    assert "Invalid horizon" in data["detail"]


def test_api_history_endpoint():
    response = client.get("/api/history?limit=30")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 30
    assert len(data["records"]) == 30
    assert "modal_price" in data["records"][0]
    assert "date" in data["records"][0]


def test_api_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "test_metrics" in data
    assert "MAE" in data["test_metrics"]
