
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").status_code == 200


def test_readiness():
    response = client.get("/ready")
    assert response.status_code == 200


def test_model_info():
    assert client.get("/model-info").status_code == 200


def test_valid_prediction():
    response = client.post("/predict", json={
        "consumption_kwh": 2.5,
        "hour": 18,
        "day_of_week": 2,
        "tariff_per_kwh": 8.0
    })
    assert response.status_code == 200


def test_invalid_hour():
    response = client.post("/predict", json={
        "consumption_kwh": 2.5,
        "hour": 25,
        "day_of_week": 2
    })
    assert response.status_code == 422


def test_negative_consumption():
    response = client.post("/predict", json={
        "consumption_kwh": -1,
        "hour": 18,
        "day_of_week": 2
    })
    assert response.status_code == 422
