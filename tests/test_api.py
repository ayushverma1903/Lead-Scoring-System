"""
test_api.py
-----------
Tests for the FastAPI backend endpoints.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

# We use the startup event to load the model for tests
@pytest.fixture(autouse=True)
def run_lifespan():
    with TestClient(app) as client:
        yield client

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "accuracy" in response.json()

def test_single_predict():
    data = {
        "data": {
            "TotalVisits": 5,
            "Total Time Spent on Website": 800,
            "Page Views Per Visit": 2.5
        }
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "Conversion_Probability" in response.json()

def test_batch_predict():
    data = {
        "leads": [
            {
                "TotalVisits": 5,
                "Total Time Spent on Website": 800
            },
            {
                "TotalVisits": 2,
                "Total Time Spent on Website": 100
            }
        ]
    }
    response = client.post("/batch_predict", json=data)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 2
