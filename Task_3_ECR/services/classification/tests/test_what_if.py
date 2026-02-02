from unittest.mock import Mock

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


# Mock authenticated user
def mock_user():
    return {"user_id": "test_user", "token": "mock-token"}


@pytest.fixture
def client():
    """Mock FastAPI app with What-If endpoint"""
    app = FastAPI()

    mock_model_service = Mock()
    mock_model_service.perform_what_if = Mock(return_value={"impact": "positive"})

    @app.post("/classification/what-if")
    async def perform_what_if(payload: dict, user: dict = Depends(mock_user)):
        features = payload.get("features")
        try:
            result = mock_model_service.perform_what_if(features)
            return {"status": "success", "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    app.mock_model_service = mock_model_service

    yield TestClient(app)


# =================== UNIT TESTS =================== #


def test_what_if_success(client):
    """✔ Should return success with valid features list."""
    payload = {"features": {"age": 45, "income": 70000}}

    response = client.post("/classification/what-if", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    client.app.mock_model_service.perform_what_if.assert_called_once_with(payload["features"])


def test_what_if_missing_features(client):
    """❌ Missing features should still call function with None, but success should return."""
    payload = {}  # Missing "features"

    response = client.post("/classification/what-if", json=payload)

    assert response.status_code == 200  # Endpoint doesn’t error on missing features
    client.app.mock_model_service.perform_what_if.assert_called_once_with(None)


def test_what_if_internal_error(client):
    """💥 Internal exception should produce graceful failure."""
    client.app.mock_model_service.perform_what_if.side_effect = Exception("Unexpected Failure")

    response = client.post("/classification/what-if", json={"features": {}})

    assert response.status_code == 500
    assert "Error:" in response.json()["detail"]
