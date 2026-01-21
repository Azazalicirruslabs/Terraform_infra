from unittest.mock import Mock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient


# Mock authenticated user dependency
def mock_user():
    return {"user_id": "test", "email": "test@xai.com", "token": "mock-token"}


@pytest.fixture
def client():
    app = FastAPI()

    mock_model_service = Mock()
    mock_model_service.threshold_analysis = Mock(return_value={"thresholds": []})

    @app.post("/classification/api/threshold-analysis")
    async def post_threshold_analysis(num_thresholds: int = 50, user: dict = Depends(mock_user)):
        try:
            result = mock_model_service.threshold_analysis(num_thresholds)
            return {"status": "success", "analysis": "threshold", "result": result}
        except Exception as e:
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    app.mock_model_service = mock_model_service

    yield TestClient(app)


# -------------------------------- Test Cases -------------------------------- #


def test_threshold_analysis_success(client):
    """✔ Should return success when default threshold is used."""
    res = client.post("/classification/api/threshold-analysis")

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    client.app.mock_model_service.threshold_analysis.assert_called_once()
    client.app.mock_model_service.threshold_analysis.assert_called_with(50)


def test_threshold_analysis_custom_input(client):
    """✔ Should accept custom query parameter."""
    res = client.post("/classification/api/threshold-analysis?num_thresholds=80")

    assert res.status_code == 200
    client.app.mock_model_service.threshold_analysis.assert_called_with(80)


def test_threshold_analysis_internal_failure(client):
    """💥 Simulates unexpected internal error from model service."""
    client.app.mock_model_service.threshold_analysis.side_effect = Exception("Internal Failure")

    res = client.post("/classification/api/threshold-analysis?num_thresholds=60")

    assert res.status_code == 500
    assert "Error:" in res.json()["detail"]
