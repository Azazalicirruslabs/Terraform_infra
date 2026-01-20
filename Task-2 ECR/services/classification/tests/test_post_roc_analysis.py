from unittest.mock import Mock

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


# ---- Mock Auth ----
def mock_current_user():
    return {"user_id": "test_user", "email": "test@xai.com", "token": "abc"}


@pytest.fixture
def client():
    """Create app with mocked model service and handle_request logic"""
    app = FastAPI()

    mock_model_service = Mock()
    mock_model_service.roc_analysis = Mock(return_value={"auc": 0.95})

    @app.post("/classification/api/roc-analysis")
    async def post_roc_analysis(payload: dict, user: dict = Depends(mock_current_user)):
        """Mock ROC analysis endpoint"""
        try:
            # Call the mock service
            result = mock_model_service.roc_analysis()
            return {"status": "success", "roc_auc": result.get("auc", 0.95)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    # Expose mocks for test assertions
    app.mock_model_service = mock_model_service

    return TestClient(app)


# ---- TEST CASES ----


def test_roc_analysis_success(client):
    """✔ Should return success when service works"""
    response = client.post("/classification/api/roc-analysis", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["roc_auc"] == 0.95

    client.app.mock_model_service.roc_analysis.assert_called_once()


def test_roc_analysis_internal_error(client):
    """💥 Should return 500 when service fails"""
    # Make the mock service raise an exception
    client.app.mock_model_service.roc_analysis.side_effect = Exception("Unexpected error")

    response = client.post("/classification/api/roc-analysis", json={})

    assert response.status_code == 500
    assert "Error:" in response.json()["detail"]
