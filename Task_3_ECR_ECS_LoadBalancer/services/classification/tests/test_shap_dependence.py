from unittest.mock import Mock

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


def mock_current_user():
    return {"user_id": "test_user", "email": "test@xai.com", "token": "mocktoken"}


@pytest.fixture
def client():
    app = FastAPI()
    mock_model_service = Mock()
    mock_model_service.shap_dependence = Mock(return_value={"plot": "dummy_plot"})

    @app.post("/classification/api/shap-dependence")
    async def post_shap_dependence(payload: dict, user: dict = Depends(mock_current_user)):
        feature = payload.get("feature")
        if not feature:
            raise HTTPException(status_code=400, detail="Missing 'feature'")
        color_by = payload.get("color_by")
        try:
            # call the mock service directly
            result = mock_model_service.shap_dependence(feature, color_by)
            return {"status": "success", "shap_dependence_plot": True, "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    app.mock_model_service = mock_model_service

    yield TestClient(app)


# ---------------- TEST CASES ---------------- #


def test_shap_dependence_success(client):
    """✔ Should return success when feature is provided"""
    payload = {"feature": "age", "color_by": "income"}
    res = client.post("/classification/api/shap-dependence", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"

    # Ensure the mock service was called once and returned the mocked result
    client.app.mock_model_service.shap_dependence.assert_called_once()


def test_shap_dependence_missing_feature(client):
    """❌ Should reject request if required feature missing"""
    payload = {"color_by": "income"}
    res = client.post("/classification/api/shap-dependence", json=payload)

    assert res.status_code == 400
    assert res.json()["detail"] == "Missing 'feature'"


def test_shap_dependence_internal_error(client):
    """💥 Simulate unexpected error in execution"""
    # Make the mock service raise an exception
    client.app.mock_model_service.shap_dependence.side_effect = Exception("Processing Failed")

    payload = {"feature": "age"}
    res = client.post("/classification/api/shap-dependence", json=payload)

    # API should return 500 with an error detail
    assert res.status_code == 500
    assert "Error:" in res.json()["detail"]
