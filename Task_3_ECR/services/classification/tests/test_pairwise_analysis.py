from unittest.mock import Mock

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

# ---------------- Test Setup ---------------- #


def mock_user():
    return {"user_id": "test_user", "token": "mock-token"}


@pytest.fixture
def client():
    app = FastAPI()

    # Mock Service
    mock_model_service = Mock()
    mock_model_service.pairwise_analysis = Mock(return_value={"status": "success"})

    @app.post("/classification/api/pairwise-analysis")
    async def post_pairwise_analysis(payload: dict, user: dict = Depends(mock_user)):
        try:
            f1 = payload.get("feature1")
            f2 = payload.get("feature2")
            if not f1 or not f2:
                raise HTTPException(status_code=400, detail="Missing 'feature1' or 'feature2'")

            color_by = payload.get("color_by")
            sample_size = int(payload.get("sample_size", 1000))

            return mock_model_service.pairwise_analysis(f1, f2, color_by, sample_size)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    app.mock_model_service = mock_model_service
    return TestClient(app)


# ---------------- Test Cases ---------------- #


def test_pairwise_analysis_success(client):
    """✔ API returns success for valid input"""
    payload = {"feature1": "age", "feature2": "income", "color_by": "loan_approved", "sample_size": 500}

    res = client.post("/classification/api/pairwise-analysis", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    client.app.mock_model_service.pairwise_analysis.assert_called_once()


def test_pairwise_analysis_missing_features(client):
    """❌ Missing feature parameters → 400"""
    payload = {"feature1": "age"}  # feature2 missing

    res = client.post("/classification/api/pairwise-analysis", json=payload)

    assert res.status_code == 400
    assert res.json()["detail"] == "Missing 'feature1' or 'feature2'"


def test_pairwise_analysis_internal_error(client):
    """💥 Exception while processing → 500 response"""
    client.app.mock_model_service.pairwise_analysis.side_effect = Exception("Boom!")

    payload = {"feature1": "age", "feature2": "income"}

    res = client.post("/classification/api/pairwise-analysis", json=payload)

    assert res.status_code == 500
    assert "Internal error" in res.json()["detail"]
