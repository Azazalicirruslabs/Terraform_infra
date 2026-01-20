from unittest.mock import Mock

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

# -------------------------- Test Setup --------------------------


def mock_user():
    return {"user_id": "test_user", "token": "mock-token"}


def mock_db():
    """Fake DB session"""
    return None


@pytest.fixture
def client():
    """Create a test app with a mock /classification/analysis/overview endpoint"""
    app = FastAPI()

    # Mock model service
    mock_model_service = Mock()
    mock_model_service.get_model_overview = Mock(return_value={"status": "success"})
    mock_model_service.load_model_and_datasets = Mock(return_value=None)

    @app.post("/classification/analysis/overview")
    async def get_overview(payload: dict, user: dict = Depends(mock_user)):
        """Mock overview endpoint matching expected behavior"""
        try:
            model_url = payload.get("model")
            train_url = payload.get("ref_dataset")
            test_url = payload.get("cur_dataset")
            target_col = payload.get("target_column")

            # Validate required URLs
            if not model_url or not train_url:
                raise HTTPException(status_code=400, detail="Missing S3 URLs")

            # Simulate loading
            mock_model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_url,
                test_data_path=test_url,
                target_column=target_col,
            )

            # Return overview response
            return mock_model_service.get_model_overview()

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model/data: {str(e)}")

    # Store mock on app for test access
    app.mock_model_service = mock_model_service
    return TestClient(app)


# -------------------------- Test Data --------------------------

VALID_PAYLOAD = {
    "model": "s3://bucket/model.onnx",
    "ref_dataset": "s3://bucket/train.csv",
    "cur_dataset": "s3://bucket/test.csv",
    "target_column": "target",
}


# -------------------------- Tests --------------------------


def test_overview_success(client):
    """✔ SUCCESS: Valid request returns overview response"""
    payload = {
        "model": "s3://bucket/model.onnx",
        "ref_dataset": "s3://bucket/train.csv",
        "cur_dataset": "s3://bucket/test.csv",
        "target_column": "target",
    }

    res = client.post("/classification/analysis/overview", json=payload)

    assert res.status_code == 200
    assert res.json()["status"] == "success"
    client.app.mock_model_service.load_model_and_datasets.assert_called_once()


def test_overview_missing_urls(client):
    """❌ 400: Missing model or dataset URLs"""
    invalid_payload = {
        "model": "",
        "ref_dataset": "",
        "cur_dataset": "",
        "target_column": "target",
    }

    res = client.post("/classification/analysis/overview", json=invalid_payload)

    assert res.status_code == 400
    assert "Missing S3 URLs" in res.json()["detail"]


def test_overview_model_load_failure(client):
    """💥 500: Model loading failure handled gracefully"""
    # Make the mock service raise an exception
    client.app.mock_model_service.load_model_and_datasets.side_effect = Exception("Load Failed")

    payload = {
        "model": "s3://bucket/model.onnx",
        "ref_dataset": "s3://bucket/train.csv",
        "cur_dataset": "s3://bucket/test.csv",
        "target_column": "target",
    }

    res = client.post("/classification/analysis/overview", json=payload)

    assert res.status_code == 500
    assert "Error loading model/data" in res.json()["detail"]
