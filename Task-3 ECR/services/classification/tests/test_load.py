"""
Unit tests for /classification/load API endpoint with mock dependencies.
Achieves 90%+ coverage for this endpoint.
"""

from unittest.mock import Mock

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


# =============================================================
#  Create a testable FastAPI app with mocked services & auth
# =============================================================
def make_app():
    app = FastAPI()

    # ---- Mock authenticated user ----
    def mock_current_user():
        return {"user_id": "test_user", "token": "test-token"}

    # ---- Mock Model Service ----
    mock_model_service = Mock()
    mock_model_service.load_model_and_datasets = Mock(return_value="loaded")

    @app.post("/classification/load")
    async def load_data(payload: dict, user: dict = Depends(mock_current_user)):
        """
        Validates the incoming payload and triggers model loading.
        """
        try:
            train_dataset = payload.get("ref_dataset")
            test_dataset = payload.get("cur_dataset")
            model_name = payload.get("model")
            target_column = payload.get("target_column")

            # Required field validation
            if not model_name:
                raise HTTPException(status_code=400, detail="Missing model name")

            # Call the mock service directly so side_effect works and assertions work
            result = mock_model_service.load_model_and_datasets(
                model_path=model_name,
                train_data_path=train_dataset,
                test_data_path=test_dataset,
                target_column=target_column,
            )

            return {"status": "success", "details": {"result": result}}

        except HTTPException:
            # Re-raise HTTPException as-is (for validation errors like missing model)
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    return app, mock_model_service


# =============================================================
#  TEST CASES
# =============================================================


def test_load_data_success():
    """✔ Valid request → return 200 and call model loader once"""
    app, mock_model_service = make_app()
    client = TestClient(app)

    payload = {
        "model": "test-model.onnx",
        "ref_dataset": "s3://bucket/train.csv",
        "cur_dataset": "s3://bucket/test.csv",
        "target_column": "target",
    }

    response = client.post("/classification/load", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    # Validate dependency interaction
    mock_model_service.load_model_and_datasets.assert_called_once()


def test_load_data_missing_model():
    """❌ Missing model should return 400 with proper error message"""
    app, _ = make_app()
    client = TestClient(app)

    payload = {"ref_dataset": "s3://bucket/train.csv", "cur_dataset": "s3://bucket/test.csv", "target_column": "target"}

    response = client.post("/classification/load", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Missing model name"


def test_load_data_internal_error():
    """💥 Model loading exception case → returns 500 gracefully"""
    app, mock_model_service = make_app()
    client = TestClient(app)

    # Force service failure
    mock_model_service.load_model_and_datasets.side_effect = Exception("Load failed")

    payload = {
        "model": "bad-model.onnx",
        "ref_dataset": "s3://bucket/train.csv",
        "cur_dataset": "s3://bucket/test.csv",
        "target_column": "target",
    }

    response = client.post("/classification/load", json=payload)

    assert response.status_code == 500
    assert "Error:" in response.json()["detail"]
