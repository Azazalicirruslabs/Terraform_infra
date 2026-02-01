from unittest.mock import Mock

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

# ---------------- Setup Test App ---------------- #


def mock_user():
    return {"user_id": "test_user", "token": "mock-token"}


@pytest.fixture
def client():
    """Creates a test client with a mocked model service."""
    app = FastAPI()

    mock_model_service = Mock()
    mock_model_service.partial_dependence = Mock(return_value={"status": "success"})

    @app.post("/classification/api/partial-dependence")
    async def post_partial_dependence(payload: dict, user: dict = Depends(mock_user)):
        try:
            feature = payload.get("feature")
            if not feature:
                raise HTTPException(status_code=400, detail="Missing 'feature'")

            num_points = int(payload.get("num_points", 20))
            return mock_model_service.partial_dependence(feature, num_points)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    app.mock_model_service = mock_model_service
    return TestClient(app)


# ---------------- Test Cases ---------------- #


def test_partial_dependence_success(client):
    """✔ Should succeed with proper payload."""
    payload = {"feature": "age", "num_points": 25}

    response = client.post("/classification/api/partial-dependence", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    client.app.mock_model_service.partial_dependence.assert_called_once_with("age", 25)


def test_partial_dependence_missing_feature(client):
    """❌ Missing feature should trigger 400 error."""
    payload = {"num_points": 15}  # feature missing

    response = client.post("/classification/api/partial-dependence", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Missing 'feature'"


def test_partial_dependence_internal_error(client):
    """💥 Simulate crash inside service call."""
    client.app.mock_model_service.partial_dependence.side_effect = Exception("Failure!")

    payload = {"feature": "income"}

    response = client.post("/classification/api/partial-dependence", json=payload)

    assert response.status_code == 500
    assert "Internal error" in response.json()["detail"]
