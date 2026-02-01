from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


def make_app(success=True):
    """
    success=True  -> valid explanation returned
    success=False -> simulate failure inside service
    """
    app = FastAPI()

    class MockService:
        def explain_instance(self, idx: int):
            if not success:
                raise ValueError("Failed to explain instance")
            if idx < 0:
                raise ValueError("Invalid instance index")
            return {
                "instance_id": idx,
                "prediction": 1,
                "features": {"age": 35, "income": 75000},
                "explanation": "Mock explanation",
            }

    model_service = MockService()

    def mock_handle_request(func, *args):
        """Mocking common error-handling wrapper."""
        try:
            return func(*args)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def mock_get_current_user():
        return {"user_id": "test-user"}

    @app.get("/classification/explain-instance/{instance_idx}")
    async def explain_instance(instance_idx: int, user=Depends(mock_get_current_user)):
        return mock_handle_request(model_service.explain_instance, instance_idx)

    return app


def test_explain_instance_success():
    """➡ Validate correct formatted response on success."""
    app = make_app(success=True)
    client = TestClient(app)

    resp = client.get("/classification/explain-instance/5")

    assert resp.status_code == 200
    data = resp.json()

    assert data["instance_id"] == 5
    assert "features" in data
    assert "explanation" in data
    assert data["prediction"] in [0, 1]


def test_explain_instance_invalid_index():
    """➡ Invalid index should return 500 with message."""
    app = make_app(success=True)
    client = TestClient(app)

    resp = client.get("/classification/explain-instance/-1")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Invalid instance index"


def test_explain_instance_failure():
    """➡ Service failure should return HTTP 500."""
    app = make_app(success=False)
    client = TestClient(app)

    resp = client.get("/classification/explain-instance/10")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to explain instance"
