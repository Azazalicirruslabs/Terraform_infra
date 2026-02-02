from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


def make_app():
    app = FastAPI()

    class MockService:
        def get_dataset_comparison(self):
            return {"matches": 95, "differences": 5}

    model_service = MockService()

    def handle_request(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def mock_get_current_user():
        return {"user_id": "test-user"}

    @app.get("/classification/dataset-comparison")
    async def get_dataset_comparison(user=Depends(mock_get_current_user)):
        return handle_request(model_service.get_dataset_comparison)

    return app


def test_dataset_comparison_success():
    app = make_app()
    client = TestClient(app)

    resp = client.get("/classification/dataset-comparison")
    assert resp.status_code == 200

    data = resp.json()
    assert "matches" in data
    assert "differences" in data
    assert data["matches"] == 95


def test_dataset_comparison_failure():
    app = FastAPI()

    def failing_service():
        raise ValueError("Processing error")

    @app.get("/classification/dataset-comparison")
    async def get_dataset_comparison():
        try:
            failing_service()
        except Exception:
            raise HTTPException(status_code=500, detail="Processing error")

    client = TestClient(app)
    resp = client.get("/classification/dataset-comparison")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Processing error"
