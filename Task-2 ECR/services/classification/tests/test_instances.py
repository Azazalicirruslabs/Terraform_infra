from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


def make_app(success=True):
    app = FastAPI()

    # Mock service
    class MockModelService:
        def list_instances(self, sort_by, limit):
            if not success:
                raise RuntimeError("List instances failed")
            return {
                "instances": [{"id": 1, "prediction": 0.78}, {"id": 2, "prediction": 0.64}],
                "sort_by": sort_by,
                "limit": limit,
            }

    model_service = MockModelService()

    # Mock Auth
    def mock_current_user():
        return {"user_id": "test-user"}

    # Mock Handle Request wrapper
    def mock_handle_request(func, *args):
        try:
            result = func(*args)
            return {"status": "ok", "data": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/classification/instances")
    async def list_instances(sort_by: str = "prediction", limit: int = 100, user=Depends(mock_current_user)):
        return mock_handle_request(model_service.list_instances, sort_by, limit)

    return app


# ---------------------------------------------------
# TEST CASES
# ---------------------------------------------------


def test_list_instances_success():
    """✔ Validate successful response with expected fields"""
    app = make_app(success=True)
    client = TestClient(app)

    resp = client.get("/classification/instances?sort_by=prediction&limit=50")

    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "ok"
    assert "instances" in body["data"]
    assert body["data"]["sort_by"] == "prediction"
    assert body["data"]["limit"] == 50


def test_list_instances_default_params():
    """✔ Ensure defaults apply when no query params provided"""
    app = make_app()
    client = TestClient(app)

    resp = client.get("/classification/instances")

    assert resp.status_code == 200
    data = resp.json()["data"]

    assert data["sort_by"] == "prediction"  # Default
    assert data["limit"] == 100  # Default


def test_list_instances_failure():
    """❌ Ensure runtime exception is converted to HTTP 500"""
    app = make_app(success=False)
    client = TestClient(app)

    resp = client.get("/classification/instances?sort_by=age&limit=20")

    assert resp.status_code == 500
    assert "List instances failed" in resp.json()["detail"]
