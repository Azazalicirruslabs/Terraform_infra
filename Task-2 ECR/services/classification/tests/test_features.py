from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


def make_app(success=True):
    app = FastAPI()

    # Mock ModelService
    class MockModelService:
        def get_feature_metadata(self):
            if not success:
                raise ValueError("Feature metadata fetch failed")

            return {
                "features": [
                    {"name": "feature_1", "type": "numeric"},
                    {"name": "feature_2", "type": "categorical"},
                ]
            }

    model_service = MockModelService()

    def mock_current_user():
        return {"user_id": "test-user"}

    # Mock handle_request
    def mock_handle_request(func, *args):
        try:
            result = func(*args)
            return {"status": "ok", "data": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/classification/features")
    async def get_features_metadata(user=Depends(mock_current_user)):
        return mock_handle_request(model_service.get_feature_metadata)

    return app


# --------------------------------------------------------
# TEST CASES
# --------------------------------------------------------


def test_get_features_metadata_success():
    """✔ Should return metadata when model service works normally."""
    app = make_app(success=True)
    client = TestClient(app)

    resp = client.get("/classification/features")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert "data" in body
    assert "features" in body["data"]
    assert len(body["data"]["features"]) == 2


def test_get_features_metadata_failure():
    """✔ Should return 500 when metadata retrieval fails."""
    app = make_app(success=False)
    client = TestClient(app)

    resp = client.get("/classification/features")
    assert resp.status_code == 500
    assert "Feature metadata fetch failed" in resp.json()["detail"]
