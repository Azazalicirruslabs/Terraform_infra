from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


# ----------------------------------------------------
# Mock Minimal App for Unit Testing
# ----------------------------------------------------
def make_app(success=True):
    app = FastAPI()

    class MockModelService:
        def get_feature_importance(self, method):
            if not success:
                raise ValueError("Error computing feature importance")
            return {
                "method": method,
                "importance": {
                    "feature_1": 0.42,
                    "feature_2": -0.18,
                },
            }

    model_service = MockModelService()

    def mock_current_user():
        return {"user_id": "test-user"}

    def mock_handle_request(func, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return {"status": "ok", "data": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/classification/feature-importance")
    async def get_feature_importance(method: str = "shap", user=Depends(mock_current_user)):
        return mock_handle_request(model_service.get_feature_importance, method)

    return app


# ----------------------------------------------------
# TEST CASES
# ----------------------------------------------------


def test_feature_importance_default_method():
    """✔ Should return success response with default 'shap' method."""
    app = make_app(success=True)
    client = TestClient(app)

    resp = client.get("/classification/feature-importance")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["data"]["method"] == "shap"
    assert "importance" in data["data"]


def test_feature_importance_custom_method():
    """✔ Should support alternate method name."""
    app = make_app(success=True)
    client = TestClient(app)

    resp = client.get("/classification/feature-importance?method=xi")
    assert resp.status_code == 200

    data = resp.json()
    assert data["data"]["method"] == "xi"


def test_feature_importance_internal_failure():
    """✔ Should return 500 when backend fails."""
    app = make_app(success=False)
    client = TestClient(app)

    resp = client.get("/classification/feature-importance")
    assert resp.status_code == 500
    assert "Error computing feature importance" in resp.json()["detail"]
