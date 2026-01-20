from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


# ----------------------------------------------------
# Mock App for Endpoint Testing
# ----------------------------------------------------
def make_app(success=True):
    app = FastAPI()

    class MockModelService:
        def get_feature_dependence(self, feature_name):
            if not success:
                raise ValueError("Failed to compute dependence")
            return {
                "feature": feature_name,
                "dependence_values": [0.12, 0.34],
                "shap_values": [0.01, -0.02],
            }

    model_service = MockModelService()

    def mock_get_current_user():
        return {"user_id": "test-user"}

    def mock_handle_request(func, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return {"status": "ok", "data": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/classification/feature-dependence/{feature_name}")
    async def get_feature_dependence(feature_name: str, user=Depends(mock_get_current_user)):
        if not feature_name:
            raise HTTPException(status_code=422, detail="Feature name required")
        return mock_handle_request(model_service.get_feature_dependence, feature_name)

    return app


# ----------------------------------------------------
# TEST CASES
# ----------------------------------------------------


def test_feature_dependence_success():
    """✔ Should return dependence values when feature name is provided"""
    app = make_app(success=True)
    client = TestClient(app)

    resp = client.get("/classification/feature-dependence/age")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["data"]["feature"] == "age"
    assert "dependence_values" in data["data"]
    assert "shap_values" in data["data"]


def test_feature_dependence_internal_failure():
    """✔ Should return 500 when backend processing fails"""
    app = make_app(success=False)
    client = TestClient(app)

    resp = client.get("/classification/feature-dependence/income")
    assert resp.status_code == 500
    assert "Failed to compute dependence" in resp.json()["detail"]


def test_feature_dependence_missing_feature():
    """✔ Should return 422 if path param is empty / invalid"""
    app = make_app()
    client = TestClient(app)

    resp = client.get("/classification/feature-dependence/")
    assert resp.status_code in [404, 422]  # ❗ FastAPI may interpret missing path as 404
