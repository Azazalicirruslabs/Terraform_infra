from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


# ----------------------------------------------------
# Mock FastAPI app for isolated unit testing
# ----------------------------------------------------
def make_app(success=True):
    app = FastAPI()

    class MockModelService:
        def get_feature_interactions(self, feature1, feature2):
            if not success:
                raise ValueError("Failed to compute interactions")

            return {
                "feature1": feature1,
                "feature2": feature2,
                "interaction_strength": 0.85,
                "pairs": [[feature1, feature2]],
            }

    model_service = MockModelService()

    def mock_current_user():
        return {"user_id": "test-user"}

    def mock_handle_request(func, *args):
        try:
            result = func(*args)
            return {"status": "ok", "data": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/classification/analysis/feature-interactions")
    async def get_feature_interactions(feature1: str, feature2: str, user=Depends(mock_current_user)):
        return mock_handle_request(model_service.get_feature_interactions, feature1, feature2)

    return app


# ----------------------------------------------------
# TEST CASES
# ----------------------------------------------------


def test_feature_interactions_success():
    """✔ Should return valid response when both query params are provided."""
    app = make_app(success=True)
    client = TestClient(app)

    resp = client.get("/classification/analysis/feature-interactions?feature1=f1&feature2=f2")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["data"]["feature1"] == "f1"
    assert data["data"]["feature2"] == "f2"
    assert "interaction_strength" in data["data"]


def test_feature_interactions_missing_query_params():
    """✔ Should fail with 422 when query parameters are missing."""
    app = make_app(success=True)
    client = TestClient(app)

    resp = client.get("/classification/analysis/feature-interactions")
    assert resp.status_code == 422  # FastAPI validation error


def test_feature_interactions_internal_error():
    """✔ Should return 500 when backend service raises an error."""
    app = make_app(success=False)
    client = TestClient(app)

    resp = client.get("/classification/analysis/feature-interactions?feature1=f1&feature2=f2")
    assert resp.status_code == 500
    assert "Failed to compute interactions" in resp.json()["detail"]
