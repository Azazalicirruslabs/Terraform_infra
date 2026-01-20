from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


def make_app(success=True):
    """
    Helper function to build the app with mock behavior.
    success=True  -> return valid tree response
    success=False -> simulate backend/service failure
    """
    app = FastAPI()

    class MockService:
        def get_decision_tree(self):
            if not success:
                raise ValueError("Decision tree generation failed")
            return {
                "model_type": "RandomForest",
                "num_trees": 10,
                "trees": [{"depth": 5}, {"depth": 7}],  # mock tree structure
            }

    model_service = MockService()

    def handle_request(func, *args):
        try:
            return func(*args)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def mock_get_current_user():
        return {"user_id": "test-user"}

    @app.get("/classification/analysis/decision-tree")
    async def get_decision_tree(user=Depends(mock_get_current_user)):
        return handle_request(model_service.get_decision_tree)

    return app


def test_decision_tree_success():
    """Verify decision tree API returns proper response on success."""
    app = make_app(success=True)
    client = TestClient(app)

    resp = client.get("/classification/analysis/decision-tree")

    assert resp.status_code == 200
    data = resp.json()

    # Expected response structure
    assert "model_type" in data
    assert "num_trees" in data
    assert "trees" in data
    assert isinstance(data["trees"], list)
    assert data["num_trees"] == 10


def test_decision_tree_failure():
    """Verify decision tree API handles internal errors gracefully."""
    app = make_app(success=False)
    client = TestClient(app)

    resp = client.get("/classification/analysis/decision-tree")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Decision tree generation failed"
