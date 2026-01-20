from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


def make_app(success=True):
    app = FastAPI()

    # Mock ModelService
    class MockModelService:
        def roc_analysis(self):
            if not success:
                raise ValueError("ROC analysis failed!")
            return {
                "auc_score": 0.89,
                "fpr": [0, 0.2, 0.5, 1.0],
                "tpr": [0, 0.5, 0.8, 1.0],
            }

    model_service = MockModelService()

    def mock_current_user():
        return {"user_id": "test-user", "email": "test@test.com"}

    # Mock handle_request behavior
    def mock_handle_request(func, *args):
        try:
            data = func()
            return {"status": "ok", "data": data}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/classification/api/roc-analysis")
    async def get_roc_analysis(user=Depends(mock_current_user)):
        return mock_handle_request(model_service.roc_analysis)

    return app


# ---------------------------------------------------
# TEST CASES
# ---------------------------------------------------


def test_roc_analysis_success():
    """✔ Should return ROC data successfully"""
    app = make_app(success=True)
    client = TestClient(app)

    response = client.get("/classification/api/roc-analysis")
    assert response.status_code == 200

    result = response.json()
    assert result["status"] == "ok"
    assert "data" in result
    assert "auc_score" in result["data"]
    assert result["data"]["auc_score"] == 0.89


def test_roc_analysis_failure():
    """✔ Should return 500 when roc_analysis errors"""
    app = make_app(success=False)
    client = TestClient(app)

    response = client.get("/classification/api/roc-analysis")
    assert response.status_code == 500
    assert "ROC analysis failed!" in response.json()["detail"]
