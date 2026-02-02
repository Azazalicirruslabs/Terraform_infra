from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


def make_app(success=True):
    app = FastAPI()

    # Mock model service
    class MockModelService:
        def individual_prediction(self, instance_idx):
            if not success:
                raise RuntimeError("Prediction failed")
            return {"instance_idx": instance_idx, "predicted_class": 1, "probability": 0.87}

    model_service = MockModelService()

    def mock_current_user():
        return {"user_id": "test-user"}

    def mock_handle_request(func, *args):
        try:
            result = func(*args)
            return {"status": "ok", "data": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/classification/api/individual-prediction")
    async def post_individual_prediction(payload: dict, user=Depends(mock_current_user)):
        instance_idx = int(payload.get("instance_idx", 0))
        return mock_handle_request(model_service.individual_prediction, instance_idx)

    return app


# ---------------------------------------------------
# TEST CASES
# ---------------------------------------------------


def test_individual_prediction_success():
    """✔ Valid request returns prediction successfully"""
    app = make_app(success=True)
    client = TestClient(app)

    payload = {"instance_idx": 10}
    resp = client.post("/classification/api/individual-prediction", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["data"]["instance_idx"] == 10
    assert "predicted_class" in data["data"]
    assert "probability" in data["data"]


def test_individual_prediction_default_to_zero():
    """✔ Default instance index should be 0 when not provided"""
    app = make_app()
    client = TestClient(app)

    payload = {}
    resp = client.post("/classification/api/individual-prediction", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["data"]["instance_idx"] == 0


def test_individual_prediction_service_failure():
    """❌ Service exception should return 500"""
    app = make_app(success=False)
    client = TestClient(app)

    payload = {"instance_idx": 5}
    resp = client.post("/classification/api/individual-prediction", json=payload)

    assert resp.status_code == 500
    assert "Prediction failed" in resp.json()["detail"]
