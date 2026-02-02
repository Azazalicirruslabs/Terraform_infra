from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


def make_app(success=True):
    app = FastAPI()

    # Mock service
    class MockModelService:
        def ice_plot(self, feature, num_points, num_instances):
            if not success:
                raise RuntimeError("ICE computation failed")
            return {
                "feature": feature,
                "ice_values": [[0.1, 0.2], [0.3, 0.4]],
                "num_points": num_points,
                "num_instances": num_instances,
            }

    model_service = MockModelService()

    def mock_current_user():
        return {"user_id": "test-user"}

    # Mock handle_request
    def mock_handle_request(func, *args):
        try:
            data = func(*args)
            return {"status": "ok", "data": data}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/classification/api/ice-plot")
    async def post_ice_plot(payload: dict, user=Depends(mock_current_user)):
        feature = payload.get("feature")
        if not feature:
            raise HTTPException(status_code=400, detail="Missing 'feature'")

        num_points = int(payload.get("num_points", 20))
        num_instances = int(payload.get("num_instances", 20))

        return mock_handle_request(model_service.ice_plot, feature, num_points, num_instances)

    return app


# ---------------------------------------------------
# TEST CASES
# ---------------------------------------------------


def test_ice_plot_success():
    """✔ Success case: Valid payload should return ICE values"""
    app = make_app(success=True)
    client = TestClient(app)

    payload = {"feature": "age", "num_points": 10, "num_instances": 5}
    response = client.post("/classification/api/ice-plot", json=payload)

    assert response.status_code == 200
    result = response.json()

    assert result["status"] == "ok"
    data = result["data"]
    assert data["feature"] == "age"
    assert data["num_points"] == 10
    assert data["num_instances"] == 5
    assert "ice_values" in data


def test_ice_plot_missing_feature():
    """❌ Missing feature should raise 400"""
    app = make_app()
    client = TestClient(app)

    response = client.post("/classification/api/ice-plot", json={})

    assert response.status_code == 400
    assert response.json()["detail"] == "Missing 'feature'"


def test_ice_plot_internal_failure():
    """❌ Mocked failure should return 500"""
    app = make_app(success=False)
    client = TestClient(app)

    payload = {"feature": "age"}
    response = client.post("/classification/api/ice-plot", json=payload)

    assert response.status_code == 500
    assert "ICE computation failed" in response.json()["detail"]
