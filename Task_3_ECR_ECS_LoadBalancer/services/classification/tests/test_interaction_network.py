from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


def make_app(success=True):
    app = FastAPI()

    # Mock Model Service
    class MockModelService:
        def interaction_network(self, top_k, sample_rows):
            if not success:
                raise RuntimeError("Interaction network failed")
            return {
                "nodes": [{"id": 1}, {"id": 2}],
                "edges": [{"source": 1, "target": 2}],
                "top_k": top_k,
                "sample_rows": sample_rows,
            }

    model_service = MockModelService()

    # Mock current user
    def mock_current_user():
        return {"user_id": "test-user"}

    # Mock handle_request to simulate wrapper behavior
    def mock_handle_request(func, *args):
        try:
            result = func(*args)
            return {"status": "ok", "data": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/classification/api/interaction-network")
    async def post_interaction_network(payload: dict = Body({}), user=Depends(mock_current_user)):
        top_k = int(payload.get("top_k", 30))
        sample_rows = int(payload.get("sample_rows", 200))
        return mock_handle_request(model_service.interaction_network, top_k, sample_rows)

    return app


# ---------------------------------------------------
# TEST CASES
# ---------------------------------------------------


def test_interaction_network_success():
    """✔ Validate successful response with given payload"""
    app = make_app(success=True)
    client = TestClient(app)

    payload = {"top_k": 50, "sample_rows": 100}

    resp = client.post("/classification/api/interaction-network", json=payload)
    assert resp.status_code == 200

    data = resp.json()["data"]
    assert data["top_k"] == 50
    assert data["sample_rows"] == 100
    assert "nodes" in data and "edges" in data


def test_interaction_network_default_values():
    """✔ Verify default top_k and sample_rows are applied"""
    app = make_app()
    client = TestClient(app)

    resp = client.post("/classification/api/interaction-network", json={})
    assert resp.status_code == 200

    data = resp.json()["data"]
    assert data["top_k"] == 30  # Default
    assert data["sample_rows"] == 200  # Default


def test_interaction_network_failure():
    """❌ Ensure runtime exception produces HTTP 500"""
    app = make_app(success=False)
    client = TestClient(app)

    resp = client.post("/classification/api/interaction-network", json={"top_k": 40})
    assert resp.status_code == 500
    assert "Interaction network failed" in resp.json()["detail"]
