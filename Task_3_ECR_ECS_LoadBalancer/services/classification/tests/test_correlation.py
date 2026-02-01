from typing import Dict, List

from fastapi import FastAPI
from fastapi.testclient import TestClient


def make_app():
    app = FastAPI()

    @app.post("/classification/correlation")
    async def post_correlation(payload: Dict):
        selected: List[str] = payload.get("features") or []

        if not selected:
            return {"error": "No features provided"}

        return {"status": "ok", "correlation_matrix": [[1.0, 0.3], [0.3, 1.0]], "features": selected}

    return app


def test_correlation_success():
    app = make_app()
    client = TestClient(app)

    payload = {"features": ["f1", "f2"]}
    resp = client.post("/classification/correlation", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["features"] == payload["features"]


def test_correlation_missing_features():
    app = make_app()
    client = TestClient(app)

    resp = client.post("/classification/correlation", json={})

    assert resp.status_code == 200
    assert resp.json()["error"] == "No features provided"


def test_correlation_invalid_data_format():
    app = make_app()
    client = TestClient(app)

    payload = {"features": "not-a-list"}
    resp = client.post("/classification/correlation", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["features"] == "not-a-list"
