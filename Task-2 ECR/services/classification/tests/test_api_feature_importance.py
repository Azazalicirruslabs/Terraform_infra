from typing import Dict

from fastapi import FastAPI
from fastapi.testclient import TestClient


def make_app():
    app = FastAPI()

    @app.post("/classification/api/feature-importance")
    async def post_feature_importance(payload: Dict):
        method = payload.get("method", "shap")
        sort_by = payload.get("sort_by", "importance")

        try:
            top_n = int(payload.get("top_n", 20))  # enforce integer
        except:
            return {"error": "Invalid top_n format"}

        visualization = payload.get("visualization", "bar")

        return {
            "status": "ok",
            "method": method,
            "sort_by": sort_by,
            "top_n": top_n,
            "visualization": visualization,
            "features": ["f1", "f2", "f3"],  # mocked feature list
        }

    return app


def test_feature_importance_success():
    app = make_app()
    client = TestClient(app)

    payload = {"method": "shap", "sort_by": "importance", "top_n": 10, "visualization": "bar"}

    resp = client.post("/classification/api/feature-importance", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["top_n"] == 10
    assert data["method"] == "shap"
    assert "features" in data


def test_feature_importance_defaults():
    """Test with NO parameters — falls back to defaults"""
    app = make_app()
    client = TestClient(app)

    resp = client.post("/classification/api/feature-importance", json={})
    assert resp.status_code == 200

    data = resp.json()
    assert data["method"] == "shap"
    assert data["sort_by"] == "importance"
    assert data["top_n"] == 20
    assert data["visualization"] == "bar"


def test_feature_importance_invalid_top_n():
    """Test top_n wrong format raises error message"""
    app = make_app()
    client = TestClient(app)

    payload = {"top_n": "invalid_number"}
    resp = client.post("/classification/api/feature-importance", json=payload)

    assert resp.status_code == 200
    assert resp.json()["error"] == "Invalid top_n format"
