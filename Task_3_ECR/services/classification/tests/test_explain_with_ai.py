from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


# -----------------------
# Mocked App for Testing
# -----------------------
def make_app(ai_success=True):
    app = FastAPI()

    class MockAIService:
        def generate_explanation(self, analysis_data, analysis_type):
            if not ai_success:
                raise ValueError("AI failed to generate explanation")
            return {
                "analysis_type": analysis_type,
                "explanation": "Mock AI explanation generated.",
                "summary": "This is test AI output",
            }

    ai_service = MockAIService()

    def mock_get_current_user():
        return {"user_id": "test-user"}

    @app.post("/classification/analysis/explain-with-ai")
    async def explain_with_ai(payload: dict = Body(...), user: dict = Depends(mock_get_current_user)):
        analysis_type = payload.get("analysis_type")
        analysis_data = payload.get("analysis_data", {})

        if not analysis_type:
            raise HTTPException(status_code=400, detail="Missing 'analysis_type' in payload")

        try:
            response = ai_service.generate_explanation(analysis_data, analysis_type)
            return JSONResponse(status_code=200, content=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating AI explanation: {str(e)}")

    return app


# -----------------------
# TEST CASES
# -----------------------


def test_explain_with_ai_success():
    """✔ Validate a successful explanation response"""
    app = make_app(ai_success=True)
    client = TestClient(app)

    payload = {"analysis_type": "feature_importance", "analysis_data": {"feature": "age"}}

    resp = client.post("/classification/analysis/explain-with-ai", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["analysis_type"] == "feature_importance"
    assert "explanation" in data
    assert "summary" in data


def test_missing_analysis_type():
    """✔ Ensure 400 returned if analysis_type missing"""
    app = make_app()
    client = TestClient(app)

    payload = {"analysis_data": {"feature": "income"}}

    resp = client.post("/classification/analysis/explain-with-ai", json=payload)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Missing 'analysis_type' in payload"


def test_ai_explanation_failure():
    """✔ Internal failure inside AI service should return 500"""
    app = make_app(ai_success=False)  # Trigger error
    client = TestClient(app)

    payload = {"analysis_type": "classification_stats", "analysis_data": {"accuracy": 0.85}}

    resp = client.post("/classification/analysis/explain-with-ai", json=payload)
    assert resp.status_code == 500
    assert "Error generating AI explanation" in resp.json()["detail"]
