from typing import Dict

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import JSONResponse

from services.classification.app.core.ai_explanation_service import AIExplanationService
from shared.auth import get_current_user

router = APIRouter(prefix="/classification", tags=["AI Analysis"])


@router.post("/analysis/explain-with-ai", tags=["AI Analysis"])
async def explain_with_ai(payload: Dict = Body(...), user: str = Depends(get_current_user)):

    ai_explanation_service = AIExplanationService()

    try:
        # ✅ Validate payload
        analysis_type = payload.get("analysis_type")
        analysis_data = payload.get("analysis_data", {})

        if not analysis_type:
            raise HTTPException(status_code=400, detail="❌ Missing 'analysis_type' in payload")

        if not isinstance(analysis_data, dict):
            raise HTTPException(status_code=400, detail="❌ 'analysis_data' must be a dictionary")

        # ✅ Generate AI explanation
        try:
            explanation = ai_explanation_service.generate_explanation(analysis_data, analysis_type)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f"❌ AI explanation could not be generated: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"⚠️ Unexpected error during AI explanation: {str(e)}")

        # ✅ Return explanation
        return JSONResponse(status_code=200, content=explanation)

    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"⚠️ Unexpected internal error while generating AI explanation: {str(e)}"
        )
