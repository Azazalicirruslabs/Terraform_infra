from typing import List, Optional, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from services.data_drift.app.routers import datadrift as data_drift_router
from shared.auth import get_current_user

router = APIRouter(prefix="/data_drift")


DriftAnalyzer = data_drift_router.DriftAnalyzer
LLMAnalyzer = data_drift_router.LLMAnalyzer


class LlmRequest(BaseModel):
    split_pct: Optional[int] = 70
    drop_cols: Optional[List[str]] = None
    date_col: Optional[str] = None
    ref_range: Optional[Tuple[str, str]] = None
    cur_range: Optional[Tuple[str, str]] = None


@router.post("/llm_report/{analysis_type}/{project_name}/", tags=["Old Data Drift"])
def run_llm_drift_analysis(
    analysis_type: str,
    project_name: str,
    payload: Optional[LlmRequest] = Body(None),
    current_user: dict = Depends(get_current_user),
):
    access_token = current_user.get("token")
    llm_analyzer = LLMAnalyzer()
    drift_analyzer = DriftAnalyzer()

    # Handle empty body
    if payload is None:
        split_pct = 70
        drop_cols = []
        date_col = None
        ref_range = None
        cur_range = None
    else:
        split_pct = payload.split_pct or 70
        drop_cols = payload.drop_cols or []
        date_col = payload.date_col or None
        ref_range = payload.ref_range
        cur_range = payload.cur_range

    try:
        ref_df, cur_df = drift_analyzer.load_data_from_s3(
            access_token,
            split_pct=split_pct,
            drop_cols=drop_cols,
            date_col=date_col,
            ref_range=ref_range,
            cur_range=cur_range,
            analysis_type=analysis_type,
            project_name=project_name,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load data for S3 drift analysis: {str(e)}")

    try:
        if ref_df is not None:
            html, drift_dict = drift_analyzer.generate_full_drift_report()
            summary_text = drift_analyzer.get_drift_summary(drift_dict)
            data = llm_analyzer.analyze_full_drift(ref_df, cur_df, summary_text)

            return {
                "status": "success",
                "llm_analysis": data,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {str(e)}")
