from typing import List, Optional, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from services.data_drift.app.routers import datadrift as data_drift_router
from shared.auth import get_current_user

DriftAnalyzer = data_drift_router.DriftAnalyzer
LLMAnalyzer = data_drift_router.LLMAnalyzer

router = APIRouter(prefix="/data_drift")


class DriftColumnRequest(BaseModel):
    split_pct: Optional[int] = 70
    drop_cols: Optional[List[str]] = None
    date_col: Optional[str] = None
    ref_range: Optional[Tuple[str, str]] = None
    cur_range: Optional[Tuple[str, str]] = None


@router.post("/column_drift_report/{analysis_type}/{project_name}/{column}", tags=["Old Data Drift"])
def run_drift_analysis(
    analysis_type: str,
    project_name: str,
    column: str,
    payload: Optional[DriftColumnRequest] = Body(None),
    current_user: dict = Depends(get_current_user),
):
    analyzer = DriftAnalyzer()
    access_token = current_user.get("token")
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
    # Load data with 60% reference, 40% current
    ref_df, cur_df = analyzer.load_data_from_s3(
        access_token,
        split_pct=split_pct,
        drop_cols=drop_cols,
        date_col=date_col,
        ref_range=ref_range,
        cur_range=cur_range,
        analysis_type=analysis_type,
        project_name=project_name,
    )

    if ref_df is None or cur_df is None:
        raise HTTPException(status_code=400, detail="Failed to load data for drift analysis")

    # Set reference and current dataframes
    analyzer.reference_df = ref_df
    analyzer.current_df = cur_df

    # Run column drift report
    try:
        html, drift_dict = analyzer.generate_column_drift_report(column)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift report generation failed: {str(e)}")

    return {
        "status": "success",
        "column_repoart": html,
    }
