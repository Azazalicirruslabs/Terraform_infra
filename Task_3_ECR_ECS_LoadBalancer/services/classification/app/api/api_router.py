# api/api_router.py

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from services.classification.app.orchestrator.analysis_orchestrator import AnalysisOrchestrator
from services.classification.app.schemas.classification_schema import (
    DashboardURLResponse,
    HTMLReportSchema,
    MarkdownAnalysisResponse,
)
from shared.auth import get_current_user

router = APIRouter(prefix="/classification", tags=["Old Classification"])
orchestrator = AnalysisOrchestrator()


class ReportResponse(BaseModel):
    target_column: str


@router.post("/report/html", response_model=HTMLReportSchema)
def get_report_html(request: ReportResponse, current_user: dict = Depends(get_current_user)):

    token = current_user.get("token")
    target = request.target_column
    if not target:
        return {"error": "Target column is required."}

    model, train_df, test_df = orchestrator.load_data_and_model(token)
    html, _ = orchestrator.get_evidently_report(model, train_df, test_df, target)
    return {
        "Status": "Success",
        "html": html,
    }


class MarkdownResponse(BaseModel):
    target_column: str


@router.post("/analysis/markdown", response_model=MarkdownAnalysisResponse)
def get_llm_markdown(request: MarkdownResponse, current_user: dict = Depends(get_current_user)):
    token = current_user.get("token")
    target = request.target_column
    if not target:
        return {"error": "Target column is required."}

    # Load model and data
    model, train_df, test_df = orchestrator.load_data_and_model(token)
    if model is None or train_df is None or test_df is None:
        return {"error": "Failed to load model or data from S3."}

    # Get Evidently report JSON
    _, json_data = orchestrator.get_evidently_report(model, train_df, test_df, target)

    # Get LLM analysis markdown
    md = orchestrator.get_llm_analysis(json_data, train_df, test_df, target)
    return {"markdown": md}


class DashboardRequest(BaseModel):
    target_column: str


@router.post("/dashboard/url", response_model=DashboardURLResponse)
def get_dashboard_url(request: DashboardRequest, current_user: dict = Depends(get_current_user)):
    token = current_user.get("token")
    target = request.target_column
    if not target:
        return {"error": "Target column is required."}
    # Load model and data
    model, train_df, test_df = orchestrator.load_data_and_model(token)
    url = orchestrator.get_explainer_dashboard(model, train_df, test_df, target)
    return {"url": url}
