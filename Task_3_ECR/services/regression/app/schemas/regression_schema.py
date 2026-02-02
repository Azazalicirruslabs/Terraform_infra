from typing import Literal, Optional, Tuple

from pydantic import BaseModel


class RequestPayload(BaseModel):
    cur_dataset: Optional[str] = None
    ref_dataset: Optional[str] = None
    model: Optional[str] = None
    target_column: Optional[str] = None


class RegressionAnalysisResponse(BaseModel):
    status: Literal["success"]
    evidently_report_html: str
    llm_insights: str


class DataInfo(BaseModel):
    reference_shape: Tuple[int, int]
    current_shape: Tuple[int, int]
    model_type: str
    target_column: str


class ExplainerDashboardResponse(BaseModel):
    status: Literal["success", "already_running"]
    explainer_url: str
    data_info: DataInfo | None = None  # Only present on fresh launch


class CorrelationPayload(BaseModel):
    cur_dataset: Optional[str] = None
    ref_dataset: Optional[str] = None
    model: Optional[str] = None
    target_column: Optional[str] = None
    features: list[str] = []
