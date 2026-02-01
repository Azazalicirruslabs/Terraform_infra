from typing import Optional

from pydantic import BaseModel


class RequestPayload(BaseModel):
    cur_dataset: Optional[str] = None
    ref_dataset: Optional[str] = None
    model: Optional[str] = None
    target_column: Optional[str] = None


class HTMLReportSchema(BaseModel):
    html: str


class MarkdownAnalysisResponse(BaseModel):
    markdown: str


class DashboardURLResponse(BaseModel):
    url: str


class CorrelationPayload(BaseModel):
    cur_dataset: Optional[str] = None
    ref_dataset: Optional[str] = None
    model: Optional[str] = None
    target_column: str = None
    features: list[str] = []
