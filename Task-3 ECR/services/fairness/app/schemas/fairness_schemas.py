from typing import Any, Dict, List, Optional

from pydantic import BaseModel, HttpUrl


class Overview(BaseModel):
    fairness_score_before: Dict[str, Any]
    fairness_score_after: Dict[str, Any]
    mitigation_strategy_used: str


class FairnessMetrics(BaseModel):
    fairness_metrics_before: Dict[str, Any]
    fairness_metrics_after: Dict[str, Any]
    group_performance_before: Dict[str, Any]
    group_performance_after: Dict[str, Any]
    bias_detected: bool
    recommendations: Optional[Any]  # Can be made stricter if needed


class Plots(BaseModel):
    before_mitigation: Any  # Could be base64, dict, or plotly JSON, define properly if known
    after_mitigation: Any


class FullResponse(BaseModel):
    overview: Overview
    features: List[str]
    metrics: Dict[str, FairnessMetrics]
    plots: Plots
    llm_analysis_report: Optional[Any]


class DatasetPreviewResponse(BaseModel):
    columns: List[str]
    data: List[Dict[str, Any]]


class FileItem(BaseModel):
    file_name: str
    folder: str
    url: HttpUrl


class FilesAndModels(BaseModel):
    files: List[FileItem]
    models: List[FileItem]


class ListDatasetsResponse(BaseModel):
    files: FilesAndModels
