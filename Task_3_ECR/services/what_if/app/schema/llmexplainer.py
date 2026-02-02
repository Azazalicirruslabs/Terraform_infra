from typing import List

from pydantic import BaseModel


# LLM Analysis Models
class ModelInfo(BaseModel):
    n_features: int
    n_samples: int
    target_column: str


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    rank: int


class ShapAnalysis(BaseModel):
    explanation: str
    confidence_level: str
    interactions: str


class CurrentPrediction(BaseModel):
    value: float
    interpretation: str
    confidence: str


class WhatIfInsights(BaseModel):
    sensitivity: str
    key_drivers: str
    scenarios: str


class DashboardSummary(BaseModel):
    model_info: ModelInfo
    feature_importance: List[FeatureImportance]
    shap_analysis: ShapAnalysis
    current_prediction: CurrentPrediction
    what_if_insights: WhatIfInsights


class LLMExplainerResponse(BaseModel):
    session_id: str
    business_analysis: str
    technical_analysis: str
    timestamp: str
