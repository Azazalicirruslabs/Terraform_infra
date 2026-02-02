"""
Pydantic schemas for explainability analysis requests and responses.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ThresholdConfig(BaseModel):
    """Threshold configuration for metric evaluation."""

    acceptable_threshold: float = Field(default=0.70, ge=0, le=1, description="Acceptable threshold (≥ this value)")
    warning_threshold: float = Field(default=0.50, ge=0, le=1, description="Warning threshold (≥ this value)")
    breach_threshold: float = Field(default=0.30, ge=0, le=1, description="Breach threshold (< this value)")

    @field_validator("warning_threshold", "breach_threshold")
    @classmethod
    def validate_threshold_order(cls, v, info):
        """Ensure thresholds are in correct order."""
        if info.field_name == "warning_threshold":
            acceptable = info.data.get("acceptable_threshold", 0.70)
            if v >= acceptable:
                raise ValueError("warning_threshold must be less than acceptable_threshold")
        elif info.field_name == "breach_threshold":
            warning = info.data.get("warning_threshold", 0.50)
            if v >= warning:
                raise ValueError("breach_threshold must be less than warning_threshold")
        return v


class ExplainabilityRequest(BaseModel):
    """Request schema for explainability analysis."""

    ref_dataset: str = Field(..., description="Path or URL to the training/reference dataset (.csv or .parquet)")
    cur_dataset: str = Field(..., description="Path or URL to the test/current dataset (.csv or .parquet)")
    model: str = Field(..., description="Path or URL to the model file (.joblib, .pkl, or .pickle)")
    target_column: str = Field(..., description="Name of the target column in the datasets")

    # Feature importance method selection
    feature_importance: bool = Field(default=True, description="Whether to compute feature importance")
    feature_importance_method: str = Field(
        default="shap", description="Method for feature importance: 'shap', 'gain', or 'permutation'"
    )

    # Threshold configuration
    thresholds: Optional[ThresholdConfig] = Field(
        default=None, description="Threshold configuration for metrics evaluation"
    )

    @field_validator("feature_importance_method")
    @classmethod
    def validate_method(cls, v):
        """Ensure method is one of the supported values."""
        allowed_methods = ["shap", "gain", "permutation"]
        if v.lower() not in allowed_methods:
            raise ValueError(f"feature_importance_method must be one of {allowed_methods}, got '{v}'")
        return v.lower()


class MetricsResponse(BaseModel):
    """Performance metrics for classification or regression."""

    train: Dict[str, float] = Field(..., description="Training metrics")
    test: Dict[str, float] = Field(..., description="Test metrics")
    overfitting_score: float = Field(..., description="Overfitting indicator")


class FeatureImportanceItem(BaseModel):
    """Individual feature importance item."""

    name: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Importance value")
    impact_direction: str = Field(..., description="Impact direction: 'positive' or 'negative'")
    rank: int = Field(..., description="Rank based on importance")


class LLMAnalysisResponse(BaseModel):
    """LLM-generated analysis insights."""

    what_this_means: str = Field(..., description="Plain language explanation of results")
    why_it_matters: str = Field(..., description="Business impact and implications")
    risk_signal: str = Field(..., description="Risk level and key concerns")


class ThresholdEvaluation(BaseModel):
    """Threshold evaluation result for a metric."""

    metric_name: str = Field(..., description="Name of the evaluated metric")
    metric_value: float = Field(..., description="Current value of the metric")
    status: str = Field(..., description="Status: 'acceptable', 'warning', or 'breach'")
    threshold_used: float = Field(..., description="The threshold value used for evaluation")
    message: str = Field(..., description="Human-readable status message")


class ExplainabilityResponse(BaseModel):
    """Response schema for explainability analysis."""

    performance_metrics: MetricsResponse = Field(..., description="Model performance metrics")
    shap_available: bool = Field(..., description="Whether SHAP analysis is available")
    total_features: int = Field(..., description="Total number of features")
    positive_impact_count: int = Field(..., description="Number of features with positive impact")
    negative_impact_count: int = Field(..., description="Number of features with negative impact")
    features: List[FeatureImportanceItem] = Field(..., description="List of feature importance items")
    computation_method: str = Field(..., description="Method used for feature importance computation")
    computed_at: str = Field(..., description="ISO timestamp when analysis was computed")
    llm_analysis: LLMAnalysisResponse = Field(..., description="LLM-generated insights and explanations")
    threshold_evaluations: Optional[List[ThresholdEvaluation]] = Field(
        default=None, description="Threshold evaluation results for key metrics"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "performance_metrics": {
                    "train": {
                        "accuracy": 0.758,
                        "precision": 0.765,
                        "recall": 0.758,
                        "f1_score": 0.760,
                        "auc": 0.899,
                    },
                    "test": {
                        "accuracy": 0.604,
                        "precision": 0.630,
                        "recall": 0.604,
                        "f1_score": 0.610,
                        "auc": 0.755,
                    },
                    "overfitting_score": 0.154,
                },
                "shap_available": True,
                "total_features": 10,
                "positive_impact_count": 3,
                "negative_impact_count": 7,
                "features": [
                    {
                        "name": "feature1",
                        "importance": 0.25,
                        "impact_direction": "positive",
                        "rank": 1,
                    },
                    {
                        "name": "feature2",
                        "importance": 0.18,
                        "impact_direction": "negative",
                        "rank": 2,
                    },
                ],
                "computation_method": "shap",
                "computed_at": "2026-01-09T10:04:33.307486+00:00",
                "llm_analysis": {
                    "what_this_means": "The model shows moderate performance with 60.4% accuracy on test data...",
                    "why_it_matters": "This indicates potential overfitting that could impact production reliability...",
                    "risk_signal": "Medium risk - significant performance gap between training and test sets.",
                },
                "threshold_evaluations": [
                    {
                        "metric_name": "accuracy",
                        "metric_value": 0.604,
                        "status": "warning",
                        "threshold_used": 0.50,
                        "message": "Accuracy (60.4%) is in warning range (≥50% but <70%)",
                    },
                    {
                        "metric_name": "f1_score",
                        "metric_value": 0.610,
                        "status": "warning",
                        "threshold_used": 0.50,
                        "message": "F1 Score (61.0%) is in warning range (≥50% but <70%)",
                    },
                ],
            }
        }
