"""
BiasLens™ Analyzer - Fairness Analysis Schemas

This module contains all Pydantic models for the fairness analysis API endpoints.

Models:
    MetricSelectionModel: User-selectable fairness metrics configuration
    ThresholdsModel: User-configurable thresholds for fairness assessment
    FairnessAnalysisRequest: Complete request schema for fairness analysis
    LLMAnalysisModel: LLM-generated fairness analysis insights
    FairnessAnalysisResponse: Complete fairness analysis response
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class MetricSelectionModel(BaseModel):
    """
    User-selectable fairness metrics configuration.

    Attributes:
        equal_opportunity: Enable Equal Opportunity metric (TPR parity)
        disparate_impact: Enable Disparate Impact metric (four-fifths rule)
        statistical_parity: Enable Statistical Parity metric (demographic parity)
        equalized_odds: Enable Equalized Odds metric (TPR/FPR equality)
    """

    equal_opportunity: bool = Field(True, description="Enable Equal Opportunity metric")
    disparate_impact: bool = Field(True, description="Enable Disparate Impact metric")
    statistical_parity: bool = Field(True, description="Enable Statistical Parity metric")
    equalized_odds: bool = Field(True, description="Enable Equalized Odds metric")

    @model_validator(mode="after")
    def check_at_least_one_metric(self):
        """Ensure at least one metric is enabled."""
        if not any([self.equal_opportunity, self.disparate_impact, self.statistical_parity, self.equalized_odds]):
            raise ValueError("At least one fairness metric must be enabled")
        return self


class ThresholdsModel(BaseModel):
    """
    User-configurable thresholds for fairness assessment.

    Thresholds define three zones:
        - Acceptable: Metric >= acceptable_threshold
        - Warning: breach_threshold < Metric < acceptable_threshold
        - Breach: Metric <= breach_threshold

    Attributes:
        acceptable_threshold: Upper threshold for acceptable fairness (0-100)
        warning_threshold: Middle threshold for warning zone (0-100)
        breach_threshold: Lower threshold for breach zone (0-100)
    """

    acceptable_threshold: float = Field(80.0, ge=0.0, le=100.0, description="Threshold for acceptable fairness (0-100)")
    warning_threshold: float = Field(60.0, ge=0.0, le=100.0, description="Threshold for warning zone (0-100)")
    breach_threshold: float = Field(40.0, ge=0.0, le=100.0, description="Threshold for breach zone (0-100)")

    @model_validator(mode="after")
    def validate_thresholds(self):
        """Ensure thresholds are in correct order."""
        if self.warning_threshold >= self.acceptable_threshold:
            raise ValueError("warning_threshold must be less than acceptable_threshold")
        if self.breach_threshold >= self.warning_threshold:
            raise ValueError("breach_threshold must be less than warning_threshold")
        return self


class FairnessAnalysisRequest(BaseModel):
    """
    Complete request schema for fairness analysis.

    Attributes:
        reference_url: Pre-signed S3 URL for reference/training dataset CSV
        current_url: Pre-signed S3 URL for current/testing dataset CSV (optional)
        model_url: Pre-signed S3 URL for trained model file (optional)
        target_column: Name of the target/label column (required)
        sensitive_feature: Name of the sensitive/protected attribute column (required)
        metric_selection: Which metrics to calculate
        thresholds: Custom threshold values for assessment
    """

    reference_url: str = Field(..., description="Pre-signed S3 URL for reference/training dataset CSV")
    current_url: Optional[str] = Field(None, description="Pre-signed S3 URL for current/testing dataset CSV")
    model_url: Optional[str] = Field(None, description="Pre-signed S3 URL for trained model (.pkl)")

    target_column: str = Field(..., description="Target/label column name")
    sensitive_feature: str = Field(..., description="Sensitive/protected attribute column name")

    metric_selection: MetricSelectionModel = Field(
        default_factory=MetricSelectionModel, description="Metrics to calculate"
    )
    thresholds: ThresholdsModel = Field(default_factory=ThresholdsModel, description="Threshold configuration")

    @model_validator(mode="after")
    def validate_data_sources(self):
        """Ensure either current_url or model_url is provided."""
        if not self.current_url and not self.model_url:
            raise ValueError(
                "Either current_url or model_url must be provided. "
                "If current_url is missing, model_url is required to generate predictions."
            )
        return self


class LLMAnalysisModel(BaseModel):
    """LLM-generated fairness analysis insights."""

    what_this_means: str = Field("", description="Plain-language explanation of key findings")
    why_it_matters: str = Field("", description="Real-world implications and risks")
    risk_signal: str = Field("", description="Overall fairness status and guidance")


class FairnessAnalysisResponse(BaseModel):
    """
    Complete fairness analysis response.

    Contains calculated metrics, group performance comparison, LLM analysis, and overall assessment.
    """

    status: str = Field(..., description="Analysis status: 'success' or 'error'")
    metrics: Dict[str, Any] = Field(..., description="Calculated fairness metrics")
    group_performance: Dict[str, Any] = Field(..., description="Performance comparison by group")
    llm_analysis: LLMAnalysisModel = Field(
        default_factory=LLMAnalysisModel, description="LLM-generated fairness insights"
    )
    overall_assessment: str = Field(..., description="Overall fairness assessment")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Analysis metadata")
    applied_thresholds: Dict[str, Any] = Field(default_factory=dict, description="Thresholds used for assessment")
    warnings: List[str] = Field(default_factory=list, description="Any warnings generated during analysis")
    message: Optional[str] = Field(None, description="Additional information or warnings")
