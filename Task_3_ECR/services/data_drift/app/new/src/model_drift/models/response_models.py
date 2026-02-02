"""
Response Models for Model Drift Detection API
Defines Pydantic models for structured API responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field


class DriftSeverity(str, Enum):
    """Enum for drift severity levels."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalysisStatus(str, Enum):
    """Enum for analysis status."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"


class MetricResult(BaseModel):
    """Individual metric calculation result."""

    name: str = Field(description="Metric name")
    reference_value: float = Field(description="Metric value on reference data")
    current_value: float = Field(description="Metric value on current data")
    difference: float = Field(description="Absolute difference between values")
    percentage_change: float = Field(description="Percentage change from reference to current")
    relative_difference: float = Field(description="Relative difference as proportion")
    drift_severity: DriftSeverity = Field(description="Drift severity classification")
    confidence_interval: Optional[List[float]] = Field(default=None, description="Confidence interval for difference")
    is_degraded: bool = Field(description="Whether this metric shows degradation")
    is_critical: bool = Field(default=False, description="Whether this is a critical metric")
    threshold_exceeded: bool = Field(description="Whether drift threshold was exceeded")

    class Config:
        json_encoders = {float: lambda v: round(v, 6) if v is not None and not (np.isnan(v) or np.isinf(v)) else None}


class StatisticalTestResult(BaseModel):
    """Statistical test result."""

    test_name: str = Field(description="Name of the statistical test")
    test_type: str = Field(description="Type/category of test")
    statistic: float = Field(description="Test statistic value")
    p_value: float = Field(description="P-value of the test")
    is_significant: bool = Field(description="Whether the result is statistically significant")
    alpha_level: float = Field(default=0.05, description="Significance level used")
    confidence_interval: Optional[List[float]] = Field(default=None, description="Confidence interval")
    effect_size: Optional[float] = Field(default=None, description="Effect size measure")
    power: Optional[float] = Field(default=None, description="Statistical power")
    interpretation: str = Field(description="Human-readable interpretation of the result")
    method_description: str = Field(description="Description of the statistical method")
    assumptions_met: Optional[bool] = Field(default=None, description="Whether test assumptions are satisfied")

    class Config:
        json_encoders = {float: lambda v: round(v, 6) if v is not None and not (np.isnan(v) or np.isinf(v)) else None}


class FeatureImportanceResult(BaseModel):
    """Feature importance comparison result."""

    feature_name: str = Field(description="Name of the feature")
    feature_index: Optional[int] = Field(default=None, description="Feature index in dataset")
    reference_importance: float = Field(description="Importance in reference model")
    current_importance: float = Field(description="Importance in current model")
    importance_change: float = Field(description="Change in importance (absolute)")
    importance_change_percent: float = Field(description="Percentage change in importance")
    rank_reference: int = Field(description="Rank in reference model")
    rank_current: int = Field(description="Rank in current model")
    rank_change: int = Field(description="Change in importance ranking")
    drift_detected: bool = Field(description="Whether drift was detected for this feature")
    drift_severity: DriftSeverity = Field(description="Severity of importance drift")
    stability_score: Optional[float] = Field(default=None, description="Stability score across time")


class ModelDisagreementResult(BaseModel):
    """Model disagreement analysis result."""

    overall_disagreement_rate: float = Field(description="Overall rate of prediction disagreement")
    class_level_disagreement: Optional[Dict[str, float]] = Field(default=None, description="Disagreement by class")
    confidence_weighted_disagreement: float = Field(description="Disagreement weighted by prediction confidence")
    disagreement_severity: DriftSeverity = Field(description="Severity of disagreement")
    problematic_regions: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Regions with high disagreement"
    )
    agreement_patterns: Dict[str, Any] = Field(description="Patterns in model agreement/disagreement")


class ConfidenceAnalysisResult(BaseModel):
    """Model confidence and calibration analysis result."""

    reference_mean_confidence: float = Field(description="Mean confidence of reference model")
    current_mean_confidence: float = Field(description="Mean confidence of current model")
    confidence_shift: float = Field(description="Shift in confidence distribution")
    calibration_analysis: Optional[Dict[str, Any]] = Field(description="Calibration analysis results")
    confidence_distribution_drift: Dict[str, Any] = Field(description="Changes in confidence distribution")
    overconfidence_analysis: Dict[str, Any] = Field(description="Analysis of overconfident predictions")


class PSIResult(BaseModel):
    """Population Stability Index result."""

    overall_psi: float = Field(description="Overall PSI value")
    psi_interpretation: str = Field(description="PSI interpretation")
    drift_level: DriftSeverity = Field(description="Drift severity based on PSI")
    is_stable: bool = Field(description="Whether the population is stable")
    feature_level_psi: Optional[Dict[str, float]] = Field(default=None, description="PSI by feature")
    bin_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Detailed bin-level analysis")
    stability_trends: Optional[Dict[str, Any]] = Field(default=None, description="Stability trend analysis")

    class Config:
        json_encoders = {float: lambda v: round(v, 6) if v is not None and not (np.isnan(v) or np.isinf(v)) else None}


class CalibrationResult(BaseModel):
    """Model calibration analysis result."""

    reference_brier_score: float = Field(description="Brier score on reference data")
    current_brier_score: float = Field(description="Brier score on current data")
    brier_score_change: float = Field(description="Change in Brier score")
    reference_ece: float = Field(description="Expected Calibration Error on reference data")
    current_ece: float = Field(description="Expected Calibration Error on current data")
    ece_change: float = Field(description="Change in Expected Calibration Error")
    calibration_change: float = Field(description="Overall change in calibration")
    is_well_calibrated: bool = Field(description="Whether the model is well calibrated")
    drift_severity: DriftSeverity = Field(description="Calibration drift severity")
    calibration_plot_data: Optional[Dict[str, Any]] = Field(default=None, description="Data for calibration plots")
    reliability_diagram_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Data for reliability diagrams"
    )


class PerformanceComparisonResponse(BaseModel):
    """Response model for performance comparison analysis (Tab 1)."""

    # Analysis metadata
    analysis_id: str = Field(description="Unique analysis identifier")
    analysis_name: str = Field(description="Name of the analysis")
    analysis_type: str = Field(default="performance_comparison", description="Type of analysis performed")
    timestamp: datetime = Field(description="Analysis timestamp")
    status: AnalysisStatus = Field(description="Analysis status")
    execution_time: Optional[float] = Field(default=None, description="Analysis execution time in seconds")

    # Dataset information
    dataset_info: Dict[str, Any] = Field(description="Information about datasets used")
    model_info: Dict[str, Any] = Field(description="Information about models analyzed")

    # Core metrics comparison results
    metrics_comparison: List[MetricResult] = Field(description="Detailed metrics comparison results")
    metrics_summary: Dict[str, Any] = Field(description="Summary of metrics comparison")

    # Drift analysis
    overall_drift_severity: DriftSeverity = Field(description="Overall drift severity assessment")
    drift_detected: bool = Field(description="Whether significant drift was detected")
    critical_degradations: int = Field(description="Number of critical performance degradations")

    # Statistical analysis results
    statistical_tests: List[StatisticalTestResult] = Field(description="Statistical significance test results")
    statistical_summary: Dict[str, Any] = Field(description="Summary of statistical test results")

    # Effect size and practical significance
    effect_size_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Effect size analysis results")
    practical_significance: Dict[str, Any] = Field(description="Assessment of practical significance")

    # Advanced analysis components
    feature_importance_analysis: Optional[List[FeatureImportanceResult]] = Field(
        default=None, description="Feature importance drift analysis"
    )
    psi_analysis: Optional[PSIResult] = Field(default=None, description="Population Stability Index analysis")
    calibration_analysis: Optional[CalibrationResult] = Field(default=None, description="Model calibration analysis")
    prediction_drift_analysis: Optional[Dict[str, Any]] = Field(
        default=None, description="Prediction distribution drift analysis"
    )

    # Performance degradation analysis
    degradation_analysis: Dict[str, Any] = Field(description="Detailed performance degradation analysis")

    # Executive summary and insights
    executive_summary: str = Field(description="High-level executive summary of findings")
    key_findings: List[str] = Field(description="Key findings from the analysis")
    recommendations: List[str] = Field(description="Recommended actions based on analysis")
    risk_assessment: Dict[str, Any] = Field(description="Risk level and impact assessment")

    # AI-powered explanations
    ai_insights: Optional[Dict[str, Any]] = Field(default=None, description="AI-generated insights and explanations")
    business_impact: Optional[Dict[str, Any]] = Field(default=None, description="Business impact assessment")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            float: lambda v: round(v, 6) if v is not None and not (np.isnan(v) or np.isinf(v)) else None,
        }
        schema_extra = {
            "example": {
                "analysis_id": "perf_comp_20250910_001",
                "analysis_name": "Monthly Production Model Review",
                "timestamp": "2025-09-10T10:00:00Z",
                "status": "success",
                "overall_drift_severity": "medium",
                "drift_detected": True,
                "critical_degradations": 2,
                "executive_summary": "Model performance has degraded moderately with significant accuracy drop requiring attention.",
                "key_findings": [
                    "Accuracy decreased by 5.2% (statistically significant)",
                    "Precision maintained but recall degraded",
                    "Feature importance shifted significantly",
                ],
                "recommendations": [
                    "Consider model retraining with recent data",
                    "Investigate data quality issues",
                    "Monitor performance closely",
                ],
            }
        }


class DegradationMetricsResponse(BaseModel):
    """Response model for degradation metrics analysis (Tab 2)."""

    # Analysis metadata
    analysis_id: str = Field(description="Unique analysis identifier")
    analysis_name: str = Field(default="Degradation Metrics Analysis", description="Analysis name")
    timestamp: datetime = Field(description="Analysis timestamp")
    status: AnalysisStatus = Field(description="Analysis status")
    execution_time: Optional[float] = Field(default=None, description="Analysis execution time in seconds")

    # Sub-tab 1: Model Disagreement Analysis
    disagreement_analysis: ModelDisagreementResult = Field(description="Model prediction disagreement analysis")

    # Sub-tab 2: Confidence and Calibration Analysis
    confidence_analysis: ConfidenceAnalysisResult = Field(description="Model confidence and calibration analysis")

    # Sub-tab 3: Feature Importance Drift Analysis
    feature_importance_analysis: List[FeatureImportanceResult] = Field(description="Feature importance drift analysis")

    # Overall degradation assessment
    overall_degradation: Dict[str, Any] = Field(description="Overall degradation assessment across all sub-tabs")
    degradation_score: float = Field(description="Numerical degradation score (0-10)")
    degradation_level: DriftSeverity = Field(description="Overall degradation severity level")

    # Key degradation indicators
    key_degradation_indicators: List[str] = Field(description="Key indicators of model degradation")
    sub_tab_summaries: Dict[str, str] = Field(description="Summary for each sub-tab analysis")

    # Risk and impact assessment
    risk_assessment: Dict[str, Any] = Field(description="Risk level and business impact assessment")
    business_impact_assessment: str = Field(description="Assessment of business impact")

    # Recommendations by priority
    high_priority_actions: List[str] = Field(description="High-priority recommended actions")
    medium_priority_actions: List[str] = Field(description="Medium-priority recommended actions")
    monitoring_recommendations: List[str] = Field(description="Ongoing monitoring recommendations")

    # Trend analysis (if historical data available)
    trend_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Degradation trend over time")

    # AI-generated insights
    ai_insights: Optional[Dict[str, Any]] = Field(default=None, description="AI-generated insights and explanations")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            float: lambda v: round(v, 4) if v is not None and not (np.isnan(v) or np.isinf(v)) else None,
        }


class StatisticalSignificanceResponse(BaseModel):
    """Response model for statistical significance analysis (Tab 3)."""

    # Analysis metadata
    analysis_id: str = Field(description="Unique analysis identifier")
    analysis_name: str = Field(default="Statistical Significance Analysis", description="Analysis name")
    timestamp: datetime = Field(description="Analysis timestamp")
    status: AnalysisStatus = Field(description="Analysis status")
    execution_time: Optional[float] = Field(default=None, description="Analysis execution time in seconds")

    # Hypothesis testing results
    hypothesis_testing: Dict[str, Any] = Field(description="Comprehensive hypothesis testing results")
    individual_tests: Dict[str, StatisticalTestResult] = Field(description="Individual statistical test results")

    # Statistical significance summary
    statistical_tests: List[StatisticalTestResult] = Field(description="All statistical test results")
    significant_tests: List[StatisticalTestResult] = Field(description="Only statistically significant tests")
    significance_summary: Dict[str, Any] = Field(description="Summary of significance findings")

    # Effect size analysis
    effect_size_analysis: Dict[str, Any] = Field(description="Comprehensive effect size analysis")
    effect_magnitude_assessment: str = Field(description="Overall effect magnitude (Small/Medium/Large)")
    practical_significance: Dict[str, Any] = Field(description="Assessment of practical significance")

    # Power analysis
    power_analysis: Dict[str, Any] = Field(description="Statistical power analysis results")

    # Multiple comparisons correction
    multiple_comparisons: Dict[str, Any] = Field(description="Multiple comparisons correction results")
    bonferroni_corrected: bool = Field(description="Whether Bonferroni correction was applied")
    adjusted_alpha: Optional[float] = Field(default=None, description="Adjusted significance level")

    # Overall statistical assessment
    overall_significance: bool = Field(description="Whether overall change is statistically significant")
    confidence_level: float = Field(description="Confidence level used for tests")
    alpha_level: float = Field(description="Significance level used")

    # Statistical interpretation
    statistical_interpretation: str = Field(description="Statistical interpretation of results")
    practical_interpretation: str = Field(description="Practical interpretation of statistical findings")
    methodological_notes: List[str] = Field(description="Important methodological considerations")

    # Recommendations
    statistical_recommendations: List[str] = Field(description="Recommendations based on statistical analysis")
    methodological_recommendations: List[str] = Field(description="Methodological improvement recommendations")

    # Future analysis suggestions
    future_analysis_suggestions: List[str] = Field(description="Suggestions for future analyses")

    # AI-generated insights
    ai_insights: Optional[Dict[str, Any]] = Field(default=None, description="AI-generated statistical insights")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            float: lambda v: round(v, 6) if v is not None and not (np.isnan(v) or np.isinf(v)) else None,
        }


class ComprehensiveAnalysisResponse(BaseModel):
    """Response model for comprehensive model drift analysis (all tabs combined)."""

    # Analysis metadata
    analysis_id: str = Field(description="Unique analysis identifier")
    analysis_name: str = Field(description="Name of the comprehensive analysis")
    timestamp: datetime = Field(description="Analysis timestamp")
    status: AnalysisStatus = Field(description="Overall analysis status")
    total_execution_time: float = Field(description="Total analysis execution time in seconds")

    # Tab-specific results
    performance_comparison: PerformanceComparisonResponse = Field(description="Performance comparison analysis results")
    degradation_metrics: DegradationMetricsResponse = Field(description="Degradation metrics analysis results")
    statistical_significance: StatisticalSignificanceResponse = Field(
        description="Statistical significance analysis results"
    )

    # Overall assessment across all tabs
    overall_assessment: Dict[str, Any] = Field(description="Comprehensive assessment across all analyses")
    overall_drift_severity: DriftSeverity = Field(description="Overall drift severity across all tabs")
    overall_risk_level: DriftSeverity = Field(description="Overall business risk level")

    # Consolidated insights
    key_findings: List[str] = Field(description="Key findings from all analyses")
    critical_issues: List[str] = Field(description="Critical issues requiring immediate attention")
    opportunities: List[str] = Field(description="Identified opportunities for improvement")

    # Prioritized recommendations
    immediate_actions: List[str] = Field(description="Actions requiring immediate attention")
    short_term_actions: List[str] = Field(description="Short-term recommended actions")
    long_term_strategies: List[str] = Field(description="Long-term strategic recommendations")

    # Business impact assessment
    business_impact: Dict[str, Any] = Field(description="Comprehensive business impact assessment")
    financial_impact_estimate: Optional[Dict[str, Any]] = Field(default=None, description="Estimated financial impact")

    # Confidence and reliability
    analysis_confidence: float = Field(description="Confidence score for the overall analysis (0-1)")
    reliability_assessment: Dict[str, Any] = Field(description="Assessment of analysis reliability")

    # AI-generated insights
    executive_ai_summary: Optional[Dict[str, Any]] = Field(default=None, description="AI-generated executive summary")
    technical_ai_insights: Optional[Dict[str, Any]] = Field(default=None, description="AI-generated technical insights")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            float: lambda v: round(v, 4) if v is not None and not (np.isnan(v) or np.isinf(v)) else None,
        }


class ModelDriftSummary(BaseModel):
    """High-level summary of model drift analysis."""

    analysis_id: str = Field(description="Analysis identifier")
    model_name: str = Field(description="Name of the model analyzed")
    analysis_date: datetime = Field(description="Date of analysis")

    # Quick status indicators
    drift_detected: bool = Field(description="Whether drift was detected")
    severity_level: DriftSeverity = Field(description="Overall severity level")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in drift detection")

    # Key metrics
    performance_change: float = Field(description="Overall performance change percentage")
    critical_metrics_affected: int = Field(description="Number of critical metrics affected")
    statistical_significance: bool = Field(description="Whether changes are statistically significant")

    # Quick recommendations
    action_required: bool = Field(description="Whether immediate action is required")
    recommended_action: str = Field(description="Primary recommended action")

    # Links to detailed analysis
    detailed_report_available: bool = Field(default=True, description="Whether detailed report is available")


class AnalysisMetadata(BaseModel):
    """Metadata for model drift analysis."""

    # Analysis identification
    analysis_id: str = Field(description="Unique analysis identifier")
    analysis_version: str = Field(default="1.0", description="Version of the analysis")
    created_at: datetime = Field(description="Analysis creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")

    # Configuration used
    configuration_snapshot: Dict[str, Any] = Field(description="Configuration used for this analysis")

    # Data information
    reference_dataset_info: Dict[str, Any] = Field(description="Information about reference dataset")
    current_dataset_info: Dict[str, Any] = Field(description="Information about current dataset")

    # Model information
    reference_model_info: Dict[str, Any] = Field(description="Reference model information")
    current_model_info: Dict[str, Any] = Field(description="Current model information")

    # Analysis execution details
    execution_environment: Dict[str, Any] = Field(description="Execution environment details")
    processing_time_breakdown: Dict[str, float] = Field(description="Time breakdown by analysis component")

    # Quality metrics
    data_quality_score: Optional[float] = Field(default=None, description="Data quality assessment score")
    analysis_completeness: float = Field(description="Completeness of analysis (0-1)")

    # Reproducibility information
    random_seed: Optional[int] = Field(default=None, description="Random seed used for reproducibility")
    library_versions: Dict[str, str] = Field(description="Version of key libraries used")


class ErrorResponse(BaseModel):
    """Enhanced error response model."""

    status: str = Field(default="error", description="Response status")
    error_code: str = Field(description="Standardized error code")
    error_type: str = Field(description="Type of error (validation, processing, system)")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

    # Context information
    analysis_id: Optional[str] = Field(default=None, description="Analysis ID where error occurred")
    component: Optional[str] = Field(default=None, description="Component where error occurred")

    # Error recovery information
    recoverable: bool = Field(default=False, description="Whether the error is recoverable")
    suggested_actions: List[str] = Field(default=[], description="Suggested actions to resolve the error")

    # Technical details (for debugging)
    stack_trace: Optional[str] = Field(default=None, description="Stack trace (if available)")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


class HealthCheckResponse(BaseModel):
    """Enhanced health check response model."""

    status: str = Field(description="Overall service status")
    timestamp: datetime = Field(description="Health check timestamp")
    version: str = Field(description="API version")
    uptime: str = Field(description="Service uptime")

    # Detailed component status
    components: Dict[str, Dict[str, Any]] = Field(description="Status of individual components")
    dependencies: Dict[str, str] = Field(description="External dependency status")

    # Performance metrics
    performance_metrics: Dict[str, Any] = Field(description="Performance metrics")
    resource_usage: Dict[str, Any] = Field(description="Resource usage information")

    # Service capabilities
    available_features: List[str] = Field(description="List of available features")
    supported_model_types: List[str] = Field(description="Supported model types")
    supported_file_formats: List[str] = Field(description="Supported file formats")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-09-10T10:00:00Z",
                "version": "2.0.0",
                "uptime": "5 days, 12 hours",
                "components": {
                    "database": {"status": "healthy", "latency_ms": 15},
                    "ai_service": {"status": "healthy", "latency_ms": 250},
                    "file_storage": {"status": "healthy", "available_space_gb": 500},
                },
                "dependencies": {"aws_bedrock": "healthy", "redis_cache": "healthy", "postgresql": "healthy"},
                "available_features": [
                    "performance_comparison",
                    "degradation_metrics",
                    "statistical_significance",
                    "ai_explanations",
                ],
            }
        }


# Utility classes for response construction
class ResponseBuilder:
    """Utility class for building structured responses."""

    @staticmethod
    def build_success_response(
        analysis_results: Dict[str, Any], analysis_type: str, analysis_id: str, execution_time: float
    ) -> Dict[str, Any]:
        """Build a successful analysis response."""
        return {
            "status": AnalysisStatus.SUCCESS,
            "analysis_id": analysis_id,
            "analysis_type": analysis_type,
            "timestamp": datetime.now(),
            "execution_time": execution_time,
            **analysis_results,
        }

    @staticmethod
    def build_error_response(
        error_message: str,
        error_code: str = "ANALYSIS_ERROR",
        error_type: str = "processing",
        analysis_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> ErrorResponse:
        """Build an error response."""
        return ErrorResponse(
            error_code=error_code,
            error_type=error_type,
            message=error_message,
            analysis_id=analysis_id,
            details=details or {},
        )

    @staticmethod
    def build_partial_response(
        successful_analyses: Dict[str, Any], failed_analyses: Dict[str, Any], analysis_id: str
    ) -> Dict[str, Any]:
        """Build a partial success response."""
        return {
            "status": AnalysisStatus.PARTIAL,
            "analysis_id": analysis_id,
            "timestamp": datetime.now(),
            "successful_analyses": successful_analyses,
            "failed_analyses": failed_analyses,
            "summary": f"Completed {len(successful_analyses)} of {len(successful_analyses) + len(failed_analyses)} analyses",
        }


# Response validation utilities
class ResponseValidator:
    """Utility class for validating response structures."""

    @staticmethod
    def validate_drift_severity(severity: str) -> bool:
        """Validate drift severity value."""
        return severity in [s.value for s in DriftSeverity]

    @staticmethod
    def validate_metric_result(result: Dict[str, Any]) -> List[str]:
        """Validate metric result structure."""
        required_fields = ["name", "reference_value", "current_value", "difference", "drift_severity"]
        missing_fields = [field for field in required_fields if field not in result]
        return missing_fields

    @staticmethod
    def validate_statistical_test_result(result: Dict[str, Any]) -> List[str]:
        """Validate statistical test result structure."""
        required_fields = ["test_name", "statistic", "p_value", "is_significant"]
        missing_fields = [field for field in required_fields if field not in result]

        # Validate p_value range
        if "p_value" in result and not (0 <= result["p_value"] <= 1):
            missing_fields.append("p_value_out_of_range")

        return missing_fields


# Core response models for enhanced_model_service
class ModelInfo(BaseModel):
    """Model information and metadata."""

    model_type: str = Field(description="Type of model (sklearn, xgboost, etc.)")
    model_class: str = Field(description="Full class path of the model")
    reference_samples: int = Field(description="Number of reference samples")
    current_samples: int = Field(description="Number of current samples")
    features: List[str] = Field(description="List of feature names")
    is_pipeline: bool = Field(description="Whether the model is a pipeline")


class PerformanceMetrics(BaseModel):
    """Performance metrics for reference and current datasets."""

    reference: Dict[str, float] = Field(description="Metrics calculated on reference dataset")
    current: Dict[str, float] = Field(description="Metrics calculated on current dataset")

    class Config:
        json_encoders = {np.generic: lambda v: v.item()}


class DriftMetrics(BaseModel):
    """Drift severity and threshold analysis."""

    metric_drifts: Dict[str, float] = Field(description="Drift values for each metric")
    overall_drift: float = Field(description="Overall drift score")
    drift_severity: str = Field(description="Drift severity level (None, Low, Medium, High)")
    drift_detected: bool = Field(description="Whether drift was detected")

    class Config:
        json_encoders = {np.generic: lambda v: v.item()}


class AnalysisResults(BaseModel):
    """Complete analysis results from enhanced model service."""

    performance_comparison: Optional[Dict[str, Any]] = Field(
        default=None, description="Performance comparison analysis"
    )
    degradation_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Degradation metrics analysis")
    statistical_significance: Optional[Dict[str, Any]] = Field(
        default=None, description="Statistical significance analysis"
    )
    feature_importance_drift: Optional[List[FeatureImportanceResult]] = Field(
        default=None, description="Feature importance drift analysis"
    )
    model_disagreement: Optional[ModelDisagreementResult] = Field(
        default=None, description="Model disagreement analysis"
    )
    confidence_analysis: Optional[ConfidenceAnalysisResult] = Field(
        default=None, description="Confidence analysis results"
    )
    psi_analysis: Optional[PSIResult] = Field(default=None, description="Population Stability Index analysis")
    calibration_analysis: Optional[CalibrationResult] = Field(default=None, description="Model calibration analysis")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat(), np.generic: lambda v: v.item()}


class ModelDriftResponse(BaseModel):
    """Main response model for enhanced model drift analysis."""

    analysis_id: str = Field(description="Unique analysis identifier")
    model_info: ModelInfo = Field(description="Model information")
    timestamp: str = Field(description="Analysis timestamp")
    config: Dict[str, Any] = Field(description="Analysis configuration")
    performance_comparison: Optional[Dict[str, Any]] = Field(default=None, description="Performance comparison results")
    degradation_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Degradation metrics results")
    statistical_significance: Optional[Dict[str, Any]] = Field(
        default=None, description="Statistical significance results"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat(), np.generic: lambda v: v.item()}

    warnings: Optional[List[str]] = Field(default=None, description="Analysis warnings")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat(), np.generic: lambda v: v.item()}


# Legacy compatibility models (for backward compatibility)
class LegacyModelDriftResponse(BaseModel):
    """Legacy response model for backward compatibility."""

    status: str = Field(description="Analysis status")
    results: Dict[str, Any] = Field(description="Analysis results")
    timestamp: datetime = Field(description="Analysis timestamp")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


# Sanity Check Response Models
class SanityCheckScatterPoint(BaseModel):
    """Individual feature data point for sanity check scatter plot."""

    feature: str = Field(description="Feature name")
    similarity: float = Field(description="Similarity score (0-1, where 1 is identical)")
    expected_drop: float = Field(description="Expected performance drop (0-1)")
    importance: Optional[float] = Field(default=None, description="Feature importance weight (0-1)")
    severity: str = Field(description="Drift severity: stable, low, moderate, high, critical")
    drift_type: str = Field(description="Feature type: numerical, categorical")
    test_statistic: Optional[float] = Field(default=None, description="Statistical test value")
    p_value: Optional[float] = Field(default=None, description="P-value from statistical test")

    class Config:
        json_encoders = {float: lambda v: round(v, 6) if v is not None and not (np.isnan(v) or np.isinf(v)) else None}


class DistributionComparison(BaseModel):
    """Distribution comparison statistics for a single feature."""

    feature: str = Field(description="Feature name")
    reference_stats: Dict[str, Any] = Field(description="Reference dataset statistics")
    current_stats: Dict[str, Any] = Field(description="Current dataset statistics")
    test_statistic: float = Field(description="Statistical test statistic")
    p_value: float = Field(description="P-value from statistical test")
    drift_detected: bool = Field(description="Whether significant drift was detected")

    class Config:
        json_encoders = {float: lambda v: round(v, 6) if v is not None and not (np.isnan(v) or np.isinf(v)) else None}


class SanityCheckSummary(BaseModel):
    """Overall sanity check summary."""

    expected_metric: str = Field(description="Expected performance metric name (e.g., F1, accuracy)")
    predicted_value: float = Field(description="Predicted metric value")
    confidence: float = Field(description="Confidence level in prediction (0-1)")
    drift_type: str = Field(description="Inferred drift type: data, concept, or stable")
    severity: str = Field(description="Overall severity: stable, low, moderate, high, critical")
    alert_level: str = Field(description="Alert level with emoji: 🟢 Stable, 🟡 Monitor, 🔴 Warning")
    overall_similarity_score: float = Field(description="Aggregated similarity score (0-1)")
    num_features_analyzed: int = Field(description="Number of features analyzed")
    num_drifted_features: int = Field(description="Number of features with significant drift")

    class Config:
        json_encoders = {float: lambda v: round(v, 6) if v is not None and not (np.isnan(v) or np.isinf(v)) else None}


class SanityCheckResponse(BaseModel):
    """Complete sanity check analysis response."""

    analysis_id: str = Field(description="Unique analysis identifier")
    timestamp: str = Field(description="Analysis timestamp")
    model_info: Optional[Dict[str, Any]] = Field(default=None, description="Model information if available")
    sanity_check_summary: SanityCheckSummary = Field(description="Overall sanity check summary")
    scatter_points: List[SanityCheckScatterPoint] = Field(description="Feature-level scatter plot data")
    distribution_comparisons: List[DistributionComparison] = Field(
        description="Top drifted features distribution comparison"
    )
    recommendations: List[str] = Field(description="Actionable recommendations")
    llm_response: Optional[Dict[str, Any]] = Field(default=None, description="AI-generated explanation")
    warnings: Optional[List[str]] = Field(default=None, description="Analysis warnings")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None, np.generic: lambda v: v.item()}
