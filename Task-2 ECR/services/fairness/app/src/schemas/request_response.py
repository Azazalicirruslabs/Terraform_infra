from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    status: str
    message: str
    train_shape: Optional[List[int]] = None
    test_shape: Optional[List[int]] = None
    columns: Optional[List[str]] = None

    class Config:
        str_strip_whitespace = True
        validate_assignment = True


class TargetColumnRequest(BaseModel):
    target_column: str


class ConfigurationResponse(BaseModel):
    status: str
    message: str
    configured: bool


class AnalysisRequest(BaseModel):
    target_column: Optional[str] = None  # Optional for backward compatibility with global_session
    n_top_features: int = 10


class FeatureSensitivityScore(BaseModel):
    feature: str
    nocco_score: float
    is_sensitive: bool
    percentile_rank: float
    test_type: str = "unknown"


class AnalysisResponse(BaseModel):
    status: str
    analysis_type: str
    target_column: str
    total_features: int
    sensitive_features_count: int
    threshold_value: float
    feature_scores: List[FeatureSensitivityScore]
    sensitive_features: List[str]
    analysis_summary: Dict[str, Any]


class StatusResponse(BaseModel):
    status: str
    train_uploaded: bool
    test_uploaded: bool
    model_uploaded: bool
    target_column: Optional[str] = None
    train_shape: Optional[tuple] = None
    test_shape: Optional[tuple] = None
    columns: Optional[List[str]] = None


class FeatureRelationshipResponse(BaseModel):
    primary_feature: str
    related_features: List[Dict[str, Any]]


class BiasAnalysisRequest(BaseModel):
    target_column: str
    sensitive_feature_column: str
    prediction_column: Optional[str] = None
    prediction_proba_column: Optional[str] = None
    feature_types: Optional[Dict[str, str]] = None  # Mapping of column name -> "categorical" or "numerical"
    task_type: Optional[str] = None  # "classification" or "regression"


class BiasMetricResult(BaseModel):
    metric: str
    is_biased: bool
    threshold: float
    description: str


class BiasAnalysisResponse(BaseModel):
    status: str
    analysis_type: str
    sensitive_feature: str
    sensitive_feature_type: Optional[str] = "categorical"  # "categorical" or "numerical"
    total_metrics: int
    biased_metrics_count: int
    overall_bias_status: str
    metrics_results: Dict[str, Any]
    recommendations: List[str]


class PipelineRequest(BaseModel):
    strategies: List[str]
    sensitive_feature_column: str
    target_column: str
    prediction_column: Optional[str] = None


class MitigationRequest(BaseModel):
    strategy: str
    sensitive_feature_column: str
    target_column: str
    prediction_column: Optional[str] = None


class BiasDetectionResponse(BaseModel):
    status: str
    biased_metrics_count: int
    biased_metrics: List[str]
    metrics: Dict[str, Any]


class MitigationResponse(BaseModel):
    status: str
    strategy_applied: str
    bias_assessment: Dict[str, Any]
    improvement_analysis: Dict[str, Any]
    recommendations: Optional[List[str]] = None


class PipelineResponse(BaseModel):
    status: str
    strategies_applied: List[str]
    initial_bias: Dict[str, Any]
    final_bias: Dict[str, Any]
    overall_improvement: int
    stage_results: Dict[str, Any]
    recommendations: Optional[List[str]] = None


class FeatureInteractionRequest(BaseModel):
    columns: List[str]
    interaction_type: str = "concatenate"
    prediction_column: Optional[str] = None
    combinations: Optional[List[List[str]]] = None
    max_combinations: Optional[int] = 10
    include_individual_bias: Optional[bool] = True
    target_column: str


class InteractionBiasDetail(BaseModel):
    interaction_name: str
    source_columns: List[str]
    interaction_type: str
    unique_groups: int
    sample_values: List[str] = []
    bias_metrics: Dict[str, Any]
    biased_metrics_count: int
    total_metrics: int
    bias_percentage: float
    bias_status: str
    bias_amplification: float
    severity_score: float
    recommendations: List[str] = []


class FeatureInteractionResponse(BaseModel):
    status: str
    total_interactions: int
    interactions_analyzed: int
    baseline_bias: Dict[str, Any]
    interaction_results: List[InteractionBiasDetail]
    most_biased_interaction: Optional[str] = None
    least_biased_interaction: Optional[str] = None
    summary: Dict[str, Any]
    visualization_data: Optional[Dict[str, Any]] = None
    execution_time_seconds: Optional[float] = None


class InteractionPreviewRequest(BaseModel):
    columns: List[str]
    interaction_type: str = "concatenate"
    sample_size: Optional[int] = 5
    target_column: str


class InteractionPreviewResponse(BaseModel):
    interaction_name: str
    source_columns: List[str]
    interaction_type: str
    unique_groups: int
    sample_values: List[Dict[str, Any]]
    group_distribution: Dict[str, int]
    estimated_analysis_time: str


class InteractionComparisonRequest(BaseModel):
    columns: List[str]
    interaction_types: List[str] = ["concatenate", "multiply"]
    prediction_column: Optional[str] = None
    target_column: str


class SavedInteractionRequest(BaseModel):
    interaction_name: str
    notes: Optional[str] = None


class SmartInteractionRequest(BaseModel):
    columns: List[str]
    interaction_size: Optional[int] = None

    def validate_interaction_size(self, v):
        if v is not None and v < 2:
            raise ValueError("interaction_size must be >= 2")
        if v is not None and v > 5:
            raise ValueError("interaction_size must be <= 5")
        return v

    interaction_type: Optional[str] = None
    auto_detect_types: bool = True
    prediction_column: Optional[str] = None
    max_interactions: int = 30
    include_type_recommendations: bool = True
    target_column: str


class InteractionTypeRecommendation(BaseModel):
    columns: List[str]
    recommended_type: str
    reason: str
    column_types: Dict[str, str]
    alternatives: List[str]


class SmartInteractionResponse(BaseModel):
    status: str
    analysis_mode: str
    total_combinations: int
    interactions_analyzed: int
    baseline_bias: Dict[str, Any]
    interaction_results: List[InteractionBiasDetail]
    type_recommendations: Optional[List[InteractionTypeRecommendation]] = None
    most_biased_interaction: Optional[str] = None
    least_biased_interaction: Optional[str] = None
    summary: Dict[str, Any]
    visualization_data: Optional[Dict[str, Any]] = None
    execution_time_seconds: Optional[float] = None


class FlexibleInteractionRequest(BaseModel):
    interactions: List[Dict[str, Any]]
    prediction_column: Optional[str] = None
    auto_detect_missing_types: bool = True
    target_column: str


class ColumnAnalysisResponse(BaseModel):
    column_name: str
    data_type: str
    unique_values: int
    missing_percentage: float
    sample_values: List[str]
    recommended_interaction_types: List[str]
    compatible_columns: List[str]
