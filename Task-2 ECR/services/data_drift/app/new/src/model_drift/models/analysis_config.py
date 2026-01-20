"""
Analysis Configuration Models for Model Drift Detection
Defines configuration structures for drift analysis parameters and thresholds.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Supported model types for drift analysis."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


class StatisticalTest(str, Enum):
    """Available statistical tests for model performance comparison."""

    MCNEMAR = "mcnemar"
    DELONG = "delong"
    BOOTSTRAP = "bootstrap_confidence"
    FIVE_TWO_CV = "five_two_cv"
    PAIRED_TTEST = "paired_ttest"
    DIEBOLD_MARIANO = "diebold_mariano"
    WILCOXON = "wilcoxon"
    CHI_SQUARE = "chi_square"


class DriftSeverity(str, Enum):
    """Drift severity levels."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalysisType(str, Enum):
    """Available analysis types."""

    PERFORMANCE_COMPARISON = "performance_comparison"
    DEGRADATION_METRICS = "degradation_metrics"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    COMPREHENSIVE = "comprehensive"


class FeatureAnalysisConfig(BaseModel):
    """Configuration for feature-level analysis."""

    analyze_feature_importance: bool = Field(default=True, description="Analyze feature importance drift")
    feature_importance_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Threshold for significant importance change"
    )
    top_n_features: int = Field(default=10, ge=1, le=50, description="Number of top features to analyze")
    include_permutation_importance: bool = Field(default=False, description="Calculate permutation importance")
    feature_stability_analysis: bool = Field(default=True, description="Analyze feature stability over time")


class DriftThresholds(BaseModel):
    """Threshold configuration for drift severity classification."""

    low_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Low drift threshold (5%)")
    medium_threshold: float = Field(default=0.15, ge=0.0, le=1.0, description="Medium drift threshold (15%)")
    high_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="High drift threshold (25%)")
    critical_threshold: float = Field(default=0.40, ge=0.0, le=1.0, description="Critical drift threshold (40%)")

    def classify_drift(self, change_magnitude: float) -> DriftSeverity:
        """Classify drift severity based on change magnitude."""
        abs_change = abs(change_magnitude)
        if abs_change >= self.critical_threshold:
            return DriftSeverity.CRITICAL
        elif abs_change >= self.high_threshold:
            return DriftSeverity.HIGH
        elif abs_change >= self.medium_threshold:
            return DriftSeverity.MEDIUM
        elif abs_change >= self.low_threshold:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.MINIMAL


class StatisticalTestConfig(BaseModel):
    """Configuration for statistical testing."""

    primary_test: Union[str, StatisticalTest] = Field(
        default=StatisticalTest.MCNEMAR, description="Primary statistical test"
    )
    secondary_tests: List[Union[str, StatisticalTest]] = Field(
        default=[], description="Additional statistical tests to run"
    )
    alpha_level: float = Field(default=0.05, ge=0.01, le=0.10, description="Significance level for hypothesis testing")
    bonferroni_correction: bool = Field(default=True, description="Apply Bonferroni correction for multiple testing")
    effect_size_threshold: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Minimum effect size for practical significance"
    )
    power_analysis: bool = Field(default=True, description="Perform statistical power analysis")


class CalibrationAnalysisConfig(BaseModel):
    """Configuration for model calibration analysis."""

    analyze_calibration: bool = Field(default=True, description="Analyze model calibration drift")
    calibration_bins: int = Field(default=10, ge=5, le=20, description="Number of bins for calibration analysis")
    brier_score_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Threshold for significant Brier score change"
    )
    reliability_diagram: bool = Field(default=True, description="Generate reliability diagrams")
    ece_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Expected Calibration Error threshold")


class PSIAnalysisConfig(BaseModel):
    """Configuration for Population Stability Index analysis."""

    calculate_psi: bool = Field(default=True, description="Calculate Population Stability Index")
    psi_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="PSI threshold for drift detection")
    psi_bins: int = Field(default=10, ge=5, le=20, description="Number of bins for PSI calculation")
    feature_level_psi: bool = Field(default=True, description="Calculate PSI at feature level")
    prediction_level_psi: bool = Field(default=True, description="Calculate PSI for predictions")


class ModelDisagreementConfig(BaseModel):
    """Configuration for model disagreement analysis."""

    analyze_disagreement: bool = Field(default=True, description="Analyze model prediction disagreement")
    disagreement_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Threshold for significant disagreement"
    )
    class_level_disagreement: bool = Field(default=True, description="Analyze disagreement by class")
    confidence_weighted_disagreement: bool = Field(
        default=True, description="Weight disagreement by prediction confidence"
    )


class MetricConfiguration(BaseModel):
    """Configuration for specific metrics calculation."""

    include_confidence_intervals: bool = Field(default=True, description="Include confidence intervals for metrics")
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0, description="Confidence level for intervals")
    bootstrap_samples: int = Field(default=1000, ge=100, le=10000, description="Number of bootstrap samples")
    cross_validation_folds: int = Field(default=5, ge=3, le=10, description="Number of CV folds for robust estimation")
    metric_stability_analysis: bool = Field(default=True, description="Analyze metric stability across subsets")
    outlier_detection: bool = Field(default=False, description="Detect and handle outliers in metric calculation")


class ReportingConfig(BaseModel):
    """Configuration for analysis reporting and output."""

    generate_plots: bool = Field(default=True, description="Generate visualization plots")
    include_raw_data: bool = Field(default=False, description="Include raw data in response")
    executive_summary: bool = Field(default=True, description="Generate executive summary")
    technical_details: bool = Field(default=True, description="Include technical analysis details")
    business_impact_assessment: bool = Field(default=True, description="Generate business impact assessment")
    export_format: List[str] = Field(default=["json"], description="Export formats for reports")
    ai_explanation: bool = Field(default=True, description="Generate AI-powered explanations")


class AnalysisConfiguration(BaseModel):
    """Complete configuration for model drift analysis."""

    # Basic analysis metadata
    analysis_id: Optional[str] = Field(default=None, description="Unique analysis identifier")
    analysis_name: str = Field(default="Model Drift Analysis", description="Name of the analysis")
    description: Optional[str] = Field(default=None, description="Description of the analysis purpose and scope")
    analysis_type: AnalysisType = Field(default=AnalysisType.COMPREHENSIVE, description="Type of analysis to perform")
    timestamp: Optional[datetime] = Field(default=None, description="Analysis execution timestamp")

    # Model configuration
    model_type: ModelType = Field(default=ModelType.CLASSIFICATION, description="Type of ML model being analyzed")
    model_name: Optional[str] = Field(default=None, description="Name/identifier of the model")
    model_version_ref: Optional[str] = Field(default="reference", description="Reference model version")
    model_version_curr: Optional[str] = Field(default="current", description="Current model version")

    # Metrics selection and configuration
    selected_metrics: List[str] = Field(
        default=["accuracy", "precision", "recall", "f1_score"], description="List of metrics to calculate and compare"
    )
    custom_metrics: Optional[Dict[str, str]] = Field(
        default=None, description="Custom metric definitions (name: formula)"
    )

    # Statistical testing configuration
    statistical_config: StatisticalTestConfig = Field(
        default_factory=StatisticalTestConfig, description="Configuration for statistical significance testing"
    )

    # Drift thresholds and classification
    drift_thresholds: DriftThresholds = Field(
        default_factory=DriftThresholds, description="Thresholds for drift severity classification"
    )

    # Advanced metric calculation settings
    metric_config: MetricConfiguration = Field(
        default_factory=MetricConfiguration, description="Advanced metric calculation settings"
    )

    # Feature analysis configuration
    feature_analysis: FeatureAnalysisConfig = Field(
        default_factory=FeatureAnalysisConfig, description="Feature-level analysis configuration"
    )

    # Calibration analysis configuration
    calibration_config: CalibrationAnalysisConfig = Field(
        default_factory=CalibrationAnalysisConfig, description="Model calibration analysis settings"
    )

    # PSI analysis configuration
    psi_config: PSIAnalysisConfig = Field(
        default_factory=PSIAnalysisConfig, description="Population Stability Index analysis settings"
    )

    # Model disagreement analysis
    disagreement_config: ModelDisagreementConfig = Field(
        default_factory=ModelDisagreementConfig, description="Model disagreement analysis configuration"
    )

    # Reporting and output configuration
    reporting_config: ReportingConfig = Field(
        default_factory=ReportingConfig, description="Report generation and output configuration"
    )

    # Advanced options
    parallel_processing: bool = Field(default=True, description="Enable parallel processing where possible")
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    memory_efficient: bool = Field(default=False, description="Use memory-efficient algorithms for large datasets")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat() if v else None}
        use_enum_values = True
        validate_assignment = True

    def get_metrics_for_model_type(self) -> List[str]:
        """Get appropriate metrics based on model type."""
        if self.model_type == ModelType.CLASSIFICATION:
            return CLASSIFICATION_METRICS
        elif self.model_type == ModelType.BINARY_CLASSIFICATION:
            return BINARY_CLASSIFICATION_METRICS
        elif self.model_type == ModelType.REGRESSION:
            return REGRESSION_METRICS
        else:
            return self.selected_metrics

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the configuration and return any issues."""
        issues = []
        warnings = []

        # Check metric compatibility
        available_metrics = self.get_metrics_for_model_type()
        invalid_metrics = [m for m in self.selected_metrics if m not in available_metrics]
        if invalid_metrics:
            issues.append(f"Invalid metrics for {self.model_type}: {invalid_metrics}")

        # Check threshold consistency
        thresholds = [
            self.drift_thresholds.low_threshold,
            self.drift_thresholds.medium_threshold,
            self.drift_thresholds.high_threshold,
            self.drift_thresholds.critical_threshold,
        ]
        if thresholds != sorted(thresholds):
            issues.append("Drift thresholds should be in ascending order")

        # Check statistical test compatibility
        if self.model_type == ModelType.REGRESSION and self.statistical_config.primary_test == StatisticalTest.MCNEMAR:
            warnings.append("McNemar test is not suitable for regression models")

        return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings}


class AnalysisConfig(BaseModel):
    """Simplified analysis configuration for API requests."""

    analysis_name: str = Field(default="Model Drift Analysis")
    description: str = Field(default="Model drift analysis")
    model_type: str = Field(default="classification")
    selected_metrics: List[str] = Field(default=["accuracy", "precision", "recall"])
    statistical_test: str = Field(default="mcnemar")
    low_threshold: float = Field(default=0.05)
    medium_threshold: float = Field(default=0.15)
    high_threshold: float = Field(default=0.25)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat() if v else None}
        schema_extra = {
            "example": {
                "analysis_name": "Production Model Analysis",
                "description": "Monthly model drift analysis",
                "model_type": "classification",
                "selected_metrics": ["accuracy", "precision", "recall", "f1_score"],
                "statistical_test": "mcnemar",
                "low_threshold": 0.05,
                "medium_threshold": 0.15,
                "high_threshold": 0.25,
            }
        }

    def to_full_configuration(self) -> AnalysisConfiguration:
        """Convert simplified config to full configuration."""
        thresholds = DriftThresholds(
            low_threshold=self.low_threshold, medium_threshold=self.medium_threshold, high_threshold=self.high_threshold
        )

        statistical_config = StatisticalTestConfig(primary_test=self.statistical_test)

        return AnalysisConfiguration(
            analysis_name=self.analysis_name,
            description=self.description,
            model_type=ModelType(self.model_type),
            selected_metrics=self.selected_metrics,
            statistical_config=statistical_config,
            drift_thresholds=thresholds,
        )


# Predefined analysis templates for common use cases
class AnalysisTemplate(BaseModel):
    """Predefined analysis templates for common scenarios."""

    @staticmethod
    def production_monitoring() -> AnalysisConfiguration:
        """Template for production model monitoring."""
        return AnalysisConfiguration(
            analysis_name="Production Model Monitoring",
            description="Comprehensive monitoring for production model drift",
            analysis_type=AnalysisType.COMPREHENSIVE,
            statistical_config=StatisticalTestConfig(
                primary_test=StatisticalTest.MCNEMAR,
                secondary_tests=[StatisticalTest.BOOTSTRAP, StatisticalTest.DELONG],
                bonferroni_correction=True,
            ),
            feature_analysis=FeatureAnalysisConfig(
                analyze_feature_importance=True, top_n_features=15, include_permutation_importance=True
            ),
            reporting_config=ReportingConfig(
                generate_plots=True, executive_summary=True, business_impact_assessment=True, ai_explanation=True
            ),
        )

    @staticmethod
    def model_validation() -> AnalysisConfiguration:
        """Template for model validation and A/B testing."""
        return AnalysisConfiguration(
            analysis_name="Model Validation Analysis",
            description="Statistical validation for model comparison",
            analysis_type=AnalysisType.STATISTICAL_SIGNIFICANCE,
            statistical_config=StatisticalTestConfig(
                primary_test=StatisticalTest.FIVE_TWO_CV,
                secondary_tests=[StatisticalTest.BOOTSTRAP, StatisticalTest.PAIRED_TTEST],
                effect_size_threshold=0.1,
                power_analysis=True,
            ),
            metric_config=MetricConfiguration(
                include_confidence_intervals=True, bootstrap_samples=5000, cross_validation_folds=5
            ),
            reporting_config=ReportingConfig(technical_details=True, include_raw_data=False),
        )

    @staticmethod
    def quick_check() -> AnalysisConfiguration:
        """Template for quick drift detection."""
        return AnalysisConfiguration(
            analysis_name="Quick Drift Check",
            description="Fast drift detection for routine monitoring",
            analysis_type=AnalysisType.PERFORMANCE_COMPARISON,
            selected_metrics=["accuracy", "precision", "recall"],
            feature_analysis=FeatureAnalysisConfig(analyze_feature_importance=False),
            calibration_config=CalibrationAnalysisConfig(analyze_calibration=False),
            metric_config=MetricConfiguration(bootstrap_samples=100, cross_validation_folds=3),
            reporting_config=ReportingConfig(generate_plots=False, technical_details=False),
        )

    @staticmethod
    def regression_model() -> AnalysisConfiguration:
        """Template for regression model drift analysis."""
        return AnalysisConfiguration(
            analysis_name="Regression Model Drift Analysis",
            description="Comprehensive drift analysis for regression models",
            model_type=ModelType.REGRESSION,
            selected_metrics=["mse", "mae", "rmse", "r2", "explained_variance_score"],
            statistical_config=StatisticalTestConfig(
                primary_test=StatisticalTest.PAIRED_TTEST,
                secondary_tests=[StatisticalTest.DIEBOLD_MARIANO, StatisticalTest.BOOTSTRAP],
            ),
            calibration_config=CalibrationAnalysisConfig(analyze_calibration=False),  # Not applicable for regression
            feature_analysis=FeatureAnalysisConfig(analyze_feature_importance=True, top_n_features=20),
        )


# Default configurations for different model types
DEFAULT_CLASSIFICATION_CONFIG = AnalysisConfiguration(
    analysis_name="Classification Model Drift Analysis",
    model_type=ModelType.CLASSIFICATION,
    selected_metrics=["accuracy", "precision", "recall", "f1_score", "roc_auc", "cohen_kappa"],
    statistical_config=StatisticalTestConfig(
        primary_test=StatisticalTest.MCNEMAR, secondary_tests=[StatisticalTest.DELONG, StatisticalTest.BOOTSTRAP]
    ),
)

DEFAULT_REGRESSION_CONFIG = AnalysisConfiguration(
    analysis_name="Regression Model Drift Analysis",
    model_type=ModelType.REGRESSION,
    selected_metrics=["mse", "mae", "rmse", "r2", "explained_variance_score"],
    statistical_config=StatisticalTestConfig(
        primary_test=StatisticalTest.PAIRED_TTEST,
        secondary_tests=[StatisticalTest.DIEBOLD_MARIANO, StatisticalTest.BOOTSTRAP],
    ),
    calibration_config=CalibrationAnalysisConfig(analyze_calibration=False),
)

DEFAULT_BINARY_CLASSIFICATION_CONFIG = AnalysisConfiguration(
    analysis_name="Binary Classification Model Drift Analysis",
    model_type=ModelType.BINARY_CLASSIFICATION,
    selected_metrics=["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "mcc"],
    statistical_config=StatisticalTestConfig(
        primary_test=StatisticalTest.MCNEMAR, secondary_tests=[StatisticalTest.DELONG, StatisticalTest.FIVE_TWO_CV]
    ),
)

# Comprehensive metric mappings for different model types
CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "log_loss",
    "matthews_corrcoef",
    "balanced_accuracy",
    "cohen_kappa",
    "specificity",
    "sensitivity",
    "npv",
    "ppv",
    "mcc",
]

REGRESSION_METRICS = [
    "mse",
    "mae",
    "rmse",
    "r2",
    "adjusted_r2",
    "mean_absolute_percentage_error",
    "median_absolute_error",
    "explained_variance_score",
    "max_error",
    "mean_squared_log_error",
    "mean_poisson_deviance",
    "mean_gamma_deviance",
]

BINARY_CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "average_precision",
    "log_loss",
    "matthews_corrcoef",
    "balanced_accuracy",
    "specificity",
    "sensitivity",
    "npv",
    "ppv",
    "mcc",
    "cohen_kappa",
]

# Statistical test mappings for model types
CLASSIFICATION_STATISTICAL_TESTS = [
    StatisticalTest.MCNEMAR,
    StatisticalTest.DELONG,
    StatisticalTest.FIVE_TWO_CV,
    StatisticalTest.BOOTSTRAP,
    StatisticalTest.PAIRED_TTEST,
]

REGRESSION_STATISTICAL_TESTS = [
    StatisticalTest.PAIRED_TTEST,
    StatisticalTest.DIEBOLD_MARIANO,
    StatisticalTest.FIVE_TWO_CV,
    StatisticalTest.BOOTSTRAP,
    StatisticalTest.WILCOXON,
]

# Critical metrics that require special attention
CRITICAL_CLASSIFICATION_METRICS = ["accuracy", "f1_score", "roc_auc"]
CRITICAL_REGRESSION_METRICS = ["mse", "mae", "r2"]


# Utility functions for configuration management
def get_default_config(model_type: Union[str, ModelType]) -> AnalysisConfiguration:
    """Get default configuration for a model type."""
    if isinstance(model_type, str):
        model_type = ModelType(model_type)

    if model_type == ModelType.CLASSIFICATION:
        return DEFAULT_CLASSIFICATION_CONFIG.copy()
    elif model_type == ModelType.BINARY_CLASSIFICATION:
        return DEFAULT_BINARY_CLASSIFICATION_CONFIG.copy()
    elif model_type == ModelType.REGRESSION:
        return DEFAULT_REGRESSION_CONFIG.copy()
    else:
        return DEFAULT_CLASSIFICATION_CONFIG.copy()


def get_available_metrics(model_type: Union[str, ModelType]) -> List[str]:
    """Get available metrics for a model type."""
    if isinstance(model_type, str):
        model_type = ModelType(model_type)

    if model_type == ModelType.CLASSIFICATION:
        return CLASSIFICATION_METRICS.copy()
    elif model_type == ModelType.BINARY_CLASSIFICATION:
        return BINARY_CLASSIFICATION_METRICS.copy()
    elif model_type == ModelType.REGRESSION:
        return REGRESSION_METRICS.copy()
    else:
        return CLASSIFICATION_METRICS.copy()


def get_available_statistical_tests(model_type: Union[str, ModelType]) -> List[StatisticalTest]:
    """Get available statistical tests for a model type."""
    if isinstance(model_type, str):
        model_type = ModelType(model_type)

    if model_type in [ModelType.CLASSIFICATION, ModelType.BINARY_CLASSIFICATION]:
        return CLASSIFICATION_STATISTICAL_TESTS.copy()
    elif model_type == ModelType.REGRESSION:
        return REGRESSION_STATISTICAL_TESTS.copy()
    else:
        return CLASSIFICATION_STATISTICAL_TESTS.copy()


def validate_metrics_for_model_type(metrics: List[str], model_type: Union[str, ModelType]) -> Dict[str, Any]:
    """Validate that metrics are compatible with model type."""
    available_metrics = get_available_metrics(model_type)
    invalid_metrics = [m for m in metrics if m not in available_metrics]

    return {
        "valid": len(invalid_metrics) == 0,
        "invalid_metrics": invalid_metrics,
        "available_metrics": available_metrics,
        "suggested_metrics": available_metrics[:5],  # Top 5 recommended metrics
    }
