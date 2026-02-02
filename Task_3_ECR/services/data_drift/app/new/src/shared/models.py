"""
Pydantic models for S3-based stateless data loading
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class LoadDataRequest(BaseModel):
    """Request model for loading data and models from S3"""

    reference_url: str = Field(..., description="S3 URL of the reference/baseline dataset CSV")
    current_url: str = Field(..., description="S3 URL of the current dataset CSV")
    model_url: Optional[str] = Field(None, description="Optional S3 URL of the trained model file (pickle/joblib)")
    target_column: Optional[str] = Field(None, description="Optional target column name for classification analysis")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration dictionary for analysis")

    @validator("reference_url", "current_url")
    def validate_urls(cls, v):
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()

    @validator("model_url")
    def validate_model_url(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Model URL cannot be empty string")
        return v.strip() if v else None

    @validator("target_column")
    def validate_target_column(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Target column cannot be empty string")
        return v.strip() if v else None


class LoadDataResponse(BaseModel):
    """Response model for successful data loading"""

    status: str = "success"
    message: str
    reference_dataset: Dict[str, Any]
    current_dataset: Dict[str, Any]
    model_loaded: bool = False
    model_info: Optional[Dict[str, Any]] = None
    target_column: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    common_columns: List[str]
    validation_status: str = "passed"


class S3FileMetadata(BaseModel):
    """Model for S3 file metadata"""

    files: List[Dict[str, Any]]
    models: List[Dict[str, Any]]


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoints (replaces session_id)"""

    reference_url: str = Field(..., description="S3 URL of the reference dataset")
    current_url: str = Field(..., description="S3 URL of the current dataset")
    model_url: Optional[str] = Field(None, description="Optional S3 URL of the model")
    target_column: Optional[str] = Field(None, description="Optional target column name")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration")

    @validator("reference_url", "current_url")
    def validate_urls(cls, v):
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()


class ModelDriftAnalysisRequest(BaseModel):
    """Request model for model drift analysis endpoints - supports both single and multi-cohort analysis"""

    reference_url: str = Field(..., description="S3 URL of the reference dataset CSV")
    cohort_urls: List[str] = Field(
        ...,
        description="List of S3 URLs for cohort datasets. Single URL = normal drift analysis, Multiple URLs = multi-cohort time-series analysis",
    )
    cohort_labels: Optional[List[str]] = Field(
        None, description="Optional labels for cohorts (e.g., ['Week 1', 'Week 2']). Auto-generated if not provided."
    )
    model_url: str = Field(..., description="S3 URL of the trained model file (required for model drift)")
    target_column: Optional[str] = Field(
        None, description="Target column name for predictions (auto-detected from model if not provided)"
    )

    @validator("reference_url", "model_url")
    def validate_required_urls(cls, v):
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()

    @validator("cohort_urls")
    def validate_cohort_urls(cls, v):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("cohort_urls must be a non-empty list with at least one URL")

        # Validate each URL
        cleaned_urls = []
        for url in v:
            if not url or not url.strip():
                raise ValueError("Cohort URLs cannot contain empty strings")
            cleaned_urls.append(url.strip())

        return cleaned_urls

    @validator("cohort_labels")
    def validate_cohort_labels(cls, v, values):
        if v is not None:
            cohort_urls = values.get("cohort_urls")
            if cohort_urls and len(v) != len(cohort_urls):
                raise ValueError(f"cohort_labels length ({len(v)}) must match cohort_urls length ({len(cohort_urls)})")
        return v


class AIExplanationRequest(BaseModel):
    """Request model for AI explanation endpoint"""

    reference_url: str = Field(..., description="S3 URL of the reference dataset CSV")
    current_url: str = Field(..., description="S3 URL of the current dataset CSV")
    model_url: Optional[str] = Field(
        None,
        description="S3 URL of the trained model file (required for model drift analysis, optional for data drift analysis)",
    )
    analysis_type: str = Field(
        ...,
        description="""Type of analysis to perform:
    Model Drift: model_performance, degradation_metrics, statistical_significance, sanity_check
    Data Drift: class_imbalance, statistical_analysis, feature_analysis, data_drift_dashboard""",
    )
    target_column: Optional[str] = Field(None, description="Target column name for predictions")
    analysis_config: Optional[Dict[str, Any]] = Field(
        None, description="Analysis configuration (thresholds, metrics, etc.)"
    )

    @validator("reference_url", "current_url", "model_url")
    def validate_urls(cls, v):
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()


# ===== Multi-Cohort Models =====


class FileInfo(BaseModel):
    """Information about a single file in a multi-cohort upload"""

    file_name: str = Field(..., description="Name of the CSV file")
    folder: str = Field(..., description="Folder/directory path")
    url: str = Field(..., description="S3 presigned URL for the file")

    @validator("url")
    def validate_url(cls, v):
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()

    @validator("file_name")
    def validate_file_name(cls, v):
        if not v or not v.strip():
            raise ValueError("File name cannot be empty")
        if not v.lower().endswith(".csv"):
            raise ValueError("File must be a CSV file")
        return v.strip()


class MultiCohortAnalysisRequest(BaseModel):
    """Request model for multi-cohort data drift analysis"""

    files: List[FileInfo] = Field(
        ..., description="List of files (1 baseline + up to 4 cohorts)", min_items=2, max_items=5
    )
    target_column: Optional[str] = Field(None, description="Optional target column name")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional analysis configuration")
    selected_features: Optional[List[str]] = Field(None, description="Optional list of features to analyze")
    selected_tests: Optional[List[str]] = Field(
        None, description="Optional list of tests to run (KS, PSI, KL, JS, Wasserstein)"
    )
    cohort_pairs: Optional[List[str]] = Field(
        None, description="Optional list of cohort pairs to compare (e.g., 'baseline-cohort_1')"
    )

    @validator("files")
    def validate_files(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 files required (1 baseline + 1 cohort)")
        if len(v) > 5:
            raise ValueError("Maximum 5 files allowed (1 baseline + 4 cohorts)")
        return v

    @validator("selected_tests")
    def validate_tests(cls, v):
        if v is not None:
            valid_tests = {"KS", "PSI", "KL", "JS", "Wasserstein", "Chi-square"}
            invalid_tests = set(v) - valid_tests
            if invalid_tests:
                raise ValueError(f"Invalid tests: {invalid_tests}. Valid tests: {valid_tests}")
        return v


class CohortMetadata(BaseModel):
    """Metadata about a single cohort"""

    cohort_name: str = Field(..., description="Name of the cohort (baseline, cohort_1, etc.)")
    cohort_type: str = Field(..., description="Type of cohort: baseline or current")
    file_name: str = Field(..., description="Original file name")
    url: str = Field(..., description="S3 URL")
    shape: List[int] = Field(..., description="Shape of dataframe [rows, columns]")
    total_features: int = Field(..., description="Total number of features")
    total_rows: int = Field(..., description="Total number of rows")
    duplicate_rows: int = Field(0, description="Number of duplicate rows")
    features: List[Dict[str, Any]] = Field(default_factory=list, description="List of feature metadata")

    class Config:
        schema_extra = {
            "example": {
                "cohort_name": "baseline",
                "cohort_type": "baseline",
                "file_name": "ref_cohort_1.csv",
                "url": "https://s3.amazonaws.com/...",
                "shape": [1000, 20],
                "total_features": 20,
                "total_rows": 1000,
                "duplicate_rows": 5,
                "features": [
                    {"feature_name": "Age", "data_type": "numerical", "missing_percent": 2.5, "unique_count": 65}
                ],
            }
        }


class CohortComparisonRequest(BaseModel):
    """Request for cohort comparison endpoint"""

    files: List[FileInfo] = Field(..., description="All uploaded files")
    selected_cohorts: List[str] = Field(
        ..., description="List of cohort names to compare (up to 3)", min_items=2, max_items=3
    )
    feature: str = Field(..., description="Single feature name to analyze")

    @validator("selected_cohorts")
    def validate_cohort_count(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 cohorts required for comparison")
        if len(v) > 3:
            raise ValueError("Maximum 3 cohorts allowed for comparison")
        return v


class AIExplanationRequest(BaseModel):
    """Request model for AI explanation endpoint"""

    reference_url: str = Field(..., description="S3 URL of the reference dataset CSV")
    current_url: str = Field(..., description="S3 URL of the current dataset CSV")
    model_url: str = Field(..., description="S3 URL of the trained model file (required for model drift analysis)")
    analysis_type: str = Field(
        ...,
        description="Type of analysis (model_performance, degradation_metrics, statistical_significance, sanity_check)",
    )
    target_column: Optional[str] = Field(None, description="Target column name for predictions")
    analysis_config: Optional[Dict[str, Any]] = Field(
        None, description="Analysis configuration (thresholds, metrics, etc.)"
    )

    @validator("reference_url", "current_url")
    def validate_required_urls(cls, v):
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()

    @validator("model_url")
    def validate_model_url(cls, v):
        if v is not None and v.strip():
            return v.strip()
        return None

    @validator("analysis_type")
    def validate_analysis_type(cls, v):
        valid_types = {
            # Model Drift Analysis Types (require model)
            "model_performance",
            "degradation_metrics",
            "statistical_significance",
            "sanity_check",
            # Data Drift Analysis Types (model optional)
            "class_imbalance",
            "statistical_analysis",
            "feature_analysis",
            "data_drift_dashboard",
        }
        if v not in valid_types:
            raise ValueError(f"analysis_type must be one of: {', '.join(sorted(valid_types))}")
        return v
