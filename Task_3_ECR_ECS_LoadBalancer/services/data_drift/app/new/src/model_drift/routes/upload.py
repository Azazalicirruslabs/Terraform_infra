import io
import json
import math
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ValidationError

from shared.auth import get_current_user

from ...shared.ai_explanation_service import ai_explanation_service
from ...shared.session_manager import get_session_manager
from ..models.analysis_config import AnalysisConfiguration, DriftThresholds, ModelType
from ..services.enhanced_model_service import enhanced_model_service
from ..services.model_service import run_model_drift


def create_ai_summary_for_performance_comparison(analysis_data: dict) -> dict:
    """
    Summarizes performance comparison analysis for AI explanation.
    """
    summary = {
        "model_type": analysis_data.get("model_info", {}).get("model_type"),
        "analysis_name": analysis_data.get("analysis_info", {}).get("analysis_name"),
        "reference_performance": analysis_data.get("reference_performance", {}),
        "current_performance": analysis_data.get("current_performance", {}),
        "statistical_test_results": analysis_data.get("statistical_tests", {}),
        "overall_drift_status": analysis_data.get("drift_assessment", {}).get("overall_status"),
        "recommendations": analysis_data.get("recommendations", []),
    }

    # Add key metrics comparison (avoid full metric arrays)
    ref_perf = analysis_data.get("reference_performance", {})
    curr_perf = analysis_data.get("current_performance", {})

    key_metrics = []
    for metric_name in ["accuracy", "precision", "recall", "f1_score", "mse", "rmse", "r2"]:
        if metric_name in ref_perf and metric_name in curr_perf:
            key_metrics.append(
                {
                    "metric": metric_name,
                    "reference_value": round(ref_perf[metric_name], 4),
                    "current_value": round(curr_perf[metric_name], 4),
                    "change": round(curr_perf[metric_name] - ref_perf[metric_name], 4),
                }
            )

    summary["key_metrics_comparison"] = key_metrics[:5]  # Limit to top 5 metrics
    return summary


def create_ai_summary_for_degradation_metrics(analysis_data: dict) -> dict:
    """
    Summarizes degradation metrics analysis for AI explanation.
    """
    summary = {
        "model_type": analysis_data.get("model_info", {}).get("model_type"),
        "analysis_name": analysis_data.get("analysis_info", {}).get("analysis_name"),
        "disagreement_analysis": analysis_data.get("disagreement_analysis", {}),
        "confidence_analysis": analysis_data.get("confidence_analysis", {}),
        "feature_importance_drift": analysis_data.get("feature_importance_drift", {}),
        "overall_assessment": analysis_data.get("overall_assessment", {}),
        "recommendations": analysis_data.get("recommendations", []),
    }

    # Extract key degradation indicators
    disagreement = analysis_data.get("disagreement_analysis", {})
    if disagreement:
        summary["disagreement_rate"] = disagreement.get("disagreement_rate")
        summary["prediction_stability"] = disagreement.get("stability_score")

    confidence = analysis_data.get("confidence_analysis", {})
    if confidence:
        summary["confidence_decline"] = confidence.get("confidence_decline")
        summary["low_confidence_predictions"] = confidence.get("low_confidence_count")

    return summary


def create_ai_summary_for_statistical_significance(analysis_data: dict) -> dict:
    """
    Summarizes statistical significance analysis for AI explanation.
    """
    summary = {
        "model_type": analysis_data.get("model_info", {}).get("model_type"),
        "analysis_name": analysis_data.get("analysis_info", {}).get("analysis_name"),
        "hypothesis_test_results": analysis_data.get("hypothesis_tests", {}),
        "effect_size_analysis": analysis_data.get("effect_sizes", {}),
        "power_analysis": analysis_data.get("power_analysis", {}),
        "multiple_comparisons": analysis_data.get("multiple_comparisons", {}),
        "overall_significance": analysis_data.get("overall_assessment", {}),
        "recommendations": analysis_data.get("recommendations", []),
    }

    # Extract key significance indicators
    hypothesis_tests = analysis_data.get("hypothesis_tests", {})
    if hypothesis_tests:
        summary["significant_tests"] = len([test for test in hypothesis_tests.values() if test.get("is_significant")])
        summary["total_tests"] = len(hypothesis_tests)

    effect_sizes = analysis_data.get("effect_sizes", {})
    if effect_sizes:
        summary["large_effect_sizes"] = len(
            [effect for effect in effect_sizes.values() if effect.get("magnitude") == "large"]
        )

    return summary


# Session-to-UploadFile Adapter Classes
class SessionUploadFile:
    """Adapter class to convert session data back to UploadFile-like object"""

    def __init__(self, content: bytes, filename: str, content_type: str = "text/csv"):
        self.content = content
        self.filename = filename
        self.content_type = content_type
        self._bytes_io = io.BytesIO(content)

    async def read(self) -> bytes:
        self._bytes_io.seek(0)
        return self._bytes_io.read()

    def seek(self, offset: int, whence: int = 0):
        return self._bytes_io.seek(offset, whence)

    def tell(self) -> int:
        return self._bytes_io.tell()


def create_upload_files_from_session(session_id: str):
    """Convert session data back to UploadFile-like objects for existing services"""
    session_manager = get_session_manager()
    if not session_manager.session_exists(session_id):
        return None, None, None

    session_data = session_manager.get_model_drift_format(session_id)
    if not session_data:
        return None, None, None

    # Convert DataFrames back to CSV bytes
    reference_csv = session_data["reference_df"].to_csv(index=False).encode("utf-8")
    current_csv = session_data["current_df"].to_csv(index=False).encode("utf-8")

    # Create UploadFile-like objects
    reference_file = SessionUploadFile(
        content=reference_csv, filename=session_data["reference_filename"], content_type="text/csv"
    )

    current_file = SessionUploadFile(
        content=current_csv, filename=session_data["current_filename"], content_type="text/csv"
    )

    model_file = None
    if session_data["model_file_content"]:
        model_file = SessionUploadFile(
            content=session_data["model_file_content"],
            filename=session_data["model_filename"],
            content_type="application/octet-stream",
        )

    return reference_file, current_file, model_file, session_data.get("config")


def clean_float_values(obj):
    """Recursively clean non-finite float values from nested data structures"""
    if isinstance(obj, dict):
        return {k: clean_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_float_values(item) for item in obj]
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None  # or 0, or "N/A" depending on your preference
        return float(obj)
    elif isinstance(obj, np.generic):
        # Handle other numpy types
        item = obj.item()
        return clean_float_values(item)
    else:
        return obj


def serialize_response(result):
    """Serialize response with proper handling of numpy types and non-finite values"""
    cleaned_result = clean_float_values(result)
    return jsonable_encoder(cleaned_result, custom_encoder={np.generic: lambda x: x.item()})


class AnalysisConfig(BaseModel):
    analysis_name: str
    description: str = ""
    model_type: str
    selected_metrics: List[str]
    statistical_test: str
    low_threshold: float = 0.05
    medium_threshold: float = 0.15
    high_threshold: float = 0.25


router = APIRouter(prefix="/data_drift/v1/model-drift", tags=["model-drift"])


@router.post("/upload")
async def model_drift_upload(
    reference_data: UploadFile = File(...),
    current_data: UploadFile = File(...),
    model_file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Legacy model drift upload endpoint (backward compatibility)
    Requires both datasets and a model file
    """
    result = await run_model_drift(reference_data, current_data, model_file)
    return result


@router.post("/analyze")
async def model_drift_analyze(
    reference_data: UploadFile = File(..., description="Reference dataset CSV file"),
    current_data: UploadFile = File(..., description="Current dataset CSV file"),
    model_file: UploadFile = File(..., description="Trained model file (.pkl, .pickle, .joblib)"),
    config: str = Form(..., description="JSON configuration string"),
    user: dict = Depends(get_current_user),
):
    """
    Enhanced model drift analysis endpoint with JSON configuration

    For the config parameter, send a JSON string like:
    {"analysis_name": "My Analysis", "model_type": "classification", "selected_metrics": ["accuracy", "precision"], "statistical_test": "mcnemar"}
    """
    try:
        # Parse config from JSON string
        try:
            config_dict = json.loads(config)
            parsed_config = AnalysisConfig(**config_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {str(e)}")

        # Validate file types
        if not reference_data.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Reference data must be a CSV file")
        if not current_data.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Current data must be a CSV file")
        if not (model_file.filename.lower().endswith((".pkl", ".pickle", ".joblib", ".pkl.joblib"))):
            raise HTTPException(
                status_code=400, detail="Model file must be a pickle (.pkl, .pickle) or joblib (.joblib) file"
            )

        # Validate model type
        try:
            model_type_enum = ModelType(parsed_config.model_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type: {parsed_config.model_type}. Must be 'classification' or 'regression'",
            )

        # Create full configuration object
        try:
            full_config = AnalysisConfiguration(
                analysis_name=parsed_config.analysis_name,
                description=parsed_config.description,
                model_type=model_type_enum,
                selected_metrics=parsed_config.selected_metrics,
                statistical_test=parsed_config.statistical_test,
                drift_thresholds=DriftThresholds(
                    low_threshold=parsed_config.low_threshold,
                    medium_threshold=parsed_config.medium_threshold,
                    high_threshold=parsed_config.high_threshold,
                ),
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Configuration validation error: {str(e)}")

        # Run enhanced analysis
        result = await enhanced_model_service.run_configured_analysis(
            reference_data, current_data, model_file, full_config
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/metrics")
async def get_available_metrics(user: dict = Depends(get_current_user)):
    """Get available performance metrics for each model type"""
    return {
        "classification": [
            {"id": "accuracy", "label": "Accuracy", "description": "Overall prediction accuracy"},
            {
                "id": "precision",
                "label": "Precision",
                "description": "True positives / (True positives + False positives)",
            },
            {
                "id": "recall",
                "label": "Recall (Sensitivity)",
                "description": "True positives / (True positives + False negatives)",
            },
            {"id": "f1_score", "label": "F1-Score", "description": "Harmonic mean of precision and recall"},
            {
                "id": "specificity",
                "label": "Specificity",
                "description": "True negatives / (True negatives + False positives)",
            },
            {"id": "roc_auc", "label": "ROC AUC", "description": "Area under ROC curve"},
            {"id": "pr_auc", "label": "PR AUC", "description": "Area under precision-recall curve"},
            {"id": "cohen_kappa", "label": "Cohen's Kappa", "description": "Inter-rater reliability metric"},
            {
                "id": "mcc",
                "label": "Matthews Correlation Coefficient",
                "description": "Correlation between predictions and actual",
            },
        ],
        "regression": [
            {"id": "mse", "label": "Mean Squared Error (MSE)", "description": "Average squared prediction errors"},
            {"id": "rmse", "label": "Root Mean Squared Error (RMSE)", "description": "Square root of MSE"},
            {"id": "mae", "label": "Mean Absolute Error (MAE)", "description": "Average absolute prediction errors"},
            {"id": "r2", "label": "R-squared (R²)", "description": "Coefficient of determination"},
            {"id": "adjusted_r2", "label": "Adjusted R-squared", "description": "R² adjusted for number of predictors"},
            {
                "id": "mape",
                "label": "Mean Absolute Percentage Error (MAPE)",
                "description": "Average absolute percentage errors",
            },
            {
                "id": "explained_variance",
                "label": "Explained Variance Score",
                "description": "Proportion of variance explained",
            },
            {"id": "max_error", "label": "Max Error", "description": "Maximum residual error"},
        ],
    }


@router.get("/tests")
async def get_available_tests(user: dict = Depends(get_current_user)):
    """Get available statistical tests for each model type"""
    return {
        "classification": [
            {
                "id": "mcnemar",
                "label": "McNemar's Test",
                "description": "Compares paired categorical data for classification models",
                "complexity": "Simple",
                "category": "Non-parametric",
            },
            {
                "id": "delong",
                "label": "DeLong Test",
                "description": "Compares ROC curves for statistical significance",
                "complexity": "Moderate",
                "category": "ROC-based",
            },
            {
                "id": "five_two_cv",
                "label": "5×2 Cross-Validation F-Test",
                "description": "Robust cross-validation based comparison",
                "complexity": "Complex",
                "category": "Cross-validation",
            },
            {
                "id": "bootstrap_confidence",
                "label": "Bootstrap Confidence Intervals",
                "description": "Non-parametric confidence interval estimation",
                "complexity": "Moderate",
                "category": "Resampling",
            },
            {
                "id": "paired_ttest",
                "label": "Paired t-Test",
                "description": "Classical statistical test for paired samples",
                "complexity": "Simple",
                "category": "Parametric",
            },
        ],
        "regression": [
            {
                "id": "five_two_cv",
                "label": "5×2 Cross-Validation F-Test",
                "description": "Robust cross-validation based comparison",
                "complexity": "Complex",
                "category": "Cross-validation",
            },
            {
                "id": "bootstrap_confidence",
                "label": "Bootstrap Confidence Intervals",
                "description": "Non-parametric confidence interval estimation",
                "complexity": "Moderate",
                "category": "Resampling",
            },
            {
                "id": "diebold_mariano",
                "label": "Diebold-Mariano Test",
                "description": "Compares predictive accuracy of forecasting models",
                "complexity": "Moderate",
                "category": "Time series",
            },
            {
                "id": "paired_ttest",
                "label": "Paired t-Test",
                "description": "Classical statistical test for paired samples",
                "complexity": "Simple",
                "category": "Parametric",
            },
        ],
    }


@router.post("/performance-comparison")
async def performance_comparison_analysis(
    reference_data: UploadFile = File(..., description="Reference dataset CSV file"),
    current_data: UploadFile = File(..., description="Current dataset CSV file"),
    model_file: UploadFile = File(..., description="Trained model file (.pkl, .pickle, .joblib)"),
    config: str = Form(..., description="JSON configuration string"),
    user: dict = Depends(get_current_user),
):
    """
    Performance Comparison Analysis (Tab 1)

    Returns comprehensive performance comparison between reference and current model performance
    """
    try:
        # Parse config from JSON string
        try:
            config_dict = json.loads(config)
            parsed_config = AnalysisConfig(**config_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {str(e)}")

        # Validate file types
        if not reference_data.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Reference data must be a CSV file")
        if not current_data.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Current data must be a CSV file")
        if not (model_file.filename.lower().endswith((".pkl", ".pickle", ".joblib", ".pkl.joblib"))):
            raise HTTPException(
                status_code=400, detail="Model file must be a pickle (.pkl, .pickle) or joblib (.joblib) file"
            )

        # Validate model type
        try:
            model_type_enum = ModelType(parsed_config.model_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type: {parsed_config.model_type}. Must be 'classification' or 'regression'",
            )

        # Create full configuration object
        try:
            full_config = AnalysisConfiguration(
                analysis_name=parsed_config.analysis_name,
                description=parsed_config.description,
                model_type=model_type_enum,
                selected_metrics=parsed_config.selected_metrics,
                statistical_test=parsed_config.statistical_test,
                drift_thresholds=DriftThresholds(
                    low_threshold=parsed_config.low_threshold,
                    medium_threshold=parsed_config.medium_threshold,
                    high_threshold=parsed_config.high_threshold,
                ),
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Configuration validation error: {str(e)}")

        # Run performance comparison analysis only
        result = await enhanced_model_service.run_performance_comparison_analysis(
            reference_data, current_data, model_file, full_config
        )

        return serialize_response(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance comparison analysis failed: {str(e)}")


@router.post("/degradation-metrics")
async def degradation_metrics_analysis(
    reference_data: UploadFile = File(..., description="Reference dataset CSV file"),
    current_data: UploadFile = File(..., description="Current dataset CSV file"),
    model_file: UploadFile = File(..., description="Trained model file (.pkl, .pickle, .joblib)"),
    config: str = Form(..., description="JSON configuration string"),
    user: dict = Depends(get_current_user),
):
    """
    Degradation Metrics Analysis (Tab 2)

    Returns comprehensive degradation analysis with 3 sub-tabs:
    - Model Disagreement
    - Confidence Analysis
    - Feature Importance Drift
    """
    try:
        # Parse config from JSON string
        try:
            config_dict = json.loads(config)
            parsed_config = AnalysisConfig(**config_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {str(e)}")

        # Validate file types
        if not reference_data.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Reference data must be a CSV file")
        if not current_data.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Current data must be a CSV file")
        if not (model_file.filename.lower().endswith((".pkl", ".pickle", ".joblib", ".pkl.joblib"))):
            raise HTTPException(
                status_code=400, detail="Model file must be a pickle (.pkl, .pickle) or joblib (.joblib) file"
            )

        # Validate model type
        try:
            model_type_enum = ModelType(parsed_config.model_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type: {parsed_config.model_type}. Must be 'classification' or 'regression'",
            )

        # Create full configuration object
        try:
            full_config = AnalysisConfiguration(
                analysis_name=parsed_config.analysis_name,
                description=parsed_config.description,
                model_type=model_type_enum,
                selected_metrics=parsed_config.selected_metrics,
                statistical_test=parsed_config.statistical_test,
                drift_thresholds=DriftThresholds(
                    low_threshold=parsed_config.low_threshold,
                    medium_threshold=parsed_config.medium_threshold,
                    high_threshold=parsed_config.high_threshold,
                ),
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Configuration validation error: {str(e)}")

        # Run degradation metrics analysis only
        result = await enhanced_model_service.run_degradation_metrics_analysis(
            reference_data, current_data, model_file, full_config
        )

        return serialize_response(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Degradation metrics analysis failed: {str(e)}")


@router.post("/statistical-significance")
async def statistical_significance_analysis(
    reference_data: UploadFile = File(..., description="Reference dataset CSV file"),
    current_data: UploadFile = File(..., description="Current dataset CSV file"),
    model_file: UploadFile = File(..., description="Trained model file (.pkl, .pickle, .joblib)"),
    config: str = Form(..., description="JSON configuration string"),
    user: dict = Depends(get_current_user),
):
    """
    Statistical Significance Analysis (Tab 3)

    Returns comprehensive statistical significance testing with:
    - Hypothesis testing
    - Effect size analysis
    - Power analysis
    - Multiple comparisons correction
    """
    try:
        # Parse config from JSON string
        try:
            config_dict = json.loads(config)
            parsed_config = AnalysisConfig(**config_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {str(e)}")

        # Validate file types
        if not reference_data.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Reference data must be a CSV file")
        if not current_data.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Current data must be a CSV file")
        if not (model_file.filename.lower().endswith((".pkl", ".pickle", ".joblib", ".pkl.joblib"))):
            raise HTTPException(
                status_code=400, detail="Model file must be a pickle (.pkl, .pickle) or joblib (.joblib) file"
            )

        # Validate model type
        try:
            model_type_enum = ModelType(parsed_config.model_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type: {parsed_config.model_type}. Must be 'classification' or 'regression'",
            )

        # Create full configuration object
        try:
            full_config = AnalysisConfiguration(
                analysis_name=parsed_config.analysis_name,
                description=parsed_config.description,
                model_type=model_type_enum,
                selected_metrics=parsed_config.selected_metrics,
                statistical_test=parsed_config.statistical_test,
                drift_thresholds=DriftThresholds(
                    low_threshold=parsed_config.low_threshold,
                    medium_threshold=parsed_config.medium_threshold,
                    high_threshold=parsed_config.high_threshold,
                ),
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Configuration validation error: {str(e)}")

        # Run statistical significance analysis only
        result = await enhanced_model_service.run_statistical_significance_analysis(
            reference_data, current_data, model_file, full_config
        )

        return serialize_response(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistical significance analysis failed: {str(e)}")


# Clean Session-Based Model Drift Endpoints (Using Adapter Pattern)


@router.get("/session/performance-comparison/{session_id}")
async def session_performance_comparison(session_id: str, user: dict = Depends(get_current_user)):
    """Performance Comparison Analysis using session data with adapter pattern"""
    try:
        # Convert session data to UploadFile-like objects
        reference_file, current_file, model_file, config = create_upload_files_from_session(session_id)

        if not reference_file or not current_file or not model_file:
            raise HTTPException(status_code=404, detail="Session not found or does not contain required files")

        # Use configuration from session or default
        if config:
            # Handle statistical_test as either string or list (take first item if list)
            statistical_test = config.get("statistical_test", "mcnemar")
            if isinstance(statistical_test, list):
                statistical_test = statistical_test[0] if statistical_test else "mcnemar"

            parsed_config = AnalysisConfig(
                analysis_name=config.get("analysis_name", "Performance Comparison"),
                description=config.get("description", "Session-based performance comparison analysis"),
                model_type=config.get("model_type", "classification"),
                selected_metrics=config.get("selected_metrics", ["accuracy", "precision", "recall"]),
                statistical_test=statistical_test,
                low_threshold=config.get("low_threshold", 0.05),
                medium_threshold=config.get("medium_threshold", 0.15),
                high_threshold=config.get("high_threshold", 0.25),
            )
        else:
            parsed_config = AnalysisConfig(
                analysis_name="Performance Comparison",
                description="Session-based performance comparison analysis",
                model_type="classification",
                selected_metrics=["accuracy", "precision", "recall"],
                statistical_test="mcnemar",
                low_threshold=0.05,
                medium_threshold=0.15,
                high_threshold=0.25,
            )

        # Create full configuration for enhanced service
        model_type_enum = ModelType(parsed_config.model_type.lower())
        full_config = AnalysisConfiguration(
            analysis_name=parsed_config.analysis_name,
            description=parsed_config.description,
            model_type=model_type_enum,
            selected_metrics=parsed_config.selected_metrics,
            statistical_test=parsed_config.statistical_test,
            drift_thresholds=DriftThresholds(
                low_threshold=parsed_config.low_threshold,
                medium_threshold=parsed_config.medium_threshold,
                high_threshold=parsed_config.high_threshold,
            ),
        )

        # Use existing enhanced model service (preserves all your complex logic)
        result = await enhanced_model_service.run_performance_comparison_analysis(
            reference_file, current_file, model_file, full_config
        )

        serialized_result = serialize_response(result)

        # Generate AI explanation for the performance comparison analysis
        try:
            ai_summary_payload = create_ai_summary_for_performance_comparison(serialized_result)
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, analysis_type="model_performance"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            # Continue without AI explanation
            serialized_result["llm_response"] = {
                "summary": "Model performance comparison analysis completed successfully.",
                "detailed_explanation": "Performance comparison between reference and current model predictions has been completed. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Performance comparison analysis completed successfully",
                    "Review performance metrics for degradation patterns",
                    "AI explanations will return when service is restored",
                ],
            }

        return serialized_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance comparison analysis failed: {str(e)}")


@router.get("/session/degradation-metrics/{session_id}")
async def session_degradation_metrics(session_id: str, user: dict = Depends(get_current_user)):
    """Degradation Metrics Analysis using session data with adapter pattern"""
    try:
        # Convert session data to UploadFile-like objects
        reference_file, current_file, model_file, config = create_upload_files_from_session(session_id)

        if not reference_file or not current_file or not model_file:
            raise HTTPException(status_code=404, detail="Session not found or does not contain required files")

        # Use configuration from session or default
        if config:
            # Handle statistical_test as either string or list (take first item if list)
            statistical_test = config.get("statistical_test", "mcnemar")
            if isinstance(statistical_test, list):
                statistical_test = statistical_test[0] if statistical_test else "mcnemar"

            parsed_config = AnalysisConfig(
                analysis_name=config.get("analysis_name", "Degradation Metrics"),
                description=config.get("description", "Session-based degradation metrics analysis"),
                model_type=config.get("model_type", "classification"),
                selected_metrics=config.get("selected_metrics", ["accuracy", "precision", "recall"]),
                statistical_test=statistical_test,
                low_threshold=config.get("low_threshold", 0.05),
                medium_threshold=config.get("medium_threshold", 0.15),
                high_threshold=config.get("high_threshold", 0.25),
            )
        else:
            parsed_config = AnalysisConfig(
                analysis_name="Degradation Metrics",
                description="Session-based degradation metrics analysis",
                model_type="classification",
                selected_metrics=["accuracy", "precision", "recall"],
                statistical_test="mcnemar",
                low_threshold=0.05,
                medium_threshold=0.15,
                high_threshold=0.25,
            )

        # Create full configuration for enhanced service
        model_type_enum = ModelType(parsed_config.model_type.lower())
        full_config = AnalysisConfiguration(
            analysis_name=parsed_config.analysis_name,
            description=parsed_config.description,
            model_type=model_type_enum,
            selected_metrics=parsed_config.selected_metrics,
            statistical_test=parsed_config.statistical_test,
            drift_thresholds=DriftThresholds(
                low_threshold=parsed_config.low_threshold,
                medium_threshold=parsed_config.medium_threshold,
                high_threshold=parsed_config.high_threshold,
            ),
        )

        # Use existing enhanced model service (preserves all your complex logic)
        result = await enhanced_model_service.run_degradation_metrics_analysis(
            reference_file, current_file, model_file, full_config
        )

        serialized_result = serialize_response(result)

        # Generate AI explanation for the degradation metrics analysis
        try:
            ai_summary_payload = create_ai_summary_for_degradation_metrics(serialized_result)
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, analysis_type="degradation_metrics"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            # Continue without AI explanation
            serialized_result["llm_response"] = {
                "summary": "Model degradation metrics analysis completed successfully.",
                "detailed_explanation": "Comprehensive degradation analysis including model disagreement and confidence patterns has been completed. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Degradation metrics analysis completed successfully",
                    "Review model disagreement and confidence trends",
                    "AI explanations will return when service is restored",
                ],
            }

        return serialized_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Degradation metrics analysis failed: {str(e)}")


@router.get("/session/statistical-significance/{session_id}")
async def session_statistical_significance(session_id: str, user: dict = Depends(get_current_user)):
    """Statistical Significance Analysis using session data with adapter pattern"""
    try:
        # Convert session data to UploadFile-like objects
        reference_file, current_file, model_file, config = create_upload_files_from_session(session_id)

        if not reference_file or not current_file or not model_file:
            raise HTTPException(status_code=404, detail="Session not found or does not contain required files")

        # Use configuration from session or default
        if config:
            # Handle statistical_test as either string or list (take first item if list)
            statistical_test = config.get("statistical_test", "mcnemar")
            if isinstance(statistical_test, list):
                statistical_test = statistical_test[0] if statistical_test else "mcnemar"

            parsed_config = AnalysisConfig(
                analysis_name=config.get("analysis_name", "Statistical Significance"),
                description=config.get("description", "Session-based statistical significance analysis"),
                model_type=config.get("model_type", "classification"),
                selected_metrics=config.get("selected_metrics", ["accuracy", "precision", "recall"]),
                statistical_test=statistical_test,
                low_threshold=config.get("low_threshold", 0.05),
                medium_threshold=config.get("medium_threshold", 0.15),
                high_threshold=config.get("high_threshold", 0.25),
            )
        else:
            parsed_config = AnalysisConfig(
                analysis_name="Statistical Significance",
                description="Session-based statistical significance analysis",
                model_type="classification",
                selected_metrics=["accuracy", "precision", "recall"],
                statistical_test="mcnemar",
                low_threshold=0.05,
                medium_threshold=0.15,
                high_threshold=0.25,
            )

        # Create full configuration for enhanced service
        model_type_enum = ModelType(parsed_config.model_type.lower())
        full_config = AnalysisConfiguration(
            analysis_name=parsed_config.analysis_name,
            description=parsed_config.description,
            model_type=model_type_enum,
            selected_metrics=parsed_config.selected_metrics,
            statistical_test=parsed_config.statistical_test,
            drift_thresholds=DriftThresholds(
                low_threshold=parsed_config.low_threshold,
                medium_threshold=parsed_config.medium_threshold,
                high_threshold=parsed_config.high_threshold,
            ),
        )

        # Use existing enhanced model service (preserves all your complex logic)
        result = await enhanced_model_service.run_statistical_significance_analysis(
            reference_file, current_file, model_file, full_config
        )

        serialized_result = serialize_response(result)

        # Generate AI explanation for the statistical significance analysis
        try:
            ai_summary_payload = create_ai_summary_for_statistical_significance(serialized_result)
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, analysis_type="statistical_significance"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            # Continue without AI explanation
            serialized_result["llm_response"] = {
                "summary": "Statistical significance analysis completed successfully.",
                "detailed_explanation": "Statistical testing of model performance changes has been completed with hypothesis testing and effect size analysis. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Statistical significance analysis completed successfully",
                    "Review statistical test results and confidence intervals",
                    "AI explanations will return when service is restored",
                ],
            }

        return serialized_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistical significance analysis failed: {str(e)}")


def create_ai_summary_for_sanity_check(analysis_data: dict) -> dict:
    """
    Summarizes sanity check analysis for AI explanation.
    """
    summary = {
        "analysis_type": "sanity_check",
        "overall_similarity": analysis_data.get("sanity_check_summary", {}).get("overall_similarity_score"),
        "severity": analysis_data.get("sanity_check_summary", {}).get("severity"),
        "alert_level": analysis_data.get("sanity_check_summary", {}).get("alert_level"),
        "expected_performance": {
            "metric": analysis_data.get("sanity_check_summary", {}).get("expected_metric"),
            "predicted_value": analysis_data.get("sanity_check_summary", {}).get("predicted_value"),
            "confidence": analysis_data.get("sanity_check_summary", {}).get("confidence"),
        },
        "drift_info": {
            "num_features_analyzed": analysis_data.get("sanity_check_summary", {}).get("num_features_analyzed"),
            "num_drifted_features": analysis_data.get("sanity_check_summary", {}).get("num_drifted_features"),
            "drift_type": analysis_data.get("sanity_check_summary", {}).get("drift_type"),
        },
        "top_drifted_features": analysis_data.get("scatter_points", [])[:5],  # Top 5
        "recommendations": analysis_data.get("recommendations", []),
    }
    return summary


@router.post("/sanity-check")
async def sanity_check_analysis(
    reference_data: UploadFile = File(..., description="Reference dataset CSV file"),
    current_data: UploadFile = File(..., description="Current dataset CSV file"),
    model_file: Optional[UploadFile] = File(None, description="Trained model file (.pkl or .joblib) - Optional"),
    config: str = Form(None, description="JSON configuration for analysis (optional)"),
    user: dict = Depends(get_current_user),
):
    """
    **Sanity Check & Short-Term Performance Estimation**

    Pre-flight diagnostic analysis that estimates expected model performance on new data
    by analyzing distributional and feature similarity between reference and current datasets.

    **Features:**
    - Feature-level similarity analysis (KS test for numerical, Chi-square for categorical)
    - Expected performance drop estimation
    - Drift type inference (data drift vs concept drift)
    - Severity classification (🟢 Stable / 🟡 Monitor / 🔴 Warning)
    - Actionable recommendations

    **Returns:**
    - Scatter plot data (similarity vs expected drop per feature)
    - Distribution comparison for top drifted features
    - Overall assessment with alert level
    - AI-generated business-friendly explanation
    """
    try:
        # Parse config if provided
        parsed_config = None
        if config:
            try:
                config_dict = json.loads(config)

                # Create a minimal config object
                class SanityCheckConfig(BaseModel):
                    model_type: str = "classification"
                    target_column: Optional[str] = None

                parsed_config = SanityCheckConfig(**config_dict)
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid config JSON: {str(e)}")

        # Validate file types
        if not reference_data.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Reference data must be a CSV file")
        if not current_data.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Current data must be a CSV file")
        if model_file and not model_file.filename.lower().endswith((".pkl", ".pickle", ".joblib")):
            raise HTTPException(status_code=400, detail="Model file must be .pkl, .pickle, or .joblib format")

        # Build config for enhanced service
        full_config = None
        if parsed_config:
            model_type_enum = ModelType(parsed_config.model_type.lower())
            full_config = AnalysisConfiguration(
                analysis_name="Sanity Check",
                description="Pre-flight diagnostic analysis",
                model_type=model_type_enum,
                selected_metrics=["accuracy"] if parsed_config.model_type == "classification" else ["r2"],
                statistical_test="ks_test",
            )
            if hasattr(parsed_config, "target_column"):
                full_config.target_column = parsed_config.target_column

        # Run sanity check analysis
        result = await enhanced_model_service.run_sanity_check_analysis(
            reference_data, current_data, model_file, full_config
        )

        serialized_result = serialize_response(result)

        # Generate AI explanation
        try:
            ai_summary_payload = create_ai_summary_for_sanity_check(serialized_result)
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, analysis_type="sanity_check"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            serialized_result["llm_response"] = {
                "summary": "Sanity check completed successfully.",
                "detailed_explanation": "Pre-flight diagnostic analysis has identified the similarity between reference and current data distributions. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Sanity check analysis completed successfully",
                    "Review similarity scores and expected performance drop",
                    "AI explanations will return when service is restored",
                ],
            }

        return serialized_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sanity check analysis failed: {str(e)}")


@router.get("/session/sanity-check/{session_id}")
async def session_sanity_check(session_id: str, user: dict = Depends(get_current_user)):
    """
    **Sanity Check Analysis using Session Data**

    Performs sanity check analysis using previously uploaded data from session storage.
    This endpoint is useful when data has already been uploaded via the unified upload endpoint.

    **Parameters:**
    - session_id: UUID of the upload session

    **Returns:**
    Same structure as POST /sanity-check endpoint with AI-generated explanations.
    """
    try:
        # Convert session data to UploadFile-like objects
        reference_file, current_file, model_file, config = create_upload_files_from_session(session_id)

        if not reference_file or not current_file:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found or incomplete")

        # Create config from session data
        parsed_config = None
        if config:

            class SanityCheckConfig(BaseModel):
                model_type: str = "classification"
                target_column: Optional[str] = None

            parsed_config = SanityCheckConfig(
                model_type=config.get("model_type", "classification"), target_column=config.get("target_column")
            )

        # Build full config
        full_config = None
        if parsed_config:
            model_type_enum = ModelType(parsed_config.model_type.lower())
            full_config = AnalysisConfiguration(
                analysis_name="Sanity Check (Session)",
                description="Pre-flight diagnostic analysis from session data",
                model_type=model_type_enum,
                selected_metrics=["accuracy"] if parsed_config.model_type == "classification" else ["r2"],
                statistical_test="ks_test",
            )
            if hasattr(parsed_config, "target_column") and parsed_config.target_column:
                full_config.target_column = parsed_config.target_column

        # Run sanity check analysis
        result = await enhanced_model_service.run_sanity_check_analysis(
            reference_file, current_file, model_file, full_config
        )

        serialized_result = serialize_response(result)

        # Generate AI explanation
        try:
            ai_summary_payload = create_ai_summary_for_sanity_check(serialized_result)
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, analysis_type="sanity_check"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            serialized_result["llm_response"] = {
                "summary": "Sanity check completed successfully.",
                "detailed_explanation": "Pre-flight diagnostic analysis has identified the similarity between reference and current data distributions. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Sanity check analysis completed successfully",
                    "Review similarity scores and expected performance drop",
                    "AI explanations will return when service is restored",
                ],
            }

        return serialized_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session-based sanity check failed: {str(e)}")
