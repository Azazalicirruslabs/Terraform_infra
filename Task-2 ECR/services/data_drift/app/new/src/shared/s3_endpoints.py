"""
S3-based stateless endpoints for data discovery and loading
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sklearn.preprocessing import LabelEncoder

from services.data_drift.app.new.src.model_drift.services.analysis.degradation_metrics_service import (
    degradation_metrics_service,
)
from services.data_drift.app.new.src.model_drift.services.analysis.performance_comparison_service import (
    performance_comparison_service,
)
from services.data_drift.app.new.src.model_drift.services.analysis.statistical_significance_service import (
    statistical_significance_service,
)
from services.data_drift.app.new.src.model_drift.services.core.metrics_calculation_service import (
    metrics_calculation_service,
)
from services.data_drift.app.new.src.shared.ai_explanation_service import ai_explanation_service
from services.data_drift.app.new.src.shared.models import (
    AIExplanationRequest,
    AnalysisRequest,
    LoadDataRequest,
    LoadDataResponse,
    ModelDriftAnalysisRequest,
    S3FileMetadata,
)
from services.data_drift.app.new.src.shared.s3_utils import (
    get_s3_file_metadata,
    load_s3_csv,
    load_s3_model,
    validate_dataframe,
    validate_target_column,
)
from shared.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

# ===== AI SUMMARY HELPER FUNCTIONS =====


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
    from fastapi.encoders import jsonable_encoder

    cleaned_result = clean_float_values(result)
    return jsonable_encoder(cleaned_result, custom_encoder={np.generic: lambda x: x.item()})


def create_ai_summary_for_performance_comparison(analysis_data: dict) -> dict:
    """
    Summarizes performance comparison analysis for AI explanation.
    Supports both single cohort and multi-cohort analysis.
    """
    # Check if this is multi-cohort analysis
    is_multi_cohort = analysis_data.get("is_multi_cohort", False)

    if is_multi_cohort:
        # Multi-cohort summary
        performance_over_time = analysis_data.get("performance_over_time", [])
        metric_trends = analysis_data.get("metric_trends", {})
        cohort_summary = analysis_data.get("cohort_summary", {})

        summary = {
            "analysis_type": "multi_cohort_time_series",
            "model_type": analysis_data.get("analysis_metadata", {}).get("model_type"),
            "num_cohorts": len(performance_over_time),
            "cohort_labels": [item.get("cohort") for item in performance_over_time],
            "overall_trend": cohort_summary.get("overall_trend", "unknown"),
            "overall_change": cohort_summary.get("overall_change"),
            "trend_metric": cohort_summary.get("overall_trend_metric"),
            "significant_degradations": cohort_summary.get("significant_degradations", []),
            "significant_improvements": cohort_summary.get("significant_improvements", []),
            "recommendations": [],
        }

        # Add metric progression details for key metrics
        key_metrics = ["accuracy", "precision", "recall", "f1_score", "mse", "rmse", "r2"]
        metric_progressions = {}

        for metric in key_metrics:
            if metric in metric_trends:
                values = metric_trends[metric]
                clean_values = [v for v in values if v is not None]
                if clean_values:
                    metric_progressions[metric] = {
                        "start_value": round(clean_values[0], 4) if clean_values else None,
                        "end_value": round(clean_values[-1], 4) if clean_values else None,
                        "change": round(clean_values[-1] - clean_values[0], 4) if len(clean_values) >= 2 else 0,
                        "values": [round(v, 4) if v is not None else None for v in values],
                    }

        summary["metric_progressions"] = metric_progressions

        # Generate recommendations based on trends
        if cohort_summary.get("overall_trend") == "degrading":
            summary["recommendations"].extend(
                [
                    "Model performance is degrading over time - investigate data drift",
                    "Consider retraining model with recent data",
                    "Review significant metric degradations for root causes",
                ]
            )
        elif cohort_summary.get("overall_trend") == "improving":
            summary["recommendations"].extend(
                [
                    "Model performance is improving - continue current approach",
                    "Monitor for consistency across future cohorts",
                ]
            )
        else:
            summary["recommendations"].extend(
                ["Model performance is stable across cohorts", "Continue regular monitoring for early drift detection"]
            )

        return summary

    else:
        # Single cohort summary (original logic)
        summary = {
            "analysis_type": "single_cohort_comparison",
            "model_type": analysis_data.get("analysis_metadata", {}).get("model_type")
            or analysis_data.get("model_info", {}).get("model_type"),
            "analysis_name": analysis_data.get("analysis_metadata", {}).get("analysis_name")
            or analysis_data.get("analysis_info", {}).get("analysis_name")
            or "performance_comparison",
            "reference_performance": analysis_data.get("reference_performance", {}),
            "current_performance": analysis_data.get("current_performance", {}),
            "performance_comparison": analysis_data.get("performance_comparison", {}),
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
    Extracts data from the nested structure returned by degradation metrics service.
    """
    # Extract the actual degradation metrics from the nested structure
    degradation_metrics = analysis_data.get(
        "degradation_metrics", analysis_data
    )  # Handle both wrapped and unwrapped formats
    sub_tabs = degradation_metrics.get("sub_tabs", {})
    overall_assessment = degradation_metrics.get("overall_degradation_assessment", {})

    # Log the structure for debugging
    logger.info(f"AI Summary extraction - degradation_metrics keys: {list(degradation_metrics.keys())}")
    logger.info(f"AI Summary extraction - sub_tabs keys: {list(sub_tabs.keys())}")
    logger.info(f"AI Summary extraction - overall_assessment keys: {list(overall_assessment.keys())}")

    summary = {
        "model_type": analysis_data.get("analysis_metadata", {}).get("model_type")
        or analysis_data.get("model_info", {}).get("model_type"),
        "analysis_name": analysis_data.get("analysis_metadata", {}).get("analysis_name")
        or analysis_data.get("analysis_info", {}).get("analysis_name")
        or "degradation_metrics",
        "disagreement_analysis": sub_tabs.get("model_disagreement", {}),
        "confidence_analysis": sub_tabs.get("confidence_analysis", {}),
        "feature_importance_drift": sub_tabs.get("feature_importance_drift", {}),
        "overall_assessment": overall_assessment,
        "recommendations": overall_assessment.get("recommendations", []),
    }

    # Extract key degradation indicators with better error handling
    disagreement = sub_tabs.get("model_disagreement", {})
    if disagreement and "error" not in disagreement:
        # Handle nested structure from disagreement service
        disagreement_stats = disagreement.get("disagreement_analysis", {})
        if disagreement_stats:
            summary["disagreement_rate"] = disagreement_stats.get("mean_absolute_difference")
            summary["prediction_stability"] = disagreement_stats.get("pearson_correlation")

        # Alternative paths for disagreement data
        if "disagreement_rate" not in summary and disagreement.get("disagreement_rate"):
            summary["disagreement_rate"] = disagreement.get("disagreement_rate")
        if "prediction_stability" not in summary and disagreement.get("stability_score"):
            summary["prediction_stability"] = disagreement.get("stability_score")

    confidence = sub_tabs.get("confidence_analysis", {})
    if confidence and "error" not in confidence:
        # Handle nested structure from confidence service
        calibration = confidence.get("calibration_assessment", {})
        if calibration:
            summary["confidence_decline"] = calibration.get("drift_severity")

        confidence_metrics = confidence.get("confidence_metrics", {})
        if confidence_metrics:
            low_conf_samples = confidence_metrics.get("low_confidence_samples", [])
            summary["low_confidence_predictions"] = (
                len(low_conf_samples) if isinstance(low_conf_samples, list) else low_conf_samples
            )

        # Alternative paths for confidence data
        if "confidence_decline" not in summary and confidence.get("confidence_decline"):
            summary["confidence_decline"] = confidence.get("confidence_decline")
        if "low_confidence_predictions" not in summary and confidence.get("low_confidence_count"):
            summary["low_confidence_predictions"] = confidence.get("low_confidence_count")

    # Add overall degradation indicators
    if overall_assessment and "error" not in overall_assessment:
        summary["degradation_level"] = overall_assessment.get("overall_degradation_level")
        summary["degradation_score"] = overall_assessment.get("degradation_score")
        summary["key_indicators"] = overall_assessment.get("key_degradation_indicators", [])

    # Log what we extracted
    extracted_values = {k: v for k, v in summary.items() if v is not None and v != {} and v != []}
    logger.info(f"AI Summary extracted values: {list(extracted_values.keys())}")

    return summary


def create_ai_summary_for_statistical_significance(analysis_data: dict) -> dict:
    """
    Summarizes statistical significance analysis for AI explanation.
    """
    summary = {
        "model_type": analysis_data.get("analysis_metadata", {}).get("model_type")
        or analysis_data.get("model_info", {}).get("model_type"),
        "analysis_name": analysis_data.get("analysis_metadata", {}).get("analysis_name")
        or analysis_data.get("analysis_info", {}).get("analysis_name")
        or "statistical_significance",
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


# ===== MODEL WRAPPING UTILITIES =====


def apply_consistent_preprocessing(ref_df, curr_df, feature_columns, target_column, preprocessing_info=None):
    """
    Apply consistent preprocessing to both reference and current datasets

    Args:
        ref_df: Reference DataFrame
        curr_df: Current DataFrame
        feature_columns: List of expected feature columns
        target_column: Target column name
        preprocessing_info: Preprocessing information from model wrapper

    Returns:
        Tuple of (X_ref, y_ref, X_curr, y_curr, preprocessing_metadata)
    """
    # Check if model was likely trained on raw categorical columns (no preprocessing)
    # This is a heuristic - if feature_columns contains known categorical columns, assume
    # the model expects raw categorical data
    categorical_columns = []
    for col in ref_df.columns:
        if col != target_column and ref_df[col].dtype == "object":
            categorical_columns.append(col)

    # If any of our categorical columns are in feature_columns, the model likely
    # expects raw categorical data without one-hot encoding
    raw_categorical_model = False
    if feature_columns:
        for col in categorical_columns:
            if col in feature_columns:
                raw_categorical_model = True
                logger.info(f"Detected model trained on raw categorical features: {col} found in feature_columns")
                break

    # If we detect the model expects raw categorical features, use a simplified preprocessing approach
    if raw_categorical_model:
        logger.info("Using simplified preprocessing to maintain original column structure")

        # For reference data
        if feature_columns:
            X_ref = ref_df[feature_columns].copy()
        else:
            X_ref = ref_df.drop(columns=[target_column]) if target_column in ref_df.columns else ref_df.copy()

        y_ref = ref_df[target_column] if target_column in ref_df.columns else None

        # For current data
        if feature_columns:
            X_curr = curr_df[feature_columns].copy()
        else:
            X_curr = curr_df.drop(columns=[target_column]) if target_column in curr_df.columns else curr_df.copy()

        y_curr = curr_df[target_column] if target_column in curr_df.columns else None

        # Simple metadata
        preprocessing_metadata = {
            "categorical_columns": categorical_columns,
            "encoded_columns": [],
            "original_columns": list(X_ref.columns),
            "raw_categorical_model": True,
        }

        # Handle categorical columns with label encoding (most models expect this)

        label_encoders = {}

        for col in categorical_columns:
            if col in X_ref.columns:
                logger.info(f"Label encoding categorical column: {col}")

                # Create label encoder and fit on combined data to ensure consistency
                le = LabelEncoder()
                combined_values = pd.concat([X_ref[col].astype(str), X_curr[col].astype(str)])
                le.fit(combined_values.dropna())

                # Transform both datasets
                X_ref[col] = le.transform(X_ref[col].astype(str))
                X_curr[col] = le.transform(X_curr[col].astype(str))

                label_encoders[col] = le
                logger.info(f"Encoded {col} with {len(le.classes_)} unique values: {le.classes_[:5]}...")

        # CRITICAL FIX: Ensure ALL columns are numeric (prevent "ufunc 'isnan' not supported" error)
        for col in X_ref.columns:
            # Check if column is still object/string dtype
            if X_ref[col].dtype == "object" or X_curr[col].dtype == "object":
                logger.warning(f"Column {col} still has object dtype after label encoding. Converting to numeric.")
                try:
                    # Try to convert to numeric (coerce will turn non-numeric to NaN)
                    X_ref[col] = pd.to_numeric(X_ref[col], errors="coerce")
                    X_curr[col] = pd.to_numeric(X_curr[col], errors="coerce")
                except Exception as e:
                    logger.error(f"Failed to convert {col} to numeric: {e}")
                    # Last resort: convert to codes (ordinal encoding)
                    X_ref[col] = pd.Categorical(X_ref[col]).codes
                    X_curr[col] = pd.Categorical(X_curr[col]).codes
            elif col not in categorical_columns:
                # Ensure numeric columns are actually numeric
                try:
                    X_ref[col] = pd.to_numeric(X_ref[col], errors="coerce")
                    X_curr[col] = pd.to_numeric(X_curr[col], errors="coerce")
                except:
                    pass

        # Fill any NaN values AFTER ensuring all columns are numeric
        X_ref = X_ref.fillna(0)
        X_curr = X_curr.fillna(0)

        # Final validation: ensure all columns are numeric types
        for col in X_ref.columns:
            if X_ref[col].dtype == "object":
                logger.error(f"CRITICAL: Column {col} is still object dtype! Force converting to float.")
                X_ref[col] = pd.to_numeric(X_ref[col].astype(str), errors="coerce").fillna(0)
                X_curr[col] = pd.to_numeric(X_curr[col].astype(str), errors="coerce").fillna(0)

        preprocessing_metadata["label_encoders"] = label_encoders

        logger.info(f"Raw categorical preprocessing complete. Shape: Ref {X_ref.shape}, Curr {X_curr.shape}")
        logger.info(f"Data types: {dict(X_ref.dtypes.value_counts())}")

        return X_ref, y_ref, X_curr, y_curr, preprocessing_metadata

    # Otherwise, use our standard one-hot encoding approach
    else:
        logger.info("Using one-hot encoding preprocessing for model trained on encoded features")

        # Prepare reference data
        X_ref, y_ref, ref_metadata = prepare_data_for_model(ref_df, feature_columns, target_column, preprocessing_info)

        # For current data, we need to apply the SAME preprocessing as reference
        # This ensures consistent feature encoding between datasets
        X_curr, y_curr, curr_metadata = prepare_data_for_model(
            curr_df, feature_columns, target_column, preprocessing_info
        )

        # Ensure both datasets have the same columns after preprocessing
        ref_columns = set(X_ref.columns)
        curr_columns = set(X_curr.columns)

        if ref_columns != curr_columns:
            logger.warning(f"Column mismatch between datasets. Ref: {len(ref_columns)}, Curr: {len(curr_columns)}")

            # Get the union of all columns and add missing columns as zeros
            all_columns = sorted(ref_columns.union(curr_columns))

            for col in all_columns:
                if col not in X_ref.columns:
                    X_ref[col] = 0
                if col not in X_curr.columns:
                    X_curr[col] = 0

            # Reorder columns to match
            X_ref = X_ref[all_columns]
            X_curr = X_curr[all_columns]

            logger.info(f"Aligned datasets to {len(all_columns)} columns")

        return X_ref, y_ref, X_curr, y_curr, ref_metadata


def extract_model_from_wrapper(wrapped_model):
    """
    Extract the actual model from a wrapped model, handling both wrapped and unwrapped cases

    Args:
        wrapped_model: Either a wrapped model dict or direct model object

    Returns:
        Tuple of (model, metadata)
    """
    if isinstance(wrapped_model, dict) and "model" in wrapped_model:
        # This is a wrapped model
        logger.info("Found wrapped model with metadata")
        return wrapped_model["model"], {
            "feature_columns": wrapped_model.get("feature_columns", []),
            "target_column": wrapped_model.get("target_column"),
            "model_type": wrapped_model.get("model_type"),
            "preprocessing_info": wrapped_model.get("preprocessing_info", {}),
            "wrapper_version": wrapped_model.get("wrapper_version"),
        }
    else:
        # This is an unwrapped model - try to extract what we can
        logger.info("Found unwrapped model, attempting to extract feature information")

        feature_columns = []

        # Try to extract feature names from sklearn models
        if hasattr(wrapped_model, "feature_names_in_"):
            feature_columns = list(wrapped_model.feature_names_in_)
            logger.info(f"Extracted {len(feature_columns)} feature names from sklearn model: {feature_columns[:5]}...")
        elif hasattr(wrapped_model, "feature_importances_"):
            # For models with feature importance, we can at least get the count
            n_features = len(wrapped_model.feature_importances_)
            feature_columns = [f"feature_{i}" for i in range(n_features)]
            logger.info(f"Model has {n_features} features but no feature names available")

        return wrapped_model, {
            "feature_columns": feature_columns,
            "target_column": None,
            "model_type": str(type(wrapped_model).__name__),
            "preprocessing_info": {},
            "wrapper_version": None,
        }


def prepare_data_for_model(df, feature_columns, target_column, preprocessing_info=None):
    """
    Prepare DataFrame for model prediction, handling data type conversion and categorical encoding

    Args:
        df: Input DataFrame
        feature_columns: List of expected feature columns
        target_column: Target column name (optional)
        preprocessing_info: Dict with preprocessing information from model wrapper

    Returns:
        Tuple of (X, y, preprocessing_metadata) where y is None if target_column not provided
    """
    try:
        # If feature columns are specified, use only those
        if feature_columns:
            # Check if all required features are present
            missing_features = set(feature_columns) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            X = df[feature_columns].copy()
        else:
            # If no feature columns specified, use all except target
            if target_column and target_column in df.columns:
                X = df.drop(columns=[target_column])
            else:
                X = df.copy()

        # Handle target column
        y = None
        if target_column and target_column in df.columns:
            y = df[target_column]

        # Track preprocessing metadata for consistency
        preprocessing_metadata = {"categorical_columns": [], "encoded_columns": [], "original_columns": list(X.columns)}

        # First pass: identify categorical and numeric columns
        categorical_columns = []
        for col in X.columns:
            if X[col].dtype == "object":
                # Try to convert to numeric first
                numeric_series = pd.to_numeric(X[col], errors="coerce")
                # If conversion creates too many NaNs, treat as categorical
                if numeric_series.isna().sum() / len(numeric_series) > 0.5:
                    categorical_columns.append(col)
                    logger.info(f"Column {col} identified as categorical")
                else:
                    X[col] = numeric_series
                    logger.info(f"Column {col} converted to numeric")

        # Handle categorical columns with one-hot encoding
        if categorical_columns:
            logger.info(f"One-hot encoding categorical columns: {categorical_columns}")
            preprocessing_metadata["categorical_columns"] = categorical_columns

            # Store original values for reference
            for col in categorical_columns:
                preprocessing_metadata[f"{col}_unique_values"] = list(X[col].unique())

            # Apply one-hot encoding
            X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=False, dummy_na=True)
            preprocessing_metadata["encoded_columns"] = [col for col in X_encoded.columns if col not in X.columns]

            logger.info(f"Original shape: {X.shape}, Encoded shape: {X_encoded.shape}")
            logger.info(f"New encoded columns: {preprocessing_metadata['encoded_columns'][:5]}...")  # Show first 5

            X = X_encoded

        # Final data type conversion and validation
        for col in X.columns:
            if X[col].dtype == "object":
                # Try one more time to convert any remaining object columns
                try:
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                except:
                    pass

        # Fill any remaining NaN values
        X = X.fillna(0)

        logger.info(f"Final prepared data shape: {X.shape}")
        logger.info(f"Data types: {dict(X.dtypes.value_counts())}")

        return X, y, preprocessing_metadata

    except Exception as e:
        logger.error(f"Error preparing data for model: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Data preparation failed: {str(e)}. This might be due to categorical data that needs preprocessing.",
        )


# ===== ENDPOINTS =====


@router.get("/s3/files/{analysis_type}/{project_id}", response_model=S3FileMetadata)
async def get_s3_files(analysis_type: str, project_id: str, user: Dict = Depends(get_current_user)):
    """
    Get list of files and models available in S3 for a specific project

    Args:
        project_id: The project ID to fetch files for

    Returns:
        Dictionary containing files and models lists with metadata
    """
    token = user.get("token") if user else None
    try:
        logger.info(f"Fetching S3 files for project: {project_id}")
        result = get_s3_file_metadata(analysis_type, project_id, token)
        logger.info(f"Found {len(result.get('files', []))} files and {len(result.get('models', []))} models")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching S3 files for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch S3 files: {str(e)}")


@router.post("/load", response_model=LoadDataResponse)
async def load_data(payload: LoadDataRequest, user: Dict = Depends(get_current_user)):
    """
    Load datasets and optional model from S3 URLs for analysis

    Args:
        payload: LoadDataRequest containing S3 URLs and configuration

    Returns:
        LoadDataResponse with loaded data information and validation status
    """
    try:
        logger.info(f"Loading data from URLs: ref={payload.reference_url}, curr={payload.current_url}")

        # Load datasets from S3
        reference_df = load_s3_csv(payload.reference_url)
        current_df = load_s3_csv(payload.current_url)

        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")

        # Find common columns
        ref_columns = set(reference_df.columns)
        curr_columns = set(current_df.columns)
        common_columns = list(ref_columns.intersection(curr_columns))

        if not common_columns:
            raise HTTPException(
                status_code=400, detail="No common columns found between reference and current datasets"
            )

        # Validate target column if provided
        if payload.target_column:
            validate_target_column(reference_df, payload.target_column, "reference")
            validate_target_column(current_df, payload.target_column, "current")

        # Load model if provided
        model = None
        model_info = None
        model_loaded = False

        if payload.model_url:
            try:
                model = load_s3_model(payload.model_url)
                model_loaded = True
                model_info = {"type": str(type(model).__name__), "url": payload.model_url, "loaded_successfully": True}
                logger.info(f"Model loaded successfully: {type(model).__name__}")
            except Exception as e:
                logger.warning(f"Failed to load model from {payload.model_url}: {e}")
                model_info = {
                    "type": "unknown",
                    "url": payload.model_url,
                    "loaded_successfully": False,
                    "error": str(e),
                }

        # Prepare response
        response = LoadDataResponse(
            message=f"Successfully loaded datasets with {len(common_columns)} common columns",
            reference_dataset={
                "shape": reference_df.shape,
                "columns": list(reference_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in reference_df.dtypes.items()},
                "url": payload.reference_url,
            },
            current_dataset={
                "shape": current_df.shape,
                "columns": list(current_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in current_df.dtypes.items()},
                "url": payload.current_url,
            },
            model_loaded=model_loaded,
            model_info=model_info,
            target_column=payload.target_column,
            config=payload.config,
            common_columns=common_columns,
        )

        logger.info(f"Data loading completed successfully. Ref: {reference_df.shape}, Curr: {current_df.shape}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=f"Data loading failed: {str(e)}")


# ===== MODEL DRIFT S3 ENDPOINTS =====
# These load data directly from S3 URLs and perform model drift analysis

# ===== HELPER FUNCTIONS FOR MULTI-COHORT ANALYSIS =====


async def _evaluate_single_cohort(
    cohort_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    model: Any,
    model_metadata: Dict[str, Any],
    target_column: str,
    cohort_label: str,
) -> Dict[str, Any]:
    """
    Evaluate a single cohort dataset and return performance metrics

    Args:
        cohort_df: Cohort dataset to evaluate
        reference_df: Reference dataset (for preprocessing consistency)
        model: Trained model
        model_metadata: Model metadata from wrapper
        target_column: Name of target column
        cohort_label: Label for this cohort (e.g., "week1", "current")

    Returns:
        Dictionary with cohort metrics and metadata
    """
    try:
        # Validate cohort dataset
        validate_dataframe(cohort_df, cohort_label)

        # Prepare data using consistent preprocessing
        # apply_consistent_preprocessing returns: (X_ref, y_ref, X_curr, y_curr, metadata)
        # We want the cohort data (X_curr, y_curr), not the reference data
        _, _, X_cohort, y_true_cohort, preprocessing_metadata = apply_consistent_preprocessing(
            reference_df,
            cohort_df,
            model_metadata.get("feature_columns"),
            target_column,
            model_metadata.get("preprocessing_info"),
        )

        # Generate predictions
        pred_cohort = model.predict(X_cohort)

        # Get prediction probabilities if available
        pred_cohort_proba = None
        if hasattr(model, "predict_proba"):
            try:
                pred_cohort_proba = model.predict_proba(X_cohort)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities for {cohort_label}: {e}")

        # Calculate performance metrics directly (more reliable than comparison service for single cohort)
        logger.info(f"Calculating metrics for cohort '{cohort_label}'...")
        logger.info(f"  - Dataset size: {len(cohort_df)}")
        logger.info(f"  - Predictions shape: {pred_cohort.shape}")
        logger.info(f"  - Has probabilities: {pred_cohort_proba is not None}")
        logger.info(f"  - Ground truth shape: {y_true_cohort.shape}")

        # Detect model type based on target variable characteristics
        unique_ratio = len(np.unique(y_true_cohort)) / len(y_true_cohort) if len(y_true_cohort) > 0 else 0
        is_regression = unique_ratio > 0.1  # More than 10% unique values suggests regression

        # Also check if model has predict_proba (indicates classification)
        if pred_cohort_proba is not None:
            is_regression = False

        detected_model_type = "regression" if is_regression else "classification"
        logger.info(f"  - Model type detected: {detected_model_type}")
        logger.info(f"  - Unique ratio: {unique_ratio:.4f}")

        # Calculate metrics based on model type
        try:
            if is_regression:
                logger.info("  - Calculating regression metrics...")
                metrics = metrics_calculation_service.regression_metrics(y_true_cohort, pred_cohort)
            else:
                logger.info("  - Calculating classification metrics...")
                metrics = metrics_calculation_service.classification_metrics(
                    y_true_cohort, pred_cohort, pred_cohort_proba
                )

            # Check if metrics calculation returned an error
            if "error" in metrics:
                logger.error(f"Error calculating metrics for {cohort_label}: {metrics.get('error')}")
                metrics = {}
            else:
                logger.info(f"✓ Metrics calculated for {cohort_label}: {len(metrics)} metrics")
                if metrics:
                    # Log sample of metrics for debugging
                    sample_metrics = list(metrics.keys())[:5]
                    logger.info(f"  Sample metrics: {sample_metrics}")
                else:
                    logger.warning(f"  WARNING: No metrics returned for {cohort_label}")

        except Exception as e:
            logger.error(f"Exception calculating metrics for {cohort_label}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            metrics = {}

        return {
            "cohort": cohort_label,
            "dataset_size": len(cohort_df),
            "metrics": metrics,
            "predictions_shape": list(pred_cohort.shape),
            "has_probabilities": pred_cohort_proba is not None,
            "model_type": detected_model_type,
        }

    except Exception as e:
        logger.error(f"Failed to evaluate cohort '{cohort_label}': {e}")
        raise HTTPException(status_code=400, detail=f"Cohort '{cohort_label}' evaluation failed: {str(e)}")


def generate_metric_trends(performance_over_time: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate time-series metric trends from cohort performance data

    Args:
        performance_over_time: List of cohort performance dictionaries

    Returns:
        Dictionary with metric trends suitable for visualization
    """
    if not performance_over_time:
        return {}

    # Extract cohort labels
    cohorts = [item["cohort"] for item in performance_over_time]

    # Initialize trend dictionary
    trends = {"cohorts": cohorts}

    # Collect all unique metric names from all cohorts
    all_metrics = set()
    for cohort_data in performance_over_time:
        metrics = cohort_data.get("metrics", {})
        all_metrics.update(metrics.keys())

    # Build trend arrays for each metric
    # Skip complex nested metrics but explode per-class arrays
    skip_metrics = {"confusion_matrix", "classification_report"}

    for metric_name in all_metrics:
        if metric_name in skip_metrics:
            continue  # Skip complex nested metrics

        metric_values = []
        is_array_metric = False

        for cohort_data in performance_over_time:
            metrics = cohort_data.get("metrics", {})
            value = metrics.get(metric_name)

            # Handle nested metric structures (e.g., {"value": 0.95, "confidence": ...})
            if isinstance(value, dict) and "value" in value:
                metric_values.append(value["value"])
            elif isinstance(value, (int, float)):
                metric_values.append(value)
            elif isinstance(value, (list, np.ndarray)):
                # Mark as array metric for later processing
                is_array_metric = True
                metric_values.append(value)
            else:
                metric_values.append(None)

        # For array metrics (per-class), explode into separate trends
        if is_array_metric and metric_values[0] is not None:
            num_classes = len(metric_values[0]) if isinstance(metric_values[0], (list, np.ndarray)) else 0
            if num_classes > 0:
                # Create separate trend for each class
                for class_idx in range(num_classes):
                    class_trend = []
                    for val in metric_values:
                        if isinstance(val, (list, np.ndarray)) and len(val) > class_idx:
                            class_trend.append(float(val[class_idx]))
                        else:
                            class_trend.append(None)
                    trends[f"{metric_name}_class_{class_idx}"] = class_trend

                # Also add aggregated statistics
                avg_trend = []
                min_trend = []
                max_trend = []
                std_trend = []
                for val in metric_values:
                    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                        avg_trend.append(float(np.mean(val)))
                        min_trend.append(float(np.min(val)))
                        max_trend.append(float(np.max(val)))
                        std_trend.append(float(np.std(val)))
                    else:
                        avg_trend.append(None)
                        min_trend.append(None)
                        max_trend.append(None)
                        std_trend.append(None)

                trends[f"{metric_name}_avg"] = avg_trend
                trends[f"{metric_name}_min"] = min_trend
                trends[f"{metric_name}_max"] = max_trend
                trends[f"{metric_name}_std"] = std_trend
        else:
            # Regular scalar metric
            trends[metric_name] = metric_values

    # Calculate metric deltas (change from reference to each cohort)
    if len(performance_over_time) > 1:
        reference_metrics = performance_over_time[0].get("metrics", {})
        deltas = {}

        for i, cohort_data in enumerate(performance_over_time[1:], start=1):
            cohort_metrics = cohort_data.get("metrics", {})
            cohort_label = cohort_data["cohort"]
            deltas[cohort_label] = {}

            for metric_name in all_metrics:
                ref_value = reference_metrics.get(metric_name)
                cohort_value = cohort_metrics.get(metric_name)

                # Extract numeric values
                if isinstance(ref_value, dict) and "value" in ref_value:
                    ref_value = ref_value["value"]
                if isinstance(cohort_value, dict) and "value" in cohort_value:
                    cohort_value = cohort_value["value"]

                # Calculate delta
                if isinstance(ref_value, (int, float)) and isinstance(cohort_value, (int, float)):
                    delta = cohort_value - ref_value
                    percent_change = (delta / ref_value * 100) if ref_value != 0 else 0
                    deltas[cohort_label][metric_name] = {
                        "absolute_change": round(delta, 4),
                        "percent_change": round(percent_change, 2),
                    }

        trends["deltas"] = deltas

    return trends


def create_cohort_summary(performance_over_time: List[Dict[str, Any]], metric_trends: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create executive summary for multi-cohort analysis

    Args:
        performance_over_time: List of cohort performance data
        metric_trends: Metric trends dictionary

    Returns:
        Summary dictionary with key insights
    """
    if not performance_over_time or len(performance_over_time) < 2:
        return {}

    summary = {
        "total_cohorts": len(performance_over_time),
        "cohort_labels": [item["cohort"] for item in performance_over_time],
        "analysis_type": "multi_cohort_time_series",
    }

    # Identify metrics with significant changes
    deltas = metric_trends.get("deltas", {})
    if deltas:
        significant_degradations = []
        significant_improvements = []

        # Get last cohort's deltas (most recent comparison)
        last_cohort_label = list(deltas.keys())[-1]
        last_cohort_deltas = deltas[last_cohort_label]

        for metric_name, delta_info in last_cohort_deltas.items():
            percent_change = delta_info.get("percent_change", 0)

            if percent_change < -5:  # More than 5% degradation
                significant_degradations.append({"metric": metric_name, "change": percent_change})
            elif percent_change > 5:  # More than 5% improvement
                significant_improvements.append({"metric": metric_name, "change": percent_change})

        summary["significant_degradations"] = sorted(significant_degradations, key=lambda x: x["change"])[
            :5
        ]  # Top 5 degradations

        summary["significant_improvements"] = sorted(significant_improvements, key=lambda x: x["change"], reverse=True)[
            :5
        ]  # Top 5 improvements

    # Calculate overall trend (improving, stable, degrading)
    # Use primary metric (accuracy for classification, r2 for regression)
    primary_metrics = ["accuracy", "r2", "f1_score"]
    trend_metric = None

    for metric in primary_metrics:
        if metric in metric_trends:
            trend_metric = metric
            break

    if trend_metric:
        values = metric_trends[trend_metric]
        # Remove None values
        clean_values = [v for v in values if v is not None]

        if len(clean_values) >= 2:
            first_value = clean_values[0]
            last_value = clean_values[-1]
            change = last_value - first_value

            if change < -0.05:
                summary["overall_trend"] = "degrading"
            elif change > 0.05:
                summary["overall_trend"] = "improving"
            else:
                summary["overall_trend"] = "stable"

            summary["overall_trend_metric"] = trend_metric
            summary["overall_change"] = round(change, 4)

    # ===== CRITICAL DRIFT METRICS =====
    # Calculate additional drift-specific indicators

    # 1. Class Distribution Shift (works for binary and multiclass)
    # For binary: calculate positive class ratio
    # For multiclass: calculate distribution entropy and per-class shifts
    class_distribution_metrics = {}

    # Get all TP/TN metrics or look for per-class support in classification_report
    if "true_positives" in metric_trends and "true_negatives" in metric_trends:
        # Binary classification case
        tp_trend = metric_trends.get("true_positives", [])
        tn_trend = metric_trends.get("true_negatives", [])

        if len(tp_trend) >= 2 and None not in tp_trend[:2]:
            ref_ratio = tp_trend[0] / (tp_trend[0] + tn_trend[0]) if (tp_trend[0] + tn_trend[0]) > 0 else 0
            current_ratio = tp_trend[-1] / (tp_trend[-1] + tn_trend[-1]) if (tp_trend[-1] + tn_trend[-1]) > 0 else 0

            class_distribution_metrics = {
                "type": "binary",
                "reference_positive_ratio": round(ref_ratio, 4),
                "current_positive_ratio": round(current_ratio, 4),
                "shift": round(current_ratio - ref_ratio, 4),
                "shift_percent": round((current_ratio - ref_ratio) * 100, 2),
            }

    # For multiclass: extract class distributions from performance_over_time
    elif len(performance_over_time) >= 2:
        # Try to extract class distribution from confusion matrix or classification report
        ref_metrics = performance_over_time[0].get("metrics", {})
        curr_metrics = performance_over_time[-1].get("metrics", {})

        ref_cm = ref_metrics.get("confusion_matrix")
        curr_cm = curr_metrics.get("confusion_matrix")

        if ref_cm and curr_cm:
            # Calculate class distributions from confusion matrix
            ref_totals = [sum(row) for row in ref_cm]
            curr_totals = [sum(row) for row in curr_cm]

            ref_total = sum(ref_totals)
            curr_total = sum(curr_totals)

            ref_dist = [count / ref_total for count in ref_totals] if ref_total > 0 else []
            curr_dist = [count / curr_total for count in curr_totals] if curr_total > 0 else []

            if len(ref_dist) > 0 and len(curr_dist) > 0:
                # Calculate per-class shifts
                class_shifts = []
                for i in range(len(ref_dist)):
                    if i < len(curr_dist):
                        shift = curr_dist[i] - ref_dist[i]
                        class_shifts.append(
                            {
                                "class": i,
                                "reference_proportion": round(ref_dist[i], 4),
                                "current_proportion": round(curr_dist[i], 4),
                                "shift": round(shift, 4),
                                "shift_percent": round(shift * 100, 2),
                            }
                        )

                # Calculate distribution entropy (measure of imbalance)
                ref_entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in ref_dist)
                curr_entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in curr_dist)

                class_distribution_metrics = {
                    "type": "multiclass",
                    "num_classes": len(ref_dist),
                    "per_class_shifts": class_shifts,
                    "reference_entropy": round(ref_entropy, 4),
                    "current_entropy": round(curr_entropy, 4),
                    "entropy_change": round(curr_entropy - ref_entropy, 4),
                    "interpretation": (
                        "more_balanced"
                        if curr_entropy > ref_entropy
                        else "more_imbalanced" if curr_entropy < ref_entropy else "stable"
                    ),
                }

    if class_distribution_metrics:
        summary["class_distribution_drift"] = class_distribution_metrics

    # 2. Performance Volatility (standard deviation across cohorts)
    if trend_metric and len(clean_values) >= 3:
        volatility = float(np.std(clean_values))
        summary["performance_volatility"] = {
            "metric": trend_metric,
            "std_deviation": round(volatility, 4),
            "coefficient_of_variation": (
                round(volatility / np.mean(clean_values) * 100, 2) if np.mean(clean_values) > 0 else 0
            ),
            "interpretation": "high" if volatility > 0.1 else "moderate" if volatility > 0.05 else "low",
        }

    # 3. Monotonic Degradation Check (is performance consistently declining?)
    if trend_metric and len(clean_values) >= 3:
        is_monotonic_decline = all(clean_values[i] >= clean_values[i + 1] for i in range(len(clean_values) - 1))
        is_monotonic_improve = all(clean_values[i] <= clean_values[i + 1] for i in range(len(clean_values) - 1))

        summary["trend_pattern"] = {
            "is_monotonic_decline": is_monotonic_decline,
            "is_monotonic_improvement": is_monotonic_improve,
            "is_volatile": not (is_monotonic_decline or is_monotonic_improve),
        }

    # 4. Critical Threshold Violations (performance below acceptable levels)
    critical_thresholds = {"accuracy": 0.7, "f1_score": 0.7, "precision": 0.7, "recall": 0.7, "roc_auc": 0.7, "r2": 0.5}

    threshold_violations = []
    for metric, threshold in critical_thresholds.items():
        if metric in metric_trends:
            values = [v for v in metric_trends[metric] if v is not None]
            if values:
                min_value = min(values)
                if min_value < threshold:
                    threshold_violations.append(
                        {
                            "metric": metric,
                            "threshold": threshold,
                            "min_value": round(min_value, 4),
                            "violation_severity": round((threshold - min_value) / threshold * 100, 2),
                        }
                    )

    if threshold_violations:
        summary["threshold_violations"] = sorted(
            threshold_violations, key=lambda x: x["violation_severity"], reverse=True
        )

    # 5. Prediction Shift Magnitude (based on confusion matrix changes)
    if "false_positives" in metric_trends and "false_negatives" in metric_trends:
        fp_trend = metric_trends.get("false_positives", [])
        fn_trend = metric_trends.get("false_negatives", [])

        if len(fp_trend) >= 2 and None not in [fp_trend[0], fp_trend[-1], fn_trend[0], fn_trend[-1]]:
            fp_change = fp_trend[-1] - fp_trend[0]
            fn_change = fn_trend[-1] - fn_trend[0]
            total_error_change = abs(fp_change) + abs(fn_change)

            summary["prediction_shift"] = {
                "false_positive_change": int(fp_change),
                "false_negative_change": int(fn_change),
                "total_error_shift": int(total_error_change),
                "error_type_shift": (
                    "more_false_positives"
                    if fp_change > fn_change
                    else "more_false_negatives" if fn_change > fp_change else "balanced"
                ),
            }

    return summary


async def enhance_multi_cohort_analysis(
    performance_over_time: List[Dict[str, Any]],
    metric_trends: Dict[str, Any],
    cohort_summary: Dict[str, Any],
    reference_df: pd.DataFrame,
    model: Any,
    cohort_dfs: List[pd.DataFrame],
    cohort_labels: List[str],
    model_metadata: Dict[str, Any],
    target_column: str,
) -> Dict[str, Any]:
    """
    Enhance multi-cohort analysis with comprehensive drift detection components:
    - PSI (Prediction Stability Index) for each cohort
    - Effect size analysis (Cohen's d, Hedges' g, etc.) for each cohort
    - Degradation analysis with severity classifications
    - Detailed metric comparison tables
    - Statistical significance tests
    - Enhanced overall assessment with risk levels

    Args:
        performance_over_time: List of cohort performance data
        metric_trends: Metric trends dictionary
        cohort_summary: Executive summary
        reference_df: Reference dataset
        model: Trained model
        cohort_dfs: List of cohort dataframes
        cohort_labels: List of cohort labels
        model_metadata: Model metadata
        target_column: Target column name

    Returns:
        Enhanced analysis dictionary with additional components
    """
    from services.data_drift.app.new.src.model_drift.services.advanced.psi_service import psi_service
    from services.data_drift.app.new.src.model_drift.services.core.effect_size_service import effect_size_service
    from services.data_drift.app.new.src.model_drift.services.core.metrics_calculation_service import (
        metrics_calculation_service,
    )

    enhancement = {"cohort_comparisons": []}  # Detailed comparison for each cohort vs reference

    # Reference cohort data
    reference_cohort = performance_over_time[0]
    reference_metrics = reference_cohort.get("metrics", {})

    # Get reference predictions (needed for PSI and effect size)
    try:
        # Get reference predictions
        X_ref, y_ref, _, _, _ = apply_consistent_preprocessing(
            reference_df,
            reference_df,
            model_metadata.get("feature_columns"),
            target_column,
            model_metadata.get("preprocessing_info"),
        )
        pred_ref = model.predict(X_ref)
        pred_ref_proba = None
        if hasattr(model, "predict_proba"):
            try:
                pred_ref_proba = model.predict_proba(X_ref)
            except Exception as e:
                logger.warning(f"Could not get reference prediction probabilities: {e}")
    except Exception as e:
        logger.warning(f"Could not get reference predictions for enhancements: {e}")
        return enhancement

    # Detect model type
    is_classification = pred_ref_proba is not None or len(np.unique(pred_ref)) < 30

    # Process each cohort (skip reference)
    for idx, (cohort_data, cohort_df, cohort_label) in enumerate(
        zip(performance_over_time[1:], cohort_dfs, cohort_labels), start=1
    ):
        cohort_comparison = {"cohort_label": cohort_label, "cohort_index": idx}

        try:
            # Get cohort predictions
            _, _, X_cohort, y_cohort, _ = apply_consistent_preprocessing(
                reference_df,
                cohort_df,
                model_metadata.get("feature_columns"),
                target_column,
                model_metadata.get("preprocessing_info"),
            )
            pred_cohort = model.predict(X_cohort)
            pred_cohort_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    pred_cohort_proba = model.predict_proba(X_cohort)
                except Exception as e:
                    logger.warning(f"Could not get prediction probabilities for {cohort_label}: {e}")

            # ===== 1. PSI (Prediction Stability Index) - Classification Only =====
            if is_classification and pred_ref_proba is not None and pred_cohort_proba is not None:
                try:
                    psi_result = psi_service.calculate_prediction_psi(pred_ref_proba, pred_cohort_proba)
                    cohort_comparison["prediction_drift"] = psi_result
                except Exception as e:
                    logger.warning(f"PSI calculation failed for {cohort_label}: {e}")
            elif not is_classification:
                # For regression, calculate prediction drift using statistical measures
                try:
                    from scipy import stats

                    # KS test for distribution shift
                    ks_stat, ks_pval = stats.ks_2samp(pred_ref, pred_cohort)

                    # Calculate mean and std shift
                    ref_mean, ref_std = np.mean(pred_ref), np.std(pred_ref)
                    cohort_mean, cohort_std = np.mean(pred_cohort), np.std(pred_cohort)

                    mean_shift = cohort_mean - ref_mean
                    std_shift = cohort_std - ref_std

                    # Determine drift severity based on KS statistic
                    if ks_stat > 0.3:
                        severity = "High"
                    elif ks_stat > 0.15:
                        severity = "Medium"
                    else:
                        severity = "Low"

                    cohort_comparison["prediction_drift"] = {
                        "analysis_type": "regression_prediction_drift",
                        "ks_statistic": float(ks_stat),
                        "ks_p_value": float(ks_pval),
                        "drift_severity": severity,
                        "reference_mean": float(ref_mean),
                        "reference_std": float(ref_std),
                        "current_mean": float(cohort_mean),
                        "current_std": float(cohort_std),
                        "mean_shift": float(mean_shift),
                        "std_shift": float(std_shift),
                        "interpretation": {
                            "level": severity,
                            "description": f"{'Major' if severity == 'High' else 'Moderate' if severity == 'Medium' else 'Minor'} shift in prediction distribution detected",
                            "recommended_action": (
                                "Investigate data drift"
                                if severity == "High"
                                else "Monitor closely" if severity == "Medium" else "Continue monitoring"
                            ),
                        },
                    }
                except Exception as e:
                    logger.warning(f"Regression prediction drift calculation failed for {cohort_label}: {e}")

            # ===== 2. Effect Size Analysis =====
            try:
                # Extract primary metric for effect size (accuracy for classification, r2 for regression)
                primary_metric = "accuracy" if is_classification else "r2"
                ref_metric_value = reference_metrics.get(primary_metric)
                cohort_metric_value = cohort_data.get("metrics", {}).get(primary_metric)

                if ref_metric_value is not None and cohort_metric_value is not None:
                    effect_size_result = effect_size_service.comprehensive_effect_analysis(
                        np.array([ref_metric_value]),  # Reference as array
                        np.array([cohort_metric_value]),  # Cohort as array
                        group1_name="Reference",
                        group2_name=cohort_label,
                    )
                    cohort_comparison["effect_size_analysis"] = effect_size_result
            except Exception as e:
                logger.warning(f"Effect size calculation failed for {cohort_label}: {e}")

            # ===== 3. Metrics Differences with Drift Analysis =====
            try:
                cohort_metrics = cohort_data.get("metrics", {})
                metrics_differences = {}

                for metric_name in reference_metrics.keys():
                    if metric_name in ["confusion_matrix", "classification_report"]:
                        continue  # Skip complex nested metrics

                    ref_value = reference_metrics.get(metric_name)
                    curr_value = cohort_metrics.get(metric_name)

                    # Handle numeric values only
                    if isinstance(ref_value, (int, float)) and isinstance(curr_value, (int, float)):
                        abs_diff = curr_value - ref_value
                        rel_diff = (abs_diff / ref_value * 100) if ref_value != 0 else 0

                        metrics_differences[metric_name] = {
                            "reference": float(ref_value),
                            "current": float(curr_value),
                            "absolute_difference": round(abs_diff, 6),
                            "relative_difference": round(rel_diff, 2),
                            "drift_magnitude": abs(abs_diff),
                        }

                # Calculate drift summary
                if metrics_differences:
                    relative_drifts = [
                        abs(m["relative_difference"]) for m in metrics_differences.values() if m["reference"] != 0
                    ]

                    if relative_drifts:
                        avg_drift = np.mean(relative_drifts)
                        max_drift = np.max(relative_drifts)

                        # Determine severity
                        if max_drift > 25:
                            severity = "High"
                        elif max_drift > 10:
                            severity = "Medium"
                        else:
                            severity = "Low"

                        metrics_differences["drift_summary"] = {
                            "average_relative_drift": round(avg_drift, 2),
                            "maximum_relative_drift": round(max_drift, 2),
                            "drift_severity": severity,
                            "metrics_analyzed": len(metrics_differences) - 1,  # Exclude drift_summary itself
                        }

                cohort_comparison["metrics_differences"] = metrics_differences
            except Exception as e:
                logger.warning(f"Metrics differences calculation failed for {cohort_label}: {e}")

            # ===== 4. Degradation Analysis =====
            try:
                degradation_result = metrics_calculation_service.performance_degradation_analysis(
                    reference_metrics, cohort_metrics
                )
                cohort_comparison["degradation_analysis"] = degradation_result
            except Exception as e:
                logger.warning(f"Degradation analysis failed for {cohort_label}: {e}")

            # ===== 5. Detailed Metric Comparison Table =====
            try:
                table_data = []

                # Determine which metrics are "higher is better"
                higher_is_better_metrics = {
                    # Classification metrics
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc",
                    "average_precision",
                    "sensitivity",
                    "specificity",
                    "positive_predictive_value",
                    "negative_predictive_value",
                    "matthews_correlation",
                    "true_positives",
                    "true_negatives",
                    # Regression metrics (higher is better)
                    "r2",
                    "r2_score",
                    "explained_variance",
                }
                lower_is_better_metrics = {
                    # Classification metrics
                    "log_loss",
                    "false_positives",
                    "false_negatives",
                    # Regression metrics (lower is better)
                    "mse",
                    "rmse",
                    "mae",
                    "mape",
                    "mean_squared_error",
                    "mean_absolute_error",
                    "root_mean_squared_error",
                    "mean_absolute_percentage_error",
                    "median_absolute_error",
                }

                for metric_name in reference_metrics.keys():
                    if metric_name in ["confusion_matrix", "classification_report"]:
                        continue

                    ref_value = reference_metrics.get(metric_name)
                    curr_value = cohort_metrics.get(metric_name)

                    if isinstance(ref_value, (int, float)) and isinstance(curr_value, (int, float)):
                        abs_change = curr_value - ref_value
                        pct_change = (abs_change / ref_value * 100) if ref_value != 0 else 0

                        # Determine if higher is better
                        higher_better = (
                            1
                            if metric_name in higher_is_better_metrics
                            else (0 if metric_name in lower_is_better_metrics else 1)
                        )

                        # Determine improvement status
                        if higher_better:
                            improved = abs_change > 0
                        else:
                            improved = abs_change < 0

                        # Determine magnitude
                        abs_pct = abs(pct_change)
                        if abs_pct > 20:
                            magnitude = "Large"
                        elif abs_pct > 10:
                            magnitude = "Medium"
                        elif abs_pct > 3:
                            magnitude = "Small"
                        else:
                            magnitude = "Minimal"

                        table_data.append(
                            {
                                "metric": metric_name.replace("_", " ").title(),
                                "reference_value": float(ref_value),
                                "current_value": float(curr_value),
                                "absolute_change": round(abs_change, 6),
                                "percentage_change": round(pct_change, 2),
                                "change_direction": "increase" if abs_change > 0 else "decrease",
                                "improvement_status": "Improved" if improved else "Degraded",
                                "change_magnitude": magnitude,
                                "higher_is_better": higher_better,
                            }
                        )

                # Sort by absolute percentage change (descending)
                table_data.sort(key=lambda x: abs(x["percentage_change"]), reverse=True)

                cohort_comparison["detailed_metric_comparison"] = {
                    "table_data": table_data,
                    "summary": {
                        "total_metrics": len(table_data),
                        "improved_metrics": sum(1 for m in table_data if m["improvement_status"] == "Improved"),
                        "degraded_metrics": sum(1 for m in table_data if m["improvement_status"] == "Degraded"),
                        "unchanged_metrics": 0,
                    },
                }
            except Exception as e:
                logger.warning(f"Detailed metric comparison failed for {cohort_label}: {e}")

            # ===== 6. Statistical Tests =====
            if len(y_ref) == len(y_cohort):
                try:
                    stat_tests = statistical_significance_service.analyze_statistical_significance(
                        y_true=y_ref,
                        pred_ref=pred_ref,
                        pred_curr=pred_cohort,
                        pred_ref_proba=pred_ref_proba,
                        pred_curr_proba=pred_cohort_proba,
                        X=X_ref,
                        model_ref=model,
                        model_curr=model,
                    )
                    cohort_comparison["statistical_tests"] = stat_tests
                except Exception as e:
                    logger.warning(f"Statistical tests failed for {cohort_label}: {e}")
                    cohort_comparison["statistical_tests"] = {
                        "skipped": True,
                        "reason": f"Statistical tests failed: {str(e)}",
                    }
            else:
                cohort_comparison["statistical_tests"] = {
                    "skipped": True,
                    "reason": "Different dataset sizes - statistical comparison not applicable",
                    "reference_size": len(y_ref),
                    "current_size": len(y_cohort),
                }

        except Exception as e:
            logger.error(f"Enhancement failed for cohort {cohort_label}: {e}")
            cohort_comparison["error"] = str(e)

        enhancement["cohort_comparisons"].append(cohort_comparison)

    # ===== 7. Enhanced Overall Assessment =====
    try:
        # Analyze last cohort (most recent) for overall assessment
        if enhancement["cohort_comparisons"]:
            last_comparison = enhancement["cohort_comparisons"][-1]

            # Determine drift detection
            drift_detected = False
            drift_severity = "Low"

            # Check PSI
            psi_result = last_comparison.get("prediction_drift", {})
            if psi_result:
                psi_level = psi_result.get("interpretation", {}).get("level", "Low")
                if psi_level in ["High", "Major"]:
                    drift_detected = True
                    drift_severity = "High"
                elif psi_level == "Medium":
                    drift_severity = "Medium"

            # Check metrics drift
            metrics_diff = last_comparison.get("metrics_differences", {})
            drift_sum = metrics_diff.get("drift_summary", {})
            if drift_sum:
                metric_severity = drift_sum.get("drift_severity", "Low")
                if metric_severity == "High":
                    drift_detected = True
                    drift_severity = "High"
                elif metric_severity == "Medium" and drift_severity == "Low":
                    drift_severity = "Medium"

            # Check degradation
            deg_analysis = last_comparison.get("degradation_analysis", {})
            deg_summary = deg_analysis.get("summary", {})
            critical_count = deg_summary.get("critical_degradations", 0)
            if critical_count > 0:
                drift_detected = True
                if critical_count >= 3:
                    drift_severity = "High"

            # Generate key findings
            key_findings = []
            if psi_result:
                psi_value = psi_result.get("psi", 0)
                psi_interp = psi_result.get("interpretation", {}).get("description", "")
                key_findings.append(f"Prediction drift PSI: {psi_value:.3f} - {psi_interp}")

            if drift_sum:
                avg_drift = drift_sum.get("average_relative_drift", 0)
                key_findings.append(f"Average metric drift: {avg_drift:.1f}%")

            if deg_summary:
                overall_status = deg_summary.get("overall_status", "Unknown")
                key_findings.append(f"Performance status: {overall_status}")

            # Generate recommendations
            recommendations = []
            if drift_severity == "High":
                recommendations.extend(
                    [
                        "Immediate model retraining recommended",
                        "Investigate root causes of performance degradation",
                        "Consider emergency rollback if business impact is severe",
                    ]
                )
            elif drift_severity == "Medium":
                recommendations.extend(
                    [
                        "Schedule model retraining in next cycle",
                        "Monitor performance closely",
                        "Investigate data distribution changes",
                    ]
                )
            else:
                recommendations.extend(["Continue regular monitoring", "Performance is stable"])

            # Build delta summary
            detailed_comparison = last_comparison.get("detailed_metric_comparison", {})
            table_data = detailed_comparison.get("table_data", [])

            significant_changes = []
            largest_improvement = None
            largest_degradation = None

            for metric in table_data[:5]:  # Top 5 by magnitude
                if metric["improvement_status"] == "Degraded":
                    significant_changes.append(
                        {"metric": metric["metric"], "change": f"{metric['percentage_change']:.1f}%", "improved": False}
                    )
                    # Compare numeric values, not formatted strings
                    if largest_degradation is None or abs(metric["percentage_change"]) > abs(
                        float(largest_degradation.get("degradation_value", 0))
                    ):
                        largest_degradation = {
                            "metric": metric["metric"],
                            "degradation": f"{abs(metric['percentage_change']):.1f}%",
                            "degradation_value": metric["percentage_change"],  # Store numeric value for comparison
                        }
                elif metric["improvement_status"] == "Improved":
                    if largest_improvement is None or metric["percentage_change"] > float(
                        largest_improvement.get("improvement", "0%").rstrip("%")
                    ):
                        largest_improvement = {
                            "metric": metric["metric"],
                            "improvement": f"{metric['percentage_change']:.1f}%",
                        }

            enhancement["overall_assessment"] = {
                "drift_detected": int(drift_detected),
                "drift_severity": drift_severity,
                "key_findings": key_findings,
                "recommendations": recommendations,
                "risk_level": drift_severity,
                "delta_summary": {
                    "significant_changes": significant_changes,
                    "largest_improvement": largest_improvement,
                    "largest_degradation": largest_degradation,
                },
            }
    except Exception as e:
        logger.warning(f"Overall assessment enhancement failed: {e}")

    return enhancement


# ===== PERFORMANCE COMPARISON ENDPOINT =====


@router.post(
    "/model-drift/performance-comparison",
    summary="S3-based Performance Comparison Analysis (Single or Multi-Cohort)",
    description="Loads data directly from S3 URLs and performs model performance comparison analysis. Supports both single cohort comparison and multi-cohort time-series analysis.",
)
async def s3_model_drift_performance_comparison(
    request: ModelDriftAnalysisRequest, user: dict = Depends(get_current_user)
):
    """
    S3-based Performance Comparison Analysis - supports both single and multi-cohort analysis

    Single Cohort Mode:
    - Provide: reference_url, cohort_urls with 1 URL, model_url
    - Returns: Traditional comparison between reference and cohort

    Multi-Cohort Mode (Time-Series Analysis):
    - Provide: reference_url, cohort_urls with 2+ URLs, model_url
    - Optional: cohort_labels for custom naming (auto-generated if not provided)
    - Returns: Performance over time with metric trends

    Model type (classification/regression) is auto-detected from the model and data.
    """
    try:
        # Load model first (shared for all modes)
        wrapped_model = load_s3_model(request.model_url)
        model, model_metadata = extract_model_from_wrapper(wrapped_model)

        # Load reference dataset (shared for all modes)
        reference_df = load_s3_csv(request.reference_url)
        validate_dataframe(reference_df, "Reference")

        # Determine target column (auto-detect from model if not provided)
        target_column = request.target_column or model_metadata.get("target_column") or reference_df.columns[-1]

        # Determine analysis mode based on number of cohort URLs
        num_cohorts = len(request.cohort_urls)
        is_multi_cohort = num_cohorts > 1

        logger.info(
            f"Analysis mode: {'Multi-cohort' if is_multi_cohort else 'Single cohort'} ({num_cohorts} cohort(s))"
        )

        # ===== MULTI-COHORT MODE =====
        if is_multi_cohort:
            logger.info(f"Multi-cohort mode: Analyzing {num_cohorts} cohorts")

            # Generate cohort labels if not provided (auto-detection)
            if request.cohort_labels:
                cohort_labels = request.cohort_labels
                logger.info(f"Using provided cohort labels: {cohort_labels}")
            else:
                cohort_labels = [f"cohort_{i+1}" for i in range(num_cohorts)]
                logger.info(f"Auto-generated cohort labels: {cohort_labels}")

            # Evaluate reference dataset first
            logger.info("Evaluating reference dataset...")
            reference_performance = await _evaluate_single_cohort(
                cohort_df=reference_df,
                reference_df=reference_df,
                model=model,
                model_metadata=model_metadata,
                target_column=target_column,
                cohort_label="reference",
            )

            # Initialize performance over time with reference
            performance_over_time = [reference_performance]

            # Evaluate each cohort
            for idx, cohort_url in enumerate(request.cohort_urls):
                cohort_label = cohort_labels[idx]
                logger.info(f"Evaluating cohort '{cohort_label}' from {cohort_url}...")

                try:
                    # Load cohort dataset
                    cohort_df = load_s3_csv(cohort_url)

                    # Evaluate cohort
                    cohort_performance = await _evaluate_single_cohort(
                        cohort_df=cohort_df,
                        reference_df=reference_df,
                        model=model,
                        model_metadata=model_metadata,
                        target_column=target_column,
                        cohort_label=cohort_label,
                    )

                    performance_over_time.append(cohort_performance)

                except Exception as e:
                    logger.error(f"Failed to process cohort '{cohort_label}': {e}")
                    # Add error entry but continue with other cohorts
                    performance_over_time.append(
                        {"cohort": cohort_label, "error": str(e), "dataset_size": 0, "metrics": {}}
                    )

            # Generate metric trends
            metric_trends = generate_metric_trends(performance_over_time)

            # Create cohort summary
            cohort_summary = create_cohort_summary(performance_over_time, metric_trends)

            # Extract detected model type from first cohort evaluation
            detected_model_type = (
                performance_over_time[0].get("model_type", "unknown") if performance_over_time else "unknown"
            )

            # ===== ENHANCED MULTI-COHORT ANALYSIS =====
            # Add comprehensive drift detection components (PSI, effect sizes, degradation analysis, etc.)
            logger.info("Enhancing multi-cohort analysis with comprehensive drift detection...")
            try:
                # Load all cohort dataframes for enhancement
                cohort_dataframes = []
                for cohort_url in request.cohort_urls:
                    cohort_df = load_s3_csv(cohort_url)
                    cohort_dataframes.append(cohort_df)

                enhanced_analysis = await enhance_multi_cohort_analysis(
                    performance_over_time=performance_over_time,
                    metric_trends=metric_trends,
                    cohort_summary=cohort_summary,
                    reference_df=reference_df,
                    model=model,
                    cohort_dfs=cohort_dataframes,
                    cohort_labels=cohort_labels,
                    model_metadata=model_metadata,
                    target_column=target_column,
                )
                logger.info(
                    f"Enhancement completed: Added {len(enhanced_analysis.get('cohort_comparisons', []))} detailed cohort comparisons"
                )
            except Exception as e:
                logger.warning(f"Multi-cohort enhancement failed (continuing with basic analysis): {e}")
                enhanced_analysis = {"enhancement_error": str(e), "cohort_comparisons": []}

            # Build multi-cohort response
            result = {
                "is_multi_cohort": True,
                "performance_over_time": performance_over_time,
                "metric_trends": metric_trends,
                "cohort_summary": cohort_summary,
                "cohort_comparisons": enhanced_analysis.get("cohort_comparisons", []),  # NEW: Detailed comparisons
                "overall_assessment": enhanced_analysis.get("overall_assessment", {}),  # NEW: Enhanced assessment
                "analysis_metadata": {
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "analysis_type": "multi_cohort_time_series",
                    "num_cohorts": num_cohorts,
                    "cohort_labels": cohort_labels,
                    "reference_dataset_size": len(reference_df),
                    "model_type": detected_model_type,
                    "feature_columns": model_metadata.get("feature_columns", []),
                    "target_column": target_column,
                },
            }

            # Serialize response
            serialized_result = serialize_response(result)

            # Generate AI explanation for multi-cohort
            try:
                ai_summary_payload = {
                    "analysis_type": "multi_cohort_performance",
                    "num_cohorts": len(performance_over_time),
                    "cohort_summary": cohort_summary,
                    "metric_trends": metric_trends,
                    "performance_over_time": performance_over_time[:3],  # Send first 3 for context
                    "overall_assessment": enhanced_analysis.get(
                        "overall_assessment", {}
                    ),  # NEW: Include enhanced assessment
                    "cohort_comparisons": enhanced_analysis.get("cohort_comparisons", [])[
                        :2
                    ],  # NEW: Send first 2 comparisons
                }

                ai_explanation = ai_explanation_service.generate_explanation(
                    analysis_data=ai_summary_payload, analysis_type="model_performance"
                )
                serialized_result["llm_response"] = ai_explanation
            except Exception as e:
                logger.warning(f"AI explanation failed: {e}")

                # Generate fallback explanation based on enhanced assessment
                fallback_key_findings = []
                if enhanced_analysis.get("overall_assessment"):
                    overall = enhanced_analysis["overall_assessment"]
                    fallback_key_findings.extend(overall.get("key_findings", []))
                else:
                    fallback_key_findings.extend(
                        [
                            "AI explanation service temporarily unavailable",
                            f"Analyzed {len(performance_over_time)} cohorts including reference",
                            f"Overall trend: {cohort_summary.get('overall_trend', 'unknown')}",
                        ]
                    )

                serialized_result["llm_response"] = {
                    "summary": f"AI explanation service failed. Multi-cohort performance analysis completed for {len(performance_over_time)} cohorts.",
                    "detailed_explanation": f"AI explanation generation encountered an error: {str(e)}. However, comprehensive performance analysis including PSI drift, effect sizes, and degradation analysis has been completed across {len(performance_over_time)} time periods. Review cohort_comparisons for detailed insights on each cohort.",
                    "key_takeaways": fallback_key_findings,
                    "error": str(e),
                }

            return serialized_result

        # ===== SINGLE COHORT MODE (Backward Compatible) =====
        else:
            logger.info("Single cohort mode: Traditional reference vs cohort comparison")

            # Load the single cohort dataset
            current_url = request.cohort_urls[0]
            current_df = load_s3_csv(current_url)
            validate_dataframe(current_df, "Current")

            # Prepare data using enhanced preprocessing for consistent processing
            try:
                X_ref, y_true_ref, X_curr, y_true_curr, preprocessing_metadata = apply_consistent_preprocessing(
                    reference_df,
                    current_df,
                    model_metadata.get("feature_columns"),
                    target_column,
                    model_metadata.get("preprocessing_info"),
                )
            except Exception as e:
                logger.error(f"Data preparation failed: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Data preparation failed: {str(e)}. Please ensure your data is properly formatted and matches the model's expected input.",
                )

            # Generate predictions with error handling
            try:
                pred_ref = model.predict(X_ref)
                pred_curr = model.predict(X_curr)
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Model prediction failed: {str(e)}. This might be due to data format mismatch or categorical variables.",
                )

            # Get prediction probabilities if available
            pred_ref_proba = None
            pred_curr_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    pred_ref_proba = model.predict_proba(X_ref)
                    pred_curr_proba = model.predict_proba(X_curr)
                except Exception as e:
                    logger.warning(f"Could not get prediction probabilities: {e}")

            # Run performance comparison analysis (model type auto-detected)
            result = performance_comparison_service.analyze_performance_comparison(
                y_true=y_true_ref,  # DEPRECATED - kept for backward compatibility
                pred_ref=pred_ref,
                pred_curr=pred_curr,
                pred_ref_proba=pred_ref_proba,
                pred_curr_proba=pred_curr_proba,
                X=X_ref,  # Pass reference features
                model_ref=model,
                model_curr=model,
                y_true_ref=y_true_ref,  # NEW: Separate reference ground truth
                y_true_curr=y_true_curr,  # NEW: Separate current ground truth
            )

            # Add metadata - use detected model type from analysis result
            result["is_multi_cohort"] = False
            result["analysis_metadata"] = {
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "analysis_type": "single_cohort_comparison",
                "reference_dataset_size": len(reference_df),
                "current_dataset_size": len(current_df),
                "model_type": result.get("model_type", "unknown"),
                "feature_columns": model_metadata.get("feature_columns", []),
                "target_column": target_column,
            }

            # Serialize response
            serialized_result = serialize_response(result)

            # Generate AI explanation
            try:
                ai_summary_payload = create_ai_summary_for_performance_comparison(serialized_result)
                ai_explanation = ai_explanation_service.generate_explanation(
                    analysis_data=ai_summary_payload, analysis_type="model_performance"
                )
                serialized_result["llm_response"] = ai_explanation
            except Exception as e:
                logger.warning(f"AI explanation failed: {e}")
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
        logger.error(f"Model drift performance comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance comparison analysis failed: {str(e)}")


@router.post(
    "/model-drift/degradation-metrics",
    summary="S3-based Degradation Metrics Analysis",
    description="Loads data directly from S3 URLs and performs model degradation metrics analysis",
)
async def s3_model_drift_degradation_metrics(
    request: ModelDriftAnalysisRequest, user: dict = Depends(get_current_user)
):
    """S3-based Degradation Metrics Analysis - loads data directly from S3 URLs"""
    try:
        # Load data and model directly from S3 URLs
        reference_df = load_s3_csv(request.reference_url)
        current_df = load_s3_csv(request.current_url)
        wrapped_model = load_s3_model(request.model_url)

        # Extract model and metadata from wrapper
        model, model_metadata = extract_model_from_wrapper(wrapped_model)

        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")

        # Determine target column (use from model metadata if available, then request, then last column)
        target_column = model_metadata.get("target_column") or request.target_column or reference_df.columns[-1]

        # Prepare data using enhanced preprocessing for consistent processing
        try:
            X_ref, y_true_ref, X_curr, y_true_curr, preprocessing_metadata = apply_consistent_preprocessing(
                reference_df,
                current_df,
                model_metadata.get("feature_columns"),
                target_column,
                model_metadata.get("preprocessing_info"),
            )
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Data preparation failed: {str(e)}. Please ensure your data is properly formatted and matches the model's expected input.",
            )

        # Generate predictions with enhanced error handling and format consistency
        try:
            # Get binary predictions
            pred_ref = model.predict(X_ref)
            pred_curr = model.predict(X_curr)

            # Ensure predictions are in consistent format
            pred_ref = np.asarray(pred_ref).flatten()
            pred_curr = np.asarray(pred_curr).flatten()

            logger.info(f"Binary predictions generated. Shapes: ref={pred_ref.shape}, curr={pred_curr.shape}")

        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Model prediction failed: {str(e)}. This might be due to data format mismatch or categorical variables.",
            )

        # Get prediction probabilities with enhanced handling
        pred_ref_proba = None
        pred_curr_proba = None
        if hasattr(model, "predict_proba"):
            try:
                pred_ref_proba_raw = model.predict_proba(X_ref)
                pred_curr_proba_raw = model.predict_proba(X_curr)

                # Handle different probability formats
                if len(pred_ref_proba_raw.shape) > 1 and pred_ref_proba_raw.shape[1] > 1:
                    if pred_ref_proba_raw.shape[1] == 2:
                        # Binary classification - use positive class probability
                        pred_ref_proba = pred_ref_proba_raw[:, 1]
                        pred_curr_proba = pred_curr_proba_raw[:, 1]
                        logger.info(f"Binary classification detected - using positive class probabilities")
                    else:
                        # Multi-class - use maximum probability for confidence analysis
                        pred_ref_proba = np.max(pred_ref_proba_raw, axis=1)
                        pred_curr_proba = np.max(pred_curr_proba_raw, axis=1)
                        logger.info(f"Multi-class classification detected - using max probabilities")
                else:
                    # Single probability per prediction
                    pred_ref_proba = pred_ref_proba_raw.flatten()
                    pred_curr_proba = pred_curr_proba_raw.flatten()

                # Ensure probabilities are in valid range [0, 1]
                pred_ref_proba = np.clip(pred_ref_proba, 0, 1)
                pred_curr_proba = np.clip(pred_curr_proba, 0, 1)

                # Validate probability quality
                ref_min, ref_max = np.min(pred_ref_proba), np.max(pred_ref_proba)
                curr_min, curr_max = np.min(pred_curr_proba), np.max(pred_curr_proba)

                logger.info(
                    f"Probability ranges: ref=[{ref_min:.6f}, {ref_max:.6f}], curr=[{curr_min:.6f}, {curr_max:.6f}]"
                )

                # Check for extreme probability distributions
                if ref_max - ref_min < 0.01 or curr_max - curr_min < 0.01:
                    logger.warning("Detected very narrow probability distribution - model may not be well calibrated")

                if ref_min < 1e-10 or curr_min < 1e-10:
                    logger.warning("Detected extremely small probabilities - may cause numerical issues")
                    # Set minimum threshold to avoid numerical problems
                    pred_ref_proba = np.maximum(pred_ref_proba, 1e-10)
                    pred_curr_proba = np.maximum(pred_curr_proba, 1e-10)

            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
                pred_ref_proba = None
                pred_curr_proba = None

        # Enhanced logging before analysis
        logger.info(f"Starting degradation metrics analysis")
        logger.info(f"Data shapes - X_ref: {X_ref.shape}, X_curr: {X_curr.shape}")
        logger.info(f"Prediction shapes - pred_ref: {pred_ref.shape}, pred_curr: {pred_curr.shape}")
        if pred_ref_proba is not None:
            logger.info(
                f"Probability shapes - pred_ref_proba: {pred_ref_proba.shape}, pred_curr_proba: {pred_curr_proba.shape}"
            )
        else:
            logger.warning("No prediction probabilities available - some analyses may be limited")

        # Validate data before analysis
        if len(pred_ref) == 0 or len(pred_curr) == 0:
            raise HTTPException(status_code=400, detail="Empty prediction arrays detected")

        if y_true_ref is None or len(y_true_ref) == 0:
            raise HTTPException(status_code=400, detail="No ground truth labels available for analysis")

        # Ensure we have ground truth for analysis
        if y_true_curr is None:
            logger.warning("No current ground truth available, using reference ground truth")
            y_true_curr = y_true_ref

        # Get analysis configuration
        config = request.analysis_config or {}

        # Run degradation metrics analysis with enhanced error capture
        # Note: For S3 analysis, we compare the same model's performance on different datasets
        # rather than comparing two different models
        result = degradation_metrics_service.analyze_degradation_metrics(
            y_true=y_true_ref,  # Use reference ground truth for evaluation
            pred_ref=pred_ref,
            pred_curr=pred_curr,
            pred_ref_proba=pred_ref_proba,
            pred_curr_proba=pred_curr_proba,
            X_ref=X_ref,
            y_ref=y_true_ref,
            X_curr=X_curr,
            y_curr=y_true_curr,
            model_ref=model,
            model_curr=model,  # Same model, different datasets
            feature_names=list(X_ref.columns) if hasattr(X_ref, "columns") else None,
        )

        # Log analysis result structure
        logger.info(f"Analysis result keys: {list(result.keys())}")
        if "sub_tabs" in result:
            logger.info(f"Sub-tabs keys: {list(result['sub_tabs'].keys())}")
            for tab_name, tab_data in result["sub_tabs"].items():
                if isinstance(tab_data, dict) and "error" in tab_data:
                    logger.error(f"Error in {tab_name}: {tab_data['error']}")
                elif isinstance(tab_data, dict):
                    logger.info(f"{tab_name} completed successfully with {len(tab_data)} keys")

        # Add metadata including model wrapper info
        result["analysis_metadata"] = {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_dataset_size": len(reference_df),
            "current_dataset_size": len(current_df),
            "model_type": model_metadata.get("model_type", str(type(model).__name__)),
            "model_wrapper_version": model_metadata.get("wrapper_version"),
            "feature_columns": model_metadata.get("feature_columns", []),
            "target_column": target_column,
            "analysis_config": config,
            "data_sources": {
                "reference_url": request.reference_url,
                "current_url": request.current_url,
                "model_url": request.model_url,
            },
        }

        # Serialize response
        serialized_result = serialize_response(result)

        # Generate AI explanation with better error handling
        try:
            ai_summary_payload = create_ai_summary_for_degradation_metrics(serialized_result)

            # Debug: Log AI summary structure
            logger.info("=== DEGRADATION METRICS RESULT STRUCTURE ===")
            logger.info(f"Serialized result keys: {list(serialized_result.keys())}")
            logger.info(f"AI Summary payload keys: {list(ai_summary_payload.keys())}")
            logger.info(f"AI Summary non-empty values: {[(k, v) for k, v in ai_summary_payload.items() if v]}")

            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, analysis_type="degradation_metrics"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            logger.error(f"AI explanation failed: {e}")
            # Enhanced fallback response with actual results summary
            fallback_summary = []
            if "sub_tabs" in serialized_result:
                for tab_name, tab_data in serialized_result["sub_tabs"].items():
                    if isinstance(tab_data, dict) and "error" not in tab_data:
                        fallback_summary.append(f"{tab_name.replace('_', ' ').title()} analysis completed")
                    elif isinstance(tab_data, dict) and "error" in tab_data:
                        fallback_summary.append(
                            f"{tab_name.replace('_', ' ').title()} analysis failed: {tab_data.get('error', 'Unknown error')}"
                        )

            if not fallback_summary:
                fallback_summary = ["Degradation metrics analysis completed with limited data"]

            serialized_result["llm_response"] = {
                "summary": "Model degradation metrics analysis completed. Review detailed results below for comprehensive insights.",
                "detailed_explanation": f"Comprehensive degradation analysis has been completed. {' '.join(fallback_summary)}. AI explanations are temporarily unavailable, but detailed metrics are available in the analysis results.",
                "key_takeaways": fallback_summary
                + [
                    "Review model disagreement and confidence trends in detailed results",
                    "AI explanations will return when service is restored",
                ],
            }

        return serialized_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model drift degradation metrics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Degradation metrics analysis failed: {str(e)}")


@router.post(
    "/model-drift/statistical-significance",
    summary="S3-based Statistical Significance Analysis",
    description="Loads data directly from S3 URLs and performs statistical significance analysis",
)
async def s3_model_drift_statistical_significance(
    request: ModelDriftAnalysisRequest, user: dict = Depends(get_current_user)
):
    """S3-based Statistical Significance Analysis - loads data directly from S3 URLs"""
    try:
        # Load data and model directly from S3 URLs
        reference_df = load_s3_csv(request.reference_url)
        current_df = load_s3_csv(request.current_url)
        wrapped_model = load_s3_model(request.model_url)

        # Extract model and metadata from wrapper
        model, model_metadata = extract_model_from_wrapper(wrapped_model)

        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")

        # Determine target column (use from model metadata if available, then request, then last column)
        target_column = model_metadata.get("target_column") or request.target_column or reference_df.columns[-1]

        # Prepare data using enhanced preprocessing for consistent processing
        try:
            X_ref, y_true_ref, X_curr, y_true_curr, preprocessing_metadata = apply_consistent_preprocessing(
                reference_df,
                current_df,
                model_metadata.get("feature_columns"),
                target_column,
                model_metadata.get("preprocessing_info"),
            )
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Data preparation failed: {str(e)}. Please ensure your data is properly formatted and matches the model's expected input.",
            )

        # Generate predictions with error handling
        try:
            pred_ref = model.predict(X_ref)
            pred_curr = model.predict(X_curr)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Model prediction failed: {str(e)}. This might be due to data format mismatch or categorical variables.",
            )

        # Get prediction probabilities if available
        pred_ref_proba = None
        pred_curr_proba = None
        if hasattr(model, "predict_proba"):
            try:
                pred_ref_proba = model.predict_proba(X_ref)
                pred_curr_proba = model.predict_proba(X_curr)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")

        # Get analysis configuration
        config = request.analysis_config or {}

        # Run statistical significance analysis
        result = statistical_significance_service.analyze_statistical_significance(
            y_true=y_true_ref,  # Use reference ground truth
            pred_ref=pred_ref,
            pred_curr=pred_curr,
            pred_ref_proba=pred_ref_proba,
            pred_curr_proba=pred_curr_proba,
            X=X_ref,
            model_ref=model,
            model_curr=model,
            alpha=config.get("alpha", 0.05),
        )

        # Add metadata including model wrapper info
        result["analysis_metadata"] = {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_dataset_size": len(reference_df),
            "current_dataset_size": len(current_df),
            "model_type": model_metadata.get("model_type", str(type(model).__name__)),
            "model_wrapper_version": model_metadata.get("wrapper_version"),
            "feature_columns": model_metadata.get("feature_columns", []),
            "target_column": target_column,
            "analysis_config": config,
            "data_sources": {
                "reference_url": request.reference_url,
                "current_url": request.current_url,
                "model_url": request.model_url,
            },
        }

        # Serialize response
        serialized_result = serialize_response(result)

        # Generate AI explanation
        try:
            ai_summary_payload = create_ai_summary_for_statistical_significance(serialized_result)
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, analysis_type="statistical_significance"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            logger.warning(f"AI explanation failed: {e}")
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
        logger.error(f"Model drift statistical significance failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistical significance analysis failed: {str(e)}")


@router.get(
    "/model-drift/metrics",
    summary="Get Available Metrics",
    description="Get available performance metrics for each model type - S3 compatible",
)
async def get_available_metrics(user: dict = Depends(get_current_user)):
    """Get available performance metrics for each model type - S3 compatible"""
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


@router.get(
    "/model-drift/tests",
    summary="Get Available Statistical Tests",
    description="Get available statistical tests for each model type - S3 compatible",
)
async def get_available_tests(user: dict = Depends(get_current_user)):
    """Get available statistical tests for each model type - S3 compatible"""
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


@router.post(
    "/model-drift/sanity-check",
    summary="S3-based Sanity Check Analysis",
    description="Pre-flight diagnostic analysis that estimates expected model performance on new data by analyzing distributional similarity",
)
async def s3_model_drift_sanity_check(request: AnalysisRequest, user: dict = Depends(get_current_user)):
    """
    S3-based Sanity Check Analysis - Pre-flight diagnostic

    Loads data from S3 URLs and performs similarity analysis to estimate expected performance.
    Model is optional - if provided, feature importances will be used for weighting.
    """
    try:
        # Import sanity check service
        from services.data_drift.app.new.src.model_drift.services.analysis.sanity_check_service import (
            sanity_check_service,
        )

        # Load data from S3
        reference_df = load_s3_csv(request.reference_url)
        current_df = load_s3_csv(request.current_url)

        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")

        # Load model if provided (optional for sanity check)
        model = None
        model_metadata = {}
        if request.model_url:
            try:
                wrapped_model = load_s3_model(request.model_url)
                model, model_metadata = extract_model_from_wrapper(wrapped_model)
            except Exception as e:
                logger.warning(f"Could not load model (optional for sanity check): {e}")
                model = None

        # Determine target column
        target_column = (
            model_metadata.get("target_column") or request.target_column or None  # Target is optional for sanity check
        )

        # Get configuration
        config = request.config or {}
        model_type = config.get("model_type", "classification")

        # Get feature names (all columns except target)
        feature_names = list(reference_df.columns)
        if target_column and target_column in feature_names:
            feature_names.remove(target_column)

        # Run sanity check analysis
        result = sanity_check_service.analyze_sanity_check(
            reference_df=reference_df,
            current_df=current_df,
            target_column=target_column,
            model=model,
            feature_names=feature_names,
            model_type=model_type,
        )

        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=f"Sanity check analysis failed: {result['error']}")

        # Build response with metadata
        analysis_id = f"sanity_check_s3_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        response = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_info": {
                "model_type": model_metadata.get("model_type", model_type),
                "has_model": model is not None,
                "model_class": str(type(model).__name__) if model else None,
            },
            "sanity_check_summary": result["sanity_check_summary"],
            "scatter_points": result["scatter_points"],
            "distribution_comparisons": result["distribution_comparisons"],
            "recommendations": result["recommendations"],
            "warnings": result.get("warnings", []),
            "analysis_metadata": {
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "reference_dataset_size": len(reference_df),
                "current_dataset_size": len(current_df),
                "model_type": model_type,
                "target_column": target_column,
                "num_features": len(feature_names),
                "data_sources": {
                    "reference_url": request.reference_url,
                    "current_url": request.current_url,
                    "model_url": request.model_url,
                },
            },
        }

        # Serialize response
        serialized_result = serialize_response(response)

        # Generate AI explanation
        try:
            ai_summary_payload = create_ai_summary_for_sanity_check(serialized_result)
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, analysis_type="sanity_check"
            )
            serialized_result["llm_response"] = ai_explanation
        except Exception as e:
            logger.warning(f"AI explanation failed: {e}")
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
        logger.error(f"S3 sanity check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sanity check analysis failed: {str(e)}")


@router.post(
    "/model-drift/explain_with_ai",
    summary="S3-based AI Explanation for Model Drift Analysis",
    description="Loads data directly from S3 URLs, performs the specified analysis, and generates AI explanations",
)
async def s3_model_drift_explain_with_ai(request: AIExplanationRequest, user: dict = Depends(get_current_user)):
    """
    S3-based AI Explanation for Model Drift and Data Drift Analysis

    This endpoint loads data directly from S3, performs the specified analysis type,
    and generates business-friendly AI explanations without requiring session management.

    Model Drift Analysis Types (require model):
    - model_performance: Performance comparison analysis with AI explanation
    - degradation_metrics: Model degradation analysis with AI explanation
    - statistical_significance: Statistical significance testing with AI explanation
    - sanity_check: Pre-flight diagnostic analysis with AI explanation

    Data Drift Analysis Types (model optional):
    - class_imbalance: Class distribution analysis with AI explanation
    - statistical_analysis: Statistical reports and tests with AI explanation
    - feature_analysis: Individual feature drift analysis with AI explanation
    - data_drift_dashboard: Comprehensive data drift overview with AI explanation
    """
    try:
        logger.info(f"S3-based AI explanation for {request.analysis_type} analysis")

        # Load datasets from S3
        logger.info(f"Loading datasets - Reference: {request.reference_url}, Current: {request.current_url}")
        reference_df = load_s3_csv(request.reference_url)
        current_df = load_s3_csv(request.current_url)

        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")

        # Determine if this analysis requires a model
        model_required_types = {"model_performance", "degradation_metrics", "statistical_significance", "sanity_check"}
        requires_model = request.analysis_type in model_required_types

        # Load model if required or provided
        model = None
        model_metadata = {}
        if request.model_url or requires_model:
            if not request.model_url and requires_model:
                raise HTTPException(
                    status_code=400, detail=f"model_url is required for {request.analysis_type} analysis"
                )

            logger.info(f"Loading model from: {request.model_url}")
            wrapped_model = load_s3_model(request.model_url)
            model, model_metadata = extract_model_from_wrapper(wrapped_model)
            logger.info(f"Model loaded successfully: {model_metadata.get('model_type', 'Unknown')}")
        logger.info(f"Feature columns: {len(model_metadata.get('feature_columns', []))}")

        # Validate target column if provided
        target_column = request.target_column or model_metadata.get("target_column")
        if target_column:
            validate_target_column(reference_df, target_column, "reference")
            validate_target_column(current_df, target_column, "current")

        # Create configuration for analysis
        analysis_config = request.analysis_config or {}
        analysis_config.get("model_type", "classification")
        model_metadata.get("feature_columns", [])

        # Run the appropriate analysis based on analysis_type
        logger.info(f"Running {request.analysis_type} analysis via direct endpoint calls")

        if request.analysis_type == "model_performance":
            # Model drift analysis
            drift_request = ModelDriftAnalysisRequest(
                reference_url=request.reference_url,
                current_url=request.current_url,
                model_url=request.model_url,
                target_column=target_column,
                analysis_config=analysis_config,
            )
            logger.info("Calling direct performance-comparison endpoint")
            analysis_result = await s3_model_drift_performance_comparison(drift_request, user)
            llm_response = analysis_result.get("llm_response")

        elif request.analysis_type == "degradation_metrics":
            # Model drift analysis
            drift_request = ModelDriftAnalysisRequest(
                reference_url=request.reference_url,
                current_url=request.current_url,
                model_url=request.model_url,
                target_column=target_column,
                analysis_config=analysis_config,
            )
            logger.info("Calling direct degradation-metrics endpoint")
            analysis_result = await s3_model_drift_degradation_metrics(drift_request, user)
            llm_response = analysis_result.get("llm_response")

        elif request.analysis_type == "statistical_significance":
            # Model drift analysis
            drift_request = ModelDriftAnalysisRequest(
                reference_url=request.reference_url,
                current_url=request.current_url,
                model_url=request.model_url,
                target_column=target_column,
                analysis_config=analysis_config,
            )
            logger.info("Calling direct statistical-significance endpoint")
            analysis_result = await s3_model_drift_statistical_significance(drift_request, user)
            llm_response = analysis_result.get("llm_response")

        elif request.analysis_type == "sanity_check":
            # Model drift analysis
            logger.info("Calling direct sanity-check endpoint")
            # Convert to AnalysisRequest for sanity check
            sanity_request = AnalysisRequest(
                reference_url=request.reference_url,
                current_url=request.current_url,
                model_url=request.model_url,
                target_column=target_column,
                config=analysis_config,
            )
            logger.info("Calling direct sanity-check endpoint")
            analysis_result = await s3_model_drift_sanity_check(sanity_request, user)
            llm_response = analysis_result.get("llm_response")

        elif request.analysis_type == "class_imbalance":
            # Data drift analysis
            logger.info("Calling direct class-imbalance endpoint")
            # Import the endpoint function directly
            from services.data_drift.app.new.src.data_drift.routes.class_imbalance import class_imbalance_analysis_s3

            analysis_request = AnalysisRequest(
                reference_url=request.reference_url,
                current_url=request.current_url,
                model_url=request.model_url,
                target_column=target_column,
                config=analysis_config,
            )
            analysis_result = await class_imbalance_analysis_s3(analysis_request, user)
            llm_response = analysis_result.get("llm_response")

        elif request.analysis_type == "statistical_analysis":
            # Data drift analysis
            logger.info("Calling direct statistical-reports endpoint")
            from services.data_drift.app.new.src.data_drift.routes.statistical import get_statistical_reports

            analysis_request = AnalysisRequest(
                reference_url=request.reference_url,
                current_url=request.current_url,
                model_url=request.model_url,
                target_column=target_column,
                config=analysis_config,
            )
            analysis_result = await get_statistical_reports(analysis_request, user)
            llm_response = analysis_result.get("llm_response")

        elif request.analysis_type == "feature_analysis":
            # Data drift analysis
            logger.info("Calling direct feature-analysis endpoint")
            from services.data_drift.app.new.src.data_drift.routes.feature_analysis import get_feature_analysis

            analysis_request = AnalysisRequest(
                reference_url=request.reference_url,
                current_url=request.current_url,
                model_url=request.model_url,
                target_column=target_column,
                config=analysis_config,
            )
            analysis_result = await get_feature_analysis(analysis_request, user)
            llm_response = analysis_result.get("llm_response")

        elif request.analysis_type == "data_drift_dashboard":
            # Comprehensive data drift overview
            logger.info("Calling comprehensive data drift dashboard analysis")
            # Note: This would need to be implemented in the data drift services
            # For now, delegate to statistical analysis as the closest equivalent
            from services.data_drift.app.new.src.data_drift.routes.statistical import get_statistical_reports

            analysis_request = AnalysisRequest(
                reference_url=request.reference_url,
                current_url=request.current_url,
                model_url=request.model_url,
                target_column=target_column,
                config=analysis_config,
            )
            analysis_result = await get_statistical_reports(analysis_request, user)
            llm_response = analysis_result.get("llm_response")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported analysis type: {request.analysis_type}")

        # Extract LLM response from the analysis result
        if not llm_response:
            logger.warning("No LLM response found in analysis result, using fallback")
            llm_response = {
                "summary": f"{request.analysis_type.replace('_', ' ').title()} analysis completed successfully.",
                "detailed_explanation": f"The {request.analysis_type} analysis has been completed using S3 data sources.",
                "key_takeaways": [
                    f"{request.analysis_type.replace('_', ' ').title()} analysis completed successfully",
                    "Review the technical results for insights",
                ],
            }

        logger.info(f"AI explanation extracted successfully for {request.analysis_type}")

        # Return simplified response focused on AI explanation
        return {
            "success": True,
            "analysis_type": request.analysis_type,
            "llm_response": llm_response,
            "note": f"Generated by calling direct {request.analysis_type} endpoint",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"S3-based AI explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI explanation generation failed: {str(e)}")
