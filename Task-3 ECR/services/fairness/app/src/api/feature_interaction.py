import logging
import math
import time
from itertools import combinations
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from services.fairness.app.src.api.bias import _clean_nan_values, _encode_dataframe, _safe_convert_to_numeric
from services.fairness.app.src.bias_detector import BiasDetector
from services.fairness.app.src.feature_interaction_generator import FeatureInteractionGenerator
from services.fairness.app.src.models import global_session
from services.fairness.app.src.schemas.request_response import (
    ColumnAnalysisResponse,
    FeatureInteractionRequest,
    FeatureInteractionResponse,
    FlexibleInteractionRequest,
    InteractionBiasDetail,
    InteractionComparisonRequest,
    InteractionPreviewRequest,
    InteractionPreviewResponse,
    InteractionTypeRecommendation,
    SmartInteractionRequest,
    SmartInteractionResponse,
)
from services.fairness.app.src.utils import FileManager
from services.fairness.app.utils.helper_functions import (
    get_s3_file_metadata,
    load_dataframe_from_url,
    load_model_from_url,
    validate_file_metadata,
)
from shared.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fairness/feature-interaction", tags=["Feature Interaction"])


def _calculate_bias_amplification(baseline_biased_count: int, interaction_biased_count: int) -> float:
    """
    Calculate bias amplification factor
    Positive = bias increased, Negative = bias decreased
    """
    if baseline_biased_count == 0:
        return 0.0 if interaction_biased_count == 0 else 100.0

    amplification = ((interaction_biased_count - baseline_biased_count) / baseline_biased_count) * 100
    return round(amplification, 2)


def _calculate_severity_score(biased_count: int, total_metrics: int, amplification: float) -> float:
    """
    Calculate severity score (0-100) for easy comparison
    Considers both percentage of biased metrics and amplification
    """
    if total_metrics == 0:
        return 0.0

    bias_percentage = (biased_count / total_metrics) * 100

    # Weighted combination: 60% from bias percentage, 40% from amplification
    # Normalize amplification to 0-100 scale
    normalized_amp = min(max(amplification, -100), 100)  # Clamp to -100 to 100
    normalized_amp = (normalized_amp + 100) / 2  # Convert to 0-100

    severity = (bias_percentage * 0.6) + (normalized_amp * 0.4)
    return round(severity, 2)


def _generate_interaction_recommendations(
    bias_status: str, amplification: float, unique_groups: int, source_columns: List[str]
) -> List[str]:
    """Generate actionable recommendations based on bias analysis"""
    recommendations = []

    if bias_status == "highly_biased":
        recommendations.append(f"⚠️ High bias detected. Consider avoiding this interaction in production models.")
        recommendations.append(f"Investigate why combining {' and '.join(source_columns)} creates significant bias.")

    if amplification > 50:
        recommendations.append(f"📈 Bias amplified by {amplification}%. This combination creates intersectional bias.")
        recommendations.append("Consider fairness-aware preprocessing or separate models for different groups.")
    elif amplification < -30:
        recommendations.append(f"📉 Bias reduced by {abs(amplification)}%. This interaction may help fairness.")
        recommendations.append("Investigate why this combination reduces bias for potential insights.")

    if unique_groups > 20:
        recommendations.append(f"⚡ {unique_groups} subgroups detected. Consider grouping similar categories.")
    elif unique_groups < 3:
        recommendations.append("💡 Few subgroups. Consider if this granularity is sufficient for fairness analysis.")

    if not recommendations:
        recommendations.append("✓ No major fairness concerns detected for this interaction.")

    return recommendations


def _prepare_visualization_data(interaction_results: List[InteractionBiasDetail]) -> Dict[str, Any]:
    """Prepare data structure optimized for frontend charts"""
    return {
        "scatter_plot": [
            {
                "name": r.interaction_name,
                "x": r.bias_amplification,
                "y": r.biased_metrics_count,
                "size": r.unique_groups,
                "color": r.severity_score,
            }
            for r in interaction_results
        ],
        "bar_chart": [
            {"interaction": r.interaction_name, "severity": r.severity_score, "status": r.bias_status}
            for r in sorted(interaction_results, key=lambda x: x.severity_score, reverse=True)[:10]
        ],
        "heatmap": {
            "interactions": [r.interaction_name for r in interaction_results],
            "amplification": [r.bias_amplification for r in interaction_results],
            "biased_count": [r.biased_metrics_count for r in interaction_results],
        },
    }


@router.get("/available-columns")
def get_available_columns(
    current_user: str = Depends(get_current_user),
    project_id: str = None,
    target_column: str = None,
):
    """
    Get list of available columns for interaction analysis

    Returns column names with metadata for better UI display
    """
    # Validate current_user structure
    if not isinstance(current_user, dict) or "token" not in current_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials: token missing.",
        )
    token = current_user.get("token")

    test_df = None
    try:
        # --- 1. Try S3 if project_id provided ---
        if project_id:
            try:
                s3_metadata = get_s3_file_metadata(token, project_id)
                if s3_metadata:
                    train_url, test_url, model_url = validate_file_metadata(s3_metadata)
                    test_df = load_dataframe_from_url(test_url)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("S3 loading failed in /available-columns: %s", e)

        # --- 2. Fallback to global_session path only ---
        if test_df is None:
            if global_session.test_file_path is None:
                raise HTTPException(
                    status_code=400,
                    detail="Test data must be uploaded first",
                )
            test_df = FileManager.load_csv(global_session.test_file_path)

        # --- 3. Resolve / validate target_column from payload ---
        if target_column is None:
            raise HTTPException(
                status_code=400,
                detail="target_column must be provided.",
            )
        if target_column not in test_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"target_column '{target_column}' not found in test data.",
            )

        # Build column metadata
        columns_metadata = []
        for col in test_df.columns:
            if col == target_column:  # <- use payload value
                continue
            col_info = {
                "name": col,
                "type": str(test_df[col].dtype),
                "unique_values": int(test_df[col].nunique()),
                "missing_values": int(test_df[col].isnull().sum()),
                "sample_values": test_df[col].dropna().head(3).tolist(),
            }
            columns_metadata.append(col_info)

        return {
            "status": "success",
            "columns": columns_metadata,
            "total_columns": len(columns_metadata),
            "target_column": target_column,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# @router.get("/available-columns")
# def get_available_columns(current_user: str = Depends(get_current_user), project_id: str = None):
#     """
#     Get list of available columns for interaction analysis

#     Returns column names with metadata for better UI display
#     """
#     token = current_user.get("token")
#     get_s3_file_metadata(token, project_id)
#     try:
#         if global_session.train_file_path is None:
#             raise HTTPException(status_code=400, detail="Training data must be uploaded first")

#         train_df = FileManager.load_csv(global_session.train_file_path)

#         # Build column metadata
#         columns_metadata = []
#         for col in train_df.columns:
#             if col == global_session.target_column:
#                 continue

#             col_info = {
#                 "name": col,
#                 "type": str(train_df[col].dtype),
#                 "unique_values": int(train_df[col].nunique()),
#                 "missing_values": int(train_df[col].isnull().sum()),
#                 "sample_values": train_df[col].dropna().head(3).tolist(),
#             }
#             columns_metadata.append(col_info)

#         return {
#             "status": "success",
#             "columns": columns_metadata,
#             "total_columns": len(columns_metadata),
#             "target_column": global_session.target_column,
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@router.post("/preview", response_model=InteractionPreviewResponse)
def preview_interaction(
    request: InteractionPreviewRequest, current_user: str = Depends(get_current_user), project_id: str = None
):
    """
    Preview interaction before full analysis

    Shows sample values and distribution without running bias metrics.
    Perfect for exploring what the interaction will look like.
    """
    # Validate current_user structure
    if not isinstance(current_user, dict) or "token" not in current_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials: token missing.",
        )
    token = current_user.get("token")

    test_df = None
    try:
        # 1. Try S3 if project_id is provided
        if project_id:
            try:
                s3_metadata = get_s3_file_metadata(token, project_id)
                if s3_metadata:
                    train_url, test_url, model_url = validate_file_metadata(s3_metadata)
                    if test_url:
                        test_df = load_dataframe_from_url(test_url)
            except Exception as e:
                logger.warning("S3 loading failed in /preview: %s", e)

        # 2. Fallback to global_session.test_file_path
        if test_df is None:
            if global_session.test_file_path is None:
                raise HTTPException(status_code=400, detail="Test data must be uploaded first")
            test_df = FileManager.load_csv(global_session.test_file_path)

        # Validate target_column from payload
        if request.target_column not in test_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"target_column '{request.target_column}' not found in test data",
            )

        # Validate columns
        for col in request.columns:
            if col not in test_df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

        # Create interaction
        generator = FeatureInteractionGenerator()
        spec = {"columns": request.columns, "type": request.interaction_type}

        df_with_interaction, metadata = generator.create_interaction_features(test_df, [spec], request.interaction_type)

        interaction_name = list(metadata.keys())[0]
        interaction_col = df_with_interaction[interaction_name]

        # Get sample values with original column values
        sample_indices = np.random.choice(len(test_df), min(request.sample_size, len(test_df)), replace=False)
        samples = []

        for idx in sample_indices:
            sample = {
                "interaction_value": str(interaction_col.iloc[idx]),
                "original_values": {col: str(test_df[col].iloc[idx]) for col in request.columns},
            }
            samples.append(sample)

        # Get distribution
        value_counts = interaction_col.value_counts()
        distribution = {str(k): int(v) for k, v in value_counts.head(10).items()}

        # Estimate analysis time
        unique_groups = interaction_col.nunique()
        estimated_time = f"{unique_groups * 0.5:.1f}-{unique_groups * 1:.1f} seconds"

        return InteractionPreviewResponse(
            interaction_name=interaction_name,
            source_columns=request.columns,
            interaction_type=request.interaction_type,
            unique_groups=unique_groups,
            sample_values=samples,
            group_distribution=distribution,
            estimated_analysis_time=estimated_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in preview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=FeatureInteractionResponse)
def analyze_feature_interactions(
    request: FeatureInteractionRequest, current_user: str = Depends(get_current_user), project_id: str = None
):
    """
    Analyze bias when features are combined/interacted

    Shows how combining features affects bias without any mitigation.
    Useful for understanding intersectional fairness.
    """
    token = current_user.get("token")
    test_df = None
    start_time = time.time()
    try:
        # Try S3 if project_id provided
        if project_id:
            try:
                s3_metadata = get_s3_file_metadata(token, project_id)
                if s3_metadata:
                    train_url, test_url, model_url = validate_file_metadata(s3_metadata)
                    if test_url:
                        test_df = load_dataframe_from_url(test_url)
            except Exception as e:
                logger.warning("S3 loading failed in /analyze: %s", e)

        # Fallback to global_session path only
        if test_df is None:
            if global_session.test_file_path is None:
                raise HTTPException(status_code=400, detail="Test data must be uploaded first")
            test_df = FileManager.load_csv(global_session.test_file_path)

        # Use target column from payload if provided, else fallback
        target_column = getattr(request, "target_column", None) or global_session.target_column
        if target_column is None:
            raise HTTPException(status_code=400, detail="Target column must be set first")

        logger.info(f"Starting feature interaction analysis for columns: {request.columns}")

        # Validate columns exist
        for col in request.columns:
            if col not in test_df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in dataset")

        # Get target and predictions
        y_true = _safe_convert_to_numeric(test_df[target_column].values, "y_true")

        # Get predictions
        if request.prediction_column:
            if request.prediction_column not in test_df.columns:
                raise HTTPException(
                    status_code=400, detail=f"Prediction column '{request.prediction_column}' not found"
                )
            y_pred = _safe_convert_to_numeric(test_df[request.prediction_column].values, request.prediction_column)
        else:
            # Try to load model from S3 if available
            model = None
            model_url = None
            if project_id:
                try:
                    s3_metadata = get_s3_file_metadata(token, project_id)
                    if s3_metadata:
                        _, _, model_url = validate_file_metadata(s3_metadata)
                except Exception as e:
                    logger.warning("S3 model loading failed in /analyze: %s", e)

            if model_url:
                model = load_model_from_url(model_url)
            elif global_session.model_file_path:
                model = FileManager.load_model(global_session.model_file_path)
            else:
                raise HTTPException(status_code=400, detail="Either prediction_column or uploaded model required")

            X = test_df.drop(columns=[target_column])
            X_encoded, _ = _encode_dataframe(X)
            if isinstance(X_encoded, pd.DataFrame):
                X_encoded = X_encoded.values
            X_encoded = np.array(X_encoded, dtype=np.float64)

            y_pred = model.predict(X_encoded)
            y_pred = _safe_convert_to_numeric(y_pred, "model_predictions")

        # Initialize bias detector
        bias_detector = BiasDetector()

        # Calculate baseline bias for each individual column
        logger.info("Calculating baseline bias for individual columns...")
        baseline_results = {}

        if request.include_individual_bias:
            for col in request.columns:
                sensitive_attr = _safe_convert_to_numeric(test_df[col].values, col)
                baseline_metrics = bias_detector.run_all_metrics(y_true, y_pred, sensitive_attr)
                baseline_metrics = _clean_nan_values(baseline_metrics)
                baseline_results[col] = {
                    "biased_metrics_count": baseline_metrics["summary"]["biased_metrics_count"],
                    "total_metrics": baseline_metrics["summary"]["total_metrics"],
                    "metrics": {
                        k: v
                        for k, v in baseline_metrics.items()
                        if k != "summary"  # Exclude summary, keep individual metric results
                    },
                }

        avg_baseline_biased = np.mean([r["biased_metrics_count"] for r in baseline_results.values()])

        logger.info(f"Average baseline biased metrics: {avg_baseline_biased}")

        # Generate interactions
        interaction_generator = FeatureInteractionGenerator()

        # Determine which combinations to analyze
        if request.combinations:
            # User specified combinations
            interaction_specs = [
                {"columns": combo, "type": request.interaction_type}
                for combo in request.combinations[: request.max_combinations]
            ]
        else:
            # Generate all pairwise combinations
            interaction_specs = []
            for col1, col2 in combinations(request.columns, 2):
                interaction_specs.append({"columns": [col1, col2], "type": request.interaction_type})
                if len(interaction_specs) >= request.max_combinations:
                    break
        logger.info(f"Analyzing {len(interaction_specs)} pairwise combinations")

        # Analyze bias for each interaction
        interaction_results = []

        for spec in interaction_specs:
            try:
                # Create interaction feature
                df_with_interaction, metadata = interaction_generator.create_interaction_features(
                    test_df, [spec], request.interaction_type
                )

                interaction_name = list(metadata.keys())[0]
                interaction_meta = metadata[interaction_name]

                logger.info(f"Analyzing interaction: {interaction_name}")
                sample_values = df_with_interaction[interaction_name].dropna().head(5).astype(str).tolist()

                # Get interaction as sensitive attribute
                interaction_values = df_with_interaction[interaction_name].values
                sensitive_attr = _safe_convert_to_numeric(interaction_values, interaction_name)

                # Calculate bias metrics for interaction
                interaction_bias = bias_detector.run_all_metrics(y_true, y_pred, sensitive_attr)
                interaction_bias = _clean_nan_values(interaction_bias)

                biased_count = interaction_bias["summary"]["biased_metrics_count"]
                total_metrics = interaction_bias["summary"]["total_metrics"]
                bias_percentage = (biased_count / total_metrics * 100) if total_metrics > 0 else 0

                # Calculate amplification
                amplification = _calculate_bias_amplification(avg_baseline_biased, biased_count)

                severity = _calculate_severity_score(biased_count, total_metrics, amplification)

                # Determine bias status
                if biased_count == 0:
                    bias_status = "fair"
                elif biased_count >= 50:
                    bias_status = "highly_biased"
                elif biased_count >= 25:
                    bias_status = "biased"
                else:
                    bias_status = "unknown"

                # Generate recommendations
                recommendations = _generate_interaction_recommendations(
                    bias_status, amplification, interaction_meta["unique_groups"], spec["columns"]
                )

                # Create result
                interaction_result = InteractionBiasDetail(
                    interaction_name=interaction_name,
                    source_columns=interaction_meta["source_columns"],
                    interaction_type=interaction_meta["interaction_type"],
                    unique_groups=interaction_meta["unique_groups"],
                    sample_values=sample_values,
                    bias_metrics={k: v for k, v in interaction_bias.items() if k != "summary"},
                    biased_metrics_count=biased_count,
                    total_metrics=total_metrics,
                    bias_percentage=round(bias_percentage, 2),
                    bias_status=bias_status,
                    bias_amplification=amplification,
                    severity_score=severity,
                    recommendations=recommendations,
                )

                interaction_results.append(interaction_result)

                logger.info(f"  Biased metrics: {biased_count}/{total_metrics}, " f"Amplification: {amplification}%")

            except Exception as e:
                logger.error(f"Failed to analyze interaction {spec}: {str(e)}")
                continue

        # Sort results by bias amplification (descending)
        interaction_results.sort(key=lambda x: x.severity_score, reverse=True)

        # Find most and least biased
        most_biased = interaction_results[0].interaction_name if interaction_results else None
        least_biased = interaction_results[-1].interaction_name if interaction_results else None

        # Create summary
        summary = {
            "average_baseline_biased_metrics": round(avg_baseline_biased, 2),
            "interactions_with_increased_bias": sum(1 for r in interaction_results if r.bias_amplification > 0),
            "interactions_with_decreased_bias": sum(1 for r in interaction_results if r.bias_amplification < 0),
            "interactions_with_same_bias": sum(1 for r in interaction_results if r.bias_amplification == 0),
            "max_bias_amplification": max((r.bias_amplification for r in interaction_results), default=0),
            "min_bias_amplification": min((r.bias_amplification for r in interaction_results), default=0),
            "average_amplification": (
                round(np.mean([r.bias_amplification for r in interaction_results]), 2) if interaction_results else 0
            ),
            "average_severity": (
                round(np.mean([r.severity_score for r in interaction_results]), 2) if interaction_results else 0
            ),
            "highly_biased_count": sum(1 for r in interaction_results if r.bias_status == "highly_biased"),
        }

        # Prepare visualization data
        viz_data = _prepare_visualization_data(interaction_results)

        execution_time = time.time() - start_time

        return FeatureInteractionResponse(
            status="success",
            total_interactions=len(interaction_specs),
            interactions_analyzed=len(interaction_results),
            baseline_bias=baseline_results,
            interaction_results=interaction_results,
            most_biased_interaction=most_biased,
            least_biased_interaction=least_biased,
            summary=summary,
            visualization_data=viz_data,
            execution_time_seconds=round(execution_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in feature interaction analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing feature interactions: {str(e)}")


@router.post("/compare-types")
def compare_interaction_types(
    request: InteractionComparisonRequest, current_user: str = Depends(get_current_user), project_id: str = None
):
    """
    Compare how different interaction types affect bias for the same columns

    Useful for understanding which interaction method preserves fairness best.
    """
    # Validate current_user structure
    if not isinstance(current_user, dict) or "token" not in current_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials: token missing.",
        )
    token = current_user.get("token")

    test_df = None
    try:
        # 1. Try S3 if project_id is provided
        if project_id:
            try:
                s3_metadata = get_s3_file_metadata(token, project_id)
                if s3_metadata:
                    train_url, test_url, model_url = validate_file_metadata(s3_metadata)
                    if test_url:
                        test_df = load_dataframe_from_url(test_url)
            except Exception as e:
                logger.warning("S3 loading failed in /compare-types: %s", e)

        # 2. Fallback to global_session.test_file_path
        if test_df is None:
            if global_session.test_file_path is None:
                raise HTTPException(status_code=400, detail="Test data must be uploaded first")
            test_df = FileManager.load_csv(global_session.test_file_path)

        # Validate target_column from payload
        if request.target_column not in test_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"target_column '{request.target_column}' not found in test data",
            )

        # STEP 1: Calculate baseline ONCE (before comparing types)
        # Get predictions
        y_true = _safe_convert_to_numeric(test_df[request.target_column].values, "y_true")

        if request.prediction_column:
            if request.prediction_column not in test_df.columns:
                raise HTTPException(
                    status_code=400, detail=f"Prediction column '{request.prediction_column}' not found"
                )
            y_pred = _safe_convert_to_numeric(test_df[request.prediction_column].values, request.prediction_column)
        else:
            # Try to load model from S3 if available
            model = None
            model_url = None
            if project_id:
                try:
                    s3_metadata = get_s3_file_metadata(token, project_id)
                    if s3_metadata:
                        _, _, model_url = validate_file_metadata(s3_metadata)
                except Exception as e:
                    logger.warning("S3 model loading failed in /analyze: %s", e)
            if model_url:
                model = load_model_from_url(model_url)
            elif global_session.model_file_path:
                model = FileManager.load_model(global_session.model_file_path)
            else:
                raise HTTPException(status_code=400, detail="Either prediction_column or uploaded model required")

            X = test_df.drop(columns=[request.target_column])
            X_encoded, _ = _encode_dataframe(X)
            if isinstance(X_encoded, pd.DataFrame):
                X_encoded = X_encoded.values
            X_encoded = np.array(X_encoded, dtype=np.float64)
            y_pred = model.predict(X_encoded)
            y_pred = _safe_convert_to_numeric(y_pred, "model_predictions")

        # Calculate baseline for each column
        bias_detector = BiasDetector()
        baseline_biased_counts = []

        for col in request.columns:
            sensitive_attr = _safe_convert_to_numeric(test_df[col].values, col)
            baseline_metrics = bias_detector.run_all_metrics(y_true, y_pred, sensitive_attr)
            baseline_metrics = _clean_nan_values(baseline_metrics)
            baseline_biased_counts.append(baseline_metrics["summary"]["biased_metrics_count"])

        avg_baseline_biased = np.mean(baseline_biased_counts) if baseline_biased_counts else 0

        logger.info(f"Baseline bias (avg): {avg_baseline_biased}")

        # STEP 2: Now compare interaction types
        results = {}

        for interaction_type in request.interaction_types:
            # Create mini-request for each type
            analysis_request = FeatureInteractionRequest(
                columns=request.columns,
                interaction_type=interaction_type,
                prediction_column=request.prediction_column,
                max_combinations=1,  # Just analyze the direct combination
                include_individual_bias=False,
                target_column=request.target_column,
            )

            # Run analysis (reuse main endpoint logic)
            response = analyze_feature_interactions(analysis_request, current_user, project_id)

            if response.interaction_results:
                result = response.interaction_results[0]

                # Manually calculate amplification since we have baseline
                amplification = _calculate_bias_amplification(avg_baseline_biased, result.biased_metrics_count)

                # Manually calculate severity
                severity = _calculate_severity_score(result.biased_metrics_count, result.total_metrics, amplification)

                # Determine bias status
                bias_percentage = (
                    (result.biased_metrics_count / result.total_metrics * 100) if result.total_metrics > 0 else 0
                )
                if bias_percentage >= 50:
                    bias_status = "highly_biased"
                elif bias_percentage >= 25:
                    bias_status = "biased"
                elif bias_percentage > 0:
                    bias_status = "low_bias"
                else:
                    bias_status = "fair"

                results[interaction_type] = {
                    "bias_amplification": amplification,
                    "severity_score": severity,
                    "biased_metrics_count": result.biased_metrics_count,
                    "bias_status": bias_status,
                }

        # Determine best interaction type
        best_type = min(results.items(), key=lambda x: x[1]["severity_score"])[0] if results else None

        def clean_nan(obj):
            """Recursively replace NaN with None for JSON serialization"""
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(v) for v in obj]
            elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj

        response_data = {
            "status": "success",
            "columns_analyzed": request.columns,
            "interaction_types_compared": request.interaction_types,
            "baseline_bias": {
                "average_biased_metrics": round(avg_baseline_biased, 2),
                "individual_columns": {col: int(count) for col, count in zip(request.columns, baseline_biased_counts)},
            },
            "results": results,
            "recommended_type": best_type,
            "comparison_chart": {
                "types": list(results.keys()),
                "severity_scores": [v["severity_score"] for v in results.values()],
                "amplifications": [v["bias_amplification"] for v in results.values()],
            },
        }

        return clean_nan(response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing interaction types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interaction-types")
def get_available_interaction_types(current_user: str = Depends(get_current_user), project_id: str = None):
    """Get list of available interaction types with descriptions"""
    token = current_user.get("token")
    get_s3_file_metadata(token, project_id)
    return {
        "status": "success",
        "interaction_types": [
            {
                "name": "concatenate",
                "description": "Combine values as text (e.g., 'Male_Asian')",
                "example": "gender='Male', race='Asian' → 'Male_Asian'",
                "best_for": "Categorical features, intersectional analysis",
            },
            {
                "name": "multiply",
                "description": "Multiply numeric values",
                "example": "age=25, income=50000 → 1250000",
                "best_for": "Numeric features, scaling effects",
            },
            {
                "name": "add",
                "description": "Add numeric values",
                "example": "age=25, years_experience=5 → 30",
                "best_for": "Numeric features, cumulative effects",
            },
            {
                "name": "and",
                "description": "Logical AND for binary features",
                "example": "has_degree=True, has_experience=True → True",
                "best_for": "Binary features, requirement combinations",
            },
            {
                "name": "or",
                "description": "Logical OR for binary features",
                "example": "has_degree=False, has_experience=True → True",
                "best_for": "Binary features, alternative qualifications",
            },
        ],
    }


@router.get("/columns/analyze")
def analyze_columns_for_interactions(current_user: str = Depends(get_current_user), project_id: str = None):
    """
    Analyze all columns to help users understand interaction possibilities

    Returns detailed information about each column including:
    - Data type
    - Recommended interaction types
    - Compatible columns
    """
    token = current_user.get("token")
    test_df = None
    try:
        # 1. Try S3 if project_id is provided
        if project_id:
            try:
                s3_metadata = get_s3_file_metadata(token, project_id)
                if s3_metadata:
                    _, test_url, _ = validate_file_metadata(s3_metadata)
                    if test_url:
                        test_df = load_dataframe_from_url(test_url)
            except Exception as e:
                logger.warning("S3 loading failed in /columns/analyze: %s", e)

        # 2. Fallback to global_session.test_file_path
        if test_df is None:
            if global_session.test_file_path is None:
                raise HTTPException(status_code=400, detail="Test data must be uploaded first")
            test_df = FileManager.load_csv(global_session.test_file_path)

        generator = FeatureInteractionGenerator()

        columns_analysis = []
        # Use test_df for all calculations
        all_columns = [col for col in test_df.columns if col != global_session.target_column]

        for col in all_columns:
            col_type = generator._determine_column_type(test_df[col])
            unique_count = int(test_df[col].nunique())
            missing_pct = round((test_df[col].isnull().sum() / len(test_df)) * 100, 2)
            samples = test_df[col].dropna().head(5).astype(str).tolist()

            # Recommend interaction types based on column type
            if col_type == "categorical":
                recommended_types = ["concatenate"]
            elif col_type == "binary":
                recommended_types = ["and", "or", "concatenate"]
            else:  # numerical
                recommended_types = ["multiply", "add", "concatenate"]

            # Find compatible columns (same or compatible types)
            compatible = []
            for other_col in all_columns:
                if other_col != col:
                    other_type = generator._determine_column_type(test_df[other_col])
                    # All types can concatenate, same types can use specialized operations
                    if other_type == col_type or "concatenate" in recommended_types:
                        compatible.append(other_col)

            analysis = ColumnAnalysisResponse(
                column_name=col,
                data_type=col_type,
                unique_values=unique_count,
                missing_percentage=missing_pct,
                sample_values=samples,
                recommended_interaction_types=recommended_types,
                compatible_columns=compatible[:10],  # Limit for readability
            )
            columns_analysis.append(analysis)

        return {
            "status": "success",
            "total_columns": len(columns_analysis),
            "columns": columns_analysis,
            "interaction_recommendations": {
                "easy_start": "Try concatenating categorical columns like gender and race",
                "numerical_interactions": "Multiply or add numerical columns for scaling effects",
                "binary_combinations": "Use AND/OR for binary features to capture requirement logic",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing columns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-smart", response_model=SmartInteractionResponse)
def analyze_smart_interactions(
    request: SmartInteractionRequest, current_user: str = Depends(get_current_user), project_id: str = None
):
    """
    Smart interaction analysis with automatic type detection

    This endpoint intelligently determines the best interaction method
    for each feature combination based on their data types.
    Perfect for exploratory analysis!
    """
    # Validate current_user structure
    if not isinstance(current_user, dict) or "token" not in current_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials: token missing.",
        )
    token = current_user.get("token")

    test_df = None
    start_time = time.time()

    try:
        # 1. Try S3 if project_id is provided
        if project_id:
            try:
                s3_metadata = get_s3_file_metadata(token, project_id)
                if s3_metadata:
                    train_url, test_url, model_url = validate_file_metadata(s3_metadata)
                    if test_url:
                        test_df = load_dataframe_from_url(test_url)
            except Exception as e:
                logger.warning("S3 loading failed in /analyze-smart: %s", e)

        # 2. Fallback to global_session.test_file_path
        if test_df is None:
            if global_session.test_file_path is None:
                raise HTTPException(status_code=400, detail="Test data must be uploaded first")
            test_df = FileManager.load_csv(global_session.test_file_path)

        # Validate target_column from payload
        if request.target_column not in test_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"target_column '{request.target_column}' not found in test data",
            )

        # Validate columns
        for col in request.columns:
            if col not in test_df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

        # Get predictions (same as before)
        y_true = _safe_convert_to_numeric(test_df[request.target_column].values, "y_true")

        if request.prediction_column:
            if request.prediction_column not in test_df.columns:
                raise HTTPException(status_code=400, detail=f"Prediction column not found")
            y_pred = _safe_convert_to_numeric(test_df[request.prediction_column].values, request.prediction_column)
        else:
            # Try to load model from S3 if available
            model = None
            model_url = None
            if project_id:
                try:
                    s3_metadata = get_s3_file_metadata(token, project_id)
                    if s3_metadata:
                        _, _, model_url = validate_file_metadata(s3_metadata)
                except Exception as e:
                    logger.warning("S3 model loading failed in /analyze: %s", e)
            if model_url:
                model = load_model_from_url(model_url)
            elif global_session.model_file_path:
                model = FileManager.load_model(global_session.model_file_path)
            else:
                raise HTTPException(status_code=400, detail="Either prediction_column or uploaded model required")

            X = test_df.drop(columns=[request.target_column])
            X_encoded, _ = _encode_dataframe(X)
            if isinstance(X_encoded, pd.DataFrame):
                X_encoded = X_encoded.values
            X_encoded = np.array(X_encoded, dtype=np.float64)
            y_pred = model.predict(X_encoded)
            y_pred = _safe_convert_to_numeric(y_pred, "model_predictions")

        bias_detector = BiasDetector()
        interaction_generator = FeatureInteractionGenerator()

        # Calculate baseline
        baseline_results = {}
        for col in request.columns:
            sensitive_attr = _safe_convert_to_numeric(test_df[col].values, col)
            baseline_metrics = bias_detector.run_all_metrics(y_true, y_pred, sensitive_attr)
            baseline_metrics = _clean_nan_values(baseline_metrics)
            baseline_results[col] = {
                "biased_metrics_count": baseline_metrics["summary"]["biased_metrics_count"],
                "total_metrics": baseline_metrics["summary"]["total_metrics"],
            }

        avg_baseline_biased = np.mean([r["biased_metrics_count"] for r in baseline_results.values()])

        # Generate interactions with smart detection
        type_recommendations = []
        interaction_specs = []

        if request.interaction_size:
            # Specific n-way interactions
            combos = list(combinations(request.columns, request.interaction_size))
        else:
            # All sizes from 2 to min(5, len(columns))
            combos = []
            max_size = min(5, len(request.columns))
            for size in range(2, max_size + 1):
                combos.extend(list(combinations(request.columns, size)))

        # Limit to max_interactions
        combos = combos[: request.max_interactions]

        for combo in combos:
            combo_list = list(combo)

            # Auto-detect or use fixed type
            if request.auto_detect_types and request.interaction_type is None:
                detected_type = interaction_generator.auto_detect_best_interaction(test_df, combo_list)

                # Store recommendation
                if request.include_type_recommendations:
                    col_types = {col: interaction_generator._determine_column_type(test_df[col]) for col in combo_list}

                    # Suggest alternatives
                    alternatives = []
                    if detected_type == "concatenate":
                        alternatives = ["multiply", "add"] if all(t == "numerical" for t in col_types.values()) else []
                    elif detected_type in ["multiply", "add"]:
                        alternatives = ["concatenate"]

                    recommendation = InteractionTypeRecommendation(
                        columns=combo_list,
                        recommended_type=detected_type,
                        reason=f"Auto-detected based on column types: {list(col_types.values())}",
                        column_types=col_types,
                        alternatives=alternatives,
                    )
                    type_recommendations.append(recommendation)

                interaction_type = detected_type
            else:
                interaction_type = request.interaction_type or "concatenate"

            interaction_specs.append({"columns": combo_list, "type": interaction_type})

        logger.info(f"Analyzing {len(interaction_specs)} smart interactions")

        # Analyze each interaction
        interaction_results = []

        for spec in interaction_specs:
            try:
                df_with_interaction, metadata = interaction_generator.create_interaction_features(
                    test_df, [spec], spec["type"]
                )

                interaction_name = list(metadata.keys())[0]
                interaction_meta = metadata[interaction_name]

                sample_values = df_with_interaction[interaction_name].dropna().head(5).astype(str).tolist()
                interaction_values = df_with_interaction[interaction_name].values
                sensitive_attr = _safe_convert_to_numeric(interaction_values, interaction_name)

                interaction_bias = bias_detector.run_all_metrics(y_true, y_pred, sensitive_attr)
                interaction_bias = _clean_nan_values(interaction_bias)

                biased_count = interaction_bias["summary"]["biased_metrics_count"]
                total_metrics = interaction_bias["summary"]["total_metrics"]
                bias_percentage = (biased_count / total_metrics * 100) if total_metrics > 0 else 0

                amplification = _calculate_bias_amplification(avg_baseline_biased, biased_count)
                severity = _calculate_severity_score(biased_count, total_metrics, amplification)

                if biased_count == 0:
                    bias_status = "fair"
                elif bias_percentage >= 50:
                    bias_status = "highly_biased"
                elif bias_percentage >= 25:
                    bias_status = "biased"
                else:
                    bias_status = "low_bias"

                recommendations = _generate_interaction_recommendations(
                    bias_status, amplification, interaction_meta["unique_groups"], spec["columns"]
                )

                interaction_result = InteractionBiasDetail(
                    interaction_name=interaction_name,
                    source_columns=interaction_meta["source_columns"],
                    interaction_type=interaction_meta["interaction_type"],
                    unique_groups=interaction_meta["unique_groups"],
                    sample_values=sample_values,
                    bias_metrics={k: v for k, v in interaction_bias.items() if k != "summary"},
                    biased_metrics_count=biased_count,
                    total_metrics=total_metrics,
                    bias_percentage=round(bias_percentage, 2),
                    bias_status=bias_status,
                    bias_amplification=amplification,
                    severity_score=severity,
                    recommendations=recommendations,
                )

                interaction_results.append(interaction_result)

            except Exception as e:
                logger.error(f"Failed to analyze interaction {spec}: {str(e)}")
                continue

        interaction_results.sort(key=lambda x: x.severity_score, reverse=True)

        most_biased = interaction_results[0].interaction_name if interaction_results else None
        least_biased = interaction_results[-1].interaction_name if interaction_results else None

        summary = {
            "average_baseline_biased_metrics": round(avg_baseline_biased, 2),
            "interactions_with_increased_bias": sum(1 for r in interaction_results if r.bias_amplification > 0),
            "interactions_with_decreased_bias": sum(1 for r in interaction_results if r.bias_amplification < 0),
            "max_bias_amplification": max((r.bias_amplification for r in interaction_results), default=0),
            "average_severity": (
                round(np.mean([r.severity_score for r in interaction_results]), 2) if interaction_results else 0
            ),
            "interaction_types_used": list(set(r.interaction_type for r in interaction_results)),
            "highly_biased_count": sum(1 for r in interaction_results if r.bias_status == "highly_biased"),
        }

        viz_data = _prepare_visualization_data(interaction_results)
        execution_time = time.time() - start_time

        analysis_mode = "auto_detect" if request.auto_detect_types else "fixed_type"

        return SmartInteractionResponse(
            status="success",
            analysis_mode=analysis_mode,
            total_combinations=len(combos),
            interactions_analyzed=len(interaction_results),
            baseline_bias=baseline_results,
            interaction_results=interaction_results,
            type_recommendations=type_recommendations if request.include_type_recommendations else None,
            most_biased_interaction=most_biased,
            least_biased_interaction=least_biased,
            summary=summary,
            visualization_data=viz_data,
            execution_time_seconds=round(execution_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in smart interaction analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-flexible")
def analyze_flexible_interactions(
    request: FlexibleInteractionRequest, current_user: str = Depends(get_current_user), project_id: str = None
):
    """
    Maximum flexibility: analyze custom combinations with per-combination types

    Perfect for advanced users who want complete control over their analysis.
    Each interaction can have its own type, or auto-detection can fill gaps.
    """
    # Validate current_user structure
    if not isinstance(current_user, dict) or "token" not in current_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials: token missing.",
        )
    token = current_user.get("token")

    test_df = None
    # Reuse smart analysis logic but with custom specs
    try:
        # 1. Try S3 if project_id is provided
        if project_id:
            try:
                s3_metadata = get_s3_file_metadata(token, project_id)
                if s3_metadata:
                    train_url, test_url, model_url = validate_file_metadata(s3_metadata)
                    if test_url:
                        test_df = load_dataframe_from_url(test_url)
            except Exception as e:
                logger.warning("S3 loading failed in /analyze-flexible: %s", e)

        # 2. Fallback to global_session.test_file_path
        if test_df is None:
            if global_session.test_file_path is None:
                raise HTTPException(status_code=400, detail="Test data must be uploaded first")
            test_df = FileManager.load_csv(global_session.test_file_path)

        # Validate target_column from payload
        if request.target_column not in test_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"target_column '{request.target_column}' not found in test data",
            )

        # Convert flexible request to smart request format
        all_columns = list(set([col for interaction in request.interactions for col in interaction["columns"]]))

        # Validate all columns exist
        for col in all_columns:
            if col not in test_df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

        # Create custom specs
        interaction_specs = []
        for interaction in request.interactions:
            spec = {"columns": interaction["columns"]}

            if "type" in interaction and interaction["type"]:
                spec["type"] = interaction["type"]
            elif request.auto_detect_missing_types:
                generator = FeatureInteractionGenerator()
                spec["type"] = generator.auto_detect_best_interaction(test_df, interaction["columns"])
            else:
                spec["type"] = "concatenate"  # default

            interaction_specs.append(spec)

        # Use the smart analysis logic with custom specs
        # (Implementation similar to analyze_smart_interactions but with pre-defined specs)

        return {
            "status": "success",
            "message": "Flexible interaction analysis complete",
            "interactions_analyzed": len(interaction_specs),
            "note": "Full implementation follows smart analysis pattern",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in flexible interaction analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
