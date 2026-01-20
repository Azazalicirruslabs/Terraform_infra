"""
Router for explainability analysis endpoints.
"""

import logging

from fastapi import APIRouter, HTTPException

from services.mainflow.app.core.explainability_logic import ExplainabilityService
from services.mainflow.app.schemas.explainability import ExplainabilityRequest, ExplainabilityResponse
from services.mainflow.app.utils.explainability_mainflow import sanitize_for_json

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/mainflow/explainability",
    tags=["Explainability"],
)


@router.post("/analyze", response_model=ExplainabilityResponse)
async def analyze_explainability(request: ExplainabilityRequest):
    """
    Analyze model explainability with performance metrics and feature importance.

    This endpoint:
    1. Loads the model and datasets from the provided paths/URLs
    2. Automatically detects whether it's a classification or regression model
    3. Computes appropriate performance metrics (train/test/overfitting)
    4. Calculates feature importance using the selected method (SHAP, Gain, or Permutation)
    5. Evaluates metrics against configured thresholds (if provided)
    6. Generates LLM-based insights about model performance

    Args:
        request: ExplainabilityRequest containing:
            - model: Path or URL to the model file
            - ref_dataset: Path or URL to the training/reference dataset
            - cur_dataset: Path or URL to the test/current dataset
            - target_column: Name of the target column
            - compute_feature_importance: Whether to compute feature importance (default: True)
            - feature_importance_method: Method to use - 'shap', 'gain', or 'permutation' (default: 'shap')
            - thresholds: Optional threshold configuration for metric evaluation

    Returns:
        ExplainabilityResponse with:
            - performance_metrics: Model performance on train/test sets
            - features: Feature importance with impact directions
            - llm_analysis: AI-generated insights and explanations
            - threshold_evaluations: Status of metrics vs thresholds (if configured)

    Raises:
        HTTPException: If validation fails or processing encounters errors
    """
    try:
        logger.info("Starting explainability analysis...")

        # Initialize service
        service = ExplainabilityService()

        # Load model and datasets
        logger.info("Loading model and datasets...")
        load_result = service.load_model_and_datasets(
            model_path=request.model,
            train_data_path=request.ref_dataset,
            test_data_path=request.cur_dataset,
            target_column=request.target_column,
        )

        logger.info(
            "Model loaded successfully. Type: %s, Features: %s",
            load_result["model_type"],
            load_result["features_count"],
        )

        # Compute explainability analysis
        logger.info("Computing explainability analysis...")

        # Prepare threshold configuration if provided
        threshold_config = None
        if request.thresholds:
            threshold_config = {
                "acceptable_threshold": request.thresholds.acceptable_threshold,
                "warning_threshold": request.thresholds.warning_threshold,
                "breach_threshold": request.thresholds.breach_threshold,
            }

        result = service.compute_explainability_analysis(
            compute_feature_importance=request.feature_importance,
            method=request.feature_importance_method,
            thresholds=threshold_config,
        )

        # Sanitize result for JSON serialization
        sanitized_result = sanitize_for_json(result)

        logger.info("Explainability analysis completed successfully")
        return sanitized_result

    except ValueError as e:
        logger.error("Validation error: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}") from e

    except FileNotFoundError as e:
        logger.error("File not found: %s", str(e))
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}") from e

    except RuntimeError as e:
        logger.error("Runtime error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}") from e

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Unexpected error in explainability analysis: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during explainability analysis: {str(e)}",
        ) from e
