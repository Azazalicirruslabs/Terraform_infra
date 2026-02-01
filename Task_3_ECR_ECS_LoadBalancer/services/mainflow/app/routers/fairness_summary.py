"""
BiasLens™ Analyzer - Fairness Analysis API Router

This module provides RESTful API endpoints for fairness analysis in machine learning models.

Endpoints:
    POST /mainflow/fairness/analyze - Run complete fairness analysis
    GET /mainflow/fairness/health - Health check endpoint

Workflow:
    1. Users upload files via the main API: POST /api/files_upload
    2. Users retrieve file URLs: GET /api/files_download/{analysis_type}/{project_name}
    3. Users call /analyze with URLs and fairness configuration
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from services.mainflow.app.core.fairness_summary import FairnessSummaryCalculator, FairnessThresholds
from services.mainflow.app.schemas.fairness_schema import (
    FairnessAnalysisRequest,
    FairnessAnalysisResponse,
)
from shared.auth import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/mainflow/fairness", tags=["BiasLens Fairness"])


# ==================== API Endpoints ====================


@router.post("/analyze", response_model=FairnessAnalysisResponse, status_code=200)
async def analyze_fairness(
    request: FairnessAnalysisRequest, current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Run complete fairness analysis on provided data.

    **Authentication:** Requires Bearer token in Authorization header.

    This endpoint performs comprehensive fairness analysis including:
    - Data loading and preprocessing
    - Optional prediction generation from model
    - Calculation of selected fairness metrics
    - Group performance comparison
    - Threshold-based assessment

    Args:
        request: FairnessAnalysisRequest containing all configuration and data sources

    Returns:
        FairnessAnalysisResponse with calculated metrics and assessment

    Raises:
        HTTPException 400: Invalid input data or configuration
        HTTPException 401: Unauthorized - invalid or missing token
        HTTPException 500: Internal processing error

    Example:
        ```json
        {
            "reference_url": "https://s3.amazonaws.com/training_data.csv",
            "current_url": "https://s3.amazonaws.com/testing_data.csv",
            "model_url": "https://s3.amazonaws.com/model.pkl",
            "target_column": "income",
            "sensitive_feature": "sex_Male",
            "metric_selection": {
                "equal_opportunity": true,
                "disparate_impact": true,
                "statistical_parity": true,
                "equalized_odds": true
            },
            "thresholds": {
                "acceptable_threshold": 80.0,
                "warning_threshold": 60.0,
                "breach_threshold": 40.0
            }
        }
        ```
    """
    logger.info("=" * 80)
    logger.info("BiasLens™ Fairness Analysis Request Received")
    logger.info(f"User: {current_user.get('username')} (ID: {current_user.get('user_id')})")
    logger.info("=" * 80)
    logger.debug(f"Reference URL: {request.reference_url}")
    logger.debug(f"Current URL: {request.current_url}")
    logger.debug(f"Model URL: {request.model_url}")
    logger.debug(f"Target Column: {request.target_column}")
    logger.debug(f"Sensitive Feature: {request.sensitive_feature}")
    logger.debug(f"Metrics Enabled: {request.metric_selection.model_dump()}")
    logger.debug(f"Thresholds: {request.thresholds.model_dump()}")

    try:
        # Convert Pydantic models to dataclasses for calculator initialization
        thresholds_dataclass = FairnessThresholds(
            acceptable=request.thresholds.acceptable_threshold,
            warning=request.thresholds.warning_threshold,
            breach=request.thresholds.breach_threshold,
        )

        # Convert metric selection to dict (required by process_fairness_request)
        metric_selection_dict = {
            "disparate_impact": request.metric_selection.disparate_impact,
            "statistical_parity": request.metric_selection.statistical_parity,
            "equalized_odds": request.metric_selection.equalized_odds,
            "equal_opportunity": request.metric_selection.equal_opportunity,
        }

        # Convert thresholds to dict (required by process_fairness_request)
        thresholds_dict = {
            "acceptable": int(request.thresholds.acceptable_threshold),
            "warning": int(request.thresholds.warning_threshold),
            "breach": int(request.thresholds.breach_threshold),
        }

        # Initialize calculator with default thresholds only
        calculator = FairnessSummaryCalculator(default_thresholds=thresholds_dataclass)

        # Process fairness analysis
        logger.info("Starting fairness analysis workflow...")
        result = calculator.process_fairness_request(
            reference_url=request.reference_url,
            target_column=request.target_column,
            sensitive_feature=request.sensitive_feature,
            metric_selection=metric_selection_dict,
            current_url=request.current_url,
            model_url=request.model_url,
            thresholds=thresholds_dict,
        )

        logger.info("=" * 80)
        logger.info(f"Analysis Complete - Status: {result.get('success', False)}")
        logger.info("=" * 80)

        # Extract data from nested structure
        data = result.get("data", {})

        return {
            "status": "success" if result.get("success") else "error",
            "metrics": data.get("key_metrics", {}),
            "group_performance": data.get("group_performance", {}),
            "llm_analysis": data.get("llm_analysis", {}),
            "overall_assessment": data.get("overall_fairness_assessment", "Unknown"),
            "metadata": data.get("metadata", {}),
            "applied_thresholds": data.get("applied_thresholds", {}),
            "warnings": result.get("warnings", []),
            "message": result.get("message"),
        }

    except ValueError as ve:
        # Client errors (invalid input)
        logger.error(f"Validation Error: {str(ve)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))

    except FileNotFoundError as fnf:
        # Missing data files
        logger.error(f"Data Not Found: {str(fnf)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Data source not accessible: {str(fnf)}")

    except Exception as e:
        # Server errors
        logger.error(f"Internal Error During Fairness Analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during fairness analysis: {str(e)}")


@router.get("/health", status_code=200)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for BiasLens™ Analyzer service.

    Returns:
        Service status information
    """
    logger.debug("BiasLens™ Analyzer health check")
    return {"status": "healthy", "service": "BiasLens™ Fairness Analyzer", "version": "1.0.0"}
