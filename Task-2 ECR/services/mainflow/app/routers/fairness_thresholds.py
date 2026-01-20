"""
BiasLens™ Analyzer - Fairness Thresholds Configuration Router

This module provides the API endpoint for retrieving default threshold values.

Endpoints:
    GET /mainflow/fairness/default-thresholds - Get default threshold values
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter

from services.mainflow.app.core.fairness_summary import FairnessThresholds

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/mainflow/fairness", tags=["BiasLens Fairness"])


@router.get("/default-thresholds", status_code=200)
async def get_default_thresholds() -> Dict[str, Any]:
    """
    Get default threshold values for fairness assessment.

    Returns the standard threshold configuration used when no custom thresholds are provided.
    These values define the boundaries between acceptable, warning, and breach zones.

    Returns:
        Dictionary with default threshold values

    Example Response:
        ```json
        {
            "acceptable_threshold": 80.0,
            "warning_threshold": 60.0,
            "breach_threshold": 40.0,
            "description": "Metric >= 80: Acceptable, 60-80: Warning, <= 40: Breach"
        }
        ```
    """
    logger.debug("Fetching default threshold configuration")

    default_thresholds = FairnessThresholds()

    return {
        "acceptable_threshold": default_thresholds.acceptable,
        "warning_threshold": default_thresholds.warning,
        "breach_threshold": default_thresholds.breach,
        "description": (
            f"Metric >= {default_thresholds.acceptable}: Acceptable, "
            f"{default_thresholds.breach}-{default_thresholds.acceptable}: Warning, "
            f"<= {default_thresholds.breach}: Breach"
        ),
    }
