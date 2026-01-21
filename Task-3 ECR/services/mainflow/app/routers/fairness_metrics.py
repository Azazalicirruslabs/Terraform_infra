"""
BiasLens™ Analyzer - Fairness Metrics Configuration Router

This module provides the API endpoint for retrieving available fairness metrics information.

Endpoints:
    GET /mainflow/fairness/metrics - Get available fairness metrics
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/mainflow/fairness", tags=["BiasLens Fairness"])


@router.get("/metrics", status_code=200)
async def get_available_metrics() -> Dict[str, Any]:
    """
    Get information about available fairness metrics.

    Returns detailed descriptions of all fairness metrics supported by BiasLens™ Analyzer,
    including their formulas, interpretation guidelines, and threshold behavior.

    Returns:
        Dictionary containing metric metadata

    Example Response:
        ```json
        {
            "metrics": {
                "equal_opportunity": {
                    "name": "Equal Opportunity",
                    "description": "Measures parity in True Positive Rates...",
                    "formula": "(min_TPR / max_TPR) × 100",
                    "range": "0-100",
                    "interpretation": "100 = perfect parity, lower = greater disparity"
                },
                ...
            }
        }
        ```
    """
    logger.debug("Fetching available fairness metrics metadata")

    return {
        "metrics": {
            "equal_opportunity": {
                "name": "Equal Opportunity",
                "description": "Measures parity in True Positive Rates (TPR) between protected and non-protected groups",
                "formula": "(min_TPR / max_TPR) × 100",
                "range": "0-100",
                "interpretation": "100 = perfect parity, lower values indicate greater disparity in opportunity",
                "threshold_behavior": "Higher is better. Compares against acceptable/warning/breach thresholds.",
            },
            "disparate_impact": {
                "name": "Disparate Impact",
                "description": "Four-fifths rule - ratio of favorable outcome rates between groups",
                "formula": "(Protected favorable rate / Non-protected favorable rate) × 100",
                "range": "0-∞ (typically 0-200)",
                "interpretation": "100 = equal rates, <80 may indicate adverse impact, >125 may indicate reverse impact",
                "threshold_behavior": "Closer to 100 is better. Evaluated as min(DI, 200-DI) against thresholds.",
            },
            "statistical_parity": {
                "name": "Statistical Parity",
                "description": "Demographic parity - absolute difference in positive prediction rates",
                "formula": "|P(Ŷ=1|Protected) - P(Ŷ=1|Non-protected)| × 100",
                "range": "0-100",
                "interpretation": "0 = perfect parity, higher values indicate greater disparity",
                "threshold_behavior": "Lower is better. Inverted for comparison (100 - SP) against thresholds.",
            },
            "equalized_odds": {
                "name": "Equalized Odds",
                "description": "Measures equality of TPR and FPR between groups",
                "formula": "min(TPR_ratio, FPR_ratio) × 100",
                "range": "0-100",
                "interpretation": "100 = perfect equality in both TPR and FPR, lower = greater disparity",
                "threshold_behavior": "Higher is better. Compares against acceptable/warning/breach thresholds.",
            },
            "group_performance_comparison": {
                "name": "Group Performance Comparison",
                "description": "Compares standard ML metrics (accuracy, precision, recall, F1) between groups",
                "metrics_included": ["accuracy", "precision", "recall", "f1_score"],
                "interpretation": "Identifies performance gaps between protected and non-protected groups",
            },
        },
        "default_thresholds": {"acceptable_threshold": 80.0, "warning_threshold": 60.0, "breach_threshold": 40.0},
    }
