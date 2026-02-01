"""
Cohort Comparison Endpoint
Provides detailed comparison between selected cohorts for a specific feature
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from shared.auth import get_current_user

from ...shared.models import CohortComparisonRequest
from ...shared.multi_cohort_utils import load_multiple_cohorts, validate_cohort_compatibility
from ..services.multi_cohort_drift_service import multi_cohort_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data_drift", tags=["Data Drift - Cohort Comparison"])


@router.post("/cohort_comparison")
async def compare_cohorts(request: CohortComparisonRequest, user: Dict = Depends(get_current_user)):
    """
    Compare selected cohorts for a specific feature.

    Provides:
    - Distribution overlay data (histograms for numerical, frequencies for categorical)
    - Central tendency metrics (mean, median, std, quartiles)
    - Pairwise drift matrix with all 5 statistical tests
    - Statistical comparison metrics (mean change, missing value change, etc.)

    Args:
        request: CohortComparisonRequest with selected cohorts and feature

    Returns:
        Dictionary with comparison data and drift metrics
    """
    try:
        logger.info(
            f"Cohort comparison request for feature '{request.feature}' across {len(request.selected_cohorts)} cohorts"
        )

        # Load all cohorts
        cohorts = load_multiple_cohorts(request.files)

        # Validate cohorts
        validation_result = validate_cohort_compatibility(cohorts)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400, detail=f"Cohort validation failed: {', '.join(validation_result.errors)}"
            )

        # Check if selected cohorts exist
        available_cohorts = set(cohorts.keys())
        requested_cohorts = set(request.selected_cohorts)
        missing_cohorts = requested_cohorts - available_cohorts

        if missing_cohorts:
            raise HTTPException(
                status_code=400,
                detail=f"Selected cohorts not found: {list(missing_cohorts)}. Available: {list(available_cohorts)}",
            )

        # Check if feature exists in all selected cohorts
        feature = request.feature
        for cohort_name in request.selected_cohorts:
            df, _ = cohorts[cohort_name]
            if feature not in df.columns:
                raise HTTPException(status_code=400, detail=f"Feature '{feature}' not found in cohort '{cohort_name}'")

        # Extract data for selected cohorts
        selected_data = {cohort_name: cohorts[cohort_name][0] for cohort_name in request.selected_cohorts}

        # Determine feature type
        first_cohort_df = list(selected_data.values())[0]
        is_numerical = pd.api.types.is_numeric_dtype(first_cohort_df[feature])
        feature_type = "numerical" if is_numerical else "categorical"

        logger.info(f"Feature '{feature}' type: {feature_type}")

        # Compute overlay data
        overlay_data = {}
        for cohort_name, df in selected_data.items():
            if is_numerical:
                overlay_data[cohort_name] = _compute_numerical_overlay(df[feature])
            else:
                overlay_data[cohort_name] = _compute_categorical_overlay(df[feature])

        # Compute central tendency
        central_tendency = {}
        for cohort_name, df in selected_data.items():
            if is_numerical:
                central_tendency[cohort_name] = _compute_numerical_central_tendency(df[feature])
            else:
                central_tendency[cohort_name] = _compute_categorical_central_tendency(df[feature])

        # Compute pairwise drift matrix for all combinations
        drift_matrix = []
        cohort_names = list(selected_data.keys())

        for i in range(len(cohort_names)):
            for j in range(i + 1, len(cohort_names)):
                cohort1_name = cohort_names[i]
                cohort2_name = cohort_names[j]

                df1 = selected_data[cohort1_name]
                df2 = selected_data[cohort2_name]

                # Compute pairwise drift
                drift_result = multi_cohort_analyzer.compute_pairwise_drift(
                    df1, df2, feature, cohort1_name, cohort2_name
                )

                # Extract test results
                test_results = {}
                for test_name, test in drift_result.tests.items():
                    if test.status != "N/A":
                        test_results[test_name] = {
                            "statistic": test.statistic,
                            "p_value": test.p_value,
                            "threshold": test.threshold,
                            "status": test.status,
                            "confidence": test.confidence,
                        }

                drift_matrix.append(
                    {
                        "pair": f"{cohort1_name}-{cohort2_name}",
                        "cohort1": cohort1_name,
                        "cohort2": cohort2_name,
                        "tests": test_results,
                        "overall_drift_score": drift_result.overall_drift_score,
                        "overall_status": drift_result.overall_status,
                    }
                )

        # Compute statistical comparison
        statistical_comparison = _compute_statistical_comparison(selected_data, feature, is_numerical)

        # Build response
        response = {
            "status": "success",
            "data": {
                "selected_cohorts": request.selected_cohorts,
                "feature": feature,
                "feature_type": feature_type,
                "overlay_data": overlay_data,
                "central_tendency": central_tendency,
                "drift_matrix": drift_matrix,
                "statistical_comparison": statistical_comparison,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

        # Generate AI explanation
        try:
            from ...shared.ai_explanation_service import ai_explanation_service

            # Create compact summary for AI (exclude overlay_data for brevity)
            ai_summary = {
                "feature": feature,
                "feature_type": feature_type,
                "selected_cohorts": request.selected_cohorts,
                "total_pairs": len(drift_matrix),
                "central_tendency": central_tendency,
                "drift_matrix_summary": [
                    {
                        "pair": dm["pair"],
                        "overall_status": dm["overall_status"],
                        "overall_drift_score": dm["overall_drift_score"],
                        "tests": {
                            test_name: {"status": test_info["status"], "confidence": test_info["confidence"]}
                            for test_name, test_info in dm["tests"].items()
                        },
                    }
                    for dm in drift_matrix
                ],
                "statistical_comparison_summary": statistical_comparison[:5],  # Top 5 metrics
            }

            llm_response = ai_explanation_service.generate_explanation(ai_summary, "multi_cohort_comparison")
            response["llm_response"] = llm_response
            logger.info("AI explanation generated successfully for cohort comparison")
        except Exception as e:
            logger.warning(f"AI explanation generation failed: {e}")
            # Add fallback explanation
            response["llm_response"] = {
                "summary": "Cohort comparison analysis completed. AI explanations are temporarily unavailable.",
                "detailed_explanation": f"Your detailed comparison for feature '{feature}' has been completed. Review distribution plots and drift test results for insights.",
                "key_takeaways": [
                    "Feature-level cohort comparison completed successfully",
                    "Review drift matrix for pairwise comparisons",
                    "AI-powered insights will return when service is restored",
                ],
            }

        logger.info(f"Cohort comparison completed for feature '{feature}'")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cohort comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cohort comparison failed: {str(e)}")


def _compute_numerical_overlay(series: pd.Series, bins: int = 30) -> Dict[str, Any]:
    """Compute histogram data for numerical feature overlay"""
    clean_data = series.dropna()

    if len(clean_data) == 0:
        return {"histogram": {"bins": [], "counts": []}, "raw_sample": []}

    hist, bin_edges = np.histogram(clean_data, bins=bins)

    return {
        "histogram": {
            "bins": bin_edges.tolist(),
            "counts": hist.tolist(),
            "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
        },
        "raw_sample": clean_data.sample(min(1000, len(clean_data))).tolist(),
    }


def _compute_categorical_overlay(series: pd.Series) -> Dict[str, Any]:
    """Compute frequency data for categorical feature overlay"""
    value_counts = series.value_counts()

    return {
        "frequencies": {
            "categories": value_counts.index.tolist(),
            "counts": value_counts.values.tolist(),
            "percentages": (value_counts / len(series) * 100).tolist(),
        },
        "raw_sample": series.sample(min(1000, len(series))).tolist(),
    }


def _compute_numerical_central_tendency(series: pd.Series) -> Dict[str, float]:
    """Compute central tendency metrics for numerical feature"""
    clean_data = series.dropna()

    if len(clean_data) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "missing_percent": 100.0,
        }

    return {
        "mean": float(clean_data.mean()),
        "median": float(clean_data.median()),
        "std": float(clean_data.std()),
        "min": float(clean_data.min()),
        "max": float(clean_data.max()),
        "q25": float(clean_data.quantile(0.25)),
        "q75": float(clean_data.quantile(0.75)),
        "missing_percent": float((series.isna().sum() / len(series)) * 100),
    }


def _compute_categorical_central_tendency(series: pd.Series) -> Dict[str, Any]:
    """Compute central tendency metrics for categorical feature"""
    value_counts = series.value_counts()

    mode_value = series.mode()[0] if len(series.mode()) > 0 else None

    return {
        "mode": str(mode_value) if mode_value is not None else None,
        "unique_count": int(series.nunique()),
        "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
        "most_frequent_percent": float((value_counts.iloc[0] / len(series)) * 100) if len(value_counts) > 0 else 0.0,
        "missing_percent": float((series.isna().sum() / len(series)) * 100),
    }


def _compute_statistical_comparison(
    cohorts_data: Dict[str, pd.DataFrame], feature: str, is_numerical: bool
) -> List[Dict[str, Any]]:
    """Compute statistical comparison metrics between cohorts"""
    comparison_metrics = []
    cohort_names = list(cohorts_data.keys())

    # Assume first cohort is baseline for comparison
    baseline_name = cohort_names[0]
    baseline_series = cohorts_data[baseline_name][feature]

    for cohort_name in cohort_names[1:]:
        cohort_series = cohorts_data[cohort_name][feature]

        if is_numerical:
            # Numerical metrics
            baseline_mean = baseline_series.mean()
            cohort_mean = cohort_series.mean()
            mean_change = cohort_mean - baseline_mean
            mean_change_pct = (mean_change / baseline_mean * 100) if baseline_mean != 0 else 0

            baseline_std = baseline_series.std()
            cohort_std = cohort_series.std()
            std_change = cohort_std - baseline_std
            std_change_pct = (std_change / baseline_std * 100) if baseline_std != 0 else 0

            baseline_missing = (baseline_series.isna().sum() / len(baseline_series)) * 100
            cohort_missing = (cohort_series.isna().sum() / len(cohort_series)) * 100
            missing_change = cohort_missing - baseline_missing
            missing_change_pct = (missing_change / baseline_missing * 100) if baseline_missing != 0 else 0

            comparison_metrics.extend(
                [
                    {
                        "metric": "mean",
                        "baseline_cohort": baseline_name,
                        "comparison_cohort": cohort_name,
                        "baseline_value": float(baseline_mean),
                        "comparison_value": float(cohort_mean),
                        "change_absolute": float(mean_change),
                        "change_percent": float(mean_change_pct),
                    },
                    {
                        "metric": "std",
                        "baseline_cohort": baseline_name,
                        "comparison_cohort": cohort_name,
                        "baseline_value": float(baseline_std),
                        "comparison_value": float(cohort_std),
                        "change_absolute": float(std_change),
                        "change_percent": float(std_change_pct),
                    },
                    {
                        "metric": "missing_percent",
                        "baseline_cohort": baseline_name,
                        "comparison_cohort": cohort_name,
                        "baseline_value": float(baseline_missing),
                        "comparison_value": float(cohort_missing),
                        "change_absolute": float(missing_change),
                        "change_percent": float(missing_change_pct),
                    },
                ]
            )
        else:
            # Categorical metrics
            baseline_unique = baseline_series.nunique()
            cohort_unique = cohort_series.nunique()
            unique_change = cohort_unique - baseline_unique
            unique_change_pct = (unique_change / baseline_unique * 100) if baseline_unique != 0 else 0

            baseline_missing = (baseline_series.isna().sum() / len(baseline_series)) * 100
            cohort_missing = (cohort_series.isna().sum() / len(cohort_series)) * 100
            missing_change = cohort_missing - baseline_missing
            missing_change_pct = (missing_change / baseline_missing * 100) if baseline_missing != 0 else 0

            comparison_metrics.extend(
                [
                    {
                        "metric": "unique_count",
                        "baseline_cohort": baseline_name,
                        "comparison_cohort": cohort_name,
                        "baseline_value": int(baseline_unique),
                        "comparison_value": int(cohort_unique),
                        "change_absolute": int(unique_change),
                        "change_percent": float(unique_change_pct),
                    },
                    {
                        "metric": "missing_percent",
                        "baseline_cohort": baseline_name,
                        "comparison_cohort": cohort_name,
                        "baseline_value": float(baseline_missing),
                        "comparison_value": float(cohort_missing),
                        "change_absolute": float(missing_change),
                        "change_percent": float(missing_change_pct),
                    },
                ]
            )

    # Add duplicate metrics for all cohorts
    baseline_duplicates = cohorts_data[baseline_name].duplicated().sum()
    baseline_dup_pct = (baseline_duplicates / len(cohorts_data[baseline_name])) * 100

    for cohort_name in cohort_names[1:]:
        cohort_duplicates = cohorts_data[cohort_name].duplicated().sum()
        cohort_dup_pct = (cohort_duplicates / len(cohorts_data[cohort_name])) * 100
        dup_change = cohort_dup_pct - baseline_dup_pct
        dup_change_pct = (dup_change / baseline_dup_pct * 100) if baseline_dup_pct != 0 else 0

        comparison_metrics.append(
            {
                "metric": "duplicate_percent",
                "baseline_cohort": baseline_name,
                "comparison_cohort": cohort_name,
                "baseline_value": float(baseline_dup_pct),
                "comparison_value": float(cohort_dup_pct),
                "change_absolute": float(dup_change),
                "change_percent": float(dup_change_pct),
            }
        )

    return comparison_metrics
