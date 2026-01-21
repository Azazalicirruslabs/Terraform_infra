import logging

# Configure the logger (adjust this based on your application's setup)
# This example sets up a basic logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the minimum level to log

# Optionally, add a handler to see the output (e.g., to console)
# If your main application already configures handlers, this might not be needed
if not logger.handlers:
    # Create handler
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

from datetime import datetime, timezone
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from scipy.stats import chi2_contingency, ks_2samp

from shared.auth import get_current_user

from ...shared.ai_explanation_service import ai_explanation_service
from ...shared.models import AnalysisRequest, MultiCohortAnalysisRequest
from ...shared.multi_cohort_utils import load_multiple_cohorts, validate_cohort_compatibility
from ...shared.s3_utils import load_s3_csv, validate_dataframe
from ..services.multi_cohort_drift_service import multi_cohort_analyzer

router = APIRouter(prefix="/data_drift", tags=["Data Drift - Statistical Reports"])


def psi(ref, curr, bins=10):
    """Population Stability Index (simplified)"""
    try:
        if len(ref) == 0 or len(curr) == 0:
            return 0.0
        ref_hist, bin_edges = np.histogram(ref, bins=bins)
        curr_hist, _ = np.histogram(curr, bins=bin_edges)
        ref_pct = ref_hist / max(np.sum(ref_hist), 1)
        curr_pct = curr_hist / max(np.sum(curr_hist), 1)
        curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        return float(np.sum((ref_pct - curr_pct) * np.log(ref_pct / curr_pct)))
    except:
        return 0.0


def create_ai_summary_for_statistical_analysis(analysis_data: dict) -> dict:
    """
    Summarizes the detailed statistical analysis into a compact format suitable for an LLM prompt.
    """
    summary = {
        "total_features": analysis_data.get("total_features", 0),
        "overall_drift_score": analysis_data.get("overall_drift_score"),
        "overall_status": analysis_data.get("overall_status"),
        "data_quality_score": analysis_data.get("data_quality_score"),
        "executive_summary": analysis_data.get("executive_summary", ""),
    }

    # Add summary statistics instead of full feature arrays
    summary_stats = analysis_data.get("summary_stats", {})
    summary.update(
        {
            "high_drift_features": summary_stats.get("high_drift_features", 0),
            "medium_drift_features": summary_stats.get("medium_drift_features", 0),
            "low_drift_features": summary_stats.get("low_drift_features", 0),
            "significant_ks_tests": summary_stats.get("significant_ks_tests", 0),
            "significant_chi_tests": summary_stats.get("significant_chi_tests", 0),
        }
    )

    # Add top 5 most drifted features only
    feature_analysis = analysis_data.get("feature_analysis", [])
    sorted_features = sorted(feature_analysis, key=lambda x: x.get("drift_score", 0), reverse=True)

    top_drifted_features = []
    for feature in sorted_features[:5]:
        top_drifted_features.append(
            {
                "feature": feature.get("feature"),
                "data_type": feature.get("data_type"),
                "drift_score": round(feature.get("drift_score", 0), 3),
                "status": feature.get("status"),
                "p_value": round(feature.get("p_value", 1), 4),
                "ks_statistic": round(feature.get("ks_statistic", 0), 3),
            }
        )

    summary["top_drifted_features"] = top_drifted_features

    return summary


@router.post("/statistical-reports")
async def get_statistical_reports(
    request: Union[AnalysisRequest, MultiCohortAnalysisRequest] = Body(...),
    user: dict = Depends(get_current_user),
    selected_features: Optional[List[str]] = Query(None, description="Filter by specific features"),
    selected_tests: Optional[List[str]] = Query(
        None, description="Filter by specific tests (KS, PSI, KL, JS, Wasserstein, Chi-square)"
    ),
    cohort_pairs: Optional[List[str]] = Query(
        None, description="Filter by specific cohort pairs (e.g., 'baseline-cohort_1')"
    ),
):
    """
    Get statistical reports analysis for datasets loaded from S3.

    Supports two formats:
    1. Legacy: Single reference vs single current comparison (AnalysisRequest)
    2. Multi-Cohort: 1 baseline vs up to 4 current cohorts (MultiCohortAnalysisRequest)

    Args:
        request: Either AnalysisRequest (legacy) or MultiCohortAnalysisRequest (multi-cohort)
        selected_features: Optional list of features to filter results
        selected_tests: Optional list of statistical tests to filter results
        cohort_pairs: Optional list of cohort pairs to filter results

    Returns:
        Statistical reports analysis results with all test details
    """
    # Detect request format and route to appropriate handler
    if hasattr(request, "files"):
        # Multi-cohort format
        return await _handle_multi_cohort_statistical(request, user, selected_features, selected_tests, cohort_pairs)
    else:
        # Legacy format
        return await _handle_legacy_statistical(request, user)


async def _handle_legacy_statistical(request: AnalysisRequest, user: dict):
    """
    Handle legacy single reference vs single current comparison.
    Preserves exact original functionality.
    """
    try:
        # Load data from S3 URLs
        reference_df = load_s3_csv(request.reference_url)
        current_df = load_s3_csv(request.current_url)

        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")

        feature_analysis_list = []
        ks_tests = []
        chi_tests = []

        # Only analyze common columns
        common_columns = list(set(reference_df.columns) & set(current_df.columns))

        if not common_columns:
            raise HTTPException(status_code=400, detail="No common columns found between datasets")

        for col in common_columns:
            try:
                ref_series = reference_df[col].dropna()
                curr_series = current_df[col].dropna()

                # Skip if insufficient data
                if len(ref_series) == 0 or len(curr_series) == 0:
                    continue

                dtype = "numerical" if reference_df[col].dtype in ["int64", "float64"] else "categorical"
                missing_ref = int(reference_df[col].isna().sum())
                missing_curr = int(current_df[col].isna().sum())

                if dtype == "numerical":
                    # Use same approach as dashboard - KS test with p-value based severity
                    ks_stat, p_value = ks_2samp(ref_series, curr_series)

                    # Unified severity classification
                    if p_value < 0.01:
                        status = "high"
                    elif p_value < 0.05:
                        status = "medium"
                    else:
                        status = "low"

                    drift_score = abs(ks_stat) * 5  # Same scaling as dashboard

                    feature_stats = {
                        "feature": col,
                        "data_type": dtype,
                        "ref_mean": float(ref_series.mean()),
                        "ref_std": float(ref_series.std()),
                        "ref_min": float(ref_series.min()),
                        "ref_max": float(ref_series.max()),
                        "curr_mean": float(curr_series.mean()),
                        "curr_std": float(curr_series.std()),
                        "curr_min": float(curr_series.min()),
                        "curr_max": float(curr_series.max()),
                        "missing_values_ref": missing_ref,
                        "missing_values_current": missing_curr,
                        "drift_score": drift_score,
                        "ks_statistic": float(ks_stat),
                        "p_value": float(p_value),
                        "status": status,
                    }

                    ks_tests.append(
                        {
                            "feature": col,
                            "ks_statistic": float(ks_stat),
                            "p_value": float(p_value),
                            "result": "Significant" if p_value < 0.05 else "Not Significant",
                        }
                    )

                else:  # categorical
                    # Use same approach as dashboard - Chi-square with p-value based severity
                    ref_counts = reference_df[col].value_counts()
                    curr_counts = current_df[col].value_counts()

                    # Align categories
                    all_cats = ref_counts.index.union(curr_counts.index)
                    ref_aligned = ref_counts.reindex(all_cats, fill_value=0)
                    curr_aligned = curr_counts.reindex(all_cats, fill_value=0)

                    if len(all_cats) > 1 and (ref_aligned > 0).sum() > 0 and (curr_aligned > 0).sum() > 0:
                        contingency_table = np.array([ref_aligned.values, curr_aligned.values])
                        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                    else:
                        chi2_stat, p_value = 0.0, 1.0

                    # Unified severity classification
                    if p_value < 0.01:
                        status = "high"
                    elif p_value < 0.05:
                        status = "medium"
                    else:
                        status = "low"

                    drift_score = chi2_stat / 10  # Same scaling as dashboard

                    feature_stats = {
                        "feature": col,
                        "data_type": dtype,
                        "ref_unique_values": len(ref_counts),
                        "curr_unique_values": len(curr_counts),
                        "ref_mode": ref_series.mode().iloc[0] if len(ref_series.mode()) > 0 else None,
                        "curr_mode": curr_series.mode().iloc[0] if len(curr_series.mode()) > 0 else None,
                        "missing_values_ref": missing_ref,
                        "missing_values_current": missing_curr,
                        "drift_score": drift_score,
                        "chi2_statistic": float(chi2_stat),
                        "p_value": float(p_value),
                        "status": status,
                    }

                    chi_tests.append(
                        {
                            "feature": col,
                            "chi2_statistic": float(chi2_stat),
                            "p_value": float(p_value),
                            "result": "Significant" if p_value < 0.05 else "Not Significant",
                        }
                    )

                feature_analysis_list.append(feature_stats)

            except Exception as e:
                # Skip problematic features but log for debugging
                print(f"Warning: Could not analyze feature {col}: {e}")
                continue

        if len(feature_analysis_list) == 0:
            raise HTTPException(status_code=400, detail="No features could be analyzed")

        # Use same overall calculation approach as dashboard
        total_features = len(feature_analysis_list)
        high_drift_features = sum(1 for f in feature_analysis_list if f.get("status") == "high")
        medium_drift_features = sum(1 for f in feature_analysis_list if f.get("status") == "medium")
        low_drift_features = sum(1 for f in feature_analysis_list if f.get("status") == "low")

        # Overall drift score as average
        overall_drift_score = sum(f["drift_score"] for f in feature_analysis_list) / total_features

        # Overall status based on feature counts
        if high_drift_features > total_features * 0.3:  # >30% high drift features
            overall_status = "high"
        elif medium_drift_features + high_drift_features > total_features * 0.5:  # >50% medium+ drift
            overall_status = "medium"
        else:
            overall_status = "low"

        # Safer data quality score calculation
        total_cells = current_df.shape[0] * current_df.shape[1]
        missing_cells = sum(f.get("missing_values_current", 0) for f in feature_analysis_list)
        data_quality_score = 1 - (missing_cells / max(total_cells, 1))

        # Dynamic executive summary using consistent metrics
        count_high = high_drift_features
        count_medium = medium_drift_features
        count_low = low_drift_features
        executive_summary = (
            f"Analyzed {len(feature_analysis_list)} features: "
            f"{count_high} high drift, {count_medium} medium drift, {count_low} low drift. "
            f"Overall drift status: {overall_status.upper()}."
        )

        # Correlation analysis - safer implementation
        correlation_analysis = []
        numerical_cols = [
            c
            for c in common_columns
            if reference_df[c].dtype in ["int64", "float64"] and current_df[c].dtype in ["int64", "float64"]
        ]
        for i in range(len(numerical_cols)):
            for j in range(i + 1, min(len(numerical_cols), i + 21)):  # Limit to prevent too many correlations
                f1, f2 = numerical_cols[i], numerical_cols[j]
                try:
                    ref_corr = reference_df[f1].corr(reference_df[f2])
                    curr_corr = current_df[f1].corr(current_df[f2])
                    if pd.notna(ref_corr) and pd.notna(curr_corr):
                        correlation_analysis.append(
                            {
                                "feature1": f1,
                                "feature2": f2,
                                "correlation": float(ref_corr),
                                "drift_correlation": float(curr_corr),
                                "correlation_change": float(curr_corr - ref_corr),
                            }
                        )

                except:
                    continue

        result = {
            "status": "success",
            "data": {
                "feature_analysis": feature_analysis_list,
                "ks_tests": ks_tests,
                "chi_tests": chi_tests,
                "correlation_analysis": correlation_analysis,
                "total_features": len(feature_analysis_list),
                "overall_drift_score": round(overall_drift_score, 3),
                "overall_status": overall_status,
                "data_quality_score": round(data_quality_score, 3),
                "executive_summary": executive_summary,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "summary_stats": {
                    "high_drift_features": count_high,
                    "medium_drift_features": count_medium,
                    "low_drift_features": count_low,
                    "significant_ks_tests": len([t for t in ks_tests if t["result"] == "Significant"]),
                    "significant_chi_tests": len([t for t in chi_tests if t["result"] == "Significant"]),
                },
            },
        }

        # Generate AI explanation for the analysis results
        try:
            ai_summary_payload = create_ai_summary_for_statistical_analysis(result["data"])
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, analysis_type="statistical_analysis"
            )
            result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            # Continue without AI explanation
            result["llm_response"] = {
                "summary": "Statistical drift analysis completed successfully.",
                "detailed_explanation": "The statistical analysis has been completed using various drift detection methods including KS tests and chi-square tests. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Statistical analysis completed",
                    "Review drift metrics for insights",
                    "AI explanations will return when service is restored",
                ],
            }

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistical reports analysis failed: {str(e)}")


async def _handle_multi_cohort_statistical(
    request: MultiCohortAnalysisRequest,
    user: dict,
    selected_features: Optional[List[str]] = None,
    selected_tests: Optional[List[str]] = None,
    cohort_pairs: Optional[List[str]] = None,
):
    """
    Handle multi-cohort statistical analysis (1 baseline vs up to 4 current cohorts).

    Returns detailed test results for all feature-test-cohort combinations.
    Supports query parameter filtering for features, tests, and cohort pairs.
    """
    try:
        # Load all cohorts
        cohorts = load_multiple_cohorts(request.files)

        # Validate cohort compatibility
        validation_result = validate_cohort_compatibility(cohorts)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400, detail=f"Cohort validation failed: {', '.join(validation_result.errors)}"
            )

        # Identify baseline and current cohorts
        baseline_name = None
        baseline_df = None
        current_cohorts = {}

        for cohort_name, (df, info) in cohorts.items():
            if info.is_baseline:
                baseline_name = cohort_name
                baseline_df = df
            else:
                current_cohorts[cohort_name] = df

        if baseline_name is None:
            raise HTTPException(status_code=400, detail="No baseline cohort found")
        if len(current_cohorts) == 0:
            raise HTTPException(status_code=400, detail="No current cohorts found")

        # Get common features
        common_features = validation_result.common_features

        # Apply feature filter if specified
        if selected_features:
            common_features = [f for f in common_features if f in selected_features]

        if not common_features:
            raise HTTPException(status_code=400, detail="No common features found after applying filters")

        # Build list of cohort pairs to analyze
        pairs_to_analyze = []
        for cohort_name in current_cohorts.keys():
            pair_name = f"{baseline_name}-{cohort_name}"
            # Apply cohort pair filter if specified
            if cohort_pairs is None or pair_name in cohort_pairs:
                pairs_to_analyze.append((baseline_name, cohort_name, pair_name))

        if not pairs_to_analyze:
            raise HTTPException(status_code=400, detail="No cohort pairs match the specified filters")

        # Compute statistical tests for all feature-test-cohort combinations
        comparisons = []

        for baseline_name, cohort_name, pair_name in pairs_to_analyze:
            cohort_df = current_cohorts[cohort_name]

            for feature in common_features:
                # Compute pairwise drift with all 5 tests
                drift_result = multi_cohort_analyzer.compute_pairwise_drift(
                    baseline_df, cohort_df, feature, baseline_name, cohort_name
                )

                # Extract test results with plot data
                for test_name, test in drift_result.tests.items():
                    # Skip N/A tests or apply test filter if specified
                    if test.status == "N/A":
                        continue
                    if selected_tests and test_name not in selected_tests:
                        continue

                    comparisons.append(
                        {
                            "cohort_pair": pair_name,
                            "baseline_cohort": baseline_name,
                            "current_cohort": cohort_name,
                            "feature": feature,
                            "feature_type": drift_result.feature_type,
                            "test": test_name,
                            "score": test.statistic,
                            "p_value": test.p_value if test.p_value is not None else None,
                            "threshold": test.threshold,
                            "status": test.status,
                            "confidence": test.confidence,
                            "plot_data": test.plot_data,  # Include plot data for frontend visualization
                        }
                    )

        # Compute correlation analysis for all pairs
        correlation_analysis = []

        logger.info(f"Computing correlation analysis for {len(pairs_to_analyze)} cohort pair(s)")

        for baseline_name, cohort_name, pair_name in pairs_to_analyze:
            cohort_df = current_cohorts[cohort_name]

            # Get numerical columns common to both
            numerical_cols = [
                c
                for c in common_features
                if pd.api.types.is_numeric_dtype(baseline_df[c]) and pd.api.types.is_numeric_dtype(cohort_df[c])
            ]

            logger.info(f"Pair '{pair_name}': Found {len(numerical_cols)} numerical columns for correlation analysis")

            if len(numerical_cols) < 2:
                logger.warning(
                    f"Pair '{pair_name}': Need at least 2 numerical columns for correlation. Skipping correlation analysis for this pair."
                )
                continue

            correlation_count = 0
            for i in range(len(numerical_cols)):
                for j in range(i + 1, min(len(numerical_cols), i + 21)):  # Limit pairs
                    f1, f2 = numerical_cols[i], numerical_cols[j]
                    try:
                        # Compute correlations
                        baseline_corr = baseline_df[f1].corr(baseline_df[f2])
                        cohort_corr = cohort_df[f1].corr(cohort_df[f2])

                        # Check for valid correlations
                        if pd.notna(baseline_corr) and pd.notna(cohort_corr):
                            correlation_analysis.append(
                                {
                                    "cohort_pair": pair_name,
                                    "baseline_cohort": baseline_name,
                                    "current_cohort": cohort_name,
                                    "feature1": f1,
                                    "feature2": f2,
                                    "baseline_correlation": float(baseline_corr),
                                    "current_correlation": float(cohort_corr),
                                    "correlation_change": float(cohort_corr - baseline_corr),
                                }
                            )
                            correlation_count += 1
                        else:
                            logger.debug(
                                f"Pair '{pair_name}': Correlation between {f1} and {f2} resulted in NaN values (baseline={baseline_corr}, current={cohort_corr})"
                            )
                    except Exception as e:
                        logger.warning(f"Pair '{pair_name}': Failed to compute correlation between {f1} and {f2}: {e}")
                        continue

            logger.info(f"Pair '{pair_name}': Successfully computed {correlation_count} correlations")

        logger.info(f"Total correlation pairs computed: {len(correlation_analysis)}")

        # Provide test formulas for frontend tooltips
        test_formulas = {
            "KS": {
                "name": "Kolmogorov-Smirnov Test",
                "formula": "D = sup|F₁(x) - F₂(x)|",
                "description": "Measures maximum distance between empirical cumulative distribution functions",
                "threshold": "p-value < 0.05 indicates significant drift",
            },
            "PSI": {
                "name": "Population Stability Index",
                "formula": "PSI = Σ(actual% - expected%) × ln(actual% / expected%)",
                "description": "Measures shift in distribution by comparing bin proportions",
                "threshold": "PSI < 0.1: stable, 0.1-0.2: moderate, > 0.2: severe drift",
            },
            "KL": {
                "name": "Kullback-Leibler Divergence",
                "formula": "KL(P||Q) = Σ P(x) × log(P(x) / Q(x))",
                "description": "Measures how one probability distribution diverges from a reference distribution",
                "threshold": "KL < 0.1: stable, 0.1-0.3: moderate, > 0.3: severe drift",
            },
            "JS": {
                "name": "Jensen-Shannon Divergence",
                "formula": "JS(P||Q) = 0.5 × KL(P||M) + 0.5 × KL(Q||M), where M = 0.5(P+Q)",
                "description": "Symmetric and smoothed version of KL divergence, bounded between 0 and 1",
                "threshold": "JS < 0.1: stable, 0.1-0.3: moderate, > 0.3: severe drift",
            },
            "Wasserstein": {
                "name": "Wasserstein Distance (Earth Mover's Distance)",
                "formula": "W(P,Q) = inf E[||X-Y||] over all joint distributions of (X,Y)",
                "description": "Minimum 'cost' to transform one distribution into another (numerical features only)",
                "threshold": "Normalized: < 0.1: stable, 0.1-0.3: moderate, > 0.3: severe drift",
            },
            "Chi-square": {
                "name": "Chi-Square Test of Independence",
                "formula": "χ² = Σ(Observed - Expected)² / Expected",
                "description": "Tests independence between categorical variables using contingency tables",
                "threshold": "p-value < 0.05 indicates significant drift",
            },
        }

        # Build response
        response = {
            "status": "success",
            "data": {
                "comparisons": comparisons,
                "correlation_analysis": correlation_analysis,
                "test_formulas": test_formulas,
                "metadata": {
                    "total_comparisons": len(comparisons),
                    "total_features": len(common_features),
                    "total_cohort_pairs": len(pairs_to_analyze),
                    "baseline_cohort": baseline_name,
                    "current_cohorts": list(current_cohorts.keys()),
                    "filters_applied": {
                        "selected_features": selected_features,
                        "selected_tests": selected_tests,
                        "cohort_pairs": cohort_pairs,
                    },
                },
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

        # Generate AI explanation
        try:
            # Create compact summary for AI (exclude plot_data)
            ai_summary = {
                "total_comparisons": len(comparisons),
                "total_features": len(common_features),
                "total_cohort_pairs": len(pairs_to_analyze),
                "baseline_cohort": baseline_name,
                "current_cohorts": list(current_cohorts.keys()),
                "tests_used": list(set(c["test"] for c in comparisons)),
                "drift_status_summary": {
                    "stable": sum(1 for c in comparisons if c["status"] == "stable"),
                    "moderate": sum(1 for c in comparisons if c["status"] == "moderate"),
                    "severe": sum(1 for c in comparisons if c["status"] == "severe"),
                },
                "top_drifted_features": sorted(
                    [
                        (c["feature"], c["score"], c["test"])
                        for c in comparisons
                        if c["status"] in ["moderate", "severe"]
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )[
                    :10
                ],  # Top 10 most drifted feature-test combinations
            }

            llm_response = ai_explanation_service.generate_explanation(ai_summary, "multi_cohort_statistical")
            response["llm_response"] = llm_response
            logger.info("AI explanation generated successfully for statistical reports")
        except Exception as e:
            logger.warning(f"AI explanation generation failed: {e}")
            # Add fallback explanation
            response["llm_response"] = {
                "summary": "Multi-cohort statistical analysis completed. AI explanations are temporarily unavailable.",
                "detailed_explanation": "Your statistical drift analysis has been completed with all requested tests. Review test results and plot data for insights.",
                "key_takeaways": [
                    "Statistical tests completed successfully",
                    "Review drift scores and confidence levels",
                    "AI-powered insights will return when service is restored",
                ],
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-cohort statistical analysis failed: {str(e)}")
