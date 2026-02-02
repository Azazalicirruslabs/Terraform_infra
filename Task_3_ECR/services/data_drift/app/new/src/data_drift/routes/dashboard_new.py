import logging

from fastapi import APIRouter, Body, Depends, HTTPException

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
from typing import Union

import numpy as np
from scipy.stats import chi2_contingency, entropy, ks_2samp

from shared.auth import get_current_user

from ...shared.ai_explanation_service import ai_explanation_service
from ...shared.models import AnalysisRequest, MultiCohortAnalysisRequest
from ...shared.multi_cohort_utils import load_multiple_cohorts, validate_cohort_compatibility
from ...shared.s3_utils import load_s3_csv, validate_dataframe
from ..services.multi_cohort_drift_service import multi_cohort_analyzer

router = APIRouter(prefix="/data_drift", tags=["Data Drift - Dashboard"])


def compute_kl_divergence(p, q):
    """Compute KL divergence between two probability distributions"""
    # Add small constant to avoid log(0)
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    return entropy(p, q)


def create_ai_summary_for_dashboard(analysis_data: dict) -> dict:
    """
    Summarizes the detailed dashboard analysis into a compact format suitable for an LLM prompt.
    This prevents token limit issues by sending only high-level insights instead of raw data.
    """
    # Extract top-level KPIs
    summary = {
        "overall_status": analysis_data.get("overall_status"),
        "overall_drift_score": round(analysis_data.get("overall_drift_score", 0), 2),
        "total_features_analyzed": analysis_data.get("total_features"),
        "high_drift_features_count": analysis_data.get("high_drift_features"),
        "medium_drift_features_count": analysis_data.get("medium_drift_features"),
        "data_quality_score": round(analysis_data.get("data_quality_score", 0), 2),
        "executive_summary": analysis_data.get("executive_summary", ""),
    }

    # Extract info for ONLY the top N most drifted features
    # This is the most important step to reduce token count
    feature_analysis = analysis_data.get("feature_analysis", [])

    # Sort features by drift score to find the most impactful ones
    sorted_features = sorted(feature_analysis, key=lambda x: x.get("drift_score", 0), reverse=True)

    top_n = 5  # Limit to top 5 features to keep prompt manageable
    top_drifted_features_summary = []

    for feature in sorted_features[:top_n]:
        feature_summary = {
            "feature_name": feature.get("feature"),
            "drift_status": feature.get("status"),
            "drift_score": round(feature.get("drift_score", 0), 2),
            "feature_type": feature.get("feature_type", "unknown"),
        }

        # Add a simple change description without raw distribution data
        if feature.get("status") in ["high", "critical"]:
            if feature.get("feature_type") == "numerical":
                ref_mean = feature.get("ref_mean", 0)
                curr_mean = feature.get("curr_mean", 0)
                if ref_mean != 0:
                    pct_change = round(((curr_mean - ref_mean) / ref_mean) * 100, 1)
                    feature_summary["change_description"] = f"Mean changed by {pct_change}%"
            elif feature.get("feature_type") == "categorical":
                feature_summary["change_description"] = "Category distribution has shifted significantly"

        top_drifted_features_summary.append(feature_summary)

    summary["top_drifted_features"] = top_drifted_features_summary

    # Add recommendations from the original analysis
    summary["recommendations"] = analysis_data.get("recommendations", [])

    return summary


@router.post("/dashboard")
async def get_drift_dashboard(
    request: Union[AnalysisRequest, MultiCohortAnalysisRequest] = Body(
        ...,
        examples={
            "legacy_format": {
                "summary": "Legacy Format (Single Pair)",
                "description": "Compare one reference dataset against one current dataset",
                "value": {
                    "reference_url": "https://s3.amazonaws.com/bucket/ref_data.csv",
                    "current_url": "https://s3.amazonaws.com/bucket/current_data.csv",
                    "target_column": "target",
                    "config": {},
                },
            },
            "multi_cohort_format": {
                "summary": "Multi-Cohort Format (1 Baseline + Up to 4 Cohorts)",
                "description": "Compare one baseline (ref_*) against multiple current cohorts (cur_*, cohort_*)",
                "value": {
                    "files": [
                        {
                            "file_name": "ref_cohort_1.csv",
                            "folder": "files",
                            "url": "https://s3.amazonaws.com/bucket/ref_cohort_1.csv",
                        },
                        {
                            "file_name": "cur_baseline.csv",
                            "folder": "files",
                            "url": "https://s3.amazonaws.com/bucket/cur_baseline.csv",
                        },
                        {
                            "file_name": "cohort_2.csv",
                            "folder": "files",
                            "url": "https://s3.amazonaws.com/bucket/cohort_2.csv",
                        },
                    ],
                    "target_column": "target",
                    "config": {},
                },
            },
        },
    ),
    user: dict = Depends(get_current_user),
):
    """
    Get drift dashboard analysis for datasets loaded from S3.

    Supports two formats:

    **1. Legacy Format (AnalysisRequest):** Single reference vs single current comparison
    ```json
    {
        "reference_url": "https://s3.amazonaws.com/...",
        "current_url": "https://s3.amazonaws.com/...",
        "target_column": "optional",
        "config": {}
    }
    ```

    **2. Multi-Cohort Format (MultiCohortAnalysisRequest):** 1 baseline vs up to 4 cohorts
    ```json
    {
        "files": [
            {"file_name": "ref_cohort_1.csv", "folder": "files", "url": "https://..."},
            {"file_name": "cur_baseline.csv", "folder": "files", "url": "https://..."},
            {"file_name": "cohort_2.csv", "folder": "files", "url": "https://..."}
        ],
        "target_column": "optional",
        "config": {}
    }
    ```

    **File Naming Convention:**
    - `ref_*` or `ref_*.csv` → Baseline/Reference cohort
    - `cur_*` or `cohort_*` → Current cohorts

    Args:
        request: Either AnalysisRequest or MultiCohortAnalysisRequest

    Returns:
        Dashboard analysis results (format depends on request type)
    """
    try:
        # Detect request format and route to appropriate handler
        if hasattr(request, "files"):
            # New multi-cohort format
            logger.info(f"Dashboard request: Multi-cohort format detected with {len(request.files)} files")
            return await _handle_multi_cohort_dashboard(request, user)
        else:
            # Legacy single-pair format
            logger.info("Dashboard request: Legacy format detected (single reference vs current)")
            return await _handle_legacy_dashboard(request, user)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dashboard analysis failed: {str(e)}")


async def _handle_legacy_dashboard(request: AnalysisRequest, user: dict):
    """Handle legacy single reference vs single current comparison"""
    try:
        # Load data from S3 URLs
        reference_df = load_s3_csv(request.reference_url)
        current_df = load_s3_csv(request.current_url)

        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")

        drifted_features = []
        feature_analysis_list = []

        # Only analyze common columns
        common_columns = list(set(reference_df.columns) & set(current_df.columns))

        if not common_columns:
            raise HTTPException(status_code=400, detail="No common columns found between datasets")

        for col in common_columns:
            if reference_df[col].dtype in ["int64", "float64"]:
                # Numerical features - use KS test
                ref_vals = reference_df[col].dropna()
                curr_vals = current_df[col].dropna()

                if len(ref_vals) == 0 or len(curr_vals) == 0:
                    continue

                ks_stat, p_value = ks_2samp(ref_vals, curr_vals)

                # Unified severity classification based on p-value
                if p_value < 0.01:
                    drift_status = "high"
                elif p_value < 0.05:
                    drift_status = "medium"
                else:
                    drift_status = "low"

                drift_score = abs(ks_stat) * 5  # Scale for display

                # Compute histograms for visualization
                bins = np.histogram_bin_edges(ref_vals, bins="auto")
                ref_hist, _ = np.histogram(ref_vals, bins=bins, density=True)
                curr_hist, _ = np.histogram(curr_vals, bins=bins, density=True)

                distribution_ref = (ref_hist * 100).tolist()
                distribution_current = (curr_hist * 100).tolist()
                bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)]

                feature_analysis_list.append(
                    {
                        "feature": col,
                        "feature_type": "numerical",
                        "drift_score": drift_score,
                        "status": drift_status,
                        "p_value": p_value,
                        "ks_statistic": float(ks_stat),
                        "distribution_ref": distribution_ref,
                        "distribution_current": distribution_current,
                        "bin_labels": bin_labels,
                    }
                )

            else:
                # Categorical features - use Chi-square test
                try:
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

                    # Unified severity classification based on p-value
                    if p_value < 0.01:
                        drift_status = "high"
                    elif p_value < 0.05:
                        drift_status = "medium"
                    else:
                        drift_status = "low"

                    drift_score = chi2_stat / 10  # Scale for display

                    distribution_ref = ref_counts.to_dict()
                    distribution_current = curr_counts.to_dict()

                    feature_analysis_list.append(
                        {
                            "feature": col,
                            "feature_type": "categorical",
                            "drift_score": drift_score,
                            "status": drift_status,
                            "p_value": p_value,
                            "chi2_statistic": float(chi2_stat),
                            "distribution_ref": distribution_ref,
                            "distribution_current": distribution_current,
                        }
                    )

                except Exception as e:
                    continue

            if drift_status in ["high", "medium"]:
                drifted_features.append(col)

        if len(feature_analysis_list) == 0:
            raise HTTPException(status_code=400, detail="No features could be analyzed")

        # Calculate overall metrics using consistent approach
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

        top_features = ", ".join(drifted_features[:2]) if drifted_features else "no significant features"
        executive_summary = (
            f"Analysis shows {overall_status}-level drift primarily driven by {top_features}. "
            "The model performance may be impacted and retraining should be considered within the next quarter."
        )

        data_quality_score = reference_df.notnull().mean().mean()

        result = {
            "status": "success",
            "data": {
                "high_drift_features": high_drift_features,
                "medium_drift_features": medium_drift_features,
                "data_quality_score": float(data_quality_score),
                "total_features": total_features,
                "overall_drift_score": round(overall_drift_score, 2),
                "executive_summary": executive_summary,
                "overall_status": overall_status,
                "analysis_timestamp": datetime.now(timezone.utc).strftime("%d/%m/%Y"),
                "feature_analysis": feature_analysis_list,
                "recommendations": [
                    "Monitor features with high drift closely",
                    "Retrain model if drift persists",
                ],
            },
        }

        # Generate AI explanation for the dashboard analysis
        try:
            # *** NEW STEP: Create the summary FIRST ***
            ai_summary_payload = create_ai_summary_for_dashboard(result["data"])

            ai_explanation = ai_explanation_service.generate_explanation(
                # *** CHANGE: Send the summary, NOT the full result["data"] ***
                analysis_data=ai_summary_payload,
                analysis_type="data_drift_dashboard",
            )
            result["llm_response"] = ai_explanation
        except Exception as e:
            print(f"Warning: AI explanation failed: {e}")
            # Continue without AI explanation
            result["llm_response"] = {
                "summary": "Data drift dashboard analysis completed successfully.",
                "detailed_explanation": "Your comprehensive data drift dashboard has been generated, showing drift patterns across all features. AI explanations are temporarily unavailable.",
                "key_takeaways": [
                    "Dashboard analysis completed successfully",
                    "Review drift scores and feature patterns",
                    "AI explanations will return when service is restored",
                ],
            }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Legacy dashboard analysis failed: {str(e)}")


async def _handle_multi_cohort_dashboard(request: MultiCohortAnalysisRequest, user: dict):
    """Handle new multi-cohort dashboard analysis (1 baseline vs up to 4 cohorts)"""
    import logging

    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Multi-cohort dashboard request for {len(request.files)} files")

        # Load all cohorts
        cohorts = load_multiple_cohorts(request.files)

        # Validate cohorts
        validation_result = validate_cohort_compatibility(cohorts)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400, detail=f"Cohort validation failed: {', '.join(validation_result.errors)}"
            )

        common_features = validation_result.common_features
        logger.info(f"Found {len(common_features)} common features")

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

        if baseline_name is None or baseline_df is None:
            raise HTTPException(status_code=400, detail="No baseline cohort found")

        if not current_cohorts:
            raise HTTPException(status_code=400, detail="No current cohorts found")

        logger.info(f"Baseline: {baseline_name}, Current cohorts: {list(current_cohorts.keys())}")

        # Compute drift for each baseline-cohort pair
        cohort_summaries = []

        for cohort_name, cohort_df in current_cohorts.items():
            logger.info(f"Analyzing drift: {baseline_name} vs {cohort_name}")

            # Compute pairwise drift for all features
            pairwise_results = []
            for feature in common_features:
                try:
                    result = multi_cohort_analyzer.compute_pairwise_drift(
                        baseline_df, cohort_df, feature, baseline_name, cohort_name
                    )
                    pairwise_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to compute drift for feature {feature}: {e}")

            # Aggregate results for this cohort
            aggregated = multi_cohort_analyzer.aggregate_drift_across_features(pairwise_results)

            # Build cohort summary
            cohort_summary = {
                "cohort_name": cohort_name,
                "total_features": aggregated["total_features"],
                "drifted_features": aggregated["drifted_features"],
                "drift_percentage": aggregated["drift_percentage"],
                "overall_drift_score": aggregated["overall_drift_score"],
                "overall_status": aggregated["overall_status"],
                "tests_summary": aggregated["tests_summary"],
                "severity_breakdown": aggregated["severity_breakdown"],
            }

            cohort_summaries.append(cohort_summary)

        # Compute aggregated summary across all cohorts
        avg_drift_percentage = np.mean([c["drift_percentage"] for c in cohort_summaries])

        # Find most and least drifted cohorts
        most_drifted = max(cohort_summaries, key=lambda x: x["drift_percentage"])
        least_drifted = min(cohort_summaries, key=lambda x: x["drift_percentage"])

        # Find consistently drifted features (drifted in majority of cohorts)
        feature_drift_counts = {}
        for pairwise_result in pairwise_results:
            if pairwise_result.overall_status in ["moderate", "severe"]:
                feature_name = pairwise_result.feature_name
                feature_drift_counts[feature_name] = feature_drift_counts.get(feature_name, 0) + 1

        consistently_drifted = [
            feat for feat, count in feature_drift_counts.items() if count >= len(current_cohorts) / 2
        ]

        # Determine which tests are most sensitive (detect drift most frequently)
        test_sensitivity = {}
        for cohort_summary in cohort_summaries:
            for test_name, test_info in cohort_summary["tests_summary"].items():
                if test_name not in test_sensitivity:
                    test_sensitivity[test_name] = {"total_drifted": 0, "total_tested": 0}
                test_sensitivity[test_name]["total_drifted"] += test_info.get("drifted_count", 0)
                test_sensitivity[test_name]["total_tested"] += test_info.get("total_tested", 0)

        # Sort tests by sensitivity
        tests_most_sensitive = sorted(
            test_sensitivity.keys(),
            key=lambda t: test_sensitivity[t]["total_drifted"] / max(test_sensitivity[t]["total_tested"], 1),
            reverse=True,
        )[
            :3
        ]  # Top 3 most sensitive tests

        # Build aggregated summary
        aggregated_summary = {
            "avg_drift_percentage": float(avg_drift_percentage),
            "most_drifted_cohort": most_drifted["cohort_name"],
            "least_drifted_cohort": least_drifted["cohort_name"],
            "consistently_drifted_features": consistently_drifted,
            "tests_most_sensitive": tests_most_sensitive,
        }

        # Build response
        response = {
            "status": "success",
            "data": {
                "analysis_type": "multi_cohort",
                "baseline_cohort": baseline_name,
                "total_features": len(common_features),
                "cohort_summaries": cohort_summaries,
                "aggregated_summary": aggregated_summary,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

        # Generate AI explanation
        try:
            # Create compact summary for AI (exclude large data structures)
            ai_summary = {
                "total_cohorts": len(current_cohorts),
                "baseline_cohort": baseline_name,
                "total_features": len(common_features),
                "avg_drift_percentage": aggregated_summary["avg_drift_percentage"],
                "most_drifted_cohort": aggregated_summary["most_drifted_cohort"],
                "least_drifted_cohort": aggregated_summary["least_drifted_cohort"],
                "consistently_drifted_features": aggregated_summary["consistently_drifted_features"],
                "tests_most_sensitive": aggregated_summary["tests_most_sensitive"],
                "cohort_summaries": [
                    {
                        "cohort_name": cs["cohort_name"],
                        "drift_percentage": cs["drift_percentage"],
                        "overall_status": cs["overall_status"],
                        "drifted_features": cs["drifted_features"],
                    }
                    for cs in cohort_summaries
                ],
            }

            llm_response = ai_explanation_service.generate_explanation(ai_summary, "multi_cohort_dashboard")
            response["llm_response"] = llm_response
            logger.info("AI explanation generated successfully")
        except Exception as e:
            logger.warning(f"AI explanation generation failed: {e}")
            # Add fallback explanation
            response["llm_response"] = {
                "summary": "Multi-cohort drift analysis completed. AI explanations are temporarily unavailable.",
                "detailed_explanation": "Your multi-cohort drift dashboard analysis has been completed successfully. Review cohort summaries to identify drift patterns.",
                "key_takeaways": [
                    "Multi-cohort analysis completed successfully",
                    "Review cohort summaries for drift insights",
                    "AI-powered insights will return when service is restored",
                ],
            }

        logger.info(f"Multi-cohort dashboard completed for {len(current_cohorts)} cohorts")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-cohort dashboard failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-cohort dashboard analysis failed: {str(e)}")
