"""
Data Overview Endpoint
Provides metadata and summary statistics for all uploaded cohorts
"""

import logging
from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException

from shared.auth import get_current_user

from ...shared.models import MultiCohortAnalysisRequest
from ...shared.multi_cohort_utils import (
    check_data_quality,
    compute_cohort_metadata,
    load_multiple_cohorts,
    validate_cohort_compatibility,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data_drift", tags=["Data Drift - Data Overview"])


@router.post("/data_overview")
async def get_data_overview(request: MultiCohortAnalysisRequest, user: Dict = Depends(get_current_user)):
    """
    Get comprehensive overview of all uploaded cohorts.

    This endpoint provides:
    - Metadata for each cohort (shape, features, missing values, duplicates)
    - Common features across all cohorts
    - Data quality assessment
    - Validation warnings

    Args:
        request: MultiCohortAnalysisRequest with file list

    Returns:
        Dictionary with cohort metadata and data quality metrics
    """
    try:
        logger.info(f"Data overview request received for {len(request.files)} files")

        # Load all cohorts
        cohorts = load_multiple_cohorts(request.files)
        logger.info(f"Successfully loaded {len(cohorts)} cohorts")

        # Validate cohort compatibility
        validation_result = validate_cohort_compatibility(cohorts)

        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400, detail=f"Cohort validation failed: {', '.join(validation_result.errors)}"
            )

        # Compute metadata for each cohort
        cohorts_metadata = []
        for file_info in request.files:
            # Find matching cohort
            cohort_entry = None
            for cohort_name, (df, info) in cohorts.items():
                if info.original_filename == file_info.file_name:
                    cohort_entry = (cohort_name, df, info)
                    break

            if cohort_entry:
                cohort_name, df, cohort_info = cohort_entry
                metadata = compute_cohort_metadata(
                    cohort_name=cohort_name, df=df, url=file_info.url, file_name=file_info.file_name
                )
                cohorts_metadata.append(metadata.dict())

        # Check data quality
        quality_report = check_data_quality(cohorts)

        # Build response
        response = {
            "status": "success",
            "data": {
                "cohorts": cohorts_metadata,
                "common_features": validation_result.common_features,
                "total_cohorts": len(cohorts_metadata),
                "baseline_cohort": next(
                    (c["cohort_name"] for c in cohorts_metadata if c["cohort_type"] == "baseline"), None
                ),
                "current_cohorts": [c["cohort_name"] for c in cohorts_metadata if c["cohort_type"] == "current"],
                "validation": {
                    "is_valid": validation_result.is_valid,
                    "warnings": validation_result.warnings,
                    "missing_features": validation_result.missing_features,
                    "data_type_mismatches": validation_result.data_type_mismatches,
                },
                "data_quality": quality_report,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

        # Generate AI explanation
        try:
            from ...shared.ai_explanation_service import ai_explanation_service

            # Create compact summary for AI
            ai_summary = {
                "total_cohorts": len(cohorts_metadata),
                "baseline_cohort": response["data"]["baseline_cohort"],
                "current_cohorts": response["data"]["current_cohorts"],
                "total_common_features": len(validation_result.common_features),
                "validation_status": "valid" if validation_result.is_valid else "invalid",
                "validation_warnings_count": len(validation_result.warnings),
                "overall_data_quality": quality_report.get("overall_quality", "unknown"),
                "quality_warnings_count": len(quality_report.get("warnings", [])),
                "cohort_summaries": [
                    {
                        "cohort_name": c["cohort_name"],
                        "cohort_type": c["cohort_type"],
                        "total_rows": c["total_rows"],
                        "total_features": c["total_features"],
                        "duplicate_rows": c["duplicate_rows"],
                    }
                    for c in cohorts_metadata
                ],
            }

            llm_response = ai_explanation_service.generate_explanation(ai_summary, "multi_cohort_data_overview")
            response["llm_response"] = llm_response
            logger.info("AI explanation generated successfully for data overview")
        except Exception as e:
            logger.warning(f"AI explanation generation failed: {e}")
            # Add fallback explanation
            response["llm_response"] = {
                "summary": "Multi-cohort data overview completed. AI explanations are temporarily unavailable.",
                "detailed_explanation": "Your data quality overview has been completed. Review cohort metadata and validation results for insights.",
                "key_takeaways": [
                    "Data overview completed successfully",
                    "Review validation status and data quality metrics",
                    "AI-powered insights will return when service is restored",
                ],
            }

        logger.info(f"Data overview completed for {len(cohorts_metadata)} cohorts")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data overview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data overview analysis failed: {str(e)}")
