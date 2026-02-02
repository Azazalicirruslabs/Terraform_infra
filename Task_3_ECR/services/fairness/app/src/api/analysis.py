import logging

from fastapi import APIRouter, Depends, HTTPException  # pylint: disable=import-error

from services.fairness.app.src.hsic_detector import AnalysisEngine
from services.fairness.app.src.models import global_session
from services.fairness.app.src.schemas.request_response import (
    AnalysisRequest,
    AnalysisResponse,
    FeatureSensitivityScore,
)
from services.fairness.app.src.utils import FileManager
from services.fairness.app.utils.helper_functions import (
    get_s3_file_metadata,
    load_dataframe_from_url,
    validate_file_metadata,
)
from shared.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fairness/analysis", tags=["Analysis"])


@router.post("/run", response_model=AnalysisResponse)
async def run_fairness_analysis(
    request: AnalysisRequest, current_user: str = Depends(get_current_user), project_id: str = None
):
    """
    Run fairness analysis using HSIC Lasso (pyHSICLasso library) with S3-based file loading.

    Args:
        request: AnalysisRequest with optional target_column and n_top_features
        current_user: Authenticated user info
        project_id: Project ID to fetch S3 metadata
    """
    token = current_user.get("token")

    try:

        # Try S3-based approach first
        train_df = None
        target_column = request.target_column

        if project_id:
            try:
                # Get S3 metadata
                s3_metadata = get_s3_file_metadata(token, project_id)

                if s3_metadata:
                    # Validate and extract URLs
                    train_url, test_url, model_url = validate_file_metadata(s3_metadata)

                    # Load training data from S3
                    print(f"[ANALYSIS] Loading training data from S3: {train_url}")
                    train_df = load_dataframe_from_url(train_url)
                    print(f"[ANALYSIS] Loaded data: shape={train_df.shape}, columns={list(train_df.columns)}")

                    # Validate target column if provided
                    if target_column and target_column not in train_df.columns:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Target column '{target_column}' not found. Available: {list(train_df.columns)}",
                        )
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("S3 loading failed, falling back to global_session: %s", e)

        # Fallback to global_session if S3 loading failed or no project_id
        if train_df is None:
            print("[ANALYSIS] Using fallback: global_session")
            if not global_session.is_ready_for_analysis():
                raise HTTPException(
                    status_code=400,
                    detail="Training data and target column must be configured. Provide project_id or upload data first.",
                )

            train_df = FileManager.load_csv(global_session.train_file_path)
            target_column = target_column or global_session.target_column

        # Ensure we have a target column
        if not target_column:
            raise HTTPException(
                status_code=400,
                detail="target_column is required. Provide it in the request or configure global_session.",
            )

        # Initialize analysis engine
        print(f"[ANALYSIS] Initializing engine with target: {target_column}")
        engine = AnalysisEngine()
        engine.train_df = train_df
        engine.target_column = target_column

        # Run analysis
        print(f"[ANALYSIS] Running analysis with {request.n_top_features} top features")
        results = engine.run_analysis(n_top_features=request.n_top_features)
        results["detector_obj"] = engine.detector

        # Store results in session
        global_session.analysis_results = results

        # Convert to response format
        feature_sensitivity_scores = [
            FeatureSensitivityScore(
                feature=score["feature"],
                nocco_score=score["nocco_score"],
                is_sensitive=score["is_sensitive"],
                percentile_rank=score["percentile_rank"],
            )
            for score in results["feature_sensitivity_scores"]
        ]

        logger.info("Analysis finished; stored detector instance in session")

        return AnalysisResponse(
            status="success",
            analysis_type=results["analysis_summary"]["analysis_type"],
            target_column=target_column,
            total_features=results["analysis_summary"]["total_features"],
            sensitive_features_count=results["analysis_summary"]["sensitive_features_count"],
            threshold_value=results["threshold_value"],
            feature_scores=feature_sensitivity_scores,
            sensitive_features=results["sensitive_features"],
            analysis_summary=results["analysis_summary"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in run_fairness_analysis")
        raise HTTPException(status_code=500, detail=f"Error running analysis: {str(e)}") from e


@router.get("/results")
async def get_analysis_results(
    current_user: str = Depends(get_current_user),
    project_id: str = None,
    target_column: str = None,
    n_top_features: int = 20,  # or any default you want
):
    """Run (if needed) and get analysis results"""
    # Validate current_user structure
    if not isinstance(current_user, dict) or "token" not in current_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials: token missing.",
        )
    token = current_user.get("token")

    try:
        # --- 1. Load data (S3 first, fallback to global_session) ---
        train_df = None

        if project_id:
            try:
                # Get S3 metadata
                s3_metadata = get_s3_file_metadata(token, project_id)

                if s3_metadata:
                    # Validate and extract URLs
                    train_url, test_url, model_url = validate_file_metadata(s3_metadata)

                    # Load training data from S3
                    print(f"[RESULTS] Loading training data from S3: {train_url}")
                    train_df = load_dataframe_from_url(train_url)
                    print(f"[RESULTS] Loaded data: shape={train_df.shape}, columns={list(train_df.columns)}")

                    # Validate target column if provided
                    if target_column and target_column not in train_df.columns:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Target column '{target_column}' not found. Available: {list(train_df.columns)}",
                        )
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("S3 loading failed, falling back to global_session: %s", e)

        # Fallback to global_session if S3 loading failed or no project_id
        if train_df is None:
            print("[RESULTS] Using fallback: global_session")
            if not global_session.is_ready_for_analysis():
                raise HTTPException(
                    status_code=400,
                    detail="Training data and target column must be configured. Provide project_id or upload data first.",
                )

            train_df = FileManager.load_csv(global_session.train_file_path)
            target_column = target_column or global_session.target_column

        # Ensure we have a target column
        if not target_column:
            raise HTTPException(
                status_code=400,
                detail="target_column is required. Provide it as a query param or configure global_session.",
            )

        # --- 2. Run analysis here (moved from /run) ---
        print(f"[RESULTS] Initializing engine with target: {target_column}")
        engine = AnalysisEngine()
        engine.train_df = train_df
        engine.target_column = target_column

        print(f"[RESULTS] Running analysis with {n_top_features} top features")
        results = engine.run_analysis(n_top_features=n_top_features)
        results["detector_obj"] = engine.detector

        # Optionally still store in session if you want
        global_session.analysis_results = results

        # --- 3. Prepare response (remove non-serializable parts) ---
        results_copy = {key: value for key, value in results.items() if key != "detector_obj"}

        return {"status": "success", "results": results_copy}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}") from e


@router.get("/feature-relationships/{feature_index}")
async def get_feature_relationships(
    feature_index: int,
    num_neighbors: int = 5,
    current_user: str = Depends(get_current_user),
    project_id: str = None,
    target_column: str = None,
    n_top_features: int = 20,  # optional; used when running analysis
):
    """
    Get features that are related to a specific feature using the detector.
    Tries S3 + fresh analysis first; falls back to session-stored detector.
    """
    if not isinstance(current_user, dict) or "token" not in current_user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials: token missing.")
    token = current_user.get("token")

    try:
        detector = None
        total_features = None

        # --- 1. Try S3-based approach + fresh analysis ---
        train_df = None
        if project_id:
            try:
                s3_metadata = get_s3_file_metadata(token, project_id)
                if s3_metadata:
                    # Validate and extract URLs
                    train_url, test_url, model_url = validate_file_metadata(s3_metadata)

                    # Load training data from S3
                    print(f"[RELATIONSHIPS] Loading training data from S3: {train_url}")
                    train_df = load_dataframe_from_url(train_url)
                    print(f"[RELATIONSHIPS] Loaded data: shape={train_df.shape}, columns={list(train_df.columns)}")

                    # If no target_column provided here, you need some way to get it
                    # from metadata, model, or global_session as a last resort.
                    if target_column is None and global_session.target_column:
                        target_column = global_session.target_column

                    # Validate target column if provided
                    if target_column and target_column not in train_df.columns:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Target column '{target_column}' not found. Available: {list(train_df.columns)}",
                        )

                    if not target_column:
                        raise HTTPException(
                            status_code=400,
                            detail="target_column is required. Provide it as a query param or configure global_session.",
                        )

                    # Run analysis to build detector
                    print(f"[RELATIONSHIPS] Initializing engine with target: {target_column}")
                    engine = AnalysisEngine()
                    engine.train_df = train_df
                    engine.target_column = target_column

                    print(f"[RELATIONSHIPS] Running analysis with {n_top_features} top features")
                    results = engine.run_analysis(n_top_features=n_top_features)
                    results["detector_obj"] = engine.detector

                    # Optionally store in session for later use
                    global_session.analysis_results = results

                    detector = engine.detector
                    total_features = results.get("analysis_summary", {}).get("total_features", None)

            except HTTPException:
                # re-raise HTTP-specific problems (like bad target_column)
                raise
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("S3 + analysis failed, falling back to global_session: %s", e)
                detector = None
                total_features = None

        # --- 2. Fallback: use detector from global_session.analysis_results ---
        if detector is None:
            if global_session.analysis_results is None:
                raise HTTPException(status_code=404, detail="No analysis results found. Run analysis first.")

            detector = global_session.analysis_results.get("detector_obj", None)
            if detector is None:
                raise HTTPException(
                    status_code=500,
                    detail="Detector instance not found in session. Re-run analysis to rebuild detector.",
                )

            total_features = global_session.analysis_results.get("analysis_summary", {}).get("total_features", None)

        # --- 3. Validate feature_index using total_features (if available) ---
        if total_features is not None and (feature_index < 0 or feature_index >= total_features):
            raise HTTPException(
                status_code=400,
                detail=f"feature_index {feature_index} out of range (0..{total_features - 1})",
            )

        # --- 4. Get relationships from detector ---
        relationships = detector.get_feature_relationships(feature_index, num_neighbors)

        return {"status": "success", "relationships": relationships}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error retrieving feature relationships")
        raise HTTPException(status_code=500, detail=f"Error retrieving relationships: {str(e)}") from e
