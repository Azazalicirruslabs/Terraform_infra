import io
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from shared.auth import get_current_user

from .session_manager import get_session_manager

router = APIRouter(prefix="/data_drift/v1", tags=["Unified Upload"])


@router.post("/upload")
async def unified_upload(
    reference_data: UploadFile = File(..., description="Reference dataset CSV file"),
    current_data: UploadFile = File(..., description="Current dataset CSV file"),
    model_file: Optional[UploadFile] = File(None, description="Optional: Trained model pickle file"),
    config: Optional[str] = Form(None, description="Optional JSON configuration for Model Drift analysis"),
    target_column: Optional[str] = Form(
        None, description="Optional: Name of target column for class imbalance analysis"
    ),
    reference_predictions: Optional[str] = Form(
        None, description="Optional: JSON array of model predictions for reference data"
    ),
    current_predictions: Optional[str] = Form(
        None, description="Optional: JSON array of model predictions for current data"
    ),
    session_id: Optional[str] = None,
    user: dict = Depends(get_current_user),  # Assuming user authentication is required
):
    """
    Unified upload endpoint for both Data Drift and Model Drift analysis

    Args:
        reference_data: CSV file containing the reference/baseline dataset
        current_data: CSV file containing the current dataset to compare against reference
        model_file: Optional pickle or joblib file containing trained model (enables model drift analysis)
        session_id: Optional session ID, if not provided a new one will be generated

    Returns:
        Session information that can be used for both data drift and model drift analysis
    """
    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Validate file types
        if not reference_data.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Reference data must be a CSV file")
        if not current_data.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Current data must be a CSV file")
        if model_file and not (model_file.filename.lower().endswith((".pkl", ".pickle", ".joblib", ".pkl.joblib"))):
            raise HTTPException(
                status_code=400, detail="Model file must be a pickle (.pkl, .pickle) or joblib (.joblib) file"
            )

        # Validate and parse configuration if provided
        parsed_config = None
        if config:
            try:
                parsed_config = json.loads(config)
                # Basic configuration validation for Model Drift
                if model_file:  # Only validate if model is provided
                    required_fields = ["model_type", "selected_metrics"]
                    for field in required_fields:
                        if field not in parsed_config:
                            raise HTTPException(
                                status_code=400, detail=f"Missing required configuration field: {field}"
                            )

                    # Validate model_type
                    if parsed_config["model_type"].lower() not in ["classification", "regression"]:
                        raise HTTPException(
                            status_code=400, detail="model_type must be 'classification' or 'regression'"
                        )

                    # Validate selected_metrics is a list
                    if not isinstance(parsed_config["selected_metrics"], list) or not parsed_config["selected_metrics"]:
                        raise HTTPException(status_code=400, detail="selected_metrics must be a non-empty list")

            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON configuration: {str(e)}")

        # Read CSV files
        reference_content = await reference_data.read()
        current_content = await current_data.read()

        # Convert to pandas DataFrames
        try:
            reference_df = pd.read_csv(io.StringIO(reference_content.decode("utf-8")))
            current_df = pd.read_csv(io.StringIO(current_content.decode("utf-8")))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV files: {str(e)}")

        # Basic validation
        if reference_df.empty or current_df.empty:
            raise HTTPException(status_code=400, detail="Uploaded files cannot be empty")

        # Handle model file if provided
        model_file_content = None
        model_filename = None
        if model_file:
            model_file_content = await model_file.read()
            model_filename = model_file.filename

        # Parse optional predictions if provided
        parsed_reference_predictions = None
        parsed_current_predictions = None

        if reference_predictions:
            try:
                parsed_reference_predictions = json.loads(reference_predictions)
                if len(parsed_reference_predictions) != len(reference_df):
                    raise ValueError(
                        f"Reference predictions length ({len(parsed_reference_predictions)}) must match reference dataset length ({len(reference_df)})"
                    )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid reference predictions: {str(e)}")

        if current_predictions:
            try:
                parsed_current_predictions = json.loads(current_predictions)
                if len(parsed_current_predictions) != len(current_df):
                    raise ValueError(
                        f"Current predictions length ({len(parsed_current_predictions)}) must match current dataset length ({len(current_df)})"
                    )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid current predictions: {str(e)}")

        # Validate target column if provided
        if target_column:
            if target_column not in reference_df.columns:
                raise HTTPException(
                    status_code=400, detail=f"Target column '{target_column}' not found in reference dataset"
                )
            if target_column not in current_df.columns:
                raise HTTPException(
                    status_code=400, detail=f"Target column '{target_column}' not found in current dataset"
                )

        # Store in unified session manager
        session_manager = get_session_manager()

        # Store configuration in session if provided
        session_config = parsed_config if parsed_config else None

        created_session_id = session_manager.create_session(
            reference_df=reference_df,
            current_df=current_df,
            reference_filename=reference_data.filename,
            current_filename=current_data.filename,
            model_file_content=model_file_content,
            model_filename=model_filename,
            session_id=session_id,
        )

        # Store configuration in session after creation
        if session_config and model_file:
            session_manager._storage[created_session_id]["config"] = session_config

        # Store class imbalance analysis configuration if provided
        class_imbalance_config = {}
        if target_column:
            class_imbalance_config["target_column"] = target_column
        if parsed_reference_predictions:
            class_imbalance_config["reference_predictions"] = parsed_reference_predictions
        if parsed_current_predictions:
            class_imbalance_config["current_predictions"] = parsed_current_predictions

        if class_imbalance_config:
            session_manager._storage[created_session_id].update(class_imbalance_config)

        # Prepare response
        response_data = {
            "session_id": created_session_id,
            "message": "Files uploaded successfully",
            "data": {
                "reference_shape": reference_df.shape,
                "current_shape": current_df.shape,
                "reference_columns": list(reference_df.columns),
                "current_columns": list(current_df.columns),
                "common_columns": list(set(reference_df.columns) & set(current_df.columns)),
                "has_model": model_file is not None,
                "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                # Class imbalance configuration status
                "class_imbalance_ready": bool(target_column),
                "target_column": target_column,
                "predictions_provided": {
                    "reference": parsed_reference_predictions is not None,
                    "current": parsed_current_predictions is not None,
                },
            },
            "analysis_endpoints": {
                "data_drift": {
                    "dashboard": f"/data-drift/dashboard/{created_session_id}",
                    "class_imbalance": f"/data-drift/class-imbalance/analysis/{created_session_id}",
                    "statistical_reports": f"/data-drift/statistical-reports/{created_session_id}",
                    "feature_analysis": f"/data-drift/feature-analysis/{created_session_id}",
                }
            },
        }

        # Add model drift endpoints if model was provided
        if model_file:
            response_data["analysis_endpoints"]["model_drift"] = {
                "performance_comparison": f"/model-drift/session/performance-comparison/{created_session_id}",
                "degradation_metrics": f"/model-drift/session/degradation-metrics/{created_session_id}",
                "statistical_significance": f"/model-drift/session/statistical-significance/{created_session_id}",
            }

        return {"status": "success", **response_data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/session/{session_id}/info")
async def get_unified_session_info(session_id: str, user: dict = Depends(get_current_user)):
    """
    Get information about a unified session

    Args:
        session_id: Session identifier

    Returns:
        Session information including available analysis types
    """
    try:
        session_manager = get_session_manager()
        if not session_manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = session_manager.get_session(session_id)

        response_data = {
            "session_id": session_id,
            "data": {
                "reference_filename": session_data["reference_filename"],
                "current_filename": session_data["current_filename"],
                "reference_shape": session_data["reference_shape"],
                "current_shape": session_data["current_shape"],
                "common_columns": session_data["common_columns"],
                "common_columns_count": len(session_data["common_columns"]),
                "has_model": session_data["has_model"],
                "upload_timestamp": session_data["upload_timestamp"],
            },
            "analysis_endpoints": {
                "data_drift": {
                    "dashboard": f"/data-drift/dashboard/{session_id}",
                    "class_imbalance": f"/data-drift/class-imbalance/analysis/{session_id}",
                    "statistical_reports": f"/data-drift/statistical-reports/{session_id}",
                    "feature_analysis": f"/data-drift/feature-analysis/{session_id}",
                }
            },
        }

        # Add model drift endpoints if model is available
        if session_data["has_model"]:
            response_data["analysis_endpoints"]["model_drift"] = {
                "performance_comparison": f"/model-drift/session/performance-comparison/{session_id}",
                "degradation_metrics": f"/model-drift/session/degradation-metrics/{session_id}",
                "statistical_significance": f"/model-drift/session/statistical-significance/{session_id}",
            }
            response_data["data"]["model_filename"] = session_data.get("model_filename", "")

        return {"status": "success", **response_data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")


@router.delete("/session/{session_id}")
async def delete_unified_session(session_id: str, user: dict = Depends(get_current_user)):
    """
    Delete a unified session and its associated data

    Args:
        session_id: Session identifier

    Returns:
        Confirmation of deletion
    """
    try:
        session_manager = get_session_manager()
        if session_manager.delete_session(session_id):
            return {"status": "success", "message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.get("/sessions")
async def list_unified_sessions(user: dict = Depends(get_current_user)):
    """
    List all active unified sessions

    Returns:
        List of active sessions with their information
    """
    try:
        session_manager = get_session_manager()
        sessions = session_manager.list_sessions()

        return {"status": "success", "total_sessions": len(sessions), "sessions": sessions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for the unified upload service"""
    return {
        "status": "healthy",
        "service": "unified-upload",
        "version": "2.0.0",
        "endpoints": {
            "upload": "/api/v1/upload",
            "session_info": "/api/v1/session/{session_id}/info",
            "delete_session": "/api/v1/session/{session_id}",
            "list_sessions": "/api/v1/sessions",
        },
    }
