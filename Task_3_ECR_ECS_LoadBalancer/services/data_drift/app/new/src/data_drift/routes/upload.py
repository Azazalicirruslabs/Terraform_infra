import io
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from shared.auth import get_current_user

router = APIRouter(prefix="/data_drift", tags=["data-drift"])

# Simple session storage for uploaded files (shared across all data drift routes)
session_storage = {}


@router.post("/upload")
async def upload_datasets(
    reference_file: UploadFile = File(...),
    current_file: UploadFile = File(...),
    session_id: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    """
    Upload reference and current datasets for data drift analysis

    Args:
        reference_file: Reference dataset CSV file
        current_file: Current dataset CSV file
        session_id: Optional session ID, if not provided a new one will be generated

    Returns:
        Session ID for accessing analysis results
    """
    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Validate file types
        if not reference_file.filename.endswith(".csv") or not current_file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Read CSV files
        reference_content = await reference_file.read()
        current_content = await current_file.read()

        # Convert to pandas DataFrames
        try:
            reference_df = pd.read_csv(io.StringIO(reference_content.decode("utf-8")))
            current_df = pd.read_csv(io.StringIO(current_content.decode("utf-8")))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV files: {str(e)}")

        # Basic validation
        if reference_df.empty or current_df.empty:
            raise HTTPException(status_code=400, detail="Uploaded files cannot be empty")

        # Store in session
        session_storage[session_id] = {
            "reference": reference_df,
            "current": current_df,
            "upload_timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_filename": reference_file.filename,
            "current_filename": current_file.filename,
            "reference_shape": reference_df.shape,
            "current_shape": current_df.shape,
        }

        return {
            "status": "success",
            "session_id": session_id,
            "message": "Datasets uploaded successfully",
            "data": {
                "reference_shape": reference_df.shape,
                "current_shape": current_df.shape,
                "reference_columns": list(reference_df.columns),
                "current_columns": list(current_df.columns),
                "common_columns": list(set(reference_df.columns) & set(current_df.columns)),
                "upload_timestamp": session_storage[session_id]["upload_timestamp"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/session/{session_id}/info")
async def get_session_info(session_id: str, user: dict = Depends(get_current_user)):
    """
    Get information about an existing session

    Args:
        session_id: Session identifier

    Returns:
        Session information
    """
    try:
        if session_id not in session_storage:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = session_storage[session_id]
        reference_df = session_data["reference"]
        current_df = session_data["current"]

        return {
            "status": "success",
            "session_id": session_id,
            "data": {
                "reference_filename": session_data["reference_filename"],
                "current_filename": session_data["current_filename"],
                "reference_shape": session_data["reference_shape"],
                "current_shape": session_data["current_shape"],
                "reference_columns": list(reference_df.columns),
                "current_columns": list(current_df.columns),
                "common_columns": list(set(reference_df.columns) & set(current_df.columns)),
                "upload_timestamp": session_data["upload_timestamp"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    """
    Delete a session and its associated data

    Args:
        session_id: Session identifier

    Returns:
        Confirmation of deletion
    """
    try:
        if session_id not in session_storage:
            raise HTTPException(status_code=404, detail="Session not found")

        del session_storage[session_id]

        return {"status": "success", "message": f"Session {session_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.get("/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    """
    List all active sessions

    Returns:
        List of active sessions
    """
    try:
        sessions = []
        for session_id, data in session_storage.items():
            sessions.append(
                {
                    "session_id": session_id,
                    "upload_timestamp": data["upload_timestamp"],
                    "reference_filename": data["reference_filename"],
                    "current_filename": data["current_filename"],
                    "reference_shape": data["reference_shape"],
                    "current_shape": data["current_shape"],
                }
            )

        return {"status": "success", "total_sessions": len(sessions), "sessions": sessions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


# Helper function to get session data (used by other route files)
def get_session_storage(user: dict = Depends(get_current_user)):
    """Get reference to session storage for other modules"""
    return session_storage
