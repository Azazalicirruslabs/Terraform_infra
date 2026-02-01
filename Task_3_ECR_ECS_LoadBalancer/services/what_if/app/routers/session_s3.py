from fastapi import APIRouter, Depends, HTTPException

from services.what_if.app.schema.session_s3 import SessionRequest, SessionResponse
from services.what_if.app.utils.session_manager import SessionManager
from shared.auth import get_current_user

# Initialize session manager
session_manager = SessionManager()

router = APIRouter(prefix="/what_if", tags=["What If Analysis"])


@router.post("/session/s3", response_model=SessionResponse)
async def create_session_from_s3(request: SessionRequest, current_user: dict = Depends(get_current_user)):
    """Create a new analysis session with data from S3"""
    access_token = current_user.get("token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Unauthorized - Access token is required")
    target_column = request.target_column
    if not target_column:
        raise HTTPException(status_code=400, detail="Target column is required")
    analysis_type = request.analysis_type
    if not analysis_type:
        raise HTTPException(status_code=400, detail="Analysis type is required")
    # Validate analysis type
    if analysis_type not in ["Classification", "Regression"]:
        raise HTTPException(status_code=400, detail="Invalid analysis type. Must be 'Classification' or 'Regression'.")
    try:
        # Use the hardcoded access token from session manager
        session_id, metadata = session_manager.create_session_from_s3(
            access_token, analysis_type, target_column  # Pass empty string since token is hardcoded
        )

        return {
            "session_id": session_id,
            "metadata": metadata,
            "message": "Datasets and model fetched successfully from S3!",
        }

    except Exception as e:
        raise HTTPException(500, f"Error creating S3 session: {str(e)}")
