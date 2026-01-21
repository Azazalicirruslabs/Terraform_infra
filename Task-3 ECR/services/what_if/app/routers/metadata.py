from fastapi import APIRouter, Depends, HTTPException

from services.what_if.app.schema.metadata import MetadataResponse
from services.what_if.app.utils.session_manager import SessionManager
from shared.auth import get_current_user

session_manager = SessionManager()

router = APIRouter(prefix="/what_if", tags=["What If Analysis"])


@router.get("/metadata/{session_id}", response_model=MetadataResponse)
async def get_metadata(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get session metadata and data profile"""
    try:
        _, _, _, metadata = session_manager.get_session_artifacts(session_id)
        return metadata
    except Exception as e:
        raise HTTPException(500, f"Metadata error: {str(e)}")
