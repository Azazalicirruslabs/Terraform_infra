from fastapi import APIRouter, Depends, HTTPException

from services.what_if.app.utils.session_manager import SessionManager
from shared.auth import get_current_user

session_manager = SessionManager()

router = APIRouter(prefix="/what_if", tags=["What If Analysis"])


@router.delete("/session/{session_id}")
async def cleanup_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Cleanup a specific session and all its data"""
    try:
        result = session_manager.cleanup_session(session_id)
        return result
    except Exception as e:
        raise HTTPException(500, f"Error cleaning up session: {str(e)}")
