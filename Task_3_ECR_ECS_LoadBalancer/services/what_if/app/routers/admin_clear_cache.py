from fastapi import APIRouter, Depends, HTTPException

from services.what_if.app.utils.session_manager import SessionManager
from shared.auth import get_current_user

session_manager = SessionManager()
router = APIRouter(prefix="/what_if", tags=["What If Analysis"])


@router.get("/admin/clear-cache/{session_id}")
async def clear_session_cache(session_id: str, current_user: dict = Depends(get_current_user)):
    """Clear cache for a specific session (for debugging)"""
    try:
        session_manager.clear_cache(session_id)
        return {"message": f"Cache cleared for session {session_id}"}
    except Exception as e:
        raise HTTPException(500, f"Error clearing cache: {str(e)}")
