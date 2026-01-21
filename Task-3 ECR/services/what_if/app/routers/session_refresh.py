from fastapi import APIRouter, Depends, HTTPException

from services.what_if.app.utils.session_manager import SessionManager
from shared.auth import get_current_user

session_manager = SessionManager()

router = APIRouter(prefix="/what_if", tags=["What If Analysis"])


@router.get("/session/{session_id}/refresh")
async def refresh_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Refresh a session to extend its lifetime"""
    try:
        result = session_manager.refresh_session(session_id)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error refreshing session: {str(e)}")
