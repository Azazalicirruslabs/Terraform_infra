from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from services.what_if.app.utils.session_manager import SessionManager
from shared.auth import get_current_user

session_manager = SessionManager()

router = APIRouter(prefix="/what_if", tags=["What If Analysis"])


@router.get("/profile/{session_id}")
async def get_profile(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get data profiling report"""
    profile_path = session_manager.sessions_dir / session_id / "profile.html"
    if not profile_path.exists():
        raise HTTPException(404, "Profile not found")
    return FileResponse(profile_path, media_type="text/html")
