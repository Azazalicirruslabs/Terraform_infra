from typing import Any, Dict

from pydantic import BaseModel


class SessionRequest(BaseModel):
    analysis_type: str
    target_column: str


class SessionResponse(BaseModel):
    session_id: str
    metadata: Dict[str, Any]
    message: str
