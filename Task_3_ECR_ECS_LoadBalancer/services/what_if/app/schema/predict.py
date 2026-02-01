from typing import Any, Dict, List

from pydantic import BaseModel


class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    status: str
    session_id: str
    predictions: List[float]
