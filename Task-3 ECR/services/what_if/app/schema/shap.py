from typing import Any, Dict, List

from pydantic import BaseModel


class ShapeRequest(BaseModel):
    data: List[Dict[str, Any]]


class ShapResponse(BaseModel):
    values: List[List[float]]
    base_values: List[float]
    data: List[List[Any]]
    feature_names: List[str]
