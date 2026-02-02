from pydantic import BaseModel


class ResponseModel(BaseModel):
    feature_info: dict[str, dict]
    session_id: str
