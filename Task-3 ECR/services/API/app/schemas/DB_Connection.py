from typing import Any, Dict, Optional

from pydantic import BaseModel


class Response_Schema(BaseModel):
    status: str
    message: str
    folder_status: Optional[str] = None  # Optional field for folder status
    data: Optional[Dict[str, Any]] = None  # Use `Any` for flexible data type


class TransferToS3Request(BaseModel):
    status: str
    message: str
    additional_info: Optional[str] = None  # Optional field for additional information
