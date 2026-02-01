from typing import List, Optional

from pydantic import BaseModel


class RegisteredAsset(BaseModel):
    asset_type: str
    asset_name: str
    version: str
    framework: str
    lifecycle_state: str
    artifact_uri: str
    owner_team: str
    description: Optional[str] = None
    tags: Optional[List[str]] = []


class RegistrationFetchResponse(BaseModel):
    count: int
    data: List[RegisteredAsset]
