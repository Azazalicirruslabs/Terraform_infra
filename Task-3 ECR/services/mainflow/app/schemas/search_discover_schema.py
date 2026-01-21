from typing import List, Optional

from pydantic import BaseModel


class AssetInfo(BaseModel):
    asset_type: Optional[str]
    asset_name: Optional[str]
    version: Optional[str]
    framework: Optional[str]
    lifecycle_state: Optional[str]
    artifact_uri: Optional[str]
    owner_team: Optional[str]
    description: Optional[str]
    tags: Optional[List[str]]


class SearchDiscoverResponse(BaseModel):
    count: int
    data: List[AssetInfo]
