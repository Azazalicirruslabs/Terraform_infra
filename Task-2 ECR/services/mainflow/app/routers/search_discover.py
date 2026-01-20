from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session

from services.mainflow.app.database.connections import get_db
from shared.auth import get_current_user
from shared_migrations.models.discover import Discover


class AssetInfo(BaseModel):
    registration_id: int
    asset_type: Optional[str]
    asset_name: Optional[str]
    version: Optional[str]
    framework: Optional[str]
    lifecycle_state: Optional[str]
    artifact_uri: Optional[str]
    owner_team: Optional[str]
    description: Optional[str]
    tags: Optional[List[str]]
    created_at: Optional[str]


class SearchDiscoverResponse(BaseModel):
    count: int
    data: List[AssetInfo]


router = APIRouter(prefix="/mainflow", tags=["Discover Search"])


@router.get("/search-discover", response_model=SearchDiscoverResponse)
def search_discover_projects(
    search: str = Query(..., min_length=3, description="Search by project name or asset type"),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    results = (
        db.query(Discover)
        .filter(or_(Discover.project_name.ilike(f"{search}%"), Discover.asset_type.ilike(f"{search}%")))
        .all()
    )

    if not results:
        raise HTTPException(status_code=404, detail=f"Asset name or asset type '{search}' not found.")

    assets = []
    for asset in results:
        assets.append(
            AssetInfo(
                registration_id=asset.id,
                asset_type=asset.asset_type,
                asset_name=asset.project_name,
                version=getattr(asset, "version", None),
                framework=getattr(asset, "model_type", None),
                lifecycle_state=getattr(asset, "lifecycle_state", None),
                artifact_uri=getattr(asset, "uri", None),
                owner_team=getattr(asset, "owner", None),
                description=getattr(asset, "description", None),
                tags=getattr(asset, "tags", None),
                created_at=asset.created_at.isoformat() if isinstance(asset.created_at, datetime) else None,
            )
        )

    return SearchDiscoverResponse(count=len(assets), data=assets)
