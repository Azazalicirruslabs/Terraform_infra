from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from services.mainflow.app.database.connections import get_db
from shared.auth import get_current_user
from shared_migrations.models.discover import Discover

router = APIRouter(
    prefix="/mainflow",
    tags=["Manual Registration Fetch"],
)

ALLOWED_ASSET_TYPES = {"ML Model", "LLM API", "Agentic AI"}


def map_discover_to_response(asset: Discover) -> dict:
    """
    Maps Discover DB model to API response format
    """
    return {
        "asset_id": asset.id,  # ✅ include registration ID
        "asset_type": asset.asset_type,
        "asset_name": asset.project_name,
        "version": asset.version,
        "framework": asset.model_type,
        "lifecycle_state": asset.lifecycle_state,
        "artifact_uri": asset.uri,
        "owner_team": asset.owner,
        "description": asset.description,
        "tags": asset.tags if asset.tags else [],
        "created_at": asset.created_at,
    }


@router.get("/get_discover")
def fetch_registered_assets(
    asset_type: str | None = Query(None, description="ML Model | LLM API | Agentic AI (optional)"),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if asset_type and asset_type not in ALLOWED_ASSET_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid asset_type. Allowed values are: {', '.join(ALLOWED_ASSET_TYPES)}",
        )

    query = db.query(Discover)

    if asset_type:
        query = query.filter(Discover.asset_type == asset_type)

    assets = query.all()

    if asset_type and not assets:
        raise HTTPException(
            status_code=404,
            detail=f"No assets found for asset_type: {asset_type}. "
            f"Allowed values: {', '.join(ALLOWED_ASSET_TYPES)}",
        )

    response_data = list(map(map_discover_to_response, assets))

    return {
        "message": "Registered assets fetched successfully",
        "status": 200,
        "data": response_data,
    }
