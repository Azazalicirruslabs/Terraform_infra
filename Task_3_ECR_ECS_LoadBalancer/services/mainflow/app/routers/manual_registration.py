import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from services.mainflow.app.database.connections import get_db
from services.mainflow.app.schemas.manual_registration_schema import (
    ManualRegistrationRequest,
    ManualRegistrationResponse,
)
from shared.auth import get_current_user
from shared_migrations.models.discover import Discover
from shared_migrations.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/mainflow",
    tags=["Manual Registration"],
)


@router.post(
    "/manual-registration",
    response_model=ManualRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
)
def manual_registration(
    payload: ManualRegistrationRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        user_id = user.get("user_id")

        mandatory_fields = {
            "asset_type": "Asset type",
            "asset_name": "Asset name",
            "version": "Version",
            "framework": "Framework / Model type",
            "lifecycle_state": "Lifecycle state",
            "artifact_uri": "Artifact URI / Endpoint",
            "owner_team": "Owner / Responsible team",
        }

        for field, display_name in mandatory_fields.items():
            value = getattr(payload, field)
            if not value or (isinstance(value, str) and not value.strip()):
                logger.error(f"{display_name} is required | user_id={user_id}")
                raise HTTPException(status_code=400, detail=f"{display_name} must be provided.")

        if not payload.asset_type.strip():
            logger.error(f"Asset type not selected | user_id={user_id}")
            raise HTTPException(status_code=400, detail="Asset type must be selected.")

        user_obj = db.query(User).filter_by(id=user_id).first()
        if not user_obj:
            logger.error(f"User not found | user_id={user_id}")
            raise HTTPException(status_code=400, detail="User not found.")

        user_obj = db.query(User).filter_by(id=user_id).first()
        if not user_obj:
            logger.error(f"User not found | user_id={user_id}")
            raise HTTPException(status_code=400, detail="User not found.")

        user_obj = db.query(User).filter_by(id=user_id).first()
        if not user_obj:
            logger.error(f"User not found | user_id={user_id}")
            raise HTTPException(status_code=400, detail="User not found.")

        tenant_id = user_obj.tenant_id

        existing = (
            db.query(Discover)
            .filter_by(
                project_name=payload.asset_name,
                asset_type=payload.asset_type,
                user_id=user_id,
            )
            .first()
        )

        if existing:
            logger.error(
                f"Duplicate asset registration | "
                f"user_id={user_id}, asset_type={payload.asset_type}, asset_name={payload.asset_name}"
            )
            raise HTTPException(
                status_code=409,
                detail=f"An asset with the name '{payload.asset_name}' already exists under the asset type '{payload.asset_type}'.",
            )

        logger.info(
            f"Creating Discover entry | "
            f"asset_type={payload.asset_type}, "
            f"asset_name={payload.asset_name}, "
            f"version={payload.version}, "
            f"user_id={user_id}"
        )

        new_entry = Discover(
            asset_type=payload.asset_type,
            project_name=payload.asset_name,
            version=payload.version,
            model_type=payload.framework,
            lifecycle_state=payload.lifecycle_state,
            uri=payload.artifact_uri,
            owner=payload.owner_team,
            description=payload.description,
            tags=payload.tags,
            tenant_id=tenant_id,
            user_id=user_id,
            is_active=True,
        )

        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)

        logger.info(f"Manual registration successful | " f"registration_id={new_entry.id}, user_id={user_id}")

        return ManualRegistrationResponse(
            message="Asset registered successfully",
            status=status.HTTP_201_CREATED,
            data=payload,
            registration_id=new_entry.id,
            created_at=new_entry.created_at,
        )

    except HTTPException as e:
        logger.error(f"Manual registration failed | " f"status_code={e.status_code}, detail={e.detail}")
        raise

    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error during manual registration | " f"user_id={user_id}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to register manually",
        )
