from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.API.app.database.connections import get_db
from services.API.app.models import Role  # Assuming Role is a SQLAlchemy model for roles
from shared.auth import get_current_user

router = APIRouter(prefix="/api", tags=["API"])


class RquestData(BaseModel):
    """Data model for adding a new role."""

    name: str
    tenant_id: str
    description: str = None


@router.post("/add_role", status_code=status.HTTP_201_CREATED)
async def add_role(
    payload: Optional[RquestData], db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Add a new role to the system.
    Only accessible by authenticated users.
    """

    if payload is None:
        raise HTTPException(status_code=400, detail="Request body cannot be empty")

    # Check if the role already exists for this tenant
    existing_role = db.query(Role).filter_by(name=payload.name, tenant_id=payload.tenant_id).first()

    if existing_role:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Role already exists for this tenant")

    # Create a new role instance
    new_role = Role(name=payload.name, tenant_id=payload.tenant_id, description=payload.description)
    # Add the role to the database
    db.add(new_role)
    db.commit()

    return {"message": "Role added successfully", "role": payload.name}
