from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.API.app.database.connections import get_db
from services.API.app.schemas.assign_role import AssignRolesRequest
from shared_migrations.models import Role, User, UserRole
from shared_migrations.models.role import Role

router = APIRouter(prefix="/api", tags=["API"])


@router.post("/assign-roles")
def assign_roles(payload: AssignRolesRequest, db: Session = Depends(get_db)):
    # Fetch user
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Optional: Remove old roles
    # db.query(UserRole).filter(UserRole.user_id == payload.user_id).delete()

    # Add new roles
    for role_id in payload.role_ids:
        role = db.query(Role).filter(Role.id == role_id).first()
        if not role:
            raise HTTPException(status_code=400, detail=f"Role ID {role_id} does not exist")
        db.add(UserRole(user_id=payload.user_id, role_id=role_id))

    db.commit()
    return {"message": f"Roles assigned successfully to user {user.email}"}
