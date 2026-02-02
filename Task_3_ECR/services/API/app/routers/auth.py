from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from services.API.app.database.connections import get_db
from services.API.app.models import User
from services.API.app.utils.security import verify_password
from shared.auth import create_access_token

router = APIRouter(prefix="/api", tags=["Auth"])


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not registered")

    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    # get roles from user.roles
    role_name = []
    for role in user.roles:
        if not role.role:
            raise HTTPException(status_code=400, detail="User has no roles assigned")
        else:
            role_name.append(role.role.name)

    # get permissions from user.roles
    # Step 2: Get permissions from all roles
    permission_names = set()
    for ur in user.roles:
        for rp in ur.role.permissions:
            permission_names.add(rp.permission.name)

    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id, "roles": role_name, "permissions": list(permission_names)}
    )
    user_profile = {
        "name": user.name,
        "username": user.username,
        "email": user.email,
        "tenant_id": user.tenant_id,
        "user_id": user.id,
        "roles": role_name,
        "permissions": list(permission_names),
    }
    return {"access_token": access_token, "token_type": "Bearer", "user_profile": user_profile}
