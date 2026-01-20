import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from services.API.app.database.connections import get_db
from services.API.app.models import User
from services.API.app.schemas.users import UserCreate
from services.API.app.utils.security import hash_password

router = APIRouter(prefix="/api", tags=["API"])

logger = logging.getLogger(__name__)


@router.post("/users")
# @router.post("/", response_model=UserRead)
def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create a new user in the system."""
    logger.info("Connected Database: %s", db.bind.url.database)
    # Print database name
    print(f"Connected Database: {db.bind.url.database}")
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists")

    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")

    hashed_pwd = hash_password(user_data.password)
    db_user = User(
        name=user_data.name,
        tenant_id=user_data.tenant_id,
        username=user_data.username,
        email=user_data.email,
        password=hashed_pwd,
    )
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        return {"status": status.HTTP_409_CONFLICT, "message": "User already exist"}
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"status": status.HTTP_201_CREATED, "message": "User created successfully"}
