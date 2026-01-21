from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from services.API.app.database.connections import get_db

# from app.models import Plan # Assuming Plan is a SQLAlchemy model
from services.API.app.models import Plan

# from app.schemas.plan import CreatePlane, ReadPlane
from services.API.app.schemas.plan import CreatePlane, ReadPlane
from services.API.app.utils.security import get_current_user

router = APIRouter(prefix="/api", tags=["API"])


@router.post("/plan", response_model=ReadPlane)
def create_plan(plan: CreatePlane, db: Session = Depends(get_db), current_user=Depends(get_current_user)):

    db_plan = Plan(**plan.dict())
    db.add(db_plan)
    db.commit()
    db.refresh(db_plan)

    return db_plan
