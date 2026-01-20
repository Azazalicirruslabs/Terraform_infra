from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from services.API.app.database.connections import get_db
from services.API.app.models import TenantSubscription
from services.API.app.schemas.subscription import CreateSubscription, ReadSubscription

router = APIRouter(prefix="/api", tags=["API"])


@router.post("/subscription", response_model=ReadSubscription)
def create_subscription(Subscription_data: CreateSubscription, db: Session = Depends(get_db)):

    db_subscription = TenantSubscription(**Subscription_data.dict())
    db.add(db_subscription)
    db.commit()
    db.refresh(db_subscription)

    return db_subscription
