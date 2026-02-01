from datetime import datetime

from pydantic import BaseModel


class CreateSubscription(BaseModel):
    tenant_id: int
    plan_id: int
    start_date: datetime
    end_date: datetime
    payment_status: str
    status: str


class ReadSubscription(BaseModel):
    id: int
    tenant_id: int
    plan_id: int
    start_date: datetime
    end_date: datetime
    payment_status: str
    status: str

    class Config:
        from_attributes = True
