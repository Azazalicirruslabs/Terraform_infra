from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from services.API.app.database.connections import get_db
from services.API.app.models import Tenant
from services.API.app.schemas.tenant import CreateTenant

router = APIRouter(prefix="/api", tags=["API"])


@router.post("/tenant")
def create_tenant(tenant: CreateTenant, db: Session = Depends(get_db)):

    name = tenant.name
    tenant_exists = db.query(Tenant).filter(Tenant.name == name).first()
    if tenant_exists:
        return {"status": status.HTTP_409_CONFLICT, "error": "Tenant with this name already exists"}

    email = tenant.email
    email_exists = db.query(Tenant).filter(Tenant.email == email).first()
    if email_exists:
        return {"status": status.HTTP_409_CONFLICT, "error": "Tenant with this email already exists"}

    mobile_no = tenant.mobile_no
    mobile_no_exists = db.query(Tenant).filter(Tenant.mobile_no == mobile_no).first()
    if mobile_no_exists:
        return {"status": status.HTTP_409_CONFLICT, "error": "Tenant with this mobile number already exists"}

    domain = tenant.domain
    domain_exists = db.query(Tenant).filter(Tenant.domain == domain).first()
    if domain_exists:
        return {"status": status.HTTP_409_CONFLICT, "error": "Tenant with this domain already exists"}

    db_tenant = Tenant(**tenant.dict())
    db.add(db_tenant)
    db.commit()
    db.refresh(db_tenant)
    return db_tenant
