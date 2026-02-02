from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from shared_migrations.models.base import Base


class TenantSubscription(Base):
    __tablename__ = "tenant_subscriptions"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    plan_id = Column(Integer, ForeignKey("plans.id"))
    start_date = Column(String)
    end_date = Column(String)
    payment_status = Column(String)
    status = Column(String)  # active, cancelled, expired
    created_at = Column(DateTime(timezone=True), default=func.now())

    tenant = relationship("Tenant", back_populates="subscriptions")
    plan = relationship("Plan", back_populates="subscriptions")
