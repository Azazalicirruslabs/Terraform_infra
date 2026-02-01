from sqlalchemy import Boolean, Column, DateTime, Integer, String, func
from sqlalchemy.orm import relationship

from shared_migrations.models.base import Base


class Plan(Base):
    __tablename__ = "plans"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    price = Column(Integer)
    user_limit = Column(Integer)
    role_limit = Column(Integer)
    description = Column(String)
    status = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    subscriptions = relationship("TenantSubscription", back_populates="plan")
