from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, func
from sqlalchemy.orm import relationship

from shared_migrations.models.base import Base


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    domain = Column(String, unique=True, nullable=True)
    revenue = Column(Float, nullable=True)
    no_of_employee = Column(Integer, nullable=True)
    status = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    users = relationship("User", back_populates="tenant")
    # roles = relationship("Role", back_populates="tenant")
    subscriptions = relationship("TenantSubscription", back_populates="tenant")
    logs = relationship("AuditLog", back_populates="tenant")
    files = relationship("FileStorage", back_populates="tenant", cascade="all, delete-orphan")
