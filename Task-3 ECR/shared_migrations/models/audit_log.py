from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from shared_migrations.models.base import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    action = Column(String)
    ip_address = Column(String)
    created_at = Column(DateTime(timezone=True), default=func.now())

    user = relationship("User", back_populates="logs")
    tenant = relationship("Tenant", back_populates="logs")
