from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from shared_migrations.models.base import Base


class User(Base):

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    username = Column(String, unique=True, nullable=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    roles = relationship("UserRole", back_populates="user")
    tenant = relationship("Tenant", back_populates="users")
    logs = relationship("AuditLog", back_populates="user")

    files = relationship("FileStorage", back_populates="user", cascade="all, delete-orphan")
