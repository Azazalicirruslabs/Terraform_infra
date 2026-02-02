from sqlalchemy import Column, DateTime, Integer, String, func
from sqlalchemy.orm import relationship

from shared_migrations.models.base import Base


class Permission(Base):

    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    created_at = Column(DateTime(timezone=True), default=func.now())

    roles = relationship("RolePermission", back_populates="permission")
