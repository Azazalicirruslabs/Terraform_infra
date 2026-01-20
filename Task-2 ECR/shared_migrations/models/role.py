from sqlalchemy import Boolean, Column, DateTime, Integer, String, func
from sqlalchemy.orm import relationship

from shared_migrations.models.base import Base


class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    status = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    permissions = relationship(
        "RolePermission", back_populates="role"
    )  # It defines a one-to-many relationship from Role → RolePermission
    users = relationship(
        "UserRole", back_populates="role"
    )  # back_populates="role"-> This creates a bi-directional relationship
