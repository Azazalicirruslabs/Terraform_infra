from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from shared_migrations.models.base import Base

# Many-to-Many Tables


class RolePermission(Base):

    __tablename__ = "role_permissions"
    id = Column(Integer, primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.id"), primary_key=True)
    permission_id = Column(Integer, ForeignKey("permissions.id"), primary_key=True)

    role = relationship("Role", back_populates="permissions")
    permission = relationship("Permission", back_populates="roles")
