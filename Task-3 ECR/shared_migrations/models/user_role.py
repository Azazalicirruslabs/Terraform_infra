from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from shared_migrations.models.base import Base

# Many-to-Many Tables


class UserRole(Base):

    __tablename__ = "user_roles"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.id"), primary_key=True)

    user = relationship("User", back_populates="roles")
    role = relationship("Role", back_populates="users")
