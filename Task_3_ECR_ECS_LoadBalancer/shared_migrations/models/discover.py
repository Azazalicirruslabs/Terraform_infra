from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import ARRAY

from shared_migrations.models.base import Base


class Discover(Base):

    __tablename__ = "discover"

    id = Column(Integer, primary_key=True, index=True)
    asset_type = Column(String)
    project_name = Column(String, nullable=True)
    version = Column(String, nullable=True)
    model_type = Column(String, nullable=True)
    lifecycle_state = Column(String, nullable=True)
    uri = Column(String, nullable=True)
    owner = Column(String, nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    description = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
