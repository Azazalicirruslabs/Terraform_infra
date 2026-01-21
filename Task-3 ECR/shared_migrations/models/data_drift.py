from sqlalchemy import JSON, Column, DateTime, Integer, String, func

from shared_migrations.models.base import Base


class DataDrift(Base):
    __tablename__ = "data_drift"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True, index=True)
    tenant_id = Column(Integer, nullable=True, index=True)
    project_id = Column(Integer, nullable=True, index=True)
    file_name = Column(String, nullable=True, unique=True, index=True)
    drift_json_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
