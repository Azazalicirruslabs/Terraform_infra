from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship

from shared_migrations.models.base import Base


class FileStorage(Base):

    __tablename__ = "file_storage"

    id = Column(Integer, primary_key=True)
    file_name = Column(String, nullable=False)
    s3_url = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))  # Foreign key to users table
    tenant_id = Column(Integer, ForeignKey("tenants.id"))  # Foreign key to users table
    created_at = Column(DateTime(timezone=True), default=func.now())
    project_name = Column(String, nullable=True)
    analysis_type = Column(String, nullable=True)

    user = relationship("User", back_populates="files")  # Define reverse relationship
    tenant = relationship("Tenant", back_populates="files")
