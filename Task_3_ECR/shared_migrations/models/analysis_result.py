from sqlalchemy import JSON, Column, DateTime, Integer, String, func

from shared_migrations.models.base import Base


class AnalysisResult(Base):

    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True)
    analysis_id = Column(String, unique=False, index=True)  # Frontend Fetches Processed Result by analysis_id
    user_id = Column(String, nullable=True)  # drift, regression, classification, etc.
    analysis_type = Column(String, nullable=True)  # drift, regression, classification, etc.
    project_id = Column(String, nullable=True)  # project identifier
    analysis_tab = Column(String, nullable=True)  # overview, detailed, etc.
    input_hash = Column(String, nullable=True)  # hash of uploaded data
    json_result = Column(JSON, nullable=True)  #  structured JSON
    created_at = Column(DateTime(timezone=True), default=func.now())
