from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.classification.app.config.utils import (
    generate_analysis_id,
    handle_request,
    insert_analyzed_data_to_database,
)
from services.classification.app.core.model_service import ModelService
from services.classification.app.database.connections import get_db
from services.classification.app.logging_config import logger
from services.classification.app.schemas.classification_schema import RequestPayload
from shared.auth import get_current_user
from shared_migrations.models.analysis_result import AnalysisResult

router = APIRouter(prefix="/classification", tags=["Analysis"])
import json


@router.post("/analysis/overview")
async def get_overview(payload: RequestPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    model_service = ModelService()
    train_data_url = payload.ref_dataset
    test_data_url = payload.cur_dataset
    model_url = payload.model
    target_column = payload.target_column

    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    logger.info(f"Received analysis overview request with model_url")

    user_id = user.get("user_id")
    data_preview_response = handle_request(
        model_service.get_classification_stats,
        {"model_url": model_url, "dataset_url": train_data_url, "target_column": target_column},
    )
    data_preview = json.loads(data_preview_response.body)

    # For true microservice architecture, always process fresh data
    # Generate analysis_id for potential database storage (optional)
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column,
        )

        # Optional: Store analysis result in database (could be made configurable)
        if not existing:
            insert_analyzed_data_to_database(
                analysis_id=analysis_id,
                user_id=user_id,
                data_preview=data_preview,
                analysis_tab="Overview",
                project_id="N/A",
                db=db,
            )

        # Get model overview with dynamic loading support
        model_overview_payload = {"model_url": model_url, "dataset_url": train_data_url, "target_column": target_column}
        return model_service.get_model_overview(model_overview_payload)
    except Exception as e:
        logger.error(
            f"Error loading model/data from S3 for user_id: {user_id} with model_url: {model_url}, train_data_url: {train_data_url}, test_data_url: {test_data_url}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")
