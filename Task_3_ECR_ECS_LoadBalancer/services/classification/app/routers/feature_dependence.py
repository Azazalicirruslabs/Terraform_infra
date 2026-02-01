import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.classification.app.config.utils import generate_analysis_id, handle_request
from services.classification.app.core.model_service import ModelService
from services.classification.app.database.connections import get_db
from services.classification.app.logging_config import logger
from services.classification.app.schemas.classification_schema import RequestPayload
from shared.auth import get_current_user
from shared_migrations.models.analysis_result import AnalysisResult

router = APIRouter(prefix="/classification", tags=["Analysis"])


@router.post("/analysis/feature-dependence/{feature_name}", tags=["Analysis"])
async def get_feature_dependence(
    payload: RequestPayload, feature_name: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    try:
        model_service = ModelService()
        user_id = user.get("user_id")

        # ✅ Validate payload fields
        if not payload.ref_dataset:
            raise HTTPException(status_code=400, detail="❌ 'ref_dataset' (training dataset) is required.")
        if not payload.cur_dataset:
            raise HTTPException(status_code=400, detail="❌ 'cur_dataset' (test dataset) is required.")
        if not payload.model:
            raise HTTPException(status_code=400, detail="❌ 'model' path or URL is required.")
        if not payload.target_column:
            raise HTTPException(status_code=400, detail="❌ 'target_column' is required.")

        # ✅ Validate feature_name
        if not feature_name:
            raise HTTPException(status_code=400, detail="❌ 'feature_name' must be provided in the path parameter.")

        train_data_url = payload.ref_dataset
        test_data_url = payload.cur_dataset
        model_url = payload.model
        target_column = payload.target_column
        logger.info(f"Feature Dependence Request for feature: {feature_name}")
        model_overview_payload = {"model_url": model_url, "dataset_url": train_data_url, "target_column": target_column}

        # ✅ Load model and datasets
        try:
            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="❌ Model file not found at the specified path.")
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f"❌ Target column '{target_column}' does not exist in the dataset."
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f"❌ Dataset could not be loaded: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"⚠️ Unexpected error while loading model or datasets: {str(e)}")

        # ✅ Perform feature dependence analysis
        try:
            data_preview_response = handle_request(
                model_service.get_feature_dependence, model_overview_payload, feature_name
            )
            data_preview = json.loads(data_preview_response.body)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"⚠️ Failed to generate feature dependence for '{feature_name}': {str(e)}"
            )

        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="classification",
                analysis_tab="feature-dependence",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
            logger.info(f"New AnalysisResult entry created with analysis_id: {analysis_id}")
        return data_preview

    except HTTPException:
        # Preserve custom HTTP exceptions
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"⚠️ Failed to perform feature dependence analysis due to an unexpected internal error: {str(e)}",
        )
