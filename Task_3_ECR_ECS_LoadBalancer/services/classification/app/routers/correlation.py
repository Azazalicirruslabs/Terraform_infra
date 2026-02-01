import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.classification.app.config.utils import generate_analysis_id, handle_request
from services.classification.app.core.model_service import ModelService
from services.classification.app.database.connections import get_db
from services.classification.app.schemas.classification_schema import CorrelationPayload
from shared.auth import get_current_user
from shared_migrations.models.analysis_result import AnalysisResult

router = APIRouter(prefix="/classification", tags=["Features"])


@router.post("/api/correlation", tags=["Features"])
async def post_correlation(
    payload: CorrelationPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()

        # Extract payload values
        selected: List[str] = payload.features
        train_data_url = payload.ref_dataset
        test_data_url = payload.cur_dataset
        model_url = payload.model
        target_column = payload.target_column

        # ✅ Validate required fields
        missing_fields = []
        if not train_data_url:
            missing_fields.append("ref_dataset")
        if not model_url:
            missing_fields.append("model")
        if not target_column:
            missing_fields.append("target_column")
        if missing_fields:
            raise HTTPException(status_code=400, detail=f"❌ Missing required fields: {', '.join(missing_fields)}")

        model_overview_payload = {"model_url": model_url, "dataset_url": train_data_url, "target_column": target_column}

        # ✅ Load model and dataset safely
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
            raise HTTPException(status_code=400, detail=f"❌ Target column '{target_column}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f"❌ Dataset could not be loaded: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"⚠️ Unexpected error while loading model/dataset: {str(e)}")

        # ✅ Compute correlation
        try:
            data_preview_response = handle_request(model_service.compute_correlation, model_overview_payload, selected)
            data_preview = json.loads(data_preview_response.body)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"⚠️ Failed to compute correlation: {str(e)}")

        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="classification",
                analysis_tab="correlation",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()

        return data_preview

    except HTTPException:
        raise  # Preserve custom HTTP exceptions

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"⚠️ Unexpected internal error while computing correlation: {str(e)}"
        )
