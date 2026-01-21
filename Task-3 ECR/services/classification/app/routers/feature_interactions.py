import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.classification.app.config.utils import generate_analysis_id, handle_request
from services.classification.app.core.model_service import ModelService
from services.classification.app.database.connections import get_db
from services.classification.app.schemas.classification_schema import RequestPayload
from shared.auth import get_current_user
from shared_migrations.models.analysis_result import AnalysisResult

router = APIRouter(prefix="/classification", tags=["Analysis"])


@router.post("/analysis/feature-interactions", tags=["Analysis"])
async def get_feature_interactions(
    url_payload: RequestPayload,
    feature1: str,
    feature2: str,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()

        # ✅ Validate URL payload fields
        missing_fields = []
        if not url_payload.ref_dataset:
            missing_fields.append("ref_dataset")
        if not url_payload.model:
            missing_fields.append("model")
        if not url_payload.target_column:
            missing_fields.append("target_column")
        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f"❌ Missing required fields in URL payload: {', '.join(missing_fields)}"
            )

        # ✅ Validate features
        if not feature1 or not feature2:
            raise HTTPException(status_code=400, detail="❌ Both 'feature1' and 'feature2' must be provided.")
        if feature1 == feature2:
            raise HTTPException(
                status_code=400, detail="❌ 'feature1' and 'feature2' must be different for interaction analysis."
            )

        train_data_url = url_payload.ref_dataset
        test_data_url = url_payload.cur_dataset
        model_url = url_payload.model
        target_column = url_payload.target_column

        model_overview_payload = {"model_url": model_url, "dataset_url": train_data_url, "target_column": target_column}

        # ✅ Load model and dataset
        try:
            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="❌ Model file or dataset not found at the specified path.")
        except KeyError:
            raise HTTPException(status_code=400, detail=f"❌ Target column '{target_column}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f"❌ Dataset could not be loaded: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"⚠️ Unexpected error while loading model/dataset: {str(e)}")

        # ✅ Compute feature interactions
        try:
            data_preview_response = handle_request(
                model_service.get_feature_interactions, model_overview_payload, feature1, feature2
            )
            data_preview = json.loads(data_preview_response.body)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f"❌ Invalid feature names or data: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"⚠️ Failed to compute feature interactions: {str(e)}")

        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="classification",
                analysis_tab="feature-interactions",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()

        return data_preview

    except HTTPException:
        raise  # preserve custom HTTP messages

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"⚠️ Unexpected internal error while performing feature interaction analysis: {str(e)}",
        )
