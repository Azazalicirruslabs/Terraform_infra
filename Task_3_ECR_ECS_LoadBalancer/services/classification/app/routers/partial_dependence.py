import json
from typing import Dict

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.orm import Session

from services.classification.app.config.utils import generate_analysis_id, handle_request
from services.classification.app.core.model_service import ModelService
from services.classification.app.database.connections import get_db
from services.classification.app.schemas.classification_schema import RequestPayload
from shared.auth import get_current_user
from shared_migrations.models.analysis_result import AnalysisResult

router = APIRouter(prefix="/classification", tags=["Dependence"])


@router.post("/api/partial-dependence", tags=["Dependence"])
async def post_partial_dependence(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()

        # ✅ Validate required payload fields
        feature = payload.get("feature")
        if not feature:
            raise HTTPException(status_code=400, detail="❌ Missing required field in request body: 'feature'.")

        # ✅ Validate num_points (must be a positive integer)
        try:
            num_points = int(payload.get("num_points", 20))
            if num_points <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="❌ 'num_points' must be a positive integer.")

        # ✅ Validate URL payload
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

        train_data_url = url_payload.ref_dataset
        test_data_url = url_payload.cur_dataset
        model_url = url_payload.model
        target_column = url_payload.target_column

        model_overview_payload = {"model_url": model_url, "dataset_url": train_data_url, "target_column": target_column}

        # ✅ Safely load model and dataset
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
            raise HTTPException(status_code=400, detail=f"❌ Dataset could not be loaded properly: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"⚠️ Unexpected error while loading model/dataset: {str(e)}")

        # ✅ Compute partial dependence
        try:
            data_preview_response = handle_request(
                model_service.partial_dependence, model_overview_payload, feature, num_points
            )
            data_preview = json.loads(data_preview_response.body)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"❌ Feature '{feature}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(
                status_code=400,
                detail=f"❌ Failed to compute partial dependence due to invalid parameters or data: {str(ve)}",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"⚠️ Failed to compute partial dependence: {str(e)}")

        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="classification",
                analysis_tab="partial-dependence",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()

        return data_preview

    except HTTPException:
        raise  # keep custom messages

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"⚠️ Unexpected internal error while performing partial dependence analysis: {str(e)}",
        )
