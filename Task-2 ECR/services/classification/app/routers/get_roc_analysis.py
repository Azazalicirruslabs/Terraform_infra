import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.classification.app.config.utils import handle_request
from services.classification.app.core.model_service import ModelService
from services.classification.app.database.connections import get_db
from services.classification.app.schemas.classification_schema import RequestPayload
from shared.auth import get_current_user

router = APIRouter(prefix="/classification", tags=["Classification"])


@router.post("/api/roc-analysis", tags=["AI Analysis"])
async def get_roc_analysis(
    url_payload: RequestPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    try:
        user.get("user_id")
        model_service = ModelService()

        # ✅ Validate required URL payload fields
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

        # ✅ Load model and dataset safely
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

        # ✅ Perform ROC analysis
        try:
            roc_response = handle_request(model_service.roc_analysis, model_overview_payload)
            data_preview = json.loads(roc_response.body)
        except ValueError as ve:
            raise HTTPException(
                status_code=400, detail=f"❌ Failed to compute ROC analysis due to invalid model or data: {str(ve)}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"⚠️ Failed to compute ROC analysis: {str(e)}")

        # ✅ Return final ROC analysis result
        return data_preview

    except HTTPException:
        raise  # preserve our custom messages

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"⚠️ Unexpected internal error while performing ROC analysis: {str(e)}"
        )
