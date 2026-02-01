from fastapi import APIRouter, Depends, HTTPException

from services.classification.app.config.utils import handle_request
from services.classification.app.core.model_service import ModelService
from services.classification.app.schemas.classification_schema import RequestPayload
from shared.auth import get_current_user

router = APIRouter(prefix="/classification", tags=["Classification"])


model_service = ModelService()


@router.post("/load")
async def load_data(payload: RequestPayload, user: str = Depends(get_current_user)):

    try:
        train_dataset = payload.ref_dataset
        test_dataset = payload.cur_dataset
        model_name = payload.model
        target_column = payload.target_column

        if not model_name:
            raise HTTPException(status_code=400, detail="Missing model name")

        return handle_request(
            model_service.load_model_and_datasets,
            model_path=model_name,
            train_data_path=train_dataset,
            test_data_path=test_dataset,
            target_column=target_column,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
