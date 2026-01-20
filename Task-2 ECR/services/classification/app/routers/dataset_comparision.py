from fastapi import APIRouter, Depends

from services.classification.app.config.utils import handle_request
from services.classification.app.core.model_service import ModelService
from shared.auth import get_current_user

router = APIRouter(prefix="/classification", tags=["Classification"])

model_service = ModelService()


@router.get("/dataset-comparison")
async def get_dataset_comparison(user: str = Depends(get_current_user)):
    return handle_request(model_service.get_dataset_comparison)
