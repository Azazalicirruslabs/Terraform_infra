from fastapi import APIRouter, Depends, HTTPException

from services.fairness.app.src.models import global_session
from services.fairness.app.src.schemas.request_response import (
    ConfigurationResponse,
    StatusResponse,
    TargetColumnRequest,
)
from services.fairness.app.src.utils import DataValidator, FileManager
from services.fairness.app.utils.helper_functions import get_s3_file_metadata
from shared.auth import get_current_user

router = APIRouter(prefix="/fairness/config", tags=["Configuration"])


@router.post("/set-target-column", response_model=ConfigurationResponse)
async def set_target_column(
    request: TargetColumnRequest, current_user: str = Depends(get_current_user), project_id: str = None
):
    """Set target column for analysis"""
    token = current_user.get("token")
    get_s3_file_metadata(token, project_id)
    try:
        if global_session.train_file_path is None:
            raise HTTPException(status_code=400, detail="Please upload training data first")

        # Load train data
        train_df = FileManager.load_csv(global_session.train_file_path)

        # Validate target column
        is_valid, message = DataValidator.validate_target_column(train_df, request.target_column)

        if not is_valid:
            raise HTTPException(status_code=400, detail=message)

        global_session.target_column = request.target_column

        return ConfigurationResponse(
            status="success", message=f"Target column '{request.target_column}' set successfully", configured=True
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting target column: {str(e)}")


@router.get("/status", response_model=StatusResponse)
async def get_status(current_user: str = Depends(get_current_user), project_id: str = None):
    """Get current configuration status"""
    token = current_user.get("token")
    get_s3_file_metadata(token, project_id)
    try:
        train_shape = None
        test_shape = None
        columns = None

        if global_session.train_file_path:
            train_df = FileManager.load_csv(global_session.train_file_path)
            train_shape = train_df.shape
            columns = train_df.columns.tolist()

        if global_session.test_file_path:
            test_df = FileManager.load_csv(global_session.test_file_path)
            test_shape = test_df.shape

        return StatusResponse(
            status="success",
            train_uploaded=global_session.train_file_path is not None,
            test_uploaded=global_session.test_file_path is not None,
            model_uploaded=global_session.model_file_path is not None,
            target_column=global_session.target_column,
            train_shape=train_shape,
            test_shape=test_shape,
            columns=columns,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")


@router.post("/reset")
async def reset_session(current_user: str = Depends(get_current_user), project_id: str = None):
    """Reset entire session"""
    try:
        global_session.clear()
        FileManager.clear_files()

        return {"status": "success", "message": "Session reset successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting session: {str(e)}")
