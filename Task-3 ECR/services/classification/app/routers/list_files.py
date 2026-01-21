import os

import requests
from fastapi import APIRouter, Depends

from services.classification.app.logging_config import logger
from shared.auth import get_current_user

router = APIRouter(prefix="/classification", tags=["Mandatory"])


@router.get("/api/files")
def get_s3_file_metadata(project_id: str, user: str = Depends(get_current_user)):
    """
    Lists files and models from the external S3 API and returns their metadata (name, URL, folder).
    Separates files and models based on the folder field.
    """
    file_api = os.getenv("FILES_API_BASE_URL")
    token = user.get("token")  # Assuming the token is stored in the user object
    user_id = user.get("user_id")
    EXTERNAL_S3_API_URL = f"{file_api}/Classification/{project_id}"
    headers = {"Authorization": f"Bearer {token}"}
    logger.info(f"Fetching S3 file metadata for project_id: {project_id} and user_id: {user_id}")
    try:
        response = requests.get(EXTERNAL_S3_API_URL, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        json_data = response.json()
        all_items = json_data.get("files", [])

        # Separate files and models based on folder
        files = [item for item in all_items if item.get("folder") == "files"]
        models = [item for item in all_items if item.get("folder") == "models"]

        return {"files": files, "models": models}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to external S3 API for project_id: {project_id} and user_id: {user_id}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Error processing external S3 API response for project_id: {project_id} and user_id: {user_id}: {e}"
        )
        return None
