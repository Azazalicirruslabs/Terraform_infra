"""
BiasLens™ Analyzer - Fairness Files Router

This module provides the API endpoint for listing uploaded project files.

Endpoints:
    GET /mainflow/fairness/files/{analysis_type}/{project_name} - List uploaded files
"""

import logging
import os
from typing import Any, Dict

import requests
from fastapi import APIRouter, Depends, HTTPException

from shared.auth import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

# Get API URLs from environment
FILES_API_BASE_URL = os.getenv("FILES_API_BASE_URL")

# Create router
router = APIRouter(prefix="/mainflow/fairness", tags=["BiasLens Fairness"])


@router.get("/files/{analysis_type}/{project_name}", status_code=200)
async def list_project_files(
    analysis_type: str, project_name: str, current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all uploaded files for a specific project.

    **Authentication:** Requires Bearer token in Authorization header.

    This endpoint retrieves metadata for all files and models that users have
    already uploaded via the main API's /api/files_upload endpoint.

    Users can then extract the URLs from the response and pass them to
    the /analyze endpoint along with fairness configuration.

    Args:
        analysis_type: Analysis type (e.g., "Fairness")
        project_name: Project name

    Returns:
        Dictionary with file URLs ready to use in /analyze endpoint

    Example Response:
        ```json
        {
            "analysis_type": "Fairness",
            "project_name": "my_project",
            "files": [
                {
                    "name": "training_data.csv",
                    "url": "https://s3.amazonaws.com/...",
                    "type": "dataset"
                },
                {
                    "name": "testing_data.csv",
                    "url": "https://s3.amazonaws.com/...",
                    "type": "dataset"
                },
                {
                    "name": "model.pkl",
                    "url": "https://s3.amazonaws.com/...",
                    "type": "model"
                }
            ]
        }
        ```
    """
    logger.info(f"Listing files for project: {project_name}, analysis_type: {analysis_type}")
    logger.info(f"User: {current_user.get('username')}")

    try:
        if not FILES_API_BASE_URL:
            raise HTTPException(
                status_code=500, detail="File download service not configured. Please contact administrator."
            )

        # Request file list from storage service using the user's token
        list_url = f"{FILES_API_BASE_URL}/{analysis_type}/{project_name}"
        headers = {"Authorization": f"Bearer {current_user.get('token')}"}

        logger.debug(f"Fetching files from: {list_url}")

        response = requests.get(list_url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        logger.info(f"Retrieved file list successfully for {project_name}")

        return {
            "analysis_type": analysis_type,
            "project_name": project_name,
            "files": data.get("files", []),
            "message": "Use the file URLs in the /analyze endpoint",
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to list files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file list from storage service: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")
