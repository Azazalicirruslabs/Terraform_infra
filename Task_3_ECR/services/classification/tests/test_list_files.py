import os
from unittest.mock import Mock, patch

import requests
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient


def make_app():
    app = FastAPI()

    # Mock current user dependency
    def mock_current_user():
        return {
            "user_id": "test-user",
            "token": "test-token",
        }

    @app.get("/classification/api/files")
    async def get_s3_file_metadata(
        project_id: str,
        user: dict = Depends(mock_current_user),
    ):
        file_api = os.getenv("FILES_API_BASE_URL", "https://mock-api")
        token = user.get("token")
        user.get("user_id")

        EXTERNAL_S3_API_URL = f"{file_api}/Classification/{project_id}"
        headers = {"Authorization": f"Bearer {token}"}

        try:
            response = requests.get(EXTERNAL_S3_API_URL, headers=headers)
            response.raise_for_status()
            json_data = response.json()

            all_items = json_data.get("files", [])
            files = [i for i in all_items if i.get("folder") == "files"]
            models = [i for i in all_items if i.get("folder") == "models"]

            return {"files": files, "models": models}
        except requests.exceptions.RequestException:
            return None
        except Exception:
            return None

    return app


# -------------------------------------------------------------------
# TEST CASES
# -------------------------------------------------------------------


@patch("requests.get")
def test_get_files_success(mock_get):
    """✔ Valid response from external S3 API should return separated files and models."""
    app = make_app()
    client = TestClient(app)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "files": [{"file_name": "data.csv", "folder": "files"}, {"file_name": "model.pkl", "folder": "models"}]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    resp = client.get("/classification/api/files?project_id=test-project")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["files"]) == 1
    assert len(data["models"]) == 1


@patch("requests.get")
def test_get_files_external_api_failure(mock_get):
    """❌ If external API fails, endpoint should return None."""
    app = make_app()
    client = TestClient(app)

    mock_get.side_effect = requests.exceptions.RequestException("Network error")

    resp = client.get("/classification/api/files?project_id=test-project")
    assert resp.status_code == 200
    assert resp.json() is None


@patch("requests.get")
def test_get_files_empty_list(mock_get):
    """⚠ Empty response should return empty lists, not errors."""
    app = make_app()
    client = TestClient(app)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"files": []}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    resp = client.get("/classification/api/files?project_id=test-project")
    assert resp.status_code == 200
    data = resp.json()
    assert data["files"] == []
    assert data["models"] == []
