import mimetypes
import os
from io import BytesIO
from typing import List, Optional
from urllib.parse import unquote, urlparse

import requests
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from services.API.app.core.config import S3_BUCKET, TEMPORARY_FILES, get_s3_client
from services.API.app.database.connections import get_db
from services.API.app.schemas.DB_Connection import TransferToS3Request
from shared.auth import get_current_user
from shared_migrations.models.user import User

router = APIRouter(prefix="/api", tags=["API"])


def fetch_from_presigned_urls(presigned_urls: List[str]) -> List[tuple[str, bytes]]:
    """
    Downloads files from the given list of presigned S3 URLs.
    Returns a list of (filename, content) tuples.
    """
    results = []
    for url in presigned_urls:
        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"Failed to download from URL: {url}")
            parsed_url = urlparse(url)
            filename = unquote(os.path.basename(parsed_url.path))  # Handles %20, %2F etc.
            results.append((filename, response.content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error fetching file from presigned URL")
    return results


def fetch_temp_files_from_s3(username: str, tenant_name: str, analysis_type: str) -> List[tuple[str, bytes]]:
    """
    Fetches 'temporary_files' from S3 instead of local storage.
    Looks inside: {tenant_name}/{username}/{analysis_type}/files/
    Returns a list of (filename, content) tuples.
    """
    try:
        s3_client = get_s3_client()
        prefix = f"{tenant_name}/{username}/{analysis_type.lower()}/files/"
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

        if "Contents" not in response:
            return []

        temp_files = []
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith("/"):
                continue  # Skip folder marker
            file_content = s3_client.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
            filename = os.path.basename(key)
            temp_files.append((filename, file_content))

        return temp_files
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error fetching temporary files from S3: {str(e)}")


@router.post("/transfer-to-s3", response_model=TransferToS3Request)
async def upload_db_files(
    analysis_type: str = Form(...),
    files: Optional[list[UploadFile]] = File(default=None),
    ref_presigned_urls: Optional[List[str]] = Form(default=[]),
    cur_presigned_urls: Optional[List[str]] = Form(default=[]),
    model_presigned_urls: Optional[List[str]] = Form(default=[]),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Uploads files to an external API, combining files from user upload and a temporary directory.
    """
    email_id = current_user.get("username") if current_user else None
    if not email_id:
        raise HTTPException(status_code=401, detail="Unauthorized: User not found.")

    user_obj = db.query(User).filter(User.email == email_id).first() if email_id else None
    username = user_obj.username if user_obj else None
    if not user_obj or not username:
        raise HTTPException(status_code=401, detail="Unauthorized: User not found.")
    token = current_user.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized: Token not found.")
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized: User ID not found.")

    tenant_id = user_obj.tenant_id if user_obj else None
    tenant_name = None
    if user_obj and user_obj.tenant:
        tenant_name = user_obj.tenant.name
    if not tenant_name:
        raise HTTPException(status_code=404, detail="Tenant name not found.")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Unauthorized: Tenant ID not found.")

    response_message = ""
    # Fetch temp files from S3 (instead of local)
    temp_dir_files = fetch_temp_files_from_s3(username, tenant_name, TEMPORARY_FILES)
    if not temp_dir_files:
        response_message += "Not Uploading Data From Database (S3 temporary files not found)"

    # Gather user-uploaded files
    uploaded_files = []
    if files is not None:
        # Gather user-uploaded files
        if files is not None:
            for file in files:
                try:
                    if not file.filename:
                        continue
                    content = await file.read()
                    uploaded_files.append((file.filename, content))
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error reading file {file.filename}: {str(e)}")
        else:
            response_message += f"Not Uploading Data From User Uploads"

    # Gather files from presigned URLs
    if not any([ref_presigned_urls, cur_presigned_urls, model_presigned_urls]):
        response_message += "No presigned URLs provided for file upload."

    presigned_files = (
        fetch_from_presigned_urls(ref_presigned_urls)
        + fetch_from_presigned_urls(cur_presigned_urls)
        + fetch_from_presigned_urls(model_presigned_urls)
    )

    all_files = temp_dir_files + uploaded_files + presigned_files

    if not all_files:
        raise HTTPException(
            status_code=400, detail="No files found to upload (neither uploaded nor in temporary directory)."
        )

    # Upload all files to external API

    multipart_files = []
    for fname, content in all_files:
        mime_type, _ = mimetypes.guess_type(fname)
        mime_type = mime_type or "application/octet-stream"

        file_stream = BytesIO(content)
        file_stream.seek(0)
        # Each file entry must have the same field name "files"
        multipart_files.append(("files", (fname, file_stream, mime_type)))
    files_upload_api = os.getenv("FILES_UPLOAD_API", "http://localhost:8000/api/files_upload")
    if not files_upload_api:
        raise HTTPException(status_code=500, detail="Files upload API URL is not configured.")
    response = requests.post(
        files_upload_api,
        data={
            "user_id": str(user_id),
            "tenant_id": str(tenant_id),
            "analysis_type": analysis_type,
        },
        files=multipart_files,
        headers={"Authorization": f"Bearer {token}"},
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Failed to upload file {fname}: {response.text}")

    if response.status_code == 200:
        # ---------- Delete old files after uploading to S3 ----------
        s3_client = get_s3_client()

        for base_path in ["files", "models"]:
            prefix = f"{tenant_name}/{username}/{TEMPORARY_FILES}/{base_path}/"
            try:
                existing = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
                if "Contents" in existing:
                    for obj in existing["Contents"]:
                        print(f"[DEBUG] Deleting old file: {obj['Key']}")
                        s3_client.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
            except Exception as e:
                print(f"[ERROR] Failed to delete old files in {prefix}: {e}")

    return {"status": "success", "message": "Files uploaded successfully", "additional_info": response_message}
