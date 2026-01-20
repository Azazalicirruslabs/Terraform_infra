import logging
import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.API.app.core.config import check_analysis, get_s3_client
from services.API.app.database.connections import get_db
from services.API.app.utils.security import get_current_user
from shared_migrations.models.tenant import Tenant

load_dotenv()

router = APIRouter(prefix="/api", tags=["API"])

logger = logging.getLogger(__name__)


@router.get("/files_download/{analysis_type}/{project_name}")
def list_files_and_generate_presigned_urls(
    analysis_type: str, project_name: str, current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)
):
    S3_BUCKET = os.getenv("S3_BUCKET")

    analysis = check_analysis(analysis_type)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis Not Found.")

    s3_client = get_s3_client()
    logger.info(f"S3 client initialized and Fetching files for analysis: {analysis}, project: {project_name}")

    username = current_user.username
    tenant_id = current_user.tenant_id
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    tenant_name = tenant.name
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized: User not found.")

    # Generate presigned URLs for all files
    presigned_files = []
    folder_types = ["files", "models"]  # Check both folders
    try:

        for folder_type in folder_types:

            FOLDER_PREFIX = f"{tenant_name}/{username}/{analysis.lower()}/{project_name}/{folder_type}/"
            response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=FOLDER_PREFIX)

            if "Contents" not in response:
                continue

            for obj in response["Contents"]:
                key = obj["Key"]
                if key.endswith("/"):
                    continue  # Skip folder marker

                url = s3_client.generate_presigned_url(
                    "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=3600  # 1 hour
                )
                presigned_files.append({"file_name": key[len(FOLDER_PREFIX) :], "folder": folder_type, "url": url})

        if not presigned_files:
            raise HTTPException(status_code=404, detail="No files found in S3 folder.")
        logger.info(f"Generated presigned URLs for user: {username}")
        return {"files": presigned_files}

    except Exception as e:
        logger.error(f"Error generating presigned URLs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
