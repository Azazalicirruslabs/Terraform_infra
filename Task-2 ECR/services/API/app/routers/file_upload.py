import logging
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from services.API.app.core.config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_MODEL_TYPES,
    S3_BUCKET,
    check_analysis,
    s3_client,
    upload_to_s3,
)
from services.API.app.database.connections import get_db
from services.API.app.models import FileStorage
from services.API.app.utils.project_name_helper import get_next_project_folder
from services.API.app.utils.security import get_current_user
from shared_migrations.models.tenant import Tenant

router = APIRouter(prefix="/api", tags=["API"])

logger = logging.getLogger(__name__)


@router.post("/files_upload")
async def upload_files(
    user_id: int = Form(...),
    tenant_id: int = Form(...),
    analysis_type: str = Form(...),
    files: list[UploadFile] = File(...),
    project_name: str = Form(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    # current_user = Depends(require_permission('upload_data')) # enable this when role and permission based access required
):

    logger.info(f"Starting file upload for {analysis_type} under project {project_name}")
    username = current_user.username if current_user else None
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized: User not found.")
    tenant_data = db.query(Tenant).filter_by(id=tenant_id).first()
    if not tenant_data:
        raise HTTPException(status_code=404, detail="Tenant not found")
    tenant_name = tenant_data.name

    if tenant_name is None:
        raise HTTPException(status_code=404, detail="Tenant name not found")

    analysis = check_analysis(analysis_type)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis Not Found.")

    uploaded_files = []

    if project_name is not None:

        # ---------- Delete old files once BEFORE uploading ----------
        for content_type_group, base_path in [(ALLOWED_CONTENT_TYPES, "files"), (ALLOWED_MODEL_TYPES, "models")]:
            # If any file in current request belongs to this group
            if any(
                (
                    os.path.splitext(file.filename)[1].lower() == ".parquet"
                    and file.content_type == "application/octet-stream"
                )
                or (file.content_type in content_type_group)
                for file in files
            ):
                prefix = f"{tenant_name}/{username}/{analysis.lower()}/{project_name}/{base_path}/"
                response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
                if "Contents" in response:
                    for obj in response["Contents"]:
                        print(f"[DEBUG] Deleting old file: {obj['Key']}")
                        s3_client.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
    else:
        # project_name = get_next_project_folder(s3_client, tenant_name, username, analysis)
        project_name = (
            project_name if project_name else get_next_project_folder(s3_client, tenant_name, username, analysis)
        )

    # ---------- Now upload all new files ----------
    for idx, file in enumerate(files):
        # Add prefix based on file position
        if idx == 0:
            unique_filename = f"ref_{file.filename}"
        elif idx == 1:
            unique_filename = f"cur_{file.filename}"
        else:
            unique_filename = file.filename  # Or you can define other prefixes

        file_bytes = await file.read()

        # Upload to S3
        s3_url = upload_to_s3(
            username, file_bytes, unique_filename, file.content_type, tenant_id, db, project_name, analysis
        )

        # Store in DB
        file_record = FileStorage(
            file_name=unique_filename,
            s3_url=s3_url,
            user_id=user_id,
            tenant_id=tenant_id,
            project_name=project_name,
            analysis_type=analysis,
        )
        db.add(file_record)
        db.commit()
        db.refresh(file_record)

        uploaded_files.append(
            {
                "original_filename": file.filename,
                "stored_filename": unique_filename,
                "s3_url": s3_url,
                "id": file_record.id,
            }
        )
    logger.info(f"Completed file upload for {username} analysis: {analysis_type}, project: {project_name}")
    return {"uploaded_files": uploaded_files}
