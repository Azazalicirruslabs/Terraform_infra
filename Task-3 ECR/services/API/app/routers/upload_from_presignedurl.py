import os
from itertools import chain
from typing import List, Optional

from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session

from services.API.app.core.config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_MODEL_TYPES,
    S3_BUCKET,
    check_analysis,
    fetch_from_presigned_urls,
    get_s3_client,
    is_valid_model_file,
)
from services.API.app.database.connections import get_db
from services.API.app.models import FileStorage
from services.API.app.utils.project_name_helper import get_next_project_folder
from shared.auth import get_current_user
from shared_migrations.models.user import User

router = APIRouter(prefix="/api", tags=["API"])


@router.post("/upload_from_presignedurl")
async def upload_from_presignedurl(
    analysis_type: str = Form(...),
    project_name: str = Form(None),
    ref_presigned_urls: Optional[List[str]] = Form(default=[]),
    cur_presigned_urls: Optional[List[str]] = Form(default=[]),
    model_presigned_urls: Optional[List[str]] = Form(default=[]),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Uploads files from presigned URLs to S3.
    """
    try:
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

        analysis = check_analysis(analysis_type)
        if not analysis:
            raise HTTPException(status_code=404, detail=f"Analysis Not Found.")
        # Fetch files from the provided presigned URLs
        presigned_files = list(
            chain(
                fetch_from_presigned_urls(ref_presigned_urls),
                fetch_from_presigned_urls(cur_presigned_urls),
                fetch_from_presigned_urls(model_presigned_urls),
            )
        )
        if len(presigned_files) > 3:
            raise HTTPException(status_code=400, detail="Too many files uploaded. max 3 files allowed")
        # Upload each file to S3
        s3_client = get_s3_client()

        # ---------- Delete old files once BEFORE uploading ----------
        if project_name is not None:
            for content_type_group, base_path in [(ALLOWED_CONTENT_TYPES, "files"), (ALLOWED_MODEL_TYPES, "models")]:
                # If any file in current request belongs to this group
                if any(content_type in content_type_group for filename, content_type, content in presigned_files):
                    prefix = f"{tenant_name}/{username}/{analysis.lower()}/{project_name}/{base_path}/"
                    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
                    if "Contents" in response:
                        for obj in response["Contents"]:
                            print(f"[DEBUG] Deleting old file: {obj['Key']}")
                            s3_client.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
        else:
            project_name = (
                project_name if project_name else get_next_project_folder(s3_client, tenant_name, username, analysis)
            )

        # ---------- Now upload all new files ----------
        uploaded_files = []
        db_records = []
        for idx, (filename, content_type, content) in enumerate(presigned_files):
            # Add prefix based on file position
            if idx == 0:
                unique_filename = f"ref_{filename}"
            elif idx == 1:
                unique_filename = f"cur_{filename}"
            else:
                unique_filename = filename  # Or you can define other prefixes

            ext = os.path.splitext(filename)[1].lower()

            if content_type in ALLOWED_CONTENT_TYPES or (ext == ".csv" and content_type == "binary/octet-stream"):
                base_prefix = f"{tenant_name}/{username}/{analysis}/{project_name}/files/"
            elif is_valid_model_file(filename, content_type):
                base_prefix = f"{tenant_name}/{username}/{analysis}/{project_name}/models/"
            else:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported content type or extension: type={content_type}, ext={ext}"
                )

            key = f"{base_prefix}{unique_filename}"
            s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=content)

            AWS_REGION = os.getenv("AWS_REGION")

            s3_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
            # Store in DB
            db_records.append(
                FileStorage(
                    file_name=unique_filename,
                    s3_url=s3_url,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    project_name=project_name,
                    analysis_type=analysis,
                )
            )

            uploaded_files.append(
                {
                    "original_filename": filename,
                    "stored_filename": unique_filename,
                    "s3_url": s3_url,
                }
            )
        # ---------- DB COMMIT IN BULK ----------
        db.add_all(db_records)
        db.commit()

        return {"uploaded_files": uploaded_files}

    except NoCredentialsError:
        raise HTTPException(status_code=403, detail="AWS credentials not found")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Error uploading files to S3: {str(e)}")
