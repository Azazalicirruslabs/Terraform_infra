import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from services.API.app.core.config import ALLOWED_MODEL_TYPES, upload_to_s3
from services.API.app.database.connections import get_db
from services.API.app.models import FileStorage
from services.API.app.utils.security import get_current_user

router = APIRouter(prefix="/api", tags=["API"])


@router.post("/models_upload")
async def upload_files(
    user_id: int = Form(...),
    tenant_id: int = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    uploaded_files = []

    for file in files:
        if file.content_type not in ALLOWED_MODEL_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename} ({file.content_type})")

        unique_filename = f"model_{uuid.uuid4()}_{file.filename}"
        file_bytes = await file.read()

        # Upload to S3
        s3_url = upload_to_s3(file_bytes, unique_filename, file.content_type, tenant_id, db=db)

        # Store in DB
        file_record = FileStorage(file_name=unique_filename, s3_url=s3_url, user_id=user_id, tenant_id=tenant_id)
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

    return {"uploaded_models": uploaded_files}
