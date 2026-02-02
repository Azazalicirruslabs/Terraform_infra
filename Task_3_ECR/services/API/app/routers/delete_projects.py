import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.API.app.core.config import check_analysis, get_s3_client
from services.API.app.database.connections import get_db
from shared.auth import get_current_user

load_dotenv()
from shared_migrations.models.file_storage import FileStorage
from shared_migrations.models.user import User

router = APIRouter(prefix="/api", tags=["API"])


@router.delete("/delete_projects")
def delete_projects(
    project_name: list[str],
    analysis_type: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Function implementation goes here
    s3_client = get_s3_client()

    S3_BUCKET = os.getenv("S3_BUCKET")

    analysis = check_analysis(analysis_type)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis Not Found.")

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

    folder_types = ["files", "models"]  # Check both folders

    # Delete project files from S3
    try:
        for project in project_name:
            for folder_type in folder_types:
                FOLDER_PREFIX = f"{tenant_name}/{username}/{analysis.lower()}/{project}/{folder_type}/"
                response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=FOLDER_PREFIX)

                if "Contents" not in response:
                    continue

                objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]

                if objects_to_delete:
                    s3_client.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": objects_to_delete})
            # Also delete entries from the database

            db.query(FileStorage).filter(
                FileStorage.tenant_id == tenant_id,
                FileStorage.user_id == user_id,
                FileStorage.project_name == project,
                FileStorage.analysis_type.ilike(analysis),
            ).delete(synchronize_session=False)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error deleting projects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting projects: {str(e)}")

    return {"message": "Projects deleted successfully."}
