import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.API.app.core.config import check_analysis, s3_client
from services.API.app.database.connections import get_db
from shared.auth import get_current_user
from shared_migrations.models.user import User

router = APIRouter(prefix="/api", tags=["API"])


@router.get("/projects_list")
def list_projects(
    analysis_type: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)
) -> list[str]:
    """
    List all project folders under analysis.
    Example path: tenant_name/username/analysis/project_1/
    """
    analysis = check_analysis(analysis_type)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis Not Found.")

    user_id = current_user.get("user_id")
    user = db.query(User).filter_by(id=user_id).first()
    tenant_name = user.tenant.name
    username = user.username

    prefix_base = f"{tenant_name}/{username}/{analysis}/"
    response = s3_client.list_objects_v2(Bucket=os.getenv("S3_BUCKET"), Prefix=prefix_base, Delimiter="/")

    projects = []
    if "CommonPrefixes" in response:
        for cp in response["CommonPrefixes"]:
            folder = cp["Prefix"].replace(prefix_base, "").split("/")[0]
            projects.append(folder)

    return projects
