from services.API.app.logging_config import setup_logging

setup_logging()

from fastapi import FastAPI

from services.API.app.routers import (
    addrole,
)
from services.API.app.routers import assign_role_to_users as addrole
from services.API.app.routers import (
    auth,
    database_connection_test,
    delete_projects,
    download_files,
    file_upload,
    plan,
    project_list,
    subscription,
    tenant,
    upload_files_from_db,
    upload_from_presignedurl,
    users,
)

app = FastAPI(
    title="Welcome to API Service",
    description="Service for managing API requests.",
    version="3.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url="/api/redocs",
)

from fastapi.middleware.cors import CORSMiddleware

# Allow CORS for all origins (you can customize this for your needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with ["http://localhost:3000"] or your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", status_code=200)
def health_check():
    return {"status": "API-healthy"}


app.include_router(plan.router)
app.include_router(subscription.router)
app.include_router(users.router)
app.include_router(tenant.router)
app.include_router(auth.router)
app.include_router(file_upload.router)
app.include_router(download_files.router)
app.include_router(addrole.router)
app.include_router(database_connection_test.router)
app.include_router(upload_files_from_db.router)
app.include_router(upload_from_presignedurl.router)
app.include_router(project_list.router)
app.include_router(delete_projects.router)
