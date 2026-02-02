import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.data_drift.app.routers import (
    columns,
    data_drift_column_report,
    data_drift_full_report,
    llm_column_report,
    llm_full_report,
)
from shared.auth import get_current_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Welcome to Data Drift Detection Service",
    description="Service for detecting data drift in machine learning models.",
    version="3.1.0",
    docs_url="/data_drift/docs",
    openapi_url="/data_drift/openapi.json",
    redoc_url="/data_drift/redocs",
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with ["http://localhost:3000"] or your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/data_drift/health", tags=["health"])
async def health_check():
    return {"status": "Data-drift healthy"}


# Import routers with error handling
try:
    from services.data_drift.app.new.src.shared.upload import router as upload_router

    logger.info("✅ Imported shared upload router")
except ImportError as e:
    logger.error(f"❌ Failed to import shared upload router: {e}")
    upload_router = None

try:
    from services.data_drift.app.new.src.shared.s3_endpoints import router as s3_router

    logger.info("✅ Imported S3 endpoints router")
    # Log the routes in s3_router for debugging
    if hasattr(s3_router, "routes"):
        logger.info(f"S3 Router has {len(s3_router.routes)} routes:")
        for route in s3_router.routes:
            if hasattr(route, "path") and hasattr(route, "methods"):
                logger.info(f"  - {list(route.methods)} {route.path}")
except ImportError as e:
    logger.error(f"❌ Failed to import S3 router: {e}")
    s3_router = None


try:
    from services.data_drift.app.new.src.data_drift.routes.dashboard_new import router as data_drift_dashboard_router

    logger.info("✅ Imported data drift dashboard router")
except ImportError as e:
    logger.error(f"❌ Failed to import data drift dashboard router: {e}")
    data_drift_dashboard_router = None


try:
    from services.data_drift.app.new.src.data_drift.routes.statistical import router as data_drift_statistical_router

    logger.info("✅ Imported data drift statistical router")
except ImportError as e:
    logger.error(f"❌ Failed to import data drift statistical router: {e}")
    data_drift_statistical_router = None

try:
    from services.data_drift.app.new.src.data_drift.routes.feature_analysis import (
        router as data_drift_feature_analysis_router,
    )

    logger.info("✅ Imported data drift feature analysis router")
except ImportError as e:
    logger.error(f"❌ Failed to import data drift feature analysis router: {e}")
    data_drift_feature_analysis_router = None

try:
    from services.data_drift.app.new.src.data_drift.routes.data_overview import (
        router as data_drift_data_overview_router,
    )

    logger.info("✅ Imported data drift data overview router")
except ImportError as e:
    logger.error(f"❌ Failed to import data drift data overview router: {e}")
    data_drift_data_overview_router = None

try:
    from services.data_drift.app.new.src.data_drift.routes.cohort_comparison import (
        router as data_drift_cohort_comparison_router,
    )

    logger.info("✅ Imported data drift cohort comparison router")
except ImportError as e:
    logger.error(f"❌ Failed to import data drift cohort comparison router: {e}")
    data_drift_cohort_comparison_router = None


# Mount routers with better organization and error handling
logger.info("🚀 Mounting routers...")

# Shared services (no prefix - they define their own)
if upload_router:
    app.include_router(upload_router)
    logger.info("✅ Mounted shared upload router")

# S3 services with specific prefix to avoid conflicts
if s3_router:
    app.include_router(s3_router, prefix="/data_drift/v1/s3", tags=["S3 Services"])
    logger.info("✅ Mounted S3 router at /data_drift/v1/s3")

# Data Drift services
if data_drift_dashboard_router:
    app.include_router(data_drift_dashboard_router)
    logger.info("✅ Mounted data drift dashboard router")

if data_drift_statistical_router:
    app.include_router(data_drift_statistical_router)
    logger.info("✅ Mounted data drift statistical router")

if data_drift_feature_analysis_router:
    app.include_router(data_drift_feature_analysis_router)
    logger.info("✅ Mounted data drift feature analysis router")

if data_drift_data_overview_router:
    app.include_router(data_drift_data_overview_router)
    logger.info("✅ Mounted data drift data overview router")

if data_drift_cohort_comparison_router:
    app.include_router(data_drift_cohort_comparison_router)
    logger.info("✅ Mounted data drift cohort comparison router")

logger.info("🎉 All routers mounted successfully!")


# Debug endpoint to show all registered routes
@app.get("/data_drift/v1/debug/routes")
async def debug_routes(user: dict = Depends(get_current_user)):
    """Debug endpoint to show all registered routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            routes.append(
                {"path": route.path, "methods": list(route.methods), "name": getattr(route, "name", "Unknown")}
            )
    return {"total_routes": len(routes), "routes": sorted(routes, key=lambda x: x["path"])}


# Startup event to log all routes
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Application startup complete!")
    logger.info("📋 Registered routes:")
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            logger.info(f"  {list(route.methods)} {route.path}")


# Include Old routers
app.include_router(data_drift_full_report.router)
app.include_router(data_drift_column_report.router)
app.include_router(llm_full_report.router)
app.include_router(llm_column_report.router)
app.include_router(columns.router)
