import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.mainflow.app.core import logging_config
from services.mainflow.app.routers import (
    explainability_analysis,
    fairness_files,
    fairness_metrics,
    fairness_summary,
    fairness_thresholds,
    health,
    manual_registration,
    registration_fetch,
    search_discover,
)

logging_config.setup_logging()

app = FastAPI(
    title="Welcome to Mainflow Service",
    description="Service for managing Mainflow requests.",
    version="3.1.0",
    docs_url="/mainflow/docs",
    openapi_url="/mainflow/openapi.json",
    redoc_url="/mainflow/redocs",
)

logger = logging.getLogger(__name__)
logger.info("Mainflow service starting up.")

# Allow CORS for all origins (you can customize this for your needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(manual_registration.router)
app.include_router(registration_fetch.router)
app.include_router(search_discover.router)

# Fairness (BiasLens) routers
app.include_router(fairness_summary.router)
app.include_router(fairness_metrics.router)
app.include_router(fairness_thresholds.router)
app.include_router(fairness_files.router)
app.include_router(explainability_analysis.router)
