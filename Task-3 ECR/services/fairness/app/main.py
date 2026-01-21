from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.fairness.app.routers import fairness
from services.fairness.app.src.api import analysis, bias, configuration, feature_interaction, mitigation

app = FastAPI(
    title="Welcome to Fairness Service",
    description="Service for assessing and ensuring fairness in machine learning models.",
    version="3.1.0",
    docs_url="/fairness/docs",
    openapi_url="/fairness/openapi.json",
    redoc_url="/fairness/redocs",
)


# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with ["http://localhost:3000"] or your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/fairness/health", status_code=200)
def health_check():
    return {"status": "Fairness-healthy"}


app.include_router(configuration.router)
app.include_router(analysis.router)
app.include_router(bias.router)
app.include_router(mitigation.router)
app.include_router(feature_interaction.router)
app.include_router(fairness.routers)
