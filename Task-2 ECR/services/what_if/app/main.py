from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.what_if.app.routers.admin_clear_cache import router as admin_clear_cache_router
from services.what_if.app.routers.features import router as features_router
from services.what_if.app.routers.llmexplainer import router as llmexplainer_router
from services.what_if.app.routers.metadata import router as metadata_router
from services.what_if.app.routers.predict import router as predict_router
from services.what_if.app.routers.profile import router as profile_router
from services.what_if.app.routers.session_delete import router as session_delete_router
from services.what_if.app.routers.session_refresh import router as session_refresh_router
from services.what_if.app.routers.session_s3 import router as session_s3_router
from services.what_if.app.routers.shap import router as shap_router

app = FastAPI(
    title="Welcome to What If Service",
    description="This service handles regression and classification tasks, provides endpoints for model predictions, and includes performance and explainability analysis capabilities",
    version="3.1.0",
    docs_url="/what_if/docs",
    openapi_url="/what_if/openapi.json",
    redoc_url="/what_if/redocs",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_origins=["*"],  # You can replace "*" with specific origins for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/what_if/health", status_code=200)
def health_check():
    return {"status": "what_if-healthy"}


app.include_router(session_s3_router)
app.include_router(predict_router)
app.include_router(shap_router)
app.include_router(metadata_router)
app.include_router(features_router)
app.include_router(llmexplainer_router)
app.include_router(profile_router)
app.include_router(admin_clear_cache_router)
app.include_router(session_refresh_router)
app.include_router(session_delete_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
