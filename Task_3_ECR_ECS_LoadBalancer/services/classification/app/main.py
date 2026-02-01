import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.classification.app.api.api_router import router as api_router
from services.classification.app.routers.api_feature_impotance import router as api_feature_importance
from services.classification.app.routers.classification_stats import router as classification_stats
from services.classification.app.routers.correlation import router as correlation
from services.classification.app.routers.decision_tree import router as decision_tree
from services.classification.app.routers.explain_instance import router as explain_instance
from services.classification.app.routers.explain_with_ai import router as explain_with_ai
from services.classification.app.routers.feature_dependence import router as feature_dependence
from services.classification.app.routers.feature_impotance import router as feature_importance
from services.classification.app.routers.feature_interactions import router as feature_interactions
from services.classification.app.routers.features import router as features
from services.classification.app.routers.get_roc_analysis import router as get_roc_analysis
from services.classification.app.routers.ice_plot import router as ice_plot
from services.classification.app.routers.individual_prediction import router as individual_prediction
from services.classification.app.routers.instances import router as instances
from services.classification.app.routers.interaction_network import router as interaction_network
from services.classification.app.routers.list_files import router as files
from services.classification.app.routers.load import router as load
from services.classification.app.routers.overview import router as overview
from services.classification.app.routers.pairwise_analysis import router as pairwise_analysis
from services.classification.app.routers.partial_dependence import router as partial_dependence
from services.classification.app.routers.post_roc_analysis import router as post_roc_analysis
from services.classification.app.routers.shap_dependence import router as shap_dependence
from services.classification.app.routers.threshold_analysis import router as threshold_analysis
from services.classification.app.routers.what_if import router as what_if

app = FastAPI(
    title="Welcome to Classification Service",
    description="Service for managing classification endpoints.",
    version="3.1.0",
    docs_url="/classification/docs",
    openapi_url="/classification/openapi.json",
    redoc_url="/classification/redocs",
)
# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with ["http://localhost:3000"] or your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/classification/health", status_code=200)
def health_check():
    return {"status": "classification-healthy"}


app.include_router(files)
app.include_router(load)
app.include_router(overview)
app.include_router(classification_stats)
app.include_router(api_feature_importance)
app.include_router(correlation)
app.include_router(decision_tree)
app.include_router(explain_instance)
app.include_router(explain_with_ai)
app.include_router(feature_dependence)
app.include_router(feature_importance)
app.include_router(feature_interactions)
app.include_router(features)
app.include_router(get_roc_analysis)
app.include_router(ice_plot)
app.include_router(individual_prediction)
app.include_router(instances)
app.include_router(interaction_network)
app.include_router(pairwise_analysis)
app.include_router(partial_dependence)
app.include_router(post_roc_analysis)
app.include_router(shap_dependence)
app.include_router(threshold_analysis)
app.include_router(what_if)


app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
