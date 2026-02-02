import json
import os
from typing import Dict, List, Optional

import requests
from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.regression.app.core.ai_explanation_service import AIExplanationService
from services.regression.app.core.model_service import ModelService
from services.regression.app.database.connections import get_db
from services.regression.app.logging_config import logger
from services.regression.app.routers import regression
from services.regression.app.schemas.regression_schema import CorrelationPayload, RequestPayload
from services.regression.app.utils.analysisid import generate_analysis_id
from services.regression.app.utils.error_handler import handle_request
from shared.auth import get_current_user
from shared_migrations.models.analysis_result import AnalysisResult

app = FastAPI(
    title="Welcome to Regression Service",
    description="This service handles regression tasks, provides endpoints for model predictions, and includes performance and explainability analysis capabilities.",
    version="3.1.0",
    docs_url="/regression/docs",
    openapi_url="/regression/openapi.json",
    redoc_url="/regression/redocs",
)


@app.get("/regression/health", tags=["health"])
async def health_check():
    return {"status": "Regression healthy"}


# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with ["http://localhost:3000"] or your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- API Endpoints ---


@app.get("/regression/api/files", tags=["Mandatory"])
def get_s3_file_metadata(project_id: str, user: str = Depends(get_current_user)):
    """
    Lists files and models from the external S3 API and returns their metadata (name, URL, folder).
    Separates files and models based on the folder field.
    """
    file_api = os.getenv("FILES_API_BASE_URL")
    token = user.get("token")
    if not file_api:
        raise HTTPException(status_code=500, detail="FILES_API_BASE_URL environment variable is not set.")
    if not token:
        raise HTTPException(status_code=401, detail="User token is missing or invalid.")

    EXTERNAL_S3_API_URL = f"{file_api}/Regression/{project_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(EXTERNAL_S3_API_URL, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        json_data = response.json()
        all_items = json_data.get("files", [])

        # Separate files and models based on folder
        files = [item for item in all_items if item.get("folder") == "files"]
        models = [item for item in all_items if item.get("folder") == "models"]

        return {
            "success": True,
            "files": files,
            "models": models,
            "total_files": len(files),
            "total_models": len(models),
        }
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to external S3 API: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to external S3 API: {str(e)}")
    except Exception as e:
        print(f"Error processing external S3 API response: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing S3 API response: {str(e)}")


class LoadDataRequest(BaseModel):
    cur_dataset: Optional[str] = None
    ref_dataset: Optional[str] = None
    model: str
    target_column: str = "target"


@app.post("/regression/load", tags=["Mandatory"])
async def load_data(payload: LoadDataRequest, user: str = Depends(get_current_user)):

    try:
        model_service = ModelService()
        model_name = payload.model
        train_dataset = payload.ref_dataset
        test_dataset = payload.cur_dataset
        target_column = payload.target_column

        if not model_name:
            raise HTTPException(status_code=400, detail="Missing model name")

        return handle_request(
            model_service.load_model_and_datasets,
            model_path=model_name,
            train_data_path=train_dataset,
            test_data_path=test_dataset,
            target_column=target_column,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/regression/analysis/overview", tags=["Analysis"])
async def get_overview(payload: RequestPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    model_service = ModelService()
    user_id = user.get("user_id")
    train_data_url = payload.ref_dataset
    test_data_url = payload.cur_dataset
    model_url = payload.model
    target_column = payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.get_model_overview)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="overview",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.get_model_overview)


@app.post("/regression/analysis/regression-stats", tags=["Analysis"])
async def get_regression_statistics(
    payload: RequestPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    model_service = ModelService()
    user_id = user.get("user_id")
    train_data_url = payload.ref_dataset
    test_data_url = payload.cur_dataset
    model_url = payload.model
    target_column = payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.get_regression_stats)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="regression-stats",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()

    return handle_request(model_service.get_regression_stats)


@app.post("/regression/analysis/feature-importance", tags=["Analysis"])
async def get_feature_importance(
    payload: RequestPayload, method: str = "shap", user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    model_service = ModelService()
    user_id = user.get("user_id")
    train_data_url = payload.ref_dataset
    test_data_url = payload.cur_dataset
    model_url = payload.model
    target_column = payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    # Use advanced method with default parameters to get comprehensive response including impact directions
    data_preview_response = handle_request(
        model_service.compute_feature_importance_advanced, method, "importance", 1000, "bar"
    )
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="feature-importance",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.compute_feature_importance_advanced, method, "importance", 1000, "bar")


@app.post("/regression/analysis/explain-instance/{instance_idx}", tags=["Analysis"])
async def explain_instance(
    payload: RequestPayload,
    instance_idx: int,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
    method: str = "shap",
):
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = payload.ref_dataset
    test_data_url = payload.cur_dataset
    model_url = payload.model
    target_column = payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.explain_instance, instance_idx)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="explain-instance",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.explain_instance, instance_idx)


@app.post("/regression/analysis/what-if", tags=["Analysis"])
async def perform_what_if(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = url_payload.ref_dataset
    test_data_url = url_payload.cur_dataset
    model_url = url_payload.model
    target_column = url_payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.perform_what_if, payload.get("features"))
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="what-if",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.perform_what_if, payload.get("features"))


@app.post("/regression/analysis/feature-dependence/{feature_name}", tags=["Analysis"])
async def get_feature_dependence(
    payload: RequestPayload, feature_name: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = payload.ref_dataset
    test_data_url = payload.cur_dataset
    model_url = payload.model
    target_column = payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.get_feature_dependence, feature_name)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="feature-dependence",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.get_feature_dependence, feature_name)


@app.post("/regression/analysis/instances", tags=["Analysis"])
async def list_instances(
    payload: RequestPayload,
    sort_by: str = "prediction",
    limit: int = 100,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = user.get("user_id")
    logger.info(f"User {user_id} started regression instance analysis")
    model_service = ModelService()
    train_data_url = payload.ref_dataset
    test_data_url = payload.cur_dataset
    model_url = payload.model
    target_column = payload.target_column

    logger.info(
        f"Parameters received | train_data: {train_data_url}, test_data: {test_data_url}, "
        f"model: {model_url}, target_column: {target_column}, sort_by: {sort_by}, limit: {limit}"
    )
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column,
        )
        logger.info("Model and datasets loaded successfully.")

        # data_preview_response = handle_request(model_service.list_instances, sort_by, limit)
        # data_preview = json.loads(data_preview_response.body)
        # # Generate analysis_id
        # analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        # existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()
        # logger.info(f"Generated analysis_id: {analysis_id}")

        # if not existing:
        #     new_entry = AnalysisResult(
        #         analysis_id=analysis_id,
        #         user_id=user_id,
        #         analysis_type="regression",
        #         analysis_tab="instances",
        #         project_id="N/A",
        #         json_result=json.dumps(data_preview),
        #     )
        #     db.add(new_entry)
        #     db.commit()
        #     logger.info(f"Analysis result saved to database (analysis_id={analysis_id}).")
        # else:
        #     logger.info(f"Analysis result already exists in database (analysis_id={analysis_id}). Skipping save.")
        return handle_request(model_service.list_instances, sort_by, limit)
    except Exception as e:
        logger.error(f"Error during regression instance analysis for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/regression/analysis/dataset-comparison", tags=["Analysis"])
async def get_dataset_comparison(
    payload: RequestPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = payload.ref_dataset
    test_data_url = payload.cur_dataset
    model_url = payload.model
    target_column = payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.get_dataset_comparison)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="dataset-comparison",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.get_dataset_comparison)


# --- New enterprise feature endpoints ---
@app.post("/regression/api/features", tags=["Features"])
async def get_features_metadata(
    payload: RequestPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = payload.ref_dataset
    test_data_url = payload.cur_dataset
    model_url = payload.model
    target_column = payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.get_feature_metadata)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="features",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.get_feature_metadata)


@app.post("/regression/api/correlation", tags=["Features"])
async def post_correlation(
    payload: CorrelationPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    # Extract URL payload data from the main payload
    model_service = ModelService()
    selected: List[str] = payload.features or []
    user_id = user.get("user_id")
    train_data_url = payload.ref_dataset
    test_data_url = payload.cur_dataset
    model_url = payload.model
    target_column = payload.target_column

    # Validate required fields
    if not all([train_data_url, model_url, target_column]):
        raise HTTPException(status_code=400, detail="Missing required fields: ref_dataset, model, target_column")

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.compute_correlation, selected)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="correlation",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.compute_correlation, selected)


@app.post("/regression/api/feature-importance", tags=["Features"])
async def post_feature_importance(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    method = payload.get("method", "shap")
    sort_by = payload.get("sort_by", "importance")
    top_n = int(payload.get("top_n", 1000))
    visualization = payload.get("visualization", "bar")
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = url_payload.ref_dataset
    test_data_url = url_payload.cur_dataset
    model_url = url_payload.model
    target_column = url_payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(
        model_service.compute_feature_importance_advanced, method, sort_by, top_n, visualization
    )
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="feature-importance",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.compute_feature_importance_advanced, method, sort_by, top_n, visualization)


@app.post("/regression/analysis/feature-interactions", tags=["Analysis"])
async def get_feature_interactions(
    url_payload: RequestPayload,
    feature1: str,
    feature2: str,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = url_payload.ref_dataset
    test_data_url = url_payload.cur_dataset
    model_url = url_payload.model
    target_column = url_payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.get_feature_interactions, feature1, feature2)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="feature-interactions",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.get_feature_interactions, feature1, feature2)


@app.post("/regression/analysis/decision-tree", tags=["Analysis"])
async def get_decision_tree(
    url_payload: RequestPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = url_payload.ref_dataset
    test_data_url = url_payload.cur_dataset
    model_url = url_payload.model
    target_column = url_payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.get_decision_tree)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="decision-tree",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.get_decision_tree)


# --- Individual Prediction API ---
@app.post("/regression/api/individual-prediction", tags=["Prediction"])
async def post_individual_prediction(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    instance_idx = int(payload.get("instance_idx", 0))
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = url_payload.ref_dataset
    test_data_url = url_payload.cur_dataset
    model_url = url_payload.model
    target_column = url_payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.individual_prediction, instance_idx)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="individual-prediction",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.individual_prediction, instance_idx)


# --- Regression Analysis Endpoints ---
@app.post("/regression/api/partial-dependence", tags=["Dependence"])
async def post_partial_dependence(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    feature = payload.get("feature")
    if not feature:
        raise HTTPException(status_code=400, detail="Missing 'feature'")
    num_points = int(payload.get("num_points", 20))
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = url_payload.ref_dataset
    test_data_url = url_payload.cur_dataset
    model_url = url_payload.model
    target_column = url_payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.partial_dependence, feature, num_points)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="partial-dependence",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.partial_dependence, feature, num_points)


@app.post("/regression/api/shap-dependence", tags=["Dependence"])
async def post_shap_dependence(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    feature = payload.get("feature")
    if not feature:
        raise HTTPException(status_code=400, detail="Missing 'feature'")
    color_by = payload.get("color_by")
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = url_payload.ref_dataset
    test_data_url = url_payload.cur_dataset
    model_url = url_payload.model
    target_column = url_payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.shap_dependence, feature, color_by)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="shap-dependence",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.shap_dependence, feature, color_by)


@app.post("/regression/api/ice-plot", tags=["Dependence"])
async def post_ice_plot(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    feature = payload.get("feature")
    if not feature:
        raise HTTPException(status_code=400, detail="Missing 'feature'")
    num_points = int(payload.get("num_points", 20))
    num_instances = int(payload.get("num_instances", 20))
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = url_payload.ref_dataset
    test_data_url = url_payload.cur_dataset
    model_url = url_payload.model
    target_column = url_payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.ice_plot, feature, num_points, num_instances)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="ice-plot",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.ice_plot, feature, num_points, num_instances)


# --- Section 5 APIs ---
@app.post("/regression/api/interaction-network", tags=["Interactions"])
async def post_interaction_network(
    url_payload: RequestPayload,
    payload: Dict = Body({}),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    top_k = int(payload.get("top_k", 30))
    sample_rows = int(payload.get("sample_rows", 200))
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = url_payload.ref_dataset
    test_data_url = url_payload.cur_dataset
    model_url = url_payload.model
    target_column = url_payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.interaction_network, top_k, sample_rows)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="interaction-network",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.interaction_network, top_k, sample_rows)


@app.post("/regression/api/pairwise-analysis", tags=["Interactions"])
async def post_pairwise_analysis(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    f1 = payload.get("feature1")
    f2 = payload.get("feature2")
    if not f1 or not f2:
        raise HTTPException(status_code=400, detail="Missing 'feature1' or 'feature2'")
    color_by = payload.get("color_by")
    sample_size = int(payload.get("sample_size", 1000))
    user_id = user.get("user_id")
    model_service = ModelService()
    train_data_url = url_payload.ref_dataset
    test_data_url = url_payload.cur_dataset
    model_url = url_payload.model
    target_column = url_payload.target_column

    model_service.load_model_and_datasets(
        model_path=model_url, train_data_path=train_data_url, test_data_path=test_data_url, target_column=target_column
    )

    data_preview_response = handle_request(model_service.pairwise_analysis, f1, f2, color_by, sample_size)
    data_preview = json.loads(data_preview_response.body)
    # Generate analysis_id
    analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
    existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

    if not existing:
        new_entry = AnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analysis_type="regression",
            analysis_tab="pairwise-analysis",
            project_id="N/A",
            json_result=json.dumps(data_preview),
        )
        db.add(new_entry)
        db.commit()
    return handle_request(model_service.pairwise_analysis, f1, f2, color_by, sample_size)


# --- AI Explanation Endpoint ---
@app.post("/regression/analysis/explain-with-ai", tags=["AI Analysis"])
async def explain_with_ai(payload: Dict = Body(...), user: str = Depends(get_current_user)):
    """
    Generate an AI-powered explanation of the current analysis results.

    Expected payload:
    {
        "analysis_type": "overview|feature_importance|classification_stats|...",
        "analysis_data": {...}  # The data to be explained
    }
    """
    try:
        analysis_type = payload.get("analysis_type")
        analysis_data = payload.get("analysis_data", {})
        ai_explanation_service = AIExplanationService()
        if not analysis_type:
            raise HTTPException(status_code=400, detail="Missing 'analysis_type' in payload")

        # Generate AI explanation
        explanation = ai_explanation_service.generate_explanation(analysis_data, analysis_type)

        return JSONResponse(status_code=200, content=explanation)

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating AI explanation: {str(e)}")


app.include_router(regression.routers)
