# api_backend.py
import os
import socket
import tempfile
import threading
import time
import traceback

import psutil
from explainerdashboard import ExplainerDashboard, RegressionExplainer
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.regression.app.routers.logic import LLMAnalyzer, RegressionService
from services.regression.app.schemas.regression_schema import ExplainerDashboardResponse, RegressionAnalysisResponse
from shared.auth import get_current_user

routers = APIRouter(prefix="/regression", tags=["Old regression"])

# Constants

DASHBOARD_INSTANCES = {}


# Helper functions (unchanged)
def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_dashboard_thread(explainer, host, port):
    try:
        db = ExplainerDashboard(explainer, title="Interactive Model Explainer", shap_interaction=False)
        db.run(host=host, port=port, threaded=False)
        print(f"Dashboard on port {port} has shut down.")
    except Exception as e:
        print(f"Error running dashboard on port {port}: {e}")


class RequestRegression(BaseModel):
    target_column: str
    analysis_type: str


@routers.post("/analyze/regression", response_model=RegressionAnalysisResponse)
async def analyze_regression(
    request: RequestRegression,
    current_user: dict = Depends(get_current_user),
):
    """
    Analyzes regression model performance by fetching data from the cloud based on analysis_type.
    """

    try:
        # Extract the token
        access_token = current_user.get("token")
        if not access_token:
            raise HTTPException(status_code=401, detail="Unauthorized: No access token provided")

        # Use your existing RegressionService which handles the data fetching
        regression_service = RegressionService()
        llm_analyzer = LLMAnalyzer()

        # Your RegressionService already handles fetching data from the cloud
        snapshot, report_dict = regression_service.generate_performance_report(
            analysis_type=request.analysis_type, auth_token=access_token, target_column=request.target_column
        )

        # Generate HTML report
        with tempfile.TemporaryDirectory() as temp_dir:
            report_file = os.path.join(temp_dir, "evidently_report.html")
            snapshot.save_html(report_file)
            with open(report_file, "r", encoding="utf-8") as f:
                html_report = f.read()

        # Get LLM insights using Claude
        llm_insights = llm_analyzer.analyze_with_claude(report_dict)

        return JSONResponse(
            content={"status": "success", "evidently_report_html": html_report, "llm_insights": llm_insights}
        )

    except HTTPException:
        raise
    except (ValueError, FileNotFoundError, ConnectionError) as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")


class RequestExplainer(BaseModel):
    target_column: str
    analysis_type: str


@routers.post("/explain/model", response_model=ExplainerDashboardResponse)
async def explain_model(
    request: RequestExplainer,
    current_user: dict = Depends(get_current_user),
):
    """
    Launches an ExplainerDashboard by fetching the model and reference dataset from the cloud.
    """

    access_token = current_user.get("token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Unauthorized: No access token provided")
    dashboard_key = request.analysis_type

    if dashboard_key in DASHBOARD_INSTANCES:
        instance = DASHBOARD_INSTANCES[dashboard_key]
        process = instance.get("process")
        if process and process.is_running():
            print(f"Dashboard for '{dashboard_key}' already running at {instance['url']}")
            return JSONResponse(content={"status": "already_running", "explainer_url": instance["url"]})
        else:
            DASHBOARD_INSTANCES.pop(dashboard_key, None)

    try:
        # Use your existing RegressionService to fetch the data and model
        service = RegressionService()
        # Your _fetch_and_load_assets method returns (ref_df, cur_df, model)
        ref_df, cur_df, model = service._fetch_and_load_assets(request.analysis_type, access_token)

        # Use the reference dataset for the explainer
        target_column = request.target_column
        X = ref_df.drop(columns=[target_column])
        y = ref_df[target_column]

        explainer = RegressionExplainer(model, X, y, X_background=X.head(100))

        host = os.getenv("HOST")
        port = _find_free_port()

        dashboard_thread = threading.Thread(target=_run_dashboard_thread, args=(explainer, host, port), daemon=True)
        dashboard_thread.start()

        time.sleep(5)

        url = f"http://{host}:{port}"
        DASHBOARD_INSTANCES[dashboard_key] = {
            "process": psutil.Process(os.getpid()),
            "url": url,
            "thread": dashboard_thread,
        }

        print(f"Launched new dashboard for '{dashboard_key}' at {url}")
        return JSONResponse(
            content={
                "status": "success",
                "explainer_url": url,
                "data_info": {
                    "reference_shape": ref_df.shape,
                    "current_shape": cur_df.shape,
                    "model_type": type(model).__name__,
                    "target_column": target_column,
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to launch explainer dashboard: {str(e)}")
