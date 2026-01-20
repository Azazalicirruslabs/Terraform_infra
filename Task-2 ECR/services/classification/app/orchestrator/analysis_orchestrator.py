# orchestrator/analysis_orchestrator.py
import os

from dotenv import load_dotenv

from services.classification.app.logic.evidentlyrunner import prepare_evidently_report
from services.classification.app.logic.explainer_runner import ExplainerRunner
from services.classification.app.logic.explainerengine import ExplainerEngine
from services.classification.app.logic.llmanalyzer import LLMAnalyzer
from services.classification.app.logic.s3handler import S3Handler

load_dotenv()


BASE_URL = os.getenv("FILES_API_BASE_URL", "http://localhost")
API_URL = f"{BASE_URL}/Classification"


class AnalysisOrchestrator:
    def __init__(self):
        self.s3 = S3Handler(api_url=API_URL)
        self.llm = LLMAnalyzer()

    def load_data_and_model(self, token: str = None):
        model, train_df, test_df = self.s3.load_from_s3(access_token=token)
        return model, train_df, test_df

    def get_evidently_report(self, model, train_df, test_df, target_col: str):
        html, json_data = prepare_evidently_report(
            train_df=train_df, test_df=test_df, model=model, target_col=target_col, task_type="binary"
        )
        return html, json_data

    def get_llm_analysis(self, report_json, train_df, test_df, target_col: str):
        summary = self.llm.get_classification_summary(report_json)
        analysis_md = self.llm.analyze_classification(
            report_dict=report_json, train_df=train_df, test_df=test_df, target_name=target_col, summary_text=summary
        )
        return analysis_md

    def get_explainer_dashboard(self, model, train_df, test_df, target_col: str):
        # 1) Build the engine
        engine = ExplainerEngine(model=model, train_df=train_df, test_df=test_df)
        engine.setup_explainer(target_column=target_col, max_explainer_rows=1000)
        # 2) Launch Dash app in a thread
        runner = ExplainerRunner(engine)
        runner.start_dashboard(host="0.0.0.0", port=8050)
        # 3) Return URL for frontend embedding
        return "http://localhost:8050"
