import json
import os
from io import BytesIO

import boto3
import joblib
import pandas as pd
import requests

# import google.generativeai as genai
from dotenv import load_dotenv

# --- NEW IMPORTS based on your research ---
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, RegressionPreset

# Load environment variables from .env file
load_dotenv()


class RegressionService:
    def __init__(self):
        # NEW: Store the base URL for the file download API
        self.files_api_base_url = os.getenv("FILES_API_BASE_URL")

    # MODIFIED: This now loads data from an in-memory byte stream
    def _load_data_from_bytes(self, content: bytes, file_type: str) -> pd.DataFrame:
        """Loads data from a byte stream into a pandas DataFrame."""
        try:
            if file_type == "csv":
                return pd.read_csv(BytesIO(content))
            elif file_type == "parquet":
                return pd.read_parquet(BytesIO(content), engine="pyarrow")
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise ValueError(f"Failed to parse content from bytes: {e}")

    # MODIFIED: This now loads a model from an in-memory byte stream
    def _load_model_from_bytes(self, content: bytes):
        """Loads a pickled model from a byte stream."""
        try:
            return joblib.load(BytesIO(content))
        except Exception as e:
            raise ValueError(f"Failed to load model content from bytes: {e}")

    # REMOVED: The old _load_data(file_path) and _load_model(file_path) are no longer needed.

    # NEW: This method orchestrates fetching and loading all assets from your API
    def _fetch_and_load_assets(self, analysis_type: str, auth_token: str) -> tuple:
        """
        Fetches file URLs from the download API and loads them into memory.
        Returns (reference_df, current_df, model).
        """
        # api_url = f"{self.files_api_base_url}/files_download/{analysis_type}"
        base_url = os.getenv("FILES_API_BASE_URL")
        if not base_url:
            raise ValueError("FILES_API_BASE_URL environment variable is not set.")
        api_url = f"{base_url}/{analysis_type}"
        headers = {"Authorization": f"Bearer {auth_token}"}

        print(f"Fetching file list from: {api_url}")
        response = requests.get(api_url, headers=headers)

        if response.status_code != 200:
            raise ConnectionError(
                f"API request to fetch files failed with status {response.status_code}: {response.text}"
            )

        try:
            response.json()
        except ValueError as e:
            raise ValueError(f"Invalid JSON response from {api_url}: {response.text}")

        files_metadata = response.json().get("files", [])
        if not files_metadata:
            raise FileNotFoundError("No files returned from the download API.")

        ref_df, cur_df, model = None, None, None

        for file_info in files_metadata:
            file_url = file_info.get("url")
            file_name = file_info.get("file_name", "").lower()

            if not file_url:
                print(f"⚠️ Skipping file '{file_name}' due to missing URL.")
                continue

            print(f"Downloading: {file_name}")
            file_response = requests.get(file_url)
            if file_response.status_code != 200:
                print(f"⚠️ Failed to download {file_name} from pre-signed URL. Status: {file_response.status_code}")
                continue

            file_content = file_response.content

            # Logic to identify file type based on name
            if file_name.endswith((".csv", ".parquet")):
                file_type = "parquet" if file_name.endswith(".parquet") else "csv"
                if "ref" in file_name:
                    ref_df = self._load_data_from_bytes(file_content, file_type)
                    print(f"✅ Loaded reference data: {ref_df.shape}")
                elif "cur" in file_name:
                    cur_df = self._load_data_from_bytes(file_content, file_type)
                    print(f"✅ Loaded current data: {cur_df.shape}")
            elif file_name.endswith((".pkl", ".joblib")):
                model = self._load_model_from_bytes(file_content)
                print(f"✅ Loaded model: {type(model).__name__}")

        if ref_df is None or cur_df is None or model is None:
            missing = []
            if ref_df is None:
                missing.append("reference dataset")
            if cur_df is None:
                missing.append("current dataset")
            if model is None:
                missing.append("model file")
            raise FileNotFoundError(
                f"Could not load all required assets. Missing: {', '.join(missing)}. Check filenames in S3 (must contain 'ref', 'cur', and be .csv/.pkl/.joblib)."
            )

        return ref_df, cur_df, model

    def _prepare_features(self, df: pd.DataFrame, target_column: str, model) -> list:
        """Prepare feature columns for model prediction."""
        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        else:
            features = [col for col in df.columns if col != target_column]

        missing_features = set(features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Data is missing required features the model was trained on: {missing_features}")
        return features

    # MODIFIED: The main public method now takes analysis_type and auth_token instead of file paths.
    def generate_performance_report(self, analysis_type: str, auth_token: str, target_column: str) -> tuple:
        """
        Generates an Evidently AI report by fetching assets from the cloud.
        Returns the Snapshot object and the report dictionary for LLM analysis.
        """
        # Step 1: Fetch and load all required files from S3 via your API
        ref_df, cur_df, model = self._fetch_and_load_assets(analysis_type, auth_token)

        # The rest of the logic remains the same as it works on the loaded objects
        if target_column not in ref_df.columns or target_column not in cur_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in both datasets.")

        features = self._prepare_features(ref_df, target_column, model)

        prediction_col_name = "prediction"

        ref_df[prediction_col_name] = model.predict(ref_df[features])
        cur_df[prediction_col_name] = model.predict(cur_df[features])

        data_definition = DataDefinition(regression=[Regression(target=target_column, prediction="prediction")])
        reference_dataset = Dataset.from_pandas(ref_df, data_definition=data_definition)
        current_dataset = Dataset.from_pandas(cur_df, data_definition=data_definition)

        report = Report(metrics=[RegressionPreset(), DataDriftPreset()])
        snapshot = report.run(reference_data=reference_dataset, current_data=current_dataset)

        report_json = snapshot.json()
        report_dict = json.loads(report_json) if report_json else {}

        return snapshot, report_dict


class LLMAnalyzer:
    """
    Analyzer for LLM-based explanations using AWS Bedrock runtime client.

    Initializes the LLMAnalyzer and the Bedrock runtime client.
    """

    def __init__(self):
        self.bedrock_runtime = None
        self._initialize_bedrock()
        # You can choose which Claude model to use. Sonnet is a great balance.
        # Other options: "anthropic.claude-3-haiku-v1:0", "anthropic.claude-3-opus-v1:0"
        self.claude_model_id = "anthropic.claude-3-sonnet-v1:0"

    def _initialize_bedrock(self):
        """
        Initializes the AWS Bedrock runtime client and validates the connection
        by listing foundation models with a separate management client.
        """
        try:
            # 1. Use the MANAGEMENT client ('bedrock') to check for models and validate credentials/connection.
            #    This is a temporary client just for this check.
            management_client = boto3.client(
                service_name="bedrock",
                region_name=os.getenv("REGION_LLM", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_LLM"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_LLM"),
            )
            # This call will now succeed if your credentials and region are correct.
            management_client.list_foundation_models(byProvider="anthropic")
            print("✅ AWS credentials and connection validated successfully.")

            # 2. Initialize the class's main client as the INFERENCE client ('bedrock-runtime').
            #    This is the client we will actually use to call Claude.
            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("REGION_LLM", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_LLM"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_LLM"),
            )
            print("✅ Bedrock runtime client for inference is ready.")

        except Exception as e:
            print(f"❌ Bedrock initialization failed. Please check AWS credentials and region in .env file.")
            print(f"Error: {e}")
            # This will catch credential errors, region errors, or permission errors.
            raise ConnectionError("Failed to initialize or validate AWS Bedrock client.") from e

    def _invoke_claude(self, prompt):
        """Invoke Claude via AWS Bedrock."""
        if not self.bedrock_runtime:
            return "Claude invocation failed: Bedrock not initialized"

        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 130000,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            }

            response = self.bedrock_runtime.invoke_model(
                modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0", body=json.dumps(request_body)
            )

            response_body = json.loads(response["body"].read().decode("utf-8"))
            return response_body["content"][0]["text"]
        except Exception as e:
            return f"Claude invocation failed: {e}"

    def get_regression_summary(self, report_dict: dict) -> str:
        """Extract current dataset metrics from Evidently report - FIXED VERSION"""
        summary_lines = ["**Current Dataset Performance Analysis:**"]

        current_metrics = {}
        drift_info = {}

        # Extract metrics from the report - using the CORRECT structure
        for metric in report_dict.get("metrics", []):
            metric_id = metric.get("metric_id", "")  # This is the correct key
            value = metric.get("value", {})  # This is the correct key

            # Extract regression performance metrics
            if isinstance(value, dict) and "mean" in value:
                mean_val = value["mean"]
                std_val = value.get("std", 0)

                if "MeanError" in metric_id:
                    current_metrics["Mean Error (ME)"] = {
                        "value": mean_val,
                        "std": std_val,
                        "interpretation": "Average prediction error (negative = underestimation, positive = overestimation)",
                    }
                elif "MAPE" in metric_id:
                    current_metrics["Mean Absolute Percentage Error (MAPE)"] = {
                        "value": mean_val,
                        "std": std_val,
                        "interpretation": "Average percentage error (lower is better)",
                    }
                elif "MAE" in metric_id:
                    current_metrics["Mean Absolute Error (MAE)"] = {
                        "value": mean_val,
                        "std": std_val,
                        "interpretation": "Average absolute prediction error (lower is better)",
                    }
                elif "RMSE" in metric_id:
                    current_metrics["Root Mean Squared Error (RMSE)"] = {
                        "value": mean_val,
                        "std": std_val,
                        "interpretation": "Root mean squared error, penalizes large errors more (lower is better)",
                    }

            # Handle single value metrics
            elif isinstance(value, (int, float)):
                if "R2Score" in metric_id:
                    current_metrics["R² Score"] = {
                        "value": value,
                        "std": 0,
                        "interpretation": "Coefficient of determination (1.0 = perfect fit, 0.0 = no predictive power)",
                    }
                elif "RMSE" in metric_id:
                    current_metrics["Root Mean Squared Error (RMSE)"] = {
                        "value": value,
                        "std": 0,
                        "interpretation": "Root mean squared error, penalizes large errors more (lower is better)",
                    }

            # Extract drift information
            if "DriftedColumnsCount" in metric_id and isinstance(value, dict):
                drift_info["total_drifted"] = value.get("count", 0)
                drift_info["drift_share"] = value.get("share", 0.0)

            # Extract individual column drift
            elif "ValueDrift" in metric_id:
                column_name = metric_id.split("column=")[1].split(")")[0] if "column=" in metric_id else "Unknown"
                if "drift_columns" not in drift_info:
                    drift_info["drift_columns"] = {}
                drift_info["drift_columns"][column_name] = value

        # Format the summary
        summary_lines.append("\n**📊 Performance Metrics:**")

        for metric_name, metric_data in current_metrics.items():
            value = metric_data["value"]
            std = metric_data["std"]
            interpretation = metric_data["interpretation"]

            if std > 0:
                summary_lines.append(f"- **{metric_name}**: {value:.4f} ± {std:.4f}")
            else:
                summary_lines.append(f"- **{metric_name}**: {value:.4f}")
            summary_lines.append(f"  ℹ️ {interpretation}")

        # Add drift information
        if drift_info:
            summary_lines.append("\n**🔄 Data Drift Analysis:**")

            if "total_drifted" in drift_info:
                total_columns = len(drift_info.get("drift_columns", {}))
                drifted_count = int(drift_info["total_drifted"])
                drift_share = drift_info["drift_share"]

                summary_lines.append(
                    f"- **Overall Drift**: {drifted_count} out of {total_columns} columns ({drift_share:.0%})"
                )

                if "drift_columns" in drift_info:
                    summary_lines.append("- **Column-wise Drift Scores**:")
                    for col_name, drift_score in drift_info["drift_columns"].items():
                        if isinstance(drift_score, (int, float)):
                            drift_level = (
                                "🔴 High" if drift_score > 0.5 else "🟡 Medium" if drift_score > 0.1 else "🟢 Low"
                            )
                            summary_lines.append(f"  - {col_name}: {drift_score:.4f} ({drift_level})")

        return "\n".join(summary_lines) if len(summary_lines) > 1 else "No regression metrics found in report."

    def analyze_with_claude(self, report_dict: dict) -> str:
        """
        Analyzes the current dataset performance using Claude via AWS Bedrock.
        This is the main public method to call from your API.
        """
        summary_text = self.get_regression_summary(report_dict)
        if "No regression metrics found" in summary_text:
            return f"❌ **Analysis Failed**: {summary_text}"

        # Updated prompt to match the working Gemini version's approach
        prompt = f"""
        You are an expert data scientist analyzing machine learning model performance on a current dataset.

        Based on the following performance metrics and data drift analysis, provide a comprehensive but easy-to-understand report for stakeholders that includes:

        1. **📈 Performance Assessment**:
           - Overall model performance evaluation (Excellent/Good/Fair/Poor)
           - Key strengths and weaknesses based on the metrics
           - What the metrics tell us about prediction accuracy

        2. **🎯 Key Insights**:
           - Which metrics indicate good/concerning performance
           - What the R² score tells us about model fit
           - How prediction errors are distributed (based on MAE vs RMSE)

        3. **⚠️ Areas of Concern**:
           - Any metrics that suggest potential issues
           - Data drift implications for model reliability
           - Recommendations for monitoring

        4. **📋 Recommendations**:
           - Should the model continue to be used as-is?
           - What actions should be taken (if any)?
           - When should the model be retrained?

        Keep the analysis:
        - Clear and jargon-free for business stakeholders
        - Actionable with specific recommendations
        - Focused on practical implications
        - Use emojis and formatting for readability

        **Current Dataset Analysis:**
        ---
        {summary_text}
        ---
        """

        # Use Claude instead of Gemini
        try:
            return self._invoke_claude(prompt)
        except Exception as e:
            return f"""
## 🚨 AI Analysis Error

Failed to generate analysis using Claude. Please check your AWS credentials and network connection.

**Error details:** {str(e)}

---

**Raw Metrics Summary:**
{summary_text}
            """
