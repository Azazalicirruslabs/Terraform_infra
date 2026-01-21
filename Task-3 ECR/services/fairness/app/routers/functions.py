"""
functions.py
Core utility functions for fairness analysis backend, mitigation, metrics, and ONNX support.
"""

import io
import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chardet
import joblib
import numpy as np
import pandas as pd

# Removed unused imports to clean up the code
# Removed sklearn imports as they are now in main.py
import requests  # Import requests for making HTTP calls to external API
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

try:
    import onnxruntime as rt

    # from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
except ImportError:
    rt = None
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# loading variables from .env file
load_dotenv()


from services.fairness.app.core.config import DIFFERENCE_THRESHOLD, POINTS_PENALTY_PER_THRESHOLD, THEIL_INDEX_MULTIPLIER


@dataclass
class FairnessMetrics:
    """Container for fairness metrics results for a single group."""

    demographic_parity_difference: float  # Renamed for clarity
    equalized_odds_difference: float  # Corrected calculation
    equal_opportunity_difference: float  # Corrected calculation
    disparate_impact_ratio: float  # Renamed for clarity
    average_odds_difference: float
    theil_index: float

    def to_dict(self):
        return {
            "demographic_parity_difference": self.demographic_parity_difference,
            "equalized_odds_difference": self.equalized_odds_difference,
            "equal_opportunity_difference": self.equal_opportunity_difference,
            "disparate_impact_ratio": self.disparate_impact_ratio,
            "average_odds_difference": self.average_odds_difference,
            "theil_index": self.theil_index,
        }


@dataclass
class OverallFairnessScore:
    """Container for overall fairness score calculation."""

    score: float
    grade: str
    color: str
    description: str
    improvement_potential: float

    def to_dict(self):
        return {
            "score": self.score,
            "grade": self.grade,
            "color": self.color,
            "description": self.description,
            "improvement_potential": self.improvement_potential,
        }


class ONNXModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, session: rt.InferenceSession, classes: np.ndarray = np.array([0, 1])):
        self.session = session
        self.inputs = session.get_inputs()
        self.outputs = session.get_outputs()
        self.classes_ = classes
        self._estimator_type = "classifier"

        if len(self.inputs) != 1:
            raise ValueError(f"Expected 1 input, got {len(self.inputs)}.")
        self.input_name = self.inputs[0].name

        # Detect outputs
        self.proba_name = None
        self.label_name = None
        for out in self.outputs:
            if "proba" in out.name.lower():
                self.proba_name = out.name
            else:
                self.label_name = out.name

        if self.label_name is None:
            raise ValueError("Could not find label output in ONNX model.")

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X_onnx = X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else np.array(X, dtype=np.float32)

        preds = self.session.run([self.label_name], {self.input_name: X_onnx})[0]
        return preds

    def predict_proba(self, X):
        if not self.proba_name:
            raise NotImplementedError("No probability output found in ONNX model.")

        X_onnx = X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else np.array(X, dtype=np.float32)

        raw = self.session.run([self.proba_name], {self.input_name: X_onnx})[0]

        if isinstance(raw, np.ndarray):
            return raw

        # Otherwise convert list-of-dicts → array in class order
        prob_array = np.array([[d.get(c, d.get(str(c))) for c in self.classes_] for d in raw], dtype=float)
        return prob_array

    def score(self, X, y):
        """
        Support scikit-learn’s score API so Fairlearn’s
        check_is_estimator passes.
        """
        preds = self.predict(X)
        return accuracy_score(y, preds)


class LLMBasedAnalysis:
    """
    A class to handle LLM-based fairness analysis by interacting with the Claude API via AWS Bedrock.
    """

    def __init__(self, bedrock_runtime=None):
        if bedrock_runtime is not None:
            self.bedrock_runtime = bedrock_runtime
        else:
            self.bedrock_runtime = None
            self._initialize_bedrock()

    def _initialize_bedrock(self):
        """Initialize AWS Bedrock client."""
        try:
            import boto3

            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("REGION_LLM", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_LLM"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_LLM"),
            )
        except Exception as e:
            print(f"Bedrock initialization failed: {e}")

    def _invoke_claude(self, prompt: str) -> str:
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
                modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                body=json.dumps(request_body),
            )

            response_body = json.loads(response["body"].read().decode("utf-8"))
            return response_body["content"][0]["text"]
        except Exception as e:
            return f"Claude invocation failed: {e}"

    def analyze_fairness_disparity(self, summary: str, mitigation_strategy: str = None) -> str:
        """
        Generates a detailed fairness analysis based on provided group-wise metrics using an LLM.
        """
        mitigation_info = f"\nMitigation strategy applied: {mitigation_strategy}." if mitigation_strategy else ""
        prompt = f"""
You are an expert in ethical AI, fairness auditing, and responsible machine learning.
Below is a summary of model performance across different demographic groups, generated using fairness metrics like accuracy and selection rate:{mitigation_info}
{summary}
Please analyze and provide the following:
1. **Fairness Disparities**
   - Are there notable disparities in accuracy, selection rate, or error rates across groups?
   - Quantify the disparities clearly.
2. **Potential Harms**
   - What kind of real-world or reputational risks do these disparities pose?
   - Which groups are likely to be disadvantaged?
3. **Compliance & Ethical Implications**
   - Are the disparities likely to violate fairness goals like Demographic Parity or Equal Opportunity?
   - Could this raise regulatory or legal concerns?
4. **Suggested Mitigations**
   - Recommend bias mitigation strategies (e.g., model retraining, fairness constraints)
   - Mention if post-processing or reweighting might help
   - Based on the observed results and the mitigation strategy applied, specify which class of mitigation (pre-processing, in-processing, post-processing) would be most appropriate for further improvement, and why.
5. **Communication to Stakeholders**
   - How would you explain this issue to a business or legal team?
Use markdown formatting with clear section headings and bullet points. Avoid technical jargon where possible.
"""
        return self._invoke_claude(prompt)


MITIGATION_STRATEGIES = [
    {"label": "Exponentiated Gradient (In-processing)", "value": "exponentiated_gradient"},
    {"label": "Grid Search (In-processing)", "value": "grid_search"},
    {"label": "Reweighing (Pre-processing)", "value": "reweighing"},
    {"label": "Postprocessing (Threshold Optimization)", "value": "postprocessing"},
]


class FairnessAnalyzer:
    def __init__(self):
        self.llm_fairness_analyzer = LLMBasedAnalysis()

    def generate_plot_descriptions(self, title: str, data_dict: Dict[str, Any], mitigation_strategy: str = None) -> str:
        summary = ""
        for plot_title, metrics_data in data_dict.items():
            summary += f"--- {plot_title} ---\n"
            for group_name, metric_values in metrics_data.items():
                summary += f"Group: {group_name}\n"
                if isinstance(metric_values, dict):
                    for metric, values in metric_values.items():
                        summary += f"  {metric}: {values}\n"
                else:
                    summary += f"  Value: {metric_values}\n"
            summary += "\n"
        return self.llm_fairness_analyzer.analyze_fairness_disparity(summary, mitigation_strategy)

    def load_flexible_csv(self, file_content: bytes) -> pd.DataFrame:
        """
        Loads CSV data, automatically detecting delimiter and encoding.
        """
        # Detect encoding
        result = chardet.detect(file_content)
        encoding = result["encoding"] if result["encoding"] else "utf-8"

        # Try common delimiters
        for delimiter in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(file_content), encoding=encoding, delimiter=delimiter)
                if len(df.columns) > 1:  # Assume success if more than one column is parsed
                    return df
            except Exception:
                continue

        # If no common delimiter works, try with default (comma) and let it fail if truly malformed
        try:
            df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
            if len(df.columns) == 1:  # If only one column, it might be a single-column CSV or wrong delimiter
                raise RuntimeError(
                    "Could not determine appropriate delimiter. Data might be malformed or single-column."
                )
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV: {e}")

    def get_s3_file_metadata(self, token, project_id: str):
        """
        Lists files and models from the external S3 API and returns their metadata (name, URL, folder).
        Separates files and models based on the folder field.
        """
        FILE_MODEL_DOWNLOAD_API = os.getenv("FILES_API_BASE_URL")
        if FILE_MODEL_DOWNLOAD_API is None:
            raise RuntimeError("Environment variable FILES_API_BASE_URL is required but not set.")

        EXTERNAL_S3_API_URL = f"{FILE_MODEL_DOWNLOAD_API}/Fairness/{project_id}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.get(EXTERNAL_S3_API_URL, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            json_data = response.json()
            all_items = json_data.get("files", [])

            # Separate files and models based on folder
            files = [item for item in all_items if item.get("folder") == "files"]
            models = [item for item in all_items if item.get("folder") == "models"]

            return {"files": files, "models": models}
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to external S3 API: {e}")
            return None
        except Exception as e:
            print(f"Error processing external S3 API response: {e}")
            return None

    def download_and_load_dataframe(self, file_url: str) -> pd.DataFrame:
        """
        Downloads a file from a given URL (provided by the external S3 API) and loads it into a pandas DataFrame.
        Assumes CSV format.
        """
        try:
            file_response = requests.get(file_url)
            file_response.raise_for_status()  # Raise an exception for bad status codes
            return self.load_flexible_csv(file_response.content)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download file from URL {file_url}: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load CSV from downloaded content: {e}")

    def download_and_load_model(self, model_url: str, model_format: str = "onnx") -> object:
        """
        Downloads a model from a given URL (provided by the external S3 API) and loads it.
        Supports ONNX, PKL, and JOBLIB formats.
        """
        try:
            model_response = requests.get(model_url)
            model_response.raise_for_status()  # Raise an exception for bad status codes
            contents = model_response.content

            if model_format == "onnx":
                if rt is None:
                    raise RuntimeError("ONNX runtime is not installed on the server. Please contact support.")
                try:
                    session = rt.InferenceSession(contents)
                    return ONNXModelWrapper(session)
                except Exception as e:
                    raise RuntimeError(f"Invalid ONNX file. Please ensure the model was exported correctly. Error: {e}")

            elif model_format in ["pkl", "joblib"]:
                try:
                    if model_format == "pkl":
                        model = pickle.loads(contents)
                    else:  # joblib
                        from io import BytesIO

                        model = joblib.load(BytesIO(contents))

                    if not hasattr(model, "predict"):
                        raise AttributeError("Model must have a .predict() method.")

                    return model
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load .pkl/.joblib file. This is often due to version mismatches. Please use the ONNX format for better reliability. Error: {e}"
                    )

            else:
                raise RuntimeError(f"Unsupported model format '{model_format}'.")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download model from URL {model_url}: {e}")

    def load_and_preprocess_data(
        self, df: pd.DataFrame, target_col: Optional[str] = None, sensitive_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Preprocesses the DataFrame: label-encodes categorical columns (excluding sensitive features),
        handles 'yes'/'no', and binarizes numerical target column if it's multi-class and specified.
        Sensitive features are kept in their original form for readable group names.
        """
        if df is None:
            return pd.DataFrame()
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.strip()  # Sanitize column names

        # Exclude sensitive features from encoding - keep them as original values
        sensitive_features = sensitive_features or []
        categorical_cols = [
            col
            for col in df_copy.select_dtypes(include=["object", "category"]).columns
            if col not in sensitive_features and col != target_col
        ]

        for col in categorical_cols:
            # Normalize strings
            df_copy[col] = df_copy[col].astype(str).str.strip().str.lower()
            # Binary map if values match known pairs
            binary_map = {"yes": 1, "no": 0, "approved": 1, "rejected": 0}
            unique_vals = set(df_copy[col].dropna().unique())
            if unique_vals.issubset(binary_map.keys()):
                df_copy[col] = df_copy[col].map(binary_map)
            else:
                # Label encode and preserve column name
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col])

        # Keep sensitive features in original form - just clean them
        for sensitive_col in sensitive_features:
            if sensitive_col in df_copy.columns and df_copy[sensitive_col].dtype == "object":
                df_copy[sensitive_col] = df_copy[sensitive_col].astype(str).str.strip()

        # Binarize numerical target column if needed
        if target_col and target_col in df_copy.columns:
            if pd.api.types.is_numeric_dtype(df_copy[target_col]):
                unique_vals = df_copy[target_col].unique()
                if len(unique_vals) > 2:
                    most_freq = df_copy[target_col].mode()[0]
                    df_copy[target_col] = df_copy[target_col].apply(lambda x: 0 if x == most_freq else 1)
                elif len(unique_vals) == 2 and not set(unique_vals) <= {0, 1}:
                    mapping = {min(unique_vals): 0, max(unique_vals): 1}
                    df_copy[target_col] = df_copy[target_col].map(mapping)
            # Handle target column if it's categorical
            elif df_copy[target_col].dtype == "object":
                df_copy[target_col] = df_copy[target_col].astype(str).str.strip().str.lower()
                binary_map = {"yes": 1, "no": 0, "approved": 1, "rejected": 0}
                unique_vals = set(df_copy[target_col].dropna().unique())
                if unique_vals.issubset(binary_map.keys()):
                    df_copy[target_col] = df_copy[target_col].map(binary_map)
                else:
                    le = LabelEncoder()
                    df_copy[target_col] = le.fit_transform(df_copy[target_col])

        return df_copy

    def prepare_training_and_test_data(
        self,
        df_reference_processed: pd.DataFrame,
        df_current_processed: Optional[pd.DataFrame],
        target_col: str,
        sensitive_feature: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Splits preprocessed datasets into training and testing sets.

        Args:
            df_reference_processed: Preprocessed reference dataset.
            df_current_processed: Optional preprocessed current dataset.
            target_col: The name of the target column.
            sensitive_feature: The name of the sensitive attribute.

        Returns:
            X_train, X_test, y_train, y_test, A_train, A_test
        """
        if df_current_processed is not None:
            print("--- Using provided Reference and Current datasets. ---")

            X_train = df_reference_processed.drop(columns=[target_col, sensitive_feature], errors="ignore")
            y_train = df_reference_processed[target_col]
            A_train = df_reference_processed[sensitive_feature]

            X_test = df_current_processed.drop(columns=[target_col, sensitive_feature], errors="ignore")
            y_test = df_current_processed[target_col]
            A_test = df_current_processed[sensitive_feature]

        else:
            print("--- No Current dataset provided. Splitting Reference data 70/30. ---")

            X_all = df_reference_processed.drop(columns=[target_col, sensitive_feature], errors="ignore")
            y_all = df_reference_processed[target_col]
            A_all = df_reference_processed[sensitive_feature]

            X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
                X_all, y_all, A_all, test_size=0.3, random_state=42, stratify=y_all
            )

        return X_train, X_test, y_train, y_test, A_train, A_test

    def get_mitigation_strategies(self):
        """
        Returns the available mitigation strategies for the frontend dropdown.
        """
        return MITIGATION_STRATEGIES

    def _safe_score(self, value: float, multiplier: float = 100, is_ratio: bool = False) -> float:
        """
        Safely calculate score from a metric value, handling NaN and infinity cases.

        Args:
            value: The metric value to score
            multiplier: Scaling factor for differences (default 100 for more reasonable scaling)
            is_ratio: Whether this is a ratio metric (like disparate impact)

        Returns:
            Score from 0-100 where 100 is perfectly fair
        """
        if np.isnan(value) or np.isinf(value):
            return 0  # Worst possible score for undefined metrics

        if is_ratio:
            # For ratios like disparate impact, closer to 1 is better
            return max(0, 100 - abs(1 - value) * 100)
        else:
            # For differences, closer to 0 is better
            # Use practical significance thresholds: 20 points per 0.1 difference
            return max(0, 100 - (abs(value) / DIFFERENCE_THRESHOLD) * POINTS_PENALTY_PER_THRESHOLD)

    def calculate_overall_fairness_score(self, metrics: Dict[str, float]) -> OverallFairnessScore:
        """
        Calculate an overall fairness score from 0-100 based on all fairness metrics.

        Args:
            metrics: Dictionary of fairness metrics with float values

        Score interpretation:
        - 90-100: Excellent (Bright Green)
        - 80-89: Good (Green)
        - 70-79: Fair (Yellow-Green)
        - 60-69: Poor (Yellow)
        - 50-59: Bad (Orange)
        - 0-49: Critical (Red)
        """
        if not metrics:
            return OverallFairnessScore(0, "N/A", "#808080", "No data available", 0)

        # Score each metric directly from the dictionary (0-100, where 100 is perfectly fair)
        dp_score = self._safe_score(metrics.get("demographic_parity_difference", 0.0))
        eo_score = self._safe_score(metrics.get("equalized_odds_difference", 0.0))
        eop_score = self._safe_score(metrics.get("equal_opportunity_difference", 0.0))
        di_score = self._safe_score(metrics.get("disparate_impact_ratio", 1.0), is_ratio=True)
        ao_score = self._safe_score(metrics.get("average_odds_difference", 0.0))
        theil_score = self._safe_score(metrics.get("theil_index", 0.0), multiplier=THEIL_INDEX_MULTIPLIER)

        # Weighted average of all metrics
        overall_score = (
            dp_score * 0.25  # Demographic Parity - 25%
            + eo_score * 0.20  # Equalized Odds - 20%
            + eop_score * 0.15  # Equal Opportunity - 15%
            + di_score * 0.20  # Disparate Impact - 20%
            + ao_score * 0.10  # Average Odds - 10%
            + theil_score * 0.10  # Theil Index - 10%
        )

        # Clamp between 0-100
        overall_score = max(0, min(100, overall_score))

        # Determine grade and color
        if overall_score >= 90:
            grade = "A+"
            color = "#00C851"  # Bright Green
            description = "Excellent fairness - minimal bias detected"
        elif overall_score >= 80:
            grade = "A"
            color = "#28A745"  # Green
            description = "Good fairness - low bias levels"
        elif overall_score >= 70:
            grade = "B"
            color = "#8BC34A"  # Yellow-Green
            description = "Fair fairness - moderate bias detected"
        elif overall_score >= 60:
            grade = "C"
            color = "#FFC107"  # Yellow
            description = "Poor fairness - significant bias present"
        elif overall_score >= 50:
            grade = "D"
            color = "#FF9800"  # Orange
            description = "Bad fairness - substantial bias issues"
        else:
            grade = "F"
            color = "#DC3545"  # Red
            description = "Critical fairness issues - severe bias detected"

        # Calculate improvement potential (how much could be gained)
        improvement_potential = 100 - overall_score

        return OverallFairnessScore(
            score=round(overall_score, 1),
            grade=grade,
            color=color,
            description=description,
            improvement_potential=round(improvement_potential, 1),
        )

    def estimate_post_mitigation_score(
        self, current_score: OverallFairnessScore, bias_detected: Dict[str, List[str]], mitigation_strategy: str
    ) -> OverallFairnessScore:
        """
        Estimate the expected fairness score after applying mitigation strategies.
        """
        # Strategy effectiveness mapping (conservative estimates)
        strategy_effectiveness = {
            "reweighing": 0.15,  # 15% improvement
            "exponentiated_gradient": 0.25,  # 25% improvement
            "grid_search": 0.20,  # 20% improvement
            "postprocessing": 0.10,  # 10% improvement
            "adversarial_debiasing": 0.30,  # 30% improvement
        }

        base_improvement = strategy_effectiveness.get(mitigation_strategy, 0.15)

        # Adjust based on number of bias types detected
        bias_severity = len([bias_list for bias_list in bias_detected.values() if bias_list])
        severity_multiplier = max(0.5, 1.0 - (bias_severity * 0.1))

        # Calculate expected improvement
        improvement = current_score.improvement_potential * base_improvement * severity_multiplier
        expected_score = min(100, current_score.score + improvement)

        # Create new score object
        if expected_score >= 90:
            grade = "A+"
            color = "#00C851"
            description = "Expected excellent fairness after mitigation"
        elif expected_score >= 80:
            grade = "A"
            color = "#28A745"
            description = "Expected good fairness after mitigation"
        elif expected_score >= 70:
            grade = "B"
            color = "#8BC34A"
            description = "Expected fair fairness after mitigation"
        elif expected_score >= 60:
            grade = "C"
            color = "#FFC107"
            description = "Expected moderate fairness after mitigation"
        elif expected_score >= 50:
            grade = "D"
            color = "#FF9800"
            description = "Expected limited fairness improvement"
        else:
            grade = "F"
            color = "#DC3545"
            description = "Significant mitigation efforts needed"

        return OverallFairnessScore(
            score=round(expected_score, 1),
            grade=grade,
            color=color,
            description=description,
            improvement_potential=round(100 - expected_score, 1),
        )

    def calculate_fairness_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate fairness metrics directly without group comparison keys.
        Returns simple metric dictionary that works with original string values.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            sensitive_features: Series of sensitive feature values (original strings).

        Returns:
            A dictionary with direct fairness metrics (no group comparison keys).
        """
        from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

        try:
            # Calculate metrics directly using fairlearn
            metrics = {
                "demographic_parity_difference": demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sensitive_features
                ),
                "equalized_odds_difference": equalized_odds_difference(
                    y_true, y_pred, sensitive_features=sensitive_features
                ),
                "equal_opportunity_difference": self._equal_opportunity_difference(y_true, y_pred, sensitive_features),
                "disparate_impact_ratio": self._disparate_impact_ratio(y_true, y_pred, sensitive_features),
                "average_odds_difference": self._average_odds_difference(y_true, y_pred, sensitive_features),
                "theil_index": self._calculate_theil_index_between_groups(y_pred, sensitive_features),
            }

            return metrics

        except Exception as e:
            logging.error(f"Error calculating fairness metrics: {e}")
            # Return default metrics if calculation fails
            return {
                "demographic_parity_difference": 0.0,
                "equalized_odds_difference": 0.0,
                "equal_opportunity_difference": 0.0,
                "disparate_impact_ratio": 1.0,
                "average_odds_difference": 0.0,
                "theil_index": 0.0,
            }

    def _equal_opportunity_difference(self, y_true, y_pred, sensitive_features):
        """Calculate equal opportunity difference (TPR difference)"""
        from fairlearn.metrics import MetricFrame, true_positive_rate

        mf = MetricFrame(
            metrics=true_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
        )

        tpr_by_group = mf.by_group
        if len(tpr_by_group) >= 2:
            return tpr_by_group.max() - tpr_by_group.min()
        return 0.0

    def _disparate_impact_ratio(self, y_true, y_pred, sensitive_features):
        """Calculate disparate impact ratio"""
        from fairlearn.metrics import MetricFrame, selection_rate

        mf = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)

        rates = mf.by_group
        if len(rates) >= 2 and rates.max() > 0:
            return rates.min() / rates.max()
        return 1.0

    def _average_odds_difference(self, y_true, y_pred, sensitive_features):
        """Calculate average odds difference"""
        from fairlearn.metrics import MetricFrame, false_positive_rate, true_positive_rate

        # Calculate TPR and FPR for each group
        mf_tpr = MetricFrame(
            metrics=true_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
        )

        mf_fpr = MetricFrame(
            metrics=false_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
        )

        tpr_diff = mf_tpr.by_group.max() - mf_tpr.by_group.min() if len(mf_tpr.by_group) >= 2 else 0
        fpr_diff = mf_fpr.by_group.max() - mf_fpr.by_group.min() if len(mf_fpr.by_group) >= 2 else 0

        return (tpr_diff + fpr_diff) / 2

    def _calculate_theil_index(self, benefits: np.ndarray) -> float:
        """
        Calculate Theil T index for a vector of benefits (e.g., predictions).
        A value of 0 is perfect equality. Higher values mean more inequality.

        NOTE: This is kept for backward compatibility but prefer _calculate_theil_index_between_groups
        """
        if len(benefits) == 0:
            return 0.0

        mean_benefit = np.mean(benefits)

        # Use a tolerance check for floating point numbers
        if np.isclose(mean_benefit, 0):
            return 0.0

        # Use np.where to handle log(0) case safely
        ratio = benefits / mean_benefit
        log_ratio = np.where(ratio > 0, np.log(ratio), 0)

        theil_t = np.mean(ratio * log_ratio)
        return float(theil_t)

    def _calculate_theil_index_between_groups(self, y_pred: np.ndarray, sensitive_features: pd.Series) -> float:
        """
        Calculate Theil index measuring inequality between groups.
        This is the proper way to measure fairness using Theil index.
        """
        groups = sensitive_features.unique()
        if len(groups) < 2:
            return 0.0

        # Calculate mean prediction per group
        group_means = []
        group_sizes = []

        for group in groups:
            mask = sensitive_features == group
            if np.any(mask):
                group_mean = np.mean(y_pred[mask])
                group_size = np.sum(mask)
                group_means.append(group_mean)
                group_sizes.append(group_size)

        if not group_means:
            return 0.0

        # Calculate overall mean
        overall_mean = np.average(group_means, weights=group_sizes)

        if np.isclose(overall_mean, 0):
            return 0.0

        # Calculate Theil T index between groups
        theil = 0.0
        total_size = sum(group_sizes)

        for mean, size in zip(group_means, group_sizes):
            if mean > 0:
                weight = size / total_size
                ratio = mean / overall_mean
                theil += weight * ratio * np.log(ratio)

        return float(theil)

    def detect_bias(
        self,
        metrics: Dict[str, float],  # Changed from metrics_by_comparison
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[str]]:
        """
        Detect bias based on fairness metrics and standard thresholds.
        Updated to work with direct metrics instead of comparison groups.

        Returns:
            Dictionary mapping metric names to list indicating if bias is detected.
        """
        if thresholds is None:
            thresholds = {
                "demographic_parity_difference": 0.1,
                "equalized_odds_difference": 0.1,
                "equal_opportunity_difference": 0.1,
                "disparate_impact_ratio": (0.8, 1.25),
                "average_odds_difference": 0.1,
            }

        bias_detected = {metric: [] for metric in thresholds.keys()}

        for metric_name, threshold in thresholds.items():
            value = metrics.get(metric_name, 0.0)

            if metric_name == "disparate_impact_ratio":
                lower_bound, upper_bound = threshold
                if not (lower_bound <= value <= upper_bound):
                    bias_detected[metric_name].append(metric_name)  # Just indicate the metric name
            else:
                if abs(value) > threshold:
                    bias_detected[metric_name].append(metric_name)  # Just indicate the metric name

        return bias_detected

    def recommend_mitigation_strategies(self, bias_detected: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Generate mitigation strategy recommendations based on detected biases.
        This function works as-is with the updated bias detection.
        """
        recommendations = []
        added_metrics = set()

        if any(bias_detected.get("demographic_parity_difference", [])) and "dp" not in added_metrics:
            recommendations.append(
                {
                    "metric": "Demographic Parity",
                    "issue": "Disproportionate selection rates across groups.",
                    "strategies": [
                        "Reweighing (Pre-processing)",
                        "Exponentiated Gradient (In-processing)",
                        "Grid Search (In-processing)",
                        "Threshold Optimization (Post-processing)",
                    ],
                }
            )
            added_metrics.add("dp")

        if any(bias_detected.get("equalized_odds_difference", [])) and "eo" not in added_metrics:
            recommendations.append(
                {
                    "metric": "Equalized Odds",
                    "issue": "Disparities in true positive and false positive rates.",
                    "strategies": [
                        "Calibrated Equalized Odds (Post-processing)",
                        "Adversarial Debiasing (In-processing)",
                        "Fairness Constraints in Loss Function",
                    ],
                }
            )
            added_metrics.add("eo")

        if any(bias_detected.get("disparate_impact_ratio", [])) and "di" not in added_metrics:
            recommendations.append(
                {
                    "metric": "Disparate Impact",
                    "issue": "Adverse impact on a protected group based on selection rates.",
                    "strategies": [
                        "Disparate Impact Remover (Pre-processing)",
                        "Reweighing (Pre-processing)",
                        "Review features for proxies",
                    ],
                }
            )
            added_metrics.add("di")

        return recommendations

    # --- Placeholders for Advanced Features ---
    def detect_bias_with_significance(
        self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: pd.Series, alpha: float = 0.05
    ) -> Dict[str, Dict[str, Any]]:
        """
        Placeholder for bias detection with statistical significance (e.g., Chi-squared test).
        """
        results = {}
        unique_groups = sensitive_features.unique()
        for group in unique_groups:
            mask = sensitive_features == group
            contingency_table = pd.crosstab(y_true[mask], y_pred[mask])
            if contingency_table.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(contingency_table)
                results[group] = {"chi2_statistic": chi2, "p_value": p, "is_significant": p < alpha}
        return {
            "statistical_significance_note": "This is a basic chi-squared test per group. A full implementation would compare groups.",
            "results": results,
        }

    def analyze_intersectional_bias(
        self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Placeholder for analyzing bias across intersections of multiple sensitive attributes.
        Example: Race AND Gender.
        """
        if sensitive_features_df.shape[1] < 2:
            return {"error": "Intersectional analysis requires at least two sensitive feature columns."}

        # Create a new interaction feature
        intersectional_group = sensitive_features_df.apply(lambda row: "_".join(row.values.astype(str)), axis=1)

        # Re-run the main analysis on this new intersectional group
        intersectional_metrics = self.calculate_fairness_metrics(y_true, y_pred, intersectional_group)

        return {
            "intersectional_analysis_note": "Analysis performed on combined sensitive attributes.",
            "metrics": {k: v.to_dict() for k, v in intersectional_metrics.items()},
        }
