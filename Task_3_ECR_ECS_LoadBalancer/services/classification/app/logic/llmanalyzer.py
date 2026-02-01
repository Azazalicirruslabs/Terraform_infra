# llm_analyzer.py

import json
import os

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class LLMAnalyzer:
    """
    Uses AWS Bedrock (Anthropic Claude model) to generate
    natural-language analyses of classification reports.
    """

    def __init__(self):
        self.bedrock = None
        self._initialize_bedrock()

    def _initialize_bedrock(self):
        """Initialize the AWS Bedrock client for Claude invocations."""
        try:
            self.bedrock = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("REGION_LLM", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_LLM"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_LLM"),
            )
        except Exception as e:
            print(f"[LLMAnalyzer] Bedrock init failed: {e}")
            self.bedrock = None

    def _invoke_claude(self, prompt: str) -> str:
        """
        Send `prompt` to Claude via Bedrock and return the text response.
        """
        if not self.bedrock:
            return "⛔ Bedrock not initialized, cannot invoke Claude."

        try:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 131072,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            }
            resp = self.bedrock.invoke_model(
                modelId=os.getenv("MODEL_ID_LLM", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"),
                body=json.dumps(payload),
            )
            body = resp["body"].read().decode("utf-8")
            data = json.loads(body)
            # Claude’s response under content → list[ { text: ... } ]
            return data["content"][0]["text"].strip()
        except Exception as e:
            return f"⛔ Claude invocation error: {e}"

    def get_classification_summary(self, report_dict: dict) -> str:
        """
        Turn Evidently’s classification JSON into a short bullet summary.
        """
        lines = []
        for metric in report_dict.get("metrics", []):
            mid = metric.get("metric_id", "")
            name = mid.split("(")[0]  # e.g. "Precision"
            val = metric.get("value")

            if isinstance(val, dict):
                # per-class metrics
                lines.append(f"- {name} by class:")
                for cls, score in val.items():
                    lines.append(f"    • Class {cls}: {score:.3f}")
            elif isinstance(val, (int, float)):
                lines.append(f"- {name}: {val:.3f}")
            else:
                lines.append(f"- {name}: {val}")
        return "\n".join(lines)

    def analyze_classification(
        self, report_dict: dict, train_df: pd.DataFrame, test_df: pd.DataFrame, target_name: str, summary_text: str
    ) -> str:
        """
        Build and send a classification analysis prompt to Claude,
        embedding overall metrics (accuracy, precision, recall, AUC…)
        and per-class breakdown, plus a short summary.
        """
        prompt = f"""
You are a senior data scientist and machine learning engineer.
Your task is to provide an in-depth evaluation of a binary classification model deployed in production.

1) DATASET OVERVIEW
- Training set size & shape: {train_df.shape}
- Test set size & shape:     {test_df.shape}
- Target column: {target_name}
- Class balance:
    • Positive (1): {train_df[target_name].mean():.1%} in train, {test_df[target_name].mean():.1%} in test
- Key features (first 10): {', '.join(train_df.columns.drop(target_name)[:10])}{'...' if len(train_df.columns)>11 else ''}

2) PERFORMANCE METRICS
{summary_text}

3) CONFUSION MATRIX & CLASS-WISE BEHAVIORS
- Provide a Markdown table of the confusion matrix.
- Discuss precision vs recall trade-offs per class.
- Highlight any class imbalance effects.

4) CALIBRATION & THRESHOLD ANALYSIS
- Comment on whether predicted probabilities appear well-calibrated.
- Suggest an optimal decision threshold based on business goals (e.g., maximize F1, precision, recall).

5) ERROR MODES & RISK
- Identify which type of errors (false positives vs false negatives) are more prevalent.
- Analyze business impact of each error type (cost of FP vs FN).

6) FAIRNESS & STABILITY
- Check for any signs of model bias or performance disparity across subgroups (if known).
- Comment on model stability: does test performance drift noticeably from train?

7) RECOMMENDATIONS
- Concrete steps to improve model (feature engineering, rebalancing, calibration, threshold tuning, etc.).
- Monitoring & alerting suggestions (which metrics to watch and thresholds to set).

8) NEXT STEPS & ROADMAP
- Short-term actions.
- Long-term roadmap (data collection, model retraining cadence, A/B tests).

Format your response in detailed Markdown with clear headings, subheads, bullet points, and if needed tables.

"""
        return self._invoke_claude(prompt)
