# llm_analyzer.py

import json
import os
from typing import Any, Dict

import boto3


class LLMAnalyzer:
    """
    Uses AWS Bedrock (Anthropic Claude model) to generate
    natural-language analyses of ML model explainability dashboards.
    """

    def __init__(self):
        self.bedrock = None
        self._initialize_bedrock()

    def _initialize_bedrock(self):
        """Initialize the AWS Bedrock client for Claude invocations."""

        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID_LLM")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY_LLM")
        region_name = os.getenv("REGION_LLM", "us-east-1")
        if not aws_access_key_id or not aws_secret_access_key:
            print("[LLMAnalyzer] Missing AWS credentials for Bedrock initialization.")
            self.bedrock = None
            return
        try:
            self.bedrock = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            print("[LLMAnalyzer] Bedrock client initialized successfully.")
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
            # Claude's response under content → list[ { text: ... } ]
            return data["content"][0]["text"].strip()
        except Exception as e:
            return f"⛔ Claude invocation error: {e}"

    def analyze_dashboard_metrics(self, dashboard_summary: Dict[str, Any]) -> str:
        """
        Generate an executive summary of the ML explainability dashboard
        for business stakeholders and non-technical users.
        """

        # Extract key metrics from dashboard summary
        model_info = dashboard_summary.get("model_info", {})
        feature_importance = dashboard_summary.get("feature_importance", [])
        shap_analysis = dashboard_summary.get("shap_analysis", {})
        current_prediction = dashboard_summary.get("current_prediction", {})
        what_if_insights = dashboard_summary.get("what_if_insights", {})

        prompt = f"""
You are a senior data scientist and business intelligence analyst. Your task is to provide an executive summary of a machine learning model's explainability dashboard for business stakeholders and non-technical decision makers.

## MODEL OVERVIEW
- Dataset Features: {model_info.get('n_features', 'N/A')} features
- Training Samples: {model_info.get('n_samples', 'N/A')} records
- Target Variable: {model_info.get('target_column', 'N/A')}
- Current Model Prediction: {current_prediction.get('value', 'N/A')}% confidence
- Prediction Context: {current_prediction.get('interpretation', 'N/A')}

## FEATURE IMPORTANCE ANALYSIS
Top Contributing Features:
{self._format_feature_importance(feature_importance)}

## SHAP (Model Explanation) INSIGHTS
- Model Decision Drivers: {shap_analysis.get('explanation', 'Features that most influence predictions')}
- Feature Interactions: {shap_analysis.get('interactions', 'How features work together')}
- Prediction Confidence: {shap_analysis.get('confidence_level', 'Model certainty in predictions')}

## WHAT-IF SCENARIO ANALYSIS
- Sensitivity Analysis: {what_if_insights.get('sensitivity', 'How changes in inputs affect predictions')}
- Key Influential Parameters: {what_if_insights.get('key_drivers', 'Most impactful variables')}
- Business Scenarios: {what_if_insights.get('scenarios', 'Different input combinations tested')}

## REQUESTED ANALYSIS

Please provide a comprehensive executive summary that addresses:

### 1. BUSINESS IMPACT ASSESSMENT
- What does this model predict and why is it important for business decisions?
- How confident should stakeholders be in the current prediction?
- What are the key business drivers identified by the model?

### 2. FEATURE INSIGHTS FOR DECISION MAKERS
- Explain the top 5 most important features in simple business terms
- How do these features influence the outcome?
- What actionable insights can be derived from feature importance?

### 3. MODEL RELIABILITY & TRUST
- How explainable and interpretable is this model for business use?
- What level of confidence should we have in the predictions?
- Are there any red flags or areas of concern?

### 4. SCENARIO PLANNING & WHAT-IF INSIGHTS
- How sensitive is the model to changes in key variables?
- What scenarios show the most significant prediction changes?
- How can this inform business strategy and decision-making?

### 5. RECOMMENDATIONS FOR ACTION
- What immediate actions should be taken based on current predictions?
- How can businesses leverage these insights for competitive advantage?
- What monitoring and review processes should be established?

### 6. RISK ASSESSMENT & LIMITATIONS
- What are the potential risks of relying on this model?
- What scenarios might cause the model to perform poorly?
- How should businesses hedge against model uncertainty?

**Format your response as a professional executive summary using clear business language, avoiding technical jargon. Use bullet points, headers, and structured sections for easy readability by C-level executives and business stakeholders.**

**Focus on actionable insights, business value, and practical recommendations rather than technical model details.**
"""

        return self._invoke_claude(prompt)

    def _format_feature_importance(self, feature_importance: list) -> str:
        """Format feature importance data for the prompt."""
        if not feature_importance:
            return "No feature importance data available"

        formatted = []
        for i, feature in enumerate(feature_importance[:10], 1):
            name = feature.get("feature", "Unknown")
            importance = feature.get("importance", 0)
            formatted.append(f"  {i}. {name}: {importance:.4f} importance score")

        return "\n".join(formatted)

    def generate_technical_insights(self, dashboard_summary: Dict[str, Any]) -> str:
        """
        Generate technical insights for data scientists and ML engineers.
        """
        prompt = f"""
You are a principal ML engineer providing technical insights on model explainability.

Dashboard Summary: {json.dumps(dashboard_summary, indent=2)}

Provide technical analysis covering:
1. Model interpretability assessment
2. Feature interaction analysis
3. SHAP value distribution insights
4. Prediction stability analysis
5. Recommendations for model improvement
6. Monitoring and validation strategies

Format as a technical report for ML practitioners.
"""

        return self._invoke_claude(prompt)
