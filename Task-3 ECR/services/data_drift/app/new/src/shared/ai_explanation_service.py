"""
AI Explanation Service for Drift Detection Analysis
Provides business-friendly explanations of technical drift detection results
using AWS Bedrock Claude 3.
"""

import json
import logging
import os
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging before anything else
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables using the robust find_dotenv() method
try:
    from dotenv import find_dotenv, load_dotenv

    # find_dotenv() will search up the directory tree for the .env file
    env_path = find_dotenv()
    if env_path:
        load_dotenv(env_path)
        logger.info(f".env file found and loaded from: {env_path}")
    else:
        logger.info(".env file not found. Using system environment variables.")
except ImportError:
    # This is fine in production environments where variables are set directly
    logger.info("python-dotenv not found. Using system environment variables.")


class DriftAIExplanationService:
    """
    Service for generating AI-powered explanations of drift detection results.

    Initializes the AI explanation service with AWS Bedrock.
    Uses model from environment or default.
    """

    def __init__(self):
        self.bedrock_client = None
        # Use model from env or default
        self.model_id = os.getenv("MODEL_ID_LLM", "anthropic.claude-3-sonnet-20240229-v1:0")
        self._initialize_bedrock()

    def _initialize_bedrock(self):
        """Initialize AWS Bedrock client."""
        try:
            # Get AWS credentials from environment (support both naming conventions)
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID_LLM") or os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY_LLM") or os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("REGION_LLM") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")

            if not aws_access_key or not aws_secret_key:
                raise NoCredentialsError()

            # Initialize AWS Bedrock client
            self.bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
            )
            logger.info(f"AWS Bedrock client initialized successfully with region: {aws_region}")
            logger.info(f"Using model: {self.model_id}")

        except NoCredentialsError:
            logger.warning("AWS credentials not found. AI explanations will use fallback mode.")
            self.bedrock_client = None
        except Exception as e:
            logger.error(f"Failed to initialize AWS Bedrock client: {e}")
            self.bedrock_client = None

    def generate_explanation(self, analysis_data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """
        Generate an AI explanation for drift detection analysis results.

        Args:
            analysis_data: The results from drift detection analysis
            analysis_type: The type of analysis (e.g., 'data_drift_dashboard', 'model_performance', etc.)

        Returns:
            Dictionary containing the structured explanation
        """
        logger.info(f"Generating AI explanation for analysis type: {analysis_type}")

        if not self.bedrock_client:
            logger.warning("No Bedrock client available, using fallback explanation")
            return self._generate_fallback_explanation(analysis_data, analysis_type)

        try:
            # Prepare the prompt for the LLM
            prompt = self._create_drift_prompt(analysis_data, analysis_type)
            logger.debug("Created prompt for LLM")

            # Call AWS Bedrock
            response = self._call_bedrock(prompt)
            logger.info("Received response from AWS Bedrock")

            # Parse and validate the response
            explanation = self._parse_response(response)
            logger.info("Successfully generated and parsed AI explanation")
            return explanation

        except Exception as e:
            logger.error(f"Error generating AI explanation: {e}")
            return self._generate_fallback_explanation(analysis_data, analysis_type)

    def _create_drift_prompt(self, analysis_data: Dict[str, Any], analysis_type: str) -> str:
        """Create a drift detection-specific prompt for the LLM."""
        # Base system prompt for drift detection
        system_prompt = """You are an expert ML monitoring and drift detection specialist. Your task is to explain drift detection analysis results in a clear, business-friendly way that helps stakeholders understand:
1. What drift means for their ML model and business
2. Whether the detected changes are concerning or expected
3. What actions they should consider taking

You must respond with a JSON object in exactly this format:
{
    "summary": "A concise executive summary (2-3 sentences) highlighting the most critical finding and its business impact.",
    "detailed_explanation": "A comprehensive, well-structured business report (3-5 paragraphs minimum) that:\n- Analyzes performance trends with specific metrics and percentages\n- Explains what the changes mean for business operations\n- Identifies root causes and risk factors\n- Provides context about severity and urgency\n- Uses business language while being technically accurate",
    "key_takeaways": [
        "Primary finding with specific metrics and business impact (quantify the effect)",
        "Root cause analysis or hypothesis about why performance changed",
        "Immediate action recommendation with clear rationale and timeline",
        "Long-term monitoring or improvement strategy",
        "Additional considerations or risks to be aware of"
    ]
}

Use clear, business-oriented language with specific numbers and percentages. Structure detailed_explanation as a comprehensive report with multiple paragraphs covering different aspects (trends, impact, root causes, recommendations). Make key_takeaways actionable and specific."""

        # Analysis-specific context prompts
        analysis_contexts = {
            "data_drift_dashboard": """
This is a comprehensive data drift analysis showing how input data distributions have changed compared to the reference dataset (training data). Focus on:
- Overall drift severity across features
- Which features show the most significant changes
- Business implications of data distribution shifts
- Whether model retraining might be needed
""",
            "class_imbalance": """
This analysis examines changes in target class distributions between reference and current data. Focus on:
- How class proportions have shifted
- Impact on model performance expectations
- Whether the imbalance affects business metrics
- Recommendations for handling class distribution changes
""",
            "statistical_analysis": """
This statistical drift analysis uses formal tests to detect significant changes in data distributions. Focus on:
- Which statistical tests detected drift
- Confidence levels and practical significance
- Feature-level drift patterns
- Statistical evidence for data quality concerns
""",
            "feature_analysis": """
This detailed feature analysis examines individual feature drift patterns and importance. Focus on:
- Which features are drifting most significantly
- How feature importance has changed
- Business meaning of the drifting features
- Priority features for monitoring
""",
            "model_performance": """
This analysis tracks model performance across multiple time periods (cohorts) to identify drift patterns and performance degradation over time.

**Generate a comprehensive business report format with the following sections:**

1. **Executive Summary**: Provide a clear, actionable summary of overall model health and critical findings.

2. **Performance Trend Analysis**:
   - Analyze the trajectory of key metrics (accuracy, precision, recall, F1, ROC AUC)
   - Identify monotonic trends (consistent decline/improvement) vs volatile patterns
   - Quantify the magnitude of performance shifts with specific percentages
   - Compare reference period performance to most recent cohort

3. **Critical Metric Degradations**:
   - Highlight metrics that crossed critical thresholds (e.g., accuracy < 70%)
   - Explain the business impact of each degraded metric
   - Specify which types of errors increased (false positives vs false negatives)
   - Quantify the real-world impact (e.g., "missing 62 more positive cases per 900 samples")

4. **Root Cause Indicators**:
   - Analyze class distribution shifts (class imbalance changes)
   - Examine prediction shift patterns (error type changes)
   - Identify volatility in performance (unstable model behavior)
   - Correlate performance drops with specific cohorts/time periods

5. **Business Impact Assessment**:
   - Translate technical metrics into business outcomes
   - Estimate the cost/risk of current performance levels
   - Compare current state to acceptable business thresholds
   - Assess urgency based on trend severity and direction

6. **Actionable Recommendations**:
   - Prioritized list of specific actions (investigate, retrain, monitor, deploy fixes)
   - Timeline recommendations (immediate, short-term, ongoing)
   - Data quality investigations needed
   - Model improvement strategies

Provide specific numbers, percentages, and concrete examples. Use business language while maintaining technical accuracy. Structure the response as a formal business report that executives and technical teams can both understand.
""",
            "degradation_metrics": """
This analysis tracks model performance degradation over time. Focus on:
- Trends in key business metrics
- Rate and severity of performance decline
- Correlation with data drift patterns
- Timeline for intervention needs
""",
            "statistical_significance": """
This analysis examines the statistical significance of model performance changes. Focus on:
- Confidence in observed performance differences
- Whether changes exceed normal variance
- Business relevance vs statistical significance
- Reliability of the performance assessments
""",
            "sanity_check": """
This is a pre-flight diagnostic analysis that estimates expected model performance on new data by comparing distributions. Focus on:
- Overall data similarity between reference and current datasets
- Expected performance degradation estimates
- Severity of detected drift (Stable, Monitor, Warning)
- Which features show the most significant distribution changes
- Whether the model should be deployed on this data
- Actionable recommendations for data quality and model readiness
""",
            "multi_cohort_dashboard": """
This is a multi-cohort drift dashboard comparing 1 baseline dataset against multiple current cohorts (up to 4). Focus on:
- Overall drift patterns across all cohorts
- Which cohorts show the most significant drift from baseline
- Consistently drifted features appearing across multiple cohorts
- Comparison of drift severity between different cohorts
- Business implications of having multiple drifting cohorts
- Prioritization of which cohorts need attention first
- Recommendations for model retraining or data quality interventions
""",
            "multi_cohort_statistical": """
This is a detailed statistical analysis across multiple cohorts using 5 different drift tests (KS, PSI, KL, JS, Wasserstein, Chi-square). Focus on:
- Which statistical tests are most sensitive to detected changes
- Feature-level drift patterns across different cohorts
- Statistical confidence in detected drift
- Practical significance vs statistical significance
- Which cohort pairs show the strongest drift signals
- Correlation patterns between cohorts
- Actionable insights for data monitoring and model health
""",
            "multi_cohort_data_overview": """
This is a comprehensive overview of data quality and metadata across multiple uploaded cohorts. Focus on:
- Overall data quality assessment across all cohorts
- Common features available for drift analysis
- Data quality issues (missing values, duplicates, type mismatches)
- Cohort compatibility for analysis
- Data readiness for drift detection
- Recommendations for data preparation or cleaning
- Whether cohorts are suitable for meaningful comparison
""",
            "multi_cohort_comparison": """
This is a detailed pairwise comparison between 2-3 selected cohorts for a specific feature. Focus on:
- Distribution differences for the selected feature
- Statistical significance of observed differences (using all 5 drift tests)
- Changes in central tendency (mean, median, std for numerical; mode, frequencies for categorical)
- Practical magnitude of changes
- Business implications of the feature drift
- Whether the differences warrant intervention
- Recommendations specific to this feature's behavior
""",
        }

        # Get analysis-specific context
        context = analysis_contexts.get(analysis_type, "This is a drift detection analysis.")

        # Create the full prompt
        context_prompt = f"""
Analysis Type: {analysis_type}
Context: {context}
Analysis Results: {json.dumps(analysis_data, indent=2, default=str)}

Please analyze these drift detection results and provide a business-friendly explanation following the specified JSON format. Focus on helping stakeholders understand what these findings mean for their ML model's health and business impact.
"""
        return f"{system_prompt}\n\n{context_prompt}"

    def _call_bedrock(self, prompt: str) -> str:
        """Make a call to AWS Bedrock."""
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,  # Increased for comprehensive business report format
                "temperature": 0.3,  # Lower temperature for more focused, analytical responses
                "messages": [{"role": "user", "content": prompt}],
            }

            response = self.bedrock_client.invoke_model(modelId=self.model_id, body=json.dumps(request_body))
            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"]

        except ClientError as e:
            logger.error(f"AWS Bedrock client error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Bedrock: {e}")
            raise

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate the LLM response."""
        try:
            # Clean the response text
            clean_response = response_text.strip()

            # Try to extract JSON from the response, often wrapped in code blocks
            if "```json" in clean_response:
                start = clean_response.find("```json") + 7
                end = clean_response.find("```", start)
                if end != -1:
                    clean_response = clean_response[start:end].strip()
            elif "```" in clean_response:
                start = clean_response.find("```") + 3
                end = clean_response.find("```", start)
                if end != -1:
                    clean_response = clean_response[start:end].strip()

            # If the response starts with {, it's likely raw JSON
            if clean_response.startswith("{") and clean_response.endswith("}"):
                parsed = json.loads(clean_response)
            else:
                # Try to find JSON within the text
                start_brace = clean_response.find("{")
                end_brace = clean_response.rfind("}")
                if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                    json_part = clean_response[start_brace : end_brace + 1]
                    parsed = json.loads(json_part)
                else:
                    # Fallback: try parsing the entire response
                    parsed = json.loads(clean_response)

            # Validate required fields
            required_fields = ["summary", "detailed_explanation", "key_takeaways"]
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")

            # Ensure key_takeaways is a list
            if not isinstance(parsed["key_takeaways"], list):
                parsed["key_takeaways"] = [str(parsed["key_takeaways"])]

            logger.info("Successfully parsed AI explanation response")
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response: {response_text}")
            # Return a fallback explanation
            return self._generate_fallback_explanation({}, "unknown")

    def _generate_fallback_explanation(self, analysis_data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """Generate a fallback explanation when AI service is unavailable."""
        fallback_messages = {
            "data_drift_dashboard": {
                "summary": "Data drift analysis completed. AI explanations are temporarily unavailable.",
                "detailed_explanation": "Your data drift analysis has been completed successfully. This analysis compares your current data distributions with the reference dataset to identify potential changes that could affect model performance. The results include statistical measurements of drift across all features and visualizations showing distribution changes.",
                "key_takeaways": [
                    "Data drift analysis completed successfully",
                    "Review the statistical metrics and visualizations for insights",
                    "AI-powered business explanations will be available when the service is restored",
                ],
            },
            "model_performance": {
                "summary": "Model performance analysis completed. AI explanations are temporarily unavailable.",
                "detailed_explanation": "Your model performance comparison analysis has been completed. This analysis compares performance metrics between your reference and current datasets to identify any degradation in model quality. The results include key performance indicators and statistical significance tests.",
                "key_takeaways": [
                    "Performance analysis completed successfully",
                    "Check performance metrics for any concerning trends",
                    "AI-powered insights will be available when the service is restored",
                ],
            },
            "multi_cohort_dashboard": {
                "summary": "Multi-cohort drift analysis completed. AI explanations are temporarily unavailable.",
                "detailed_explanation": "Your multi-cohort drift dashboard analysis has been completed successfully. This analysis compares your baseline dataset against multiple current cohorts to identify drift patterns. The results include cohort-specific summaries, aggregated drift metrics, and consistently drifted features across cohorts.",
                "key_takeaways": [
                    "Multi-cohort drift analysis completed successfully",
                    "Review cohort summaries to identify most drifted cohorts",
                    "AI-powered business insights will be available when the service is restored",
                ],
            },
            "multi_cohort_statistical": {
                "summary": "Multi-cohort statistical analysis completed. AI explanations are temporarily unavailable.",
                "detailed_explanation": "Your statistical drift analysis across multiple cohorts has been completed. This analysis uses 5 statistical tests (KS, PSI, KL, JS, Wasserstein, Chi-square) to detect drift at the feature level across all cohort pairs. Results include test scores, p-values, and plot data for visualization.",
                "key_takeaways": [
                    "Statistical tests completed across all cohort pairs",
                    "Review test results and plot data for detailed insights",
                    "AI-powered explanations will be available when the service is restored",
                ],
            },
            "multi_cohort_data_overview": {
                "summary": "Multi-cohort data overview completed. AI explanations are temporarily unavailable.",
                "detailed_explanation": "Your data quality and metadata overview across multiple cohorts has been completed. This analysis provides comprehensive information about each cohort including shape, features, missing values, duplicates, and data quality metrics. Validation results confirm cohort compatibility for drift analysis.",
                "key_takeaways": [
                    "Data quality assessment completed for all cohorts",
                    "Review cohort metadata and validation warnings",
                    "AI-powered insights will be available when the service is restored",
                ],
            },
            "multi_cohort_comparison": {
                "summary": "Cohort comparison analysis completed. AI explanations are temporarily unavailable.",
                "detailed_explanation": "Your detailed pairwise cohort comparison for the selected feature has been completed. This analysis provides distribution overlays, central tendency metrics, and comprehensive drift test results using all applicable statistical tests. The drift matrix shows pairwise comparisons across all selected cohorts.",
                "key_takeaways": [
                    "Feature-level cohort comparison completed successfully",
                    "Review distribution plots and drift test results",
                    "AI-powered recommendations will be available when the service is restored",
                ],
            },
        }
        # Get analysis-specific fallback or use generic
        fallback = fallback_messages.get(
            analysis_type,
            {
                "summary": "Analysis completed successfully. AI explanations are temporarily unavailable.",
                "detailed_explanation": f"Your {analysis_type} analysis has been completed and the results are available for review. The technical analysis has been performed according to best practices for ML monitoring and drift detection.",
                "key_takeaways": [
                    "Analysis completed successfully",
                    "Technical results are available for review",
                    "AI explanations will be available when the service is restored",
                ],
            },
        )
        return fallback


# Create a singleton instance
ai_explanation_service = DriftAIExplanationService()
