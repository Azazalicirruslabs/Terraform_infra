import json
import logging
import os
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class AIExplanationService:
    """
    Service for generating AI-powered explanations of data analysis results
    using AWS Bedrock.
    """

    def __init__(self):
        self.bedrock_client = None
        self.model_id = os.getenv("MODEL_ID_LLM")  # Claude 3 Sonnet
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the AWS Bedrock client."""
        try:
            # Use configuration settings
            if os.getenv("AWS_ACCESS_KEY_ID_LLM") and os.getenv("AWS_SECRET_ACCESS_KEY_LLM"):
                # Use explicit credentials if provided
                self.bedrock_client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=os.getenv("REGION_LLM"),
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_LLM"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_LLM"),
                    aws_session_token=(
                        os.getenv("AWS_SESSION_TOKEN_LLM") if os.getenv("AWS_SESSION_TOKEN_LLM") else None
                    ),
                )
            else:
                # Try to create client with default credentials
                self.bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=os.getenv("REGION_LLM"))
            logger.info(f"AWS Bedrock client initialized successfully in region {os.getenv('REGION_LLM')}")
        except NoCredentialsError:
            logger.warning("AWS credentials not found. AI explanations will be disabled.")
            self.bedrock_client = None
        except Exception as e:
            logger.error(f"Failed to initialize AWS Bedrock client: {e}")
            self.bedrock_client = None

    def generate_explanation(self, analysis_data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """
        Generate an AI explanation for the given analysis data.

        Args:
            analysis_data: The data, plots, and metrics from the analysis
            analysis_type: The type of analysis (e.g., 'overview', 'feature_importance', etc.)

        Returns:
            Dictionary containing the structured explanation
        """
        if not self.bedrock_client:
            return self._generate_fallback_explanation(analysis_data, analysis_type)

        try:
            # Prepare the prompt for the LLM
            prompt = self._create_prompt(analysis_data, analysis_type)

            # Call AWS Bedrock
            response = self._call_bedrock(prompt)

            # Parse and validate the response
            explanation = self._parse_response(response)

            return explanation

        except Exception as e:
            logger.error(f"Error generating AI explanation: {e}")
            return self._generate_fallback_explanation(analysis_data, analysis_type)

    def _create_prompt(self, analysis_data: Dict[str, Any], analysis_type: str) -> str:
        """Create a structured prompt for the LLM."""

        # Base system prompt
        system_prompt = """You are an expert data analyst and communicator. Your task is to explain data analysis results in a clear, non-technical way that makes insights accessible to users without a technical background.

You must respond with a JSON object in exactly this format:
{
  "summary": "A brief, high-level overview of the key insights. This should be no more than two sentences.",
  "detailed_explanation": "A step-by-step breakdown of the data, including what the plots show and what the metrics mean. Use simple language and analogies where appropriate to make the information accessible to someone without a technical background.",
  "key_takeaways": [
    "A bullet point summarizing the most important conclusion or action item.",
    "Another important conclusion.",
    "A third key takeaway."
  ]
}

Use simple, clear language. Avoid technical jargon. Focus on what the results mean in practical terms."""

        # Create context-specific prompt
        context_prompt = f"""
Analysis Type: {analysis_type}

Analysis Data:
{json.dumps(analysis_data, indent=2, default=str)}

Please analyze this data and provide an explanation following the specified JSON format. Focus on making the insights clear and actionable for non-technical users.
"""

        return f"{system_prompt}\n\n{context_prompt}"

    def _call_bedrock(self, prompt: str) -> str:
        """Make a call to AWS Bedrock."""
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
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
            # Try to extract JSON from the response
            # Look for JSON content between triple backticks if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()

            # Parse the JSON
            parsed = json.loads(response_text)

            # Validate required fields
            required_fields = ["summary", "detailed_explanation", "key_takeaways"]
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")

            # Ensure key_takeaways is a list
            if not isinstance(parsed["key_takeaways"], list):
                parsed["key_takeaways"] = [str(parsed["key_takeaways"])]

            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response: {response_text}")
            # Return a fallback explanation
            return self._generate_fallback_explanation({}, "unknown")

    def _generate_fallback_explanation(self, analysis_data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """Generate a fallback explanation when AI service is unavailable."""
        return {
            "summary": "AI explanation service is currently unavailable. Here's a basic overview of your data.",
            "detailed_explanation": f"This is a {analysis_type} analysis. The data has been processed and analyzed to provide insights into your model's performance and data characteristics. Please check back later for AI-powered explanations.",
            "key_takeaways": [
                "Your data has been successfully analyzed",
                "The analysis results are available for review",
                "AI explanations will be available when the service is restored",
            ],
        }
