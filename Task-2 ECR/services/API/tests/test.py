# Simple test to check Claude access via AWS Bedrock
import os

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a Bedrock Runtime client
try:
    client = boto3.client(
        "bedrock-runtime",
        region_name="ap-south-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_LLM"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_LLM"),
    )
    print("✅ Bedrock client created successfully")
except Exception as e:
    print(f"❌ Failed to create Bedrock client: {e}")
    exit(1)

# List of models to try (in order of preference)
models_to_try = ["apac.anthropic.claude-3-7-sonnet-20250219-v1:0", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"]

# Test message
user_message = "Say hello and tell me what model you are in exactly 10 words."

# Try each model
for model_id in models_to_try:
    print(f"\n🔄 Trying model: {model_id}")

    # Prepare conversation
    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    try:
        # Send the message to the model
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
        )

        # Extract and print the response text
        response_text = response["output"]["message"]["content"][0]["text"]
        print(f"✅ SUCCESS with {model_id}")
        print(f"📝 Response: {response_text}")
        break

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        print(f"❌ {error_code}: {error_message}")

        if error_code == "AccessDeniedException":
            print("   → Check your IAM permissions for Bedrock")
        elif error_code == "ValidationException":
            print("   → Model might not be available in your region")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

print("\n" + "=" * 50)
print("If no models worked, check:")
print("1. AWS credentials are correct")
print("2. IAM user has Bedrock permissions")
print("3. Model access is enabled in Bedrock console")
print("4. You're in the right AWS region")
