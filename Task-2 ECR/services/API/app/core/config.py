import os
from io import BytesIO

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from dotenv import load_dotenv

load_dotenv()
import ipaddress
import socket
from typing import List
from urllib.parse import unquote, urlparse

import requests
from fastapi import HTTPException
from sqlalchemy.orm import Session

from shared_migrations.models.tenant import Tenant


def is_allowed_presigned_s3_url(url: str) -> bool:
    """
    Check if the URL appears to be an AWS S3 presigned URL and does not resolve to a private IP.
    Allows only HTTPS URLs for *.amazonaws.com domain or regional S3 endpoints.
    Also blocks internal/private IP addresses.
    """
    try:
        parsed = urlparse(url)
        # Check scheme
        if parsed.scheme != "https":
            return False
        # Check hostname is s3.amazonaws.com or matches *.s3.*.amazonaws.com (regional)
        host = parsed.hostname
        if not host:
            return False
        allowed_patterns = [
            ".s3.amazonaws.com",
            ".s3-",
            ".s3.",
        ]
        if "amazonaws.com" not in host or not (
            host.endswith(".amazonaws.com")
            and (host == "s3.amazonaws.com" or any(pat in host for pat in allowed_patterns))
        ):
            return False
        # Optional: check for AWS presigned params
        qs = parsed.query
        if "X-Amz-Signature" not in qs or "X-Amz-Credential" not in qs:
            return False
        # Resolve the hostname (prevents DNS rebinding, direct IP).
        try:
            ip_addr = socket.gethostbyname(host)
            ip = ipaddress.ip_address(ip_addr)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        except Exception:
            return False
        return True
    except Exception:
        return False


def fetch_from_presigned_urls(presigned_urls: List[str]) -> List[tuple[str, str, bytes]]:
    """
    Downloads files from the given list of presigned S3 URLs.
    Returns a list of (filename, content_type, content) tuples.
    """
    results = []
    for url in presigned_urls:
        if not is_allowed_presigned_s3_url(url):
            raise HTTPException(status_code=400, detail=f"URL '{url}' is not a valid AWS S3 presigned URL.")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            parsed_url = urlparse(url)
            content_type = response.headers.get("Content-Type", "application/octet-stream")
            filename = unquote(os.path.basename(parsed_url.path))  # Handles %20, %2F etc.

            results.append((filename, content_type, response.content))

        except requests.exceptions.Timeout:
            raise HTTPException(status_code=408, detail=f"Timeout while downloading from presigned URL '{url}'")
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=400, detail=f"Network error fetching file from presigned URL '{url}': {str(e)}"
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error fetching file from presigned URL '{url}': {str(e)}")
    return results


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")

TEMPORARY_FILES = "temporary_files"

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

ALLOWED_CONTENT_TYPES = {
    "text/plain",  # .txt
    "text/csv",  # .csv
    "application/json",  # .json
    "application/vnd.ms-excel",  # .csv (Excel style)
    "application/vnd.apache.parquet",  # .parquet
}

ALLOWED_MODEL_TYPES = {
    "application/octet-stream",  # Often for pickle
    "application/onnx",  # ONNX
    "application/xml",  # PMML
}

ALLOWED_MODEL_EXTENSIONS = {".pkl", ".joblib", ".onnx", ".xml"}


ALLOWED_ANALYSIS_TYPES = {
    "Data Drift": "datadrift",
    "Fairness": "fairness",
    "Classification": "classification",
    "Regression": "regression",
}


def check_analysis(analysis_type):

    analysis = ""
    for key, val in ALLOWED_ANALYSIS_TYPES.items():
        if analysis_type in key:
            analysis = val
        else:
            continue
    return analysis


def is_valid_model_file(filename: str, content_type: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return content_type in ALLOWED_MODEL_TYPES and ext in ALLOWED_MODEL_EXTENSIONS


def upload_to_s3(
    username: str,
    file_content: bytes,
    filename: str,
    content_type: str,
    tenant_id,
    db: Session,
    project_name,
    analysis: str = "temparory_files",
):
    try:
        tenant_data = db.query(Tenant).filter_by(id=tenant_id).first()
        if not tenant_data:
            raise HTTPException(status_code=404, detail="Tenant not found")

        tenant_name = tenant_data.name
        ext = os.path.splitext(filename)[1].lower()

        if content_type in ALLOWED_CONTENT_TYPES or (ext == ".parquet" and content_type == "application/octet-stream"):
            base_prefix = f"{tenant_name}/{username}/{analysis}/{project_name}/files/"
        elif is_valid_model_file(filename, content_type):
            base_prefix = f"{tenant_name}/{username}/{analysis}/{project_name}/models/"
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported content type or extension: type={content_type}, ext={ext}"
            )

        s3_key = f"{base_prefix}{filename}"
        print(f"Uploading to S3 key: {s3_key}")

        s3_client = get_s3_client()

        # Upload file to S3
        s3_client.upload_fileobj(
            Fileobj=BytesIO(file_content), Bucket=S3_BUCKET, Key=s3_key, ExtraArgs={"ContentType": content_type}
        )

        # Create presigned URL for secure access instead of direct URL
        presigned_url = generate_presigned_url(s3_key, expiration=3600)  # 1 hour expiry

        print(f"File uploaded successfully. Presigned URL generated.")
        return presigned_url

    except (BotoCoreError, NoCredentialsError, ClientError) as e:
        return f"Error uploading file to S3: {str(e)}"


# file download
def get_s3_client():
    """
    Returns a properly configured S3 client for the current region.
    Uses AWS_REGION environment variable without any fallback.
    Forces regional endpoint to avoid IllegalLocationConstraintException.
    """
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


# def get_s3_client():
#     region_name = os.getenv("AWS_REGION", "ap-south-1")
#     print("Creating S3 client for region:", region_name)
#     return boto3.client(
#         "s3",
#         aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#         aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#         region_name=os.getenv("AWS_REGION")
#     )


def get_json_file_from_s3(bucket_name: str, file_key: str):
    s3_client = get_s3_client()
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        json_data = response["Body"].read().decode("utf-8")
        return json_data
    except ClientError as e:
        raise HTTPException(status_code=404, detail=f"File not found or S3 error: {str(e)}")


def generate_presigned_url(s3_key: str, expiration: int = 3600) -> str:
    """
    Generate a presigned URL for secure access to S3 objects.

    Args:
        s3_key: The S3 object key
        expiration: URL expiration time in seconds (default: 1 hour)

    Returns:
        Presigned URL string
    """
    try:
        s3_client = get_s3_client()
        presigned_url = s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": S3_BUCKET, "Key": s3_key}, ExpiresIn=expiration
        )
        return presigned_url
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Error generating presigned URL: {str(e)}")
