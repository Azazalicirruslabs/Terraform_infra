import hashlib
import io
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import chardet
import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.base import BaseEstimator

load_dotenv()

# File cache to avoid re-downloading same files
_file_cache = {}


def generate_analysis_id(user_id: str, data_preview: dict) -> str:
    """
    Generates a unique analysis_id based on user_id and dataset content.

    Args:
        user_id (str): The ID of the user performing the analysis.
        data_preview (dict): The preview data used in the analysis.

    Returns:
        str: A unique SHA256 hash string representing this specific analysis.
    """
    # Convert data to a consistent JSON string (sorted keys ensure same structure always yields same hash)
    data_str = json.dumps(data_preview, sort_keys=True)

    # Combine user_id and data content
    unique_string = f"{user_id}_{data_str}"

    # Create SHA-256 hash
    analysis_hash = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()

    # Optionally shorten it to first 16 or 32 characters
    return analysis_hash[:32]


def download_file_from_url(url: str, use_cache: bool = True) -> bytes:
    """
    Download file from URL with caching support.

    Args:
        url: The URL to download from
        use_cache: Whether to use cached version if available

    Returns:
        File content as bytes

    Raises:
        RuntimeError: If download fails
    """
    # Check cache first
    if use_cache and url in _file_cache:
        print(f"[CACHE] Using cached version of {url}")
        return _file_cache[url]

    try:
        print(f"[DOWNLOAD] Fetching file from {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        file_content = response.content

        # Cache the content
        if use_cache:
            _file_cache[url] = file_content
            print(f"[CACHE] Cached file from {url}")

        return file_content

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download file from {url}: {str(e)}") from e
    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError(f"Unexpected error downloading file from {url}: {str(e)}") from e


def load_flexible_csv(file_content: bytes) -> pd.DataFrame:
    """
    Load CSV data with automatic delimiter and encoding detection.

    Args:
        file_content: CSV file content as bytes

    Returns:
        Loaded DataFrame

    Raises:
        RuntimeError: If CSV loading fails
    """
    # Detect encoding
    result = chardet.detect(file_content)
    encoding = result["encoding"] if result["encoding"] else "utf-8"

    # Try common delimiters
    for delimiter in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.BytesIO(file_content), encoding=encoding, delimiter=delimiter)
            if len(df.columns) > 1:  # Success if more than one column
                print(f"[CSV] Successfully loaded with delimiter '{delimiter}' and encoding '{encoding}'")
                return df
        except Exception:  # pylint: disable=broad-except
            continue

    # Last resort: try default comma delimiter
    try:
        df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
        if len(df.columns) == 1:
            raise RuntimeError("Could not determine appropriate delimiter. Data might be malformed or single-column.")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {str(e)}") from e


def load_dataframe_from_url(url: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Download and load DataFrame from URL.

    Args:
        url: The URL to download CSV from
        use_cache: Whether to use cached version

    Returns:
        Loaded DataFrame

    Raises:
        RuntimeError: If loading fails
    """
    file_content = download_file_from_url(url, use_cache=use_cache)
    return load_flexible_csv(file_content)


def load_model_from_url(url: str, use_cache: bool = True) -> BaseEstimator:
    """
    Download and load model from URL with automatic format detection.
    Supports pickle (.pkl, .joblib) and ONNX (.onnx) formats.

    Args:
        url: The URL to download model from
        use_cache: Whether to use cached version

    Returns:
        Loaded model wrapped appropriately

    Raises:
        RuntimeError: If loading fails
    """
    file_content = download_file_from_url(url, use_cache=use_cache)

    # Detect format from URL extension (handle S3 presigned URLs)
    from urllib.parse import urlparse

    parsed_path = urlparse(url).path
    url_lower = parsed_path.lower()

    try:
        if url_lower.endswith(".onnx"):
            print("[MODEL] Detected ONNX format")
            return _load_onnx_model(file_content)
        elif url_lower.endswith((".pkl", ".pickle")):
            print("[MODEL] Detected pickle format")
            return pickle.loads(file_content)
        elif url_lower.endswith(".joblib"):
            print("[MODEL] Detected joblib format")
            # Save to temp file for joblib
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            try:
                model = joblib.load(tmp_path)
                return model
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            # Try pickle as default
            print("[MODEL] Unknown extension, trying pickle format")
            try:
                return pickle.loads(file_content)
            except Exception:
                print("[MODEL] Pickle failed, trying joblib format")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                try:
                    model = joblib.load(tmp_path)
                    return model
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed to load model from {url}: {str(e)}") from e


def _load_onnx_model(file_content: bytes) -> BaseEstimator:
    """
    Load ONNX model and wrap it in sklearn-compatible wrapper.

    Args:
        file_content: ONNX model file content

    Returns:
        ONNXModelWrapper instance

    Raises:
        RuntimeError: If ONNX loading fails
    """
    try:
        import onnxruntime as rt
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed. Install it to use ONNX models.") from exc

    try:
        # Import the wrapper class from functions.py
        from services.fairness.app.routers.functions import ONNXModelWrapper

        # Save to temp file for ONNX runtime
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            session = rt.InferenceSession(tmp_path)
            wrapper = ONNXModelWrapper(session)
            print("[ONNX] Successfully loaded ONNX model")
            return wrapper
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed to load ONNX model: {str(e)}") from e


def validate_file_metadata(metadata: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Validate and extract file URLs from S3 metadata.
    Expected order from backend: [test_data, train_data, model]

    Args:
        metadata: S3 metadata dictionary with 'files' and 'models' keys
        expected_order: Expected order of file types ['test', 'train', 'model']

    Returns:
        Tuple of (train_url, test_url, model_url) - any can be None

    Raises:
        RuntimeError: If validation fails or required files are missing
    """
    if not metadata:
        raise RuntimeError("S3 metadata is empty or None")

    files = metadata.get("files", [])
    models = metadata.get("models", [])

    if not files:
        raise RuntimeError("No files found in S3 metadata. Please upload data files first.")

    print(f"[VALIDATION] Found {len(files)} file(s) and {len(models)} model(s)")

    # Extract URLs based on expected backend order: [test, train, model]
    train_url = None
    test_url = None
    model_url = None

    # Files should be in order: test (index 0), train (index 1)
    if len(files) >= 1:
        test_url = files[0].get("url")
        print(f"[VALIDATION] Test data URL: {test_url}")

    if len(files) >= 2:
        train_url = files[1].get("url")
        print(f"[VALIDATION] Train data URL: {train_url}")

    # Model is separate
    if models and len(models) > 0:
        model_url = models[0].get("url")
        print(f"[VALIDATION] Model URL: {model_url}")

    # Validate that at least train data exists
    if not train_url:
        raise RuntimeError(
            "Training data is required but not found. Please ensure files are uploaded in correct order: [test, train, model]"
        )

    return train_url, test_url, model_url


def clear_file_cache():
    """
    Clear the file download cache.
    """
    _file_cache.clear()
    print("[CACHE] File cache cleared")


def get_s3_file_metadata(token: str, project_id: str):
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
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error processing external S3 API response: {e}")
        return None
