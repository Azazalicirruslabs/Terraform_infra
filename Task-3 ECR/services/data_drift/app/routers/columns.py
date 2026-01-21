import logging
import os
from io import BytesIO

import chardet
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import APIRouter, Depends

from services.data_drift.app.routers.datadrift import DriftAnalyzer
from shared.auth import get_current_user

load_dotenv()

logging.basicConfig(level=logging.INFO)

DriftAnalyzer = DriftAnalyzer()
columns = DriftAnalyzer.preprocess

router = APIRouter(prefix="/data_drift", tags=["List of Columns"])


def load_flexible_csv(file_bytes):
    """
    Try loading a CSV file by detecting encoding and trying multiple delimiters.
    """
    encoding = chardet.detect(file_bytes)["encoding"]

    # Try comma-separated files first
    try:
        df = pd.read_csv(BytesIO(file_bytes), encoding=encoding)
        if df.shape[1] == 1:
            raise ValueError("Likely wrong delimiter, trying semicolon...")
        return df
    except Exception:
        try:
            df = pd.read_csv(BytesIO(file_bytes), sep=";", encoding=encoding)
            if df.shape[1] == 1:
                raise ValueError("Still only 1 column after fallback. Malformed file?")
            return df
        except Exception as e:
            raise RuntimeError(f"❌ Could not parse the CSV. Reason: {e}")


@router.get("/columns/{analysis_type}/{project_id}")
def s3_bucket_data(analysis_type: str, project_id: str, current_user: dict = Depends(get_current_user)):
    FILE_MODEL_DOWNLOAD_API = os.getenv("FILES_API_BASE_URL")
    api_url = f"{FILE_MODEL_DOWNLOAD_API}/{analysis_type}/{project_id}"
    access_token = current_user.get("token")
    headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        return {"error": f"Failed to connect to the API. Status code: {response.status_code}"}

    try:
        json_data = response.json()
        files = json_data.get("files", [])
        if not files:
            return {"message": "No files found."}

        file_columns = {}

        for file_info in files:
            file_name = file_info["file_name"]
            file_url = file_info["url"]

            file_response = requests.get(file_url)
            if file_response.status_code != 200:
                file_columns[file_name] = f"Failed to download. Status code: {file_response.status_code}"
                continue

            file_bytes = file_response.content
            try:
                # Handle different formats
                if file_name.lower().endswith((".csv", ".txt")):
                    df = load_flexible_csv(file_bytes)

                elif file_name.lower().endswith((".xls", ".xlsx")):
                    df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")

                else:
                    file_columns[file_name] = "Unsupported file type."
                    continue

                # Keep all columns, including unnamed
                file_columns[file_name] = df.columns.tolist()

            except Exception as e:
                file_columns[file_name] = f"Error reading file: {str(e)}"
            break  # Process only the first file for now
        return file_columns

    except Exception as e:
        logging.exception("Error processing API response")
        return {"error": "Failed to process API response. Please try again later."}
