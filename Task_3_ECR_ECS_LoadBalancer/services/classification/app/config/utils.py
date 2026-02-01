import hashlib
import json
import math
from typing import Any, Dict

import numpy as np
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from shared_migrations.models.analysis_result import AnalysisResult

SHAP_SAMPLE_SIZE = 100


# --- Utility Function for Error Handling ---
def sanitize_for_json(obj):
    """Recursively sanitize an object to ensure JSON serialization compatibility."""
    if obj is None:
        return None
    elif isinstance(obj, (bool, str)):
        return obj
    elif isinstance(obj, (int, float)):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    else:
        # For any other type, try to convert to string
        return str(obj)


# --- Utility Function for Error Handling ---
def handle_request(service_func, payload: Dict[str, Any], *args, **kwargs):
    import json
    import traceback

    try:
        result = service_func(payload, *args, **kwargs)

        # Sanitize the result before JSON serialization
        sanitized_result = sanitize_for_json(result)

        # Test JSON serialization
        try:
            json_str = json.dumps(sanitized_result)
            print(f"✅ Service result successfully serialized to JSON ({len(json_str)} characters)")
        except (TypeError, ValueError) as json_error:
            print(f"❌ JSON serialization error even after sanitization: {json_error}")
            print(f"🔍 Result type: {type(sanitized_result)}")

            # Find problematic parts in the sanitized result
            if isinstance(sanitized_result, dict):
                for key, value in sanitized_result.items():
                    try:
                        json.dumps({key: value})
                    except Exception as e:
                        print(f"❌ Problematic key '{key}': {value} (type: {type(value)}) - {e}")

            raise ValueError(f"Response contains values that cannot be serialized to JSON: {json_error}")

        return JSONResponse(status_code=200, content=sanitized_result)

    except ValueError as e:
        print(f"❌ ValueError in handle_request: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"💥 Unexpected error in handle_request: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


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

    # Optionally shorten it to first 16 or 20 characters
    return analysis_hash[:20]


def insert_analyzed_data_to_database(analysis_id, user_id, data_preview, analysis_tab, project_id, db):
    """
    Inserts analyzed data into the database.
    """
    new_entry = AnalysisResult(
        analysis_id=analysis_id,
        user_id=user_id,
        analysis_type="Classification",
        analysis_tab=analysis_tab,
        project_id=project_id,
        json_result=json.dumps(data_preview),
    )
    db.add(new_entry)
    db.commit()
