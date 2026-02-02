import hashlib
import json


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
