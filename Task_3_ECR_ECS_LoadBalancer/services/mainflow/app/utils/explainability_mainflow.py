"""
Utility functions for explainability analysis.
"""

import math
from typing import Any

import numpy as np


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object to ensure JSON serialization compatibility.

    Args:
        obj: Object to sanitize

    Returns:
        JSON-serializable version of the object
    """
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
