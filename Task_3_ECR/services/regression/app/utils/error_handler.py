import math
import traceback

from fastapi import HTTPException
from fastapi.responses import JSONResponse


# --- Utility Function for Error Handling ---
def handle_request(service_func, *args, **kwargs):
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        else:
            return obj

    try:
        result = service_func(*args, **kwargs)
        sanitized_result = sanitize(result)
        return JSONResponse(status_code=200, content=sanitized_result)
    except ValueError as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
