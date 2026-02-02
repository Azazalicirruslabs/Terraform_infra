import joblib
from fastapi import APIRouter, Depends, HTTPException

from services.what_if.app.schema.features import ResponseModel
from services.what_if.app.utils.session_manager import SessionManager
from shared.auth import get_current_user

session_manager = SessionManager()

router = APIRouter(prefix="/what_if", tags=["What If Analysis"])


@router.get("/features/{session_id}", response_model=ResponseModel)
async def get_feature_info(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get detailed feature information for What-If analysis"""
    try:
        # Get the session artifacts and metadata
        session_path = session_manager.sessions_dir / session_id
        metadata = joblib.load(session_path / "metadata.pkl")

        # Check if we have pre-computed feature info in metadata
        if "feature_info" in metadata:
            return {"feature_info": metadata["feature_info"], "session_id": session_id}

        # Fallback: Use background data to estimate feature ranges
        background_data_path = session_path / "shap_background.pkl"
        if background_data_path.exists():
            background_data = joblib.load(background_data_path)
            print(f"Using background data ({len(background_data)} samples) for feature info")

            feature_info = {}

            for column in background_data.columns:
                col_info = {
                    "name": column,
                    "type": "numeric",  # Background data is already preprocessed, so all numeric
                }

                # Get statistics from background data
                col_values = background_data[column].dropna()
                if len(col_values) > 0:
                    col_info.update(
                        {
                            "min": float(col_values.min()),
                            "max": float(col_values.max()),
                            "mean": float(col_values.mean()),
                            "median": float(col_values.median()),
                            "std": float(col_values.std()),
                        }
                    )
                else:
                    # Default values if no data available
                    col_info.update({"min": 0.0, "max": 1.0, "mean": 0.5, "median": 0.5, "std": 0.25})

                feature_info[column] = col_info

            return {"feature_info": feature_info, "session_id": session_id}
        else:
            # Final fallback: create default feature info based on metadata
            feature_info = {}
            feature_names = metadata.get("feature_names", [])

            for feature_name in feature_names:
                feature_info[feature_name] = {
                    "name": feature_name,
                    "type": "numeric",
                    "min": 0.0,
                    "max": 1.0,
                    "mean": 0.5,
                    "median": 0.5,
                    "std": 0.25,
                }

            return {"feature_info": feature_info, "session_id": session_id}

    except Exception as e:
        import traceback

        print(f"Feature info error: {traceback.format_exc()}")
        raise HTTPException(500, f"Feature info error: {str(e)}")
