import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from services.what_if.app.schema.shap import ShapeRequest, ShapResponse
from services.what_if.app.utils.feature_alignment import _ensure_feature_alignment
from services.what_if.app.utils.session_manager import SessionManager
from shared.auth import get_current_user

# Initialize session manager
session_manager = SessionManager()

router = APIRouter(prefix="/what_if", tags=["What If Analysis"])


@router.post("/shap/{session_id}", response_model=ShapResponse)
async def get_shap_values(session_id: str, request: ShapeRequest, current_user: dict = Depends(get_current_user)):
    """Get SHAP values for given data"""
    try:
        preprocessor, explainer, model, metadata = session_manager.get_session_artifacts(session_id)

        # Debug information
        print(f"🔍 SHAP request - Explainer type: {type(explainer)}")
        print(f"🔍 SHAP request - Explainer is None: {explainer is None}")
        print(f"🔍 SHAP request - Metadata shap_available: {metadata.get('shap_available', 'NOT_SET')}")
        print(f"🔍 SHAP request - Model type: {type(model)}")

        # Check if SHAP is available
        if explainer is None or not metadata.get("shap_available", True):
            print(
                f"❌ SHAP not available - explainer is None: {explainer is None}, shap_available: {metadata.get('shap_available', True)}"
            )
            raise HTTPException(
                503, "SHAP explainer is not available for this session. This may be due to model compatibility issues."
            )

        print(f"✅ SHAP explainer available, proceeding with calculation...")

        # Convert and preprocess data
        df = pd.DataFrame(request.data)
        X_processed = preprocessor.transform(df)

        # Convert to DataFrame with proper feature names based on model type and metadata
        model_type = metadata.get("model_type", "sklearn")
        uses_clean_names = metadata.get("uses_clean_names", False)

        if model_type == "onnx" or uses_clean_names:
            # For ONNX models or sklearn models that expect clean names
            if hasattr(preprocessor, "get_feature_names_out"):
                try:
                    processed_feature_names = preprocessor.get_feature_names_out()
                    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)

                    # Apply feature alignment to get clean names
                    X_processed_df = _ensure_feature_alignment(X_processed_df)

                except Exception as e:
                    processed_feature_names = metadata.get(
                        "processed_feature_names", [f"feature_{i}" for i in range(X_processed.shape[1])]
                    )
                    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
            else:
                processed_feature_names = metadata.get(
                    "processed_feature_names", [f"feature_{i}" for i in range(X_processed.shape[1])]
                )
                X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)

            # Use the cleaned feature names for SHAP display
            feature_names_for_display = list(X_processed_df.columns)
        else:
            # For sklearn models that expect preprocessed names (with prefixes)
            if "processed_feature_names" in metadata:
                # Use stored processed feature names
                processed_feature_names = metadata["processed_feature_names"]
                X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
            else:
                # Fallback - get from preprocessor
                if hasattr(preprocessor, "get_feature_names_out"):
                    try:
                        processed_feature_names = preprocessor.get_feature_names_out()
                        X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
                    except Exception as e:
                        processed_feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                        X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
                else:
                    processed_feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)

            # Use the original feature names for display (cleaner for UI)
            feature_names_for_display = metadata.get("feature_names", list(X_processed_df.columns))

        if len(list(X_processed_df.columns)) == X_processed.shape[1]:
            pass  # X_processed_df is already correctly created
        else:
            # Fallback to numpy array if feature count doesn't match
            X_processed_df = X_processed

        # Calculate SHAP values
        shap_values = explainer(X_processed_df)

        # Handle different SHAP explainer types
        if hasattr(shap_values, "values"):
            values = shap_values.values
            base_values = shap_values.base_values
            data = shap_values.data
        else:
            # For older SHAP versions or different explainer types
            values = shap_values
            base_values = explainer.expected_value
            data = X_processed

        # Ensure proper shape and format
        if len(values.shape) == 3:  # Multi-class case
            values = values[:, :, 1]  # Take positive class for binary classification

        # Handle base_values format for different cases
        if isinstance(base_values, np.ndarray):
            if base_values.ndim > 1:
                # Multi-class case - take positive class
                base_values = base_values[:, 1] if base_values.shape[1] > 1 else base_values.flatten()
            elif base_values.ndim == 1 and len(base_values) > len(values):
                # If base_values has more elements than samples, take positive class
                base_values = base_values[1] if len(base_values) == 2 else base_values[0]
                base_values = [base_values] * len(values)
            else:
                base_values = base_values.flatten()
        elif isinstance(base_values, (list, tuple)):
            if len(base_values) > 0 and isinstance(base_values[0], (list, tuple, np.ndarray)):
                # Nested list case - flatten and take positive class if binary
                if len(base_values[0]) == 2:
                    base_values = [bv[1] for bv in base_values]
                else:
                    base_values = [bv[0] for bv in base_values]
            elif len(base_values) == 2 and len(values) > 2:
                # Binary base_values but multiple samples
                base_values = [base_values[1]] * len(values)
        elif isinstance(base_values, (int, float)):
            base_values = [float(base_values)] * len(values)

        # Ensure base_values is the right length
        if len(base_values) != len(values):
            if len(base_values) == 1:
                base_values = base_values * len(values)
            else:
                base_values = [base_values[0]] * len(values)

        # Get feature names for the response
        if len(feature_names_for_display) == values.shape[1]:
            response_feature_names = feature_names_for_display
        else:
            response_feature_names = [f"feature_{i}" for i in range(values.shape[1])]

        return ShapResponse(
            values=values.tolist(),
            base_values=[float(bv) for bv in base_values],
            data=data.tolist(),
            feature_names=response_feature_names,
        )

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"SHAP calculation error: {str(e)}")
        print(f"Full traceback: {error_details}")
        raise HTTPException(500, f"SHAP calculation error: {str(e)}")
