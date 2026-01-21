import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from services.what_if.app.schema.predict import PredictRequest, PredictResponse
from services.what_if.app.utils.feature_alignment import _ensure_feature_alignment
from services.what_if.app.utils.session_manager import SessionManager
from shared.auth import get_current_user

# Initialize session manager
session_manager = SessionManager()

router = APIRouter(prefix="/what_if", tags=["What If Analysis"])


@router.post("/predict/{session_id}", response_model=PredictResponse)
async def predict(session_id: str, request: PredictRequest, current_user: dict = Depends(get_current_user)):
    """Make predictions for given data"""
    try:
        preprocessor, explainer, model, metadata = session_manager.get_session_artifacts(session_id)

        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        print(f"🔍 Predict request - Input DF shape: {df.shape}")
        print(f"🔍 Predict request - Input columns: {list(df.columns)}")

        # Preprocess data
        X_processed = preprocessor.transform(df)
        print(f"🔍 Predict request - Processed shape: {X_processed.shape}")
        print(f"🔍 Predict request - Processed type: {type(X_processed)}")

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
                    print(f"⚠️ Could not get feature names from preprocessor: {e}")
                    processed_feature_names = metadata.get(
                        "processed_feature_names", [f"feature_{i}" for i in range(X_processed.shape[1])]
                    )
                    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
            else:
                processed_feature_names = metadata.get(
                    "processed_feature_names", [f"feature_{i}" for i in range(X_processed.shape[1])]
                )
                X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
        else:
            # For sklearn models that expect preprocessed names (with prefixes)
            if "processed_feature_names" in metadata:
                # Use stored processed feature names
                processed_feature_names = metadata["processed_feature_names"]
                X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
                print(
                    f"🔍 Sklearn model - Using stored processed feature names: {len(processed_feature_names)} features"
                )
            else:
                # Fallback - get from preprocessor
                if hasattr(preprocessor, "get_feature_names_out"):
                    try:
                        processed_feature_names = preprocessor.get_feature_names_out()
                        X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
                        print(
                            f"🔍 Sklearn model - Using preprocessor feature names: {len(processed_feature_names)} features"
                        )
                    except Exception as e:
                        print(f"⚠️ Could not get feature names from preprocessor: {e}")
                        processed_feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                        X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)
                else:
                    processed_feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)

        print(f"🔍 Feature names sample: {list(X_processed_df.columns)[:5]}")

        # Create DataFrame with appropriate feature names
        if len(list(X_processed_df.columns)) == X_processed.shape[1]:
            print(f"🔍 Final DF shape: {X_processed_df.shape}")
            print(f"🔍 Final DF columns sample: {list(X_processed_df.columns)[:5]}")
        else:
            print(f"⚠️ Feature count mismatch: {len(list(X_processed_df.columns))} vs {X_processed.shape[1]}")
            # Fallback to numpy array for model prediction
            X_processed_df = X_processed

        # Make predictions
        print(f"🔍 Predict request - Model type: {type(model)}")
        if hasattr(model, "predict_proba"):
            print("🔍 Predict request - Using predict_proba")
            predictions = model.predict_proba(X_processed_df)[:, 1].tolist()
        else:
            print("🔍 Predict request - Using predict")
            predictions = model.predict(X_processed_df).tolist()

        print(f"🔍 Predict request - Predictions: {predictions}")
        return {"status": "success", "session_id": session_id, "predictions": predictions}

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"❌ Prediction error: {str(e)}")
        print(f"❌ Full traceback: {error_details}")
        raise HTTPException(500, f"Prediction error: {str(e)}")
