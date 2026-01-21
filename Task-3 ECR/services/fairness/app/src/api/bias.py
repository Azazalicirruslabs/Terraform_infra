import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from services.fairness.app.src.bias_detector import BiasDetector
from services.fairness.app.src.models import global_session
from services.fairness.app.src.schemas.request_response import BiasAnalysisRequest
from services.fairness.app.src.utils import FileManager
from services.fairness.app.utils.helper_functions import (
    get_s3_file_metadata,
    load_dataframe_from_url,
    load_model_from_url,
    validate_file_metadata,
)
from shared.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fairness/bias", tags=["Bias Detection"])


@router.post("/analyze")
async def analyze_bias(
    request: BiasAnalysisRequest, current_user: str = Depends(get_current_user), project_id: str = None
):
    """
    Analyze bias using fairness metrics with S3-based file loading.

    Args:
        request: BiasAnalysisRequest containing target_column, sensitive_feature_column,
                 and optional prediction_column, prediction_proba_column
        current_user: Authenticated user info
        project_id: Project ID to fetch S3 metadata

    Returns:
        Bias analysis results with metrics and recommendations
    """
    token = current_user.get("token")

    try:
        # 1. Get S3 metadata and validate
        print(f"\\n[BIAS ANALYSIS] Starting analysis for project {project_id}")
        s3_metadata = get_s3_file_metadata(token, project_id)

        if not s3_metadata:
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve S3 metadata. Please check your project ID and ensure files are uploaded.",
            )

        # 2. Validate and extract file URLs (order: train, test, model)
        train_url, test_url, model_url = validate_file_metadata(s3_metadata)

        print(f"[BIAS ANALYSIS] Using train data from: {train_url}")
        if test_url:
            print(f"[BIAS ANALYSIS] Test data available at: {test_url}")
        if model_url:
            print(f"[BIAS ANALYSIS] Model available at: {model_url}")

        # 3. Load Analysis Data (Prioritize TEST, fallback to TRAIN)
        analysis_df = None
        data_source = "train"

        if test_url:
            print("\n[STEP 1] Test data found. Loading TEST data for analysis...")
            try:
                analysis_df = load_dataframe_from_url(test_url)
                data_source = "test"
                print(f"[STEP 1] Loaded TEST data: shape={analysis_df.shape}, columns={list(analysis_df.columns)}")
            except Exception as e:
                print(f"[WARNING] Failed to load test data: {e}. Falling back to training data.")

        if analysis_df is None:
            print("\n[STEP 1] Loading TRAINING data for analysis (Fallback)...")
            analysis_df = load_dataframe_from_url(train_url)
            data_source = "train"
            print(f"[STEP 1] Loaded TRAIN data: shape={analysis_df.shape}, columns={list(analysis_df.columns)}")

        # 4. Validate target column
        if request.target_column not in analysis_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_column}' not found in dataset. Available columns: {list(analysis_df.columns)}",
            )

        # 5. Extract y_true (target values)
        print(f"\n[STEP 2] Extracting target column: {request.target_column}")
        y_true = analysis_df[request.target_column].values

        # 5a. Determine Task Type
        feature_types = request.feature_types or {}
        task_type = request.task_type

        if not task_type:
            # Infer task type
            is_numeric = np.issubdtype(y_true.dtype, np.number)
            unique_count = len(np.unique(y_true))
            if is_numeric and unique_count > 20:
                task_type = "regression"
                print(f"[STEP 2] Inferred task_type: regression (numeric target with {unique_count} unique values)")
            else:
                task_type = "classification"
                print(f"[STEP 2] Inferred task_type: classification (unique values: {unique_count})")
        else:
            print(f"[STEP 2] Using provided task_type: {task_type}")

        # 5b. Convert y_true based on task type
        if task_type == "regression":
            y_true = pd.to_numeric(y_true, errors="coerce")
            # Handle NaNs if any?
            if np.isnan(y_true).any():
                print("[WARNING] NaN values found in regression target, filling with mean")
                mask = np.isnan(y_true)
                y_true[mask] = np.nanmean(y_true)
        else:
            y_true = _safe_convert_to_numeric(y_true, "y_true")

        print(f"[STEP 2] y_true shape: {y_true.shape}, unique values: {len(np.unique(y_true))}")

        # 6. Validate sensitive feature
        if request.sensitive_feature_column not in analysis_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Sensitive feature '{request.sensitive_feature_column}' not found in dataset. Available columns: {list(analysis_df.columns)}",
            )

        # 7. Extract sensitive feature
        print(f"\n[STEP 3] Extracting sensitive feature: {request.sensitive_feature_column}")
        sensitive_feature = analysis_df[request.sensitive_feature_column].values
        print(
            f"[STEP 3] Original sensitive_feature type: {type(sensitive_feature[0])}, sample: {sensitive_feature[:5]}"
        )

        # Check explicit feature type for binning
        sensitive_type = feature_types.get(request.sensitive_feature_column)

        # Auto-detect if not provided
        if not sensitive_type:
            # 1. Check if boolean/binary (always categorical)
            if pd.api.types.is_bool_dtype(sensitive_feature) or len(np.unique(sensitive_feature)) <= 2:
                sensitive_type = "categorical"
                print(f"[STEP 3] Auto-detected sensitive_type: categorical (binary/boolean)")

            # 2. Check if numeric
            elif np.issubdtype(sensitive_feature.dtype, np.number) or pd.api.types.is_numeric_dtype(sensitive_feature):
                unique_count = len(np.unique(sensitive_feature))
                # Heuristic: If floats or cardinality > 10 -> Numerical (binning)
                # Note: education_num (16 values) should be numerical.
                is_actually_float = False
                if pd.api.types.is_float_dtype(sensitive_feature):
                    non_nan_feature = sensitive_feature[~np.isnan(sensitive_feature)]
                    if non_nan_feature.size > 0 and not np.all(np.mod(non_nan_feature, 1) == 0):
                        is_actually_float = True

                if is_actually_float:
                    sensitive_type = "numerical"
                    print(f"[STEP 3] Auto-detected sensitive_type: numerical (float data)")
                elif unique_count > 10:
                    sensitive_type = "numerical"
                    print(f"[STEP 3] Auto-detected sensitive_type: numerical (cardinality {unique_count} > 10)")
                else:
                    sensitive_type = "categorical"
                    print(f"[STEP 3] Auto-detected sensitive_type: categorical (cardinality {unique_count} <= 10)")
            else:
                sensitive_type = "categorical"
                print(f"[STEP 3] Auto-detected sensitive_type: categorical (non-numeric)")
        else:
            print(f"[STEP 3] Using provided sensitive_type: {sensitive_type}")

        if sensitive_type == "numerical":
            print(f"[STEP 3] Sensitive feature marked as 'numerical' - applying Quartile Binning (4 bins)")
            try:
                # Ensure numeric first
                sf_numeric = pd.to_numeric(sensitive_feature, errors="coerce")
                # QCut
                sensitive_feature = pd.qcut(sf_numeric, q=4, labels=False, duplicates="drop")
                print(f"[STEP 3] Binning successful. Categories: {np.unique(sensitive_feature)}")
            except Exception as e:
                print(f"[STEP 3] Binning failed ({str(e)}), falling back to categorical conversion")

        sensitive_feature = _safe_convert_to_numeric(sensitive_feature, request.sensitive_feature_column)
        print(
            f"[STEP 3] Converted sensitive_feature shape: {sensitive_feature.shape}, unique: {np.unique(sensitive_feature)}"
        )

        # 8. Get predictions
        print("\\n[STEP 4] Obtaining predictions...")
        y_pred = None

        # Option A: Use prediction column from dataset
        if request.prediction_column:
            if request.prediction_column not in analysis_df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Prediction column '{request.prediction_column}' not found in dataset. Available columns: {list(analysis_df.columns)}",
                )
            print(f"[STEP 4] Using prediction column: {request.prediction_column}")
            y_pred = analysis_df[request.prediction_column].values
            if task_type == "regression":
                y_pred = pd.to_numeric(y_pred, errors="coerce")
            else:
                y_pred = _safe_convert_to_numeric(y_pred, request.prediction_column)

        # Option B: Load model and generate predictions
        elif model_url:
            print(f"[STEP 4] Loading model from: {model_url}")
            model = load_model_from_url(model_url)

            # Prepare features (remove target and sensitive columns)
            X = analysis_df.drop(columns=[request.target_column], errors="ignore").copy()

            # Encode categorical columns if needed
            print(f"[STEP 4] Encoding features for model prediction")
            print(f"[STEP 4] X dtypes before encoding: {X.dtypes.unique()}")
            X_encoded, encoders = _encode_dataframe(X)
            print(f"[STEP 4] X dtypes after encoding: {X_encoded.dtypes.unique()}")
            print(f"[STEP 4] X shape: {X_encoded.shape}")

            # Store encoders in global_session as fallback
            if not hasattr(global_session, "encoders"):
                global_session.encoders = encoders

            # Generate predictions
            y_pred = model.predict(X_encoded)
            if task_type == "regression":
                y_pred = pd.to_numeric(y_pred, errors="coerce")
            else:
                y_pred = _safe_convert_to_numeric(y_pred, "model_prediction")
            print(f"[STEP 4] Generated predictions from model: shape={y_pred.shape}, unique={np.unique(y_pred)}")

        else:
            # Fallback to global_session if available
            if global_session.model_file_path:
                print(f"[STEP 4] Using fallback model from global_session")
                model = FileManager.load_model(global_session.model_file_path)
                X = analysis_df.drop(columns=[request.target_column], errors="ignore").copy()
                X_encoded, encoders = _encode_dataframe(X)

                if not hasattr(global_session, "encoders"):
                    global_session.encoders = encoders

                y_pred = model.predict(X_encoded)
                if task_type == "regression":
                    y_pred = pd.to_numeric(y_pred, errors="coerce")
                else:
                    y_pred = _safe_convert_to_numeric(y_pred, "model_prediction")
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either prediction_column or model URL is required. Please provide predictions or upload a model.",
                )

        # 9. Get prediction probabilities if available
        print("\\n[STEP 5] Checking for prediction probabilities...")
        y_pred_proba = None
        if request.prediction_proba_column:
            if request.prediction_proba_column in analysis_df.columns:
                try:
                    y_pred_proba = analysis_df[request.prediction_proba_column].values.astype(float)
                    print(f"[STEP 5] Using prediction probabilities from column: {request.prediction_proba_column}")
                except Exception as e:
                    print(f"[STEP 5] Failed to load prediction probabilities: {e}")
                    y_pred_proba = None

        # 10. Validate array shapes
        print(f"\\n[VALIDATION] Final array shapes:")
        print(f"  y_true: {y_true.shape}, unique values: {np.unique(y_true)}")
        print(f"  y_pred: {y_pred.shape}, unique values: {np.unique(y_pred)}")
        print(f"  sensitive_feature: {sensitive_feature.shape}, unique values: {np.unique(sensitive_feature)}")
        if y_pred_proba is not None:
            print(f"  y_pred_proba: {y_pred_proba.shape}")

        # 11. Run bias detection
        print("\\n[STEP 6] Running bias detection metrics...")
        detector = BiasDetector()
        results = detector.run_all_metrics(y_true, y_pred, sensitive_feature, y_pred_proba, task_type=task_type)

        # 12. Clean NaN values from results
        results = _clean_nan_values(results)

        # 13. Generate recommendations
        print("\\n[STEP 7] Generating recommendations...")
        recommendations = _generate_recommendations(results)

        # 13a. Generate Feature Type Map for ALL columns
        feature_map = {}
        for col in analysis_df.columns:
            # Re-use the heuristic
            if col in (request.feature_types or {}):
                feature_map[col] = request.feature_types[col]
            else:
                col_data = analysis_df[col]
                # 1. Binary/Bool -> Categorical
                if pd.api.types.is_bool_dtype(col_data) or len(col_data.unique()) <= 2:
                    feature_map[col] = "categorical"
                    continue

                # 2. Numeric logic
                is_num = np.issubdtype(col_data.dtype, np.number) or pd.api.types.is_numeric_dtype(col_data)
                if is_num:
                    n_unique = len(col_data.unique())
                    # Float check
                    is_float = pd.api.types.is_float_dtype(col_data) and not np.all(np.mod(col_data.dropna(), 1) == 0)

                    if is_float or n_unique > 10:
                        feature_map[col] = "numerical"
                    else:
                        feature_map[col] = "categorical"
                else:
                    feature_map[col] = "categorical"

        # 14. Return comprehensive results
        return {
            "status": "success",
            "analysis_type": "Fairness Bias Detection",
            "project_id": project_id,
            "target_column": request.target_column,
            "sensitive_feature": request.sensitive_feature_column,
            "sensitive_feature_type": sensitive_type,
            "data_info": {
                "dataset_source": data_source,
                "samples": len(analysis_df),
                "features": list(analysis_df.columns),
                "feature_types": feature_map,
                "used_model": model_url is not None or request.prediction_column is not None,
            },
            "total_metrics": results["summary"]["total_metrics"],
            "biased_metrics_count": results["summary"]["biased_metrics_count"],
            "overall_bias_status": results["summary"]["overall_bias_status"],
            "metrics_results": {k: v for k, v in results.items() if k != "summary"},
            "recommendations": recommendations,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error in analyze_bias: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing bias: {str(e)}")


# Keep the old commented code as reference for now
# OLD IMPLEMENTATION (commented out):
# try:
#     if global_session.train_file_path is None:
#         raise HTTPException(status_code=400, detail="Training data must be uploaded first")

#     # Load data
#     train_df = FileManager.load_csv(global_session.train_file_path)

#     # Get y_true
#     if global_session.target_column is None:
#         raise HTTPException(status_code=400, detail="Target column must be set first")

#     y_true = train_df[global_session.target_column].values
#     y_true = _safe_convert_to_numeric(y_true, "y_true")

#     # Get sensitive feature
#     if request.sensitive_feature_column not in train_df.columns:
#         raise HTTPException(
#             status_code=400, detail=f"Sensitive feature '{request.sensitive_feature_column}' not found"
#         )

#     sensitive_feature = train_df[request.sensitive_feature_column].values
#     print(
#         f"\n[DEBUG] Original sensitive_feature type: {type(sensitive_feature[0])}, sample values: {sensitive_feature[:5]}"
#     )
#     sensitive_feature = _safe_convert_to_numeric(sensitive_feature, request.sensitive_feature_column)
#     print(f"[DEBUG] Converted sensitive_feature: {sensitive_feature[:5]}")

#     # Get predictions
#     if request.prediction_column:
#         if request.prediction_column not in train_df.columns:
#             raise HTTPException(
#                 status_code=400, detail=f"Prediction column '{request.prediction_column}' not found"
#             )
#         y_pred = train_df[request.prediction_column].values
#         y_pred = _safe_convert_to_numeric(y_pred, request.prediction_column)

#     elif global_session.model_file_path:
#         # Use uploaded model to predict
#         model = FileManager.load_model(global_session.model_file_path)
#         X = train_df.drop(columns=[global_session.target_column]).copy()

#         # Encode categorical columns in X (same way model expects)
#         print(f"\n[DEBUG] Encoding X data for model prediction")
#         print(f"  X dtypes before: {X.dtypes.unique()}")

#         X, encoders = _encode_dataframe(X)

#         print(f"  X dtypes after: {X.dtypes.unique()}")
#         print(f"  X shape: {X.shape}")

#         # Optionally store encoders for consistency
#         if not hasattr(global_session, "encoders"):
#             global_session.encoders = encoders

#         y_pred = model.predict(X)
#         y_pred = _safe_convert_to_numeric(y_pred, "model_prediction")
#     else:
#         raise HTTPException(status_code=400, detail="Either prediction_column or uploaded model required")

#     # Get prediction probabilities if available
#     y_pred_proba = None
#     if request.prediction_proba_column:
#         if request.prediction_proba_column in train_df.columns:
#             try:
#                 y_pred_proba = train_df[request.prediction_proba_column].values.astype(float)
#             except:
#                 y_pred_proba = None

#     print(f"\n[INFO] Bias Analysis Input:")
#     print(f"  y_true shape: {y_true.shape}, unique values: {np.unique(y_true)}")
#     print(f"  y_pred shape: {y_pred.shape}, unique values: {np.unique(y_pred)}")
#     print(f"  sensitive_feature shape: {sensitive_feature.shape}, unique values: {np.unique(sensitive_feature)}")

#     # Run bias detection
#     detector = BiasDetector()
#     results = detector.run_all_metrics(y_true, y_pred, sensitive_feature, y_pred_proba)

#     # Clean NaN values from results
#     results = _clean_nan_values(results)

#     # Generate recommendations
#     recommendations = _generate_recommendations(results)

#     return {
#         "status": "success",
#         "analysis_type": "Fairness Bias Detection",
#         "sensitive_feature": request.sensitive_feature_column,
#         "total_metrics": results["summary"]["total_metrics"],
#         "biased_metrics_count": results["summary"]["biased_metrics_count"],
#         "overall_bias_status": results["summary"]["overall_bias_status"],
#         "metrics_results": {k: v for k, v in results.items() if k != "summary"},
#         "recommendations": recommendations,
#     }

# except HTTPException:
#     raise
# except Exception as e:
#     print(f"[ERROR] Error in analyze_bias: {str(e)}")
#     import traceback

#     traceback.print_exc()
#     raise HTTPException(status_code=500, detail=f"Error analyzing bias: {str(e)}")


@router.get("/thresholds")
async def get_bias_thresholds(current_user: str = Depends(get_current_user), project_id: str = None):
    """Get current bias detection thresholds"""
    token = current_user.get("token")
    get_s3_file_metadata(token, project_id)
    detector = BiasDetector()
    return {
        "status": "success",
        "thresholds": detector.thresholds,
        "descriptions": {
            "statistical_parity": "Max acceptable difference in positive prediction rate",
            "disparate_impact": "Min acceptable ratio (80% rule)",
            "equal_opportunity": "Max acceptable difference in TPR",
            "equalized_odds": "Max acceptable difference in TPR and FPR",
            "calibration": "Max acceptable calibration error",
            "generalized_entropy_index": "Max acceptable entropy index",
        },
    }


def _safe_convert_to_numeric(data: np.ndarray, name: str) -> np.ndarray:
    """
    Safely convert any data to numeric integers.
    - Preserves order-based mapping for categorical values (first-seen -> 0..n-1).
    - Handles booleans, ints, floats, strings and mixed object arrays.
    - Raises a ValueError if conversion fails.
    """
    logger.info(f"[CONVERT] Converting {name}")

    try:
        # Already numeric floats/ints -> cast to int safely
        if np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating):
            return data.astype(int)

        # Boolean -> int
        if data.dtype == bool or (data.dtype == np.bool_):
            return data.astype(int)

        # Object - possibly strings / mixed
        if data.dtype == object or data.dtype == "O":

            # Try direct numeric conversion first
            try:
                numeric = pd.to_numeric(data, errors="raise")
                return numeric.astype(int).to_numpy()
            except Exception:
                # Fall back to stable categorical mapping (preserve first-seen order)
                mapping = {}
                mapped = []
                next_idx = 0
                for val in data:
                    key = val if not (isinstance(val, str) and val.strip() == "") else "__MISSING__"
                    if key not in mapping:
                        mapping[key] = next_idx
                        next_idx += 1
                    mapped.append(mapping[key])
                logger.debug(f"[CONVERT] {name} categorical mapping: {mapping}")
                return np.array(mapped, dtype=int)

        # As a last resort, attempt converting element-wise
        return np.array([int(x) for x in data], dtype=int)

    except Exception as e:
        logger.exception(f"Conversion failed for {name}")
        raise ValueError(f"Could not convert {name} to numeric: {str(e)}")


def _encode_dataframe(df: pd.DataFrame, saved_encoders: Dict[str, Dict] = None) -> pd.DataFrame:
    """
    Encode categorical columns to numeric in a stable, reproducible way.
    - If `saved_encoders` provided (dict of col -> mapping), reuse them.
      Otherwise, create mappings where needed (preserving first-seen order).
    - Returns df_encoded (numeric) and *does not* alter original df.
    """
    df_encoded = df.copy()
    encoders = saved_encoders or {}

    for col in df_encoded.columns:
        # If dtype numeric already, ensure numeric and fill NAs
        if df_encoded[col].dtype == "object" or df_encoded[col].dtype.name == "category":
            df_encoded[col] = df_encoded[col].astype(str)

            if col in encoders:
                # Use saved encoder
                le = encoders[col]
                # Handle unseen labels
                df_encoded[col] = df_encoded[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            else:
                # Create new encoder
                from sklearn.preprocessing import LabelEncoder

                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                encoders[col] = le

        # CRITICAL: Ensure all columns are 1D numeric arrays
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors="coerce").fillna(0)

    # VALIDATE: Check for homogeneous shape
    try:
        test_array = df_encoded.values
        if test_array.dtype == "object":
            # Convert object arrays to float
            df_encoded = df_encoded.astype(float)
    except Exception as e:
        logger.error(f"Error converting to numpy array: {e}")
        # Force conversion column by column
        for col in df_encoded.columns:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors="coerce").fillna(0)

    return df_encoded, encoders


def _clean_nan_values(obj):
    """Recursively remove NaN values from nested dicts/lists"""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if isinstance(v, float) and np.isnan(v):
                cleaned[k] = 0.0
            elif isinstance(v, (dict, list)):
                cleaned[k] = _clean_nan_values(v)
            else:
                cleaned[k] = v
        return cleaned
    elif isinstance(obj, list):
        return [_clean_nan_values(item) for item in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return 0.0
    return obj


def _generate_recommendations(results: Dict) -> List[str]:
    """Generate recommendations based on bias analysis"""
    recommendations = []

    if results["summary"]["overall_bias_status"] == "FAIR":
        recommendations.append("✅ Model appears fair across selected metrics")
        return recommendations

    biased = results["summary"]["biased_metrics"]

    metrics_results = {k: v for k, v in results.items() if k != "summary"}

    if "statistical_parity" in biased:
        sp = metrics_results.get("statistical_parity", {})
        recommendations.append(
            f"⚠️ Statistical Parity violated (diff: {sp.get('difference', 0):.3f}) - Different prediction rates between groups. Consider resampling or reweighting data."
        )

    if "disparate_impact" in biased:
        di = metrics_results.get("disparate_impact", {})
        recommendations.append(
            f"⚠️ Disparate Impact detected (ratio: {di.get('ratio', 0):.3f}) - Selection rate ratio < 0.80. Review decision thresholds."
        )

    if "equal_opportunity" in biased:
        eo = metrics_results.get("equal_opportunity", {})
        if eo.get("status") != "skipped":
            recommendations.append(
                f"⚠️ Equal Opportunity violated - True Positive Rates differ. Adjust decision thresholds."
            )

    if "equalized_odds" in biased:
        eq = metrics_results.get("equalized_odds", {})
        if eq.get("status") != "skipped":
            recommendations.append(f"⚠️ Equalized Odds violated - Apply fairness-aware post-processing.")

    if "calibration" in biased:
        metrics_results.get("calibration", {})
        recommendations.append(
            f"⚠️ Calibration error - Predicted probabilities differ from actual rates. Recalibrate model."
        )

    if "generalized_entropy_index" in biased:
        recommendations.append(f"⚠️ High inequality in predictions - Apply fairness constraints during training.")

    return recommendations
