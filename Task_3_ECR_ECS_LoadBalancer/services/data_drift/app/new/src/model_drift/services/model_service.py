import io
import logging
import pickle
import warnings
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import UploadFile
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")


def preprocess_data(
    X_ref: pd.DataFrame, X_curr: pd.DataFrame, y_ref: pd.Series, y_curr: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Comprehensive data preprocessing for both categorical and numerical features

    Args:
        X_ref: Reference features
        X_curr: Current features
        y_ref: Reference target
        y_curr: Current target

    Returns:
        Tuple of processed (X_ref, X_curr, y_ref, y_curr)
    """
    try:
        # Make copies to avoid modifying original data
        X_ref_processed = X_ref.copy()
        X_curr_processed = X_curr.copy()
        y_ref_processed = y_ref.copy()
        y_curr_processed = y_curr.copy()

        # 1. Handle missing values
        # For numerical columns
        numerical_cols = X_ref_processed.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            imputer_num = SimpleImputer(strategy="median")
            X_ref_processed[numerical_cols] = imputer_num.fit_transform(X_ref_processed[numerical_cols])
            X_curr_processed[numerical_cols] = imputer_num.transform(X_curr_processed[numerical_cols])

        # For categorical columns
        categorical_cols = X_ref_processed.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy="most_frequent")
            X_ref_processed[categorical_cols] = imputer_cat.fit_transform(X_ref_processed[categorical_cols])
            X_curr_processed[categorical_cols] = imputer_cat.transform(X_curr_processed[categorical_cols])

        # 2. Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on combined data to ensure consistent encoding
            combined_values = pd.concat([X_ref_processed[col], X_curr_processed[col]]).astype(str)
            le.fit(combined_values)

            X_ref_processed[col] = le.transform(X_ref_processed[col].astype(str))
            X_curr_processed[col] = le.transform(X_curr_processed[col].astype(str))
            label_encoders[col] = le

        # 3. Handle target variable encoding if it's categorical
        if y_ref_processed.dtype == "object" or y_ref_processed.dtype.name == "category":
            target_encoder = LabelEncoder()
            combined_targets = pd.concat([y_ref_processed, y_curr_processed]).astype(str)
            target_encoder.fit(combined_targets)

            y_ref_processed = target_encoder.transform(y_ref_processed.astype(str))
            y_curr_processed = target_encoder.transform(y_curr_processed.astype(str))

        # 4. Scale numerical features (optional, but often helpful)
        if len(numerical_cols) > 0:
            scaler = RobustScaler()  # More robust to outliers than StandardScaler
            X_ref_processed[numerical_cols] = scaler.fit_transform(X_ref_processed[numerical_cols])
            X_curr_processed[numerical_cols] = scaler.transform(X_curr_processed[numerical_cols])

        # Convert to numpy arrays
        X_ref_final = X_ref_processed.values.astype(np.float64)
        X_curr_final = X_curr_processed.values.astype(np.float64)
        y_ref_final = np.array(y_ref_processed, dtype=np.float64)
        y_curr_final = np.array(y_curr_processed, dtype=np.float64)

        return X_ref_final, X_curr_final, y_ref_final, y_curr_final

    except Exception as e:
        raise ValueError(f"Data preprocessing failed: {str(e)}")


async def run_model_drift(
    reference_data: UploadFile, current_data: UploadFile, model_file: UploadFile
) -> Dict[str, Any]:
    """
    Run model drift analysis comparing model performance on reference vs current data

    Args:
        reference_data: CSV file with reference dataset
        current_data: CSV file with current dataset
        model_file: Pickle file containing the trained model

    Returns:
        Dictionary containing model drift analysis results
    """
    try:
        # Read the CSV files
        ref_content = await reference_data.read()
        curr_content = await current_data.read()
        model_content = await model_file.read()

        # Load data into pandas DataFrames with error handling
        try:
            ref_df = pd.read_csv(io.StringIO(ref_content.decode("utf-8")))
            if ref_df.empty:
                raise ValueError("Reference dataset is empty")
        except Exception as e:
            raise ValueError(f"Failed to load reference data: {str(e)}")

        try:
            curr_df = pd.read_csv(io.StringIO(curr_content.decode("utf-8")))
            if curr_df.empty:
                raise ValueError("Current dataset is empty")
        except Exception as e:
            raise ValueError(f"Failed to load current data: {str(e)}")

        # Validate datasets have same columns (excluding target)
        if list(ref_df.columns) != list(curr_df.columns):
            raise ValueError(
                f"Column mismatch between datasets. Reference: {list(ref_df.columns)}, Current: {list(curr_df.columns)}"
            )

        if len(ref_df.columns) < 2:
            raise ValueError(f"Dataset must have at least 2 columns (features + target). Found: {len(ref_df.columns)}")

        # Load the model - support both pickle and joblib formats
        model_filename = model_file.filename.lower()
        try:
            if model_filename.endswith((".joblib", ".pkl.joblib")):
                # Try loading with joblib
                model = joblib.load(io.BytesIO(model_content))
            elif model_filename.endswith((".pkl", ".pickle")):
                # Try loading with pickle
                model = pickle.loads(model_content)
            else:
                # Try both methods for unknown extensions
                try:
                    model = pickle.loads(model_content)
                except:
                    model = joblib.load(io.BytesIO(model_content))
        except Exception as e:
            raise ValueError(f"Could not load model file. Supported formats: .pkl, .pickle, .joblib. Error: {str(e)}")

        # Validate model object
        if not hasattr(model, "predict"):
            # Handle complex model structures (e.g., dictionary with model key)
            if isinstance(model, dict):
                # Look for common model keys
                for key in [
                    "model",
                    "classifier",
                    "regressor",
                    "estimator",
                    "best_estimator_",
                    "final_estimator",
                    "named_steps",
                ]:
                    if key in model and hasattr(model[key], "predict"):
                        model = model[key]
                        break
                    elif key == "named_steps" and isinstance(model[key], dict):
                        # Handle pipeline named_steps
                        for step_name, step_model in model[key].items():
                            if hasattr(step_model, "predict"):
                                model = step_model
                                break
                else:
                    # If still no predict method found, list available keys
                    available_keys = list(model.keys()) if isinstance(model, dict) else []
                    raise ValueError(f"Model object does not have 'predict' method. Available keys: {available_keys}")
            else:
                raise ValueError(f"Model object does not have 'predict' method. Type: {type(model)}")

        print(f"Model validation successful. Model type: {type(model)}")

        # Check if model is a pipeline or has preprocessing steps
        is_pipeline = hasattr(model, "named_steps") or hasattr(model, "steps")
        print(f"Model is pipeline: {is_pipeline}")

        # Assume last column is target variable
        ref_X = ref_df.iloc[:, :-1]
        ref_y = ref_df.iloc[:, -1]
        curr_X = curr_df.iloc[:, :-1]
        curr_y = curr_df.iloc[:, -1]

        print(f"Data shapes - Ref: {ref_X.shape}, Curr: {curr_X.shape}")
        print(
            f"Feature types - Ref numerical: {len(ref_X.select_dtypes(include=[np.number]).columns)}, categorical: {len(ref_X.select_dtypes(include=['object']).columns)}"
        )

        # Check if model is a pipeline or has preprocessing steps
        is_pipeline = hasattr(model, "named_steps") or hasattr(model, "steps")
        print(f"Model is pipeline: {is_pipeline}")

        # Make predictions with different strategies based on model type
        try:
            if is_pipeline:
                # Pipeline models often include preprocessing, try raw data first
                print("Trying with raw data (pipeline detected)...")
                ref_predictions = model.predict(ref_X)
                curr_predictions = model.predict(curr_X)
                y_ref_processed = ref_y.values
                y_curr_processed = curr_y.values
                print("Pipeline prediction successful with raw data")
            else:
                # Regular model, needs preprocessing
                print("Preprocessing data for non-pipeline model...")
                X_ref_processed, X_curr_processed, y_ref_processed, y_curr_processed = preprocess_data(
                    ref_X, curr_X, ref_y, curr_y
                )
                print(
                    f"Preprocessing successful. Processed shapes - Ref: {X_ref_processed.shape}, Curr: {X_curr_processed.shape}"
                )

                ref_predictions = model.predict(X_ref_processed)
                curr_predictions = model.predict(X_curr_processed)
                print(
                    f"Predictions successful. Ref preds shape: {ref_predictions.shape}, Curr preds shape: {curr_predictions.shape}"
                )

        except Exception as e:
            print(f"Primary prediction strategy failed: {str(e)}")
            # Fallback: try the opposite approach
            try:
                if is_pipeline:
                    print("Pipeline failed with raw data, trying with preprocessing...")
                    X_ref_processed, X_curr_processed, y_ref_processed, y_curr_processed = preprocess_data(
                        ref_X, curr_X, ref_y, curr_y
                    )
                    ref_predictions = model.predict(X_ref_processed)
                    curr_predictions = model.predict(X_curr_processed)
                else:
                    print("Non-pipeline failed with preprocessing, trying with raw data...")
                    ref_predictions = model.predict(ref_X)
                    curr_predictions = model.predict(curr_X)
                    y_ref_processed = ref_y.values
                    y_curr_processed = curr_y.values
                print("Fallback prediction strategy successful")
            except Exception as e2:
                # Final fallback: simple categorical encoding only
                try:
                    print("Both strategies failed, trying minimal preprocessing...")
                    X_ref_simple = ref_X.copy()
                    X_curr_simple = curr_X.copy()

                    # Only encode categorical variables, no scaling
                    categorical_cols = X_ref_simple.select_dtypes(include=["object", "category"]).columns
                    for col in categorical_cols:
                        le = LabelEncoder()
                        combined_values = pd.concat([X_ref_simple[col], X_curr_simple[col]]).astype(str)
                        le.fit(combined_values)
                        X_ref_simple[col] = le.transform(X_ref_simple[col].astype(str))
                        X_curr_simple[col] = le.transform(X_curr_simple[col].astype(str))

                    ref_predictions = model.predict(X_ref_simple)
                    curr_predictions = model.predict(X_curr_simple)
                    y_ref_processed = ref_y.values
                    y_curr_processed = curr_y.values
                    print("Minimal preprocessing successful")
                except Exception as e3:
                    raise ValueError(
                        f"All prediction strategies failed. Pipeline error: {str(e)}. Preprocessing error: {str(e2)}. Minimal error: {str(e3)}. Model type: {type(model)}"
                    )

        # Calculate performance metrics for reference data
        ref_accuracy = accuracy_score(y_ref_processed, ref_predictions)
        ref_precision = precision_score(y_ref_processed, ref_predictions, average="weighted", zero_division=0)
        ref_recall = recall_score(y_ref_processed, ref_predictions, average="weighted", zero_division=0)
        ref_f1 = f1_score(y_ref_processed, ref_predictions, average="weighted", zero_division=0)

        # Calculate performance metrics for current data
        curr_accuracy = accuracy_score(y_curr_processed, curr_predictions)
        curr_precision = precision_score(y_curr_processed, curr_predictions, average="weighted", zero_division=0)
        curr_recall = recall_score(y_curr_processed, curr_predictions, average="weighted", zero_division=0)
        curr_f1 = f1_score(y_curr_processed, curr_predictions, average="weighted", zero_division=0)

        # Calculate performance drift
        accuracy_drift = abs(ref_accuracy - curr_accuracy)
        precision_drift = abs(ref_precision - curr_precision)
        recall_drift = abs(ref_recall - curr_recall)
        f1_drift = abs(ref_f1 - curr_f1)

        # Determine drift severity (thresholds can be adjusted)
        def get_drift_severity(drift_value: float) -> str:
            if drift_value < 0.05:
                return "Low"
            elif drift_value < 0.15:
                return "Medium"
            else:
                return "High"

        # Calculate confusion matrices
        ref_cm = confusion_matrix(y_ref_processed, ref_predictions).tolist()
        curr_cm = confusion_matrix(y_curr_processed, curr_predictions).tolist()

        # Generate classification reports
        ref_report = classification_report(y_ref_processed, ref_predictions, output_dict=True, zero_division=0)
        curr_report = classification_report(y_curr_processed, curr_predictions, output_dict=True, zero_division=0)

        # Overall drift assessment
        overall_drift = max(accuracy_drift, precision_drift, recall_drift, f1_drift)
        drift_severity = get_drift_severity(overall_drift)

        # Prepare the response
        result = {
            "analysis_id": f"model_drift_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            "model_info": {
                "model_type": str(type(model).__name__),
                "reference_samples": len(ref_df),
                "current_samples": len(curr_df),
                "features": list(ref_X.columns),
            },
            "performance_metrics": {
                "reference": {
                    "accuracy": float(ref_accuracy),
                    "precision": float(ref_precision),
                    "recall": float(ref_recall),
                    "f1_score": float(ref_f1),
                },
                "current": {
                    "accuracy": float(curr_accuracy),
                    "precision": float(curr_precision),
                    "recall": float(curr_recall),
                    "f1_score": float(curr_f1),
                },
            },
            "drift_metrics": {
                "accuracy_drift": float(accuracy_drift),
                "precision_drift": float(precision_drift),
                "recall_drift": float(recall_drift),
                "f1_drift": float(f1_drift),
                "overall_drift": float(overall_drift),
                "drift_severity": drift_severity,
            },
            "confusion_matrices": {"reference": ref_cm, "current": curr_cm},
            "classification_reports": {"reference": ref_report, "current": curr_report},
            "drift_detected": drift_severity in ["Medium", "High"],
        }

        return result

    except Exception as e:
        logging.exception("Exception during model drift analysis")  # Logs full stack trace
        return {
            "error": "Model drift analysis failed due to an internal error.",
            "analysis_id": f"model_drift_error_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        }
