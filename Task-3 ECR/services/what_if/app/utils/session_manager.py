import os
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import requests
import shap
from ydata_profiling import ProfileReport

from services.what_if.app.utils.config import SHAP_BACKGROUND_SAMPLE_SIZE, WHATIF_SESSIONS_DIR
from services.what_if.app.utils.preprocessing import create_preprocessing_pipeline

# Try to import onnxruntime, but don't fail if it's not available
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
    print("✅ ONNX Runtime available")
except ImportError as e:
    print(f"⚠️ ONNX Runtime not available: {e}")
    ort = None
    ONNX_AVAILABLE = False


# ONNX Model Wrapper class at module level to make it picklable
class ONNXModelWrapper:
    """Wrapper class for ONNX models to match sklearn interface"""

    def __init__(self, ort_session):
        self.session = ort_session
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def predict(self, X):
        # Convert to numpy if it's a DataFrame
        if hasattr(X, "values"):
            X = X.values

        X = X.astype(np.float32)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: X})

        # Use the first output for predictions
        predictions = outputs[0]

        # Dynamic detection: check if output is probabilities or class predictions
        flattened_predictions = predictions.flatten()
        if len(flattened_predictions) > 0:
            min_val = np.min(flattened_predictions)
            max_val = np.max(flattened_predictions)

            # Check if values look like probabilities (0-1 range and not just 0/1)
            is_probability = min_val >= 0 and max_val <= 1 and not np.all(np.isin(flattened_predictions, [0, 1]))

            if is_probability:
                # Convert probabilities to class predictions using 0.5 threshold
                result = (flattened_predictions >= 0.5).astype(int)
                return result
            else:
                # Use discrete class predictions as-is
                result = flattened_predictions.astype(int)
                return result
        else:
            return np.array([0])

    def predict_proba(self, X):
        # Convert to numpy if it's a DataFrame
        if hasattr(X, "values"):
            X = X.values

        X = X.astype(np.float32)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: X})

        # Check if we have a probability output (second output should be output_probability)
        if len(outputs) > 1:
            prob_output = outputs[1]  # output_probability

            # Try to extract probabilities from the sequence of maps
            if hasattr(prob_output, "__len__") and len(prob_output) > 0:
                try:
                    # Convert sequence of maps to probability array
                    prob_array = []
                    for prob_map in prob_output:
                        if hasattr(prob_map, "items"):
                            # Extract probabilities from the map (dict-like)
                            probs = [prob_map.get(0, 0.0), prob_map.get(1, 0.0)]
                            prob_array.append(probs)
                        else:
                            # Fallback if it's already a list/array
                            prob_array.append([1.0 - float(prob_map), float(prob_map)])

                    result = np.array(prob_array)
                    return result
                except Exception as e:
                    pass

        # Fallback: use first output and determine if it's probabilities or class predictions
        raw_output = outputs[0]

        # Dynamic detection based on output characteristics
        flattened_output = raw_output.flatten()
        if len(flattened_output) > 0:
            min_val = np.min(flattened_output)
            max_val = np.max(flattened_output)

            # Enhanced detection logic
            # Case 1: Values are exactly 0 and 1 (discrete class predictions)
            if np.all(np.isin(flattened_output, [0, 1])):
                probabilities = []
                for pred in flattened_output:
                    if pred == 0:
                        probabilities.append([1.0, 0.0])  # High confidence for class 0
                    else:
                        probabilities.append([0.0, 1.0])  # High confidence for class 1

                result = np.array(probabilities)
                return result

            # Case 2: Values are continuous probabilities (0 to 1 range, not just 0/1)
            elif min_val >= 0 and max_val <= 1:
                # For binary classification: [prob_class_0, prob_class_1]
                prob_class_1 = flattened_output  # Probability of positive class
                prob_class_0 = 1.0 - prob_class_1  # Probability of negative class

                # Stack to create [n_samples, 2] array
                result = np.column_stack([prob_class_0, prob_class_1])
                return result

            # Case 3: Values outside 0-1 range (treat as raw class predictions)
            else:
                probabilities = []
                for pred in flattened_output:
                    if pred <= 0 or pred < 0.5:
                        probabilities.append([1.0, 0.0])  # Class 0
                    else:
                        probabilities.append([0.0, 1.0])  # Class 1

                result = np.array(probabilities)
                return result
        else:
            return np.array([[0.5, 0.5]])


class SessionManager:
    def __init__(self, sessions_dir: str = None, session_timeout: int = 3600):
        default_path = "services/what_if/app/utils/sessions"
        self.sessions_dir = Path(sessions_dir or WHATIF_SESSIONS_DIR or default_path)
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create session directory at {self.sessions_dir}: {e}")

        # Cache for loaded artifacts to avoid repeated reconstruction
        self._artifact_cache = {}

        # Session management for cleanup
        self.session_timeout = session_timeout  # Default 1 hour timeout
        self.session_last_accessed = {}  # Track last access time for each session

        # Start cleanup thread
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background thread for automatic session cleanup"""

        def cleanup_worker():
            while True:
                try:
                    self._cleanup_expired_sessions()
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    print(f"⚠️ Cleanup thread error: {e}")
                    time.sleep(60)  # Wait 1 minute on error

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        print("🧹 Started automatic session cleanup thread")

    def _cleanup_expired_sessions(self):
        """Remove expired sessions and their data"""
        current_time = time.time()
        expired_sessions = []

        for session_id, last_accessed in self.session_last_accessed.items():
            if current_time - last_accessed > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self._delete_session(session_id)
            print(f"🗑️ Cleaned up expired session: {session_id}")

    def _delete_session(self, session_id: str):
        """Delete a session and all its associated data"""
        try:
            # Remove from cache
            if session_id in self._artifact_cache:
                del self._artifact_cache[session_id]

            # Remove from last accessed tracking
            if session_id in self.session_last_accessed:
                del self.session_last_accessed[session_id]

            # Remove session directory and all files
            session_path = self.sessions_dir / session_id
            if session_path.exists():
                import shutil

                shutil.rmtree(session_path)
                print(f"📁 Deleted session files: {session_path}")

        except Exception as e:
            print(f"⚠️ Error deleting session {session_id}: {e}")

    def _update_session_access(self, session_id: str):
        """Update last access time for a session"""
        self.session_last_accessed[session_id] = time.time()

    def refresh_session(self, session_id: str):
        """Refresh session access time to extend its lifetime"""
        if session_id in self.session_last_accessed:
            self._update_session_access(session_id)
            return {
                "message": f"Session {session_id} refreshed successfully",
                "last_accessed": self.session_last_accessed[session_id],
            }
        else:
            # If session not tracked, check if it exists and add it
            session_path = self.sessions_dir / session_id
            if session_path.exists():
                self._update_session_access(session_id)
                return {
                    "message": f"Session {session_id} refreshed and added to tracking",
                    "last_accessed": self.session_last_accessed[session_id],
                }
            else:
                raise ValueError(f"Session {session_id} not found")

    def cleanup_session(self, session_id: str):
        """Manually cleanup a specific session"""
        self._delete_session(session_id)
        return {"message": f"Session {session_id} cleaned up successfully"}

    def cleanup_all_sessions(self):
        """Manually cleanup all sessions"""
        session_ids = list(self.session_last_accessed.keys())
        for session_id in session_ids:
            self._delete_session(session_id)

        # Also clean any orphaned directories
        try:
            if self.sessions_dir.exists():
                for session_dir in self.sessions_dir.iterdir():
                    if session_dir.is_dir():
                        import shutil

                        shutil.rmtree(session_dir)
                        print(f"🗑️ Cleaned orphaned session directory: {session_dir}")
        except Exception as e:
            print(f"⚠️ Error cleaning orphaned directories: {e}")

        return {"message": "All sessions cleaned up successfully"}

    def load_model(self, content: bytes, filename: str):
        """Load model from bytes content - ONNX models only"""
        try:
            if filename.endswith(".onnx"):
                if not ONNX_AVAILABLE:
                    return (
                        None,
                        None,
                        "onnx",
                        "ONNX Runtime is not properly installed or has DLL issues. Please reinstall with: pip uninstall onnxruntime && pip install onnxruntime",
                    )

                try:
                    # Create ONNX runtime session
                    session = ort.InferenceSession(content)
                    return session, None, "onnx", None

                except Exception as e:
                    return None, None, "onnx", f"Error loading ONNX model: {str(e)}"
            else:
                return (
                    None,
                    None,
                    "unsupported",
                    f"Unsupported file format: {filename}. Only ONNX models (.onnx) are supported.",
                )
        except Exception as e:
            return None, None, filename.split(".")[-1], f"Unexpected error loading model: {str(e)}"

    def load_from_s3(self, access_token: str, analysis_type: str):
        """
        Fetch files from external S3 API using user-provided access token.
        Returns train_df, test_df, model.
        """
        if not access_token:
            raise ValueError("Access token is required.")

        BASE_URL = os.getenv("FILES_API_BASE_URL")
        if not BASE_URL:
            raise ValueError("FILES_API_BASE_URL environment variable is not set.")

        parsed_url = urlparse(BASE_URL)

        # Ensure scheme and netloc are present
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise ValueError("FILES_API_BASE_URL must be a valid HTTP/HTTPS URL.")

        if not BASE_URL:
            raise ValueError(
                "FILES_API_BASE_URL environment variable is not set. Please set it to the base URL of the files API."
            )

        api_url = f"{BASE_URL}/{analysis_type}"

        headers = {"Authorization": f"Bearer {access_token}"}

        response = requests.get(api_url, headers=headers)
        print(f"🔍 Response status: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ Response headers: {response.headers}")
            print(f"❌ Response content: {response.text}")
            raise Exception(f"API request failed with status {response.status_code}")

        files = response.json().get("files", [])
        train_df = None
        test_df = None
        model = None

        for file_info in files:
            file_response = requests.get(file_info["url"])
            if file_response.status_code != 200:
                continue

            fname = file_info["file_name"].lower()
            if fname.endswith(".csv"):
                df = pd.read_csv(BytesIO(file_response.content))
                print(f"✅ Loaded dataset: {fname} | Shape: {df.shape} | Rows: {len(df)}")
                if "ref" in fname:
                    train_df = df
                    print(f"🔹 Set as TRAIN dataset: {len(df)} rows")
                elif "cur" in fname:
                    test_df = df
                    print(f"🔹 Set as TEST dataset: {len(df)} rows")
                else:
                    train_df = df
                    print(f"🔹 Set as DEFAULT TRAIN dataset: {len(df)} rows")

            elif fname.endswith(".onnx"):
                print(f"🔍 Loading ONNX model file: {fname}")
                loaded_model, features, format_tag, error = self.load_model(file_response.content, fname)
                if loaded_model is not None:
                    model = loaded_model
                    print(f"✅ ONNX model loaded successfully")
                elif error:
                    print(f"❌ ONNX model loading error: {error}")

        if model is None:
            print("❌ No ONNX model found in S3 response. Only ONNX models are supported.")
            raise Exception("❌ No ONNX model found. Only ONNX models (.onnx) are supported.")

        if train_df is None and test_df is None:
            raise Exception("❌ No dataset found in S3 response.")

        return train_df, test_df, model

    def create_session_from_s3(self, access_token: str, analysis_type: str, target_column: str):
        """Create a new analysis session with data from S3"""
        session_id = uuid.uuid4().hex[:8]

        # Ensure sessions directory exists
        self.sessions_dir.mkdir(exist_ok=True)

        session_path = self.sessions_dir / session_id
        session_path.mkdir(exist_ok=True)

        try:
            # Load data from S3
            train_df, test_df, model = self.load_from_s3(access_token, analysis_type)

            # If no test_df, use train_df as test_df
            if test_df is None:
                test_df = train_df.copy()

            # Store original training data in memory for SHAP explainer recreation
            # (avoid saving to local disk - keep in memory only)

            # Continue with the same logic as create_session
            return self._create_session_common(session_id, session_path, train_df, test_df, model, target_column)

        except Exception as e:
            # Cleanup on error
            if session_path.exists():
                import shutil

                shutil.rmtree(session_path)
            import traceback

            error_details = traceback.format_exc()
            print(f"S3 session creation error: {str(e)}")
            print(f"Full traceback: {error_details}")
            raise e

    def _create_session_common(
        self,
        session_id: str,
        session_path: Path,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model,
        target_column: str,
    ):
        """Common session creation logic used by both upload and S3 methods"""
        print(f"🔍 Session creation - Train: {train_df.shape}, Test: {test_df.shape}, Target: {target_column}")

        # Create preprocessing pipeline
        preprocessor, feature_names = create_preprocessing_pipeline(train_df, target_column)

        # Fit preprocessing pipeline
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        preprocessor.fit(X_train, y_train)

        # Transform data for SHAP
        X_train_processed = preprocessor.transform(X_train)

        # Convert to DataFrame to preserve feature names for model compatibility
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                processed_feature_names = preprocessor.get_feature_names_out()
                X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names)
            except Exception as e:
                print(f"⚠️ Could not get feature names from preprocessor: {e}")
                X_train_processed_df = pd.DataFrame(X_train_processed)
        else:
            X_train_processed_df = pd.DataFrame(X_train_processed)

        # Handle ONNX models - simplified since we only support ONNX now
        if ONNX_AVAILABLE and ort and isinstance(model, ort.InferenceSession):
            print("🧠 ONNX model detected")

            # Apply feature alignment for ONNX models
            X_train_processed_df = self._ensure_feature_alignment(X_train_processed_df)

            # Wrap ONNX model to match sklearn interface
            model = self._wrap_onnx_model(model)

            model_type = "onnx"
            model_expects_clean_names = True

        else:
            raise Exception("❌ Only ONNX models are supported. Please provide a .onnx model file.")

        # Validate model compatibility
        print("🔧 Model compatibility validation...")
        if not self._validate_model_compatibility(model, X_train_processed_df):
            print("⚠️ Model compatibility issues detected, but continuing...")

        # Create SHAP explainer with background sample
        # Fixed background size for consistent performance and to avoid timeout issues
        dataset_size = len(X_train_processed_df)
        background_size = SHAP_BACKGROUND_SAMPLE_SIZE  # Fixed size for optimal performance

        # Ensure we don't exceed dataset size
        background_size = min(background_size, dataset_size)

        print(
            f"🎯 Creating background sample: {background_size} samples from {dataset_size} total (fixed for performance)"
        )

        # Use stratified sampling based on target variable if available in training data
        try:
            # Try to maintain class balance in background sample
            from sklearn.model_selection import train_test_split

            # Get target values for stratified sampling
            y_train_for_sampling = y_train.values if hasattr(y_train, "values") else y_train

            # For very small datasets, ensure we don't sample more than available
            if background_size >= len(X_train_processed_df):
                background_data = X_train_processed_df.copy()
                print(f"✅ Using entire dataset as background ({len(background_data)} samples)")
                bg_target_dist = pd.Series(y_train_for_sampling).value_counts()
                print(f"📊 Full dataset class distribution: {dict(bg_target_dist)}")
            else:
                # Stratified sampling to maintain class distribution
                _, _, background_indices, _ = train_test_split(
                    range(len(X_train_processed_df)),
                    y_train_for_sampling,
                    test_size=background_size,
                    random_state=42,  # For reproducibility
                    stratify=y_train_for_sampling,
                )

                background_data = X_train_processed_df.iloc[background_indices]
                print(f"✅ Using stratified sampling for background data ({background_size} samples)")

                # Log class distribution in background
                bg_target_dist = pd.Series(y_train_for_sampling)[background_indices].value_counts()
                print(f"📊 Background sample class distribution: {dict(bg_target_dist)}")

        except Exception as e:
            print(f"⚠️ Stratified sampling failed ({e}), falling back to random sampling")
            # Fallback to random sampling
            if background_size >= len(X_train_processed_df):
                background_data = X_train_processed_df.copy()
                print(f"📊 Using entire dataset as background (fallback)")
            else:
                background_indices = np.random.choice(len(X_train_processed_df), background_size, replace=False)
                background_data = X_train_processed_df.iloc[background_indices]
                print(f"📊 Using random sampling for background data ({background_size} samples)")

        # Save background data for SHAP explainer recreation (avoid saving training data to disk)
        background_data_path = session_path / "shap_background.pkl"
        joblib.dump(background_data, background_data_path)
        print(f"💾 Saved SHAP background data ({len(background_data)} samples) for explainer recreation")

        # Generate feature info for What-If analysis using original training data
        # (before it goes out of scope)
        feature_info = {}
        X_train_original = train_df.drop(columns=[target_column])

        for column in X_train_original.columns:
            if column in feature_names:  # Only process features that made it through preprocessing
                col_info = {
                    "name": column,
                    "type": "numeric" if X_train_original[column].dtype in ["int64", "float64"] else "categorical",
                }

                if col_info["type"] == "numeric":
                    col_info.update(
                        {
                            "min": float(X_train_original[column].min()),
                            "max": float(X_train_original[column].max()),
                            "mean": float(X_train_original[column].mean()),
                            "median": float(X_train_original[column].median()),
                            "std": float(X_train_original[column].std()),
                        }
                    )
                else:
                    # For categorical columns, get unique values
                    unique_values = X_train_original[column].dropna().unique().tolist()
                    col_info.update(
                        {
                            "unique_values": unique_values[:50],  # Limit to 50 values for performance
                            "value_counts": X_train_original[column].value_counts().head(20).to_dict(),
                        }
                    )

                feature_info[column] = col_info

        print(f"📊 Generated feature info for {len(feature_info)} features")

        # Create SHAP explainer for ONNX models
        explainer = None
        explainer_type = "unknown"

        try:
            # Use KernelExplainer for ONNX models
            if hasattr(model, "predict_proba"):
                explainer = shap.KernelExplainer(model.predict_proba, background_data)
            else:
                explainer = shap.KernelExplainer(model.predict, background_data)
            explainer_type = "KernelExplainer"
            print("Using SHAP KernelExplainer for ONNX model")
        except Exception as e:
            print(f"Failed to create SHAP explainer: {e}")
            explainer = None
            explainer_type = "none"

        if explainer is None:
            print("Warning: Could not create SHAP explainer. SHAP analysis will not be available.")

        # Save artifacts - handle ONNX models specially for explainer saving
        joblib.dump(preprocessor, session_path / "preprocessor.pkl")

        # Handle ONNX explainer saving separately since InferenceSession can't be pickled
        if ONNX_AVAILABLE and ort and isinstance(model, ONNXModelWrapper):
            print("⚠️ ONNX model detected - explainer will be recreated on demand")
            # For ONNX models, save None for explainer since it will be recreated on demand
            # But we can still save it if it exists and works
            try:
                if explainer is not None:
                    # Try to save the explainer - if it fails, we'll recreate it later
                    joblib.dump(explainer, session_path / "explainer.pkl")
                    print("✅ ONNX SHAP explainer saved successfully")
                else:
                    joblib.dump(None, session_path / "explainer.pkl")
                    print("⚠️ ONNX explainer is None, will recreate on demand")
            except Exception as explainer_save_error:
                print(f"⚠️ Failed to save ONNX explainer, will recreate on demand: {explainer_save_error}")
                joblib.dump(None, session_path / "explainer.pkl")

            # Save the ONNX model content for reconstruction
            try:
                # Extract the original ONNX content from the session
                onnx_content = model.session._model_bytes if hasattr(model.session, "_model_bytes") else None
                if onnx_content:
                    onnx_model_path = session_path / "onnx_model.onnx"
                    with open(onnx_model_path, "wb") as f:
                        f.write(onnx_content)
                    print("📁 ONNX model content saved for reconstruction")
                else:
                    print("⚠️ Could not extract ONNX model content")
            except Exception as onnx_save_error:
                print(f"⚠️ Could not save ONNX model content: {onnx_save_error}")
        else:
            # Normal explainer saving for non-ONNX models
            try:
                joblib.dump(explainer, session_path / "explainer.pkl")
            except Exception as explainer_save_error:
                print(f"⚠️ Failed to save explainer: {explainer_save_error}")
                joblib.dump(None, session_path / "explainer.pkl")

        # Save model - handle ONNX models specially
        if ONNX_AVAILABLE and ort and isinstance(model, ONNXModelWrapper):
            print("⚠️ ONNX model detected - saving model metadata for reconstruction")
            # Save comprehensive model metadata for reconstruction
            onnx_metadata = {
                "type": "onnx_model",
                "input_name": model.input_name,
                "output_names": model.output_names,
                "feature_names": list(X_train_processed_df.columns),
                "onnx_file": "onnx_model.onnx",  # Reference to saved ONNX file
            }
            joblib.dump(onnx_metadata, session_path / "model.pkl")
        else:
            # Since we only support ONNX models now, this shouldn't happen
            raise Exception("❌ Only ONNX models are supported")

        # Generate data profile (disable sampling to use full dataset)
        print(f"Generating profile for {len(train_df)} rows and {len(train_df.columns)} columns")
        print(f"📊 Columns to analyze: {list(train_df.columns)}")

        try:
            # Try with minimal config first
            profile = ProfileReport(
                train_df, title=f"Dataset Profile - {len(train_df)} rows, {len(train_df.columns)} columns", minimal=True
            )
            profile.to_file(session_path / "profile.html")
        except Exception as profile_error:
            print(f"Profile generation error: {profile_error}")
            # Create a simple HTML file as fallback
            with open(session_path / "profile.html", "w") as f:
                f.write(
                    f"""
                <html>
                <head><title>Dataset Profile</title></head>
                <body>
                    <h1>Dataset Profile - {len(train_df)} rows</h1>
                    <p>Profile generation failed, but session created successfully.</p>
                    <p>Dataset shape: {train_df.shape}</p>
                    <p>Columns: {', '.join(train_df.columns.tolist())}</p>
                </body>
                </html>
                """
                )

        # Create metadata with appropriate feature names based on model type
        if model_type == "onnx":
            # For ONNX models, use the cleaned feature names (without prefixes)
            final_feature_names = list(X_train_processed_df.columns)
            processed_feature_names_for_metadata = final_feature_names
            uses_clean_names = True
        else:
            # For sklearn models, determine if model uses clean or preprocessed names
            if model_expects_clean_names:
                # Model was fitted with clean names
                final_feature_names = list(X_train_processed_df.columns)  # Clean names for display
                processed_feature_names_for_metadata = list(X_train_processed_df.columns)  # Clean names for model
                uses_clean_names = True
            else:
                # Model uses preprocessed names
                final_feature_names = feature_names  # Original column names for display
                processed_feature_names_for_metadata = list(
                    X_train_processed_df.columns
                )  # Preprocessed names for model
                uses_clean_names = False

        metadata = {
            "session_id": session_id,
            "target_column": target_column,
            "feature_names": final_feature_names,  # For UI display
            "processed_feature_names": processed_feature_names_for_metadata,  # For model input
            "uses_clean_names": uses_clean_names,  # Whether model expects clean or preprocessed names
            "n_features": len(final_feature_names),
            "n_samples": len(train_df),
            "explainer_type": explainer_type,
            "shap_available": explainer is not None,
            "model_type": model_type,
            "feature_info": feature_info,  # Store feature statistics for What-If analysis
            "created_at": str(pd.Timestamp.now()),
        }

        joblib.dump(metadata, session_path / "metadata.pkl")

        # Track session creation time
        self._update_session_access(session_id)
        print(f"📝 Session {session_id} created and tracked for cleanup")

        return session_id, metadata

    def get_session(self, session_id: str):
        """Get session metadata if it exists"""
        session_path = self.sessions_dir / session_id

        if not session_path.exists():
            return None

        try:
            # Update access time
            self._update_session_access(session_id)

            metadata = joblib.load(session_path / "metadata.pkl")
            return metadata
        except Exception as e:
            print(f"Error loading session metadata: {e}")
            return None

    def get_session_artifacts(self, session_id: str):
        """Load session artifacts with ONNX model reconstruction and caching"""
        session_path = self.sessions_dir / session_id

        if not session_path.exists():
            raise ValueError(f"Session {session_id} not found")

        # Update access time
        self._update_session_access(session_id)

        # Check cache first
        if session_id in self._artifact_cache:
            print(f"✅ Using cached artifacts for session {session_id}")
            return self._artifact_cache[session_id]

        preprocessor = joblib.load(session_path / "preprocessor.pkl")
        explainer = joblib.load(session_path / "explainer.pkl")
        model_data = joblib.load(session_path / "model.pkl")
        metadata = joblib.load(session_path / "metadata.pkl")

        # Handle ONNX model reconstruction
        if isinstance(model_data, dict) and model_data.get("type") == "onnx_model":
            print("🔧 Reconstructing ONNX model from saved metadata...")
            try:
                if ONNX_AVAILABLE and ort:
                    # Load the ONNX model from saved file
                    onnx_file_path = session_path / model_data.get("onnx_file", "onnx_model.onnx")
                    if onnx_file_path.exists():
                        # Create new InferenceSession from saved ONNX file
                        session = ort.InferenceSession(str(onnx_file_path))
                        # Recreate the wrapper
                        model = ONNXModelWrapper(session)
                        print("✅ ONNX model reconstructed successfully")

                        # If explainer is None (ONNX case), try to recreate SHAP explainer
                        if explainer is None:
                            print("🔧 Recreating SHAP explainer for ONNX model...")
                            try:
                                # Load saved background data for SHAP explainer recreation
                                background_data_path = session_path / "shap_background.pkl"
                                if background_data_path.exists():
                                    background_data = joblib.load(background_data_path)
                                    print(f"✅ Loaded SHAP background data ({len(background_data)} samples)")

                                    # Create SHAP explainer using saved background data
                                    if hasattr(model, "predict_proba"):
                                        explainer = shap.KernelExplainer(model.predict_proba, background_data)
                                    else:
                                        explainer = shap.KernelExplainer(model.predict, background_data)
                                    print("✅ SHAP explainer recreated for ONNX model")

                                    # Update metadata to reflect SHAP availability
                                    metadata["shap_available"] = True
                                    metadata["explainer_type"] = "KernelExplainer"
                                    joblib.dump(metadata, session_path / "metadata.pkl")
                                else:
                                    print("⚠️ SHAP background data not found, cannot recreate SHAP explainer")
                            except Exception as shap_error:
                                print(f"⚠️ Failed to recreate SHAP explainer: {shap_error}")
                                explainer = None

                    else:
                        print("⚠️ ONNX model file not found, using placeholder")
                        model = model_data  # Return the metadata as fallback
                else:
                    print("⚠️ ONNX Runtime not available, using placeholder")
                    model = model_data  # Return the metadata as fallback
            except Exception as e:
                print(f"⚠️ ONNX model reconstruction failed: {e}")
                model = model_data  # Return the metadata as fallback
        else:
            model = model_data

        # Cache the artifacts for future use
        artifacts = (preprocessor, explainer, model, metadata)
        self._artifact_cache[session_id] = artifacts
        print(f"✅ Cached artifacts for session {session_id}")

        return artifacts

    def clear_cache(self, session_id: str = None):
        """Clear artifact cache for a specific session or all sessions"""
        if session_id:
            if session_id in self._artifact_cache:
                del self._artifact_cache[session_id]
                print(f"✅ Cleared cache for session {session_id}")
        else:
            self._artifact_cache.clear()
            print("✅ Cleared all cached artifacts")

    def _ensure_feature_alignment(self, X_train_processed_df):
        """Ensure feature alignment between model and data like your working code"""
        # This method handles feature alignment between the model and processed data
        # Remove preprocessing prefixes to match original model training
        if hasattr(X_train_processed_df, "columns"):
            # Create a mapping to clean column names
            new_columns = []
            for col in X_train_processed_df.columns:
                # Remove common preprocessing prefixes
                clean_col = col
                if "__" in col:
                    clean_col = col.split("__", 1)[1]  # Remove prefix like 'num__', 'cat__'
                new_columns.append(clean_col)

            # Update the DataFrame columns
            X_train_processed_df.columns = new_columns

        return X_train_processed_df

    def _wrap_onnx_model(self, onnx_session):
        """Wrap ONNX model to match sklearn interface like your working code"""
        return ONNXModelWrapper(onnx_session)

    def _validate_model_compatibility(self, model, X_train_processed_df):
        """Validate model compatibility like your working code"""
        try:
            # Test prediction on a small sample
            test_sample = X_train_processed_df.head(1)
            predictions = model.predict(test_sample)
            print(f"✅ Model compatibility validated - sample prediction: {predictions[0]:.6f}")
            return True
        except Exception as e:
            print(f"⚠️ Model compatibility validation failed: {e}")
            return False
