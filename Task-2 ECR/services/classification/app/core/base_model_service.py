import io
import os
import pickle
import tempfile
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
import requests
import shap
from fastapi import HTTPException
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from services.classification.app.logging_config import logger


class ModelWrapper:
    """Wrapper class to provide consistent interface for scikit-learn models."""

    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
        self._feature_names_expected = None

    def _detect_feature_names_requirement(self, X):
        """Detect whether the model expects feature names or not."""
        if self._feature_names_expected is not None:
            return self._feature_names_expected

        if not hasattr(X, "values"):
            self._feature_names_expected = False
            return False

        import warnings

        # Test with DataFrame (feature names)
        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always")
            try:
                _ = self.model.predict_proba(X.iloc[:1])
                df_warnings = len(w1)
            except:
                df_warnings = float("inf")

        # Test with numpy array (no feature names)
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            try:
                _ = self.model.predict_proba(X.iloc[:1].values)
                array_warnings = len(w2)
            except:
                array_warnings = float("inf")

        # Choose the approach with fewer warnings
        if df_warnings <= array_warnings:
            self._feature_names_expected = True
            print(f"📊 Model expects feature names (DataFrame input)")
        else:
            self._feature_names_expected = False
            print(f"📊 Model expects numpy arrays (no feature names)")

        return self._feature_names_expected

    def _prepare_input(self, X):
        """Prepare input data in the format expected by the model."""
        if not hasattr(X, "values"):
            return X

        expects_feature_names = self._detect_feature_names_requirement(X)
        return X if expects_feature_names else X.values

    def predict(self, X):
        """Predict class labels with adaptive input format."""
        if self.model_type == "sklearn":
            prepared_X = self._prepare_input(X)
            return self.model.predict(prepared_X)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict_proba(self, X):
        """Predict class probabilities with adaptive input format."""
        if self.model_type == "sklearn":
            prepared_X = self._prepare_input(X)
            return self.model.predict_proba(prepared_X)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        return getattr(self.model, name)


class BaseModelService:
    """Base service that holds the loaded model and shared data/state."""

    def __init__(self):
        # Core model and data
        self.model: Optional[ModelWrapper] = None

        # Separate train and test datasets
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None

        # Backward compatibility
        self.X_df: Optional[pd.DataFrame] = None  # Points to train data
        self.y_s: Optional[pd.Series] = None  # Points to train data

        # Metadata
        self.feature_names: Optional[List[str]] = None
        self.target_name: Optional[str] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.model_info: Dict[str, Any] = {}

        print("BaseModelService initialized. Waiting for model and data.")

    def _detect_model_framework(self, model_wrapper: ModelWrapper) -> str:
        """Detect the framework used to create the model."""
        return "scikit-learn" if model_wrapper.model_type == "sklearn" else "unknown"

    def _get_model_algorithm(self, model_wrapper: ModelWrapper) -> str:
        """Get the algorithm name from the model."""
        if model_wrapper.model_type == "sklearn":
            model_name = type(model_wrapper.model).__name__
            # Handle ensemble models
            if hasattr(model_wrapper.model, "base_estimator"):
                base_name = type(model_wrapper.model.base_estimator).__name__
                return f"{model_name}({base_name})"
            elif hasattr(model_wrapper.model, "estimators_"):
                if hasattr(model_wrapper.model, "base_estimator_"):
                    base_name = type(model_wrapper.model.base_estimator_).__name__
                    return f"{model_name}({base_name})"
            return model_name
        else:
            return "Unknown"

    def _load_model_by_format(self, model_path: str) -> ModelWrapper:
        """Load model based on file extension and return wrapped model."""

        # Check if it's a URL (S3 pre-signed URL)
        if model_path.startswith(("http://", "https://")):
            return self._load_model_from_presigned_url(model_path)

        # Handle local file paths
        file_extension = model_path.lower()

        if file_extension.endswith((".joblib", ".pkl", ".pickle")):
            try:
                if file_extension.endswith(".joblib"):
                    model = joblib.load(model_path)
                else:
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                return ModelWrapper(model, "sklearn")
            except Exception as e:
                raise ValueError(f"Failed to load scikit-learn model: {str(e)}")
        else:
            raise ValueError(f"Unsupported model format. Supported formats: .joblib, .pkl, .pickle. Got: {model_path}")

    def _load_model_from_presigned_url(self, url: str) -> ModelWrapper:
        """Load model directly from S3 pre-signed URL."""

        print(f"📥 Downloading model from S3...")

        # Parse URL to extract useful debugging info
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)

        # Check for common issues with pre-signed URLs
        if "X-Amz-Date" in query_params:
            print(f"🕒 Pre-signed URL date: {query_params['X-Amz-Date'][0]}")
        if "X-Amz-Expires" in query_params:
            print(f"⏰ URL expires in: {query_params['X-Amz-Expires'][0]} seconds")

        try:
            response = requests.get(url, stream=True, timeout=30)

            # Check response status before raising for status
            if response.status_code == 403:
                print(f"❌ Access denied (403). Pre-signed URL may have expired or lacks permissions.")
                print(f"🔗 URL path: {parsed.path}")
                raise ValueError(
                    f"S3 access denied. The pre-signed URL may have expired or lacks proper permissions. Status: {response.status_code}"
                )
            elif response.status_code == 404:
                print(f"❌ File not found (404). The model file may not exist at the specified location.")
                raise ValueError(f"Model file not found at S3 location. Status: {response.status_code}")
            elif response.status_code != 200:
                print(f"❌ Unexpected HTTP status: {response.status_code}")
                print(f"📄 Response content: {response.text[:200]}")
                raise ValueError(f"Failed to download model from S3. HTTP Status: {response.status_code}")

            response.raise_for_status()  # This should now only raise for 200s that somehow failed

            print(
                f"✅ Successfully connected to S3. Content-Length: {response.headers.get('Content-Length', 'unknown')}"
            )

        except requests.exceptions.Timeout:
            raise ValueError("S3 download timed out after 30 seconds. Please check your internet connection.")
        except requests.exceptions.ConnectionError:
            raise ValueError("Failed to connect to S3. Please check your internet connection.")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error while downloading from S3: {str(e)}")

        # Determine file extension from URL (before query parameters)
        file_path = parsed.path

        if file_path.lower().endswith(".joblib"):
            model_bytes = response.content
            print(f"📦 Downloaded {len(model_bytes)} bytes for .joblib model")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_file:
                tmp_file.write(model_bytes)
                tmp_path = tmp_file.name

            try:
                model = joblib.load(tmp_path)
                print(f"✅ Successfully loaded .joblib model: {type(model).__name__}")
                return ModelWrapper(model, "sklearn")
            except Exception as e:
                raise ValueError(f"Failed to load .joblib model: {str(e)}")
            finally:
                os.unlink(tmp_path)

        elif file_path.lower().endswith((".pkl", ".pickle")):
            model_bytes = response.content
            print(f"📦 Downloaded {len(model_bytes)} bytes for .pkl model")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
                tmp_file.write(model_bytes)
                tmp_path = tmp_file.name

            try:
                with open(tmp_path, "rb") as f:
                    model = pickle.load(f)
                print(f"✅ Successfully loaded .pkl model: {type(model).__name__}")
                return ModelWrapper(model, "sklearn")
            except Exception as e:
                raise ValueError(f"Failed to load .pkl model: {str(e)}")
            finally:
                os.unlink(tmp_path)
        else:
            raise ValueError(f"Unsupported model format in URL: {file_path}. Supported formats: .joblib, .pkl, .pickle")

    # New helper to load CSV or Parquet from local path or S3 presigned URL
    # ----------------------------
    def _load_data_flexibly(self, path: str) -> pd.DataFrame:
        """Loads data from CSV or Parquet (local or S3 presigned URL)."""
        if not path:
            raise ValueError("Empty path provided to _load_data_flexibly")

        # If it's an S3 presigned URL or any http(s) URL
        if path.startswith("http://") or path.startswith("https://"):
            # Download bytes/text and then use appropriate pandas reader
            try:
                resp = requests.get(path, timeout=30)
                resp.raise_for_status()
            except requests.exceptions.Timeout:
                raise ValueError("Timeout while downloading dataset from URL.")
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Network error while downloading dataset: {e}")

            # Determine extension from URL path (strip query string)
            url_no_q = path.split("?", 1)[0]
            ext = os.path.splitext(url_no_q)[1].lower()

            if ext == ".csv":
                return pd.read_csv(io.StringIO(resp.text))
            elif ext == ".parquet":
                # read_parquet can accept bytes via pyarrow engine
                try:
                    return pd.read_parquet(io.BytesIO(resp.content), engine="pyarrow")
                except Exception as e:
                    raise ValueError(f"Failed to read parquet from remote bytes: {e}")
            else:
                raise ValueError(f"Unsupported remote file format: {ext}. Supported: .csv, .parquet")

        # Local file path
        else:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".csv":
                return pd.read_csv(path)
            elif ext == ".parquet":
                return pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported local file format: {ext}. Supported: .csv, .parquet")

    # def load_model_and_datasets(self, model_path: str, data_path: str = None, train_data_path: str = None, test_data_path: str = None, target_column: Optional[str] = None, test_size: float = 0.2, random_state: int = 42):
    #     """Unified method to load model and dataset(s) from local files or S3.

    #     Args:
    #         model_path: Path to model file (local or S3 URL)
    #         data_path: Path to single dataset (for train/test splitting)
    #         train_data_path: Path to training dataset
    #         test_data_path: Path to test dataset
    #         target_column: Name of target column
    #         test_size: Proportion of data for test set (when splitting single dataset)
    #         random_state: Random state for reproducible splitting
    #     """

    #     try:
    #         # Validate input parameters
    #         if not (train_data_path and test_data_path) and not data_path:
    #             raise ValueError("Must provide either data_path OR both train_data_path and test_data_path")

    #         # Load model
    #         if model_path.startswith('https://') and 's3.amazonaws.com' in model_path:
    #             print("🔗 Detected S3 pre-signed URL for model, loading directly...")
    #             print(f"⬇️ Loading model from: {model_path}")
    #             model_wrapper = self._load_model_from_presigned_url(model_path)
    #         else:
    #             print(f"📁 Loading model from local path: {model_path}")
    #             model_wrapper = self._load_model_by_format(model_path)

    #         self.model = model_wrapper

    #         # Handle single dataset case (split into train/test)
    #         if data_path:
    #             print("📊 Single dataset mode: will split into train/test")

    #             # Load dataset
    #             if data_path.startswith('https://') and 's3.amazonaws.com' in data_path:
    #                 print(f"⬇️ Loading dataset from S3: {data_path}")
    #                 df = pd.read_csv(data_path)
    #             else:
    #                 print(f"📁 Loading dataset from local path: {data_path}")
    #                 df = pd.read_csv(data_path)

    #             # Validate target column exists
    #             if target_column not in df.columns:
    #                 raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    #             # Split features and target
    #             X = df.drop(columns=[target_column])
    #             y = df[target_column]

    #             # Split into train/test
    #             X_train, X_test, y_train, y_test = train_test_split(
    #                 X, y, test_size=test_size, random_state=random_state, stratify=y
    #             )

    #             message = f"Model and dataset loaded successfully (split into train/test)"
    #             split_info = f"test_size={test_size}, random_state={random_state}"

    #         # Handle separate datasets case
    #         else:
    #             print("📊 Separate datasets mode: using provided train/test files")

    #             # Check if paths are S3 URLs
    #             if (train_data_path.startswith('https://') and 's3.amazonaws.com' in train_data_path and
    #                 test_data_path.startswith('https://') and 's3.amazonaws.com' in test_data_path):

    #                 print("🔗 Detected S3 pre-signed URLs for datasets, loading directly...")

    #                 # Load datasets directly from pre-signed URLs
    #                 print(f"⬇️ Loading training data from: {train_data_path}")
    #                 train_df = pd.read_csv(train_data_path)

    #                 print(f"⬇️ Loading test data from: {test_data_path}")
    #                 test_df = pd.read_csv(test_data_path)

    #             else:
    #                 # Handle local file paths
    #                 print(f"📁 Loading training data from: {train_data_path}")
    #                 train_df = pd.read_csv(train_data_path)

    #                 print(f"📁 Loading test data from: {test_data_path}")
    #                 test_df = pd.read_csv(test_data_path)

    #             # Validate target column exists in both datasets
    #             if target_column not in train_df.columns:
    #                 raise ValueError(f"Target column '{target_column}' not found in the training dataset.")
    #             if target_column not in test_df.columns:
    #                 raise ValueError(f"Target column '{target_column}' not found in the test dataset.")

    #             # Validate feature columns match
    #             train_features = set(train_df.drop(columns=[target_column]).columns)
    #             test_features = set(test_df.drop(columns=[target_column]).columns)
    #             if train_features != test_features:
    #                 missing_in_test = train_features - test_features
    #                 missing_in_train = test_features - train_features
    #                 error_msg = "Feature columns mismatch between train and test datasets."
    #                 if missing_in_test:
    #                     error_msg += f" Missing in test: {missing_in_test}."
    #                 if missing_in_train:
    #                     error_msg += f" Missing in train: {missing_in_train}."
    #                 raise ValueError(error_msg)

    #             # Extract features and targets
    #             X_train = train_df.drop(columns=[target_column])
    #             y_train = train_df[target_column]
    #             X_test = test_df.drop(columns=[target_column])
    #             y_test = test_df[target_column]

    #             message = f"Model and datasets loaded successfully"
    #             split_info = None

    #         # Store data
    #         self.X_train = X_train
    #         self.y_train = y_train
    #         self.X_test = X_test
    #         self.y_test = y_test

    #         # Backward compatibility - point to train data
    #         self.X_df = self.X_train
    #         self.y_s = self.y_train
    #         self.feature_names = list(X_train.columns)
    #         self.target_name = target_column

    #         # Initialize SHAP explainer
    #         self._initialize_shap_explainer(model_wrapper)

    #         # Calculate dataset statistics for model_info
    #         # Combine train and test data for overall statistics
    #         if data_path:
    #             # For single dataset case, use the original combined data
    #             combined_df = pd.concat([X_train, X_test], ignore_index=True)
    #             data_path_used = data_path
    #         else:
    #             # For separate datasets case, combine train and test for stats
    #             combined_df = pd.concat([X_train, X_test], ignore_index=True)
    #             data_path_used = f"train: {train_data_path}, test: {test_data_path}"

    #         num_rows = len(combined_df)
    #         missing_count = combined_df.isnull().sum().sum()
    #         total_cells = combined_df.size
    #         missing_ratio = missing_count / total_cells if total_cells > 0 else 0.0

    #         # Calculate duplicate ratio
    #         duplicate_count = combined_df.duplicated().sum()
    #         duplicate_ratio = duplicate_count / num_rows if num_rows > 0 else 0.0

    #         # Set comprehensive model info
    #         self.model_info = {
    #             "target_column": target_column,
    #             "features_count": len(self.feature_names),
    #             "data_shape": combined_df.shape,
    #             "algorithm": self._get_model_algorithm(model_wrapper),
    #             "framework": self._detect_model_framework(model_wrapper),
    #             "model_type": model_wrapper.model_type,
    #             "type": "classification" if hasattr(model_wrapper.model, "predict_proba") else "regression",
    #             "version": "1.0.0",
    #             "created": pd.Timestamp.utcnow().isoformat(),
    #             "last_trained": pd.Timestamp.utcnow().isoformat(),
    #             "samples": int(num_rows),
    #             "features": int(len(self.feature_names)),
    #             "missing_pct": missing_ratio * 100.0,
    #             "duplicates_pct": duplicate_ratio * 100.0,
    #             "status": "Active",
    #             "health_score_pct": max(0.0, 100.0 - (missing_ratio * 100.0 * 0.5 + duplicate_ratio * 100.0 * 0.5)),
    #             "train_samples": len(self.X_train),
    #             "test_samples": len(self.X_test),
    #             "training_samples": len(self.X_train),  # Backward compatibility
    #             "test_samples": len(self.X_test),       # Backward compatibility
    #             "shap_available": self.explainer is not None
    #         }

    #         # Add split info if available
    #         if split_info:
    #             self.model_info["split_info"] = split_info

    #         return {
    #             "status": "success",
    #             "message": message,
    #             "model_info": self.model_info,
    #             "features": self.feature_names,
    #             "target": self.target_name,
    #             "train_shape": self.X_train.shape,
    #             "test_shape": self.X_test.shape
    #         }

    #     except Exception as e:
    #         # Reset state on failure
    #         self.__init__()
    #         raise e

    def load_model_and_datasets(
        self,
        model_path: str,
        data_path: str = None,
        train_data_path: str = None,
        test_data_path: str = None,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Unified method to load model and dataset(s) from local files or S3.

        Args:
            model_path: Path to model file (local or S3 URL)
            data_path: Path to single dataset (for train/test splitting)
            train_data_path: Path to training dataset
            test_data_path: Path to test dataset
            target_column: Name of target column
            test_size: Proportion of data for test set (when splitting single dataset)
            random_state: Random state for reproducible splitting
        """

        try:
            # Validate input parameters
            if not (train_data_path and test_data_path) and not data_path:
                raise ValueError("Must provide either data_path OR both train_data_path and test_data_path")
            logger.info(
                f"Loading model and datasets with parameters: model_path={model_path}, data_path={data_path}, train_data_path={train_data_path}, test_data_path={test_data_path}, target_column={target_column}, test_size={test_size}, random_state={random_state}"
            )

            # Load model
            print(f"🔍 Analyzing model_path: {model_path}")
            print(f"🔍 Starts with https://: {model_path.startswith('https://')}")
            parsed = urlparse(model_path)
            host = parsed.hostname
            print(f"🔍 Parsed hostname: {host}")

            if (
                model_path.startswith("https://")
                and host
                and (host == ".amazonaws.com" or host.endswith(".amazonaws.com"))
            ):
                print("🔗 Detected S3 pre-signed URL for model, loading directly...")
                print(f"⬇️ Loading model from: {model_path}")
                model_wrapper = self._load_model_from_presigned_url(model_path)
            else:
                print(f"📁 Loading model from local path: {model_path}")
                model_wrapper = self._load_model_by_format(model_path)

            self.model = model_wrapper

            # Handle single dataset case (split into train/test)
            if data_path:
                print("📊 Single dataset mode: will split into train/test")

                # Load dataset (flexible)
                print(f"⬇️ Loading dataset from: {data_path}")
                df = self._load_data_flexibly(data_path)

                # Validate target column exists
                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in the dataset.")

                # Split features and target
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Split into train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )

                message = f"Model and dataset loaded successfully (split into train/test)"
                split_info = f"test_size={test_size}, random_state={random_state}"
                logger.info(f"Dataset split into train/test with test_size={test_size}, random_state={random_state}")

            # Handle separate datasets case
            else:
                print("📊 Separate datasets mode: using provided train/test files")

                # Load datasets flexibly (S3 or local)
                print(f"⬇️ Loading training data from: {train_data_path}")
                train_df = self._load_data_flexibly(train_data_path)

                print(f"⬇️ Loading test data from: {test_data_path}")
                test_df = self._load_data_flexibly(test_data_path)

                # Validate target column exists in both datasets
                if target_column not in train_df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in the training dataset.")
                if target_column not in test_df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in the test dataset.")

                # Validate feature columns match
                train_features = set(train_df.drop(columns=[target_column]).columns)
                test_features = set(test_df.drop(columns=[target_column]).columns)
                if train_features != test_features:
                    missing_in_test = train_features - test_features
                    missing_in_train = test_features - train_features
                    error_msg = "Feature columns mismatch between train and test datasets."
                    if missing_in_test:
                        error_msg += f" Missing in test: {missing_in_test}."
                    if missing_in_train:
                        error_msg += f" Missing in train: {missing_in_train}."
                    raise ValueError(error_msg)

                # Extract features and targets
                X_train = train_df.drop(columns=[target_column])
                y_train = train_df[target_column]
                X_test = test_df.drop(columns=[target_column])
                y_test = test_df[target_column]

                message = f"Model and datasets loaded successfully"
                split_info = None

            # Store data
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test

            # Backward compatibility - point to train data
            self.X_df = self.X_train
            self.y_s = self.y_train
            self.feature_names = list(X_train.columns)
            self.target_name = target_column
            logger.info(f"Data loaded with training samples: {len(self.X_train)}, test samples: {len(self.X_test)}")
            # Initialize SHAP explainer
            self._initialize_shap_explainer(model_wrapper)

            # Calculate dataset statistics for model_info
            # Combine train and test data for overall statistics
            if data_path:
                # For single dataset case, use the original combined data
                combined_df = pd.concat([X_train, X_test], ignore_index=True)
                data_path_used = data_path
            else:
                # For separate datasets case, combine train and test for stats
                combined_df = pd.concat([X_train, X_test], ignore_index=True)
                data_path_used = f"train: {train_data_path}, test: {test_data_path}"

            num_rows = len(combined_df)
            missing_count = combined_df.isnull().sum().sum()
            total_cells = combined_df.size
            missing_ratio = missing_count / total_cells if total_cells > 0 else 0.0

            # Calculate duplicate ratio
            duplicate_count = combined_df.duplicated().sum()
            duplicate_ratio = duplicate_count / num_rows if num_rows > 0 else 0.0

            # Set comprehensive model info
            self.model_info = {
                "target_column": target_column,
                "features_count": len(self.feature_names),
                "data_shape": combined_df.shape,
                "algorithm": self._get_model_algorithm(model_wrapper),
                "framework": self._detect_model_framework(model_wrapper),
                "model_type": model_wrapper.model_type,
                "type": "classification" if hasattr(model_wrapper.model, "predict_proba") else "regression",
                "version": "1.0.0",
                "created": pd.Timestamp.utcnow().isoformat(),
                "last_trained": pd.Timestamp.utcnow().isoformat(),
                "samples": int(num_rows),
                "features": int(len(self.feature_names)),
                "missing_pct": missing_ratio * 100.0,
                "duplicates_pct": duplicate_ratio * 100.0,
                "status": "Active",
                "health_score_pct": max(0.0, 100.0 - (missing_ratio * 100.0 * 0.5 + duplicate_ratio * 100.0 * 0.5)),
                "train_samples": len(self.X_train),
                "test_samples": len(self.X_test),
                "training_samples": len(self.X_train),  # Backward compatibility
                "test_samples": len(self.X_test),  # Backward compatibility
                "shap_available": self.explainer is not None,
            }

            # Add split info if available
            if split_info:
                self.model_info["split_info"] = split_info

            return {
                "status": "success",
                "message": message,
                "model_info": self.model_info,
                "features": self.feature_names,
                "target": self.target_name,
                "train_shape": self.X_train.shape,
                "test_shape": self.X_test.shape,
            }

        except Exception as e:
            # Reset state on failure
            logger.error(f"Error loading model and datasets in load_model_and_datasets function: {e}")
            self.__init__()
            raise e

    def _initialize_shap_explainer(self, model_wrapper: ModelWrapper):
        """Initialize SHAP explainer and compute SHAP values."""
        print("Creating SHAP explainer...")
        try:
            if model_wrapper.model_type == "sklearn":
                try:
                    self.explainer = shap.TreeExplainer(model_wrapper.model)
                except Exception as e:
                    print(f"Warning: Could not create SHAP TreeExplainer: {e}")
                    try:
                        # Fallback to general explainer
                        sample_size = min(100, len(self.X_train))
                        background_data = self.X_train.values[:sample_size]
                        self.explainer = shap.Explainer(model_wrapper.predict_proba, background_data)
                    except Exception as e2:
                        print(f"Warning: Could not create SHAP Explainer: {e2}")
                        try:
                            # Try KernelExplainer as final fallback
                            sample_size = min(50, len(self.X_train))
                            background_data = self.X_train.values[:sample_size]
                            self.explainer = shap.KernelExplainer(model_wrapper.predict_proba, background_data)
                        except Exception as e3:
                            print(f"Warning: Could not create SHAP KernelExplainer: {e3}")
                            self.explainer = None
            else:
                # For non-sklearn models, use a different SHAP explainer
                try:
                    # Use a smaller sample for initialization to avoid memory issues
                    sample_size = min(50, len(self.X_train))
                    background_data = self.X_train.values[:sample_size]
                    self.explainer = shap.Explainer(model_wrapper.predict_proba, background_data)
                except Exception as e:
                    print(f"Warning: Could not create SHAP explainer for {model_wrapper.model_type} model: {e}")
                    try:
                        # Try with even smaller sample
                        sample_size = min(10, len(self.X_train))
                        background_data = self.X_train.values[:sample_size]
                        self.explainer = shap.KernelExplainer(model_wrapper.predict_proba, background_data)
                    except Exception as e2:
                        print(f"Warning: Could not create SHAP KernelExplainer: {e2}")
                        self.explainer = None

            # Compute SHAP values if explainer is available
            if self.explainer:
                try:
                    print("Computing SHAP values...")
                    # Use a small sample for SHAP values to avoid memory/computation issues
                    sample_size = min(100, len(self.X_train))
                    self.shap_values = self.explainer.shap_values(self.X_train.values[:sample_size])
                    print(f"SHAP explainer created successfully with sample size {sample_size}.")

                    # Log the shape for debugging
                    if isinstance(self.shap_values, list):
                        print(
                            f"SHAP values computed as list with {len(self.shap_values)} classes, shapes: {[arr.shape for arr in self.shap_values]}"
                        )
                    else:
                        print(f"SHAP values computed with shape: {self.shap_values.shape}")

                except Exception as e:
                    print(f"Warning: Could not compute SHAP values: {e}")
                    self.shap_values = None
                    self.explainer = None
            else:
                self.shap_values = None
                print("SHAP explainer not available for this model type.")

        except Exception as e:
            print(f"Error initializing SHAP: {e}")
            self.explainer = None
            self.shap_values = None

    # === Utility methods used by other services ===

    def safe_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Safe prediction method that handles feature name compatibility."""
        self._is_ready()
        return self.model.predict(X)

    def safe_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Safe prediction probability method that handles feature name compatibility."""
        self._is_ready()
        return self.model.predict_proba(X)

    def _ensure_loaded_with_payload(self, payload: Dict[str, Any] = None, method: str = None):
        """Helper method to ensure model and data are loaded, with optional dynamic loading from payload."""

        if self.model is None or self.X_train is None or self.y_train is None:
            if payload is None:
                raise HTTPException(
                    status_code=400,
                    detail="Model and data must be loaded before performing analysis. Use the upload endpoints first or provide payload with 'model_url', 'dataset_url', and 'target_column'.",
                )

            print("⚠️ Model or data not loaded. Attempting to load dynamically from payload.")

            model_url = payload.get("model_url")
            dataset_url = payload.get("dataset_url")
            target_column = payload.get("target_column")

            if not model_url:
                raise HTTPException(status_code=400, detail="❌ 'model' path or URL is required.")
            if not dataset_url:
                raise HTTPException(status_code=400, detail="❌ 'dataset_url' is required.")
            if not target_column:
                raise HTTPException(status_code=400, detail="❌ 'target_column' is required.")

            # Load the model
            self.load_model_from_url(model_url)
            print("✅ Model loaded successfully.")

            # Load the dataset
            self.load_data_from_url(dataset_url, target_column)
            print("✅ Dataset loaded successfully.")

            # Initialize SHAP explainer now that both model and data are loaded
            if self.explainer is None and self.model is not None and self.X_train is not None:
                print("🔧 Initializing SHAP explainer after dynamic loading...")
                self._initialize_shap_explainer(self.model)

        # Ensure everything is ready
        self._is_ready()

    def _is_ready(self):
        """Check if the service has a model and data loaded."""
        if self.model is None or self.X_train is None or self.y_train is None:
            raise ValueError(
                "Model and data must be loaded before performing analysis. Use the upload endpoints first."
            )

    def get_model_input_format(self):
        """Get information about the expected input format for the model."""
        if self.model is None:
            return {"error": "No model loaded"}

        input_format = {
            "type": "tabular_data",
            "format": "pandas_dataframe",
            "features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names if self.feature_names else [],
            "required_columns": self.feature_names if self.feature_names else [],
            "data_types": {},
            "example_shape": f"({len(self.feature_names) if self.feature_names else 0},)",
        }

        # Add data type information if training data is available
        if self.X_train is not None:
            input_format["example_shape"] = f"(n_samples, {self.X_train.shape[1]})"
            for col in self.X_train.columns:
                dtype = str(self.X_train[col].dtype)
                if dtype.startswith("int"):
                    input_format["data_types"][col] = "integer"
                elif dtype.startswith("float"):
                    input_format["data_types"][col] = "float"
                elif dtype == "object":
                    input_format["data_types"][col] = "categorical"
                else:
                    input_format["data_types"][col] = dtype

        return input_format

    def _safe_float(self, value: Any) -> Any:
        """Convert value to float safely, handling non-numeric values."""
        try:
            return float(value)
        except Exception:
            return value

    def _sanitize_metrics(self, metrics: dict) -> dict:
        """Sanitize metrics to ensure JSON serialization compatibility."""
        import math

        sanitized = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    sanitized[key] = 0.0  # Replace NaN/inf with 0.0
                else:
                    sanitized[key] = float(value)
            else:
                sanitized[key] = value
        return sanitized

    def _get_classification_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate classification metrics handling both binary and multiclass cases."""

        try:
            # Determine if binary or multiclass
            unique_classes = np.unique(y_true)
            is_binary = len(unique_classes) == 2

            if is_binary:
                # Binary classification with zero_division handling
                metrics = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
                    "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
                    "f1_score": f1_score(y_true, y_pred, average="binary", zero_division=0),
                }

                # Add AUC for binary classification if probabilities are provided
                if y_proba is not None:
                    try:
                        if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                            metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
                        elif y_proba.ndim == 1:
                            metrics["auc"] = roc_auc_score(y_true, y_proba)
                        else:
                            metrics["auc"] = 0.0
                    except Exception as e:
                        print(f"Warning: Could not calculate AUC for binary classification: {e}")
                        metrics["auc"] = 0.0
                else:
                    metrics["auc"] = 0.0

            else:
                # Multiclass classification with zero_division handling
                metrics = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                    "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                }

                # For multiclass, use macro-averaged AUC if probabilities are provided
                if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    try:
                        metrics["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                    except Exception as e:
                        print(f"Warning: Could not calculate AUC for multiclass classification: {e}")
                        metrics["auc"] = 0.0
                else:
                    metrics["auc"] = 0.0

            # Sanitize all metrics to ensure JSON compatibility
            metrics = self._sanitize_metrics(metrics)
            return metrics, is_binary

        except Exception as e:
            print(f"Error calculating classification metrics: {e}")
            # Return default metrics if calculation fails
            default_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "auc": 0.0}
            return default_metrics, True  # Assume binary for fallback

    def _get_shap_matrix(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Return SHAP values as a 2D array shaped (n_samples, n_features) for analysis.
        Handles different SHAP return shapes including binary/multiclass cases.
        """
        if self.shap_values is None:
            return np.zeros((len(self.X_df), len(self.feature_names)))

        base = self.shap_values

        # Handle list/tuple (multiclass case)
        if isinstance(base, (list, tuple)):
            if len(base) == 2:
                # Binary classification - use positive class
                mat = base[1]
            else:
                # Multiclass - average across classes or use first class
                try:
                    mat = np.mean(np.array(base, dtype=object), axis=0)
                except:
                    mat = base[0]  # Fallback to first class

        # Handle numpy array
        elif isinstance(base, np.ndarray):
            if base.ndim == 3:  # (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
                if base.shape[0] == 2:
                    # Binary case: (2, n_samples, n_features)
                    mat = base[1]  # Positive class
                elif base.shape[2] == 2:
                    # Binary case: (n_samples, n_features, 2)
                    mat = base[:, :, 1]  # Positive class
                elif base.shape[0] < base.shape[2]:
                    # Shape: (n_classes, n_samples, n_features)
                    mat = base.mean(axis=0)
                else:
                    # Shape: (n_samples, n_features, n_classes)
                    mat = base.mean(axis=2)
            elif base.ndim == 2:
                # Already in correct format: (n_samples, n_features)
                mat = base
            else:
                # 1D or other - try to squeeze
                mat = base.squeeze()
        else:
            # Compute on the fly if available
            try:
                X_calc = X if X is not None else self.X_df
                if self.explainer is not None:
                    sv = self.explainer.shap_values(X_calc.values)
                    if isinstance(sv, (list, tuple)):
                        mat = sv[1] if len(sv) == 2 else np.mean(np.array(sv, dtype=object), axis=0)
                    else:
                        mat = sv
                else:
                    mat = np.zeros((len(self.X_df), len(self.feature_names)))
            except Exception as e:
                print(f"Warning: Could not compute SHAP matrix on demand: {e}")
                mat = np.zeros((len(self.X_df), len(self.feature_names)))

        # Ensure proper shape and type
        arr = np.asarray(mat)
        if arr.ndim > 2:
            # Reshape to 2D by flattening extra dimensions
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])
        elif arr.ndim == 1:
            # If somehow 1D, reshape assuming single sample
            arr = arr.reshape(1, -1)

        # Ensure we have the right number of features
        if arr.shape[1] != len(self.feature_names):
            print(
                f"Warning: SHAP matrix feature count ({arr.shape[1]}) doesn't match expected ({len(self.feature_names)})"
            )
            # Pad or truncate as necessary
            if arr.shape[1] < len(self.feature_names):
                padding = np.zeros((arr.shape[0], len(self.feature_names) - arr.shape[1]))
                arr = np.hstack([arr, padding])
            else:
                arr = arr[:, : len(self.feature_names)]

        return arr.astype(float)

    def _get_instance_shap_vector(self, instance_idx: int) -> np.ndarray:
        """Return a 1D array of SHAP values for the specified instance.
        Handles different shapes returned by SHAP depending on model/explainer versions.
        """
        if self.shap_values is None or self.explainer is None:
            return np.zeros(len(self.feature_names))

        try:
            # Handle list/tuple (most tree models return list per class)
            if isinstance(self.shap_values, (list, tuple)):
                # Use positive class for binary, first class for multiclass
                arr = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
                if instance_idx < len(arr):
                    vec = arr[instance_idx]
                    if vec.ndim > 1:
                        vec = vec.reshape(-1)
                    return vec
                else:
                    return np.zeros(len(self.feature_names))

            # Handle numpy array of shape (n_samples, n_features)
            if isinstance(self.shap_values, np.ndarray):
                if self.shap_values.ndim == 2 and instance_idx < self.shap_values.shape[0]:
                    return self.shap_values[instance_idx]
                elif self.shap_values.ndim == 3:
                    # Handle 3D array: (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
                    if self.shap_values.shape[0] < self.shap_values.shape[2]:
                        # Shape: (n_classes, n_samples, n_features)
                        class_idx = 1 if self.shap_values.shape[0] == 2 else 0
                        if instance_idx < self.shap_values.shape[1]:
                            return self.shap_values[class_idx, instance_idx, :]
                    else:
                        # Shape: (n_samples, n_features, n_classes)
                        class_idx = 1 if self.shap_values.shape[2] == 2 else 0
                        if instance_idx < self.shap_values.shape[0]:
                            return self.shap_values[instance_idx, :, class_idx]

            # Fallback: compute per-instance using explainer
            if instance_idx < len(self.X_df):
                try:
                    one = self.X_df.iloc[[instance_idx]]
                    sv = self.explainer.shap_values(one.values)
                    if isinstance(sv, (list, tuple)):
                        vec = sv[1][0] if len(sv) > 1 else sv[0][0]
                    else:
                        vec = sv[0] if sv.ndim > 1 else sv

                    if isinstance(vec, np.ndarray) and vec.ndim > 1:
                        vec = vec.reshape(-1)
                    return vec
                except Exception as e:
                    print(f"Warning: Could not compute SHAP values on-demand for instance {instance_idx}: {e}")

        except Exception as e:
            print(f"Error getting SHAP values for instance {instance_idx}: {e}")

        # Final fallback: return zeros
        return np.zeros(len(self.feature_names))

    def _get_shap_values_for_analysis(self) -> Optional[np.ndarray]:
        """Get SHAP values appropriate for analysis, handling both binary and multiclass cases."""
        if self.shap_values is None:
            return None

        if isinstance(self.shap_values, list):
            # Multiclass: use positive class (typically last) or average
            if len(self.shap_values) == 2:
                # Binary case stored as list
                return self.shap_values[1]
            else:
                # Multiclass: average across all classes
                return np.array(self.shap_values).mean(axis=0)
        elif len(self.shap_values.shape) == 3:
            # Multiclass: shape is (n_samples, n_features, n_classes)
            # Average across classes for analysis
            return self.shap_values.mean(axis=2)
        else:
            # Binary or already processed
            return self.shap_values

    def load_model_from_url(self, url: str):
        """Public method to load a model from a given URL."""
        try:
            self.model = self._load_model_from_presigned_url(url)
            print("✅ Model loaded successfully from URL.")
        except Exception as e:
            raise ValueError(f"Failed to load model from URL: {str(e)}")

    # def load_data_from_url(self, url: str, target_column: str):
    #     """Public method to load a dataset from a given URL."""
    #     import pandas as pd
    #     import requests
    #     from io import StringIO

    #     try:
    #         print(f"📥 Downloading dataset from URL: {url}")
    #         response = requests.get(url)
    #         response.raise_for_status()

    #         # Assume the dataset is in CSV format
    #         dataset = pd.read_csv(StringIO(response.text))
    #         print(f"✅ Dataset loaded successfully. Shape: {dataset.shape}")

    #         # Split into features and target
    #         if target_column not in dataset.columns:
    #             raise ValueError(f"Target column '{target_column}' not found in dataset.")

    #         self.X_train = dataset.drop(columns=[target_column])
    #         self.y_train = dataset[target_column]

    #         # For simplicity, assign the same data to test sets
    #         self.X_test = self.X_train.copy()
    #         self.y_test = self.y_train.copy()

    #         # Set up backward compatibility and metadata
    #         self.X_df = self.X_train
    #         self.y_s = self.y_train
    #         self.feature_names = list(self.X_train.columns)
    #         self.target_name = target_column

    #         print("✅ Dataset split into features and target successfully.")
    #     except Exception as e:
    #         raise ValueError(f"Failed to load dataset from URL: {str(e)}")
    def load_data_from_url(self, url: str, target_column: str):
        """Public method to load a dataset from a given URL (supports CSV, Parquet, Excel, JSON)."""

        try:
            print(f"Downloading dataset from URL: {url}")
            response = requests.get(url)
            response.raise_for_status()

            # Extract content type (optional)
            response.headers.get("Content-Type", "").lower()

            # Detect format primarily by URL extension
            if url.lower().endswith(".parquet"):
                print("📦 Detected Parquet format.")
                dataset = pd.read_parquet(BytesIO(response.content), engine="pyarrow")
            elif url.lower().endswith(".csv"):
                print("📄 Detected CSV format.")
                dataset = pd.read_csv(StringIO(response.text))
            elif url.lower().endswith((".xls", ".xlsx")):
                print("📘 Detected Excel format.")
                dataset = pd.read_excel(BytesIO(response.content))
            elif url.lower().endswith(".json"):
                print("🧾 Detected JSON format.")
                dataset = pd.read_json(StringIO(response.text))
            else:
                # Fallback check if the file might be parquet based on content
                if b"PAR1" in response.content[:4]:
                    print("📦 Fallback: Detected Parquet magic bytes.")
                    dataset = pd.read_parquet(BytesIO(response.content), engine="pyarrow")
                else:
                    print("⚠️ Unknown file type — trying CSV as fallback.")
                    dataset = pd.read_csv(StringIO(response.text))

            print(f"✅ Dataset loaded successfully. Shape: {dataset.shape}")

            # ✅ Validate target column
            if target_column not in dataset.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset.")

            # ✅ Split into features and target
            self.X_train = dataset.drop(columns=[target_column])
            self.y_train = dataset[target_column]

            # For simplicity, assign the same data to test sets
            self.X_test = self.X_train.copy()
            self.y_test = self.y_train.copy()

            # Backward compatibility and metadata
            self.X_df = self.X_train
            self.y_s = self.y_train
            self.feature_names = list(self.X_train.columns)
            self.target_name = target_column

            print("✅ Dataset split into features and target successfully.")
        except Exception as e:
            raise ValueError(f"Failed to load dataset from URL: {str(e)}")
