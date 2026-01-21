import os
import pickle
import tempfile
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
import requests
import shap
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# We only support scikit-learn models to preserve tree structure for explainability


class ModelWrapper:
    """Wrapper class to provide consistent interface for scikit-learn models."""

    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
        self._feature_names_expected = None  # Will be determined during first prediction

    def _detect_feature_names_requirement(self, X):
        """
        Detect whether the model expects feature names or not by testing both approaches.
        This is done only once and cached for performance.
        """
        if self._feature_names_expected is not None:
            return self._feature_names_expected

        if not hasattr(X, "values"):  # Already numpy array
            self._feature_names_expected = False
            return False

        # Test both approaches to see which one works without warnings
        import warnings

        # Test with DataFrame (feature names)
        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always")
            try:
                # Use predict for regression models only
                _ = self.model.predict(X.iloc[:1])
                df_warnings = len(w1)
            except:
                df_warnings = float("inf")  # Failed completely

        # Test with numpy array (no feature names)
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            try:
                _ = self.model.predict(X.iloc[:1].values)
                array_warnings = len(w2)
            except:
                array_warnings = float("inf")  # Failed completely

        # Choose the approach with fewer warnings
        if df_warnings <= array_warnings:
            self._feature_names_expected = True
            print(f"📊 Model expects feature names (DataFrame input) - will use pandas DataFrames for predictions")
        else:
            self._feature_names_expected = False
            print(f"📊 Model expects numpy arrays (no feature names) - will use .values for predictions")

        return self._feature_names_expected

    def _prepare_input(self, X):
        """Prepare input data in the format expected by the model."""
        if not hasattr(X, "values"):  # Already numpy array
            return X

        expects_feature_names = self._detect_feature_names_requirement(X)

        if expects_feature_names:
            return X  # Return DataFrame as-is
        else:
            return X.values  # Convert to numpy array

    def predict(self, X):
        """Predict class labels with adaptive input format."""
        if self.model_type == "sklearn":
            prepared_X = self._prepare_input(X)
            return self.model.predict(prepared_X)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict_proba(self, X):
        """Regression models don't have probabilities. Always returns None."""
        return None

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        return getattr(self.model, name)


class BaseModelService:
    """
    Base service that holds the loaded model and shared data/state.
    Other services will inherit from this or receive an instance.
    """

    def __init__(self):
        # State: These will be populated when files are uploaded
        self.model: Optional[Any] = None

        # Separate train and test datasets
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None

        # Keep for backward compatibility and general analysis
        self.X_df: Optional[pd.DataFrame] = None  # Will point to train data
        self.y_s: Optional[pd.Series] = None  # Will point to train data

        self.feature_names: Optional[List[str]] = None
        self.target_name: Optional[str] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self._cached_predictions: Optional[np.ndarray] = None
        self.model_info: Dict[str, Any] = {}

        print("BaseModelService initialized. Waiting for model and data.")

    def _load_model_by_format(self, model_path: str) -> ModelWrapper:
        """Load model based on file extension and return wrapped model."""

        # Check if it's a URL (S3 pre-signed URL)
        if model_path.startswith(("http://", "https://")):
            return self._load_model_from_presigned_url(model_path)

        # Handle local file paths
        file_extension = model_path.lower()

        if file_extension.endswith((".joblib", ".pkl", ".pickle")):
            # Scikit-learn models (joblib or pickle)
            try:
                if file_extension.endswith(".joblib"):
                    model = joblib.load(model_path)
                else:
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                return ModelWrapper(model, "sklearn")
            except Exception as e:
                raise ValueError(f"Failed to load sklearn model from {model_path}: {str(e)}")
        else:
            raise ValueError(f"Unsupported model format. Expected .joblib, .pkl, or .pickle, got {file_extension}")

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

    def _load_data_from_url(self, url: str) -> pd.DataFrame:
        """Load CSV data from URL with error handling."""

        print(f"📥 Loading data from URL...")

        try:
            # Pandas can read directly from URLs, but let's add some validation
            parsed = urlparse(url)

            # if not parsed.path.lower().endswith('.csv'):
            #     print(f"⚠️  Warning: URL doesn't end with .csv: {parsed.path}")
            file_path = parsed.path.lower()

            if file_path.endswith(".parquet"):
                print(f"Detected Parquet file: {file_path}")
                df = pd.read_parquet(url, engine="pyarrow")
            elif file_path.endswith(".csv"):
                print(f"Detected CSV file: {file_path}")
                df = pd.read_csv(url)
            else:
                raise ValueError(f"Unsupported file type in URL: {file_path}. Must be .csv or .parquet")

            print(f"✅ Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df

        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file at the URL is empty")
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV from URL: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error while downloading CSV from URL: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to load data from URL: {str(e)}")

    def _detect_model_framework(self, model_wrapper: ModelWrapper) -> str:
        """Detect the framework used to create the model."""
        if model_wrapper.model_type == "sklearn":
            return "scikit-learn"
        else:
            return "unknown"

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
        from sklearn.model_selection import train_test_split

        try:
            # Validate input parameters
            if not (train_data_path and test_data_path) and not data_path:
                raise ValueError("Must provide either data_path OR both train_data_path and test_data_path")

            # Load model
            if model_path.startswith("https://") and "s3.amazonaws.com" in model_path:
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

                # Load dataset
                if data_path.startswith(("http://", "https://")):
                    print(f"⬇️ Loading dataset from URL: {data_path}")
                    df = self._load_data_from_url(data_path)
                else:
                    print(f"📁 Loading dataset from local path: {data_path}")
                    df = pd.read_csv(data_path)

                # Validate target column exists
                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in the dataset.")

                # Split features and target
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Split into train/test (no stratification for regression)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

                message = f"Model and dataset loaded successfully (split into train/test)"
                split_info = f"test_size={test_size}, random_state={random_state}"

            # Handle separate datasets case
            else:
                print("📊 Separate datasets mode: using provided train/test files")

                # Load training dataset
                if train_data_path.startswith(("http://", "https://")):
                    print(f"⬇️ Loading training data from URL: {train_data_path}")
                    train_df = self._load_data_from_url(train_data_path)
                else:
                    print(f"📁 Loading training data from local path: {train_data_path}")
                    train_df = pd.read_csv(train_data_path)

                # Load test dataset
                if test_data_path.startswith(("http://", "https://")):
                    print(f"⬇️ Loading test data from URL: {test_data_path}")
                    test_df = self._load_data_from_url(test_data_path)
                else:
                    print(f"📁 Loading test data from local path: {test_data_path}")
                    test_df = pd.read_csv(test_data_path)

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
            self.__init__()
            raise e

    def load_model_and_data(self, model_path: str, data_path: str, target_column: str):
        """Loads the model and dataset from local files and prepares for analysis."""
        try:
            # Load model
            model_wrapper = self._load_model_by_format(model_path)
            self.model = model_wrapper

            # Load data
            if data_path.startswith(("http://", "https://")):
                print(f"⬇️ Loading data from URL: {data_path}")
                df = self._load_data_from_url(data_path)
            else:
                if not data_path.endswith(".csv"):
                    raise ValueError("Only CSV data files are supported.")
                print(f"📁 Loading data from local path: {data_path}")
                df = pd.read_csv(data_path)

            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset.")

            # Store metadata
            self.model_info = {
                "model_path": model_path,
                "data_path": data_path,
                "target_column": target_column,
                "framework": self._detect_model_framework(model_wrapper),
                "algorithm": self._get_model_algorithm(model_wrapper),
                "version": "1.0.0",
                "status": "Active",
                "created": pd.Timestamp.utcnow().isoformat(),
                "last_trained": pd.Timestamp.utcnow().isoformat(),
                "data_source": "single_dataset",
            }

            # Split features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Store feature information
            self.feature_names = X.columns.tolist()
            self.target_name = target_column

            # Split into train/test (80/20 split)
            from sklearn.model_selection import train_test_split

            # For regression, don't use stratification
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Store datasets
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test

            # For backward compatibility, point to training data
            self.X_df = X_train
            self.y_s = y_train

            # Update metadata with split information
            self.model_info.update(
                {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "missing_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                    "duplicates_pct": (df.duplicated().sum() / len(df)) * 100,
                    "health_score_pct": 100.0
                    - self.model_info.get("missing_pct", 0)
                    - self.model_info.get("duplicates_pct", 0),
                }
            )

            # Initialize SHAP explainer based on model type
            try:
                if model_wrapper.model_type == "sklearn":
                    # Check for tree-based models first
                    if hasattr(model_wrapper.model, "tree_") or hasattr(model_wrapper.model, "estimators_"):
                        print("Initializing TreeExplainer for tree-based model...")
                        self.explainer = shap.TreeExplainer(model_wrapper.model)
                    # Check for linear models
                    elif hasattr(model_wrapper.model, "coef_"):
                        print("Initializing LinearExplainer for linear model...")
                        self.explainer = shap.LinearExplainer(model_wrapper.model, X_train)
                    # Fallback to general explainer for other model types
                    else:
                        print("Initializing general Explainer for other model types...")
                        # Use a sample of background data for efficiency
                        background_size = min(100, len(X_train))
                        background_data = X_train.sample(n=background_size, random_state=42)
                        self.explainer = shap.Explainer(model_wrapper.predict, background_data)

                    # Compute SHAP values on a sample for efficiency
                    sample_size = min(1000, len(X_train))
                    sample_X = X_train.sample(n=sample_size, random_state=42)
                    self.shap_values = self.explainer.shap_values(sample_X)
                    print(f"SHAP explainer initialized successfully with {sample_size} samples.")
                else:
                    print(f"SHAP explainer not supported for model type: {model_wrapper.model_type}")
                    self.explainer = None
                    self.shap_values = None
            except Exception as e:
                print(f"Failed to initialize SHAP explainer: {e}")
                import traceback

                traceback.print_exc()
                self.explainer = None
                self.shap_values = None

            # Cache predictions for the training set for faster individual predictions
            try:
                print("Caching predictions for training set...")
                self._cached_predictions = self.safe_predict(self.X_df)
                print(f"Cached {len(self._cached_predictions)} predictions.")
            except Exception as e:
                print(f"Failed to cache predictions: {e}")
                self._cached_predictions = None

            return {
                "message": "Model and data loaded successfully",
                "model_info": {
                    "framework": self.model_info["framework"],
                    "algorithm": self.model_info["algorithm"],
                    "features": len(self.feature_names),
                    "samples": len(df),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "target_column": target_column,
                    "shap_available": self.explainer is not None,
                },
            }

        except Exception as e:
            raise ValueError(f"Failed to load model and data: {str(e)}")

    def load_model_and_separate_datasets(
        self, model_path: str, train_data_path: str, test_data_path: str, target_column: str
    ):
        """Loads the model and separate train/test datasets from local files and prepares for analysis."""
        try:
            # Load model
            model_wrapper = self._load_model_by_format(model_path)
            self.model = model_wrapper

            # Load training data
            if train_data_path.startswith(("http://", "https://")):
                print(f"⬇️ Loading training data from URL: {train_data_path}")
                train_df = self._load_data_from_url(train_data_path)
            else:
                if not train_data_path.endswith(".csv"):
                    raise ValueError("Only CSV data files are supported.")
                print(f"📁 Loading training data from local path: {train_data_path}")
                train_df = pd.read_csv(train_data_path)

            if target_column not in train_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in training dataset.")

            # Load test data
            if test_data_path.startswith(("http://", "https://")):
                print(f"⬇️ Loading test data from URL: {test_data_path}")
                test_df = self._load_data_from_url(test_data_path)
            else:
                if not test_data_path.endswith(".csv"):
                    raise ValueError("Only CSV data files are supported.")
                print(f"📁 Loading test data from local path: {test_data_path}")
                test_df = pd.read_csv(test_data_path)

            if target_column not in test_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in test dataset.")

            # Store metadata
            self.model_info = {
                "model_path": model_path,
                "train_data_path": train_data_path,
                "test_data_path": test_data_path,
                "target_column": target_column,
                "framework": self._detect_model_framework(model_wrapper),
                "algorithm": self._get_model_algorithm(model_wrapper),
                "version": "1.0.0",
                "status": "Active",
                "created": pd.Timestamp.utcnow().isoformat(),
                "last_trained": pd.Timestamp.utcnow().isoformat(),
                "data_source": "separate_datasets",
            }

            # Split features and target for training data
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            # Split features and target for test data
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Ensure feature consistency between train and test
            if set(X_train.columns) != set(X_test.columns):
                raise ValueError("Training and test datasets have different features.")

            # Reorder test features to match training order
            X_test = X_test[X_train.columns]

            # Store feature information
            self.feature_names = X_train.columns.tolist()
            self.target_name = target_column

            # Store datasets
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test

            # For backward compatibility, point to training data
            self.X_df = X_train
            self.y_s = y_train

            # Update metadata with dataset information
            missing_train_pct = (train_df.isnull().sum().sum() / (train_df.shape[0] * train_df.shape[1])) * 100
            missing_test_pct = (test_df.isnull().sum().sum() / (test_df.shape[0] * test_df.shape[1])) * 100
            duplicates_train_pct = (train_df.duplicated().sum() / len(train_df)) * 100
            duplicates_test_pct = (test_df.duplicated().sum() / len(test_df)) * 100

            self.model_info.update(
                {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "missing_pct": (missing_train_pct + missing_test_pct) / 2,
                    "duplicates_pct": (duplicates_train_pct + duplicates_test_pct) / 2,
                    "health_score_pct": 100.0
                    - ((missing_train_pct + missing_test_pct) / 2)
                    - ((duplicates_train_pct + duplicates_test_pct) / 2),
                }
            )

            # Initialize SHAP explainer for tree-based models
            try:
                if model_wrapper.model_type == "sklearn" and (
                    hasattr(model_wrapper.model, "tree_") or hasattr(model_wrapper.model, "estimators_")
                ):
                    self.explainer = shap.TreeExplainer(model_wrapper.model)
                    # Compute SHAP values on a sample for efficiency
                    sample_size = min(1000, len(X_train))
                    sample_X = X_train.sample(n=sample_size, random_state=42)
                    self.shap_values = self.explainer.shap_values(sample_X)
                    print(f"SHAP explainer initialized with {sample_size} samples.")
                else:
                    print("SHAP explainer not initialized (model type not supported or not tree-based)")
            except Exception as e:
                print(f"Failed to initialize SHAP explainer: {e}")
                self.explainer = None
                self.shap_values = None

            return {
                "message": "Model and separate datasets loaded successfully",
                "model_info": {
                    "framework": self.model_info["framework"],
                    "algorithm": self.model_info["algorithm"],
                    "features": len(self.feature_names),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "target_column": target_column,
                    "shap_available": self.explainer is not None,
                },
            }

        except Exception as e:
            raise ValueError(f"Failed to load model and separate datasets: {str(e)}")

    def _get_regression_metrics(self, y_true, y_pred):
        """Calculate regression metrics for model evaluation."""
        # Convert to numpy arrays for consistent handling
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # R² Score (Coefficient of determination)
        r2 = r2_score(y_true, y_pred)

        # Mean Squared Error
        mse = mean_squared_error(y_true, y_pred)

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Mean Absolute Error
        mae = mean_absolute_error(y_true, y_pred)

        # Mean Absolute Percentage Error (handle division by zero)
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except:
            # Fallback calculation for MAPE if sklearn doesn't have it or fails
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100

        # Symmetric Mean Absolute Percentage Error (SMAPE)
        denominator = np.abs(y_true) + np.abs(y_pred)
        # Avoid division by zero
        smape = np.mean(np.where(denominator == 0, 0, 2.0 * np.abs(y_pred - y_true) / denominator)) * 100

        # Adjusted R² Score
        n_samples = len(y_true)
        n_features = len(self.feature_names) if hasattr(self, "feature_names") and self.feature_names else 1
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

        # Explained Variance
        explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)

        def safe_float(val):
            try:
                f = float(val)
                if np.isnan(f) or np.isinf(f):
                    return None
                return f
            except Exception:
                return None

        return {
            "r2_score": safe_float(r2),
            "rmse": safe_float(rmse),
            "mse": safe_float(mse),
            "mae": safe_float(mae),
            "mape": safe_float(mape),
            "smape": safe_float(smape),
            "adjusted_r2": safe_float(adj_r2),
            "explained_variance": safe_float(explained_variance),
        }

    def _is_regression_model(self) -> bool:
        """All models are now regression models by design."""
        return True

    def _is_regression_model_from_data(self, y: pd.Series) -> bool:
        """All data is now treated as regression by design."""
        return True

    def safe_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Safe prediction method that automatically handles feature name compatibility.
        Use this method instead of calling model.predict() directly.
        """
        self._is_ready()
        return self.model.predict(X)

    def safe_predict_proba(self, X: pd.DataFrame) -> None:
        """
        Regression models don't have probabilities. Always returns None.
        """
        return None

    def get_model_input_format(self) -> str:
        """
        Get information about what input format the model expects.
        Returns: 'dataframe' if model expects feature names, 'array' if it expects numpy arrays.
        """
        if self.model is None:
            return "unknown"

        if hasattr(self.model, "_feature_names_expected"):
            if self.model._feature_names_expected is True:
                return "dataframe"
            elif self.model._feature_names_expected is False:
                return "array"

        return "unknown"

    def _is_ready(self):
        """Check if the service has a model and data loaded."""
        if self.model is None or self.X_train is None or self.y_train is None:
            raise ValueError(
                "Model and data must be loaded before performing analysis. Use the upload endpoints first."
            )

    def _safe_float(self, value: Any) -> Any:
        """Convert value to float safely, handling non-numeric values."""
        try:
            return float(value)
        except Exception:
            return value

    def _initialize_shap_explainer(self, model_wrapper: ModelWrapper):
        """Initialize SHAP explainer and compute SHAP values."""
        print("Creating SHAP explainer...")
        try:
            if model_wrapper.model_type == "sklearn":
                # Check for tree-based models first
                if hasattr(model_wrapper.model, "tree_") or hasattr(model_wrapper.model, "estimators_"):
                    print("Trying TreeExplainer for tree-based model...")
                    self.explainer = shap.TreeExplainer(model_wrapper.model)
                # Check for linear models
                elif hasattr(model_wrapper.model, "coef_"):
                    print("Trying LinearExplainer for linear model...")
                    self.explainer = shap.LinearExplainer(model_wrapper.model, self.X_train)
                else:
                    print("Trying general explainer fallback...")
                    # Fallback to general explainer
                    sample_size = min(100, len(self.X_train))
                    background_data = self.X_train.values[:sample_size]
                    self.explainer = shap.Explainer(model_wrapper.predict, background_data)
            else:
                # For non-sklearn models, use a different SHAP explainer
                try:
                    # Use a smaller sample for initialization to avoid memory issues
                    sample_size = min(50, len(self.X_train))
                    background_data = self.X_train.values[:sample_size]
                    self.explainer = shap.Explainer(model_wrapper.predict, background_data)
                except Exception as e:
                    print(f"Warning: Could not create SHAP explainer for {model_wrapper.model_type} model: {e}")
                    try:
                        # Try with even smaller sample
                        sample_size = min(10, len(self.X_train))
                        background_data = self.X_train.values[:sample_size]
                        self.explainer = shap.KernelExplainer(model_wrapper.predict, background_data)
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

    def _get_shap_values_for_analysis(self) -> Optional[np.ndarray]:
        """Get SHAP values appropriate for analysis, handling both binary and multiclass cases."""
        if self.shap_values is None:
            return None

        # Handle different SHAP value formats
        if isinstance(self.shap_values, list):
            # Multi-class case: SHAP returns list of arrays, one per class
            # For binary classification, take the positive class (index 1)
            if len(self.shap_values) == 2:
                return self.shap_values[1]  # Positive class
            else:
                # Multi-class: average across classes or take first class
                return self.shap_values[0]
        elif len(self.shap_values.shape) == 3:
            # 3D array: (n_samples, n_features, n_classes)
            # For binary, take positive class; for multi-class, take first class
            if self.shap_values.shape[2] == 2:
                return self.shap_values[:, :, 1]  # Positive class
            else:
                return self.shap_values[:, :, 0]  # First class
        else:
            # 2D array: (n_samples, n_features) - already in correct format
            return self.shap_values

    def _get_shap_matrix(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Return SHAP values as a 2D array shaped (n_samples, n_features) for the positive class or
        averaged across classes when applicable. Handles different SHAP return shapes.
        """
        base = self.shap_values
        if isinstance(base, (list, tuple)):
            # Multi-class: list of arrays
            if len(base) >= 2:
                mat = base[1]  # Positive class for binary
            else:
                mat = base[0]  # First class
        elif isinstance(base, np.ndarray):
            if base.ndim == 3:
                # 3D: take positive class or first class
                if base.shape[2] >= 2:
                    mat = base[:, :, 1]  # Positive class
                else:
                    mat = base[:, :, 0]  # First class
            else:
                mat = base  # Already 2D
        else:
            # Fallback: create zeros
            mat = np.zeros((len(self.X_df), len(self.feature_names)))

        arr = np.asarray(mat)
        if arr.ndim > 2:
            # Additional safeguard: flatten extra dimensions
            arr = arr.reshape(arr.shape[0], -1)[:, : len(self.feature_names)]
        return arr.astype(float)

    def _get_instance_shap_vector(self, instance_idx: int) -> np.ndarray:
        """Return a 1D array of SHAP values for the specified instance, aligned with feature_names.
        Handles different shapes returned by SHAP depending on model/explainer versions.
        """
        if self.shap_values is None or self.explainer is None:
            return np.zeros(len(self.feature_names))

        try:
            # Get the SHAP matrix and extract the instance
            shap_matrix = self._get_shap_matrix()
            if instance_idx < len(shap_matrix):
                return shap_matrix[instance_idx]
            else:
                return np.zeros(len(self.feature_names))
        except Exception as e:
            print(f"Error getting SHAP values for instance {instance_idx}: {e}")
            return np.zeros(len(self.feature_names))
