"""
S3 utilities for loading data and models from external S3 API
"""

import logging
import os
import pickle
from io import BytesIO
from typing import Any, Dict

import joblib
import pandas as pd
import requests
from fastapi import HTTPException

logger = logging.getLogger(__name__)


# --- Metadata retrieval ---
def get_s3_file_metadata(analysis_type: str, project_id: str, token) -> Dict[str, Any]:
    """
    Lists files and models from the external S3 API and returns their metadata.
    Separates files and models based on the folder field.
    """
    file_api = os.getenv("FILES_API_BASE_URL")

    if not project_id:
        raise HTTPException(status_code=400, detail="Project ID is required.")
    if not file_api:
        raise HTTPException(status_code=500, detail="FILES_API_BASE_URL env variable is not set.")
    if not token:
        raise HTTPException(status_code=401, detail="User token is missing or invalid.")

    url = f"{file_api}/{analysis_type}/{project_id}"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        json_data = resp.json()
        all_items = json_data.get("files", [])

        return {
            "files": [i for i in all_items if i.get("folder") == "files"],
            "models": [i for i in all_items if i.get("folder") == "models"],
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"S3 API connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to S3 API: {str(e)}")
    except Exception as e:
        logger.error(f"S3 API response error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing S3 API response: {str(e)}")


# --- CSV loading ---
def load_s3_csv(url: str) -> pd.DataFrame:
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(BytesIO(resp.content))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        logger.info(f"Loaded CSV from {url}: shape={df.shape}")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download CSV file: {str(e)}")
    except pd.errors.ParserError as e:
        logger.error(f"Invalid CSV format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected CSV load error: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading CSV file: {str(e)}")


# --- Model loading ---
def load_s3_model(url: str) -> Any:
    """
    Load a model from S3 URL (supports pickle, joblib, and ONNX formats).
    This tries to return the same object you saved (pipeline or estimator).

    Includes comprehensive error handling for STACK_GLOBAL errors and file integrity checks.
    """
    if not url:
        raise HTTPException(status_code=400, detail="Model URL is required")

    # pre-import common ML classes for pickle
    _prime_globals_for_sklearn()

    try:
        # Download model file with validation
        resp = requests.get(url, timeout=60, stream=False)
        resp.raise_for_status()

        # FIX #3: File integrity checks
        content = resp.content
        if len(content) == 0:
            raise ValueError("Downloaded model file is empty (0 bytes)")
        if len(content) < 100:  # Minimum viable pickle is ~100 bytes
            raise ValueError(f"Model file too small ({len(content)} bytes) - likely corrupted or incomplete")

        # Check if it looks like a pickle file (magic bytes)
        if len(content) >= 2 and not (
            content[:2] in [b"\x80\x00", b"\x80\x01", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"]
        ):
            raise ValueError(
                "File does not appear to be a valid pickle file (wrong magic bytes). Check if correct file was uploaded."
            )

        # FIX #5: Log pickle protocol for debugging
        if len(content) >= 2:
            protocol = content[1]
            logger.info(f"Model uses pickle protocol {protocol}")

            if protocol == 5:
                logger.warning(
                    "⚠️  Protocol 5 detected. Requires Python 3.8+ and may have "
                    "stricter class resolution. If loading fails, re-save model "
                    "with protocol=4 for better compatibility."
                )
            elif protocol > 5:
                raise ValueError(f"Unsupported pickle protocol {protocol}. Max supported: 5")

        content_io = BytesIO(content)
        ext = url.split(".")[-1].lower().split("?")[0]  # Handle query params in URL

        if ext == "onnx":
            return _load_onnx(content_io, url)

        # FIX #4: Enhanced error handling with encoding fallback
        # Try joblib first (it's faster and safer for sklearn models)
        model = None
        joblib_error = None

        try:
            content_io.seek(0)
            model = joblib.load(content_io)
            logger.info(f"✅ Loaded model via joblib (protocol {protocol})")
        except Exception as e_joblib:
            joblib_error = e_joblib
            error_type = type(e_joblib).__name__
            error_msg = str(e_joblib)
            logger.error(f"❌ Joblib load failed: {error_type}: {error_msg}")

            # Diagnose STACK_GLOBAL specifically
            if "STACK_GLOBAL" in error_msg:
                missing_class = _extract_missing_class_from_error(error_msg)
                logger.error(f"🔍 STACK_GLOBAL error - likely missing class: {missing_class}")
                logger.error("💡 Solution: The model contains a preprocessing class not pre-imported.")
                logger.error("💡 Check _prime_globals_for_sklearn() function and add the missing class.")

            # Try pickle with multiple encoding strategies
            logger.info("Attempting pickle load with encoding fallbacks...")
            for encoding in ["latin1", "utf-8", "bytes", None]:
                try:
                    content_io.seek(0)
                    if encoding:
                        model = pickle.load(content_io, encoding=encoding)
                        logger.info(f"✅ Loaded model via pickle with encoding={encoding}")
                    else:
                        model = pickle.load(content_io)
                        logger.info(f"✅ Loaded model via pickle (default encoding)")
                    break
                except Exception as e_pickle:
                    if encoding is None:  # Last attempt failed
                        logger.error(f"❌ All load attempts failed.")
                        logger.error(f"Joblib error: {joblib_error}")
                        logger.error(f"Pickle error: {e_pickle}")

                        # Provide helpful error message
                        detail_msg = f"Model loading failed. "
                        if "STACK_GLOBAL" in str(joblib_error):
                            detail_msg += "STACK_GLOBAL error indicates missing preprocessing classes (e.g., StandardScaler, RobustScaler). "
                            detail_msg += (
                                "Common causes: version mismatch, custom transformers, or missing sklearn imports. "
                            )
                        else:
                            detail_msg += f"Error: {error_type}. "
                            detail_msg += (
                                "Common causes: version mismatch, file corruption, or incompatible pickle protocol. "
                            )

                        detail_msg += f"Details: {str(joblib_error)[:200]}"
                        raise HTTPException(status_code=500, detail=detail_msg)
                    continue

        # Unwrap if model is stored in dictionary
        model = _unwrap_model_dicts(model)

        # Validate model has predict method
        if not hasattr(model, "predict"):
            raise HTTPException(
                status_code=400,
                detail="Loaded model does not expose a 'predict' method. "
                "Ensure you saved a full pipeline or wrap preprocessing.",
            )
        return model

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading model from {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download model file: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected model load error from {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model file: {str(e)}")


def _extract_missing_class_from_error(error_msg: str) -> str:
    """Extract likely missing class name from STACK_GLOBAL error message."""
    import re

    # Try to find module.ClassName pattern in error message
    match = re.search(r"'([\w\.]+)'", error_msg)
    if match:
        return match.group(1)
    return "unknown (check traceback for details)"


def _prime_globals_for_sklearn():
    """
    Pre-import ALL sklearn/xgboost/lightgbm/catboost classes into globals for pickle.
    This prevents STACK_GLOBAL errors when unpickling models with preprocessing pipelines.
    Expanded to include 50+ classes commonly used in regression and classification pipelines.
    """
    try:
        # Core sklearn imports
        import sklearn
        import sklearn.ensemble
        import sklearn.linear_model
        import sklearn.naive_bayes
        import sklearn.neighbors
        import sklearn.neural_network
        import sklearn.svm
        import sklearn.tree

        # Import internal modules that are used by GradientBoosting and other ensemble models
        # These imports are critical for unpickling models that use internal sklearn components
        try:
            # Try newer sklearn structure (1.0+)
            import sklearn.ensemble._gb_losses
            import sklearn.ensemble._gradient_boosting
        except (ImportError, AttributeError):
            try:
                # Try older sklearn structure (0.24.x)
                import sklearn.ensemble.gradient_boosting
            except ImportError:
                pass

        try:
            # Tree-related internal modules
            import sklearn.tree._criterion
            import sklearn.tree._splitter
            import sklearn.tree._tree
        except (ImportError, AttributeError):
            pass

        # Import loss functions module (common source of pickle errors)
        try:
            from sklearn.ensemble import _loss
        except ImportError:
            try:
                from sklearn.ensemble import _gb as _loss  # Older versions
            except ImportError:
                pass  # Very old sklearn versions

        # Preprocessing imports (CRITICAL for regression pipelines)
        # Imputation (CRITICAL for pipelines with missing data)
        from sklearn.impute import KNNImputer, SimpleImputer
        from sklearn.preprocessing import (
            Binarizer,
            FunctionTransformer,
            LabelBinarizer,
            LabelEncoder,
            MaxAbsScaler,
            MinMaxScaler,
            Normalizer,
            OneHotEncoder,
            OrdinalEncoder,
            PolynomialFeatures,
            PowerTransformer,
            QuantileTransformer,
            RobustScaler,
            StandardScaler,
        )

        try:
            from sklearn.impute import IterativeImputer
        except ImportError:
            IterativeImputer = None

        # Composition (CRITICAL for complex pipelines)
        from sklearn.compose import ColumnTransformer, TransformedTargetRegressor

        # Decomposition
        from sklearn.decomposition import PCA, TruncatedSVD
        from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor

        # Feature selection
        from sklearn.feature_selection import (
            RFE,
            RFECV,
            SelectFromModel,
            SelectKBest,
            SelectPercentile,
            VarianceThreshold,
        )

        # All regressors (CRITICAL for regression model support)
        from sklearn.linear_model import (
            BayesianRidge,
            ElasticNet,
            HuberRegressor,
            Lars,
            Lasso,
            LassoLars,
            LinearRegression,
            OrthogonalMatchingPursuit,
            RANSACRegressor,
            Ridge,
            TheilSenRegressor,
        )
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import FeatureUnion, Pipeline

        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError:
            HistGradientBoostingRegressor = None

        # Update globals with ALL classes
        globals_dict = {
            # Pipeline components
            "Pipeline": Pipeline,
            "FeatureUnion": FeatureUnion,
            "ColumnTransformer": ColumnTransformer,
            "TransformedTargetRegressor": TransformedTargetRegressor,
            # Preprocessing (scaling/transformation)
            "StandardScaler": StandardScaler,
            "RobustScaler": RobustScaler,
            "MinMaxScaler": MinMaxScaler,
            "MaxAbsScaler": MaxAbsScaler,
            "Normalizer": Normalizer,
            "Binarizer": Binarizer,
            "QuantileTransformer": QuantileTransformer,
            "PowerTransformer": PowerTransformer,
            "PolynomialFeatures": PolynomialFeatures,
            "FunctionTransformer": FunctionTransformer,
            # Encoding
            "LabelEncoder": LabelEncoder,
            "OneHotEncoder": OneHotEncoder,
            "OrdinalEncoder": OrdinalEncoder,
            "LabelBinarizer": LabelBinarizer,
            # Imputation
            "SimpleImputer": SimpleImputer,
            "KNNImputer": KNNImputer,
            # Feature selection
            "SelectKBest": SelectKBest,
            "SelectPercentile": SelectPercentile,
            "RFE": RFE,
            "RFECV": RFECV,
            "VarianceThreshold": VarianceThreshold,
            "SelectFromModel": SelectFromModel,
            # Decomposition
            "PCA": PCA,
            "TruncatedSVD": TruncatedSVD,
            # Classifiers
            "RandomForestClassifier": sklearn.ensemble.RandomForestClassifier,
            "GradientBoostingClassifier": sklearn.ensemble.GradientBoostingClassifier,
            "LogisticRegression": sklearn.linear_model.LogisticRegression,
            "SVC": sklearn.svm.SVC,
            "DecisionTreeClassifier": sklearn.tree.DecisionTreeClassifier,
            # Regressors (ALL common ones)
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet,
            "BayesianRidge": BayesianRidge,
            "HuberRegressor": HuberRegressor,
            "RANSACRegressor": RANSACRegressor,
            "TheilSenRegressor": TheilSenRegressor,
            "Lars": Lars,
            "LassoLars": LassoLars,
            "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit,
            "RandomForestRegressor": sklearn.ensemble.RandomForestRegressor,
            "GradientBoostingRegressor": sklearn.ensemble.GradientBoostingRegressor,
            "SVR": sklearn.svm.SVR,
            "DecisionTreeRegressor": sklearn.tree.DecisionTreeRegressor,
            "KNeighborsRegressor": KNeighborsRegressor,
            "MLPRegressor": MLPRegressor,
            "AdaBoostRegressor": AdaBoostRegressor,
            "BaggingRegressor": BaggingRegressor,
            "ExtraTreesRegressor": ExtraTreesRegressor,
        }

        # Add optional classes
        if IterativeImputer:
            globals_dict["IterativeImputer"] = IterativeImputer
        if HistGradientBoostingRegressor:
            globals_dict["HistGradientBoostingRegressor"] = HistGradientBoostingRegressor

        globals().update(globals_dict)
        logger.info(f"Primed globals with {len(globals_dict)} sklearn classes for pickle compatibility")

    except ImportError as e:
        logger.warning(f"Could not import some sklearn classes: {e}")

    # XGBoost
    try:
        from xgboost import XGBClassifier, XGBRegressor

        globals().update({"XGBClassifier": XGBClassifier, "XGBRegressor": XGBRegressor})
        logger.info("Added XGBoost classes to globals")
    except ImportError:
        pass

    # LightGBM
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        globals().update({"LGBMClassifier": LGBMClassifier, "LGBMRegressor": LGBMRegressor})
        logger.info("Added LightGBM classes to globals")
    except ImportError:
        pass

    # CatBoost
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        globals().update({"CatBoostClassifier": CatBoostClassifier, "CatBoostRegressor": CatBoostRegressor})
        logger.info("Added CatBoost classes to globals")
    except ImportError:
        pass


def _load_onnx(content_io: BytesIO, url: str):
    try:
        import onnxruntime as ort

        model = ort.InferenceSession(content_io.getvalue())
        logger.info(f"Loaded ONNX model from {url}")
        return model
    except ImportError:
        raise HTTPException(status_code=500, detail="onnxruntime is required for ONNX models")


def _unwrap_model_dicts(model: Any) -> Any:
    """If the loaded object is a dict, extract the estimator or wrap it."""
    if not isinstance(model, dict):
        return model
    logger.info("Model loaded as dictionary, looking for model object within...")
    # common keys:
    for k in ["model", "estimator", "classifier", "regressor", "pipeline"]:
        if k in model and hasattr(model[k], "predict"):
            logger.info(f"Found model object in dictionary under key: {k}")
            return model[k]

    # pipeline stored as dict of steps
    if "steps" in model and isinstance(model["steps"], list):
        try:
            from sklearn.pipeline import Pipeline

            return Pipeline(model["steps"])
        except Exception:
            pass

    # fallback wrapper
    class DictModelWrapper:
        def __init__(self, d):
            self._d = d
            self.best_estimator_ = d.get("best_estimator_")

        def predict(self, X):
            if self.best_estimator_:
                return self.best_estimator_.predict(X)
            raise HTTPException(status_code=400, detail="Model dict does not contain a usable estimator.")

        def __getattr__(self, name):
            return getattr(self._d, name) if hasattr(self._d, name) else self._d.get(name)

    return DictModelWrapper(model)


# --- Validators ---
def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise HTTPException(status_code=400, detail=f"{name} dataset is empty")
    if df.shape[0] < 2:
        raise HTTPException(status_code=400, detail=f"{name} dataset must have at least 2 rows")
    if df.shape[1] < 1:
        raise HTTPException(status_code=400, detail=f"{name} dataset must have at least 1 column")


def validate_target_column(df: pd.DataFrame, target_column: str, df_name: str) -> None:
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in {df_name} dataset. "
            f"Available columns: {list(df.columns)}",
        )
