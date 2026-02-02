"""
Model Compatibility Helper
Utilities for handling model loading across different Python versions
"""

import pickle
import sys
import warnings
from io import BytesIO

import joblib


def safe_pickle_load(content, filename):
    """
    Safely load pickle files with compatibility across Python versions
    """
    try:
        # Method 1: Try joblib first (most common for sklearn models)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(BytesIO(content))
    except Exception as e1:
        try:
            # Method 2: Try pickle with default protocol
            return pickle.load(BytesIO(content))
        except Exception as e2:
            try:
                # Method 3: Try with older pickle protocol
                import pickle5 as pickle_alt

                return pickle_alt.load(BytesIO(content))
            except ImportError:
                try:
                    # Method 4: Force compatibility mode
                    import pickle

                    # Create a custom unpickler that can handle version differences
                    class CompatibilityUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            # Handle common sklearn module renames
                            if module == "sklearn.externals.joblib":
                                module = "joblib"
                            return super().find_class(module, name)

                    return CompatibilityUnpickler(BytesIO(content)).load()
                except Exception as e4:
                    # If all methods fail, provide detailed error information
                    error_msg = f"""
                    Model loading failed with all methods:
                    1. Joblib error: {str(e1)}
                    2. Pickle error: {str(e2)}
                    3. Compatibility error: {str(e4)}

                    This is likely due to Python version incompatibility.
                    Model file: {filename}
                    Current Python: {sys.version}

                    Suggestions:
                    - Retrain the model with current Python version
                    - Use a compatible Python environment
                    - Export model to ONNX format for better compatibility
                    """
                    raise Exception(error_msg)


def get_model_info(model):
    """Get information about the loaded model"""
    model_info = {
        "type": type(model).__name__,
        "module": type(model).__module__,
        "has_predict": hasattr(model, "predict"),
        "has_predict_proba": hasattr(model, "predict_proba"),
        "has_feature_importances": hasattr(model, "feature_importances_"),
    }

    # Try to get additional info
    try:
        if hasattr(model, "get_params"):
            model_info["params"] = str(model.get_params())
        if hasattr(model, "n_features_"):
            model_info["n_features"] = model.n_features_
    except:
        pass

    return model_info
