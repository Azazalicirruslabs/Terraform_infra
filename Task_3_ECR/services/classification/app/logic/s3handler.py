import os
import pickle
import sys
import tempfile
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import requests
from onnxruntime import InferenceSession
from packaging import version
from sklearn.ensemble import RandomForestClassifier


class S3Handler:
    """
    Handles S3 operations for model and data loading.

    Args:
        api_url: Static API URL. Access token will be passed per request.
    """

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.model = None
        self.train_df = None
        self.test_df = None

    def load_model(self, file_content: bytes, file_name: str):
        """
        Deserialize model from bytes. Supports .onnx, .joblib, .pkl formats.
        Returns: model/session, feature list, format tag, error message (if any)
        """
        print("🔧 Starting model loading...")
        print(f"📄 Received: {file_name} | Size: {len(file_content)} bytes")

        # Check if NumPy patching is still necessary for legacy ONNX models.
        if version.parse(np.__version__) < version.parse("2.0.0"):
            try:
                import numpy.core

                sys.modules["numpy._core"] = numpy.core
                sys.modules["numpy._core._multiarray_umath"] = numpy.core._multiarray_umath
                print("🔧 NumPy patch applied.")
            except Exception as e:
                print(f"⚠️ NumPy patching failed: {e}")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
                print(f"📁 Temp file created: {tmp_path}")

            if file_name.endswith(".onnx"):
                try:
                    print("📦 Loading ONNX model...")
                    session = InferenceSession(tmp_path)
                    features = [i.name for i in session.get_inputs()]
                    print(f"✅ ONNX loaded. Features: {features}")
                    return session, features, "onnx", None
                except Exception as e:
                    print(f"❌ ONNX loading failed: {e}")
                    return None, None, "onnx", str(e)

            last_error = None

            def pickle_loader():
                with open(tmp_path, "rb") as f:
                    return pickle.load(f)

            for name, loader in [
                ("joblib", lambda: joblib.load(tmp_path)),
                ("pickle", pickle_loader),
            ]:
                try:
                    print(f"📦 Trying {name} loader...")
                    model = loader()
                    features = getattr(model, "feature_names_in_", None)
                    features = features.tolist() if features is not None else None
                    print(f"✅ Model loaded using {name}")
                    return model, features, name, None
                except Exception as e:
                    print(f"❌ {name} failed: {e}")
                    last_error = e

            raise RuntimeError(f"🚨 All loaders failed: {last_error}")

        except Exception as outer_e:
            print(f"💥 Unexpected error: {outer_e}")
            return None, None, "unknown", str(outer_e)

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                    print(f"🧹 Temp file cleaned: {tmp_path}")
                except Exception as cleanup_error:
                    print(f"⚠️ Temp cleanup error: {cleanup_error}")

    # def load_from_s3(self, access_token: str):
    #     """
    #     Fetch files from external S3 API using user-provided access token.
    #     Populates self.train_df, self.test_df, self.model.
    #     """
    #     if not access_token:
    #         raise ValueError("Access token is required.")

    #     headers = {"Authorization": f"Bearer {access_token}"}
    #     response = requests.get(self.api_url, headers=headers)

    #     files = response.json().get("files", [])
    #     if not files:
    #         return None, None, None
    #     model = None

    #     for file_info in files:
    #         file_response = requests.get(file_info["url"])
    #         if file_response.status_code != 200:
    #             continue

    #         fname = file_info["file_name"].lower()
    #         if fname.endswith(".csv"):
    #             df = pd.read_csv(BytesIO(file_response.content))
    #             if "ref" in fname:
    #                 self.train_df = df
    #             elif "cur" in fname:
    #                 self.test_df = df
    #             else:
    #                 self.train_df = df
    #             print(f"✅ Loaded dataset: {fname} | Shape: {df.shape}")

    #         elif fname.endswith((".pkl", ".joblib", ".onnx")):
    #             print(f"🔍 Loading model file: {fname}")
    #             model, features, format_tag, error = self.load_model(file_response.content, fname)

    #     if model is None:
    #         print("⚠️ No model found. Using RandomForest fallback.")
    #         model = RandomForestClassifier(n_estimators=50, random_state=42)

    #     if self.train_df is None and self.test_df is None:
    #         raise Exception("❌ No dataset found in S3 response.")

    #     self.model = model
    #     return self.model, self.train_df, self.test_df
    def load_from_s3(self, access_token: str):
        """
        Fetch files from external S3 API using user-provided access token.
        Populates self.train_df, self.test_df, and self.model.
        Supports both .csv and .parquet datasets.
        """
        if not access_token:
            raise ValueError("Access token is required.")

        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(self.api_url, headers=headers)
        response.raise_for_status()

        files = response.json().get("files", [])
        if not files:
            print("⚠️ No files found in S3 API response.")
            return None, None, None
        model = None

        for file_info in files:
            file_url = file_info.get("url")
            fname = file_info.get("file_name", "").lower()
            if not file_url or not fname:
                continue

            print(f"⬇️ Downloading: {fname}")
            file_response = requests.get(file_url)
            if file_response.status_code != 200:
                print(f"⚠️ Skipping {fname} — unable to download ({file_response.status_code}).")
                continue

            # --- Handle datasets (.csv or .parquet) ---
            if fname.endswith((".csv", ".parquet")):
                try:
                    if fname.endswith(".csv"):
                        df = pd.read_csv(BytesIO(file_response.content))
                    else:
                        df = pd.read_parquet(BytesIO(file_response.content), engine="pyarrow")

                    # Determine dataset type
                    if "ref" in fname or "train" in fname:
                        self.train_df = df
                        print(f"✅ Loaded training dataset: {fname} | Shape: {df.shape}")
                    elif "cur" in fname or "test" in fname:
                        self.test_df = df
                        print(f"✅ Loaded test dataset: {fname} | Shape: {df.shape}")
                    else:
                        # Default to training dataset
                        self.train_df = df
                        print(f"ℹ️ Loaded dataset (defaulted to train): {fname} | Shape: {df.shape}")
                except Exception as e:
                    print(f"❌ Failed to load dataset {fname}: {e}")

            # --- Handle model files (.pkl, .joblib, .onnx) ---
            elif fname.endswith((".pkl", ".joblib", ".onnx")):
                print(f"🔍 Loading model file: {fname}")
                model, features, format_tag, error = self.load_model(file_response.content, fname)
                if error:
                    print(f"⚠️ Model loading error for {fname}: {error}")
                else:
                    print(f"✅ Model loaded successfully: {fname} ({format_tag})")

        # --- Fallback handling ---
        if model is None:
            print("⚠️ No model found. Using RandomForest fallback.")
            model = RandomForestClassifier(n_estimators=50, random_state=42)

        if self.train_df is None and self.test_df is None:
            raise Exception("❌ No dataset found in S3 response.")

        self.model = model
        return self.model, self.train_df, self.test_df
