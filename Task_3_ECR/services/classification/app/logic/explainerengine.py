import os
from datetime import datetime

import numpy as np
import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from onnxruntime import InferenceSession
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class ExplainerEngine:
    def __init__(self, model, train_df, test_df):
        print("🔧 Initializing ExplainerEngine...")
        self.model = model
        self.train_df = train_df
        self.test_df = test_df
        self.explainer = None
        self.dashboard = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.problem_type = None
        self.host = os.getenv("HOST", "127.0.0.1")
        self.port = int(os.getenv("PORT", 8050))
        print(
            f"✅ ExplainerEngine initialized. Train shape: {train_df.shape if train_df is not None else 'None'}, Test shape: {test_df.shape if test_df is not None else 'None'}"
        )

    def drop_high_cardinality_columns(self, train_df, test_df, thresh=0.9):
        """Drop high-cardinality text columns from both train and test datasets."""
        print(f"🔍 Checking for high-cardinality columns with threshold: {thresh}")

        drop_cols = [
            c
            for c in train_df.select_dtypes(include=["object", "string"]).columns
            if train_df[c].nunique() / len(train_df) > thresh
        ]

        if drop_cols:
            print(f"🧹 Dropping high-cardinality cols: {drop_cols}")
            train_df = train_df.drop(columns=drop_cols, errors="ignore")
            test_df = test_df.drop(columns=drop_cols, errors="ignore")
            print(f"✅ Dropped columns. New train shape: {train_df.shape}, New test shape: {test_df.shape}")
        else:
            print("✅ No high-cardinality columns found")

        return train_df.copy(), test_df.copy()

    def _stratified_sample(self, X, y, n_samples, random_state=42):
        """Perform stratified sampling to maintain class distribution."""
        print(f"🎯 Starting stratified sampling: {len(X)} → {n_samples}")

        if n_samples >= len(X):
            print("⚠️ Requested sample size >= dataset size, returning original data")
            return X, y

        try:
            # Check class distribution before sampling
            original_dist = y.value_counts(normalize=True).sort_index()
            print(f"📊 Original class distribution: {original_dist.to_dict()}")

            sample_ratio = n_samples / len(X)
            print(f"📊 Sample ratio: {sample_ratio:.4f}")

            # Check if stratification is possible
            min_class_count = y.value_counts().min()
            print(f"📊 Minimum class count: {min_class_count}")

            if min_class_count < 2:
                print("⚠️ Insufficient samples in minority class for stratification")
                raise ValueError("Insufficient samples for stratification")

            X_sample, _, y_sample, _ = train_test_split(
                X, y, train_size=sample_ratio, stratify=y, random_state=random_state
            )

            # Verify class distribution after sampling
            sampled_dist = y_sample.value_counts(normalize=True).sort_index()
            print(f"📊 Sampled class distribution: {sampled_dist.to_dict()}")

            # Calculate distribution difference
            dist_diff = abs(original_dist - sampled_dist).max()
            print(f"📊 Max distribution difference: {dist_diff:.4f}")

            print(f"✅ Stratified sampling successful. Final shape: {X_sample.shape}")
            return X_sample, y_sample

        except Exception as e:
            print(f"⚠️ Stratified sampling failed: {e}")
            print("🔧 Falling back to random sampling")

            try:
                idx = np.random.choice(len(X), n_samples, replace=False)
                X_sample = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
                y_sample = y.iloc[idx] if hasattr(y, "iloc") else y[idx]

                print(f"✅ Random sampling fallback successful. Final shape: {X_sample.shape}")
                return X_sample, y_sample

            except Exception as e2:
                print(f"❌ Random sampling fallback also failed: {e2}")
                raise e2

    def _safe_sample(self, X, n_samples=10, random_state=42):
        """Safely sample from X whether it's a DataFrame or numpy array."""
        print(f"🔧 Safe sampling: {len(X)} → {n_samples}")

        try:
            if hasattr(X, "sample"):
                # X is a DataFrame
                print("📊 Sampling from DataFrame")
                sample_X = X.sample(n=min(n_samples, len(X)), random_state=random_state)
            else:
                # X is a numpy array
                print("📊 Sampling from numpy array")
                idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
                sample_X = X[idx]

            print(f"✅ Safe sampling successful. Shape: {sample_X.shape}")
            return sample_X

        except Exception as e:
            print(f"❌ Safe sampling failed: {e}")
            raise e

    def _detect_problem_type(self):
        """Detect if this is binary or multiclass classification."""
        print("🔍 Detecting problem type...")

        try:
            unique_classes = np.unique(self.y_test)
            n_classes = len(unique_classes)
            print(f"📊 Found {n_classes} unique classes: {unique_classes}")

            if n_classes == 2:
                self.problem_type = "binary"
                print(f"🎯 Detected binary classification ({n_classes} classes)")
            else:
                self.problem_type = "multiclass"
                print(f"🎯 Detected multiclass classification ({n_classes} classes)")

            return self.problem_type

        except Exception as e:
            print(f"❌ Problem type detection failed: {e}")
            raise e

    def _ensure_feature_alignment(self):
        """Ensure perfect feature alignment between model and data."""
        print("🔍 Starting feature alignment...")

        try:
            # Get model's expected features
            if hasattr(self.model, "feature_names_in_"):
                model_features = list(self.model.feature_names_in_)
                print(f"📊 Found model.feature_names_in_: {len(model_features)} features")
            elif hasattr(self.model, "feature_names_"):
                model_features = list(self.model.feature_names_)
                print(f"📊 Found model.feature_names_: {len(model_features)} features")
            else:
                # Fallback: assume current X_test columns are correct
                model_features = list(self.X_test.columns)
                print(f"📊 Using X_test columns as fallback: {len(model_features)} features")

            self.feature_names = model_features

            print(f"🔍 Model expects {len(model_features)} features: {model_features}")
            print(f"🔍 Current data has {len(self.X_test.columns)} features: {list(self.X_test.columns)}")

            # Handle missing features in both train and test
            missing_features = set(model_features) - set(self.X_test.columns)
            if missing_features:
                print(f"⚠️ Adding missing features with zeros: {missing_features}")
                for col in missing_features:
                    self.X_test[col] = 0
                    if hasattr(self, "X_train") and self.X_train is not None:
                        self.X_train[col] = 0
                print(f"✅ Added {len(missing_features)} missing features")

            # Handle extra features
            extra_features = set(self.X_test.columns) - set(model_features)
            if extra_features:
                print(f"🧹 Removing extra features: {extra_features}")
                self.X_test = self.X_test.drop(columns=extra_features)
                if hasattr(self, "X_train") and self.X_train is not None:
                    self.X_train = self.X_train.drop(columns=extra_features, errors="ignore")
                print(f"✅ Removed {len(extra_features)} extra features")

            # Ensure exact column order matches model training
            print("🔧 Reordering columns to match model expectations...")
            self.X_test = self.X_test[model_features]
            if hasattr(self, "X_train") and self.X_train is not None:
                self.X_train = self.X_train[model_features]

            # Validate data types
            print("🔧 Validating data types...")
            self._validate_data_types()

            print(f"✅ Feature alignment complete. Final shape: {self.X_test.shape}")

        except Exception as e:
            print(f"❌ Feature alignment failed: {e}")
            raise e

    def _validate_data_types(self):
        """Ensure data types are compatible with the model."""
        print("🔍 Starting data type validation...")

        try:
            # Convert object columns to appropriate numeric types where possible
            object_cols = self.X_test.select_dtypes(include=["object"]).columns
            print(f"📊 Found {len(object_cols)} object columns: {list(object_cols)}")

            for col in object_cols:
                print(f"🔧 Processing column: {col}")

                try:
                    # Try to convert to numeric
                    self.X_test[col] = pd.to_numeric(self.X_test[col], errors="coerce")
                    if hasattr(self, "X_train") and self.X_train is not None:
                        self.X_train[col] = pd.to_numeric(self.X_train[col], errors="coerce")
                    print(f"✅ Converted {col} to numeric")

                except Exception as e:
                    print(f"⚠️ Numeric conversion failed for {col}: {e}")

                    # If conversion fails, use label encoding for categorical
                    print(f"🔧 Applying label encoding to {col}")
                    le = LabelEncoder()

                    try:
                        # Fit on combined data to ensure consistent encoding
                        if hasattr(self, "X_train") and self.X_train is not None:
                            combined_values = pd.concat([self.X_train[col], self.X_test[col]]).astype(str)
                            le.fit(combined_values)
                            self.X_train[col] = le.transform(self.X_train[col].astype(str))
                            self.X_test[col] = le.transform(self.X_test[col].astype(str))
                        else:
                            self.X_test[col] = le.fit_transform(self.X_test[col].astype(str))
                        print(f"✅ Label encoded {col}")

                    except Exception as e2:
                        print(f"❌ Label encoding failed for {col}: {e2}")
                        # Last resort: fill with zeros
                        self.X_test[col] = 0
                        if hasattr(self, "X_train") and self.X_train is not None:
                            self.X_train[col] = 0
                        print(f"⚠️ Filled {col} with zeros as fallback")

            # Ensure all columns are numeric
            print("🔧 Ensuring all columns are numeric...")
            numeric_cols_before = len(self.X_test.columns)
            self.X_test = self.X_test.select_dtypes(include=[np.number])
            numeric_cols_after = len(self.X_test.columns)

            if hasattr(self, "X_train") and self.X_train is not None:
                self.X_train = self.X_train.select_dtypes(include=[np.number])

            print(f"📊 Numeric columns: {numeric_cols_before} → {numeric_cols_after}")

            # Fill any remaining NaN values
            nan_count_before = self.X_test.isna().sum().sum()
            self.X_test = self.X_test.fillna(0)
            if hasattr(self, "X_train") and self.X_train is not None:
                self.X_train = self.X_train.fillna(0)

            nan_count_after = self.X_test.isna().sum().sum()
            print(f"📊 NaN values filled: {nan_count_before} → {nan_count_after}")

            print("✅ Data type validation complete")

        except Exception as e:
            print(f"❌ Data type validation failed: {e}")
            raise e

    def _validate_model_compatibility(self):
        """Validate that the current dataset is compatible with the model."""
        print("🔍 Starting model compatibility test...")

        try:
            # Test prediction on a small sample
            sample = self.X_test.iloc[:1]
            print(f"📊 Testing with sample shape: {sample.shape}")

            # Handle ONNX models differently
            if isinstance(self.model, InferenceSession):
                print("🔍 Testing ONNX model compatibility...")

                # Test ONNX model with proper input format
                input_names = [i.name for i in self.model.get_inputs()]
                input_name = input_names[0]

                # Convert to numpy array with proper dtype
                sample_array = sample[self.feature_names].values.astype(np.float32)
                inputs = {input_name: sample_array}

                # Test ONNX run
                outputs = self.model.run(None, inputs)
                test_proba = outputs[0]

                # Handle different output formats
                if test_proba.ndim == 1:
                    test_proba = np.vstack([1 - test_proba, test_proba]).T

                test_pred = np.argmax(test_proba, axis=1)

                print(f"✅ ONNX model compatibility test passed")
                print(f"   Sample prediction: {test_pred[0]}")
                print(f"   Sample probabilities: {test_proba[0]}")

            else:
                print("🔍 Testing scikit-learn model compatibility...")

                # Test scikit-learn model
                test_pred = self.model.predict(sample)
                test_proba = self.model.predict_proba(sample)

                print(f"✅ Scikit-learn model compatibility test passed")
                print(f"   Sample prediction: {test_pred[0]}")
                print(f"   Sample probabilities: {test_proba[0]}")

            return True

        except Exception as e:
            print(f"❌ Model compatibility test failed: {e}")
            print(f"❌ Error type: {type(e).__name__}")

            # Additional debugging for ONNX models
            if isinstance(self.model, InferenceSession):
                print("🔍 ONNX model debugging info:")
                print(f"   Input names: {[i.name for i in self.model.get_inputs()]}")
                print(f"   Input shapes: {[i.shape for i in self.model.get_inputs()]}")
                print(f"   Output names: {[o.name for o in self.model.get_outputs()]}")
                print(f"   Output shapes: {[o.shape for o in self.model.get_outputs()]}")
                print(f"   Feature names: {self.feature_names}")
                print(f"   Sample shape: {sample.shape}")

            return False

    def setup_explainer(self, target_column, test_size=0.2, random_state=42, max_explainer_rows=None):
        """Prepare explainer dashboard with preprocessing, SHAP patching and feature alignment."""
        print(f"🚀 Starting explainer setup for target: {target_column}")
        print(
            f"📊 Parameters: test_size={test_size}, random_state={random_state}, max_explainer_rows={max_explainer_rows}"
        )

        try:
            # 1) Split or use provided datasets
            print("🔧 Step 1: Dataset preparation...")
            if self.train_df is not None and self.test_df is not None:
                print("📊 Using provided train/test datasets")
                train_df, test_df = self.train_df.copy(), self.test_df.copy()
            else:
                print("📊 Splitting single dataset")
                full_df = self.train_df.copy()
                if target_column not in full_df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset.")
                X = full_df.drop(columns=[target_column])
                y = full_df[target_column]
                X = X.fillna(X.median(numeric_only=True))
                y = y.fillna(y.mode()[0])
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                train_df = pd.concat([X_train, y_train], axis=1)
                test_df = pd.concat([X_test, y_test], axis=1)

            print(f"✅ Dataset preparation complete. Train: {train_df.shape}, Test: {test_df.shape}")

            # 2) Drop high-cardinality features
            print("🔧 Step 2: Dropping high-cardinality features...")
            train_df, test_df = self.drop_high_cardinality_columns(train_df, test_df)

            # 3) Split features and targets
            print("🔧 Step 3: Splitting features and targets...")
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            print(f"📊 Features: {X_train.shape}, {X_test.shape}")
            print(f"📊 Targets: {y_train.shape}, {y_test.shape}")

            # 4) Impute missing
            print("🔧 Step 4: Imputing missing values...")
            missing_before_train = X_train.isna().sum().sum()
            missing_before_test = X_test.isna().sum().sum()

            X_train = X_train.fillna(X_train.median(numeric_only=True))
            X_test = X_test.fillna(X_test.median(numeric_only=True))
            y_train = y_train.fillna(y_train.mode()[0])
            y_test = y_test.fillna(y_test.mode()[0])

            missing_after_train = X_train.isna().sum().sum()
            missing_after_test = X_test.isna().sum().sum()

            print(f"📊 Missing values - Train: {missing_before_train} → {missing_after_train}")
            print(f"📊 Missing values - Test: {missing_before_test} → {missing_after_test}")

            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test

            # 5) Wrap ONNX or handle regular model
            print("🔧 Step 5: Model preparation...")
            if isinstance(self.model, InferenceSession):
                print("🧠 ONNX model detected")
                self.feature_names = [i.name for i in self.model.get_inputs()]
                print(f"📊 ONNX input features: {self.feature_names}")

                # Apply feature alignment before wrapping
                self._ensure_feature_alignment()

                # Wrap ONNX model BEFORE compatibility check
                model_for_explainer = self._wrap_onnx_model()

                # Update self.model to use the wrapper for compatibility testing
                self.model = model_for_explainer

            else:
                print("🧠 Scikit-learn model detected")
                if not hasattr(self.model, "estimators_") or len(self.model.estimators_) == 0:
                    print("⚙️ Model not fitted, fitting now...")
                    self.model.fit(X_train, y_train)
                    print("✅ Model fitted")

                # Apply robust feature alignment
                self._ensure_feature_alignment()
                model_for_explainer = self.model

            # 5.5) Validate model compatibility (now works for both ONNX and sklearn)
            print("🔧 Step 5.5: Model compatibility validation...")
            if not self._validate_model_compatibility():
                raise Exception("Model is not compatible with current dataset after feature alignment")

            # 6) Detect problem type
            print("🔧 Step 6: Problem type detection...")
            self._detect_problem_type()

            # 6.5) Stratified subsample test set for speed
            print("🔧 Step 6.5: Stratified sampling...")
            if max_explainer_rows and len(self.X_test) > max_explainer_rows:
                print(f"🎯 Applying stratified sampling: {len(self.X_test)} → {max_explainer_rows}")
                self.X_test, self.y_test = self._stratified_sample(
                    self.X_test, self.y_test, max_explainer_rows, random_state
                )
                print(f"✅ Stratified sampling complete. New shape: {self.X_test.shape}")
            else:
                print("⚠️ No sampling needed - dataset size acceptable")

            # 7) Create explainer with robust configuration
            print("🔧 Step 7: Creating explainer...")
            self._create_robust_explainer(model_for_explainer)

            # 8) Build dashboard with appropriate configuration
            print("🔧 Step 8: Building dashboard...")
            if self.problem_type == "multiclass":
                print("🎯 Creating multiclass dashboard...")
                self.dashboard = ExplainerDashboard(
                    self.explainer,
                    title="Multiclass Classification Explainer",
                    whatif=True,
                    shap_interaction=False,
                    decision_trees=False,  # Often problematic for multiclass
                )
            else:
                print("🎯 Creating binary classification dashboard...")
                self.dashboard = ExplainerDashboard(
                    self.explainer, title="Binary Classification Explainer", whatif=True, shap_interaction=False
                )

            print("✅ Classification explainer ready.")
            return True

        except Exception as e:
            print(f"❌ Setup explainer failed: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback

            print(f"❌ Traceback: {traceback.format_exc()}")
            raise e

    def _create_robust_explainer(self, model_for_explainer):
        """Create explainer with robust error handling and SHAP patching."""
        print("🔧 Starting robust explainer creation...")

        try:
            # Create stratified background sample
            background_size = min(50, len(self.X_test))
            print(f"📊 Creating background sample: {background_size} samples")

            if background_size < len(self.X_test):
                X_background, _ = self._stratified_sample(self.X_test, self.y_test, background_size, random_state=42)
                print(f"✅ Stratified background sample created: {X_background.shape}")
            else:
                X_background = self.X_test
                print(f"✅ Using full dataset as background: {X_background.shape}")

            # Ensure background has same features and types
            X_background = X_background[self.feature_names]
            print(f"📊 Background sample aligned to features: {X_background.shape}")

            print("🔧 Attempting to create ClassifierExplainer...")
            try:
                self.explainer = ClassifierExplainer(
                    model_for_explainer,
                    self.X_test[self.feature_names],
                    self.y_test,
                    model_output="raw",
                    shap_kwargs={"check_additivity": False},
                    X_background=X_background,
                )
                print("✅ Explainer created successfully with check_additivity=False")

            except Exception as e:
                print(f"❌ Primary explainer creation failed: {e}")
                print("🔧 Trying with minimal SHAP configuration...")

                # Fallback with minimal configuration
                self.explainer = ClassifierExplainer(
                    model_for_explainer,
                    self.X_test[self.feature_names],
                    self.y_test,
                    model_output="raw",
                    shap_kwargs={"check_additivity": False},
                )
                print("✅ Fallback explainer created successfully")

            # Apply SHAP patching based on problem type
            print("🔧 Applying SHAP patching...")
            self._patch_shap_based_on_type()

        except Exception as e:
            print(f"❌ Robust explainer creation failed: {e}")
            raise e

    def _patch_shap_based_on_type(self):
        """Apply appropriate SHAP patching based on problem type."""
        print(f"🔧 Applying SHAP patching for {self.problem_type} classification...")

        try:
            if self.problem_type == "binary":
                self._patch_shap_for_binary()
            else:
                self._patch_shap_for_multiclass()
            print("✅ SHAP patching completed successfully")

        except Exception as e:
            print(f"❌ SHAP patching failed: {e}")
            raise e

    def _patch_shap_for_binary(self):
        """Patch SHAP explainer to return only positive-class contributions with robust error handling."""
        print("🔧 Applying binary SHAP monkey patch…")

        try:
            orig_shap = self.explainer.shap_explainer.shap_values
            print("📊 Original SHAP function captured")

            def shap_for_pos(X, **kwargs):
                print(f"🔧 SHAP called with X shape: {X.shape if hasattr(X, 'shape') else len(X)}")
                print(f"🔧 SHAP kwargs: {kwargs}")

                try:
                    # Temporarily disable additivity check if possible
                    if hasattr(self.explainer.shap_explainer, "check_additivity"):
                        orig_check = self.explainer.shap_explainer.check_additivity
                        self.explainer.shap_explainer.check_additivity = False
                        print("🔧 Additivity check disabled")
                    else:
                        orig_check = None
                        print("🔧 No additivity check found")

                    # Pass through all kwargs to original function
                    print("🔧 Calling original SHAP function...")
                    raw = orig_shap(X, **kwargs)
                    print("✅ Original SHAP function completed")

                    # Restore original check setting
                    if orig_check is not None:
                        self.explainer.shap_explainer.check_additivity = orig_check
                        print("🔧 Additivity check restored")

                except Exception as e:
                    print(f"⚠️ SHAP calculation failed: {e}")
                    print("🔧 Retrying with smaller sample size...")

                    # Try with smaller sample - handle both DataFrame and numpy array
                    if len(X) > 10:
                        try:
                            sample_X = self._safe_sample(X, n_samples=10, random_state=42)
                            print("🔧 Calling original SHAP function with smaller sample...")
                            raw = orig_shap(sample_X, **kwargs)
                            print("✅ SHAP with smaller sample completed")
                        except Exception as e2:
                            print(f"❌ SHAP with smaller sample also failed: {e2}")
                            raise e2
                    else:
                        print("❌ Sample already too small, cannot reduce further")
                        raise e

                arr = np.array(raw)
                print(f"📊 Raw SHAP output shape: {arr.shape}")
                print(f"📊 Raw SHAP output type: {type(raw)}")

                if arr.ndim == 3 and arr.shape[2] == 2:
                    out = arr[:, :, 1]
                    print("🔧 Extracted class 1 from 3D array")
                elif isinstance(raw, list) and len(raw) == 2 and isinstance(raw[1], np.ndarray):
                    out = raw[1]
                    print("🔧 Extracted class 1 from list")
                elif arr.ndim == 2:
                    out = arr
                    print("🔧 Using 2D array as-is")
                else:
                    print(f"❌ Unexpected SHAP output shape: {arr.shape}")
                    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")

                print(f"📊 Processed SHAP shape: {out.shape}")
                return out

            self.explainer.shap_explainer.shap_values = shap_for_pos
            print("✅ SHAP function patched")

            ev = self.explainer.shap_explainer.expected_value
            print(f"📊 Expected value: {ev}")

            if isinstance(ev, (list, tuple)) and len(ev) >= 2:
                self.explainer.expected_value = ev[1]
                print(f"🔧 Patched expected_value to class-1: {ev[1]}")
            else:
                print(f"🔧 Using expected_value as-is: {ev}")

            # Use try-except for the final SHAP calculation
            print("🔧 Computing initial SHAP values...")
            try:
                self.explainer.shap_values = shap_for_pos(self.X_test[self.feature_names])
                print("✅ SHAP values computed successfully")
            except Exception as e:
                print(f"⚠️ Final SHAP calculation failed: {e}")
                print("🔧 Attempting with reduced sample size...")
                # Try with a smaller sample if full dataset fails
                sample_size = min(20, len(self.X_test))
                sample_data = self.X_test[self.feature_names].sample(n=sample_size, random_state=42)
                self.explainer.shap_values = shap_for_pos(sample_data)
                print(f"✅ SHAP values computed with reduced sample size: {sample_size}")

        except Exception as e:
            print(f"❌ Binary SHAP patching failed: {e}")
            raise e

    def _patch_shap_for_multiclass(self):
        """Enhanced SHAP patching for multiclass classification."""
        print("🔧 Applying multiclass SHAP patching…")

        try:
            orig_shap = self.explainer.shap_explainer.shap_values
            print("📊 Original SHAP function captured")

            def shap_for_multiclass(X, **kwargs):
                print(f"🔧 Multiclass SHAP called with X shape: {X.shape if hasattr(X, 'shape') else len(X)}")
                print(f"🔧 Multiclass SHAP kwargs: {kwargs}")

                try:
                    print("🔧 Calling original SHAP function...")
                    raw = orig_shap(X, **kwargs)
                    print("✅ Original SHAP function completed")

                    arr = np.array(raw)
                    print(f"📊 Raw SHAP output shape: {arr.shape}")
                    print(f"📊 Raw SHAP output type: {type(raw)}")

                    # Handle different SHAP output formats
                    if isinstance(raw, list):
                        # TreeExplainer returns list of arrays, one per class
                        # Shape: [n_classes] of (n_samples, n_features)
                        out = raw
                        print(f"🔧 Using list format: {len(out)} classes")
                    elif arr.ndim == 3:
                        # Some explainers return 3D array: (n_samples, n_features, n_classes)
                        # Convert to list format expected by dashboard
                        out = [arr[:, :, i] for i in range(arr.shape[2])]
                        print(f"🔧 Converted 3D array to list format: {len(out)} classes")
                    else:
                        print(f"❌ Unexpected SHAP output shape: {arr.shape}")
                        raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")

                    print(f"📊 Processed SHAP format: {len(out)} classes")
                    return out

                except Exception as e:
                    print(f"⚠️ SHAP calculation failed: {e}")
                    # Fallback with smaller sample
                    if len(X) > 10:
                        try:
                            sample_X = self._safe_sample(X, n_samples=10, random_state=42)
                            print("🔧 Calling original SHAP function with smaller sample...")
                            return shap_for_multiclass(sample_X, **kwargs)
                        except Exception as e2:
                            print(f"❌ SHAP with smaller sample also failed: {e2}")
                            raise e2
                    else:
                        print("❌ Sample already too small, cannot reduce further")
                        raise e

            self.explainer.shap_explainer.shap_values = shap_for_multiclass
            print("✅ Multiclass SHAP function patched")

            # Handle expected values for all classes
            ev = self.explainer.shap_explainer.expected_value
            print(f"📊 Expected values: {ev}")

            if isinstance(ev, (list, tuple)):
                self.explainer.expected_value = ev
                print(f"🔧 Using expected_values for all classes: {ev}")
            else:
                print(f"🔧 Using single expected_value: {ev}")

            # Compute SHAP values for all classes
            print("🔧 Computing initial multiclass SHAP values...")
            try:
                self.explainer.shap_values = shap_for_multiclass(self.X_test[self.feature_names])
                print("✅ Multiclass SHAP values computed successfully")
            except Exception as e:
                print(f"⚠️ Multiclass SHAP calculation failed: {e}")
                print("🔧 Attempting with reduced sample size...")
                sample_size = min(20, len(self.X_test))
                sample_data = self.X_test[self.feature_names].sample(n=sample_size, random_state=42)
                self.explainer.shap_values = shap_for_multiclass(sample_data)
                print(f"✅ Multiclass SHAP values computed with reduced sample size: {sample_size}")

        except Exception as e:
            print(f"❌ Multiclass SHAP patching failed: {e}")
            raise e

    def _wrap_onnx_model(self):
        """Wrap ONNX InferenceSession into explainer-compatible class."""
        print("🔧 Wrapping ONNX model...")

        try:
            input_names = [i.name for i in self.model.get_inputs()]
            output_names = [o.name for o in self.model.get_outputs()]
            print(f"📊 ONNX input names: {input_names}")
            print(f"📊 ONNX output names: {output_names}")

            class ONNXWrapper:
                """
                A simple wrapper for ONNX models that provides `predict_proba` and `predict` methods.
                Accepts input features and returns class probabilities or labels.
                Logs input shape on the first prediction for debugging.
                """

                def __init__(self, session, input_names, feature_names):
                    self.session = session
                    self.input_names = input_names
                    self.feature_names = feature_names
                    self.logged = False  # Track if we've logged once
                    print(f"🔧 ONNXWrapper initialized with {len(feature_names)} features")

                def predict_proba(self, X):
                    # Log only the first call
                    if not self.logged:
                        print(f"🔧 ONNXWrapper.predict_proba - Processing data with shape: {X.shape}")
                        print(f"📊 Note: Further calls will be silent, progress shown in progress bar")
                        self.logged = True

                    # Ensure X is a DataFrame and has the right columns
                    if hasattr(X, "columns"):
                        X_array = X[self.feature_names].values.astype(np.float32)
                    else:
                        X_array = X.astype(np.float32)

                    input_name = self.input_names[0]
                    inputs = {input_name: X_array}

                    outputs = self.session.run(None, inputs)
                    probs = outputs[0]

                    # Handle different output formats
                    if probs.ndim == 1:
                        probs = np.vstack([1 - probs, probs]).T
                    elif probs.shape[1] == 1:
                        probs = np.column_stack([1 - probs.flatten(), probs.flatten()])

                    return probs

                def predict(self, X):
                    # No logging here - let predict_proba handle it
                    probs = self.predict_proba(X)
                    preds = np.argmax(probs, axis=1)
                    return preds

            wrapper = ONNXWrapper(self.model, input_names, self.feature_names)
            print("✅ ONNX model wrapped successfully")

            # Test the wrapper (this will trigger the one-time log)
            test_sample = self.X_test.iloc[:1]
            test_pred = wrapper.predict(test_sample)
            test_proba = wrapper.predict_proba(test_sample)

            print(f"✅ ONNX wrapper test successful: pred={test_pred[0]}, proba={test_proba[0]}")

            return wrapper

        except Exception as e:
            print(f"❌ ONNX model wrapping failed: {e}")
            import traceback

            print(f"❌ Traceback: {traceback.format_exc()}")
            raise e

    def run_dashboard(self, host=None, port=None):
        """Run dashboard directly (can be wrapped in thread externally)."""
        print(f"🚀 Starting dashboard server...")
        host = host or self.host
        port = port or self.port
        try:
            if not self.dashboard:
                raise Exception("Dashboard not created.")
            print(f"🚀 Running dashboard at http://{host}:{port}")
            self.dashboard.run(host=host, port=port, debug=False)

        except Exception as e:
            print(f"❌ Dashboard run failed: {e}")
            raise e

    def save_dashboard_html(self, filename=None):
        """Export dashboard as static HTML."""
        print("🔧 Saving dashboard as HTML...")

        try:
            if not self.dashboard:
                raise Exception("Dashboard not created.")
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Classification_dashboard_{timestamp}.html"
            os.makedirs("dashboard_exports", exist_ok=True)
            filepath = os.path.join("dashboard_exports", filename)

            self.dashboard.to_html(filepath)
            print(f"✅ Dashboard saved: {filepath}")
            return filepath

        except Exception as e:
            print(f"❌ Save failed: {e}")
            return None

    def predict_single(self, input_dict):
        """Enhanced single prediction for both binary and multiclass."""
        print(f"🔧 Single prediction for input: {input_dict}")

        try:
            if not self.model or not self.explainer:
                raise Exception("Setup explainer first.")

            input_df = pd.DataFrame([input_dict])
            print(f"📊 Input DataFrame shape: {input_df.shape}")

            # Apply same preprocessing as during setup
            for col in self.feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Ensure column order and types match
            input_df = input_df[self.feature_names]
            print(f"📊 Aligned DataFrame shape: {input_df.shape}")

            # Apply same data type conversions
            for col in input_df.columns:
                if input_df[col].dtype == "object":
                    try:
                        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
                    except:
                        # Simple fallback encoding
                        input_df[col] = 0

            input_df = input_df.fillna(0)
            print(f"📊 Preprocessed DataFrame shape: {input_df.shape}")

            prediction = self.model.predict(input_df)[0]
            probabilities = self.model.predict_proba(input_df)[0]
            print(f"📊 Prediction: {prediction}, Probabilities: {probabilities}")

            try:
                print("🔧 Computing SHAP values for single prediction...")
                shap_raw = self.explainer.shap_explainer.shap_values(input_df)

                if self.problem_type == "multiclass":
                    # Handle multiclass SHAP values
                    print("🔧 Processing multiclass SHAP values...")
                    shap_dict = {}
                    for class_idx, class_shap in enumerate(shap_raw):
                        shap_dict[f"class_{class_idx}"] = dict(zip(self.feature_names, class_shap[0]))
                    print(f"📊 Multiclass SHAP computed for {len(shap_dict)} classes")
                else:
                    # Handle binary as before
                    print("🔧 Processing binary SHAP values...")
                    if isinstance(shap_raw, list) and len(shap_raw) > 1:
                        shap_values = shap_raw[1][0]
                    elif isinstance(shap_raw, list):
                        shap_values = shap_raw[0][0]
                    else:
                        shap_values = shap_raw[0]
                    shap_dict = dict(zip(self.feature_names, shap_values))
                    print(f"📊 Binary SHAP computed for {len(shap_dict)} features")

            except Exception as e:
                print(f"⚠️ SHAP computation failed: {e}")
                shap_dict = f"SHAP error: {e}"

            result = {
                "prediction": prediction,
                "probabilities": probabilities.tolist(),
                "shap_contributions": shap_dict,
            }
            print(f"✅ Single prediction completed")
            return result

        except Exception as e:
            print(f"❌ Single prediction failed: {e}")
            raise e

    def batch_predict(self, input_list):
        """Perform batch predictions and collect SHAP explanations for each input."""
        print(f"🔧 Batch prediction for {len(input_list)} inputs...")

        try:
            results = []
            for i, input_dict in enumerate(input_list):
                print(f"🔧 Processing input {i+1}/{len(input_list)}")
                try:
                    result = self.predict_single(input_dict)
                    result["instance_id"] = i
                    results.append(result)
                    print(f"✅ Input {i+1} processed successfully")
                except Exception as e:
                    print(f"❌ Input {i+1} failed: {e}")
                    results.append({"instance_id": i, "error": str(e)})

            print(f"✅ Batch prediction completed: {len(results)} results")
            return results

        except Exception as e:
            print(f"❌ Batch prediction failed: {e}")
            raise e
