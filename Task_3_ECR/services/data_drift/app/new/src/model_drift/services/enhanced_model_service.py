"""
Enhanced Model Service with Dynamic Configuration Support
"""

import io
import pickle
import warnings
from datetime import datetime
from typing import Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import UploadFile
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, RobustScaler

from ..models.analysis_config import AnalysisConfiguration, ModelType
from ..models.response_models import (
    AnalysisResults,
    DriftMetrics,
    ModelDriftResponse,
    ModelInfo,
    PerformanceMetrics,
    StatisticalTestResult,
)
from ..services.analysis.degradation_metrics_service import degradation_metrics_service
from ..services.analysis.performance_comparison_service import performance_comparison_service
from ..services.analysis.statistical_significance_service import statistical_significance_service
from ..services.core.statistical_tests_service import statistical_tests_service

warnings.filterwarnings("ignore")


class EnhancedModelService:
    """Enhanced model service with dynamic configuration support"""

    def __init__(self):
        # Classification metrics mapping
        self.classification_metrics = {
            "accuracy": self._calculate_accuracy,
            "precision": self._calculate_precision,
            "recall": self._calculate_recall,
            "f1_score": self._calculate_f1_score,
            "specificity": self._calculate_specificity,
            "roc_auc": self._calculate_roc_auc,
            "pr_auc": self._calculate_pr_auc,
            "cohen_kappa": self._calculate_cohen_kappa,
            "mcc": self._calculate_mcc,
        }

        # Regression metrics mapping
        self.regression_metrics = {
            "mse": self._calculate_mse,
            "rmse": self._calculate_rmse,
            "mae": self._calculate_mae,
            "r2": self._calculate_r2,
            "adjusted_r2": self._calculate_adjusted_r2,
            "mape": self._calculate_mape,
            "explained_variance": self._calculate_explained_variance,
            "max_error": self._calculate_max_error,
        }

    def _detect_model_type(self, model) -> str:
        """
        Automatically detect if model is regression or classification

        Args:
            model: The model object to inspect

        Returns:
            "regression" or "classification"
        """
        # Primary check: does the model have predict_proba method?
        if hasattr(model, "predict_proba"):
            # Classification models have predict_proba
            return "classification"

        # Secondary check: if it's a pipeline, check the final estimator
        if hasattr(model, "named_steps"):
            # Get the last step (final estimator)
            final_estimator = list(model.named_steps.values())[-1]
            if hasattr(final_estimator, "predict_proba"):
                return "classification"
            else:
                return "regression"
        elif hasattr(model, "steps"):
            # Pipeline with steps attribute
            final_estimator = model.steps[-1][1]
            if hasattr(final_estimator, "predict_proba"):
                return "classification"
            else:
                return "regression"

        # Default: if no predict_proba, assume regression
        return "regression"

    async def run_configured_analysis(
        self,
        reference_data: UploadFile,
        current_data: UploadFile,
        model_file: UploadFile,
        config: AnalysisConfiguration,
    ) -> ModelDriftResponse:
        """
        Run model drift analysis with dynamic configuration

        Args:
            reference_data: Reference dataset file
            current_data: Current dataset file
            model_file: Model file
            config: Analysis configuration from frontend

        Returns:
            Structured model drift response
        """
        try:
            analysis_id = f"model_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Load and validate data
            ref_df, curr_df, model = await self._load_and_validate_inputs(reference_data, current_data, model_file)

            # Prepare data based on model type
            X_ref, X_curr, y_ref, y_curr = await self._prepare_data(ref_df, curr_df, model, config.model_type)

            # Get model information
            model_info = self._extract_model_info(model, ref_df, curr_df)

            # Calculate selected performance metrics
            performance_metrics = self._calculate_selected_metrics(y_ref, y_curr, X_ref, X_curr, model, config)

            # Calculate drift metrics
            drift_metrics = self._calculate_drift_metrics(performance_metrics, config.drift_thresholds)

            # Run selected statistical test
            statistical_result = None
            if config.statistical_test:
                statistical_result = await self._run_selected_statistical_test(
                    y_ref, y_curr, X_ref, X_curr, model, config.statistical_test
                )

            # Generate additional outputs based on model type
            confusion_matrices = None
            classification_reports = None

            if config.model_type == ModelType.CLASSIFICATION:
                ref_pred = model.predict(X_ref)
                curr_pred = model.predict(X_curr)

                confusion_matrices = {
                    "reference": confusion_matrix(y_ref, ref_pred).tolist(),
                    "current": confusion_matrix(y_curr, curr_pred).tolist(),
                }

                classification_reports = {
                    "reference": classification_report(y_ref, ref_pred, output_dict=True, zero_division=0),
                    "current": classification_report(y_curr, curr_pred, output_dict=True, zero_division=0),
                }

            # Generate recommendations
            recommendations = self._generate_recommendations(drift_metrics, statistical_result)

            # Create analysis results
            analysis_results = AnalysisResults(
                analysis_id=analysis_id,
                analysis_name=config.analysis_name,
                timestamp=datetime.now(),
                model_info=model_info,
                configuration=config.dict(),
                performance_metrics=performance_metrics,
                drift_metrics=drift_metrics,
                statistical_test_result=statistical_result,
                confusion_matrices=confusion_matrices,
                classification_reports=classification_reports,
                recommendations=recommendations,
            )

            return ModelDriftResponse(success=True, analysis_results=analysis_results)

        except Exception as e:
            return ModelDriftResponse(success=False, error_message=str(e), error_code="ANALYSIS_FAILED")

    async def run_performance_comparison_analysis(
        self,
        reference_data: UploadFile,
        current_data: UploadFile,
        model_file: UploadFile,
        config: AnalysisConfiguration,
    ) -> dict:
        """
        Run Tab 1: Performance Comparison analysis

        Args:
            reference_data: Reference dataset file
            current_data: Current dataset file
            model_file: Model file
            config: Analysis configuration from frontend

        Returns:
            Performance comparison results
        """
        try:
            analysis_id = f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Load and validate data
            ref_df, curr_df, model = await self._load_and_validate_inputs(reference_data, current_data, model_file)

            # Prepare data based on model type
            X_ref, X_curr, y_ref, y_curr = await self._prepare_data(ref_df, curr_df, model, config.model_type)

            # Get model information
            model_info = self._extract_model_info(model, ref_df, curr_df)

            # Use performance comparison service directly
            # CRITICAL FIX: Pass separate y_true arrays for reference and current datasets
            performance_result = performance_comparison_service.analyze_performance_comparison(
                y_true=y_curr,  # DEPRECATED - kept for backward compatibility
                pred_ref=model.predict(X_ref),  # Reference predictions
                pred_curr=model.predict(X_curr),  # Current predictions
                pred_ref_proba=model.predict_proba(X_ref) if hasattr(model, "predict_proba") else None,
                pred_curr_proba=model.predict_proba(X_curr) if hasattr(model, "predict_proba") else None,
                X=X_curr,
                model_ref=model,
                model_curr=model,
                y_true_ref=y_ref,  # NEW: Separate reference ground truth
                y_true_curr=y_curr,  # NEW: Separate current ground truth
            )

            return {
                "analysis_id": analysis_id,
                "model_info": model_info,
                "timestamp": datetime.now().isoformat(),
                "config": config.dict(),
                "performance_comparison": performance_result,
            }

        except Exception as e:
            error_msg = f"Performance comparison analysis failed: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg)

    async def run_degradation_metrics_analysis(
        self,
        reference_data: UploadFile,
        current_data: UploadFile,
        model_file: UploadFile,
        config: AnalysisConfiguration,
    ) -> dict:
        """
        Run Tab 2: Degradation Metrics analysis (with sub-tabs)

        Args:
            reference_data: Reference dataset file
            current_data: Current dataset file
            model_file: Model file
            config: Analysis configuration from frontend

        Returns:
            Degradation metrics results with sub-tab structure
        """
        try:
            analysis_id = f"degradation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Load and validate data
            ref_df, curr_df, model = await self._load_and_validate_inputs(reference_data, current_data, model_file)

            # Prepare data based on model type
            X_ref, X_curr, y_ref, y_curr = await self._prepare_data(ref_df, curr_df, model, config.model_type)

            # Get model information
            model_info = self._extract_model_info(model, ref_df, curr_df)

            # Use degradation metrics service directly
            degradation_result = degradation_metrics_service.analyze_degradation_metrics(
                y_true=y_curr,  # Use current data as test set
                pred_ref=model.predict(X_ref),  # Reference predictions
                pred_curr=model.predict(X_curr),  # Current predictions
                pred_ref_proba=model.predict_proba(X_ref) if hasattr(model, "predict_proba") else None,
                pred_curr_proba=model.predict_proba(X_curr) if hasattr(model, "predict_proba") else None,
                X_ref=X_ref,
                y_ref=y_ref,
                X_curr=X_curr,
                y_curr=y_curr,
                model_ref=model,
                model_curr=model,
                feature_names=list(ref_df.columns[:-1]),  # Exclude target column
            )

            return {
                "analysis_id": analysis_id,
                "model_info": model_info,
                "timestamp": datetime.now().isoformat(),
                "config": config.dict(),
                "degradation_metrics": degradation_result,
            }

        except Exception as e:
            error_msg = f"Degradation metrics analysis failed: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg)

    async def run_statistical_significance_analysis(
        self,
        reference_data: UploadFile,
        current_data: UploadFile,
        model_file: UploadFile,
        config: AnalysisConfiguration,
    ) -> dict:
        """
        Run Tab 3: Statistical Significance analysis

        Args:
            reference_data: Reference dataset file
            current_data: Current dataset file
            model_file: Model file
            config: Analysis configuration from frontend

        Returns:
            Statistical significance results
        """
        try:
            analysis_id = f"statistical_significance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Load and validate data
            ref_df, curr_df, model = await self._load_and_validate_inputs(reference_data, current_data, model_file)

            # Prepare data based on model type
            X_ref, X_curr, y_ref, y_curr = await self._prepare_data(ref_df, curr_df, model, config.model_type)

            # Get model information
            model_info = self._extract_model_info(model, ref_df, curr_df)

            # Use statistical significance service directly
            statistical_result = statistical_significance_service.analyze_statistical_significance(
                y_true=y_curr,  # Use current data as test set
                pred_ref=model.predict(X_ref),  # Reference predictions
                pred_curr=model.predict(X_curr),  # Current predictions
                pred_ref_proba=model.predict_proba(X_ref) if hasattr(model, "predict_proba") else None,
                pred_curr_proba=model.predict_proba(X_curr) if hasattr(model, "predict_proba") else None,
                X=X_curr,
                model_ref=model,
                model_curr=model,
            )

            return {
                "analysis_id": analysis_id,
                "model_info": model_info,
                "timestamp": datetime.now().isoformat(),
                "config": config.dict(),
                "statistical_significance": statistical_result,
            }

        except Exception as e:
            error_msg = f"Statistical significance analysis failed: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg)

    async def run_sanity_check_analysis(
        self,
        reference_data: UploadFile,
        current_data: UploadFile,
        model_file: Optional[UploadFile] = None,
        config: Optional[AnalysisConfiguration] = None,
    ) -> dict:
        """
        Run Sanity Check analysis - Pre-flight diagnostic

        Args:
            reference_data: Reference dataset file
            current_data: Current dataset file
            model_file: Model file (optional, for feature importance)
            config: Analysis configuration (optional)

        Returns:
            Sanity check results with similarity analysis and performance estimation
        """
        try:
            from ..services.analysis.sanity_check_service import sanity_check_service

            analysis_id = f"sanity_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Load CSV files
            ref_content = await reference_data.read()
            curr_content = await current_data.read()

            ref_df = pd.read_csv(io.StringIO(ref_content.decode("utf-8")))
            curr_df = pd.read_csv(io.StringIO(curr_content.decode("utf-8")))

            # Validate dataframes
            if ref_df.empty or curr_df.empty:
                raise ValueError("Empty dataframe detected")

            # Load model if provided
            model = None
            model_info = None
            if model_file is not None:
                model_content = await model_file.read()
                try:
                    model = pickle.loads(model_content)
                except:
                    try:
                        model = joblib.load(io.BytesIO(model_content))
                    except Exception as e:
                        print(f"Warning: Could not load model: {str(e)}")
                        model = None

                if model is not None:
                    model_info = self._extract_model_info(model, ref_df, curr_df)

            # Determine model type
            model_type = "classification"
            if config is not None:
                model_type = config.model_type.value if isinstance(config.model_type, ModelType) else config.model_type

            # Determine target column
            target_column = None
            if config is not None and hasattr(config, "target_column"):
                target_column = config.target_column

            # Get feature names (exclude target if known)
            feature_names = list(ref_df.columns)
            if target_column and target_column in feature_names:
                feature_names.remove(target_column)

            # Run sanity check analysis
            sanity_result = sanity_check_service.analyze_sanity_check(
                reference_df=ref_df,
                current_df=curr_df,
                target_column=target_column,
                model=model,
                feature_names=feature_names,
                model_type=model_type,
            )

            # Check for errors
            if "error" in sanity_result:
                raise ValueError(sanity_result["error"])

            # Build response
            response = {
                "analysis_id": analysis_id,
                "timestamp": datetime.now().isoformat(),
                "model_info": model_info,
                "sanity_check_summary": sanity_result["sanity_check_summary"],
                "scatter_points": sanity_result["scatter_points"],
                "distribution_comparisons": sanity_result["distribution_comparisons"],
                "recommendations": sanity_result["recommendations"],
                "warnings": sanity_result.get("warnings", []),
            }

            return response

        except Exception as e:
            error_msg = f"Sanity check analysis failed: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg)

    async def _load_and_validate_inputs(
        self, reference_data: UploadFile, current_data: UploadFile, model_file: UploadFile
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """Load and validate input files"""

        # Read CSV files
        ref_content = await reference_data.read()
        curr_content = await current_data.read()
        model_content = await model_file.read()

        # Load dataframes
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

        # Validate datasets have same columns
        if list(ref_df.columns) != list(curr_df.columns):
            raise ValueError(
                f"Column mismatch between datasets. Reference: {list(ref_df.columns)}, Current: {list(curr_df.columns)}"
            )

        if len(ref_df.columns) < 2:
            raise ValueError(f"Dataset must have at least 2 columns (features + target). Found: {len(ref_df.columns)}")

        # Load model
        model = await self._load_model(model_file, model_content)

        return ref_df, curr_df, model

    async def _load_model(self, model_file: UploadFile, model_content: bytes) -> Any:
        """Load model with enhanced error handling"""

        model_filename = model_file.filename.lower()
        try:
            if model_filename.endswith((".joblib", ".pkl.joblib")):
                model = joblib.load(io.BytesIO(model_content))
            elif model_filename.endswith((".pkl", ".pickle")):
                model = pickle.loads(model_content)
            else:
                try:
                    model = pickle.loads(model_content)
                except:
                    model = joblib.load(io.BytesIO(model_content))
        except Exception as e:
            raise ValueError(f"Could not load model file. Supported formats: .pkl, .pickle, .joblib. Error: {str(e)}")

        # Validate model object
        if not hasattr(model, "predict"):
            if isinstance(model, dict):
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
                        for step_name, step_model in model[key].items():
                            if hasattr(step_model, "predict"):
                                model = step_model
                                break
                else:
                    available_keys = list(model.keys()) if isinstance(model, dict) else []
                    raise ValueError(f"Model object does not have 'predict' method. Available keys: {available_keys}")
            else:
                raise ValueError(f"Model object does not have 'predict' method. Type: {type(model)}")

        return model

    async def _prepare_data(
        self, ref_df: pd.DataFrame, curr_df: pd.DataFrame, model: Any, model_type: ModelType
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for analysis"""

        # Assume last column is target
        ref_X = ref_df.iloc[:, :-1]
        ref_y = ref_df.iloc[:, -1]
        curr_X = curr_df.iloc[:, :-1]
        curr_y = curr_df.iloc[:, -1]

        # Check if model is a pipeline
        is_pipeline = hasattr(model, "named_steps") or hasattr(model, "steps")

        try:
            if is_pipeline:
                # Try with raw data first (pipeline might include preprocessing)
                X_ref_processed = ref_X.values
                X_curr_processed = curr_X.values
                y_ref_processed = ref_y.values
                y_curr_processed = curr_y.values

                # Test prediction
                _ = model.predict(X_ref_processed[:1])

            else:
                # Preprocess data for non-pipeline models
                X_ref_processed, X_curr_processed, y_ref_processed, y_curr_processed = self._preprocess_data(
                    ref_X, curr_X, ref_y, curr_y
                )

        except Exception as e:
            # Fallback strategies
            try:
                if is_pipeline:
                    # Pipeline failed with raw data, try preprocessing
                    X_ref_processed, X_curr_processed, y_ref_processed, y_curr_processed = self._preprocess_data(
                        ref_X, curr_X, ref_y, curr_y
                    )
                else:
                    # Non-pipeline failed with preprocessing, try raw data
                    X_ref_processed = ref_X.values
                    X_curr_processed = curr_X.values
                    y_ref_processed = ref_y.values
                    y_curr_processed = curr_y.values

                # Test prediction with fallback
                _ = model.predict(X_ref_processed[:1])

            except Exception as e2:
                # Final fallback: minimal preprocessing
                X_ref_processed, X_curr_processed, y_ref_processed, y_curr_processed = self._minimal_preprocess_data(
                    ref_X, curr_X, ref_y, curr_y
                )

        return X_ref_processed, X_curr_processed, y_ref_processed, y_curr_processed

    def _preprocess_data(
        self, X_ref: pd.DataFrame, X_curr: pd.DataFrame, y_ref: pd.Series, y_curr: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Comprehensive data preprocessing"""

        # Make copies
        X_ref_processed = X_ref.copy()
        X_curr_processed = X_curr.copy()
        y_ref_processed = y_ref.copy()
        y_curr_processed = y_curr.copy()

        # Handle missing values
        numerical_cols = X_ref_processed.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            imputer_num = SimpleImputer(strategy="median")
            X_ref_processed[numerical_cols] = imputer_num.fit_transform(X_ref_processed[numerical_cols])
            X_curr_processed[numerical_cols] = imputer_num.transform(X_curr_processed[numerical_cols])

        categorical_cols = X_ref_processed.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy="most_frequent")
            X_ref_processed[categorical_cols] = imputer_cat.fit_transform(X_ref_processed[categorical_cols])
            X_curr_processed[categorical_cols] = imputer_cat.transform(X_curr_processed[categorical_cols])

        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            combined_values = pd.concat([X_ref_processed[col], X_curr_processed[col]]).astype(str)
            le.fit(combined_values)
            X_ref_processed[col] = le.transform(X_ref_processed[col].astype(str))
            X_curr_processed[col] = le.transform(X_curr_processed[col].astype(str))

        # Handle target variable encoding if categorical
        if y_ref_processed.dtype == "object" or y_ref_processed.dtype.name == "category":
            target_encoder = LabelEncoder()
            combined_targets = pd.concat([y_ref_processed, y_curr_processed]).astype(str)
            target_encoder.fit(combined_targets)
            y_ref_processed = target_encoder.transform(y_ref_processed.astype(str))
            y_curr_processed = target_encoder.transform(y_curr_processed.astype(str))

        # Scale numerical features
        if len(numerical_cols) > 0:
            scaler = RobustScaler()
            X_ref_processed[numerical_cols] = scaler.fit_transform(X_ref_processed[numerical_cols])
            X_curr_processed[numerical_cols] = scaler.transform(X_curr_processed[numerical_cols])

        return (
            X_ref_processed.values.astype(np.float64),
            X_curr_processed.values.astype(np.float64),
            np.array(y_ref_processed, dtype=np.float64),
            np.array(y_curr_processed, dtype=np.float64),
        )

    def _minimal_preprocess_data(
        self, X_ref: pd.DataFrame, X_curr: pd.DataFrame, y_ref: pd.Series, y_curr: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Minimal preprocessing - only categorical encoding"""

        X_ref_simple = X_ref.copy()
        X_curr_simple = X_curr.copy()

        # Only encode categorical variables
        categorical_cols = X_ref_simple.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            le = LabelEncoder()
            combined_values = pd.concat([X_ref_simple[col], X_curr_simple[col]]).astype(str)
            le.fit(combined_values)
            X_ref_simple[col] = le.transform(X_ref_simple[col].astype(str))
            X_curr_simple[col] = le.transform(X_curr_simple[col].astype(str))

        return (X_ref_simple.values, X_curr_simple.values, y_ref.values, y_curr.values)

    def _extract_model_info(self, model: Any, ref_df: pd.DataFrame, curr_df: pd.DataFrame) -> ModelInfo:
        """Extract model information with automatic type detection"""

        is_pipeline = hasattr(model, "named_steps") or hasattr(model, "steps")

        # Extract algorithm name (for pipelines, get the final estimator)
        if is_pipeline:
            if hasattr(model, "named_steps"):
                # Get the last step which should be the estimator
                list(model.named_steps.keys())[-1]
            elif hasattr(model, "steps"):
                model.steps[-1][1].__class__.__name__
            else:
                str(type(model).__name__)
        else:
            str(type(model).__name__)

        # Detect if model is regression or classification
        detected_type = self._detect_model_type(model)

        return ModelInfo(
            model_type=str(type(model).__name__),
            model_class=model.__class__.__module__ + "." + model.__class__.__name__,
            reference_samples=len(ref_df),
            current_samples=len(curr_df),
            features=list(ref_df.columns[:-1]),  # Exclude target column
            is_pipeline=is_pipeline,
            detected_model_type=detected_type,  # Add detected type (regression or classification)
        )

    def _calculate_selected_metrics(
        self,
        y_ref: np.ndarray,
        y_curr: np.ndarray,
        X_ref: np.ndarray,
        X_curr: np.ndarray,
        model: Any,
        config: AnalysisConfiguration,
    ) -> PerformanceMetrics:
        """Calculate only the selected performance metrics"""

        # Get predictions
        ref_pred = model.predict(X_ref)
        curr_pred = model.predict(X_curr)

        # Get prediction probabilities if available
        ref_pred_proba = None
        curr_pred_proba = None
        if hasattr(model, "predict_proba"):
            try:
                ref_pred_proba = model.predict_proba(X_ref)
                curr_pred_proba = model.predict_proba(X_curr)
            except:
                pass

        # Select appropriate metric calculators
        if config.model_type == ModelType.CLASSIFICATION:
            metric_calculators = self.classification_metrics
        else:
            metric_calculators = self.regression_metrics

        # Calculate selected metrics
        ref_metrics = {}
        curr_metrics = {}

        for metric_name in config.selected_metrics:
            if metric_name in metric_calculators:
                calculator = metric_calculators[metric_name]
                ref_metrics[metric_name] = calculator(y_ref, ref_pred, ref_pred_proba)
                curr_metrics[metric_name] = calculator(y_curr, curr_pred, curr_pred_proba)

        return PerformanceMetrics(reference=ref_metrics, current=curr_metrics)

    # Classification metric calculators
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        return accuracy_score(y_true, y_pred)

    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        return precision_score(y_true, y_pred, average="weighted", zero_division=0)

    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        return recall_score(y_true, y_pred, average="weighted", zero_division=0)

    def _calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)

    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        # Calculate specificity for binary classification
        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                return tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                # For multiclass, return macro average specificity
                specificity_scores = []
                for i in range(len(np.unique(y_true))):
                    y_true_binary = (y_true == i).astype(int)
                    y_pred_binary = (y_pred == i).astype(int)
                    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
                    if cm_binary.shape == (2, 2):
                        tn, fp, fn, tp = cm_binary.ravel()
                        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        specificity_scores.append(spec)
                return np.mean(specificity_scores) if specificity_scores else 0.0
        except:
            return 0.0

    def _calculate_roc_auc(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        try:
            if y_pred_proba is not None:
                if len(np.unique(y_true)) == 2:
                    return roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    return roc_auc_score(y_true, y_pred_proba, multi_class="ovr")
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_pr_auc(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        try:
            if y_pred_proba is not None:
                from sklearn.metrics import average_precision_score

                if len(np.unique(y_true)) == 2:
                    return average_precision_score(y_true, y_pred_proba[:, 1])
                else:
                    return average_precision_score(y_true, y_pred_proba, average="weighted")
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_cohen_kappa(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        try:
            from sklearn.metrics import cohen_kappa_score

            return cohen_kappa_score(y_true, y_pred)
        except:
            return 0.0

    def _calculate_mcc(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        try:
            from sklearn.metrics import matthews_corrcoef

            return matthews_corrcoef(y_true, y_pred)
        except:
            return 0.0

    # Regression metric calculators
    def _calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        return mean_squared_error(y_true, y_pred)

    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        return mean_absolute_error(y_true, y_pred)

    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        return r2_score(y_true, y_pred)

    def _calculate_adjusted_r2(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        try:
            r2 = r2_score(y_true, y_pred)
            n = len(y_true)
            p = 1  # Simplified assumption for number of predictors
            return 1 - (1 - r2) * (n - 1) / (n - p - 1)
        except:
            return 0.0

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        try:
            return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        except:
            return 0.0

    def _calculate_explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        return explained_variance_score(y_true, y_pred)

    def _calculate_max_error(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba=None) -> float:
        return max_error(y_true, y_pred)

    def _calculate_drift_metrics(self, performance_metrics: PerformanceMetrics, thresholds) -> DriftMetrics:
        """Calculate drift metrics based on performance differences"""

        metric_drifts = {}
        drift_values = []

        for metric_name in performance_metrics.reference.keys():
            ref_value = performance_metrics.reference[metric_name]
            curr_value = performance_metrics.current[metric_name]
            drift = abs(ref_value - curr_value)
            metric_drifts[metric_name] = drift
            drift_values.append(drift)

        overall_drift = max(drift_values) if drift_values else 0.0

        # Determine severity
        if overall_drift >= thresholds.high_threshold:
            severity = "High"
        elif overall_drift >= thresholds.medium_threshold:
            severity = "Medium"
        elif overall_drift >= thresholds.low_threshold:
            severity = "Low"
        else:
            severity = "None"

        return DriftMetrics(
            metric_drifts=metric_drifts,
            overall_drift=overall_drift,
            drift_severity=severity,
            drift_detected=severity in ["Medium", "High"],
        )

    async def _run_selected_statistical_test(
        self, y_ref: np.ndarray, y_curr: np.ndarray, X_ref: np.ndarray, X_curr: np.ndarray, model: Any, test_name: str
    ) -> Optional[StatisticalTestResult]:
        """Run the selected statistical test"""

        try:
            ref_pred = model.predict(X_ref)
            curr_pred = model.predict(X_curr)

            # Get prediction probabilities if available
            ref_pred_proba = None
            curr_pred_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    ref_pred_proba = model.predict_proba(X_ref)
                    curr_pred_proba = model.predict_proba(X_curr)
                except:
                    pass

            # Run specific test
            if test_name == "mcnemar":
                result = statistical_tests_service.mcnemar_test(y_ref, ref_pred, curr_pred)
            elif test_name == "delong":
                result = statistical_tests_service.delong_test(y_ref, ref_pred_proba, curr_pred_proba)
            elif test_name == "five_two_cv":
                result = statistical_tests_service.five_two_cv_test(y_ref, ref_pred, curr_pred, X_ref, model, model)
            elif test_name == "bootstrap_confidence":
                result = statistical_tests_service.bootstrap_test(y_ref, ref_pred, curr_pred)
            elif test_name == "diebold_mariano":
                result = statistical_tests_service.diebold_mariano_test(y_ref, ref_pred, curr_pred)
            elif test_name == "paired_ttest":
                result = statistical_tests_service.paired_t_test(y_ref, ref_pred, curr_pred)
            else:
                return None

            if "error" in result:
                return None

            return StatisticalTestResult(
                test_name=result.get("test_name", test_name),
                test_type=result.get("test_type", "Unknown"),
                p_value=result.get("p_value", 1.0),
                test_statistic=result.get("test_statistic", result.get("z_score", result.get("f_statistic", 0.0))),
                significant=result.get("significant", False),
                interpretation=result.get("interpretation", "No significant difference detected"),
                confidence_level=0.95,
            )

        except Exception as e:
            print(f"Statistical test {test_name} failed: {str(e)}")
            return None

    def _generate_recommendations(
        self, drift_metrics: DriftMetrics, statistical_result: Optional[StatisticalTestResult]
    ) -> List[str]:
        """Generate recommendations based on analysis results"""

        recommendations = []

        if drift_metrics.drift_detected:
            recommendations.append(f"Model drift detected with {drift_metrics.drift_severity.lower()} severity")
            recommendations.append("Consider retraining the model with recent data")
            recommendations.append("Monitor model performance more frequently")
        else:
            recommendations.append("No significant model drift detected")
            recommendations.append("Continue regular monitoring schedule")

        if statistical_result and statistical_result.significant:
            recommendations.append(f"Statistical significance confirmed by {statistical_result.test_name}")
            recommendations.append("The performance difference is statistically reliable")
        elif statistical_result and not statistical_result.significant:
            recommendations.append("Performance differences are not statistically significant")
            recommendations.append("Differences may be due to random variation")

        if drift_metrics.overall_drift > 0.1:
            recommendations.append("Large performance difference detected - immediate attention recommended")

        return recommendations


# Service instance
enhanced_model_service = EnhancedModelService()
