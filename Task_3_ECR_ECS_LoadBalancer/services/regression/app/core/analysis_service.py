from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

from services.regression.app.logging_config import logger

from .base_model_service import BaseModelService


class AnalysisService:
    """
    Service for regression model analysis operations.
    Provides overview, regression stats, instances listing, and dataset comparison.
    """

    def __init__(self, base_service: BaseModelService):
        self.base = base_service

    def get_model_overview(self) -> Dict[str, Any]:
        """Get comprehensive model overview including performance metrics and metadata."""
        self.base._is_ready()

        # Calculate performance metrics on training data
        y_pred_train = self.base.model.predict(self.base.X_train.values)
        train_metrics = self.base._get_regression_metrics(self.base.y_train, y_pred_train)
        model_type = "regression"

        # Calculate performance metrics on test data (if available)
        test_metrics = None
        overfitting_score = 0.0
        if self.base.X_test is not None and self.base.y_test is not None:
            y_pred_test = self.base.model.predict(self.base.X_test.values)
            test_metrics = self.base._get_regression_metrics(self.base.y_test, y_pred_test)
            # For regression, use R² difference as overfitting indicator
            overfitting_score = max(0.0, train_metrics["r2_score"] - test_metrics["r2_score"])

        # Build feature schema
        feature_schema = []
        for feature in self.base.feature_names:
            col_data = self.base.X_train[feature]
            is_numeric = pd.api.types.is_numeric_dtype(col_data.dtype)

            feature_info = {
                "name": feature,
                "type": "numerical" if is_numeric else "categorical",
                "missing_count": int(col_data.isnull().sum()),
                "unique_count": int(col_data.nunique()),
            }

            if is_numeric:
                feature_info.update(
                    {
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                    }
                )
            else:
                feature_info["categories"] = col_data.value_counts().head(10).to_dict()

            feature_schema.append(feature_info)

        performance_metrics = {"train": train_metrics, "overfitting_score": overfitting_score}

        if test_metrics:
            performance_metrics["test"] = test_metrics

        return {
            "model_id": self.base.model_info.get("model_path", "N/A"),
            "name": f"Uploaded Regression Model",
            "model_type": model_type,
            "version": self.base.model_info.get("version", "1.0.0"),
            "framework": self.base.model_info.get("framework", "scikit-learn"),
            "status": self.base.model_info.get("status", "Active"),
            "algorithm": self.base.model_info.get("algorithm", "Unknown"),
            "feature_names": self.base.feature_names,
            "schema": feature_schema,
            "performance_metrics": performance_metrics,
            "shap_available": self.base.explainer is not None,
            "input_format": self.base.get_model_input_format(),
            "metadata": {
                "created": self.base.model_info.get("created", "N/A"),
                "last_trained": self.base.model_info.get("last_trained", "N/A"),
                "samples": len(self.base.X_train) + (len(self.base.X_test) if self.base.X_test is not None else 0),
                "features": len(self.base.feature_names),
                "train_samples": self.base.model_info.get("train_samples", len(self.base.X_train)),
                "test_samples": self.base.model_info.get(
                    "test_samples", len(self.base.X_test) if self.base.X_test is not None else 0
                ),
                "dataset_split": {
                    "train": len(self.base.X_train),
                    "test": len(self.base.X_test) if self.base.X_test is not None else 0,
                },
                "missing_pct": self.base.model_info.get("missing_pct", 0.0),
                "duplicates_pct": self.base.model_info.get("duplicates_pct", 0.0),
                "health_score_pct": self.base.model_info.get("health_score_pct", 100.0),
                "data_source": self.base.model_info.get("data_source", "single_dataset"),
            },
        }

    def get_classification_stats(self) -> Dict[str, Any]:
        """This endpoint is not supported for regression models."""
        return {
            "error": "Classification stats are not available for regression models. Use get_regression_stats instead."
        }

    def get_regression_stats(self) -> Dict[str, Any]:
        """Get detailed regression statistics including diagnostic plots and performance metrics."""
        self.base._is_ready()

        # Use test data if available, otherwise fall back to training data
        if self.base.X_test is not None and self.base.y_test is not None:
            X_eval, y_eval = self.base.X_test, self.base.y_test
            data_source = "test"
        else:
            X_eval, y_eval = self.base.X_train, self.base.y_train
            data_source = "train"

        y_pred = self.base.safe_predict(X_eval)
        metrics = self.base._get_regression_metrics(y_eval, y_pred)

        # Calculate residuals
        residuals = y_eval - y_pred

        # Generate diagnostic plots
        diagnostic_plots = self._generate_diagnostic_plots(y_eval, y_pred, residuals)

        # Performance summary
        performance_summary = self._calculate_performance_summary(metrics, residuals, y_eval, y_pred)

        return {
            "metrics": metrics,
            "data_source": data_source,
            "diagnostic_plots": diagnostic_plots,
            "performance_summary": performance_summary,
            "residual_stats": {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "median": float(np.median(residuals)),
            },
        }

    def _generate_diagnostic_plots(self, y_true, y_pred, residuals) -> Dict[str, Any]:
        """Generate diagnostic plot data for regression analysis (raw data only, no images)."""
        plots = {}
        try:
            # 1. Residual Plot (Residuals vs Fitted Values)
            # Trend line
            z = np.polyfit(y_pred, residuals, 1)
            p = np.poly1d(z)
            y_pred_sorted = np.sort(y_pred)
            trend_line = p(y_pred_sorted)
            plots["residual_plot"] = {
                "fitted_values": y_pred.tolist(),
                "residuals": residuals.tolist(),
                "trend_line": {
                    "x": y_pred_sorted.tolist(),
                    "y": trend_line.tolist(),
                    "slope": float(z[0]),
                    "intercept": float(z[1]),
                },
            }

            # 2. Q-Q Plot for normality check
            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
            plots["qq_plot"] = {
                "theoretical_quantiles": osm.tolist(),
                "ordered_residuals": osr.tolist(),
                "slope": float(slope),
                "intercept": float(intercept),
                "correlation": float(r),
            }

            # 3. Predicted vs Actual Values
            min_val = float(min(min(y_true), min(y_pred)))
            max_val = float(max(max(y_true), max(y_pred)))
            plots["predicted_vs_actual"] = {
                "actual": y_true.tolist(),
                "predicted": y_pred.tolist(),
                "perfect_line": {"x": [min_val, max_val], "y": [min_val, max_val]},
            }

            # 4. Residual Distribution
            hist, bin_edges = np.histogram(residuals, bins=30, density=True)
            mu, sigma = float(np.mean(residuals)), float(np.std(residuals))
            x = np.linspace(np.min(residuals), np.max(residuals), 100)
            normal_curve = stats.norm.pdf(x, mu, sigma)
            plots["residual_distribution"] = {
                "histogram": {"bin_edges": bin_edges.tolist(), "counts": hist.tolist()},
                "normal_curve": {"x": x.tolist(), "y": normal_curve.tolist(), "mu": mu, "sigma": sigma},
            }
        except Exception as e:
            print(f"Error generating diagnostic plots: {e}")
            plots = {"error": f"Failed to generate plots: {str(e)}"}
        return plots

    def _calculate_performance_summary(
        self, metrics: Dict[str, float], residuals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate performance summary for regression model."""

        # Model Quality Assessment
        r2_score = metrics.get("r2_score", 0.0)
        explained_variance = metrics.get("explained_variance", 0.0)

        if r2_score >= 0.9:
            model_fit = "Excellent"
        elif r2_score >= 0.8:
            model_fit = "Good"
        elif r2_score >= 0.6:
            model_fit = "Moderate"
        else:
            model_fit = "Poor"

        # Overfitting Risk Assessment
        train_pred = self.base.safe_predict(self.base.X_train)
        train_metrics = self.base._get_regression_metrics(self.base.y_train, train_pred)
        overfitting_score = max(0.0, train_metrics["r2_score"] - metrics["r2_score"])

        if overfitting_score > 0.1:
            overfitting_risk = "High"
        elif overfitting_score > 0.05:
            overfitting_risk = "Medium"
        else:
            overfitting_risk = "Low"

        # Error Analysis
        mae = metrics.get("mae", 0.0)
        metrics.get("rmse", 0.0)
        metrics.get("mape", 0.0)

        # Calculate prediction spread
        prediction_range = np.max(y_pred) - np.min(y_pred)
        actual_range = np.max(y_true) - np.min(y_true)
        prediction_spread = prediction_range / actual_range if actual_range > 0 else 0.0

        # Relative error assessment
        mean_actual = np.mean(np.abs(y_true))
        relative_mae = (mae / mean_actual) * 100 if mean_actual > 0 else 0.0

        return {
            "model_quality": {
                "explained_variance": explained_variance,
                "model_fit": model_fit,
                "overfitting_risk": overfitting_risk,
            },
            "error_analysis": {
                "average_error": mae,
                "prediction_spread": prediction_spread,
                "relative_error_pct": relative_mae,
            },
        }

    def list_instances(self, sort_by: str = "prediction", limit: int = 100) -> Dict[str, Any]:
        """Return lightweight list of instances to populate selector UI."""

        logger.debug(f"Listing instances | sort_by={sort_by}, limit={limit}")
        self.base._is_ready()

        # Use training data for instance listing
        predictions = self.base.safe_predict(self.base.X_df)
        records = []

        # Calculate max_error once for all instances for consistent confidence values
        max_error = max(abs(predictions - self.base.y_s), default=1.0)
        negative_confidence_count = 0

        for idx in range(len(self.base.X_df)):
            # For regression, calculate prediction confidence based on model uncertainty
            pred_value = float(predictions[idx])
            actual_value = float(self.base.y_s.iloc[idx])
            error = abs(pred_value - actual_value)

            # Clamp confidence to valid range [0, 1]
            raw_confidence = 1.0 - (error / max_error) if max_error > 0 else 1.0
            confidence = max(0.0, min(1.0, raw_confidence))

            if raw_confidence < 0:
                negative_confidence_count += 1

            records.append(
                {
                    "id": idx,  # Changed from "index" to "id" to match frontend expectations
                    "prediction": pred_value,
                    "actual": actual_value,
                    "confidence": float(confidence),
                    "error": float(error),
                }
            )

        # Log summary if negative confidences were detected
        if negative_confidence_count > 0:
            logger.warning(
                f"Detected {negative_confidence_count} instances with negative confidence scores "
                f"(clamped to 0.0). This indicates the model has very high errors on these instances."
            )

        logger.debug(f"Generated {len(records)} records before sorting.")
        if sort_by == "prediction":
            records.sort(key=lambda x: x["prediction"], reverse=True)
        elif sort_by == "confidence":
            records.sort(key=lambda x: x["confidence"], reverse=True)
        elif sort_by == "error":
            records.sort(key=lambda x: x.get("error", 0), reverse=True)
        logger.debug(f"Returning {min(limit, len(records))} records after sorting.")
        return {"instances": records[:limit], "total": len(self.base.X_df)}

    def get_dataset_comparison(self) -> Dict[str, Any]:
        """Compare train vs test dataset characteristics for basic model evaluation."""
        self.base._is_ready()

        if self.base.X_test is None or self.base.y_test is None:
            return {
                "error": "No test dataset available for comparison. Upload separate train/test datasets to enable this analysis."
            }

        # Calculate performance metrics on training data
        y_pred_train = self.base.safe_predict(self.base.X_train)
        train_metrics = self.base._get_regression_metrics(self.base.y_train, y_pred_train)

        # Calculate performance metrics on test data
        y_pred_test = self.base.safe_predict(self.base.X_test)
        test_metrics = self.base._get_regression_metrics(self.base.y_test, y_pred_test)

        # Use R² difference for overfitting in regression
        overfitting_score = max(0.0, train_metrics["r2_score"] - test_metrics["r2_score"])

        # Basic dataset statistics
        train_missing_pct = (
            self.base.X_train.isnull().sum().sum() / (self.base.X_train.shape[0] * self.base.X_train.shape[1])
        ) * 100
        test_missing_pct = (
            self.base.X_test.isnull().sum().sum() / (self.base.X_test.shape[0] * self.base.X_test.shape[1])
        ) * 100
        train_duplicates_pct = (self.base.X_train.duplicated().sum() / len(self.base.X_train)) * 100
        test_duplicates_pct = (self.base.X_test.duplicated().sum() / len(self.base.X_test)) * 100

        # Overfitting risk assessment
        risk_level = "low"
        if overfitting_score > 0.1:
            risk_level = "high"
        elif overfitting_score > 0.05:
            risk_level = "medium"

        model_type = "regression"

        return {
            # Performance metrics comparison (what frontend uses)
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "overfitting_metrics": {
                "overfitting_score": overfitting_score,
                "risk_level": risk_level,
                "interpretation": f"Model shows {'high' if overfitting_score > 0.1 else 'moderate' if overfitting_score > 0.05 else 'low'} overfitting",
                "model_type": model_type,
            },
            # Basic dataset info (what frontend uses)
            "train_samples": len(self.base.X_train),
            "test_samples": len(self.base.X_test),
            "train_missing_pct": train_missing_pct,
            "test_missing_pct": test_missing_pct,
            "train_duplicates_pct": train_duplicates_pct,
            "test_duplicates_pct": test_duplicates_pct,
        }
