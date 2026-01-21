from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from services.classification.app.logging_config import logger

from .base_model_service import BaseModelService


class AnalysisService:
    """
    Service for basic model analysis operations like overview, classification stats,
    instances listing, and dataset comparison.
    """

    def __init__(self, base_service: BaseModelService):
        self.base = base_service

    def get_model_overview(self, payload: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get comprehensive model overview including performance metrics and metadata."""

        # Use the helper method for dynamic loading
        self.base._ensure_loaded_with_payload(payload)

        # Calculate performance metrics on training data
        y_pred_train = self.base.model.predict(self.base.X_train)
        y_proba_train = self.base.model.predict_proba(self.base.X_train)

        train_metrics, is_binary = self.base._get_classification_metrics(self.base.y_train, y_pred_train, y_proba_train)

        # Calculate performance metrics on test data (if available)
        test_metrics = None
        overfitting_score = 0.0
        if self.base.X_test is not None and self.base.y_test is not None:
            y_pred_test = self.base.model.predict(self.base.X_test)
            y_proba_test = self.base.model.predict_proba(self.base.X_test)

            test_metrics, _ = self.base._get_classification_metrics(self.base.y_test, y_pred_test, y_proba_test)

            # Calculate overfitting score (difference in accuracy)
            overfitting_score = max(0.0, train_metrics["accuracy"] - test_metrics["accuracy"])

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

        # Determine model type based on classification type
        model_type = "binary_classification" if is_binary else "multiclass_classification"

        return {
            "model_id": self.base.model_info.get("model_path", "N/A"),
            "name": "Uploaded Classification Model",
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

    def get_classification_stats(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed classification statistics and confusion matrix."""
        import json
        import traceback

        try:
            logger.info("Starting classification stats calculation...")

            # Use the helper method for dynamic loading
            self.base._ensure_loaded_with_payload(payload)

            # Use test data if available, otherwise fall back to training data
            if self.base.X_test is not None and self.base.y_test is not None:
                X_eval, y_eval = self.base.X_test, self.base.y_test
                data_source = "test"
                logger.info(f"Using test data: {X_eval.shape}")
            else:
                X_eval, y_eval = self.base.X_train, self.base.y_train
                data_source = "train"
                logger.info(f"Using train data: {X_eval.shape}")
            logger.info(f"Target values: unique={np.unique(y_eval)}, shape={y_eval.shape}")

            y_pred = self.base.safe_predict(X_eval)
            logger.info(f"Predictions: unique={np.unique(y_pred)}, shape={y_pred.shape}")
            y_proba = self.base.safe_predict_proba(X_eval)
            logger.info(f"Probabilities: shape={y_proba.shape if y_proba is not None else None}")

            metrics, is_binary = self.base._get_classification_metrics(y_eval, y_pred, y_proba)
            logger.info(f"Calculated metrics: {metrics}")

            # Test JSON serialization of metrics
            try:
                json.dumps(metrics)
                logger.info("Metrics JSON serializable")
            except Exception as e:
                logger.error(f"Metrics NOT JSON serializable: {e}")
                logger.error(f"Problematic metrics: {metrics}")
                # Find the problematic values
                for key, value in metrics.items():
                    try:
                        json.dumps({key: value})
                    except Exception as ve:
                        logger.error(f"Problematic metric '{key}': {value} (type: {type(value)}) - {ve}")

            cm = confusion_matrix(y_eval, y_pred)
            logger.info(f"Confusion matrix shape: {cm.shape}")

            # ROC curve and Precision-Recall curve only for binary classification
            roc_curve_data = {}
            pr_curve_data = {}
            if is_binary and y_proba is not None:
                try:
                    # ROC curve
                    fpr, tpr, roc_thresholds = roc_curve(y_eval, y_proba[:, 1])
                    roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()}
                    logger.info(f"ROC curve data: fpr={len(fpr)}, tpr={len(tpr)}, thresholds={len(roc_thresholds)}")

                    # Precision-Recall curve
                    precision, recall, pr_thresholds = precision_recall_curve(y_eval, y_proba[:, 1])
                    pr_curve_data = {
                        "precision": precision.tolist(),
                        "recall": recall.tolist(),
                        "thresholds": pr_thresholds.tolist(),
                    }
                    print(
                        f"📊 PR curve data: precision={len(precision)}, recall={len(recall)}, thresholds={len(pr_thresholds)}"
                    )

                    # Test JSON serialization of curve data
                    try:
                        json.dumps(roc_curve_data)
                        json.dumps(pr_curve_data)
                        print("✅ ROC and PR curve data JSON serializable")
                    except Exception as e:
                        print(f"❌ Curve data NOT JSON serializable: {e}")

                except Exception as e:
                    print(f"❌ Error calculating curves: {e}")
                    traceback.print_exc()

            # Convert confusion matrix to standard format
            if cm.shape == (2, 2):
                confusion_matrix_dict = {
                    "true_negative": int(cm[0, 0]),
                    "false_positive": int(cm[0, 1]),
                    "false_negative": int(cm[1, 0]),
                    "true_positive": int(cm[1, 1]),
                }
            else:
                # Multi-class confusion matrix
                confusion_matrix_dict = {"matrix": cm.tolist(), "classes": sorted(np.unique(y_eval).tolist())}

            print(f"🏛️ Confusion matrix dict: {confusion_matrix_dict}")

            # Test JSON serialization of confusion matrix
            try:
                json.dumps(confusion_matrix_dict)
                print("✅ Confusion matrix JSON serializable")
            except Exception as e:
                print(f"❌ Confusion matrix NOT JSON serializable: {e}")

            result = {
                "metrics": metrics,
                "data_source": data_source,
                "confusion_matrix": confusion_matrix_dict,
                "roc_curve": roc_curve_data,
                "pr_curve": pr_curve_data,
                "classification_type": "binary" if is_binary else "multiclass",
            }

            # Test JSON serialization of final result
            try:
                json.dumps(result)
                print("✅ Final result JSON serializable")
            except Exception as e:
                print(f"❌ Final result NOT JSON serializable: {e}")
                # Find problematic parts
                for key, value in result.items():
                    try:
                        json.dumps({key: value})
                        print(f"✅ Result section '{key}' is JSON serializable")
                    except Exception as ve:
                        print(f"❌ Result section '{key}' NOT JSON serializable: {ve}")
                        print(f"🔍 Value: {value}")

            return result

        except Exception as e:
            print(f"💥 Error in get_classification_stats: {e}")
            traceback.print_exc()
            raise

    def list_instances(
        self, payload: Dict[str, Any] = None, sort_by: str = "prediction", limit: int = 100
    ) -> Dict[str, Any]:
        """Return lightweight list of instances to populate selector UI."""
        self.base._ensure_loaded_with_payload(payload)

        # Use training data for instance listing
        proba = self.base.safe_predict_proba(self.base.X_df)[:, 1]
        records = []

        for idx in range(len(self.base.X_df)):
            confidence = max(proba[idx], 1.0 - proba[idx])
            records.append(
                {
                    "index": idx,
                    "prediction": float(proba[idx]),
                    "actual": int(self.base.y_s.iloc[idx]),
                    "confidence": float(confidence),
                }
            )

        if sort_by == "prediction":
            records.sort(key=lambda x: x["prediction"], reverse=True)
        elif sort_by == "confidence":
            records.sort(key=lambda x: x["confidence"], reverse=True)

        return {"instances": records[:limit], "total": len(self.base.X_df)}

    def get_dataset_comparison(self, payload: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compare train vs test dataset characteristics for basic model evaluation."""
        self.base._ensure_loaded_with_payload(payload)

        if self.base.X_test is None or self.base.y_test is None:
            return {
                "error": "No test dataset available for comparison. Upload separate train/test datasets to enable this analysis."
            }

        # Calculate performance metrics on training data
        y_pred_train = self.base.safe_predict(self.base.X_train)
        y_proba_train = self.base.safe_predict_proba(self.base.X_train)
        train_metrics, is_binary = self.base._get_classification_metrics(self.base.y_train, y_pred_train, y_proba_train)

        # Calculate performance metrics on test data
        y_pred_test = self.base.safe_predict(self.base.X_test)
        y_proba_test = self.base.safe_predict_proba(self.base.X_test)
        test_metrics, _ = self.base._get_classification_metrics(self.base.y_test, y_pred_test, y_proba_test)

        # Calculate overfitting metrics
        overfitting_score = max(0.0, train_metrics["accuracy"] - test_metrics["accuracy"])

        # Basic dataset statistics
        train_missing_pct = (
            self.base.X_train.isnull().sum().sum() / (self.base.X_train.shape[0] * self.base.X_train.shape[1])
        ) * 100
        test_missing_pct = (
            self.base.X_test.isnull().sum().sum() / (self.base.X_test.shape[0] * self.base.X_test.shape[1])
        ) * 100
        train_duplicates_pct = (self.base.X_train.duplicated().sum() / len(self.base.X_train)) * 100
        test_duplicates_pct = (self.base.X_test.duplicated().sum() / len(self.base.X_test)) * 100

        # Simple overfitting risk assessment
        risk_level = "low"
        if overfitting_score > 0.1:
            risk_level = "high"
        elif overfitting_score > 0.05:
            risk_level = "medium"

        return {
            # Performance metrics comparison (what frontend uses)
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "overfitting_metrics": {
                "overfitting_score": overfitting_score,
                "risk_level": risk_level,
                "interpretation": f"Model shows {'high' if overfitting_score > 0.1 else 'moderate' if overfitting_score > 0.05 else 'low'} overfitting",
            },
            # Basic dataset info (what frontend uses)
            "train_samples": len(self.base.X_train),
            "test_samples": len(self.base.X_test),
            "train_missing_pct": train_missing_pct,
            "test_missing_pct": test_missing_pct,
            "train_duplicates_pct": train_duplicates_pct,
            "test_duplicates_pct": test_duplicates_pct,
        }
