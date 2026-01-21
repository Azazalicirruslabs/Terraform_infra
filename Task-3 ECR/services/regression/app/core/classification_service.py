from typing import Any, Dict

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from .base_model_service import BaseModelService


class ClassificationService:
    """
    Service for classification-specific analysis like ROC analysis and threshold analysis.
    """

    def __init__(self, base_service: BaseModelService):
        self.base = base_service

    def roc_analysis(self) -> Dict[str, Any]:
        """Compute ROC analysis for both binary and multiclass classification."""
        self.base._is_ready()

        # Check if this is actually a classification model
        if self.base._is_regression_model():
            return {
                "error": "ROC analysis is only available for classification models. This appears to be a regression model."
            }

        # Use test data if available, otherwise fall back to training data
        if self.base.X_test is not None and self.base.y_test is not None:
            X_eval, y_eval = self.base.X_test, self.base.y_test
            data_source = "test"
        else:
            X_eval, y_eval = self.base.X_train, self.base.y_train
            data_source = "train"

        y_true = y_eval.values
        y_proba = self.base.safe_predict_proba(X_eval)

        if y_proba is None:
            return {"error": "Model does not support probability predictions required for ROC analysis."}

        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        is_binary = n_classes == 2

        def to_finite_list(arr: np.ndarray, clamp_low: float = 0.0, clamp_high: float = 1.0) -> list:
            """Convert array to list with finite values, clamping infinities."""
            arr_clean = np.nan_to_num(arr, nan=0.0, posinf=clamp_high, neginf=clamp_low)
            return arr_clean.tolist()

        if is_binary:
            # Binary classification
            y_score = y_proba[:, 1]  # Positive class probabilities

            try:
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                # Precision-Recall curve
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

                return {
                    "classification_type": "binary",
                    "data_source": data_source,
                    "roc_curve": {
                        "fpr": to_finite_list(fpr),
                        "tpr": to_finite_list(tpr),
                        "thresholds": to_finite_list(thresholds),
                        "auc": float(auc_score),
                    },
                    "precision_recall_curve": {
                        "precision": to_finite_list(precision),
                        "recall": to_finite_list(recall),
                        "thresholds": to_finite_list(pr_thresholds),
                    },
                }

            except Exception as e:
                return {"error": f"ROC analysis failed: {str(e)}"}
        else:
            # Multiclass classification
            try:
                # One-vs-Rest ROC for each class
                per_class_curves = {}
                auc_scores = {}

                for i, class_label in enumerate(unique_classes):
                    # Create binary labels for current class vs rest
                    y_binary = (y_true == class_label).astype(int)
                    y_score_class = y_proba[:, i]

                    fpr, tpr, thresholds = roc_curve(y_binary, y_score_class)
                    auc = roc_auc_score(y_binary, y_score_class)

                    per_class_curves[str(class_label)] = {
                        "fpr": to_finite_list(fpr),
                        "tpr": to_finite_list(tpr),
                        "thresholds": to_finite_list(thresholds),
                        "auc": float(auc),
                    }
                    auc_scores[f"class_{class_label}"] = float(auc)

                # Compute macro and micro average AUC
                macro_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                micro_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="micro")

                # Compute macro and micro ROC curves
                from sklearn.preprocessing import label_binarize

                y_bin = label_binarize(y_true, classes=unique_classes)

                # Compute micro-average ROC curve and area
                fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba.ravel())

                # Compute macro-average ROC curve
                all_fpr = np.unique(np.concatenate([per_class_curves[str(cls)]["fpr"] for cls in unique_classes]))
                mean_tpr = np.zeros_like(all_fpr)
                for cls in unique_classes:
                    mean_tpr += np.interp(all_fpr, per_class_curves[str(cls)]["fpr"], per_class_curves[str(cls)]["tpr"])
                mean_tpr /= n_classes

                return {
                    "classification_type": "multiclass",
                    "data_source": data_source,
                    "classes": unique_classes.tolist(),
                    "roc_curve": {
                        "per_class": per_class_curves,
                        "macro": {
                            "fpr": to_finite_list(all_fpr),
                            "tpr": to_finite_list(mean_tpr),
                            "auc": float(macro_auc),
                        },
                        "micro": {
                            "fpr": to_finite_list(fpr_micro),
                            "tpr": to_finite_list(tpr_micro),
                            "auc": float(micro_auc),
                        },
                    },
                    "auc_scores": auc_scores,
                    "macro_auc": float(macro_auc),
                    "micro_auc": float(micro_auc),
                }

            except Exception as e:
                return {"error": f"Multiclass ROC analysis failed: {str(e)}"}

    def threshold_analysis(self, num_thresholds: int = 50) -> Dict[str, Any]:
        """Perform threshold analysis for binary classification."""
        self.base._is_ready()

        # Check if this is actually a classification model
        if self.base._is_regression_model():
            return {
                "error": "Threshold analysis is only available for classification models. This appears to be a regression model."
            }

        # Use test data if available, otherwise fall back to training data
        if self.base.X_test is not None and self.base.y_test is not None:
            X_eval, y_eval = self.base.X_test, self.base.y_test
            data_source = "test"
        else:
            X_eval, y_eval = self.base.X_train, self.base.y_train
            data_source = "train"

        y_true = y_eval.values
        y_proba = self.base.safe_predict_proba(X_eval)

        if y_proba is None:
            return {"error": "Model does not support probability predictions required for threshold analysis."}

        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) == 2

        if not is_binary:
            return {"error": "Threshold analysis is only available for binary classification."}

        # Binary classification threshold analysis
        try:
            y_score = y_proba[:, 1]  # Positive class probabilities

            # Find optimal threshold based on Youden's J statistic
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]

        except Exception as e:
            return {"error": f"Failed to find optimal threshold: {str(e)}"}

        thresholds = np.linspace(0.0, 1.0, num=num_thresholds)
        results = []

        for thr in thresholds:
            y_pred_thresh = (y_score >= thr).astype(int)

            # Calculate metrics for this threshold
            tn = np.sum((y_true == 0) & (y_pred_thresh == 0))
            fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
            fn = np.sum((y_true == 1) & (y_pred_thresh == 0))
            tp = np.sum((y_true == 1) & (y_pred_thresh == 1))

            # Avoid division by zero
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

            results.append(
                {
                    "threshold": float(thr),
                    "precision": float(precision),
                    "recall": float(recall),
                    "specificity": float(specificity),
                    "f1_score": float(f1),
                    "accuracy": float(accuracy),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                }
            )

        # Optimal metrics at optimal threshold
        optimal_metrics = None
        for result in results:
            if abs(result["threshold"] - optimal_threshold) < 0.02:  # Close enough
                optimal_metrics = result
                break

        if optimal_metrics is None:
            # Calculate optimal metrics directly
            y_pred_optimal = (y_score >= optimal_threshold).astype(int)
            tn = np.sum((y_true == 0) & (y_pred_optimal == 0))
            fp = np.sum((y_true == 0) & (y_pred_optimal == 1))
            fn = np.sum((y_true == 1) & (y_pred_optimal == 0))
            tp = np.sum((y_true == 1) & (y_pred_optimal == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

            optimal_metrics = {
                "threshold": float(optimal_threshold),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "f1_score": float(f1),
                "accuracy": float(accuracy),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            }

        # Include ROC analysis data if available
        roc_payload = self.roc_analysis()
        if "error" not in roc_payload:
            roc_data = roc_payload.get("roc_curve", {})
        else:
            roc_data = {}

        return {
            "threshold_metrics": results,
            "optimal_metrics": optimal_metrics,
            "data_source": data_source,
            "classification_type": "binary",
            "roc_curve": roc_data,
        }
