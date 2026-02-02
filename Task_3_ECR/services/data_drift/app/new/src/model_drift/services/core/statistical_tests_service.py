"""
Statistical Tests Service - Implements all 6 statistical significance tests
Following research specifications for model drift detection
"""

import warnings
from typing import Any, Dict

import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


class StatisticalTestsService:
    """Service for conducting statistical significance tests between model performances"""

    def __init__(self):
        self.alpha = 0.05  # Default significance level

    def mcnemar_test(self, y_true: np.ndarray, pred_ref: np.ndarray, pred_curr: np.ndarray) -> Dict[str, Any]:
        """
        McNemar's Test for comparing two classification models
        Used when comparing binary classification accuracy

        Args:
            y_true: True labels
            pred_ref: Reference model predictions (binary)
            pred_curr: Current model predictions (binary)

        Returns:
            Dictionary with test statistic, p-value, and interpretation
        """
        try:
            # Create contingency table
            # a: both correct, b: ref correct curr wrong, c: ref wrong curr correct, d: both wrong
            correct_ref = pred_ref == y_true
            correct_curr = pred_curr == y_true

            a = np.sum(correct_ref & correct_curr)  # Both correct
            b = np.sum(correct_ref & ~correct_curr)  # Ref correct, Curr wrong
            c = np.sum(~correct_ref & correct_curr)  # Ref wrong, Curr correct
            d = np.sum(~correct_ref & ~correct_curr)  # Both wrong

            # McNemar's statistic with continuity correction
            if b + c == 0:
                statistic = 0
                p_value = 1.0
            else:
                statistic = (abs(b - c) - 1) ** 2 / (b + c)
                p_value = 1 - stats.chi2.cdf(statistic, df=1)

            # Effect size (odds ratio)
            odds_ratio = b / c if c > 0 else np.inf if b > 0 else 1.0

            return {
                "test_name": "McNemar's Test",
                "test_statistic": float(statistic),  # Add test_statistic for consistency
                "statistic": float(statistic),  # Keep for backward compatibility
                "p_value": float(p_value),
                "critical_value": stats.chi2.ppf(1 - self.alpha, df=1),
                "significant": p_value < self.alpha,
                "interpretation": (
                    "Current model significantly different" if p_value < self.alpha else "No significant difference"
                ),
                "method_description": "Tests significance of differences in binary classification accuracy using contingency table analysis",
                "effect_size": {
                    "odds_ratio": float(odds_ratio),
                    "contingency_table": {"a": int(a), "b": int(b), "c": int(c), "d": int(d)},
                },
                "alpha": self.alpha,
            }
        except Exception as e:
            return {
                "test_name": "McNemar's Test",
                "test_statistic": None,
                "statistic": None,
                "p_value": None,
                "significant": False,
                "interpretation": "McNemar's test failed due to error",
                "method_description": "Tests significance of differences in binary classification accuracy using contingency table analysis",
                "error": f"McNemar's test failed: {str(e)}",
            }

    def delong_test(
        self, y_true: np.ndarray, pred_ref_proba: np.ndarray, pred_curr_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        DeLong's Test for comparing ROC curves (AUC comparison)
        Used for comparing probabilistic classification models

        Args:
            y_true: True binary labels
            pred_ref_proba: Reference model prediction probabilities
            pred_curr_proba: Current model prediction probabilities

        Returns:
            Dictionary with test results
        """
        try:
            # Handle binary classification probabilities
            if pred_ref_proba.ndim == 2 and pred_ref_proba.shape[1] == 2:
                pred_ref_proba = pred_ref_proba[:, 1]  # Use positive class probabilities
            if pred_curr_proba.ndim == 2 and pred_curr_proba.shape[1] == 2:
                pred_curr_proba = pred_curr_proba[:, 1]  # Use positive class probabilities

            # Calculate AUCs
            auc_ref = roc_auc_score(y_true, pred_ref_proba)
            auc_curr = roc_auc_score(y_true, pred_curr_proba)

            # DeLong test implementation (simplified)
            n = len(y_true)
            n_pos = np.sum(y_true)
            n_neg = n - n_pos

            # Calculate structural components for variance estimation
            def structural_components(predictions, labels):
                order = np.argsort(predictions)[::-1]
                predictions_sorted = predictions[order]
                labels_sorted = labels[order]

                # Calculate V10 and V01 for DeLong variance
                V10 = np.zeros(n_pos)
                V01 = np.zeros(n_neg)

                pos_indices = np.where(labels_sorted == 1)[0]
                neg_indices = np.where(labels_sorted == 0)[0]

                for i, pos_idx in enumerate(pos_indices):
                    V10[i] = np.mean(predictions_sorted[pos_idx] > predictions_sorted[neg_indices]) + 0.5 * np.mean(
                        predictions_sorted[pos_idx] == predictions_sorted[neg_indices]
                    )

                for i, neg_idx in enumerate(neg_indices):
                    V01[i] = np.mean(predictions_sorted[neg_idx] < predictions_sorted[pos_indices]) + 0.5 * np.mean(
                        predictions_sorted[neg_idx] == predictions_sorted[pos_indices]
                    )

                return V10, V01

            # Calculate structural components for both models
            V10_ref, V01_ref = structural_components(pred_ref_proba, y_true)
            V10_curr, V01_curr = structural_components(pred_curr_proba, y_true)

            # Calculate covariance
            cov = np.cov(V10_ref, V10_curr)[0, 1] / n_pos + np.cov(V01_ref, V01_curr)[0, 1] / n_neg

            # Variance of AUC difference
            var_ref = np.var(V10_ref) / n_pos + np.var(V01_ref) / n_neg
            var_curr = np.var(V10_curr) / n_pos + np.var(V01_curr) / n_neg
            var_diff = var_ref + var_curr - 2 * cov

            # Test statistic
            if var_diff <= 0:
                z_score = 0
                p_value = 1.0
            else:
                z_score = (auc_ref - auc_curr) / np.sqrt(var_diff)
                p_value = 2 * (1 - norm.cdf(abs(z_score)))

            return {
                "test_name": "DeLong's Test",
                "auc_reference": float(auc_ref),
                "auc_current": float(auc_curr),
                "auc_difference": float(auc_ref - auc_curr),
                "z_score": float(z_score),
                "p_value": float(p_value),
                "test_statistic": float(z_score),  # Add test statistic
                "significant": p_value < self.alpha,
                "interpretation": (
                    "Significant AUC difference" if p_value < self.alpha else "No significant AUC difference"
                ),
                "variance_difference": float(var_diff),
                "alpha": self.alpha,
            }
        except Exception as e:
            return {
                "test_name": "DeLong's Test",
                "auc_reference": None,
                "auc_current": None,
                "auc_difference": None,
                "z_score": None,
                "p_value": None,
                "test_statistic": None,
                "significant": False,
                "interpretation": "DeLong's test failed due to error",
                "variance_difference": None,
                "alpha": self.alpha,
                "error": f"DeLong's test failed: {str(e)}",
            }

    def five_two_cv_test(self, X: np.ndarray, y: np.ndarray, model_ref, model_curr) -> Dict[str, Any]:
        """
        5x2 Cross-Validation F-Test for comparing two models
        Robust method that accounts for training set variance

        Args:
            X: Feature matrix
            y: Target vector
            model_ref: Reference model (fitted sklearn model)
            model_curr: Current model (fitted sklearn model)

        Returns:
            Dictionary with test results
        """
        try:
            differences = []
            variances = []

            # Perform 5 iterations of 2-fold CV
            for i in range(5):
                kf = KFold(n_splits=2, shuffle=True, random_state=i)
                fold_diffs = []

                for train_idx, test_idx in kf.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Train both models
                    model_ref_cv = type(model_ref)(**model_ref.get_params())
                    model_curr_cv = type(model_curr)(**model_curr.get_params())

                    model_ref_cv.fit(X_train, y_train)
                    model_curr_cv.fit(X_train, y_train)

                    # Calculate accuracies
                    acc_ref = accuracy_score(y_test, model_ref_cv.predict(X_test))
                    acc_curr = accuracy_score(y_test, model_curr_cv.predict(X_test))

                    fold_diffs.append(acc_ref - acc_curr)

                # Calculate statistics for this iteration
                diff_mean = np.mean(fold_diffs)
                diff_var = np.sum([(d - diff_mean) ** 2 for d in fold_diffs])

                differences.append(diff_mean)
                variances.append(diff_var)

            # Calculate F-statistic
            numerator = np.sum([d**2 for d in differences])
            denominator = 2 * np.sum(variances)

            if denominator == 0:
                f_statistic = 0
                p_value = 1.0
            else:
                f_statistic = numerator / denominator
                p_value = 1 - stats.f.cdf(f_statistic, dfn=5, dfd=5)

            return {
                "test_name": "5x2 Cross-Validation F-Test",
                "f_statistic": float(f_statistic),
                "test_statistic": float(f_statistic),  # Add test statistic
                "p_value": float(p_value),
                "degrees_freedom": {"numerator": 5, "denominator": 5},
                "significant": p_value < self.alpha,
                "interpretation": (
                    "Significant performance difference" if p_value < self.alpha else "No significant difference"
                ),
                "mean_difference": float(np.mean(differences)),
                "differences": [float(d) for d in differences],
                "alpha": self.alpha,
            }
        except Exception as e:
            return {
                "test_name": "5x2 Cross-Validation F-Test",
                "f_statistic": None,
                "test_statistic": None,
                "p_value": None,
                "degrees_freedom": {"numerator": 5, "denominator": 5},
                "significant": False,
                "interpretation": "5x2 CV test failed due to error",
                "mean_difference": None,
                "differences": None,
                "alpha": self.alpha,
                "error": f"5x2 CV test failed: {str(e)}",
            }

    def bootstrap_test(
        self,
        y_true: np.ndarray,
        pred_ref: np.ndarray,
        pred_curr: np.ndarray,
        n_bootstrap: int = 1000,
        metric_func=accuracy_score,
    ) -> Dict[str, Any]:
        """
        Bootstrap Confidence Interval Test
        Non-parametric method for comparing model performances

        Args:
            y_true: True labels
            pred_ref: Reference model predictions
            pred_curr: Current model predictions
            n_bootstrap: Number of bootstrap samples
            metric_func: Metric function to use (default: accuracy_score)

        Returns:
            Dictionary with test results
        """
        try:
            # Input validation
            if len(y_true) != len(pred_ref) or len(y_true) != len(pred_curr):
                return {
                    "test_name": "Bootstrap Test",
                    "mean_difference": None,
                    "std_difference": None,
                    "confidence_interval": {"lower": None, "upper": None, "confidence_level": 1 - self.alpha},
                    "p_value": None,
                    "significant": False,
                    "interpretation": "Error: Input arrays have different lengths",
                    "method_description": "Non-parametric test using bootstrap resampling to estimate confidence intervals",
                    "error": "Input arrays have different lengths",
                }

            n_samples = len(y_true)
            if n_samples < 2:
                return {
                    "test_name": "Bootstrap Test",
                    "mean_difference": None,
                    "std_difference": None,
                    "confidence_interval": {"lower": None, "upper": None, "confidence_level": 1 - self.alpha},
                    "p_value": None,
                    "significant": False,
                    "interpretation": "Insufficient data for bootstrap testing (need at least 2 observations)",
                    "method_description": "Non-parametric test using bootstrap resampling to estimate confidence intervals",
                }

            differences = []

            # Bootstrap sampling
            for _ in range(n_bootstrap):
                try:
                    # Sample with replacement
                    indices = np.random.choice(n_samples, size=n_samples, replace=True)

                    y_boot = y_true[indices]
                    pred_ref_boot = pred_ref[indices]
                    pred_curr_boot = pred_curr[indices]

                    # Calculate metric difference
                    metric_ref = metric_func(y_boot, pred_ref_boot)
                    metric_curr = metric_func(y_boot, pred_curr_boot)
                    differences.append(metric_ref - metric_curr)
                except Exception as e:
                    # Skip this bootstrap iteration if it fails
                    continue

            if len(differences) < 100:  # Need sufficient bootstrap samples
                return {
                    "test_name": "Bootstrap Test",
                    "mean_difference": None,
                    "std_difference": None,
                    "confidence_interval": {"lower": None, "upper": None, "confidence_level": 1 - self.alpha},
                    "p_value": None,
                    "significant": False,
                    "interpretation": "Bootstrap sampling failed - insufficient valid samples",
                    "method_description": "Non-parametric test using bootstrap resampling to estimate confidence intervals",
                }

            differences = np.array(differences)

            # Calculate confidence interval
            confidence_level = 1 - self.alpha
            lower_percentile = (1 - confidence_level) / 2 * 100
            upper_percentile = (1 + confidence_level) / 2 * 100

            ci_lower = np.percentile(differences, lower_percentile)
            ci_upper = np.percentile(differences, upper_percentile)

            # Statistical significance test
            # If CI contains 0, no significant difference
            significant = not (ci_lower <= 0 <= ci_upper)

            # Calculate p-value (proportion of bootstrap samples with difference <= 0)
            if np.mean(differences) >= 0:
                p_value = 2 * np.mean(differences <= 0)
            else:
                p_value = 2 * np.mean(differences >= 0)
            p_value = min(p_value, 1.0)

            # Calculate actual difference as test statistic
            original_ref_metric = metric_func(y_true, pred_ref)
            original_curr_metric = metric_func(y_true, pred_curr)
            test_statistic = original_ref_metric - original_curr_metric

            return {
                "test_name": "Bootstrap Test",
                "mean_difference": float(np.mean(differences)),
                "std_difference": float(np.std(differences)),
                "confidence_interval": {
                    "lower": float(ci_lower),
                    "upper": float(ci_upper),
                    "confidence_level": confidence_level,
                },
                "p_value": float(p_value),
                "test_statistic": float(test_statistic),  # Add test statistic
                "significant": significant,
                "interpretation": "Significant performance difference" if significant else "No significant difference",
                "method_description": "Non-parametric test using bootstrap resampling to estimate confidence intervals",
                "n_bootstrap": len(differences),
                "alpha": self.alpha,
            }

        except Exception as e:
            return {
                "test_name": "Bootstrap Test",
                "mean_difference": None,
                "std_difference": None,
                "confidence_interval": {"lower": None, "upper": None, "confidence_level": 1 - self.alpha},
                "p_value": None,
                "test_statistic": None,
                "significant": False,
                "interpretation": "Bootstrap test failed due to error",
                "method_description": "Non-parametric test using bootstrap resampling to estimate confidence intervals",
                "error": f"Bootstrap test failed: {str(e)}",
            }

    def diebold_mariano_test(self, errors_ref: np.ndarray, errors_curr: np.ndarray) -> Dict[str, Any]:
        """
        Diebold-Mariano Test for comparing forecast accuracy
        Primarily used for regression models

        Args:
            errors_ref: Forecast errors from reference model
            errors_curr: Forecast errors from current model

        Returns:
            Dictionary with test results
        """
        try:
            # Calculate loss differential
            loss_diff = errors_ref**2 - errors_curr**2

            # Mean loss differential
            d_bar = np.mean(loss_diff)

            # Variance of loss differential (assuming no autocorrelation)
            n = len(loss_diff)
            gamma_0 = np.var(loss_diff, ddof=1)

            # Test statistic
            if gamma_0 == 0:
                dm_statistic = 0
                p_value = 1.0
            else:
                dm_statistic = d_bar / np.sqrt(gamma_0 / n)
                p_value = 2 * (1 - norm.cdf(abs(dm_statistic)))

            return {
                "test_name": "Diebold-Mariano Test",
                "dm_statistic": float(dm_statistic),
                "test_statistic": float(dm_statistic),  # Add test statistic
                "p_value": float(p_value),
                "mean_loss_differential": float(d_bar),
                "variance_loss_differential": float(gamma_0),
                "significant": p_value < self.alpha,
                "interpretation": (
                    "Significant forecast accuracy difference" if p_value < self.alpha else "No significant difference"
                ),
                "alpha": self.alpha,
            }
        except Exception as e:
            return {
                "test_name": "Diebold-Mariano Test",
                "dm_statistic": None,
                "test_statistic": None,
                "p_value": None,
                "mean_loss_differential": None,
                "variance_loss_differential": None,
                "significant": False,
                "interpretation": "Diebold-Mariano test failed due to error",
                "alpha": self.alpha,
                "error": f"Diebold-Mariano test failed: {str(e)}",
            }

    def paired_t_test(self, metric_ref: np.ndarray, metric_curr: np.ndarray) -> Dict[str, Any]:
        """
        Paired t-Test for comparing model performance metrics
        Classical statistical test for paired observations

        Args:
            metric_ref: Performance metrics from reference model
            metric_curr: Performance metrics from current model

        Returns:
            Dictionary with test results
        """
        try:
            # Input validation
            if len(metric_ref) != len(metric_curr):
                return {
                    "test_name": "Paired t-Test",
                    "t_statistic": None,
                    "p_value": None,
                    "degrees_freedom": 0,
                    "mean_difference": None,
                    "std_difference": None,
                    "confidence_interval": {"lower": None, "upper": None},
                    "significant": False,
                    "interpretation": "Error: Arrays have different lengths",
                    "method_description": "Classical parametric test for comparing paired observations of model performance",
                    "error": "Input arrays have different lengths",
                }

            # Calculate differences
            differences = metric_ref - metric_curr

            # Handle edge cases
            if len(differences) <= 1:
                return {
                    "test_name": "Paired t-Test",
                    "t_statistic": None,
                    "p_value": None,
                    "degrees_freedom": len(differences) - 1 if len(differences) > 0 else 0,
                    "mean_difference": float(np.mean(differences)) if len(differences) > 0 else None,
                    "std_difference": None,
                    "confidence_interval": {"lower": None, "upper": None},
                    "significant": False,
                    "interpretation": "Insufficient data for significance testing (need at least 2 observations)",
                    "method_description": "Classical parametric test for comparing paired observations of model performance",
                }

            # Calculate standard deviation with proper handling of zero variance
            std_diff = np.std(differences, ddof=1)
            mean_diff = np.mean(differences)

            if std_diff == 0 or np.isclose(std_diff, 0, atol=1e-10):
                return {
                    "test_name": "Paired t-Test",
                    "t_statistic": None,
                    "p_value": None,
                    "degrees_freedom": len(differences) - 1,
                    "mean_difference": float(mean_diff),
                    "std_difference": 0.0,
                    "confidence_interval": {"lower": None, "upper": None},
                    "significant": False,
                    "interpretation": "No variation in differences - cannot perform t-test",
                    "method_description": "Classical parametric test for comparing paired observations of model performance",
                }

            # Perform paired t-test
            t_statistic, p_value = stats.ttest_1samp(differences, 0)

            # Degrees of freedom
            df = len(differences) - 1

            # Effect size (Cohen's d)
            cohen_d = mean_diff / std_diff

            # Confidence interval for mean difference
            sem = stats.sem(differences)
            ci_lower, ci_upper = stats.t.interval(1 - self.alpha, df, loc=mean_diff, scale=sem)

            return {
                "test_name": "Paired t-Test",
                "t_statistic": float(t_statistic),
                "p_value": float(p_value),
                "degrees_freedom": int(df),
                "mean_difference": float(mean_diff),
                "std_difference": float(std_diff),
                "confidence_interval": {"lower": float(ci_lower), "upper": float(ci_upper)},
                "effect_size": {"cohen_d": float(cohen_d)},
                "significant": p_value < self.alpha,
                "interpretation": (
                    "Significant performance difference" if p_value < self.alpha else "No significant difference"
                ),
                "method_description": "Classical parametric test for comparing paired observations of model performance",
                "alpha": self.alpha,
            }

        except Exception as e:
            return {
                "test_name": "Paired t-Test",
                "t_statistic": None,
                "p_value": None,
                "degrees_freedom": 0,
                "mean_difference": None,
                "std_difference": None,
                "confidence_interval": {"lower": None, "upper": None},
                "significant": False,
                "interpretation": "Test failed due to error",
                "method_description": "Classical parametric test for comparing paired observations of model performance",
                "error": f"Paired t-test failed: {str(e)}",
            }

    def run_all_tests(
        self,
        y_true: np.ndarray,
        pred_ref: np.ndarray,
        pred_curr: np.ndarray,
        pred_ref_proba: np.ndarray = None,
        pred_curr_proba: np.ndarray = None,
        X: np.ndarray = None,
        model_ref=None,
        model_curr=None,
    ) -> Dict[str, Any]:
        """
        Run all applicable statistical tests

        Args:
            y_true: True labels
            pred_ref: Reference model predictions
            pred_curr: Current model predictions
            pred_ref_proba: Reference model probabilities (optional)
            pred_curr_proba: Current model probabilities (optional)
            X: Feature matrix (optional, for 5x2 CV test)
            model_ref: Reference model object (optional, for 5x2 CV test)
            model_curr: Current model object (optional, for 5x2 CV test)

        Returns:
            Dictionary with all test results
        """
        results = {"summary": {"total_tests": 0, "significant_tests": 0, "alpha": self.alpha}, "tests": {}}

        # McNemar's Test (for classification)
        mcnemar_result = self.mcnemar_test(y_true, pred_ref, pred_curr)
        results["tests"]["mcnemar"] = mcnemar_result
        if "error" not in mcnemar_result:
            results["summary"]["total_tests"] += 1
            if mcnemar_result["significant"]:
                results["summary"]["significant_tests"] += 1

        # DeLong's Test (if probabilities available)
        if pred_ref_proba is not None and pred_curr_proba is not None:
            delong_result = self.delong_test(y_true, pred_ref_proba, pred_curr_proba)
            results["tests"]["delong"] = delong_result
            if "error" not in delong_result:
                results["summary"]["total_tests"] += 1
                if delong_result["significant"]:
                    results["summary"]["significant_tests"] += 1

        # Bootstrap Test
        bootstrap_result = self.bootstrap_test(y_true, pred_ref, pred_curr)
        results["tests"]["bootstrap"] = bootstrap_result
        if "error" not in bootstrap_result:
            results["summary"]["total_tests"] += 1
            if bootstrap_result["significant"]:
                results["summary"]["significant_tests"] += 1

        # 5x2 CV Test (if models and features available)
        if X is not None and model_ref is not None and model_curr is not None:
            cv_result = self.five_two_cv_test(X, y_true, model_ref, model_curr)
            results["tests"]["five_two_cv"] = cv_result
            if "error" not in cv_result:
                results["summary"]["total_tests"] += 1
                if cv_result["significant"]:
                    results["summary"]["significant_tests"] += 1

        # Paired t-Test
        ref_accuracy = np.array([accuracy_score(y_true, pred_ref)])
        curr_accuracy = np.array([accuracy_score(y_true, pred_curr)])
        ttest_result = self.paired_t_test(ref_accuracy, curr_accuracy)
        results["tests"]["paired_t_test"] = ttest_result
        if "error" not in ttest_result:
            results["summary"]["total_tests"] += 1
            if ttest_result["significant"]:
                results["summary"]["significant_tests"] += 1

        # Overall conclusion
        results["summary"]["conclusion"] = (
            "Significant model difference detected"
            if results["summary"]["significant_tests"] > 0
            else "No significant model difference detected"
        )

        return results


# Service instance
statistical_tests_service = StatisticalTestsService()
