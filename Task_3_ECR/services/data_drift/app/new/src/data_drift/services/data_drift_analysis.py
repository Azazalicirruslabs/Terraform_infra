"""
Comprehensive Data Drift Analysis Service
Combines all 4 data drift analysis tabs with proper error handling and data management
"""

import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, entropy, ks_2samp

warnings.filterwarnings("ignore")


class DataDriftAnalysis:
    """Comprehensive data drift analysis service"""

    def __init__(self):
        pass

    async def analyze_dashboard(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Dashboard Analysis - Overview of all features drift

        Args:
            reference_df: Reference dataset
            current_df: Current dataset

        Returns:
            Dictionary with dashboard analysis results
        """
        try:
            drifted_features = []
            feature_analysis_list = []

            # Ensure same columns in both datasets
            common_columns = list(set(reference_df.columns) & set(current_df.columns))
            if not common_columns:
                return {"error": "No common columns found between reference and current datasets"}

            # Define bins for numeric columns (using same bins for ref and curr)
            numeric_bins = {}
            for col in common_columns:
                if reference_df[col].dtype in ["int64", "float64"] and current_df[col].dtype in ["int64", "float64"]:
                    # Use reference data to define bins, handle edge cases
                    ref_values = reference_df[col].dropna()
                    if len(ref_values) > 0:
                        try:
                            bins = np.histogram_bin_edges(ref_values, bins="auto")
                            # Ensure we have at least 2 bins
                            if len(bins) < 3:
                                bins = np.linspace(ref_values.min(), ref_values.max(), 11)
                            numeric_bins[col] = bins
                        except:
                            # Fallback to simple binning
                            numeric_bins[col] = np.linspace(ref_values.min(), ref_values.max(), 11)

            for col in common_columns:
                try:
                    if col in numeric_bins:
                        # Numeric feature analysis
                        ref_values = reference_df[col].dropna()
                        curr_values = current_df[col].dropna()

                        if len(ref_values) == 0 or len(curr_values) == 0:
                            continue

                        # KS test for drift
                        stat, p_value = ks_2samp(ref_values, curr_values)
                        drift_status = "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"
                        drift_score = abs(stat) * 5

                        # Compute histograms for ref and current with same bins
                        bins = numeric_bins[col]
                        ref_hist, _ = np.histogram(ref_values, bins=bins, density=True)
                        curr_hist, _ = np.histogram(curr_values, bins=bins, density=True)

                        # Safe KL divergence calculation
                        kl_divergence = self._safe_kl_divergence(ref_hist, curr_hist)

                        # Convert histograms to list of frequencies in percentages
                        distribution_ref = (ref_hist * 100).tolist()
                        distribution_current = (curr_hist * 100).tolist()

                        # Create labels for bins
                        bin_labels = []
                        for i in range(len(bins) - 1):
                            try:
                                label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                                bin_labels.append(label)
                            except:
                                bin_labels.append(f"Bin_{i+1}")

                        feature_analysis_list.append(
                            {
                                "feature": col,
                                "drift_score": round(float(drift_score), 3),
                                "kl_divergence": round(float(kl_divergence), 3),
                                "status": drift_status,
                                "p_value": float(p_value),
                                "distribution_ref": distribution_ref,
                                "distribution_current": distribution_current,
                                "bin_labels": bin_labels,
                                "feature_type": "numerical",
                            }
                        )

                    else:
                        # Categorical feature analysis
                        ref_counts = reference_df[col].value_counts()
                        curr_counts = current_df[col].value_counts()

                        # Create contingency table for chi-square test
                        all_categories = list(set(ref_counts.index) | set(curr_counts.index))
                        ref_vals = [ref_counts.get(cat, 0) for cat in all_categories]
                        curr_vals = [curr_counts.get(cat, 0) for cat in all_categories]

                        # Only perform chi-square if we have valid data
                        if sum(ref_vals) > 0 and sum(curr_vals) > 0 and len(all_categories) > 1:
                            try:
                                # Create 2x categories contingency table
                                contingency_table = [ref_vals, curr_vals]
                                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                                drift_status = "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"
                                drift_score = min(chi2 / 10, 10)  # Cap drift score at 10
                            except:
                                chi2, p_value, drift_status, drift_score = 0, 1.0, "low", 0
                        else:
                            chi2, p_value, drift_status, drift_score = 0, 1.0, "low", 0

                        distribution_ref = ref_counts.to_dict()
                        distribution_current = curr_counts.to_dict()

                        feature_analysis_list.append(
                            {
                                "feature": col,
                                "drift_score": round(float(drift_score), 3),
                                "status": drift_status,
                                "p_value": float(p_value),
                                "distribution_ref": distribution_ref,
                                "distribution_current": distribution_current,
                                "feature_type": "categorical",
                            }
                        )

                    if drift_status in ["high", "medium"]:
                        drifted_features.append(col)

                except Exception as e:
                    # Log individual feature error but continue processing
                    print(f"Error processing feature {col}: {str(e)}")
                    continue

            # Calculate overall metrics
            total_features = len(feature_analysis_list)
            if total_features == 0:
                return {"error": "No features could be analyzed"}

            high_drift_features = sum(1 for f in feature_analysis_list if f.get("status") == "high")
            medium_drift_features = sum(1 for f in feature_analysis_list if f.get("status") == "medium")
            overall_drift_score = sum(f["drift_score"] for f in feature_analysis_list) / total_features

            overall_status = "high" if overall_drift_score > 3 else "medium" if overall_drift_score > 1 else "low"

            top_features = ", ".join(drifted_features[:3]) if drifted_features else "no significant features"
            executive_summary = (
                f"Analysis shows {overall_status}-level drift primarily driven by {top_features}. "
                f"Found {high_drift_features} high-drift and {medium_drift_features} medium-drift features. "
                "Model performance monitoring and potential retraining should be considered."
            )

            data_quality_score = float(reference_df.notnull().mean().mean())

            return {
                "status": "success",
                "data": {
                    "high_drift_features": high_drift_features,
                    "medium_drift_features": medium_drift_features,
                    "data_quality_score": round(data_quality_score, 3),
                    "total_features": total_features,
                    "overall_drift_score": round(overall_drift_score, 2),
                    "executive_summary": executive_summary,
                    "overall_status": overall_status,
                    "analysis_timestamp": datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M:%S"),
                    "feature_analysis": feature_analysis_list,
                    "recommendations": self._generate_recommendations(
                        overall_status, high_drift_features, medium_drift_features
                    ),
                },
            }

        except Exception as e:
            return {"error": f"Dashboard analysis failed: {str(e)}"}

    async def analyze_class_imbalance(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame, target_column: str = None
    ) -> Dict[str, Any]:
        """
        Class Imbalance Analysis - Analyzes class distribution changes

        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            target_column: Name of target column (if None, uses last column)

        Returns:
            Dictionary with class imbalance analysis results
        """
        try:
            # Determine target column
            if target_column is None:
                if len(reference_df.columns) > 0:
                    target_column = reference_df.columns[-1]
                else:
                    return {"error": "No columns found in reference dataset"}

            if target_column not in reference_df.columns:
                return {"error": f"Target column '{target_column}' not found in reference dataset"}
            if target_column not in current_df.columns:
                return {"error": f"Target column '{target_column}' not found in current dataset"}

            # Class counts and basic stats
            class_counts_ref = reference_df[target_column].value_counts().to_dict()
            class_counts_curr = current_df[target_column].value_counts().to_dict()

            total_ref = len(reference_df)
            total_curr = len(current_df)

            if total_ref == 0 or total_curr == 0:
                return {"error": "Empty dataset(s) provided"}

            class_percent_ref = {k: round(v / total_ref * 100, 2) for k, v in class_counts_ref.items()}
            class_percent_curr = {k: round(v / total_curr * 100, 2) for k, v in class_counts_curr.items()}

            # Safe imbalance ratio calculation
            if len(class_counts_curr) == 0:
                return {"error": "No classes found in current dataset"}

            max_class_curr = max(class_counts_curr.values())
            min_class_curr = min(class_counts_curr.values())
            imbalance_ratio = float(max_class_curr / max(min_class_curr, 1))  # Avoid division by zero

            # Severity Level
            if imbalance_ratio < 2:
                severity = "Low"
            elif imbalance_ratio < 5:
                severity = "Medium"
            else:
                severity = "High"

            # Calculate imbalance metrics with error handling
            gini_ref = self._safe_gini(class_counts_ref)
            gini_curr = self._safe_gini(class_counts_curr)

            entropy_ref = self._safe_shannon_entropy(class_counts_ref)
            entropy_curr = self._safe_shannon_entropy(class_counts_curr)

            enc_ref = self._safe_effective_number_of_classes(class_counts_ref)
            enc_curr = self._safe_effective_number_of_classes(class_counts_curr)

            cbi_ref = self._safe_class_balance_index(class_counts_ref)
            cbi_curr = self._safe_class_balance_index(class_counts_curr)

            # Chi-square test for class distribution changes
            all_classes = list(set(class_counts_ref.keys()) | set(class_counts_curr.keys()))
            ref_vals = [class_counts_ref.get(c, 0) for c in all_classes]
            curr_vals = [class_counts_curr.get(c, 0) for c in all_classes]

            try:
                chi2_stat, p_value, dof, _ = chi2_contingency([ref_vals, curr_vals])
                chi2_stat = float(chi2_stat)
                p_value = float(p_value)
                dof = int(dof)
                chi_significance = (
                    "Highly Significant" if p_value < 0.01 else ("Significant" if p_value < 0.05 else "Not Significant")
                )
            except:
                chi2_stat, p_value, dof, chi_significance = 0.0, 1.0, 0, "Not Significant"

            # KS test for numeric features (excluding target)
            ks_results = {}
            numeric_cols = [
                col for col in reference_df.select_dtypes(include=[np.number]).columns if col != target_column
            ]

            for col in numeric_cols:
                try:
                    if col in current_df.columns:
                        ref_vals = reference_df[col].dropna()
                        curr_vals = current_df[col].dropna()
                        if len(ref_vals) > 0 and len(curr_vals) > 0:
                            ks_stat, ks_p = ks_2samp(ref_vals, curr_vals)
                            ks_results[col] = {
                                "ks_statistic": float(ks_stat),
                                "p_value": float(ks_p),
                                "interpretation": "Significant" if ks_p < 0.05 else "Not Significant",
                            }
                except:
                    continue

            return {
                "status": "success",
                "data": {
                    "target_column": target_column,
                    "overall_imbalance_score": round(imbalance_ratio, 2),
                    "severity_level": severity,
                    "total_samples": {"reference": total_ref, "current": total_curr},
                    "class_counts": {
                        "reference": {str(k): int(v) for k, v in class_counts_ref.items()},
                        "current": {str(k): int(v) for k, v in class_counts_curr.items()},
                    },
                    "class_percentages": {"reference": class_percent_ref, "current": class_percent_curr},
                    "imbalance_metrics": {
                        "imbalance_ratio": round(imbalance_ratio, 3),
                        "gini_coefficient": {"reference": round(gini_ref, 3), "current": round(gini_curr, 3)},
                        "shannon_entropy": {"reference": round(entropy_ref, 3), "current": round(entropy_curr, 3)},
                        "effective_number_of_classes": {"reference": round(enc_ref, 3), "current": round(enc_curr, 3)},
                        "class_balance_index": {"reference": round(cbi_ref, 3), "current": round(cbi_curr, 3)},
                        "chi_square_test": {
                            "statistic": round(chi2_stat, 3),
                            "p_value": round(p_value, 6),
                            "degrees_of_freedom": dof,
                            "interpretation": chi_significance,
                        },
                    },
                    "statistical_significance": {"ks_test": ks_results},
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "recommendations": self._generate_imbalance_recommendations(severity, imbalance_ratio),
                },
            }

        except Exception as e:
            return {"error": f"Class imbalance analysis failed: {str(e)}"}

    def _safe_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Safely calculate KL divergence with proper handling of zero values"""
        try:
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            p = np.array(p) + epsilon
            q = np.array(q) + epsilon

            # Normalize to ensure they sum to 1
            p = p / np.sum(p)
            q = q / np.sum(q)

            return float(entropy(p, q))
        except:
            return 0.0

    def _safe_gini(self, counts: Dict) -> float:
        """Safely calculate Gini coefficient"""
        try:
            if not counts or sum(counts.values()) == 0:
                return 0.0

            array = np.array(list(counts.values()))
            array = np.sort(array)
            n = array.size
            index = np.arange(1, n + 1)
            return float((np.sum((2 * index - n - 1) * array)) / (n * array.sum()))
        except:
            return 0.0

    def _safe_shannon_entropy(self, counts: Dict) -> float:
        """Safely calculate Shannon entropy"""
        try:
            if not counts or sum(counts.values()) == 0:
                return 0.0

            total = sum(counts.values())
            probs = np.array([v / total for v in counts.values()])
            probs = probs[probs > 0]  # Remove zero probabilities
            return float(-np.sum(probs * np.log2(probs)))
        except:
            return 0.0

    def _safe_effective_number_of_classes(self, counts: Dict) -> float:
        """Safely calculate effective number of classes"""
        try:
            if not counts or sum(counts.values()) == 0:
                return 0.0

            total = sum(counts.values())
            probs = np.array([v / total for v in counts.values()])
            return float(1.0 / np.sum(probs**2))
        except:
            return 0.0

    def _safe_class_balance_index(self, counts: Dict) -> float:
        """Safely calculate class balance index"""
        try:
            if not counts or sum(counts.values()) == 0:
                return 0.0

            total = sum(counts.values())
            probs = np.array([v / total for v in counts.values()])
            n_classes = len(counts)

            if n_classes <= 1:
                return 1.0

            # Avoid issues with zero probabilities in power calculation
            probs = probs[probs > 0]
            if len(probs) == 0:
                return 0.0

            return float((np.prod(probs**probs)) * n_classes)
        except:
            return 0.0

    def _generate_recommendations(self, status: str, high_drift: int, medium_drift: int) -> List[str]:
        """Generate recommendations based on drift analysis"""
        recommendations = []

        if status == "high":
            recommendations.extend(
                [
                    "Immediate investigation required - high drift detected",
                    "Consider model retraining within 1-2 weeks",
                    "Monitor model performance closely",
                    "Investigate root causes of data distribution changes",
                ]
            )
        elif status == "medium":
            recommendations.extend(
                [
                    "Moderate drift detected - monitor closely",
                    "Schedule model retraining within 1-2 months",
                    "Investigate features with medium/high drift",
                ]
            )
        else:
            recommendations.extend(
                ["Low drift detected - continue regular monitoring", "Maintain current model deployment schedule"]
            )

        if high_drift > 0:
            recommendations.append(f"Focus on {high_drift} high-drift features for immediate attention")
        if medium_drift > 0:
            recommendations.append(f"Monitor {medium_drift} medium-drift features for trend analysis")

        return recommendations

    def _generate_imbalance_recommendations(self, severity: str, ratio: float) -> List[str]:
        """Generate recommendations based on class imbalance analysis"""
        recommendations = []

        if severity == "High":
            recommendations.extend(
                [
                    f"High class imbalance detected (ratio: {ratio:.1f}:1)",
                    "Consider using stratified sampling for model training",
                    "Apply class balancing techniques (SMOTE, undersampling)",
                    "Use balanced metrics (F1-score, balanced accuracy) for evaluation",
                    "Monitor minority class performance closely",
                ]
            )
        elif severity == "Medium":
            recommendations.extend(
                [
                    f"Moderate class imbalance detected (ratio: {ratio:.1f}:1)",
                    "Consider weighted loss functions during training",
                    "Monitor class-specific performance metrics",
                    "Evaluate if current sampling strategy is adequate",
                ]
            )
        else:
            recommendations.extend(
                [
                    f"Low class imbalance (ratio: {ratio:.1f}:1)",
                    "Current class distribution is well-balanced",
                    "Continue monitoring for changes in class distribution",
                ]
            )

        return recommendations

    async def analyze_statistical_reports(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Statistical Reports Analysis - Detailed statistical analysis of feature drift

        Args:
            reference_df: Reference dataset
            current_df: Current dataset

        Returns:
            Dictionary with statistical analysis results
        """
        try:
            # Ensure same columns in both datasets
            common_columns = list(set(reference_df.columns) & set(current_df.columns))
            if not common_columns:
                return {"error": "No common columns found between reference and current datasets"}

            feature_analysis_list = []
            ks_tests = []
            chi_tests = []
            total_drift_score = 0

            for col in common_columns:
                try:
                    dtype = "numerical" if reference_df[col].dtype in ["int64", "float64"] else "categorical"
                    missing_ref = int(reference_df[col].isna().sum())
                    missing_curr = int(current_df[col].isna().sum())
                    drift_score = 0.0
                    kl_divergence = 0.0
                    psi_value = 0.0
                    ks_statistic = 0.0
                    p_value = 1.0

                    if dtype == "numerical":
                        ref_vals = reference_df[col].dropna()
                        curr_vals = current_df[col].dropna()

                        if len(ref_vals) == 0 or len(curr_vals) == 0:
                            continue

                        # Drift score based on standardized mean difference
                        ref_std = ref_vals.std()
                        if ref_std > 0:
                            drift_score = abs(ref_vals.mean() - curr_vals.mean()) / ref_std
                        else:
                            drift_score = 0.0

                        # Safe KL divergence calculation
                        try:
                            ref_hist, bins = np.histogram(ref_vals, bins=10, density=True)
                            curr_hist, _ = np.histogram(curr_vals, bins=bins, density=True)
                            kl_divergence = self._safe_kl_divergence(ref_hist, curr_hist)
                        except:
                            kl_divergence = 0.0

                        # PSI calculation
                        psi_value = self._calculate_psi(ref_vals, curr_vals)

                        # KS test
                        try:
                            ks_statistic, p_value = ks_2samp(ref_vals, curr_vals)
                            ks_tests.append(
                                {
                                    "feature": col,
                                    "ks_statistic": float(ks_statistic),
                                    "p_value": float(p_value),
                                    "result": "Significant" if p_value < 0.05 else "Not Significant",
                                }
                            )
                        except:
                            ks_statistic, p_value = 0.0, 1.0

                        feature_stats = {
                            "feature": col,
                            "data_type": dtype,
                            "ref_mean": float(ref_vals.mean()),
                            "ref_std": float(ref_vals.std()),
                            "ref_min": float(ref_vals.min()),
                            "ref_max": float(ref_vals.max()),
                            "curr_mean": float(curr_vals.mean()),
                            "curr_std": float(curr_vals.std()),
                            "curr_min": float(curr_vals.min()),
                            "curr_max": float(curr_vals.max()),
                            "missing_values_ref": missing_ref,
                            "missing_values_current": missing_curr,
                            "drift_score": float(drift_score),
                            "kl_divergence": float(kl_divergence),
                            "psi": float(psi_value),
                            "ks_statistic": float(ks_statistic),
                            "p_value": float(p_value),
                            "status": "low" if drift_score < 0.5 else "medium" if drift_score < 1.5 else "high",
                        }

                    else:
                        # Categorical feature analysis
                        ref_counts = reference_df[col].value_counts(normalize=True).to_dict()
                        curr_counts = current_df[col].value_counts(normalize=True).to_dict()
                        all_keys = set(ref_counts.keys()) | set(curr_counts.keys())

                        # Total variation distance as drift score
                        drift_score = sum(abs(ref_counts.get(k, 0) - curr_counts.get(k, 0)) for k in all_keys) / 2

                        # Chi-Square test for categorical variables
                        try:
                            categories = list(all_keys)
                            ref_vals = [reference_df[col].value_counts().get(k, 0) for k in categories]
                            curr_vals = [current_df[col].value_counts().get(k, 0) for k in categories]

                            if sum(ref_vals) > 0 and sum(curr_vals) > 0 and len(categories) > 1:
                                chi2_stat, chi_p, _, _ = chi2_contingency([ref_vals, curr_vals])
                                chi_tests.append(
                                    {
                                        "feature": col,
                                        "chi_square": float(chi2_stat),
                                        "p_value": float(chi_p),
                                        "result": "Significant" if chi_p < 0.05 else "Not Significant",
                                    }
                                )
                        except:
                            pass

                        feature_stats = {
                            "feature": col,
                            "data_type": dtype,
                            "ref_counts": {
                                str(k): float(reference_df[col].value_counts(normalize=True).get(k, 0))
                                for k in ref_counts.keys()
                            },
                            "curr_counts": {
                                str(k): float(current_df[col].value_counts(normalize=True).get(k, 0))
                                for k in curr_counts.keys()
                            },
                            "missing_values_ref": missing_ref,
                            "missing_values_current": missing_curr,
                            "drift_score": float(drift_score),
                            "kl_divergence": float(kl_divergence),
                            "psi": float(psi_value),
                            "ks_statistic": float(ks_statistic),
                            "p_value": float(p_value),
                            "status": "low" if drift_score < 0.1 else "medium" if drift_score < 0.3 else "high",
                        }

                    total_drift_score += drift_score
                    feature_analysis_list.append(feature_stats)

                except Exception as e:
                    print(f"Error analyzing feature {col}: {str(e)}")
                    continue

            if len(feature_analysis_list) == 0:
                return {"error": "No features could be analyzed"}

            overall_drift_score = total_drift_score / len(feature_analysis_list)
            overall_status = "low" if overall_drift_score < 0.5 else "medium" if overall_drift_score < 1.5 else "high"

            # Data quality score
            total_cells_curr = current_df.shape[0] * current_df.shape[1]
            missing_cells_curr = current_df.isna().sum().sum()
            data_quality_score = 1 - (missing_cells_curr / max(total_cells_curr, 1))

            # Dynamic executive summary
            count_high = sum(1 for f in feature_analysis_list if f["status"] == "high")
            count_medium = sum(1 for f in feature_analysis_list if f["status"] == "medium")
            count_low = sum(1 for f in feature_analysis_list if f["status"] == "low")

            executive_summary = (
                f"Analyzed {len(feature_analysis_list)} features: "
                f"{count_high} high drift, {count_medium} medium drift, {count_low} low drift. "
                f"Overall drift status: {overall_status.upper()}."
            )

            # Correlation analysis for numerical features
            correlation_analysis = []
            numerical_cols = [
                col
                for col in common_columns
                if reference_df[col].dtype in ["int64", "float64"] and current_df[col].dtype in ["int64", "float64"]
            ]

            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    try:
                        f1, f2 = numerical_cols[i], numerical_cols[j]
                        ref_corr = reference_df[f1].corr(reference_df[f2])
                        curr_corr = current_df[f1].corr(current_df[f2])

                        if pd.notna(ref_corr) and pd.notna(curr_corr):
                            correlation_analysis.append(
                                {
                                    "feature1": f1,
                                    "feature2": f2,
                                    "correlation": float(ref_corr),
                                    "drift_correlation": float(curr_corr),
                                    "correlation_change": float(curr_corr - ref_corr),
                                }
                            )
                    except:
                        continue

            return {
                "status": "success",
                "data": {
                    "feature_analysis": feature_analysis_list,
                    "ks_tests": ks_tests,
                    "chi_tests": chi_tests,
                    "correlation_analysis": correlation_analysis[:20],  # Limit to top 20 to avoid huge responses
                    "total_features": len(feature_analysis_list),
                    "overall_drift_score": round(overall_drift_score, 3),
                    "overall_status": overall_status,
                    "data_quality_score": round(data_quality_score, 3),
                    "executive_summary": executive_summary,
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "summary_stats": {
                        "high_drift_features": count_high,
                        "medium_drift_features": count_medium,
                        "low_drift_features": count_low,
                        "significant_ks_tests": len([t for t in ks_tests if t["result"] == "Significant"]),
                        "significant_chi_tests": len([t for t in chi_tests if t["result"] == "Significant"]),
                    },
                },
            }

        except Exception as e:
            return {"error": f"Statistical reports analysis failed: {str(e)}"}

    async def analyze_feature_deep_dive(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Feature Deep Dive Analysis - Detailed analysis of individual features

        Args:
            reference_df: Reference dataset
            current_df: Current dataset

        Returns:
            Dictionary with feature deep dive analysis results
        """
        try:
            # Ensure same columns in both datasets
            common_columns = list(set(reference_df.columns) & set(current_df.columns))
            if not common_columns:
                return {"error": "No common columns found between reference and current datasets"}

            feature_analysis_list = []

            for col in common_columns:
                try:
                    feature_type = "numerical" if reference_df[col].dtype in ["int64", "float64"] else "categorical"

                    if feature_type == "numerical":
                        ref_vals = reference_df[col].dropna()
                        curr_vals = current_df[col].dropna()

                        if len(ref_vals) == 0 or len(curr_vals) == 0:
                            continue

                        # Statistical tests
                        ks_statistic, p_value = ks_2samp(ref_vals, curr_vals)
                        drift_score = float(ks_statistic * 5)  # Scale for interpretability

                        status = "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"

                        # PSI calculation
                        psi_value = self._calculate_psi(ref_vals, curr_vals)

                        # Distribution analysis
                        try:
                            bins = np.histogram_bin_edges(ref_vals, bins="auto")
                            ref_hist, _ = np.histogram(ref_vals, bins=bins, density=True)
                            curr_hist, _ = np.histogram(curr_vals, bins=bins, density=True)

                            distribution_ref = {"histogram": ref_hist.tolist(), "bin_edges": bins.tolist()}
                            distribution_curr = {"histogram": curr_hist.tolist(), "bin_edges": bins.tolist()}
                        except:
                            distribution_ref = {"histogram": [], "bin_edges": []}
                            distribution_curr = {"histogram": [], "bin_edges": []}

                        # Summary statistics with change calculations
                        summary_stats = {
                            "mean": {
                                "reference": float(ref_vals.mean()),
                                "current": float(curr_vals.mean()),
                                "change": self._calc_change(ref_vals.mean(), curr_vals.mean()),
                            },
                            "std": {
                                "reference": float(ref_vals.std()),
                                "current": float(curr_vals.std()),
                                "change": self._calc_change(ref_vals.std(), curr_vals.std()),
                            },
                            "min": {
                                "reference": float(ref_vals.min()),
                                "current": float(curr_vals.min()),
                                "change": self._calc_change(ref_vals.min(), curr_vals.min()),
                            },
                            "max": {
                                "reference": float(ref_vals.max()),
                                "current": float(curr_vals.max()),
                                "change": self._calc_change(ref_vals.max(), curr_vals.max()),
                            },
                            "q25": {
                                "reference": float(ref_vals.quantile(0.25)),
                                "current": float(curr_vals.quantile(0.25)),
                                "change": self._calc_change(ref_vals.quantile(0.25), curr_vals.quantile(0.25)),
                            },
                            "q50": {
                                "reference": float(ref_vals.quantile(0.5)),
                                "current": float(curr_vals.quantile(0.5)),
                                "change": self._calc_change(ref_vals.quantile(0.5), curr_vals.quantile(0.5)),
                            },
                            "q75": {
                                "reference": float(ref_vals.quantile(0.75)),
                                "current": float(curr_vals.quantile(0.75)),
                                "change": self._calc_change(ref_vals.quantile(0.75), curr_vals.quantile(0.75)),
                            },
                        }

                    else:
                        # Categorical feature analysis
                        ref_counts = reference_df[col].value_counts()
                        curr_counts = current_df[col].value_counts()

                        # Chi-square test for categorical drift
                        try:
                            all_categories = list(set(ref_counts.index) | set(curr_counts.index))
                            ref_vals = [ref_counts.get(cat, 0) for cat in all_categories]
                            curr_vals = [curr_counts.get(cat, 0) for cat in all_categories]

                            if len(all_categories) > 1 and sum(ref_vals) > 0 and sum(curr_vals) > 0:
                                chi2, p_value, _, _ = chi2_contingency([ref_vals, curr_vals])
                                drift_score = float(min(chi2 / 10, 10))  # Cap at 10
                                status = "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"
                            else:
                                chi2, p_value, drift_score, status = 0, 1.0, 0, "low"
                        except:
                            chi2, p_value, drift_score, status = 0, 1.0, 0, "low"

                        psi_value = 0.0  # PSI not applicable to categorical

                        distribution_ref = {"counts": {str(k): int(v) for k, v in ref_counts.to_dict().items()}}
                        distribution_curr = {"counts": {str(k): int(v) for k, v in curr_counts.to_dict().items()}}
                        summary_stats = {}

                    feature_analysis_list.append(
                        {
                            "feature": col,
                            "feature_type": feature_type,
                            "drift_score": round(drift_score, 3),
                            "p_value": float(p_value),
                            "status": status,
                            "psi": round(psi_value, 4),
                            "distribution_ref": distribution_ref,
                            "distribution_curr": distribution_curr,
                            "summary_stats": summary_stats,
                            "missing_values": {
                                "reference": int(reference_df[col].isna().sum()),
                                "current": int(current_df[col].isna().sum()),
                            },
                        }
                    )

                except Exception as e:
                    print(f"Error in feature deep dive for {col}: {str(e)}")
                    continue

            if len(feature_analysis_list) == 0:
                return {"error": "No features could be analyzed"}

            # Overall summary
            high_drift_count = sum(1 for f in feature_analysis_list if f["status"] == "high")
            medium_drift_count = sum(1 for f in feature_analysis_list if f["status"] == "medium")
            low_drift_count = sum(1 for f in feature_analysis_list if f["status"] == "low")

            avg_drift_score = sum(f["drift_score"] for f in feature_analysis_list) / len(feature_analysis_list)

            overall_status = "high" if avg_drift_score > 3 else "medium" if avg_drift_score > 1 else "low"

            return {
                "status": "success",
                "data": {
                    "feature_analysis": feature_analysis_list,
                    "summary": {
                        "total_features": len(feature_analysis_list),
                        "high_drift_features": high_drift_count,
                        "medium_drift_features": medium_drift_count,
                        "low_drift_features": low_drift_count,
                        "average_drift_score": round(avg_drift_score, 3),
                        "overall_status": overall_status,
                    },
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "recommendations": self._generate_feature_recommendations(
                        overall_status, high_drift_count, medium_drift_count
                    ),
                },
            }

        except Exception as e:
            return {"error": f"Feature deep dive analysis failed: {str(e)}"}

    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            if len(reference) == 0 or len(current) == 0:
                return 0.0

            # Use reference data to define bin edges
            ref_vals = reference.dropna()
            curr_vals = current.dropna()

            if len(ref_vals) < 2 or len(curr_vals) < 2:
                return 0.0

            # Create bins based on reference data
            bin_edges = np.histogram_bin_edges(ref_vals, bins=bins)

            # Calculate histograms
            ref_hist, _ = np.histogram(ref_vals, bins=bin_edges)
            curr_hist, _ = np.histogram(curr_vals, bins=bin_edges)

            # Convert to proportions
            ref_pct = ref_hist / len(ref_vals)
            curr_pct = curr_hist / len(curr_vals)

            # Handle zero values
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)

            # Calculate PSI
            psi = np.sum((ref_pct - curr_pct) * np.log(ref_pct / curr_pct))
            return float(psi)

        except Exception as e:
            return 0.0

    def _calc_change(self, ref: float, curr: float) -> float:
        """Calculate percentage change between reference and current values"""
        try:
            if pd.isna(ref) or pd.isna(curr) or ref == 0:
                return 0.0
            return round(((curr - ref) / ref) * 100, 1)
        except:
            return 0.0

    def _generate_feature_recommendations(self, overall_status: str, high_count: int, medium_count: int) -> List[str]:
        """Generate recommendations for feature deep dive analysis"""
        recommendations = []

        if overall_status == "high":
            recommendations.extend(
                [
                    "Critical feature drift detected across multiple features",
                    "Immediate model retraining recommended",
                    "Investigate data pipeline changes and feature engineering processes",
                ]
            )
        elif overall_status == "medium":
            recommendations.extend(
                [
                    "Moderate feature drift detected",
                    "Plan model retraining within 1-2 months",
                    "Monitor high and medium drift features closely",
                ]
            )
        else:
            recommendations.extend(["Feature distributions remain stable", "Continue regular monitoring schedule"])

        if high_count > 0:
            recommendations.append(f"Priority attention needed for {high_count} high-drift features")
        if medium_count > 0:
            recommendations.append(f"Monitor {medium_count} medium-drift features for trend development")

        return recommendations
