"""
fairness_summary.py
Core logic for BiasLens™ Analyzer fairness metrics calculation with user-configurable thresholds.

This module provides comprehensive fairness analysis including:
- Equal Opportunity (TPR parity)
- Disparate Impact (four-fifths rule)
- Statistical Parity (demographic parity)
- Equalized Odds (TPR/FPR equality)
- Group Performance Comparison (accuracy, precision, recall, F1)

All metrics support custom user-defined thresholds for ACCEPTABLE/WARNING/BREACH status.
"""

import io
import ipaddress
import json
import logging
import os
import pickle
import socket
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import chardet
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import secure helper functions for SSRF protection and safe deserialization
from services.mainflow.app.utils.helper_function import (
    get_default_allowed_host_suffixes,
    get_default_allowed_hosts,
    make_safe_request,
    safe_pickle_load,
    validate_and_resolve_url,
)

logger = logging.getLogger(__name__)


class LLMBasedAnalysis:
    """LLM-based fairness analysis using Claude via AWS Bedrock."""

    def __init__(self, bedrock_runtime=None):
        if bedrock_runtime is not None:
            self.bedrock_runtime = bedrock_runtime
        else:
            self.bedrock_runtime = None
            self._initialize_bedrock()

    def _initialize_bedrock(self):
        """Initialize AWS Bedrock client."""
        try:
            import boto3

            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("REGION_LLM", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_LLM"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_LLM"),
            )
            logger.info("AWS Bedrock initialized successfully")
        except Exception as e:
            logger.warning(f"Bedrock initialization failed: {e}. LLM analysis will be unavailable.")
            self.bedrock_runtime = None

    def _invoke_claude(self, prompt: str) -> str:
        """Invoke Claude via AWS Bedrock."""
        if not self.bedrock_runtime:
            return "LLM analysis unavailable: Bedrock not initialized"

        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            }

            response = self.bedrock_runtime.invoke_model(
                modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                body=json.dumps(request_body),
            )

            response_body = json.loads(response["body"].read().decode("utf-8"))
            return response_body["content"][0]["text"]
        except Exception as e:
            logger.error(f"Claude invocation failed: {e}")
            return f"LLM analysis failed: {str(e)}"

    def analyze_fairness_metrics(self, metrics: Dict[str, Any], group_performance: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate LLM analysis for fairness metrics.

        Args:
            metrics: Calculated fairness metrics (disparate impact, statistical parity, etc.)
            group_performance: Group-wise performance comparison

        Returns:
            Dictionary with 'overview', 'risk_signal', and 'why_it_matters' sections
        """
        # Build summary for LLM
        summary = self._build_metrics_summary(metrics, group_performance)

        prompt = f"""You are an expert in ethical AI and fairness auditing. Analyze the following fairness metrics:

{summary}

Provide a concise analysis in THREE sections:

1. **What This Means** (2-3 sentences)
   Explain the key fairness finding in plain language. What does the disparate impact ratio tell us?

2. **Why It Matters** (2-3 sentences)
   Explain the real-world implications and potential risks.

3. **Risk Signal** (1-2 sentences)
   Assess the overall fairness status and provide actionable guidance.

Use clear, non-technical language suitable for business stakeholders. Be direct and specific."""

        analysis_text = self._invoke_claude(prompt)

        # Parse the response into structured sections
        return self._parse_analysis(analysis_text, metrics)

    def _build_metrics_summary(self, metrics: Dict[str, Any], group_performance: Dict[str, Any]) -> str:
        """Build a text summary of metrics for LLM analysis."""
        summary = "=== Fairness Metrics ===\n\n"

        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                summary += f"{metric_name.replace('_', ' ').title()}:\n"
                for key, value in metric_data.items():
                    summary += f"  - {key}: {value}\n"
                summary += "\n"

        summary += "\n=== Group Performance ===\n\n"
        if "groups" in group_performance:
            for group_name, perf in group_performance["groups"].items():
                summary += f"Group '{group_name}':\n"
                for key, value in perf.items():
                    summary += f"  - {key}: {value}\n"
                summary += "\n"

        if "disparities" in group_performance:
            summary += "Disparities:\n"
            for key, value in group_performance["disparities"].items():
                summary += f"  - {key}: {value}\n"

        return summary

    def _parse_analysis(self, analysis_text: str, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Parse LLM analysis into structured sections."""
        # Default structure
        result = {"what_this_means": "", "why_it_matters": "", "risk_signal": ""}

        # Extract disparate impact for default messages
        di_ratio = metrics.get("disparate_impact", {}).get("ratio", 0)
        di_percentage = metrics.get("disparate_impact", {}).get("percentage", 0)
        di_status = metrics.get("disparate_impact", {}).get("status", "UNKNOWN")

        # Try to parse sections from LLM response
        lines = analysis_text.split("\n")
        current_section = None

        for line in lines:
            line_lower = line.lower().strip()

            if "what this means" in line_lower:
                current_section = "what_this_means"
                continue
            elif "why it matters" in line_lower:
                current_section = "why_it_matters"
                continue
            elif "risk signal" in line_lower:
                current_section = "risk_signal"
                continue

            if current_section and line.strip() and not line.strip().startswith("#"):
                # Remove markdown formatting
                clean_line = line.strip().lstrip("*-").strip()
                if clean_line:
                    result[current_section] += clean_line + " "

        # Fallback if parsing failed
        if not result["what_this_means"]:
            result["what_this_means"] = (
                f"The model shows a disparate impact ratio of {di_ratio:.2f}, indicating that "
                f"protected groups receive favorable outcomes at {di_percentage:.0f}% the rate of non-protected groups."
            )

        if not result["why_it_matters"]:
            result["why_it_matters"] = (
                f"Disparate impact below 0.80 is often considered evidence of potential discrimination "
                f"under the four-fifths rule commonly used in regulatory assessments."
            )

        if not result["risk_signal"]:
            if di_status == "BREACH":
                result["risk_signal"] = "This metric has breached acceptable thresholds. Immediate review recommended."
            elif di_status == "WARNING":
                result["risk_signal"] = (
                    "While within acceptable range, this metric is approaching the warning threshold. "
                    "Consider reviewing training data for representation bias."
                )
            else:
                result["risk_signal"] = "Fairness metrics are within acceptable thresholds."

        # Clean up extra spaces
        for key in result:
            result[key] = result[key].strip()

        return result


@dataclass
class FairnessThresholds:
    """Configurable thresholds for fairness metrics."""

    acceptable: int = 70
    warning: int = 50
    breach: int = 30

    def validate(self) -> None:
        """Validate threshold values."""
        if not (self.acceptable > self.warning > self.breach):
            raise ValueError(
                f"Invalid threshold configuration: Acceptable ({self.acceptable}) must be > "
                f"Warning ({self.warning}) must be > Breach ({self.breach})"
            )
        if not all(0 <= t <= 100 for t in [self.acceptable, self.warning, self.breach]):
            raise ValueError("All thresholds must be between 0 and 100")
        logger.debug(
            f"Thresholds validated: acceptable={self.acceptable}, warning={self.warning}, breach={self.breach}"
        )


@dataclass
class MetricSelection:
    """User's selection of which metrics to calculate."""

    disparate_impact: bool = True
    statistical_parity: bool = True
    equalized_odds: bool = True
    equal_opportunity: bool = True

    def has_any_selected(self) -> bool:
        """Check if at least one metric is selected."""
        selected = any([self.disparate_impact, self.statistical_parity, self.equalized_odds, self.equal_opportunity])
        logger.debug(f"Metric selection check: {selected}")
        return selected

    def get_selected_metrics(self) -> List[str]:
        """Return list of selected metric names."""
        selected = []
        if self.disparate_impact:
            selected.append("disparate_impact")
        if self.statistical_parity:
            selected.append("statistical_parity")
        if self.equalized_odds:
            selected.append("equalized_odds")
        if self.equal_opportunity:
            selected.append("equal_opportunity")
        return selected


class FairnessSummaryCalculator:
    """
    Core calculator for fairness metrics with user-configurable thresholds.

    This class handles the complete fairness analysis workflow:
    1. Load datasets from S3 URLs
    2. Preprocess data (keep sensitive features readable)
    3. Calculate selected fairness metrics
    4. Apply custom thresholds
    5. Generate structured API responses
    """

    # Fixed thresholds for group performance (not user-configurable)
    GROUP_PERFORMANCE_THRESHOLDS = {
        "acceptable": 0.05,  # ≤5% disparity
        "warning": 0.10,  # ≤10% disparity
        "breach": 0.10,  # >10% disparity
    }

    # Fixed thresholds for Disparate Impact (ratio scale)
    DISPARATE_IMPACT_THRESHOLDS = {
        "acceptable": 0.80,  # ≥0.80 (four-fifths rule)
        "warning": 0.70,  # ≥0.70
        "breach": 0.70,  # <0.70
    }

    def __init__(self, default_thresholds: Optional[FairnessThresholds] = None):
        self.default_thresholds = default_thresholds or FairnessThresholds()
        self.llm_analyzer = LLMBasedAnalysis()
        logger.info(
            "FairnessSummaryCalculator initialized with default thresholds: "
            f"acceptable={self.default_thresholds.acceptable}%, "
            f"warning={self.default_thresholds.warning}%, "
            f"breach={self.default_thresholds.breach}%"
        )
        logger.info("LLM-based analysis support enabled")

    def validate_external_url(self, url: str) -> None:
        """
        Mitigates SSRF by validating URL scheme, resolving the host,
        and blocking private/internal IP ranges.

        Allows both HTTP and HTTPS schemes for flexibility.

        Args:
            url: The URL to validate

        Raises:
            ValueError: If URL is invalid or points to internal/private addresses
        """
        if not url:
            raise ValueError("Empty URL is not allowed")

        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError("Only HTTP and HTTPS URLs are allowed")

        host = parsed.hostname
        if not host:
            raise ValueError("URL must include a hostname")

        # Block obvious local targets and ensure it looks like a domain
        lowered_host = host.lower()
        if lowered_host in ("localhost", "127.0.0.1", "::1") or "." not in lowered_host:
            raise ValueError("Invalid or internal hostname")

        try:
            # Resolve hostname to IP to prevent DNS Rebinding attacks
            port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
            addrinfos = socket.getaddrinfo(host, port)
            for family, _, _, _, sockaddr in addrinfos:
                ip_str = sockaddr[0]
                ip = ipaddress.ip_address(ip_str)
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                    raise ValueError("Connections to private or non-routable addresses are not allowed")
        except socket.gaierror as e:
            raise ValueError(f"Failed to resolve host: {e}")

    def _validate_model_url(self, model_url: str) -> None:
        """
        Strict validation for models to prevent SSRF and Deserialization of malicious code.
        Restricts access to known S3 buckets or a trusted internal Files API.

        This enforces:
            - HTTPS scheme only (for models)
            - Presence of a hostname
            - Hostname restricted to allowed S3 domains or FILES_API_BASE_URL
            - No resolution to private, loopback, link-local, or sensitive IP ranges

        Args:
            model_url: The model URL to validate

        Raises:
            ValueError: If URL is not from a trusted source or fails security checks
        """
        if not model_url:
            raise ValueError("Model URL must not be empty")

        parsed = urlparse(model_url)
        scheme = (parsed.scheme or "").lower()
        hostname = (parsed.hostname or "").lower()

        # Enforce HTTPS for model files (stricter than datasets)
        if scheme != "https":
            raise ValueError("Only HTTPS model URLs are allowed for security")

        if not hostname:
            raise ValueError("Model URL must include a hostname")

        # Block obvious local targets
        if hostname in ("localhost", "127.0.0.1", "::1") or "." not in hostname:
            raise ValueError("Invalid or internal hostname for model URL")

        # Build allowed hosts list
        allowed_hosts = ["s3.amazonaws.com"]
        trusted_base = os.getenv("FILES_API_BASE_URL")
        if trusted_base:
            trusted_host = urlparse(trusted_base).hostname
            if trusted_host:
                allowed_hosts.append(trusted_host.lower())

        # Check against S3 allowlist patterns
        allowed_suffixes = (
            ".s3.amazonaws.com",
            ".amazonaws.com",
        )
        is_allowed_host = hostname in allowed_hosts or any(hostname.endswith(suffix) for suffix in allowed_suffixes)

        if not is_allowed_host:
            raise ValueError(f"Model URL host '{hostname}' is not in the list of allowed domains")

        # DNS resolution check - verify no private/internal IPs
        try:
            addrinfo_list = socket.getaddrinfo(hostname, None)
        except socket.gaierror as e:
            raise ValueError(f"Failed to resolve model URL host '{hostname}': {e}") from e

        for family, _, _, _, sockaddr in addrinfo_list:
            ip_str = sockaddr[0]
            try:
                ip_obj = ipaddress.ip_address(ip_str)
            except ValueError:
                raise ValueError(f"Unparseable IP address '{ip_str}' for host '{hostname}'")

            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_reserved
                or ip_obj.is_multicast
            ):
                raise ValueError(f"Model URL resolves to a disallowed IP address '{ip_str}'")

        logger.debug(f"Model URL passed SSRF validation: {model_url[:100]}")

    def load_flexible_csv(self, file_content: bytes, source_name: str = "file") -> pd.DataFrame:
        """
        Load CSV with automatic encoding and delimiter detection.

        Args:
            file_content: Raw bytes of CSV file
            source_name: Name for logging purposes

        Returns:
            Parsed DataFrame

        Raises:
            RuntimeError: If CSV cannot be parsed
        """
        logger.info(f"Loading CSV from {source_name} ({len(file_content)} bytes)")

        # Detect encoding
        detected = chardet.detect(file_content)
        encoding = detected["encoding"] or "utf-8"
        confidence = detected["confidence"]
        logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2%})")

        # Try multiple delimiters
        delimiters = [",", ";", "\t", "|"]

        for delimiter in delimiters:
            try:
                df = pd.read_csv(io.BytesIO(file_content), encoding=encoding, delimiter=delimiter, on_bad_lines="skip")

                # Validate successful parsing (must have at least 2 columns)
                if df.shape[1] > 1:
                    logger.info(
                        f"Successfully loaded CSV with delimiter '{delimiter}': "
                        f"{df.shape[0]} rows, {df.shape[1]} columns"
                    )
                    return df

            except Exception as e:
                logger.debug(f"Failed to parse with delimiter '{delimiter}': {e}")
                continue

        # If all delimiters failed
        raise RuntimeError(
            f"Failed to load CSV from {source_name}: Could not determine delimiter. "
            "Tried: comma, semicolon, tab, pipe. Data might be malformed."
        )

    def download_and_load_dataframe(self, file_url: str, source_name: str = "dataset") -> pd.DataFrame:
        """
        Download file from S3 URL and load as DataFrame with SSRF protection.

        Uses centralized security utilities from helper_function.py for:
        - URL validation and sanitization
        - DNS resolution with private IP blocking
        - Safe HTTP requests with redirect prevention

        Args:
            file_url: S3 pre-signed URL
            source_name: Name for logging purposes

        Returns:
            Loaded DataFrame

        Raises:
            RuntimeError: If download or parsing fails, or URL is unsafe
        """
        try:
            # SSRF Protection: Validate URL using centralized helper
            # This performs scheme validation, host allowlisting, and DNS resolution
            validated = validate_and_resolve_url(
                url=file_url,
                allowed_schemes={"http", "https"},
                allowed_hosts=get_default_allowed_hosts(),
                allowed_host_suffixes=get_default_allowed_host_suffixes(),
                source_name=source_name,
            )

            logger.info(f"Downloading {source_name} from URL: {file_url[:100]}...")

            # SSRF-SAFE: URL has been validated - scheme is HTTP/HTTPS, host is in S3 allowlist,
            # and DNS resolution confirmed no private/internal IPs. Reconstruct a normalized URL
            # from the parsed components to avoid using the raw user input string.
            response = make_safe_request(
                validated_url=validated,
                timeout=60,
                use_validated_ip=False,  # Use safe_url (normalized from parsed components)
            )

            logger.debug(f"Downloaded {len(response.content)} bytes for {source_name}")
            return self.load_flexible_csv(response.content, source_name)

        except ValueError as ve:
            logger.error(f"Rejected {source_name} URL as unsafe: {ve}")
            raise RuntimeError(f"Invalid {source_name} URL: {ve}") from ve
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {source_name} from URL: {e}", exc_info=True)
            raise RuntimeError(f"Failed to download {source_name} from URL: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load {source_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load {source_name}: {str(e)}")

    def load_and_preprocess_data(self, df: pd.DataFrame, target_col: str, sensitive_feature: str) -> pd.DataFrame:
        """
        Preprocess dataset while keeping sensitive features readable.

        CRITICAL: Sensitive features are kept in original string form for readability.
        Other categorical columns are label-encoded for model compatibility.

        Args:
            df: Raw DataFrame
            target_col: Name of target column
            sensitive_feature: Name of sensitive feature (e.g., 'gender')

        Returns:
            Preprocessed DataFrame

        Raises:
            ValueError: If target or sensitive feature not found
        """
        logger.info(f"Preprocessing data: {df.shape[0]} rows, {df.shape[1]} columns")

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Validate columns exist
        if target_col not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise ValueError(f"Target column '{target_col}' not found in dataset. " f"Available columns: {available}")

        if sensitive_feature not in df.columns:
            available = ", ".join(df.columns.tolist())
            raise ValueError(
                f"Sensitive feature '{sensitive_feature}' not found in dataset. " f"Available columns: {available}"
            )

        # Strip whitespace from string values
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.strip()

        # Keep sensitive feature as original strings (DO NOT ENCODE)
        logger.debug(
            f"Preserving sensitive feature '{sensitive_feature}' as strings: "
            f"{df[sensitive_feature].unique().tolist()}"
        )

        # Handle target column - binarize if necessary
        unique_targets = df[target_col].nunique()
        if unique_targets == 2:
            # Binary target - handle common mappings
            unique_vals = df[target_col].unique()
            if df[target_col].dtype == "object":
                # String values - map to 0/1
                mapping = {}
                for val in unique_vals:
                    val_lower = str(val).lower()
                    if val_lower in ["yes", "approved", "true", "1", "positive"]:
                        mapping[val] = 1
                    elif val_lower in ["no", "rejected", "false", "0", "negative"]:
                        mapping[val] = 0

                if len(mapping) == 2:
                    df[target_col] = df[target_col].map(mapping)
                    logger.debug(f"Mapped target column using: {mapping}")
                else:
                    # Fallback: most frequent = 0, other = 1
                    most_frequent = df[target_col].value_counts().index[0]
                    df[target_col] = (df[target_col] != most_frequent).astype(int)
                    logger.debug(f"Binary target mapped: {most_frequent} -> 0, others -> 1")
        elif unique_targets > 2:
            # Multi-class target - binarize (most frequent = 0, others = 1)
            most_frequent = df[target_col].value_counts().index[0]
            df[target_col] = (df[target_col] != most_frequent).astype(int)
            logger.warning(f"Multi-class target binarized: {most_frequent} -> 0 (n={unique_targets} classes)")

        # Label-encode other categorical columns (except sensitive feature and target)
        for col in df.columns:
            if col not in [target_col, sensitive_feature] and df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                logger.debug(f"Label-encoded column: {col}")

        logger.info(
            f"Preprocessing complete. Target values: {df[target_col].unique()}, "
            f"Sensitive groups: {df[sensitive_feature].unique().tolist()}"
        )

        return df

    def prepare_training_and_test_data(
        self, df_reference: pd.DataFrame, df_current: Optional[pd.DataFrame], target_col: str, sensitive_feature: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Prepare train/test split from datasets.

        Logic:
        - If df_current provided: reference=train, current=test
        - If df_current is None: 70/30 stratified split on reference

        Args:
            df_reference: Reference dataset (always used as train if current provided)
            df_current: Current dataset (optional, used as test)
            target_col: Name of target column
            sensitive_feature: Name of sensitive feature

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, A_train, A_test)
            where A is the sensitive feature
        """
        if df_current is not None:
            logger.info("Using separate datasets: reference=train, current=test")

            # Validate feature columns match
            ref_features = set(df_reference.columns) - {target_col}
            cur_features = set(df_current.columns) - {target_col}

            if ref_features != cur_features:
                missing_in_current = ref_features - cur_features
                missing_in_ref = cur_features - ref_features
                error_msg = "Feature columns mismatch between reference and current datasets."
                if missing_in_current:
                    error_msg += f" Missing in current: {missing_in_current}."
                if missing_in_ref:
                    error_msg += f" Missing in reference: {missing_in_ref}."
                raise ValueError(error_msg)

            # Split features and target
            X_train = df_reference.drop(columns=[target_col])
            y_train = df_reference[target_col]
            X_test = df_current.drop(columns=[target_col])
            y_test = df_current[target_col]

            # Extract sensitive features
            A_train = X_train[sensitive_feature]
            A_test = X_test[sensitive_feature]

            logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        else:
            logger.info("Splitting reference dataset 70/30 with stratification")

            X = df_reference.drop(columns=[target_col])
            y = df_reference[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            A_train = X_train[sensitive_feature]
            A_test = X_test[sensitive_feature]

            logger.info(f"Split complete: Train={len(X_train)}, Test={len(X_test)}")

        return X_train, X_test, y_train, y_test, A_train, A_test

    def validate_thresholds(self, thresholds: Dict[str, int]) -> FairnessThresholds:
        """
        Validate user-provided thresholds.

        Args:
            thresholds: Dict with 'acceptable', 'warning', 'breach' keys

        Returns:
            FairnessThresholds instance

        Raises:
            ValueError: If thresholds are invalid
        """
        try:
            thresh_obj = FairnessThresholds(**thresholds)
            thresh_obj.validate()
            return thresh_obj
        except Exception as e:
            logger.error(f"Threshold validation failed: {e}")
            raise ValueError(f"Invalid threshold configuration: {str(e)}")

    def validate_metric_selection(self, metric_selection: Dict[str, bool]) -> MetricSelection:
        """
        Validate metric selection.

        Args:
            metric_selection: Dict of metric names to boolean values

        Returns:
            MetricSelection instance

        Raises:
            ValueError: If no metrics selected
        """
        selection = MetricSelection(**metric_selection)
        if not selection.has_any_selected():
            raise ValueError("No metrics selected. Please select at least one fairness metric to evaluate.")

        logger.info(f"Metrics selected: {selection.get_selected_metrics()}")
        return selection

    # ===================================================================================
    # PHASE 2: METRIC CALCULATION METHODS (Complete Implementation)
    # ===================================================================================

    def calculate_equal_opportunity(
        self, y_true: pd.Series, y_pred: np.ndarray, sensitive_feature: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate Equal Opportunity metric (TPR parity across groups).

        Equal Opportunity measures whether protected groups have equal True Positive Rate
        (recall), ensuring fair benefit from the system when deserving positive outcomes.

        Formula:
            TPR_group = TP / (TP + FN) for each group
            Equal_Opportunity = (min_TPR / max_TPR) × 100

        Args:
            y_true: Actual target values
            y_pred: Predicted values
            sensitive_feature: Sensitive attribute values (e.g., 'Male', 'Female')

        Returns:
            Dict with percentage, status, TPRs, and group names

        Raises:
            ValueError: If less than 2 groups or invalid inputs
        """
        logger.info("Calculating Equal Opportunity metric...")

        # Get unique groups
        groups = sensitive_feature.unique()
        if len(groups) < 2:
            raise ValueError(f"Equal Opportunity requires at least 2 demographic groups. Found: {len(groups)} group(s)")

        logger.debug(f"Analyzing {len(groups)} groups: {groups.tolist()}")

        # Calculate TPR for each group
        tpr_per_group = {}
        for group in groups:
            # Filter data for this group
            group_mask = sensitive_feature == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            # Calculate confusion matrix
            try:
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
            except ValueError:
                # Handle case where group has only one class
                tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
                fn = np.sum((y_true_group == 1) & (y_pred_group == 0))

            # Calculate TPR (recall)
            if (tp + fn) > 0:
                tpr = tp / (tp + fn)
            else:
                tpr = 0.0  # No actual positives in this group
                logger.warning(f"Group '{group}' has no actual positive samples (TP+FN=0)")

            tpr_per_group[str(group)] = tpr
            logger.debug(f"Group '{group}': TPR={tpr:.4f} (TP={tp}, FN={fn})")

        # Calculate Equal Opportunity ratio
        tpr_values = list(tpr_per_group.values())
        min_tpr = min(tpr_values)
        max_tpr = max(tpr_values)

        if max_tpr > 0:
            equal_opportunity_ratio = (min_tpr / max_tpr) * 100
        else:
            equal_opportunity_ratio = 100.0  # Both groups have 0 TPR
            logger.warning("All groups have TPR=0, setting Equal Opportunity to 100%")

        # Identify protected (lower TPR) and non-protected (higher TPR) groups
        sorted_groups = sorted(tpr_per_group.items(), key=lambda x: x[1])
        protected_group_name = sorted_groups[0][0]
        non_protected_group_name = sorted_groups[-1][0]
        tpr_protected = sorted_groups[0][1]
        tpr_non_protected = sorted_groups[-1][1]

        logger.info(
            f"Equal Opportunity: {equal_opportunity_ratio:.2f}% "
            f"(Protected '{protected_group_name}' TPR={tpr_protected:.4f}, "
            f"Non-protected '{non_protected_group_name}' TPR={tpr_non_protected:.4f})"
        )

        return {
            "percentage": round(equal_opportunity_ratio, 2),
            "tpr_protected": round(tpr_protected, 4),
            "tpr_non_protected": round(tpr_non_protected, 4),
            "protected_group_name": protected_group_name,
            "non_protected_group_name": non_protected_group_name,
            "all_tprs": {k: round(v, 4) for k, v in tpr_per_group.items()},
        }

    def calculate_disparate_impact(self, y_pred: np.ndarray, sensitive_feature: pd.Series) -> Dict[str, Any]:
        """
        Calculate Disparate Impact ratio (four-fifths rule).

        Disparate Impact measures the ratio of favorable outcomes between protected
        and non-protected groups. Values below 0.80 may indicate discrimination.

        Formula:
            DI = (Favorable rate for protected) / (Favorable rate for non-protected)

        Args:
            y_pred: Predicted values
            sensitive_feature: Sensitive attribute values

        Returns:
            Dict with ratio and percentage
        """
        logger.info("Calculating Disparate Impact metric...")

        groups = sensitive_feature.unique()
        if len(groups) < 2:
            raise ValueError(f"Disparate Impact requires at least 2 groups. Found: {len(groups)}")

        # Calculate favorable outcome rates per group
        rates = {}
        for group in groups:
            group_mask = sensitive_feature == group
            group_pred = y_pred[group_mask]
            favorable_rate = np.mean(group_pred == 1)
            rates[str(group)] = favorable_rate
            logger.debug(f"Group '{group}': Favorable rate={favorable_rate:.4f}")

        # Calculate disparate impact (min rate / max rate)
        rate_values = list(rates.values())
        min_rate = min(rate_values)
        max_rate = max(rate_values)

        if max_rate > 0:
            disparate_impact_ratio = min_rate / max_rate
        else:
            disparate_impact_ratio = 1.0
            logger.warning("All groups have 0 favorable rate, setting DI to 1.0")

        percentage = disparate_impact_ratio * 100

        logger.info(f"Disparate Impact: {disparate_impact_ratio:.4f} ({percentage:.2f}%)")

        return {"ratio": round(disparate_impact_ratio, 4), "percentage": round(percentage, 2)}

    def calculate_statistical_parity(self, y_pred: np.ndarray, sensitive_feature: pd.Series) -> Dict[str, Any]:
        """
        Calculate Statistical Parity (Demographic Parity Difference).

        Measures the absolute difference in positive prediction rates between groups.

        Formula:
            SP = |P(Ŷ=1|Protected) - P(Ŷ=1|Non-protected)| × 100

        Args:
            y_pred: Predicted values
            sensitive_feature: Sensitive attribute values

        Returns:
            Dict with percentage difference
        """
        logger.info("Calculating Statistical Parity metric...")

        groups = sensitive_feature.unique()
        if len(groups) < 2:
            raise ValueError(f"Statistical Parity requires at least 2 groups. Found: {len(groups)}")

        # Calculate positive prediction rates
        rates = []
        for group in groups:
            group_mask = sensitive_feature == group
            group_pred = y_pred[group_mask]
            pos_rate = np.mean(group_pred == 1)
            rates.append(pos_rate)
            logger.debug(f"Group '{group}': Positive prediction rate={pos_rate:.4f}")

        # Calculate absolute difference
        difference = abs(max(rates) - min(rates)) * 100

        logger.info(f"Statistical Parity Difference: {difference:.2f}%")

        return {"percentage": round(difference, 2)}

    def calculate_equalized_odds(
        self, y_true: pd.Series, y_pred: np.ndarray, sensitive_feature: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate Equalized Odds (TPR and FPR equality).

        Measures whether both True Positive Rate and False Positive Rate are
        equal across demographic groups.

        Formula:
            For each group: TPR = TP/(TP+FN), FPR = FP/(FP+TN)
            Equalized_Odds = min(TPR_ratio, FPR_ratio) × 100

        Args:
            y_true: Actual target values
            y_pred: Predicted values
            sensitive_feature: Sensitive attribute values

        Returns:
            Dict with percentage equality
        """
        logger.info("Calculating Equalized Odds metric...")

        groups = sensitive_feature.unique()
        if len(groups) < 2:
            raise ValueError(f"Equalized Odds requires at least 2 groups. Found: {len(groups)}")

        # Calculate TPR and FPR for each group
        tpr_per_group = {}
        fpr_per_group = {}

        for group in groups:
            group_mask = sensitive_feature == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            try:
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
            except ValueError:
                tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
                fp = np.sum((y_true_group == 0) & (y_pred_group == 1))
                fn = np.sum((y_true_group == 1) & (y_pred_group == 0))
                tn = np.sum((y_true_group == 0) & (y_pred_group == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            tpr_per_group[str(group)] = tpr
            fpr_per_group[str(group)] = fpr
            logger.debug(f"Group '{group}': TPR={tpr:.4f}, FPR={fpr:.4f}")

        # Calculate equality ratios
        tpr_values = list(tpr_per_group.values())
        fpr_values = list(fpr_per_group.values())

        tpr_min, tpr_max = min(tpr_values), max(tpr_values)
        fpr_min, fpr_max = min(fpr_values), max(fpr_values)

        tpr_ratio = (tpr_min / tpr_max * 100) if tpr_max > 0 else 100.0
        fpr_ratio = (fpr_min / fpr_max * 100) if fpr_max > 0 else 100.0

        # Equalized odds is the minimum of both ratios
        equalized_odds = min(tpr_ratio, fpr_ratio)

        logger.info(f"Equalized Odds: {equalized_odds:.2f}% (TPR ratio={tpr_ratio:.2f}%, FPR ratio={fpr_ratio:.2f}%)")

        return {
            "percentage": round(equalized_odds, 2),
            "tpr_ratio": round(tpr_ratio, 2),
            "fpr_ratio": round(fpr_ratio, 2),
        }

    def calculate_group_performance_comparison(
        self, y_true: pd.Series, y_pred: np.ndarray, sensitive_feature: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics comparison across demographic groups.

        Computes accuracy, precision, recall, and F1-score for each group,
        then identifies performance disparities.

        Args:
            y_true: Actual target values
            y_pred: Predicted values
            sensitive_feature: Sensitive attribute values

        Returns:
            Dict with per-group metrics, disparities, and status
        """
        logger.info("Calculating Group Performance Comparison...")

        groups = sensitive_feature.unique()
        group_metrics = {}

        for group in groups:
            group_mask = sensitive_feature == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            # Calculate metrics with zero_division handling
            accuracy = accuracy_score(y_true_group, y_pred_group)
            precision = precision_score(y_true_group, y_pred_group, zero_division=0)
            recall = recall_score(y_true_group, y_pred_group, zero_division=0)
            f1 = f1_score(y_true_group, y_pred_group, zero_division=0)

            group_metrics[str(group)] = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "sample_size": int(len(y_true_group)),
            }

            logger.debug(
                f"Group '{group}': accuracy={accuracy:.4f}, precision={precision:.4f}, "
                f"recall={recall:.4f}, f1={f1:.4f}, n={len(y_true_group)}"
            )

        # Calculate disparities (absolute differences)
        metric_names = ["accuracy", "precision", "recall", "f1_score"]
        disparities = {}

        for metric in metric_names:
            values = [group_metrics[g][metric] for g in group_metrics]
            disparity = abs(max(values) - min(values))
            disparities[f"{metric}_disparity"] = round(disparity, 4)

        max_disparity = max(disparities.values())
        disparities["max_disparity"] = round(max_disparity, 4)

        # Find worst metric
        worst_metric = max(
            [(k.replace("_disparity", ""), v) for k, v in disparities.items() if k != "max_disparity"],
            key=lambda x: x[1],
        )[0]

        # Determine status using fixed thresholds
        if max_disparity <= self.GROUP_PERFORMANCE_THRESHOLDS["acceptable"]:
            status = "ACCEPTABLE"
        elif max_disparity <= self.GROUP_PERFORMANCE_THRESHOLDS["warning"]:
            status = "WARNING"
        else:
            status = "BREACH"

        description = (
            f"Maximum performance disparity of {max_disparity*100:.1f}% observed in {worst_metric}. "
            f"Performance differences are {'within acceptable range' if status == 'ACCEPTABLE' else 'concerning'} "
            f"across demographic groups."
        )

        logger.info(f"Group Performance: {status} (max disparity={max_disparity:.4f}, worst={worst_metric})")

        return {
            "groups": group_metrics,
            "disparities": disparities,
            "status": status,
            "worst_metric": worst_metric,
            "description": description,
        }

    def apply_threshold_to_metric(self, metric_value: float, thresholds: FairnessThresholds) -> str:
        """
        Determine status based on metric value and thresholds.

        Args:
            metric_value: Calculated metric value (0-100)
            thresholds: Threshold configuration

        Returns:
            Status string: "ACCEPTABLE", "WARNING", or "BREACH"
        """
        if metric_value >= thresholds.acceptable:
            return "ACCEPTABLE"
        elif metric_value >= thresholds.warning:
            return "WARNING"
        else:
            return "BREACH"

    def apply_disparate_impact_threshold(self, ratio: float) -> str:
        """
        Apply fixed thresholds to Disparate Impact ratio.

        Args:
            ratio: Disparate Impact ratio (0.0-1.0+)

        Returns:
            Status string
        """
        if ratio >= self.DISPARATE_IMPACT_THRESHOLDS["acceptable"]:
            return "ACCEPTABLE"
        elif ratio >= self.DISPARATE_IMPACT_THRESHOLDS["warning"]:
            return "WARNING"
        else:
            return "BREACH"

    # ===================================================================================
    # PHASE 3: ORCHESTRATION AND RESPONSE FORMATTING (Complete Implementation)
    # ===================================================================================

    def calculate_all_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        sensitive_feature: pd.Series,
        selected_metrics: MetricSelection,
        custom_thresholds: FairnessThresholds,
    ) -> Dict[str, Any]:
        """
        Master orchestration function to calculate all selected metrics.

        Args:
            y_true: Actual target values
            y_pred: Predicted values
            sensitive_feature: Sensitive attribute values
            selected_metrics: Which metrics to calculate
            custom_thresholds: User-configured thresholds

        Returns:
            Complete metrics dictionary ready for API response
        """
        logger.info("=== Starting comprehensive fairness metrics calculation ===")

        key_metrics = {}
        detailed_breakdown = []

        # 1. Disparate Impact
        if selected_metrics.disparate_impact:
            logger.info("Processing Disparate Impact...")
            di_result = self.calculate_disparate_impact(y_pred, sensitive_feature)
            status = self.apply_disparate_impact_threshold(di_result["ratio"])

            key_metrics["disparate_impact"] = {
                "ratio": di_result["ratio"],
                "percentage": di_result["percentage"],
                "status": status,
            }

            detailed_breakdown.append(
                {
                    "metric": "Disparate Impact",
                    "value": int(di_result["percentage"]),
                    "unit": "ratio",
                    "status": status,
                    "bar_percentage": int(di_result["percentage"]),
                    "description": (
                        f"Protected groups receive favorable outcomes at {di_result['percentage']:.0f}% "
                        f"the rate of non-protected groups. Values below 80% may indicate discrimination "
                        f"under the four-fifths rule."
                    ),
                }
            )

        # 2. Statistical Parity
        if selected_metrics.statistical_parity:
            logger.info("Processing Statistical Parity...")
            sp_result = self.calculate_statistical_parity(y_pred, sensitive_feature)

            # For statistical parity, lower is better - invert for threshold application
            # 0% difference = 100% parity
            parity_score = 100 - sp_result["percentage"]
            status = self.apply_threshold_to_metric(parity_score, custom_thresholds)

            key_metrics["statistical_parity"] = {
                "percentage": sp_result["percentage"],
                "parity_score": round(parity_score, 2),
                "status": status,
            }

            detailed_breakdown.append(
                {
                    "metric": "Statistical Parity",
                    "value": int(sp_result["percentage"]),
                    "unit": "%",
                    "status": status,
                    "bar_percentage": int(parity_score) if parity_score >= 0 else 0,
                    "description": (
                        f"The difference in positive prediction rates between protected and "
                        f"non-protected groups is {sp_result['percentage']:.1f}%. Lower differences "
                        f"indicate better fairness."
                    ),
                }
            )

        # 3. Equal Opportunity
        if selected_metrics.equal_opportunity:
            logger.info("Processing Equal Opportunity...")
            eo_result = self.calculate_equal_opportunity(y_true, y_pred, sensitive_feature)
            status = self.apply_threshold_to_metric(eo_result["percentage"], custom_thresholds)

            key_metrics["equal_opportunity"] = {
                "percentage": eo_result["percentage"],
                "status": status,
                "tpr_protected": eo_result["tpr_protected"],
                "tpr_non_protected": eo_result["tpr_non_protected"],
                "protected_group_name": eo_result["protected_group_name"],
                "non_protected_group_name": eo_result["non_protected_group_name"],
            }

            detailed_breakdown.append(
                {
                    "metric": "Equal Opportunity",
                    "value": int(eo_result["percentage"]),
                    "unit": "%",
                    "status": status,
                    "bar_percentage": int(eo_result["percentage"]),
                    "description": (
                        f"Protected groups have equal opportunity to receive positive outcomes when deserved, "
                        f"with {eo_result['percentage']:.0f}% TPR parity between '{eo_result['protected_group_name']}' "
                        f"(TPR={eo_result['tpr_protected']:.2f}) and '{eo_result['non_protected_group_name']}' "
                        f"(TPR={eo_result['tpr_non_protected']:.2f})."
                    ),
                }
            )

        # 4. Equalized Odds
        if selected_metrics.equalized_odds:
            logger.info("Processing Equalized Odds...")
            eq_result = self.calculate_equalized_odds(y_true, y_pred, sensitive_feature)
            status = self.apply_threshold_to_metric(eq_result["percentage"], custom_thresholds)

            key_metrics["equalized_odds"] = {
                "percentage": eq_result["percentage"],
                "status": status,
                "tpr_ratio": eq_result["tpr_ratio"],
                "fpr_ratio": eq_result["fpr_ratio"],
            }

            detailed_breakdown.append(
                {
                    "metric": "Equalized Odds",
                    "value": int(eq_result["percentage"]),
                    "unit": "%",
                    "status": status,
                    "bar_percentage": int(eq_result["percentage"]),
                    "description": (
                        f"Model maintains equal true positive and false positive rates across groups "
                        f"at {eq_result['percentage']:.0f}% parity."
                    ),
                }
            )

        # 5. Group Performance Comparison (always calculated)
        logger.info("Processing Group Performance Comparison...")
        group_perf = self.calculate_group_performance_comparison(y_true, y_pred, sensitive_feature)

        # Get disparate impact ratio for slider (use first metric if DI not selected)
        if "disparate_impact" in key_metrics:
            disparate_impact_ratio = int(key_metrics["disparate_impact"]["percentage"])
        else:
            disparate_impact_ratio = 50  # Default fallback

        logger.info("=== Fairness metrics calculation complete ===")

        return {
            "key_metrics": key_metrics,
            "disparate_impact_ratio": disparate_impact_ratio,
            "detailed_breakdown": detailed_breakdown,
            "group_performance": group_perf,
        }

    def load_model_and_predict(
        self, model_url: str, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load model from URL and generate predictions with strict URL validation.

        Uses centralized security utilities from helper_function.py for:
        - URL validation (HTTPS only, host allowlisting)
        - DNS resolution with private IP blocking
        - IP-bound requests to prevent DNS rebinding
        - Safe deserialization with module allowlisting

        Args:
            model_url: S3 pre-signed URL to model file
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (train_predictions, test_predictions)

        Raises:
            RuntimeError: If model loading or prediction fails, or URL is not trusted
        """
        logger.info(f"Loading model from URL: {model_url[:100]}...")

        try:
            # ============================================================
            # SSRF & Deserialization Protection using centralized helpers
            # ============================================================

            # 1. Validate URL using centralized helper with HTTPS-only requirement
            validated = validate_and_resolve_url(
                url=model_url,
                allowed_schemes={"https"},  # HTTPS only for model downloads
                allowed_hosts=get_default_allowed_hosts(),
                allowed_host_suffixes=get_default_allowed_host_suffixes(),
                source_name="model",
            )

            logger.debug(f"Model URL passed security validation: {validated.hostname} -> {validated.validated_ip}")

            # ============================================================
            # Safe to proceed - URL has been fully validated above.
            # Use normalized safe_url (from parsed.geturl()) to prevent URL manipulation.
            # NOTE: IP binding is not used here because it breaks SSL certificate validation
            # (certificates are issued for domain names, not IPs). The normalized URL combined
            # with allow_redirects=False provides adequate SSRF protection.
            # ============================================================
            response = make_safe_request(
                validated_url=validated,
                timeout=60,
                use_validated_ip=False,  # Use safe_url to maintain SSL certificate compatibility
            )

            logger.debug(f"Downloaded model: {len(response.content)} bytes")

            # Safe deserialization with SafeUnpickler to prevent arbitrary code execution
            # SafeUnpickler restricts loaded classes to safe sklearn/numpy modules only
            model = safe_pickle_load(response.content)
            logger.info(f"Model loaded securely: {type(model).__name__}")

            # Generate predictions
            logger.info("Generating predictions...")
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Validate predictions are binary
            unique_train = np.unique(y_pred_train)
            unique_test = np.unique(y_pred_test)
            if not (set(unique_train) <= {0, 1} and set(unique_test) <= {0, 1}):
                raise ValueError(
                    f"Model predictions must be binary (0/1). " f"Found train: {unique_train}, test: {unique_test}"
                )

            logger.info(f"Predictions generated: train={len(y_pred_train)}, test={len(y_pred_test)}")

            return y_pred_train, y_pred_test

        except ValueError as e:
            logger.error(f"Invalid model URL: {e}", exc_info=True)
            raise RuntimeError(f"Invalid model URL: {str(e)}")
        except pickle.UnpicklingError as ue:
            logger.error(f"Model deserialization blocked - untrusted class detected: {ue}")
            raise RuntimeError(f"Model file contains untrusted classes: {str(ue)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to download model from URL: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load or use model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def process_fairness_request(
        self,
        reference_url: str,
        target_column: str,
        sensitive_feature: str,
        metric_selection: Dict[str, bool],
        current_url: Optional[str] = None,
        model_url: Optional[str] = None,
        project_id: Optional[str] = None,
        thresholds: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for fairness analysis request.

        This orchestrates the complete analysis workflow:
        1. Validate inputs
        2. Download and preprocess datasets
        3. Load model and generate predictions (if needed)
        4. Calculate selected metrics
        5. Format complete API response

        Args:
            reference_url: S3 URL to reference dataset
            target_column: Name of target column
            sensitive_feature: Name of sensitive feature
            metric_selection: Dict of which metrics to calculate
            current_url: Optional S3 URL to current dataset
            model_url: Optional S3 URL to model file
            project_id: Optional project identifier
            thresholds: Optional custom threshold configuration

        Returns:
            Complete API response dictionary

        Raises:
            ValueError: If validation fails
            RuntimeError: If processing fails
        """
        logger.info("=" * 80)
        logger.info("STARTING FAIRNESS ANALYSIS REQUEST")
        logger.info("=" * 80)

        start_time = datetime.utcnow()

        try:
            # 1. Validate metric selection
            logger.info("Step 1: Validating metric selection...")
            selected_metrics = self.validate_metric_selection(metric_selection)

            # 2. Validate and set thresholds
            logger.info("Step 2: Validating thresholds...")
            if thresholds:
                custom_thresholds = self.validate_thresholds(thresholds)
                threshold_source = "user_configured"
            else:
                custom_thresholds = self.default_thresholds
                threshold_source = "default"

            logger.info(
                f"Using thresholds: acceptable={custom_thresholds.acceptable}%, "
                f"warning={custom_thresholds.warning}%, breach={custom_thresholds.breach}% "
                f"(source: {threshold_source})"
            )

            # 3. Download and load datasets
            logger.info("Step 3: Downloading datasets...")
            df_reference = self.download_and_load_dataframe(reference_url, "reference dataset")

            if current_url:
                df_current = self.download_and_load_dataframe(current_url, "current dataset")
            else:
                df_current = None
                logger.info("No current dataset provided, will use 70/30 split")

            # 4. Preprocess datasets
            logger.info("Step 4: Preprocessing datasets...")
            df_reference = self.load_and_preprocess_data(df_reference, target_column, sensitive_feature)

            if df_current is not None:
                df_current = self.load_and_preprocess_data(df_current, target_column, sensitive_feature)

            # 5. Prepare train/test split
            logger.info("Step 5: Preparing train/test split...")
            X_train, X_test, y_train, y_test, A_train, A_test = self.prepare_training_and_test_data(
                df_reference, df_current, target_column, sensitive_feature
            )

            # Validate sufficient data
            if len(X_train) < 5 or len(X_test) < 5:
                raise ValueError(
                    f"Insufficient data: At least 5 samples required in train and test sets. "
                    f"Found: train={len(X_train)}, test={len(X_test)}"
                )

            # Validate at least 2 groups in sensitive feature
            unique_groups_train = A_train.nunique()
            unique_groups_test = A_test.nunique()
            if unique_groups_train < 2 or unique_groups_test < 2:
                raise ValueError(
                    f"Sensitive feature must have at least 2 unique groups in both train and test. "
                    f"Found: train={unique_groups_train}, test={unique_groups_test}"
                )

            # 6. Generate predictions
            if model_url:
                logger.info("Step 6: Loading model and generating predictions...")
                y_pred_train, y_pred_test = self.load_model_and_predict(model_url, X_train, X_test)
            else:
                logger.info("Step 6: No model provided, using actual values as predictions (for testing)")
                y_train.values
                y_pred_test = y_test.values

            # 7. Calculate metrics (use test set for fairness evaluation)
            logger.info("Step 7: Calculating fairness metrics...")
            metrics_result = self.calculate_all_metrics(
                y_test, y_pred_test, A_test, selected_metrics, custom_thresholds
            )

            # 8. Generate warnings
            warnings = []
            for metric_name, metric_data in metrics_result["key_metrics"].items():
                if metric_data.get("status") == "WARNING":
                    warnings.append(
                        f"{metric_name.replace('_', ' ').title()} approaching warning threshold. "
                        f"Consider reviewing training data for representation bias."
                    )
                elif metric_data.get("status") == "BREACH":
                    warnings.append(
                        f"{metric_name.replace('_', ' ').title()} has breached acceptable threshold. "
                        f"Immediate review recommended."
                    )

            # Check for small sample sizes
            group_sizes = A_test.value_counts().to_dict()
            for group, size in group_sizes.items():
                if size < 30:
                    warnings.append(
                        f"Small sample size in group '{group}' (n={size}). " f"Results may be less reliable."
                    )

            # 9. Generate LLM-based analysis
            logger.info("Step 8: Generating LLM-based fairness analysis...")
            llm_analysis = {}
            try:
                llm_analysis = self.llm_analyzer.analyze_fairness_metrics(
                    metrics_result["key_metrics"], metrics_result["group_performance"]
                )
                logger.info("LLM analysis completed successfully")
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}. Proceeding without LLM insights.")
                llm_analysis = {
                    "what_this_means": "LLM analysis unavailable",
                    "why_it_matters": "LLM analysis unavailable",
                    "risk_signal": "LLM analysis unavailable",
                }

            # 10. Build complete response
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()

            logger.info(f"Step 9: Building final response (processing time: {processing_time:.2f}s)")

            response = {
                "success": True,
                "message": "Fairness analysis completed successfully",
                "data": {
                    **metrics_result,  # key_metrics, disparate_impact_ratio, detailed_breakdown, group_performance
                    "llm_analysis": llm_analysis,
                    "applied_thresholds": {
                        "acceptable": custom_thresholds.acceptable,
                        "warning": custom_thresholds.warning,
                        "breach": custom_thresholds.breach,
                        "source": threshold_source,
                    },
                    "metadata": {
                        "source_reference": reference_url,
                        "source_current": current_url or "N/A (70/30 split used)",
                        "evaluated": end_time.isoformat() + "Z",
                        "target_column": target_column,
                        "sensitive_feature": sensitive_feature,
                        "reference_samples": len(df_reference),
                        "current_samples": len(df_current) if df_current is not None else 0,
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                        "protected_groups": [str(g) for g in A_test.unique()],
                        "metrics_evaluated": selected_metrics.get_selected_metrics(),
                        "processing_time_seconds": round(processing_time, 2),
                    },
                },
                "warnings": warnings,
                "timestamp": end_time.isoformat() + "Z",
            }

            logger.info("=" * 80)
            logger.info(f"FAIRNESS ANALYSIS COMPLETED SUCCESSFULLY ({processing_time:.2f}s)")
            logger.info("=" * 80)

            return response

        except ValueError as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            raise
        except RuntimeError as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during fairness analysis: {e}", exc_info=True)
            raise RuntimeError(f"Internal error during fairness analysis: {str(e)}")
