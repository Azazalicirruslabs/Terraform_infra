"""
Explainability Logic Module

This module contains the core business logic for model explainability analysis,
including model loading, metrics calculation, and SHAP-based feature importance.
"""

import io
import ipaddress
import logging
import os
import pickle
import socket
import tempfile
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import requests
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

# Security: Allowed modules for safe model deserialization
SAFE_MODULES = {
    "sklearn",
    "numpy",
    "pandas",
    "scipy",
    "xgboost",
    "lightgbm",
    "catboost",
    "joblib",
    "builtins",
    "__builtin__",
}


class RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows loading of trusted modules.
    This prevents arbitrary code execution during deserialization.
    """

    def find_class(self, module, name):
        """Only allow safe modules to be loaded."""
        # Check if the module is in our safe list
        module_root = module.split(".")[0]

        if module_root not in SAFE_MODULES:
            raise pickle.UnpicklingError(
                f"Loading module '{module}' is not allowed for security reasons. "
                f"Only whitelisted ML modules are permitted."
            )

        return super().find_class(module, name)


def _validate_model_source(path: str) -> None:
    """
    Validate that the model is being loaded from a trusted source.

    Args:
        path: Path or URL to the model file

    Raises:
        ValueError: If the source is not trusted
    """
    # Validate path is not empty
    if not path or not path.strip():
        raise ValueError("Model path must not be empty.")

    # Get allowed model storage locations from environment or use defaults
    allowed_dirs = os.getenv("ALLOWED_MODEL_DIRS", "").split(",")
    allowed_s3_buckets = os.getenv("ALLOWED_S3_BUCKETS", "").split(",")

    # For URLs, validate against allowed S3 buckets
    if path.startswith(("http://", "https://")):
        parsed = urlparse(path)
        hostname = (parsed.hostname or "").lower()

        # Helper to extract S3 bucket name from common URL patterns
        def _extract_s3_bucket(host: str, url_path: str) -> str | None:
            host = host or ""
            url_path = url_path or ""

            # Strip leading '/' from path for easier splitting
            normalized_path = url_path.lstrip("/")

            # Virtual-hosted–style URL: <bucket>.s3[.<region>].amazonaws.com
            host_parts = host.split(".")
            # Require at least 3 parts: <bucket>, 's3' or 's3-<region>', 'amazonaws', 'com', ...
            if len(host_parts) >= 3 and host_parts[-2] == "amazonaws" and host_parts[-1] == "com":
                bucket_candidate = host_parts[0]
                if bucket_candidate and bucket_candidate not in ("s3", "s3-accelerate"):
                    return bucket_candidate

            # Path-style URL: s3[.<region>].amazonaws.com/<bucket>/...
            if (
                len(host_parts) >= 3
                and host_parts[0].startswith("s3")
                and host_parts[-2] == "amazonaws"
                and host_parts[-1] == "com"
            ):
                # Path must be like '<bucket>/...' or exactly '<bucket>'
                if normalized_path:
                    path_parts = normalized_path.split("/")
                    bucket_candidate = path_parts[0]
                    if bucket_candidate:
                        return bucket_candidate

            return None

        # Check if it's an S3 URL and if the bucket is allowed
        is_s3_host = hostname.endswith(".amazonaws.com") or hostname == "amazonaws.com"
        if is_s3_host and allowed_s3_buckets and allowed_s3_buckets[0]:
            bucket_name = _extract_s3_bucket(hostname, parsed.path)
            # If we cannot determine the bucket name, treat the source as untrusted
            if not bucket_name or bucket_name not in [b for b in allowed_s3_buckets if b]:
                logger.warning("Attempted to load model from untrusted S3 bucket: %s", hostname)
                raise ValueError(
                    "Model source not trusted. Only models from configured S3 buckets are allowed. "
                    "Please contact your administrator to whitelist this source."
                )
        return  # Allow other HTTPS sources (pre-signed URLs validated by authentication)

    # For local files, validate against allowed directories
    try:
        # Security: Sanitize path to prevent directory traversal before resolving
        # Extract only the filename component to prevent path traversal attacks
        safe_filename = os.path.basename(os.path.normpath(path))
        if not safe_filename or safe_filename in (".", ".."):
            raise ValueError(f"Invalid model filename in source validation: {path}")

        # Additional validation: disallow path separators in the filename
        if os.sep in safe_filename or (os.altsep and os.altsep in safe_filename):
            raise ValueError(f"Invalid model filename contains path separators: {path}")

        # If allowed_dirs is configured, validate against it
        if allowed_dirs and allowed_dirs[0]:
            # Validate that at least one allowed directory exists and contains this file
            is_allowed = False
            for allowed_dir in allowed_dirs:
                if not allowed_dir:
                    continue
                # Resolve the trusted allowed directory
                allowed_base_resolved = str(Path(allowed_dir).resolve())
                # Construct path using os.path.join and normalize it
                candidate_path_str = os.path.normpath(os.path.join(allowed_base_resolved, safe_filename))
                # Verify the normalized path starts with the allowed base (containment check)
                if not candidate_path_str.startswith(allowed_base_resolved + os.sep):
                    # Edge case: check if it's exactly the base directory (though unlikely for a file)
                    if candidate_path_str != allowed_base_resolved:
                        continue
                is_allowed = True
                break

            if not is_allowed:
                logger.warning("Attempted to load model from untrusted directory: %s", path)
                raise ValueError(
                    "Model source not trusted. Only models from configured directories are allowed. "
                    "Please contact your administrator to whitelist this location."
                )
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Error validating model path: %s", e)
        raise ValueError(f"Invalid model path: {path}") from e


class ModelWrapper:
    """Wrapper class for different model types."""

    def __init__(self, model, model_type: str = "sklearn"):
        self.model = model
        self.model_type = model_type

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities (if applicable)."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("Model does not support predict_proba")


class ExplainabilityService:
    """Service class for handling explainability analysis."""

    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = []
        self.target_name = None
        self.explainer = None
        self.shap_values = None
        self.model_info = {}

    def load_model_and_datasets(
        self,
        model_path: str,
        train_data_path: str,
        test_data_path: str,
        target_column: str,
    ) -> Dict[str, Any]:
        """
        Load model and datasets from local files or S3 URLs.

        Args:
            model_path: Path or URL to the model file
            train_data_path: Path or URL to the training dataset
            test_data_path: Path or URL to the test dataset
            target_column: Name of the target column

        Returns:
            Dict with loading status and metadata
        """
        try:
            logger.info("Loading model from: %s", model_path)
            logger.info("Loading train data from: %s", train_data_path)
            logger.info("Loading test data from: %s", test_data_path)

            # Load model
            model_wrapper = self._load_model(model_path)
            self.model = model_wrapper

            # Load datasets
            logger.info("Loading training dataset...")
            train_df = self._load_dataset(train_data_path)

            logger.info("Loading test dataset...")
            test_df = self._load_dataset(test_data_path)

            # Validate target column
            if target_column not in train_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in training dataset.")
            if target_column not in test_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in test dataset.")

            # Validate feature columns match
            train_features = set(train_df.drop(columns=[target_column]).columns)
            test_features = set(test_df.drop(columns=[target_column]).columns)
            if train_features != test_features:
                missing_in_test = train_features - test_features
                missing_in_train = test_features - train_features
                error_msg = "Feature columns mismatch between train and test datasets."
                if missing_in_test:
                    error_msg += f" Missing in test: {missing_in_test}."
                if missing_in_train:
                    error_msg += f" Missing in train: {missing_in_train}."
                raise ValueError(error_msg)

            # Extract features and targets
            self.X_train = train_df.drop(columns=[target_column])
            self.y_train = train_df[target_column]
            self.X_test = test_df.drop(columns=[target_column])
            self.y_test = test_df[target_column]
            self.feature_names = list(self.X_train.columns)
            self.target_name = target_column

            logger.info("Training samples: %s, Test samples: %s", len(self.X_train), len(self.X_test))

            # Detect model type (classification or regression)
            model_type = self._detect_model_type(model_wrapper)

            # Initialize SHAP explainer
            self._initialize_shap_explainer(model_wrapper)

            # Store model info
            self.model_info = {
                "model_type": model_type,
                "features_count": len(self.feature_names),
                "train_samples": len(self.X_train),
                "test_samples": len(self.X_test),
                "shap_available": self.explainer is not None,
                "target_column": target_column,
            }

            return {
                "status": "success",
                "message": "Model and datasets loaded successfully",
                "model_type": model_type,
                "features_count": len(self.feature_names),
            }

        except Exception as exc:
            logger.error("Error loading model and datasets: %s", exc)
            self._reset_state()
            raise

    def compute_explainability_analysis(
        self, compute_feature_importance: bool = True, method: str = "shap", thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive explainability analysis.

        Args:
            compute_feature_importance: Whether to compute feature importance
            method: Feature importance method - 'shap', 'gain', or 'permutation'
            thresholds: Optional dict with acceptable_threshold, warning_threshold, breach_threshold

        Returns:
            Dict with performance metrics, feature importance, LLM analysis, and threshold evaluations
        """
        self._ensure_loaded()

        model_type = self.model_info.get("model_type")

        if model_type == "classification":
            performance_metrics = self._compute_classification_metrics()
        elif model_type == "regression":
            performance_metrics = self._compute_regression_metrics()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compute feature importance (if requested)
        feature_importance = {}
        if compute_feature_importance:
            feature_importance = self._compute_feature_importance_by_method(method)
        else:
            # Return empty feature importance structure
            feature_importance = {
                "shap_available": False,
                "total_features": len(self.feature_names),
                "positive_impact_count": 0,
                "negative_impact_count": 0,
                "features": [],
                "computation_method": "none",
            }

        # Evaluate thresholds if provided
        threshold_evaluations = None
        if thresholds:
            threshold_evaluations = self._evaluate_thresholds(performance_metrics, thresholds, model_type)

        # Generate LLM analysis
        llm_analysis = {}
        try:
            llm_analyzer = LLMBasedExplainabilityAnalysis()
            llm_analysis = llm_analyzer.analyze_explainability_results(
                performance_metrics=performance_metrics, feature_importance=feature_importance, model_type=model_type
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("LLM analysis failed: %s. Continuing without LLM insights.", exc)
            llm_analysis = {
                "what_this_means": "LLM analysis unavailable",
                "why_it_matters": "LLM analysis unavailable",
                "risk_signal": "LLM analysis unavailable",
            }

        result = {
            "performance_metrics": performance_metrics,
            **feature_importance,
            "llm_analysis": llm_analysis,
        }

        if threshold_evaluations:
            result["threshold_evaluations"] = threshold_evaluations

        return result

    def _validate_external_url(self, url: str) -> None:
        """
        Validate that a URL is safe to request and doesn't point to internal resources.

        This prevents Server-Side Request Forgery (SSRF) attacks by ensuring the URL:
        - Uses HTTPS scheme only (more restrictive for security)
        - Doesn't resolve to private, loopback, or reserved IP addresses
        - Doesn't target internal network resources

        Args:
            url: The URL to validate

        Raises:
            ValueError: If the URL is invalid or points to a restricted resource
        """
        try:
            parsed = urlparse(url)

            # Only allow HTTPS scheme for maximum security
            if parsed.scheme.lower() != "https":
                raise ValueError(f"Invalid URL scheme '{parsed.scheme}'. Only HTTPS is allowed.")

            # Extract hostname
            hostname = parsed.hostname
            if not hostname:
                raise ValueError("URL must contain a valid hostname.")

            # Resolve hostname to IP addresses
            try:
                # Get all IP addresses for the hostname
                addr_info = socket.getaddrinfo(hostname, None)
                ip_addresses = {info[4][0] for info in addr_info}
            except socket.gaierror as exc:
                raise ValueError(f"Unable to resolve hostname '{hostname}': {exc}") from exc

            # Check each resolved IP address
            for ip_str in ip_addresses:
                try:
                    ip_obj = ipaddress.ip_address(ip_str)

                    # Block private addresses (e.g., 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
                    if ip_obj.is_private:
                        raise ValueError(
                            f"Access to private IP addresses is not allowed. "
                            f"Hostname '{hostname}' resolves to private IP: {ip_str}"
                        )

                    # Block loopback addresses (127.0.0.0/8, ::1)
                    if ip_obj.is_loopback:
                        raise ValueError(
                            f"Access to loopback addresses is not allowed. "
                            f"Hostname '{hostname}' resolves to loopback IP: {ip_str}"
                        )

                    # Block link-local addresses (169.254.0.0/16 - AWS metadata service!)
                    if ip_obj.is_link_local:
                        raise ValueError(
                            f"Access to link-local addresses is not allowed. "
                            f"Hostname '{hostname}' resolves to link-local IP: {ip_str}"
                        )

                    # Block multicast addresses
                    if ip_obj.is_multicast:
                        raise ValueError(
                            f"Access to multicast addresses is not allowed. "
                            f"Hostname '{hostname}' resolves to multicast IP: {ip_str}"
                        )

                    # Block reserved addresses
                    if ip_obj.is_reserved:
                        raise ValueError(
                            f"Access to reserved addresses is not allowed. "
                            f"Hostname '{hostname}' resolves to reserved IP: {ip_str}"
                        )

                    # Block unspecified addresses (0.0.0.0, ::)
                    if ip_obj.is_unspecified:
                        raise ValueError(
                            f"Access to unspecified addresses is not allowed. "
                            f"Hostname '{hostname}' resolves to unspecified IP: {ip_str}"
                        )

                except ValueError as exc:
                    # Re-raise our custom validation errors
                    if "not allowed" in str(exc):
                        raise
                    # Invalid IP format - should not happen with socket.getaddrinfo
                    raise ValueError(f"Invalid IP address format: {ip_str}") from exc

            logger.info("URL validation passed for: %s (resolved to: %s)", hostname, ip_addresses)

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as exc:
            # Catch any other unexpected errors
            raise ValueError(f"URL validation failed: {exc}") from exc

    def _validate_external_url_with_allowlist(self, url: str) -> None:
        """
        Validate that a dataset URL is safe to access with SSRF mitigation.

        Rules:
        - Scheme must be HTTP or HTTPS.
        - Hostname must be present.
        - If an allow-list is configured (EXPLAINABILITY_ALLOWED_HOSTS), the host
          must match one of the allowed hosts or be a subdomain of an allowed host.
        - All resolved IP addresses must be public (not private, loopback, link-local, etc.).

        Args:
            url: The URL to validate

        Raises:
            ValueError: If the URL is invalid or points to a restricted resource
        """
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only HTTP/HTTPS URLs are allowed for datasets.")

        hostname = parsed.hostname
        if not hostname:
            raise ValueError("Dataset URL must include a valid hostname.")

        # Optional hostname allow-list, e.g. "data.example.com,s3.amazonaws.com"
        allowed_hosts_env = os.getenv("EXPLAINABILITY_ALLOWED_HOSTS", "").strip()
        if allowed_hosts_env:
            allowed_hosts = {h.strip().lower() for h in allowed_hosts_env.split(",") if h.strip()}
        else:
            allowed_hosts = set()

        host_lower = hostname.lower()
        if allowed_hosts:

            def _is_allowed(h: str) -> bool:
                for allowed in allowed_hosts:
                    if h == allowed or h.endswith("." + allowed):
                        return True
                return False

            if not _is_allowed(host_lower):
                raise ValueError(f"Host '{hostname}' is not in the list of allowed dataset hosts.")

        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            raise ValueError(f"Could not resolve dataset host '{hostname}': {exc}") from exc

        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            try:
                ip_obj = ipaddress.ip_address(ip_str)
            except ValueError:
                continue

            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
                or ip_obj.is_unspecified
            ):
                raise ValueError(f"Dataset host '{hostname}' resolves to a disallowed IP address: {ip_str}")

    def _get_safe_local_model_path(self, model_path: str) -> str:
        """
        Resolve and validate a local model file path to prevent directory traversal.

        The base directory can be configured via the EXPLAINABILITY_MODEL_BASE_DIR
        environment variable; if not set, a default of "/models" is used.

        Args:
            model_path: User-provided path to the model file.

        Returns:
            A normalized, absolute path to the model file within the allowed base dir.

        Raises:
            ValueError: If the resolved path is outside the allowed base directory
                or if the filename/extension is not allowed.
        """
        # Only allow model files with known-safe extensions
        allowed_extensions = (".joblib", ".pkl", ".pickle")

        # Determine the allowed base directory for local model files and normalize it.
        base_dir = os.environ.get("EXPLAINABILITY_MODEL_BASE_DIR", "/models")
        base_path = Path(base_dir).expanduser().resolve()

        # Sanitize the user input to prevent path traversal
        # Remove any directory traversal attempts and convert to safe filename
        safe_filename = os.path.basename(os.path.normpath(model_path))
        if not safe_filename or safe_filename in (".", ".."):
            raise ValueError(f"Invalid model filename: {model_path}")

        # Disallow any path separators in the resulting filename (defense in depth)
        if os.sep in safe_filename or (os.altsep and os.altsep in safe_filename):
            raise ValueError(f"Invalid model filename: {model_path}")

        # Enforce allowed extensions on the sanitized filename
        if not safe_filename.lower().endswith(allowed_extensions):
            raise ValueError(
                f"Unsupported model filename or extension for local path: {model_path}. "
                f"Allowed extensions: {', '.join(allowed_extensions)}"
            )

        # Construct the full path within the base directory
        candidate_path = base_path / safe_filename
        resolved_path = candidate_path.resolve()

        # Ensure the resolved path is within the allowed base directory.
        try:
            resolved_path.relative_to(base_path)
        except ValueError as exc:
            raise ValueError(
                f"Model path '{model_path}' is not allowed. "
                f"Path must be within {base_dir}. "
                "Configure EXPLAINABILITY_MODEL_BASE_DIR to change allowed directory."
            ) from exc

        # Ensure the resolved path points to an existing regular file.
        if not resolved_path.exists():
            raise ValueError(f"Model file does not exist at resolved path: {resolved_path}")
        if not resolved_path.is_file():
            raise ValueError(f"Resolved model path is not a file: {resolved_path}")

        return str(resolved_path)

    def _is_allowed_model_url(self, url: str) -> bool:
        """
        Additional hardening for model URLs downloaded from external sources.

        Enforces:
        - HTTPS scheme
        - Hostname not in private or loopback ranges
        - Hostname matches an allowed pattern (for example, S3)
        """
        parsed = urlparse(url)

        # Require HTTPS to prevent protocol abuse
        if parsed.scheme != "https":
            return False

        hostname = parsed.hostname or ""
        # Disallow obvious localhost patterns
        if hostname in {"localhost", "127.0.0.1", "::1"}:
            return False

        try:
            ip = ipaddress.ip_address(hostname)
            # Reject private, loopback, link-local, or reserved IPs
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except ValueError:
            # Hostname is not a literal IP; enforce an allowed domain pattern instead.
            # Adjust this list/pattern to match the deployment's trusted model hosts.
            allowed_suffixes = (".s3.amazonaws.com", ".amazonaws.com", "s3.amazonaws.com", "amazonaws.com")
            if not any(hostname.endswith(suffix) or hostname == suffix for suffix in allowed_suffixes):
                return False

        return True

    def _load_model(self, model_path: str) -> ModelWrapper:
        """
        Load model from file path or S3 URL with security validation.

        Security measures:
        - Validates model source against allowed locations
        - Uses restricted unpickler to prevent arbitrary code execution
        - Only allows whitelisted ML framework modules
        """
        # Security: Validate the model source
        _validate_model_source(model_path)

        # Check if it's a URL (S3 pre-signed URL)
        if model_path.startswith(("http://", "https://")):
            return self._load_model_from_url(model_path)

        # Handle local file paths with additional path hardening
        safe_model_path = self._get_safe_local_model_path(model_path)
        file_extension = safe_model_path.lower()

        if file_extension.endswith(".joblib"):
            # Use the same restricted unpickler for joblib files, since joblib
            # uses pickle under the hood and arbitrary objects could otherwise
            # be deserialized from untrusted input.
            # Temporarily replace pickle.Unpickler with our RestrictedUnpickler
            original_unpickler = pickle.Unpickler
            try:
                pickle.Unpickler = RestrictedUnpickler
                model = joblib.load(safe_model_path)
            finally:
                pickle.Unpickler = original_unpickler
            return ModelWrapper(model, "sklearn")
        elif file_extension.endswith((".pkl", ".pickle")):
            # Use restricted unpickler for security
            with open(safe_model_path, "rb") as f:
                model = RestrictedUnpickler(f).load()
            return ModelWrapper(model, "sklearn")
        else:
            raise ValueError(f"Unsupported model format. Supported formats: .joblib, .pkl, .pickle. Got: {model_path}")

    def _load_model_from_url(self, url: str) -> ModelWrapper:
        """
        Load model from S3 pre-signed URL with security validation.

        Security measures:
        - Validates URL to prevent SSRF attacks
        - Enforces HTTPS and allowed hostnames
        - Disables redirects to prevent redirect-based SSRF
        - Downloads to temporary file with size limits
        - Uses restricted unpickler for pickle files
        - Validates file content before loading
        """
        logger.info("Downloading model from URL...")

        # Security: Validate URL to prevent SSRF
        self._validate_external_url(url)

        # Parse URL once for further validation
        parsed_url = urlparse(url)

        # Additional hardening: enforce HTTPS scheme
        if parsed_url.scheme.lower() != "https":
            raise ValueError("Only HTTPS URLs are allowed for model downloads.")

        # Additional hardening: enforce allowed hostnames (e.g., S3 and compatible storage)
        hostname = (parsed_url.hostname or "").lower()
        allowed_suffixes = (
            ".amazonaws.com",
            ".s3.amazonaws.com",
            "amazonaws.com",
            "s3.amazonaws.com",
        )
        if not any(hostname.endswith(suffix) or hostname == suffix for suffix in allowed_suffixes):
            raise ValueError(f"Model URL host '{hostname}' is not allowed. Only trusted S3 domains are permitted.")

        # Use existing host allow-list logic if present
        if not self._is_allowed_model_url(url):
            raise ValueError(
                "Model URL is not allowed or is potentially unsafe. "
                "Only HTTPS URLs from trusted domains (e.g., S3) are permitted."
            )

        # Resolve hostname and validate that the target IP is not private or loopback
        try:
            addr_info_list = socket.getaddrinfo(hostname, None)
        except OSError as exc:
            raise ValueError(f"Failed to resolve model URL host '{hostname}': {exc}") from exc

        safe_ip_found = False
        for family, _, _, _, sockaddr in addr_info_list:
            if family not in (socket.AF_INET, socket.AF_INET6):
                continue
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)
            # Reject private, loopback, link-local, multicast, and reserved addresses
            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
            ):
                continue
            safe_ip_found = True
            break

        if not safe_ip_found:
            raise ValueError(
                f"Resolved IP address for host '{hostname}' is not allowed (private, loopback, or reserved)."
            )

        # Security: Set a reasonable file size limit (500 MB)
        MAX_MODEL_SIZE = 500 * 1024 * 1024  # 500 MB

        # Reconstruct URL from validated components to satisfy CodeQL
        # At this point, parsed_url, hostname, and IPs have all been validated
        validated_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        if parsed_url.query:
            validated_url += f"?{parsed_url.query}"
        if parsed_url.fragment:
            validated_url += f"#{parsed_url.fragment}"

        try:
            # Use a session and disable redirects to avoid redirect-based SSRF
            with requests.Session() as session:
                response = session.get(validated_url, stream=True, timeout=30, allow_redirects=False)

            # If the server attempted to redirect, reject it rather than following.
            if 300 <= response.status_code < 400:
                raise ValueError("Redirects are not allowed when downloading model URLs.")

            response.raise_for_status()

            # Check content length if provided
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_MODEL_SIZE:
                raise ValueError(f"Model file too large: {content_length} bytes (max: {MAX_MODEL_SIZE})")
        except requests.exceptions.Timeout as exc:
            raise ValueError("Model download timed out after 30 seconds.") from exc
        except requests.exceptions.RequestException as exc:
            raise ValueError(f"Failed to download model: {str(exc)}") from exc

        # Determine file extension from URL
        file_path = parsed_url.path
        file_extension = file_path.lower()

        model_bytes = response.content

        # Security: Validate downloaded size
        if len(model_bytes) > MAX_MODEL_SIZE:
            raise ValueError(f"Model file too large: {len(model_bytes)} bytes (max: {MAX_MODEL_SIZE})")

        logger.info("Downloaded %s bytes", len(model_bytes))

        if file_extension.endswith(".joblib"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_file:
                tmp_file.write(model_bytes)
                tmp_path = tmp_file.name

            try:
                # Security: Use restricted unpickler for joblib (which uses pickle internally)
                # Temporarily replace pickle.Unpickler with our RestrictedUnpickler
                original_unpickler = pickle.Unpickler
                try:
                    pickle.Unpickler = RestrictedUnpickler
                    model = joblib.load(tmp_path)
                finally:
                    pickle.Unpickler = original_unpickler
                return ModelWrapper(model, "sklearn")
            finally:
                os.unlink(tmp_path)

        elif file_extension.endswith((".pkl", ".pickle")):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
                tmp_file.write(model_bytes)
                tmp_path = tmp_file.name

            try:
                # Security: Use restricted unpickler
                with open(tmp_path, "rb") as f:
                    model = RestrictedUnpickler(f).load()
                return ModelWrapper(model, "sklearn")
            finally:
                os.unlink(tmp_path)
        else:
            raise ValueError(f"Unsupported model format in URL: {file_path}")

    def _load_dataset(self, path: str) -> pd.DataFrame:
        """Load dataset from file path or S3 URL."""
        if not path:
            raise ValueError("Empty path provided")

        # If it's a URL
        if path.startswith(("http://", "https://")):
            # Security: Validate URL to prevent SSRF (allows both HTTP and HTTPS for datasets)
            self._validate_external_url_with_allowlist(path)

            # Reconstruct URL from validated components to satisfy CodeQL
            parsed_path = urlparse(path)
            validated_url = f"{parsed_path.scheme}://{parsed_path.netloc}{parsed_path.path}"
            if parsed_path.query:
                validated_url += f"?{parsed_path.query}"
            if parsed_path.fragment:
                validated_url += f"#{parsed_path.fragment}"

            try:
                resp = requests.get(validated_url, timeout=30)
                resp.raise_for_status()
            except requests.exceptions.Timeout as exc:
                raise ValueError("Dataset download timed out.") from exc
            except requests.exceptions.RequestException as exc:
                raise ValueError(f"Failed to download dataset: {exc}") from exc

            # Determine file type from URL
            url_no_query = path.split("?", 1)[0]
            ext = os.path.splitext(url_no_query)[1].lower()

            if ext == ".csv":
                return pd.read_csv(io.StringIO(resp.text))
            elif ext == ".parquet":
                return pd.read_parquet(io.BytesIO(resp.content), engine="pyarrow")
            else:
                raise ValueError(f"Unsupported file format: {ext}. Supported: .csv, .parquet")

        # Local file path
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext == ".parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: .csv, .parquet")

    def _detect_model_type(self, model_wrapper: ModelWrapper) -> str:
        """Detect if model is classification or regression."""
        if hasattr(model_wrapper.model, "predict_proba"):
            return "classification"
        elif hasattr(model_wrapper.model, "_estimator_type"):
            estimator_type = getattr(model_wrapper.model, "_estimator_type", None)
            if estimator_type == "classifier":
                return "classification"
            elif estimator_type == "regressor":
                return "regression"
        # Default to regression if predict_proba is not available
        return "regression"

    def _initialize_shap_explainer(self, model_wrapper: ModelWrapper):
        """Initialize SHAP explainer and compute SHAP values."""
        logger.info("Initializing SHAP explainer...")

        try:
            # Try TreeExplainer first (faster for tree-based models)
            try:
                self.explainer = shap.TreeExplainer(model_wrapper.model)
                logger.info("Using TreeExplainer")
            except Exception:  # pylint: disable=broad-except
                # Fallback to KernelExplainer
                sample_size = min(100, len(self.X_train))
                background_data = self.X_train.values[:sample_size]

                if hasattr(model_wrapper.model, "predict_proba"):

                    def predict_fn(x):
                        return model_wrapper.model.predict_proba(x)

                else:

                    def predict_fn(x):
                        return model_wrapper.model.predict(x)

                self.explainer = shap.KernelExplainer(predict_fn, background_data)
                logger.info("Using KernelExplainer with %s background samples", sample_size)

            # Compute SHAP values on a sample
            sample_size = min(100, len(self.X_train))
            self.shap_values = self.explainer.shap_values(self.X_train.values[:sample_size])
            logger.info("SHAP values computed successfully")

        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not initialize SHAP explainer: %s", exc)
            self.explainer = None
            self.shap_values = None

    def _compute_classification_metrics(self) -> Dict[str, Any]:
        """Compute classification performance metrics."""
        # Training metrics
        y_pred_train = self.model.predict(self.X_train.values)
        train_metrics = self._calculate_classification_metrics(self.y_train, y_pred_train, self.X_train.values)

        # Test metrics
        y_pred_test = self.model.predict(self.X_test.values)
        test_metrics = self._calculate_classification_metrics(self.y_test, y_pred_test, self.X_test.values)

        # Calculate overfitting score (difference in accuracy)
        overfitting_score = max(0.0, train_metrics["accuracy"] - test_metrics["accuracy"])

        return {
            "train": train_metrics,
            "test": test_metrics,
            "overfitting_score": overfitting_score,
        }

    def _calculate_classification_metrics(self, y_true, y_pred, X_values) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        # Calculate AUC if predict_proba is available
        try:
            if hasattr(self.model.model, "predict_proba"):
                y_proba = self.model.model.predict_proba(X_values)
                classes = np.unique(y_true)

                if len(classes) == 2:
                    # Binary classification
                    auc_score = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class classification
                    y_true_bin = label_binarize(y_true, classes=classes)
                    auc_score = roc_auc_score(y_true_bin, y_proba, average="weighted", multi_class="ovr")

                metrics["auc"] = float(auc_score)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not calculate AUC: %s", exc)

        return metrics

    def _compute_regression_metrics(self) -> Dict[str, Any]:
        """Compute regression performance metrics."""
        # Training metrics
        y_pred_train = self.model.predict(self.X_train.values)
        train_metrics = self._calculate_regression_metrics(self.y_train, y_pred_train)

        # Test metrics
        y_pred_test = self.model.predict(self.X_test.values)
        test_metrics = self._calculate_regression_metrics(self.y_test, y_pred_test)

        # Calculate overfitting score (difference in R² score)
        overfitting_score = max(0.0, train_metrics["r2_score"] - test_metrics["r2_score"])

        return {
            "train": train_metrics,
            "test": test_metrics,
            "overfitting_score": overfitting_score,
        }

    def _calculate_regression_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except Exception:  # pylint: disable=broad-except
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1)))

        # SMAPE
        denominator = np.abs(y_true) + np.abs(y_pred)
        smape = np.mean(np.where(denominator == 0, 0, 2.0 * np.abs(y_pred - y_true) / denominator)) * 100

        # Adjusted R²
        n_samples = len(y_true)
        n_features = len(self.feature_names)
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

        # Explained Variance
        explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)

        return {
            "r2_score": float(r2),
            "rmse": float(rmse),
            "mse": float(mse),
            "mae": float(mae),
            "mape": float(mape),
            "smape": float(smape),
            "adjusted_r2": float(adj_r2),
            "explained_variance": float(explained_variance),
        }

    def _compute_shap_feature_importance(self) -> Dict[str, Any]:
        """Compute SHAP-based feature importance."""
        if self.explainer is None or self.shap_values is None:
            return {
                "shap_available": False,
                "total_features": 0,
                "positive_impact_count": 0,
                "negative_impact_count": 0,
                "features": [],
                "computation_method": "shap",
                "computed_at": pd.Timestamp.utcnow().isoformat(),
            }

        # Get SHAP values for analysis
        shap_vals = self._get_shap_values_for_analysis()

        # Calculate signed means (for direction) and absolute means (for importance)
        signed_means = shap_vals.mean(axis=0)
        unsigned_magnitudes = np.abs(shap_vals).mean(axis=0)

        # Build feature importance list
        features = []
        for idx, feature_name in enumerate(self.feature_names):
            importance = float(unsigned_magnitudes[idx])
            signed_value = float(signed_means[idx])
            impact_direction = "positive" if signed_value >= 0 else "negative"

            features.append(
                {
                    "name": feature_name,
                    "importance": importance,
                    "impact_direction": impact_direction,
                    "rank": 0,  # Will be set after sorting
                }
            )

        # Sort by importance (descending)
        features.sort(key=lambda x: x["importance"], reverse=True)

        # Assign ranks
        for rank, feature in enumerate(features, start=1):
            feature["rank"] = rank

        # Count positive and negative impacts
        positive_count = sum(1 for f in features if f["impact_direction"] == "positive")
        negative_count = sum(1 for f in features if f["impact_direction"] == "negative")

        return {
            "shap_available": True,
            "total_features": len(features),
            "positive_impact_count": positive_count,
            "negative_impact_count": negative_count,
            "features": features,
            "computation_method": "shap",
            "computed_at": pd.Timestamp.utcnow().isoformat(),
        }

    def _get_shap_values_for_analysis(self) -> np.ndarray:
        """Get SHAP values appropriate for analysis."""
        if isinstance(self.shap_values, list):
            # Binary classification stored as list
            if len(self.shap_values) == 2:
                return self.shap_values[1]  # Use positive class
            else:
                # Multi-class: average across all classes
                return np.array(self.shap_values).mean(axis=0)
        elif len(self.shap_values.shape) == 3:
            # Multi-class: (n_samples, n_features, n_classes)
            return self.shap_values.mean(axis=2)
        else:
            return self.shap_values

    def _compute_feature_importance_by_method(self, method: str) -> Dict[str, Any]:
        """
        Compute feature importance using the specified method.

        Args:
            method: One of 'shap', 'gain', or 'permutation'

        Returns:
            Dict with feature importance results
        """
        method = method.lower()

        # Variables to store importance values and directions
        raw_importance = None
        impact_directions = None

        try:
            if method == "shap":
                # SHAP returns both signed means (for direction) and magnitudes (for ranking)
                signed_means, unsigned_magnitudes = self._compute_shap_importance_with_direction()
                raw_importance = unsigned_magnitudes
                impact_directions = self._compute_impact_direction_from_shap(signed_means)
            elif method == "permutation":
                # Permutation returns only magnitudes, use correlation for direction
                raw_importance = self._compute_permutation_importance()
                impact_directions = self._compute_impact_direction_from_correlation()
            elif method == "gain":
                # Gain returns only magnitudes, use correlation for direction
                if hasattr(self.model.model, "feature_importances_"):
                    raw_importance = self.model.model.feature_importances_
                else:
                    raise ValueError(f"Model doesn't support {method} feature importance.")
                impact_directions = self._compute_impact_direction_from_correlation()
            else:
                raise ValueError(f"Unsupported importance method: {method}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to compute feature importance with method '%s': %s", method, exc)
            return {
                "shap_available": False,
                "total_features": len(self.feature_names),
                "positive_impact_count": 0,
                "negative_impact_count": 0,
                "features": [],
                "computation_method": method,
                "computed_at": pd.Timestamp.utcnow().isoformat(),
            }

        # Build feature importance list
        features = []
        for idx, feature_name in enumerate(self.feature_names):
            importance = float(raw_importance[idx])
            impact_direction = impact_directions[idx]

            features.append(
                {
                    "name": feature_name,
                    "importance": importance,
                    "impact_direction": impact_direction,
                    "rank": 0,  # Will be set after sorting
                }
            )

        # Sort by importance (descending)
        features.sort(key=lambda x: x["importance"], reverse=True)

        # Assign ranks
        for rank, feature in enumerate(features, start=1):
            feature["rank"] = rank

        # Count positive and negative impacts
        positive_count = sum(1 for f in features if f["impact_direction"] == "positive")
        negative_count = sum(1 for f in features if f["impact_direction"] == "negative")

        return {
            "shap_available": method == "shap",
            "total_features": len(features),
            "positive_impact_count": positive_count,
            "negative_impact_count": negative_count,
            "features": features,
            "computation_method": method,
            "computed_at": pd.Timestamp.utcnow().isoformat(),
        }

    def _compute_shap_importance_with_direction(self) -> tuple:
        """
        Compute SHAP-based feature importance with direction.

        Returns:
            tuple: (signed_means, unsigned_magnitudes)
                - signed_means: Mean SHAP values preserving sign (for direction)
                - unsigned_magnitudes: Mean absolute SHAP values (for ranking)
        """
        if self.explainer is None or self.shap_values is None:
            raise ValueError("SHAP values are not available for this model.")

        shap_vals = self._get_shap_values_for_analysis()
        if shap_vals is None:
            raise ValueError("Failed to compute SHAP-based feature importance.")

        # Compute both signed means (for direction) and absolute means (for importance ranking)
        signed_means = shap_vals.mean(axis=0)  # Preserves positive/negative direction
        unsigned_magnitudes = np.abs(shap_vals).mean(axis=0)  # For ranking by importance

        return signed_means, unsigned_magnitudes

    def _compute_permutation_importance(self) -> np.ndarray:
        """Compute permutation-based feature importance."""
        try:
            # Use test data for permutation importance
            X_test = self.X_test if self.X_test is not None else self.X_train
            y_test = self.y_test if self.y_test is not None else self.y_train

            # Determine scoring metric based on model type
            model_type = self.model_info.get("model_type", "classification")
            scoring = "accuracy" if model_type == "classification" else "r2"

            # Compute permutation importance
            perm_importance = permutation_importance(
                self.model.model,
                X_test.values,
                y_test,
                n_repeats=10,
                random_state=42,
                scoring=scoring,
                n_jobs=1,  # Avoid multiprocessing issues
            )

            return perm_importance.importances_mean

        except Exception as exc:
            raise ValueError(f"Failed to compute permutation importance: {str(exc)}") from exc

    def _compute_impact_direction_from_shap(self, signed_means: np.ndarray) -> list:
        """
        Compute impact direction from signed SHAP means.

        Args:
            signed_means: Mean SHAP values preserving sign

        Returns:
            List of direction strings ('positive' or 'negative') for each feature
        """
        directions = []
        for signed_value in signed_means:
            # Positive SHAP mean → feature increases predictions
            # Negative SHAP mean → feature decreases predictions
            if signed_value >= 0:
                directions.append("positive")
            else:
                directions.append("negative")
        return directions

    def _compute_impact_direction_from_correlation(self) -> list:
        """
        Compute impact direction using correlation between features and predictions.

        This is a fallback method for non-SHAP importance methods (gain, permutation)
        that don't provide directional information.

        Returns:
            List of direction strings ('positive' or 'negative') for each feature
        """
        try:
            # Use test data if available, otherwise use training data
            X_data = self.X_test if self.X_test is not None else self.X_train

            # Get model predictions (probabilities for classification)
            if hasattr(self.model.model, "predict_proba"):
                # For binary classification, use probability of positive class
                predictions = self.model.model.predict_proba(X_data.values)
                if predictions.shape[1] == 2:
                    predictions = predictions[:, 1]  # Probability of class 1
                else:
                    # For multi-class, use the max probability
                    predictions = predictions.max(axis=1)
            else:
                # Fallback to predict if predict_proba not available
                predictions = self.model.model.predict(X_data.values)

            directions = []
            for feature_name in self.feature_names:
                try:
                    # Get feature values
                    feature_values = X_data[feature_name].values

                    # Handle constant features or features with no variance
                    if np.std(feature_values) == 0:
                        # Constant feature - no real impact, default to positive
                        directions.append("positive")
                        continue

                    # Compute Pearson correlation coefficient
                    correlation = np.corrcoef(feature_values, predictions)[0, 1]

                    # Handle NaN correlation (e.g., if predictions are constant)
                    if np.isnan(correlation):
                        directions.append("positive")
                    elif correlation >= 0:
                        directions.append("positive")
                    else:
                        directions.append("negative")

                except Exception:  # pylint: disable=broad-except
                    # If correlation fails for any reason, default to positive
                    directions.append("positive")

            return directions

        except Exception:  # pylint: disable=broad-except
            # If entire correlation computation fails, return all positive as safe fallback
            return ["positive"] * len(self.feature_names)

    def _evaluate_thresholds(
        self, performance_metrics: Dict[str, Any], thresholds: Dict[str, float], model_type: str
    ) -> list:
        """
        Evaluate performance metrics against configured thresholds.

        Args:
            performance_metrics: Dict with train/test metrics
            thresholds: Dict with acceptable_threshold, warning_threshold, breach_threshold
            model_type: 'classification' or 'regression'

        Returns:
            List of threshold evaluation results
        """
        acceptable = thresholds.get("acceptable_threshold", 0.70)
        warning = thresholds.get("warning_threshold", 0.50)
        breach = thresholds.get("breach_threshold", 0.30)

        evaluations = []

        # Determine which metrics to evaluate based on model type
        test_metrics = performance_metrics.get("test", {})

        if model_type == "classification":
            # Evaluate classification metrics
            metrics_to_check = ["accuracy", "precision", "recall", "f1_score"]
        else:
            # For regression, we'll evaluate R² score
            metrics_to_check = ["r2_score"]

        for metric_name in metrics_to_check:
            if metric_name not in test_metrics:
                continue

            metric_value = test_metrics[metric_name]

            # Determine status based on threshold ranges
            if metric_value >= acceptable:
                status = "acceptable"
                threshold_used = acceptable
                message = f"{metric_name.replace('_', ' ').title()} ({metric_value:.1%}) meets acceptable threshold (≥{acceptable:.0%})"
            elif metric_value >= warning:
                status = "warning"
                threshold_used = warning
                message = f"{metric_name.replace('_', ' ').title()} ({metric_value:.1%}) is in warning range (≥{warning:.0%} but <{acceptable:.0%})"
            else:
                status = "breach"
                threshold_used = breach
                message = f"{metric_name.replace('_', ' ').title()} ({metric_value:.1%}) is below warning threshold (<{warning:.0%})"

            evaluations.append(
                {
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "status": status,
                    "threshold_used": float(threshold_used),
                    "message": message,
                }
            )

        return evaluations

    def _ensure_loaded(self):
        """Ensure model and data are loaded."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model_and_datasets first.")
        if self.X_train is None or self.X_test is None:
            raise RuntimeError("Datasets not loaded. Call load_model_and_datasets first.")

    def _reset_state(self):
        """Reset service state."""
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = []
        self.target_name = None
        self.explainer = None
        self.shap_values = None
        self.model_info = {}


class LLMBasedExplainabilityAnalysis:
    """LLM-based explainability analysis using Claude via AWS Bedrock."""

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
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Bedrock initialization failed: %s. LLM analysis will be unavailable.", exc)
            self.bedrock_runtime = None

    def _invoke_claude(self, prompt: str) -> str:
        """Invoke Claude via AWS Bedrock."""
        if not self.bedrock_runtime:
            return "LLM analysis unavailable: Bedrock not initialized"

        try:
            import json

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
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Claude invocation failed: %s", exc)
            return f"LLM analysis failed: {str(exc)}"

    def analyze_explainability_results(
        self, performance_metrics: Dict[str, Any], feature_importance: Dict[str, Any], model_type: str
    ) -> Dict[str, str]:
        """
        Generate LLM analysis for explainability results.

        Args:
            performance_metrics: Model performance metrics (train/test/overfitting)
            feature_importance: SHAP feature importance results
            model_type: Type of model ('classification' or 'regression')

        Returns:
            Dictionary with 'what_this_means', 'why_it_matters', and 'risk_signal' sections
        """
        # Build summary for LLM
        summary = self._build_explainability_summary(performance_metrics, feature_importance, model_type)

        prompt = f"""You are an expert in machine learning model explainability and interpretability. Analyze the following model performance and feature importance results:

{summary}

Provide a concise analysis in THREE sections:

1. **What This Means** (2-3 sentences)
   Explain the key findings about model performance and feature importance in plain language. How well is the model performing? What are the most influential features?

2. **Why It Matters** (2-3 sentences)
   Explain the business implications and practical significance of these results. What should stakeholders understand about model reliability and feature impacts?

3. **Risk Signal** (1-2 sentences)
   Assess the overall model quality, overfitting concerns, and provide actionable guidance for improvement or deployment.

Use clear, non-technical language suitable for business stakeholders. Be direct and specific."""

        analysis_text = self._invoke_claude(prompt)

        # Parse the response into structured sections
        return self._parse_analysis(analysis_text, performance_metrics, model_type)

    def _build_explainability_summary(
        self, performance_metrics: Dict[str, Any], feature_importance: Dict[str, Any], model_type: str
    ) -> str:
        """Build a text summary of explainability results for LLM analysis."""
        summary = f"=== Model Type: {model_type.title()} ===\n\n"

        # Performance Metrics
        summary += "=== Performance Metrics ===\n\n"

        train_metrics = performance_metrics.get("train", {})
        test_metrics = performance_metrics.get("test", {})
        overfitting_score = performance_metrics.get("overfitting_score", 0)

        summary += "Training Performance:\n"
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                summary += f"  - {key.replace('_', ' ').title()}: {value:.4f}\n"

        summary += "\nTest Performance:\n"
        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                summary += f"  - {key.replace('_', ' ').title()}: {value:.4f}\n"

        summary += f"\nOverfitting Score: {overfitting_score:.4f}\n"

        # Feature Importance
        summary += "\n=== Feature Importance (SHAP) ===\n\n"

        if feature_importance.get("shap_available"):
            total_features = feature_importance.get("total_features", 0)
            positive_count = feature_importance.get("positive_impact_count", 0)
            negative_count = feature_importance.get("negative_impact_count", 0)

            summary += f"Total Features Analyzed: {total_features}\n"
            summary += f"Positive Impact Features: {positive_count}\n"
            summary += f"Negative Impact Features: {negative_count}\n\n"

            # Top 5 most important features
            features = feature_importance.get("features", [])
            top_features = features[:5]

            summary += "Top 5 Most Important Features:\n"
            for i, feat in enumerate(top_features, 1):
                name = feat.get("name", "Unknown")
                importance = feat.get("importance", 0)
                direction = feat.get("impact_direction", "unknown")
                summary += f"  {i}. {name}: {importance:.6f} ({direction} impact)\n"
        else:
            summary += "SHAP analysis unavailable for this model.\n"

        return summary

    def _parse_analysis(
        self, analysis_text: str, performance_metrics: Dict[str, Any], model_type: str
    ) -> Dict[str, str]:
        """Parse LLM analysis into structured sections."""
        # Default structure
        result = {"what_this_means": "", "why_it_matters": "", "risk_signal": ""}

        # Extract key metrics for fallback messages
        overfitting_score = performance_metrics.get("overfitting_score", 0)
        test_metrics = performance_metrics.get("test", {})

        # Get primary metric based on model type
        if model_type == "classification":
            primary_metric = test_metrics.get("accuracy", 0)
            primary_metric_name = "accuracy"
        else:
            primary_metric = test_metrics.get("r2_score", 0)
            primary_metric_name = "R² score"

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
                f"The {model_type} model achieves a test {primary_metric_name} of {primary_metric:.2f}. "
                f"The overfitting score of {overfitting_score:.2f} indicates "
                f"{'minimal' if overfitting_score < 0.1 else 'moderate' if overfitting_score < 0.2 else 'significant'} "
                f"performance gap between training and test data."
            )

        if not result["why_it_matters"]:
            result["why_it_matters"] = (
                "Model performance metrics help assess reliability for production deployment. "
                "Feature importance reveals which factors most influence predictions, "
                "enabling better understanding of model behavior and potential biases."
            )

        if not result["risk_signal"]:
            if overfitting_score > 0.2:
                result["risk_signal"] = (
                    "High overfitting detected. Model may not generalize well to new data. "
                    "Consider regularization, feature engineering, or collecting more training data."
                )
            elif overfitting_score > 0.1:
                result["risk_signal"] = (
                    "Moderate overfitting observed. Monitor model performance on new data and "
                    "consider additional validation before full deployment."
                )
            else:
                result["risk_signal"] = (
                    "Model shows good generalization with minimal overfitting. "
                    "Performance metrics indicate readiness for deployment with continued monitoring."
                )

        # Clean up extra spaces
        for key in result:
            result[key] = result[key].strip()

        return result
