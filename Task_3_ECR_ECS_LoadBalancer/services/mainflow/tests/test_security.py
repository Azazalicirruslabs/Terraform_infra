"""
Tests for model loading security measures.
"""

import pickle
import tempfile
from pathlib import Path

import pytest

from services.mainflow.app.core.explainability_logic import (
    ExplainabilityService,
    RestrictedUnpickler,
    _validate_model_source,
)


class TestRestrictedUnpickler:
    """Test the RestrictedUnpickler security measures."""

    def test_restricted_unpickler_blocks_dangerous_modules(self):
        """Test that RestrictedUnpickler blocks non-whitelisted modules."""

        # Create a malicious pickle that tries to import os module
        class Malicious:
            def __reduce__(self):
                import os

                return (os.system, ("echo hacked",))

        # Save malicious object
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
            pickle.dump(Malicious(), f)
            malicious_file = f.name

        try:
            # Attempting to load with RestrictedUnpickler should fail
            with open(malicious_file, "rb") as f:
                with pytest.raises(pickle.UnpicklingError) as exc_info:
                    RestrictedUnpickler(f).load()

                assert "not allowed for security reasons" in str(exc_info.value)
        finally:
            Path(malicious_file).unlink(missing_ok=True)

    def test_restricted_unpickler_allows_safe_modules(self):
        """Test that RestrictedUnpickler allows whitelisted ML modules."""
        import numpy as np

        # Create a safe numpy array
        safe_data = {"array": np.array([1, 2, 3]), "value": 42}

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
            pickle.dump(safe_data, f)
            safe_file = f.name

        try:
            # Loading safe data should work
            with open(safe_file, "rb") as f:
                loaded_data = RestrictedUnpickler(f).load()
                assert loaded_data["value"] == 42
                assert len(loaded_data["array"]) == 3
        finally:
            Path(safe_file).unlink(missing_ok=True)


class TestSourceValidation:
    """Test model source validation."""

    def test_validate_model_source_allows_when_no_restrictions(self, monkeypatch):
        """Test that validation passes when no restrictions are configured."""
        monkeypatch.setenv("ALLOWED_MODEL_DIRS", "")
        monkeypatch.setenv("ALLOWED_S3_BUCKETS", "")

        # Should not raise
        _validate_model_source("/any/path/model.pkl")
        _validate_model_source("https://example.com/model.pkl")

    def test_validate_model_source_blocks_untrusted_dir(self, monkeypatch):
        """Test that validation blocks models from untrusted directories."""
        monkeypatch.setenv("ALLOWED_MODEL_DIRS", "/trusted/models")

        with pytest.raises(ValueError) as exc_info:
            _validate_model_source("/untrusted/path/model.pkl")

        assert "not trusted" in str(exc_info.value)

    def test_validate_model_source_allows_trusted_dir(self, monkeypatch, tmp_path):
        """Test that validation allows models from trusted directories."""
        trusted_dir = tmp_path / "trusted_models"
        trusted_dir.mkdir()

        monkeypatch.setenv("ALLOWED_MODEL_DIRS", str(trusted_dir))

        model_path = trusted_dir / "model.pkl"
        model_path.touch()

        # Should not raise
        _validate_model_source(str(model_path))

    def test_validate_model_source_blocks_untrusted_s3_bucket(self, monkeypatch):
        """Test that validation blocks S3 URLs from untrusted buckets."""
        monkeypatch.setenv("ALLOWED_S3_BUCKETS", "trusted-bucket")

        with pytest.raises(ValueError) as exc_info:
            _validate_model_source("https://untrusted-bucket.s3.amazonaws.com/model.pkl")

        assert "not trusted" in str(exc_info.value)

    def test_validate_model_source_allows_trusted_s3_bucket(self, monkeypatch):
        """Test that validation allows S3 URLs from trusted buckets."""
        monkeypatch.setenv("ALLOWED_S3_BUCKETS", "trusted-bucket")

        # Should not raise
        _validate_model_source("https://trusted-bucket.s3.amazonaws.com/model.pkl")


class TestExplainabilityServiceSecurity:
    """Integration tests for ExplainabilityService security."""

    def test_load_model_validates_source(self, monkeypatch):
        """Test that _load_model validates the source before loading."""

        monkeypatch.setenv("ALLOWED_MODEL_DIRS", "/trusted")

        service = ExplainabilityService()

        with pytest.raises(ValueError) as exc_info:
            service._load_model("/untrusted/model.pkl")

        assert "not trusted" in str(exc_info.value)

    def test_load_model_from_url_enforces_size_limit(self, monkeypatch, requests_mock):
        """Test that _load_model_from_url enforces size limits."""

        # Mock a response with large content
        large_content = b"x" * (600 * 1024 * 1024)  # 600 MB
        requests_mock.get(
            "https://example.com/model.pkl", content=large_content, headers={"Content-Length": str(len(large_content))}
        )

        service = ExplainabilityService()

        with pytest.raises(ValueError) as exc_info:
            service._load_model_from_url("https://example.com/model.pkl")

        assert "too large" in str(exc_info.value)


class TestSSRFProtection:
    """Test SSRF protection measures."""

    def test_validate_external_url_blocks_localhost(self):
        """Test that localhost URLs are blocked."""
        service = ExplainabilityService()

        with pytest.raises(ValueError) as exc_info:
            service._validate_external_url("http://localhost/model.pkl")

        assert "loopback" in str(exc_info.value).lower()

    def test_validate_external_url_blocks_127_0_0_1(self):
        """Test that 127.0.0.1 is blocked."""
        service = ExplainabilityService()

        with pytest.raises(ValueError) as exc_info:
            service._validate_external_url("http://127.0.0.1/model.pkl")

        assert "loopback" in str(exc_info.value).lower()

    def test_validate_external_url_blocks_private_ips(self):
        """Test that private IP addresses are blocked."""
        service = ExplainabilityService()

        private_ips = [
            "http://10.0.0.1/model.pkl",
            "http://172.16.0.1/model.pkl",
            "http://192.168.1.1/model.pkl",
        ]

        for url in private_ips:
            with pytest.raises(ValueError) as exc_info:
                service._validate_external_url(url)

            assert "private" in str(exc_info.value).lower()

    def test_validate_external_url_blocks_aws_metadata(self):
        """Test that AWS metadata service IP is blocked."""
        service = ExplainabilityService()

        with pytest.raises(ValueError) as exc_info:
            service._validate_external_url("http://169.254.169.254/latest/meta-data/")

        assert "link-local" in str(exc_info.value).lower()

    def test_validate_external_url_blocks_invalid_scheme(self):
        """Test that non-http(s) schemes are blocked."""
        service = ExplainabilityService()

        invalid_schemes = [
            "file:///etc/passwd",
            "ftp://example.com/model.pkl",
            "gopher://example.com/model.pkl",
        ]

        for url in invalid_schemes:
            with pytest.raises(ValueError) as exc_info:
                service._validate_external_url(url)

            assert "scheme" in str(exc_info.value).lower()

    def test_validate_external_url_allows_public_https(self):
        """Test that public HTTPS URLs are allowed."""
        service = ExplainabilityService()

        # These should not raise (assuming DNS resolution works)
        # Note: In real tests, you might want to mock socket.getaddrinfo
        try:
            service._validate_external_url("https://example.com/model.pkl")
        except ValueError as e:
            # If it fails on DNS resolution or similar, that's acceptable in unit tests
            if "Unable to resolve hostname" not in str(e):
                raise

    def test_load_dataset_validates_url(self, monkeypatch):
        """Test that _load_dataset validates URLs for SSRF."""
        service = ExplainabilityService()

        with pytest.raises(ValueError) as exc_info:
            service._load_dataset("http://localhost/data.csv")

        assert "loopback" in str(exc_info.value).lower() or "not allowed" in str(exc_info.value).lower()

    def test_load_model_from_url_validates_url(self):
        """Test that _load_model_from_url validates URLs for SSRF."""
        service = ExplainabilityService()

        with pytest.raises(ValueError) as exc_info:
            service._load_model_from_url("http://127.0.0.1/model.pkl")

        assert "loopback" in str(exc_info.value).lower() or "not allowed" in str(exc_info.value).lower()
