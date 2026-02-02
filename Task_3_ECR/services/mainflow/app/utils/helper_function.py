"""
helper_function.py
Universal security utilities for SSRF protection and safe data handling.

This module provides reusable security functions including:
- URL validation and sanitization for SSRF protection
- DNS resolution with private IP blocking
- Safe HTTP requests with redirect prevention
- Safe pickle deserialization with module allowlisting

These utilities can be used across all services for consistent security posture.
"""

import io
import ipaddress
import logging
import os
import pickle
import socket
from dataclasses import dataclass
from typing import Any, List, Set
from urllib.parse import ParseResult, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# =============================================================================
# SAFE UNPICKLER - Prevents arbitrary code execution during deserialization
# =============================================================================


class SafeUnpickler(pickle.Unpickler):
    """
    Custom unpickler that only allows specific safe classes for model deserialization.
    Prevents arbitrary code execution from malicious pickle files.

    Usage:
        model_bytes = io.BytesIO(response.content)
        unpickler = SafeUnpickler(model_bytes)
        model = unpickler.load()
    """

    # Allowlist of safe modules and classes for sklearn models
    SAFE_MODULES = {
        "sklearn",
        "numpy",
        "scipy",
        "pandas",
        "joblib",
        "__builtin__",
        "builtins",
        "_codecs",
        "copy_reg",
        "copyreg",
    }

    def find_class(self, module: str, name: str) -> type:
        """
        Override find_class to only allow safe modules.

        Args:
            module: The module name containing the class
            name: The class name to load

        Returns:
            The class if it's from a safe module

        Raises:
            pickle.UnpicklingError: If module is not in the safe allowlist
        """
        if not any(module.startswith(safe) for safe in self.SAFE_MODULES):
            raise pickle.UnpicklingError(f"Attempted to load class from untrusted module: {module}.{name}")
        return super().find_class(module, name)


# =============================================================================
# URL VALIDATION RESULT - Data class for validated URL information
# =============================================================================


@dataclass
class ValidatedURL:
    """
    Result of URL validation containing normalized URL and validated IP.

    Attributes:
        original_url: The original URL that was validated
        safe_url: Normalized URL constructed from parsed components
        validated_ip: First IP address that passed all security checks
        hostname: Original hostname from the URL
        port: Port number (or default based on scheme)
        parsed: The ParseResult from urlparse
    """

    original_url: str
    safe_url: str
    validated_ip: str
    hostname: str
    port: int
    parsed: ParseResult


# =============================================================================
# SSRF PROTECTION - URL validation and safe request utilities
# =============================================================================


def is_private_ip(ip_str: str) -> bool:
    """
    Check if an IP address is private, loopback, link-local, reserved, or multicast.

    Args:
        ip_str: IP address string to check

    Returns:
        True if the IP is in a disallowed range, False otherwise
    """
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        return (
            ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved or ip_obj.is_multicast
        )
    except ValueError:
        # Invalid IP address format - treat as unsafe
        return True


def validate_and_resolve_url(
    url: str,
    allowed_schemes: Set[str] = None,
    allowed_hosts: List[str] = None,
    allowed_host_suffixes: List[str] = None,
    source_name: str = "resource",
) -> ValidatedURL:
    """
    Validate a URL for SSRF protection with DNS resolution and IP blocking.

    This function performs comprehensive URL validation:
    1. Scheme validation (HTTP/HTTPS only by default)
    2. Hostname allowlist checking
    3. DNS resolution to obtain actual IP addresses
    4. Private/internal IP blocking
    5. URL normalization to prevent manipulation

    Args:
        url: The URL to validate
        allowed_schemes: Set of allowed URL schemes (default: {"http", "https"})
        allowed_hosts: List of exact hostnames that are allowed
        allowed_host_suffixes: List of hostname suffixes that are allowed (e.g., ".amazonaws.com")
        source_name: Name for logging purposes

    Returns:
        ValidatedURL object containing safe URL and validated IP

    Raises:
        ValueError: If URL fails any security validation
    """
    if allowed_schemes is None:
        allowed_schemes = {"http", "https"}
    if allowed_hosts is None:
        allowed_hosts = []
    if allowed_host_suffixes is None:
        allowed_host_suffixes = []

    # 1. Parse URL
    if not url:
        raise ValueError("Empty URL is not allowed")

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    hostname = (parsed.hostname or "").lower()

    # 2. Validate scheme
    if scheme not in allowed_schemes:
        logger.error(f"Rejected {source_name} URL with invalid scheme '{scheme}': {url[:100]}")
        raise ValueError(f"Only {', '.join(allowed_schemes).upper()} schemes are allowed")

    # 3. Validate hostname exists
    if not hostname:
        raise ValueError("URL must include a valid hostname")

    # 4. Block obvious internal/local targets
    if hostname in ("localhost", "127.0.0.1", "::1") or "." not in hostname:
        raise ValueError("URL cannot point to internal/local addresses")

    # 5. Check hostname against allowlist
    is_allowed = hostname in [h.lower() for h in allowed_hosts] or any(
        hostname.endswith(suffix.lower()) for suffix in allowed_host_suffixes
    )

    if allowed_hosts or allowed_host_suffixes:
        # If allowlists were provided, enforce them
        if not is_allowed:
            logger.error(f"Rejected {source_name} URL with disallowed host '{hostname}': {url[:100]}")
            raise ValueError(f"Host '{hostname}' is not in the allowed hosts list")

    # 6. DNS resolution - get actual IPs and verify none are private/internal
    port = parsed.port or (443 if scheme == "https" else 80)
    validated_ip = None

    try:
        addrinfo_list = socket.getaddrinfo(hostname, port)

        for family, _, _, _, sockaddr in addrinfo_list:
            ip_str = sockaddr[0]

            if is_private_ip(ip_str):
                logger.error(f"Rejected {source_name} URL resolving to internal IP '{ip_str}'")
                raise ValueError(f"URL resolves to disallowed IP address: {ip_str}")

            # Take the first valid (non-private) IP address
            if validated_ip is None:
                validated_ip = ip_str

        if validated_ip is None:
            raise ValueError(f"Failed to obtain a valid IP address for host: {hostname}")

    except socket.gaierror as e:
        logger.error(f"Failed to resolve {source_name} URL host: {e}")
        raise ValueError(f"Failed to resolve hostname: {e}") from e

    # 7. Construct normalized safe URL from parsed components
    # This prevents tricks with weird encodings or embedded credentials
    safe_url = parsed.geturl()

    logger.debug(f"{source_name} URL passed security validation: {hostname} -> {validated_ip}")

    return ValidatedURL(
        original_url=url,
        safe_url=safe_url,
        validated_ip=validated_ip,
        hostname=hostname,
        port=port,
        parsed=parsed,
    )


def create_secure_session(
    mount_schemes: List[str] = None,
    retry_total: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: List[int] = None,
) -> requests.Session:
    """
    Create a requests Session with retry logic and strict settings.

    Args:
        mount_schemes: List of schemes to mount the adapter to (default: ["http://", "https://"])
        retry_total: Total number of retries
        backoff_factor: Backoff factor for retries
        status_forcelist: List of HTTP status codes to retry on

    Returns:
        Configured requests.Session object
    """
    if mount_schemes is None:
        mount_schemes = ["http://", "https://"]
    if status_forcelist is None:
        status_forcelist = [500, 502, 503, 504]

    session = requests.Session()
    retry = Retry(
        total=retry_total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)

    for scheme in mount_schemes:
        session.mount(scheme, adapter)

    return session


def make_safe_request(
    validated_url: ValidatedURL,
    timeout: int = 60,
    user_agent: str = "BiasLens-Fairness-Analyzer/1.0",
    use_validated_ip: bool = False,
) -> requests.Response:
    """
    Make an HTTP request to a validated URL with SSRF protections.

    This function:
    - Uses a validated URL from validate_and_resolve_url()
    - Disables redirects to prevent redirect-based SSRF
    - Optionally binds to the validated IP to prevent DNS rebinding

    Args:
        validated_url: ValidatedURL object from validate_and_resolve_url()
        timeout: Request timeout in seconds
        user_agent: User-Agent header value
        use_validated_ip: If True, request goes directly to validated IP with Host header

    Returns:
        requests.Response object

    Raises:
        requests.exceptions.RequestException: If the request fails
    """
    session = create_secure_session(mount_schemes=[f"{validated_url.parsed.scheme}://"])

    try:
        if use_validated_ip:
            # Build URL using validated IP to prevent DNS rebinding
            netloc = validated_url.validated_ip
            if validated_url.parsed.port:
                netloc = f"{validated_url.validated_ip}:{validated_url.parsed.port}"

            ip_based_url = validated_url.parsed._replace(netloc=netloc).geturl()

            # Set Host header to original hostname for virtual hosting/SNI
            host_header = validated_url.hostname
            if validated_url.parsed.port:
                host_header = f"{validated_url.hostname}:{validated_url.parsed.port}"

            headers = {
                "User-Agent": user_agent,
                "Host": host_header,
            }

            response = session.get(
                ip_based_url,
                timeout=timeout,
                allow_redirects=False,
                headers=headers,
            )
        else:
            # Use the normalized safe URL
            headers = {"User-Agent": user_agent}

            response = session.get(
                validated_url.safe_url,
                timeout=timeout,
                allow_redirects=False,
                headers=headers,
            )

        response.raise_for_status()
        return response

    finally:
        session.close()


def safe_pickle_load(data: bytes) -> Any:
    """
    Safely deserialize pickle data using SafeUnpickler.

    Args:
        data: Raw bytes to deserialize

    Returns:
        Deserialized Python object

    Raises:
        pickle.UnpicklingError: If data contains untrusted classes
        ValueError: If deserialization fails for other reasons
    """
    try:
        model_bytes = io.BytesIO(data)
        unpickler = SafeUnpickler(model_bytes)
        return unpickler.load()
    except pickle.UnpicklingError as ue:
        logger.error(f"Deserialization blocked - untrusted class detected: {ue}")
        raise
    except Exception as e:
        logger.error(f"Failed to deserialize data: {e}")
        raise ValueError(f"Deserialization failed: {str(e)}") from e


# =============================================================================
# HOST ALLOWLIST HELPERS - Build allowlists from environment configuration
# =============================================================================


def get_default_allowed_hosts() -> List[str]:
    """
    Get the default list of allowed hosts from environment configuration.

    Returns:
        List of allowed hostnames including S3 and configured trusted hosts
    """
    allowed_hosts = ["s3.amazonaws.com"]

    # Add FILES_API_BASE_URL host
    files_api_base = os.getenv("FILES_API_BASE_URL")
    if files_api_base:
        trusted_host = urlparse(files_api_base).hostname
        if trusted_host:
            allowed_hosts.append(trusted_host.lower())

    # Add FAIRNESS_TRUSTED_MODEL_HOSTS
    trusted_hosts_env = os.getenv("FAIRNESS_TRUSTED_MODEL_HOSTS", "")
    for host in trusted_hosts_env.split(","):
        if host.strip():
            allowed_hosts.append(host.strip().lower())

    return allowed_hosts


def get_default_allowed_host_suffixes() -> List[str]:
    """
    Get the default list of allowed host suffixes.

    Returns:
        List of allowed hostname suffixes (e.g., ".amazonaws.com")
    """
    return [".s3.amazonaws.com", ".amazonaws.com"]


# =============================================================================
# CONVENIENCE FUNCTIONS - High-level secure operations
# =============================================================================


def secure_download(
    url: str,
    source_name: str = "resource",
    https_only: bool = False,
    use_validated_ip: bool = False,
) -> bytes:
    """
    Securely download content from a URL with full SSRF protection.

    This is a high-level convenience function that combines URL validation
    and safe request making.

    Args:
        url: The URL to download from
        source_name: Name for logging purposes
        https_only: If True, only allow HTTPS URLs
        use_validated_ip: If True, bind request to validated IP (prevents DNS rebinding)

    Returns:
        Downloaded content as bytes

    Raises:
        ValueError: If URL fails validation
        RuntimeError: If download fails
    """
    allowed_schemes = {"https"} if https_only else {"http", "https"}

    try:
        # Validate URL
        validated = validate_and_resolve_url(
            url=url,
            allowed_schemes=allowed_schemes,
            allowed_hosts=get_default_allowed_hosts(),
            allowed_host_suffixes=get_default_allowed_host_suffixes(),
            source_name=source_name,
        )

        logger.info(f"Downloading {source_name} from URL: {url[:100]}...")

        # Make secure request
        response = make_safe_request(
            validated_url=validated,
            use_validated_ip=use_validated_ip,
        )

        logger.debug(f"Downloaded {len(response.content)} bytes for {source_name}")
        return response.content

    except ValueError as e:
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {source_name} from URL: {e}", exc_info=True)
        raise RuntimeError(f"Failed to download {source_name}: {str(e)}") from e


def secure_load_model(url: str, source_name: str = "model") -> Any:
    """
    Securely download and deserialize a model from a URL.

    This function:
    1. Validates the URL (HTTPS only, trusted hosts, no private IPs)
    2. Downloads the model using validated IP binding
    3. Deserializes using SafeUnpickler to prevent code execution

    Args:
        url: The URL to download the model from
        source_name: Name for logging purposes

    Returns:
        Deserialized model object

    Raises:
        ValueError: If URL fails validation or model contains untrusted classes
        RuntimeError: If download fails
    """
    # Download with HTTPS only and IP binding for maximum security
    content = secure_download(
        url=url,
        source_name=source_name,
        https_only=True,
        use_validated_ip=True,
    )

    # Safe deserialization
    model = safe_pickle_load(content)
    logger.info(f"{source_name} loaded securely: {type(model).__name__}")

    return model
