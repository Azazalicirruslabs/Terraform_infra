import asyncio
import copy
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, Response

# Configure logging - reduced verbosity
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Reduce httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class ServiceConfig:
    """Configuration for a service (auto-discovered or manually added)"""

    name: str
    url: str
    prefix: str
    has_openapi: bool = True
    health_path: Optional[str] = None
    openapi_path: Optional[str] = None
    is_manual: bool = False
    custom_routing: bool = False


class ServiceRegistry:
    """
    Advanced service registry with auto-discovery and manual override capabilities

    Features:
    - Auto-discovers services from *_SERVICE_URL environment variables
    - Manual service addition with custom configurations
    - Mixed mode (auto + manual)
    - Runtime service updates
    - Environment-based overrides
    - Development/production configuration switching
    - Thread-safe operations with asyncio locks
    """

    def __init__(self, auto_discovery_enabled: bool = None):
        self.services: Dict[str, ServiceConfig] = {}
        self._manual_overrides: Dict[str, ServiceConfig] = {}
        self._lock = asyncio.Lock()  # Add thread safety

        # Check if auto-discovery is disabled via environment
        if auto_discovery_enabled is None:
            auto_discovery_enabled = os.getenv("GATEWAY_DISABLE_AUTO_DISCOVERY", "false").lower() != "true"

        self.auto_discovery_enabled = auto_discovery_enabled

        if auto_discovery_enabled:
            self._discover_services()
        else:
            pass  # Manual configuration mode

    def _discover_services(self):
        """Auto-discover services from environment variables"""
        discovered_count = 0
        for env_key, env_value in os.environ.items():
            if env_key.endswith("_SERVICE_URL") and env_value:
                service_name = env_key.replace("_SERVICE_URL", "").lower()

                # Skip if manually overridden
                if service_name in self._manual_overrides:
                    continue

                service_config = self._create_service_config(service_name, env_value)
                self.services[service_name] = service_config
                discovered_count += 1

        # Add fallback defaults if no environment variables found
        if discovered_count == 0:
            self._add_fallback_services()

    def _create_service_config(self, service_name: str, service_url: str, is_manual: bool = False) -> ServiceConfig:
        """Create service configuration with intelligent defaults"""
        # Smart defaults based on service name
        is_frontend = service_name == "frontend"
        prefix = "" if is_frontend else service_name
        has_openapi = not is_frontend

        # Convention-based paths
        health_path = f"/{service_name}/health" if not is_frontend else "/health"
        openapi_path = f"/{service_name}/openapi.json" if has_openapi else None

        return ServiceConfig(
            name=service_name,
            url=service_url,
            prefix=prefix,
            has_openapi=has_openapi,
            health_path=health_path,
            openapi_path=openapi_path,
            is_manual=is_manual,
        )

    def _add_fallback_services(self):
        """Add fallback service configurations when no environment variables exist"""
        # Core project services only (no external/testing services)
        fallback_services = {
            "api": "https://raia-api-dev-1020052770212.europe-west1.run.app",
            "fairness": "https://raia-fairness-dev-1020052770212.europe-west1.run.app",
            "classification": "https://raia-classification-dev-1020052770212.europe-west1.run.app",
            "regression": "https://raia-regression-dev-1020052770212.europe-west1.run.app",
            "data_drift": "https://raia-datadrift-dev-1020052770212.europe-west1.run.app",
            "frontend": "https://raia-frontend-dev-1020052770212.europe-west1.run.app",
        }

        for name, url in fallback_services.items():
            if name not in self.services:
                self.services[name] = self._create_service_config(name, url)

    async def add_manual_service(
        self,
        name: str,
        url: str,
        prefix: str = None,
        has_openapi: bool = True,
        health_path: str = None,
        openapi_path: str = None,
        full_control: bool = False,
        **kwargs,
    ):
        """
        Manually add a service (overrides auto-discovery) with full developer control

        Args:
            name: Service name
            url: Service URL
            prefix: URL prefix (full control - can be different from service name)
            has_openapi: Whether service has OpenAPI spec
            health_path: Custom health endpoint path
            openapi_path: Custom OpenAPI spec path
            full_control: If True, no automatic path generation - use only what's provided
        """
        if not full_control:
            # Original behavior for core services - smart defaults
            if prefix is None:
                prefix = name if name != "frontend" else ""

            if health_path is None:
                health_path = f"/{prefix}/health" if prefix else "/health"

            if openapi_path is None and has_openapi:
                openapi_path = f"/{prefix}/openapi.json" if prefix else "/openapi.json"
        # else: full_control=True means use exactly what was provided, no automatic generation

        # Clean URL (remove trailing slashes)
        clean_url = url.rstrip("/")

        service_config = ServiceConfig(
            name=name,
            url=clean_url,
            prefix=prefix,  # Exactly what you provide when full_control=True
            has_openapi=has_openapi,
            health_path=health_path,  # Exactly what you provide when full_control=True
            openapi_path=openapi_path,  # Exactly what you provide when full_control=True
            is_manual=True,
            **kwargs,
        )

        # Thread-safe updates with lock
        async with self._lock:
            self.services[name] = service_config
            self._manual_overrides[name] = service_config

            # Update global SERVICES dict for immediate availability
            global SERVICES
            SERVICES[name] = clean_url

        # Service added (reduced logging)

        return service_config

    async def update_service_url(self, name: str, new_url: str):
        """Update service URL at runtime with thread safety"""
        async with self._lock:
            if name in self.services:
                self.services[name].url
                clean_url = new_url.rstrip("/")
                self.services[name].url = clean_url

                # Update global SERVICES dict for immediate availability
                global SERVICES
                SERVICES[name] = clean_url

                return True  # Updated successfully
            else:
                return False  # Service not found

    async def update_service_config(self, name: str, **kwargs):
        """Update any service configuration at runtime with full developer control and thread safety"""
        async with self._lock:
            if name in self.services:
                service = self.services[name]

                # Update any provided fields
                if "url" in kwargs:
                    service.url = kwargs["url"].rstrip("/")
                    # Update global SERVICES dict
                    global SERVICES
                    SERVICES[name] = service.url
                if "prefix" in kwargs:
                    service.prefix = kwargs["prefix"]
                if "has_openapi" in kwargs:
                    service.has_openapi = kwargs["has_openapi"]
                if "health_path" in kwargs:
                    service.health_path = kwargs["health_path"]
                if "openapi_path" in kwargs:
                    service.openapi_path = kwargs["openapi_path"]

                return service
            else:
                return None

    def remove_service(self, name: str):
        """Remove a service from registry"""
        if name in self.services:
            removed_service = self.services.pop(name)
            self._manual_overrides.pop(name, None)
            return removed_service
        else:
            return None

    def get_service(self, name: str) -> Optional[ServiceConfig]:
        """Get service configuration by name"""
        return self.services.get(name)

    def get_service_url(self, name: str) -> Optional[str]:
        """Get service URL by name"""
        service = self.services.get(name)
        return service.url if service else None

    def get_all_services(self) -> Dict[str, ServiceConfig]:
        """Get all registered services"""
        return self.services.copy()

    def get_api_services(self) -> Dict[str, ServiceConfig]:
        """Get only API services (exclude frontend)"""
        return {name: config for name, config in self.services.items() if config.has_openapi}

    def get_service_stats(self) -> Dict:
        """Get registry statistics"""
        auto_count = sum(1 for s in self.services.values() if not s.is_manual)
        manual_count = sum(1 for s in self.services.values() if s.is_manual)

        return {
            "total_services": len(self.services),
            "auto_discovered": auto_count,
            "manually_added": manual_count,
            "auto_discovery_enabled": self.auto_discovery_enabled,
            "services": list(self.services.keys()),
        }


# Global service registry instance (with manual override support)
service_registry = ServiceRegistry()

# Legacy SERVICES dict for backward compatibility (now auto-generated)
SERVICES = {service.name: service.url for service in service_registry.get_all_services().values()}

# Service registry initialized


# Simple in-memory cache for OpenAPI specs
class SimpleCache:
    def __init__(self, default_ttl: int = 300):  # 5 minutes default TTL
        self._cache = {}
        self._timestamps = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Dict]:
        """Get cached value if still valid"""
        if key not in self._cache:
            return None

        # Check if expired
        if self._is_expired(key):
            self._remove(key)
            return None

        return self._cache[key]

    def set(self, key: str, value: Dict, ttl: Optional[int] = None):
        """Set cached value with TTL"""
        self._cache[key] = value
        self._timestamps[key] = {"created": datetime.now(), "ttl": ttl or self.default_ttl}

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self._timestamps:
            return True

        timestamp_info = self._timestamps[key]
        elapsed = (datetime.now() - timestamp_info["created"]).total_seconds()
        return elapsed > timestamp_info["ttl"]

    def _remove(self, key: str):
        """Remove cache entry"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)

    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        self._timestamps.clear()

    def list_keys(self) -> List[str]:
        """Get list of all cache keys (public method for encapsulation)"""
        return list(self._cache.keys())

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_entries = len(self._cache)
        expired_entries = sum(1 for key in self._cache.keys() if self._is_expired(key))

        return {
            "total_entries": total_entries,
            "valid_entries": total_entries - expired_entries,
            "expired_entries": expired_entries,
            "cache_hit_keys": self.list_keys(),  # Use public method
        }


# Global cache instance
openapi_cache = SimpleCache(default_ttl=300)  # 5 minutes cache


# Simple health tracking
class ServiceHealthTracker:
    def __init__(self):
        self._health_data = {}
        self._request_counts = {}
        self._last_check = {}

    def record_health(self, service_name: str, is_healthy: bool, response_time_ms: float, error: str = None):
        """Record health check result"""
        now = datetime.now()

        if service_name not in self._health_data:
            self._health_data[service_name] = {
                "success_count": 0,
                "failure_count": 0,
                "avg_response_time": 0,
                "last_error": None,
                "status": "unknown",
            }

        health = self._health_data[service_name]

        if is_healthy:
            health["success_count"] += 1
            health["status"] = "healthy"
            health["last_error"] = None
        else:
            health["failure_count"] += 1
            health["status"] = "unhealthy"
            health["last_error"] = error

        # Update average response time (simple moving average)
        total_requests = health["success_count"] + health["failure_count"]
        if total_requests == 1:
            health["avg_response_time"] = response_time_ms
        else:
            health["avg_response_time"] = (
                health["avg_response_time"] * (total_requests - 1) + response_time_ms
            ) / total_requests

        self._last_check[service_name] = now

    def record_request(self, service_name: str):
        """Record a request to a service"""
        self._request_counts[service_name] = self._request_counts.get(service_name, 0) + 1

    def get_health_summary(self) -> Dict:
        """Get health summary for all services"""
        return {
            service_name: {
                **health_data,
                "total_requests": self._request_counts.get(service_name, 0),
                "last_check": self._last_check.get(service_name, datetime.now()).isoformat(),
                "failure_rate": (
                    health_data["failure_count"] / max(health_data["success_count"] + health_data["failure_count"], 1)
                )
                * 100,
            }
            for service_name, health_data in self._health_data.items()
        }

    def is_healthy(self, service_name: str) -> bool:
        """Check if service is considered healthy (Cloud Run optimized)"""
        if service_name not in self._health_data:
            return True  # Assume healthy until proven otherwise

        health = self._health_data[service_name]
        total_checks = health["success_count"] + health["failure_count"]

        if total_checks < 5:  # Increased threshold for Cloud Run cold starts
            return True

        failure_rate = health["failure_count"] / total_checks
        # More lenient threshold for Cloud Run 0-2 scaling (cold starts expected)
        return failure_rate < 0.7  # Consider unhealthy if >70% failure rate


# Global health tracker
health_tracker = ServiceHealthTracker()

app = FastAPI(
    title="XAI Platform - API Gateway",
    description="Unified API Gateway for XAI Platform microservices - Optimized for ML workloads",
    version="1.0.0",
    docs_url=None,  # We'll provide custom docs endpoint
    redoc_url="/redoc",
    openapi_url=None,  # Disable default openapi, we'll provide our own
)

# CORS middleware with ML-specific configurations
# Environment-driven CORS origins for security
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Environment-driven for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"],  # Useful for ML debugging
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Service URLs are now auto-managed by ServiceRegistry above
# (Old manual SERVICES dict replaced by auto-discovery)


async def fetch_service_openapi(service_name: str, service_url: str) -> Dict:
    """Fetch OpenAPI specification from a service with caching and error handling"""
    # Use service registry to get configuration
    service_config = service_registry.get_service(service_name)

    # Skip services without OpenAPI (e.g., frontend)
    if not service_config or not service_config.has_openapi:
        # Service doesn't have OpenAPI spec
        return {}

    # Check cache first
    cache_key = f"openapi_{service_name}_{service_url}"
    cached_spec = openapi_cache.get(cache_key)
    if cached_spec:
        return cached_spec

    # Fetch from service with retry logic optimized for Cloud Run cold starts
    openapi_path = service_config.openapi_path or "/openapi.json"
    openapi_url = f"{service_url}{openapi_path}"

    max_retries = 3  # Increased for cold starts
    for attempt in range(max_retries):
        try:
            # Extended timeouts for Cloud Run cold starts (0-2 instances)
            base_timeout = 45.0 if attempt == 0 else 60.0  # First attempt longer for cold start
            timeout = httpx.Timeout(base_timeout + (attempt * 15))  # Progressive timeout
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(openapi_url)
                if response.status_code == 200:
                    spec = response.json()
                    # Cache successful result
                    openapi_cache.set(cache_key, spec)
                    return spec

        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff

    # Return fallback spec if all retries failed
    fallback_spec = _get_fallback_spec(service_name)
    # Cache fallback with shorter TTL
    openapi_cache.set(cache_key, fallback_spec, ttl=60)  # 1 minute for fallbacks
    return fallback_spec


def _get_fallback_spec(service_name: str) -> Dict:
    """Return a basic fallback spec when service is unavailable"""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": f"{service_name.title()} Service (Unavailable)",
            "version": "0.0.0",
            "description": f"Service {service_name} is currently unavailable. This is a fallback specification.",
        },
        "paths": {
            f"/{service_name}/health": {
                "get": {
                    "summary": f"{service_name.title()} Health Check (Unavailable)",
                    "description": f"This endpoint is currently unavailable as {service_name} service is unreachable",
                    "tags": [service_name.title()],
                    "responses": {
                        "503": {
                            "description": "Service Unavailable",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                            "service": {"type": "string"},
                                            "status": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            }
        },
    }


def update_schema_refs(obj: Any, service_name: str) -> Any:
    """
    Recursively update schema references to use prefixed schema names.

    When merging OpenAPI specs from multiple services, we prefix schema names
    to avoid conflicts (e.g., Body_login -> ApiBody_login). However, the $ref
    pointers in paths still reference the original names, causing resolution errors.
    This function updates all $ref pointers to use the prefixed schema names.

    This handles nested references in:
    - Direct $ref properties
    - Array items with $ref
    - Nested objects with $ref at any level
    - Properties, additionalProperties, etc.
    - Schema title fields that reference common schema names
    - Default values for fields with specific patterns
    - Security scheme references (OAuth2PasswordBearer -> unified scheme)

    Args:
        obj: The OpenAPI object (dict, list, or primitive) to process
        service_name: The service name to use as prefix (will be title-cased)

    Returns:
        Updated object with corrected schema references
    """
    if isinstance(obj, dict):
        updated_obj = {}
        for key, value in obj.items():
            if key == "$ref" and isinstance(value, str) and value.startswith("#/components/schemas/"):
                # Extract schema name and prefix it
                schema_name = value.replace("#/components/schemas/", "")
                prefixed_schema_name = f"{service_name.title()}{schema_name}"
                updated_obj[key] = f"#/components/schemas/{prefixed_schema_name}"
            elif key == "title" and isinstance(value, str) and value in ["ValidationError", "HTTPValidationError"]:
                # Update common schema titles that might cause conflicts
                prefixed_title = f"{service_name.title()}{value}"
                updated_obj[key] = prefixed_title
            elif key == "security" and isinstance(value, list):
                # Update security references to use unified OAuth2 scheme
                updated_security = []
                for security_item in value:
                    if isinstance(security_item, dict):
                        updated_security_item = {}
                        for sec_scheme_name, sec_scopes in security_item.items():
                            # Convert all OAuth2 scheme references to unified scheme
                            if "oauth2" in sec_scheme_name.lower() or "bearer" in sec_scheme_name.lower():
                                updated_security_item["OAuth2PasswordBearer"] = sec_scopes
                            else:
                                # Keep non-OAuth2 schemes with service prefix
                                prefixed_sec_name = (
                                    f"{service_name.title()}{sec_scheme_name}"
                                    if service_name != "api"
                                    else sec_scheme_name
                                )
                                updated_security_item[prefixed_sec_name] = sec_scopes
                        updated_security.append(updated_security_item)
                    else:
                        updated_security.append(security_item)
                updated_obj[key] = updated_security
            else:
                # Recursively process all nested values
                updated_obj[key] = update_schema_refs(value, service_name)
        return updated_obj
    elif isinstance(obj, list):
        # Process each item in the list
        return [update_schema_refs(item, service_name) for item in obj]
    else:
        # Return primitives unchanged
        return obj


def enhance_schema_defaults(schema_obj: Any) -> Any:
    """
    Enhance schema objects by adding default values for fields with specific patterns.
    This improves the Swagger UI experience by pre-filling common fields.
    """
    if isinstance(schema_obj, dict):
        updated_obj = {}
        for key, value in schema_obj.items():
            if key == "properties" and isinstance(value, dict):
                # Process properties to add defaults
                updated_properties = {}
                for prop_name, prop_def in value.items():
                    updated_prop_def = enhance_schema_defaults(prop_def)

                    # Add default values for common patterns
                    if prop_name == "grant_type" and isinstance(prop_def, dict):
                        # Handle anyOf structure for grant_type
                        if "anyOf" in prop_def:
                            for any_of_item in prop_def["anyOf"]:
                                if (
                                    isinstance(any_of_item, dict)
                                    and any_of_item.get("type") == "string"
                                    and any_of_item.get("pattern") == "^password$"
                                ):
                                    # Add default to the root level
                                    updated_prop_def["default"] = "password"
                                    break
                        # Handle direct pattern
                        elif (
                            "pattern" in prop_def
                            and prop_def.get("pattern") == "^password$"
                            and "default" not in prop_def
                        ):
                            updated_prop_def["default"] = "password"

                    updated_properties[prop_name] = updated_prop_def
                updated_obj[key] = updated_properties
            else:
                updated_obj[key] = enhance_schema_defaults(value)
        return updated_obj
    elif isinstance(schema_obj, list):
        return [enhance_schema_defaults(item) for item in schema_obj]
    else:
        return schema_obj


def merge_openapi_specs(specs: Dict[str, Dict]) -> Dict:
    """Merge multiple OpenAPI specifications into a single spec"""
    merged_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "RAIA Platform - Responsible AI Analytics & Explainability Suite",
            "description": "Unified API gateway providing comprehensive AI/ML explainability, fairness analysis, and monitoring capabilities across multiple integrated microservices.",
            "version": "2.0.0",
            "contact": {"name": "RAIA Platform Support", "url": "https://github.com/cirruslabs-io/XAI-Explainability"},
        },
        "servers": [{"url": "/", "description": "RAIA Platform Gateway"}],
        "paths": {},
        "components": {"schemas": {}, "securitySchemes": {}},
        "tags": [],
    }

    service_tags = []
    total_paths_merged = 0

    # Add Gateway tags first for better organization
    gateway_tag = {"name": "Gateway", "description": "Core gateway management and health monitoring endpoints"}
    gateway_debug_tag = {
        "name": "Gateway Debug",
        "description": "Essential production debugging and diagnostics - minimal set for troubleshooting",
    }
    health_monitoring_tag = {
        "name": "Health Monitoring",
        "description": "Health check endpoints for all services - quick status overview",
    }

    service_tags.extend([gateway_tag, gateway_debug_tag, health_monitoring_tag])

    for service_name, spec in specs.items():
        if not spec or "paths" not in spec:
            logger.warning(f"Skipping {service_name}: no paths in spec")
            continue

        service_paths = spec.get("paths", {})

        # Add service tag with descriptive information
        service_descriptions = {
            "api": "Core API - User management, authentication, file handling, and project operations",
            "fairness": "AI Fairness Analysis - Bias detection, mitigation strategies, and fairness metrics",
            "classification": "Classification Analysis - Model explainability, feature importance, and decision insights",
            "regression": "Regression Analysis - Model interpretation, feature relationships, and prediction explanations",
            "data_drift": "Data Drift Detection - Model performance monitoring, statistical analysis, and data quality checks",
        }

        if service_name != "gateway":
            service_tag = {
                "name": service_name.title(),
                "description": service_descriptions.get(service_name, f"{service_name.title()} Service API endpoints"),
            }
            service_tags.append(service_tag)

        # Merge paths - keep original paths as they already have service prefixes
        for path, path_spec in service_paths.items():
            # Processing path

            # Don't double-prefix paths that already have the service name
            final_path = path

            # Deep copy path_spec to avoid modifying original
            updated_path_spec = copy.deepcopy(path_spec)

            # Add service tag to all operations and update schema references
            for method, operation in updated_path_spec.items():
                if isinstance(operation, dict):
                    # Check if this is a health endpoint
                    is_health_endpoint = (
                        path.endswith("/health")
                        or "health" in operation.get("operationId", "").lower()
                        or "health" in operation.get("summary", "").lower()
                    )

                    # Check if this is a debug endpoint
                    is_debug_endpoint = (
                        "debug" in path.lower()
                        or "debug" in operation.get("operationId", "").lower()
                        or "debug" in operation.get("summary", "").lower()
                    )

                    # Replace tags completely to avoid duplicates
                    if is_health_endpoint:
                        # Gateway health endpoint stays in Gateway tag, others go to Health Monitoring
                        if path == "/gateway/health":
                            operation["tags"] = ["Gateway"]
                        else:
                            # All other health endpoints go to Health Monitoring
                            operation["tags"] = ["Health Monitoring"]
                    elif is_debug_endpoint and service_name == "gateway":
                        # Gateway debug endpoints get Gateway Debug tag
                        operation["tags"] = ["Gateway Debug"]
                    elif service_name == "gateway":
                        # Other gateway endpoints get Gateway tag
                        operation["tags"] = ["Gateway"]
                    else:
                        # All other service endpoints get their service tag
                        operation["tags"] = [service_name.title()]

                    # Update schema references in operation
                    updated_path_spec[method] = update_schema_refs(operation, service_name)

            merged_spec["paths"][final_path] = updated_path_spec
            total_paths_merged += 1

        # Merge components/schemas
        if "components" in spec and "schemas" in spec["components"]:
            for schema_name, schema_spec in spec["components"]["schemas"].items():
                prefixed_schema_name = f"{service_name.title()}{schema_name}"
                # Also update any schema references within the schema definitions themselves
                updated_schema_spec = update_schema_refs(schema_spec, service_name)
                # Enhance schema with default values for better UX
                enhanced_schema_spec = enhance_schema_defaults(updated_schema_spec)
                merged_spec["components"]["schemas"][prefixed_schema_name] = enhanced_schema_spec

        # Merge security schemes (consolidate OAuth2 schemes)
        oauth2_scheme_added = False
        if "components" in spec and "securitySchemes" in spec["components"]:
            for scheme_name, scheme_spec in spec["components"]["securitySchemes"].items():
                # Update OAuth2 tokenUrl to work with gateway
                if isinstance(scheme_spec, dict) and scheme_spec.get("type") == "oauth2":
                    if not oauth2_scheme_added:
                        # Only add one unified OAuth2 scheme
                        updated_scheme = scheme_spec.copy()
                        flows = updated_scheme.get("flows", {})
                        if "password" in flows:
                            password_flow = flows["password"].copy()
                            # Update tokenUrl to use gateway URL
                            if "tokenUrl" in password_flow:
                                password_flow["tokenUrl"]
                                password_flow["tokenUrl"] = "/api/login"
                            # Updated OAuth2 tokenUrl
                            flows["password"] = password_flow
                            updated_scheme["flows"] = flows

                        # Use a single unified scheme name
                        merged_spec["components"]["securitySchemes"]["OAuth2PasswordBearer"] = updated_scheme
                        oauth2_scheme_added = True
                # Added unified OAuth2 scheme
                else:
                    # For non-OAuth2 schemes, prefix with service name to avoid conflicts
                    prefixed_scheme_name = f"{service_name.title()}{scheme_name}"
                    merged_spec["components"]["securitySchemes"][prefixed_scheme_name] = scheme_spec

    merged_spec["tags"] = service_tags
    return merged_spec


@app.get("/openapi.json", include_in_schema=False)
async def get_unified_openapi():
    """Get unified OpenAPI specification for all services with caching"""
    # Check for cached unified spec
    unified_cache_key = f"unified_openapi_{hash(str(SERVICES))}"
    cached_unified = openapi_cache.get(unified_cache_key)
    if cached_unified:
        return cached_unified
    service_specs = {}

    # Fetch OpenAPI specs from API services only (exclude frontend)
    tasks = []
    api_services = []

    for service_name, service_url in SERVICES.items():
        # Skip frontend and placeholder URLs (only skip if it's exactly the placeholder)
        if service_name == "frontend" or "your-hash-ew.a.run.app" in service_url:
            continue
        tasks.append(fetch_service_openapi(service_name, service_url))
        api_services.append(service_name)

    specs = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results with better error handling
    for i, spec in enumerate(specs):
        if i < len(api_services):
            service_name = api_services[i]
            if isinstance(spec, Exception):
                logger.error(f"Error fetching OpenAPI from {service_name}: {spec}")
                # Use fallback spec for failed services
                service_specs[service_name] = _get_fallback_spec(service_name)
            elif spec:
                service_specs[service_name] = spec
            else:
                logger.warning(f"Empty OpenAPI spec from {service_name}, using fallback")
                service_specs[service_name] = _get_fallback_spec(service_name)

    # 🚀 AUTO-DISCOVER GATEWAY ENDPOINTS (No more manual maintenance!)
    def auto_discover_gateway_endpoints():
        """Automatically discover and extract gateway endpoints from FastAPI app"""
        # Get FastAPI's built-in OpenAPI spec for this app
        temp_openapi = app.openapi()
        gateway_paths = {}

        # Extract only gateway endpoints
        for path, path_spec in temp_openapi.get("paths", {}).items():
            if path.startswith("/gateway/"):
                # Update tags for better organization
                for method, operation in path_spec.items():
                    if isinstance(operation, dict):
                        # Categorize gateway endpoints
                        if "debug" in path.lower():
                            operation["tags"] = ["Gateway Debug"]
                        else:
                            operation["tags"] = ["Gateway"]

                gateway_paths[path] = path_spec

        return {
            "paths": gateway_paths,
            "info": {
                "title": "Gateway Service (Auto-Discovered)",
                "version": "1.0.0",
                "description": f"API Gateway management endpoints - {len(gateway_paths)} endpoints auto-discovered",
            },
            # Include components if they exist
            "components": temp_openapi.get("components", {}),
        }

    # Use auto-discovery instead of manual maintenance
    gateway_spec = auto_discover_gateway_endpoints()
    service_specs["gateway"] = gateway_spec

    unified_spec = merge_openapi_specs(service_specs)

    # Dynamically calculate total endpoints and update the title
    total_endpoints = len(unified_spec.get("paths", {}))
    service_count = len([name for name in service_specs.keys() if name != "frontend"])

    # Update the unified spec info with dynamic counts
    if "info" in unified_spec:
        unified_spec["info"][
            "title"
        ] = f"RAIA Platform - Responsible AI Analytics & Explainability Suite ({total_endpoints} Endpoints)"
        unified_spec["info"][
            "description"
        ] = f"Unified API gateway for RAIA platform with {total_endpoints} endpoints across {service_count} microservices. Provides AI/ML explainability, fairness analysis, classification, regression, and data drift monitoring capabilities."

    # Cache the unified spec
    openapi_cache.set(unified_cache_key, unified_spec)
    return unified_spec


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI that uses our unified OpenAPI spec"""

    # Get the unified spec to extract dynamic title
    try:
        unified_spec = await get_unified_openapi()
        dynamic_title = unified_spec.get("info", {}).get("title", "RAIA Platform - Responsible AI Analytics Suite")
    except Exception as e:
        logger.error(f"Error getting unified spec for title: {e}")
        dynamic_title = "RAIA Platform - Responsible AI Analytics & Explainability Suite"

    return get_swagger_ui_html(openapi_url="/openapi.json", title=dynamic_title)


async def proxy_request(target_url: str, request: Request, service_name: str = None) -> Any:
    """Proxy HTTP request to target service with request tracking optimized for Cloud Run and ML workloads"""
    # Track request if service name is provided
    if service_name:
        health_tracker.record_request(service_name)

    try:
        # Enhanced connection limits for high concurrency ML workloads
        limits = httpx.Limits(
            max_keepalive_connections=20,  # Keep connections alive for ML services
            max_connections=100,  # Support high concurrency
            keepalive_expiry=30.0,  # Keep connections for 30 seconds
        )

        timeout_config = httpx.Timeout(
            connect=30.0,  # Connection timeout
            read=120.0,  # ML inference timeout
            write=30.0,  # Upload timeout for large datasets
            pool=10.0,  # Pool acquisition timeout
        )

        async with httpx.AsyncClient(timeout=timeout_config, limits=limits, max_redirects=3) as client:
            # Prepare headers
            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("content-length", None)
            headers.pop("connection", None)

            # Make request to target service
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=await request.body(),
                params=request.query_params,
            )

            # Prepare response headers
            response_headers = dict(response.headers)
            response_headers.pop("content-length", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("connection", None)
            response_headers.pop("server", None)

            # Handle different content types
            content_type = response.headers.get("content-type", "")

            if response.status_code == 204 or len(response.content) == 0:
                return JSONResponse(content=None, status_code=response.status_code, headers=response_headers)
            elif content_type.startswith("application/json"):
                try:
                    return JSONResponse(
                        content=response.json(), status_code=response.status_code, headers=response_headers
                    )
                except Exception as e:
                    # Failed to parse JSON response
                    return JSONResponse(
                        content={"data": response.text}, status_code=response.status_code, headers=response_headers
                    )
            else:
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=content_type,
                )

    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=502, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def check_service_health(service_name: str, service_url: str) -> dict:
    """Check health of individual service with tracking optimized for Cloud Run scaling"""
    # Use service registry to get health endpoint
    service_config = service_registry.get_service(service_name)

    start_time = time.time()

    try:
        health_path = service_config.health_path if service_config else "/health"
        health_url = f"{service_url}{health_path}"

        # Extended timeout for Cloud Run cold starts (0-2 instance scaling)
        cold_start_timeout = 30.0  # Allow time for model loading on cold start
        async with httpx.AsyncClient(timeout=cold_start_timeout) as client:
            response = await client.get(health_url)
            response_time_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                # Record successful health check
                health_tracker.record_health(service_name, True, response_time_ms)

                try:
                    health_data = response.json()
                    return {
                        "service": service_name,
                        "status": "healthy",
                        "response_time_ms": response_time_ms,
                        "details": health_data,
                    }
                except Exception as e:
                    logger.debug(f"Failed to parse health data: {e}")
                    return {"service": service_name, "status": "healthy", "response_time_ms": response_time_ms}
            else:
                # Record failed health check
                error_msg = f"HTTP {response.status_code}"
                health_tracker.record_health(service_name, False, response_time_ms, error_msg)

                return {
                    "service": service_name,
                    "status": "unhealthy",
                    "error": error_msg,
                    "response_time_ms": response_time_ms,
                }

    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        error_msg = str(e)

        # Record failed health check
        health_tracker.record_health(service_name, False, response_time_ms, error_msg)

        return {
            "service": service_name,
            "status": "unreachable",
            "error": error_msg,
            "response_time_ms": response_time_ms,
        }


@app.get("/gateway/health", tags=["Gateway"])
async def health_check():
    """Gateway health check endpoint"""
    start_time = time.time()

    # Check all services concurrently
    service_checks = []

    for service_name, service_url in SERVICES.items():
        if service_url and not service_url.endswith("your-hash-ew.a.run.app"):
            service_checks.append(check_service_health(service_name, service_url))

    service_statuses = await asyncio.gather(*service_checks, return_exceptions=True)

    # Process results
    healthy_services = 0
    total_services = len(service_statuses)
    services_health = []

    for result in service_statuses:
        if isinstance(result, Exception):
            services_health.append({"service": "unknown", "status": "error", "error": str(result)})
        else:
            services_health.append(result)
            if result.get("status") == "healthy":
                healthy_services += 1

    overall_status = "healthy" if healthy_services == total_services else "degraded"
    if healthy_services == 0:
        overall_status = "unhealthy"

    end_time = time.time()

    return {
        "gateway": {
            "status": overall_status,
            "version": "1.0.0",
            "check_duration_ms": round((end_time - start_time) * 1000, 2),
        },
        "services": {
            "total": total_services,
            "healthy": healthy_services,
            "unhealthy": total_services - healthy_services,
            "details": services_health,
        },
    }


# 📊 Service registry status endpoint
@app.get("/gateway/registry/status", tags=["Gateway"])
async def get_registry_status():
    """
    Get comprehensive service registry status including:
    - Service statistics and management system info
    - Detailed service configurations (URL, prefix, health path, OpenAPI path)
    - Discovery method for each service
    """
    stats = service_registry.get_service_stats()

    # Get detailed service information
    services_detail = []
    for service_name, service_config in service_registry.get_all_services().items():
        services_detail.append(
            {
                "name": service_name,
                "url": service_config.url,
                "prefix": service_config.prefix,
                "health_path": service_config.health_path,
                "openapi_path": service_config.openapi_path,
                "has_openapi": service_config.has_openapi,
                "discovery_method": "manual" if service_config.is_manual else "auto-discovery",
            }
        )

    return {
        "status": "active",
        "registry": "operational",
        "management_system": "3_essential_ways",
        "service_counts": {
            "total_services": stats["total_services"],
            "auto_discovered": stats["auto_discovered"],
            "manually_managed": stats["manually_added"],
            "auto_discovery_enabled": stats["auto_discovery_enabled"],
        },
        "services": services_detail,  # Enhanced with detailed configurations
        "management_methods": {
            "1_environment_variables": "export *_SERVICE_URL for deployment automation",
            "2_fallback_defaults": "automatic core services when no env vars found",
            "3_unified_rest_api": "POST/PUT /gateway/registry/services/{name} with full control",
        },
    }


@app.get("/gateway/service-management-guide", tags=["Gateway"])
async def service_management_guide():
    """
    Complete guide to the 3 Essential Ways of Service Management with Full Developer Control
    """
    return {
        "gateway_service_management": "3 Essential Ways with Full Developer Control",
        "simplified_from": "5 ways (removed redundancy)",
        "method_1_environment_variables": {
            "purpose": "Deployment automation and DevOps workflows",
            "usage": "export API_SERVICE_URL='https://api.example.com'",
            "automatic": True,
            "when_to_use": "Production deployments, CI/CD pipelines, container orchestration",
        },
        "method_2_fallback_defaults": {
            "purpose": "Development and default setup",
            "automatic": True,
            "when_used": "When no environment variables are found",
            "services_included": ["api", "fairness", "classification", "regression", "data_drift", "frontend"],
        },
        "method_3_unified_rest_api": {
            "purpose": "Runtime management with full developer control",
            "endpoints": {
                "add_or_update": "POST/PUT /gateway/registry/services/{service_name}",
                "remove": "DELETE /gateway/registry/services/{service_name}",
                "status": "GET /gateway/registry/status",
            },
            "full_control_parameters": {
                "url": "Service base URL",
                "prefix": "URL prefix (can be different from service name or empty)",
                "has_openapi": "Whether service has OpenAPI spec",
                "health_path": "Custom health endpoint path",
                "openapi_path": "Custom OpenAPI spec path",
                "full_control": "If true, no automatic path generation",
            },
            "examples": {
                "core_service": {
                    "url": "https://api.example.com",
                    "prefix": "api",
                    "health_path": "/api/health",
                    "openapi_path": "/api/openapi.json",
                },
                "external_service": {
                    "url": "https://external.example.com",
                    "prefix": "external",
                    "health_path": "/status",
                    "openapi_path": "/docs/openapi.json",
                },
                "no_prefix_service": {
                    "url": "https://simple.example.com",
                    "prefix": "",
                    "health_path": "/health",
                    "openapi_path": "/openapi.json",
                },
            },
        },
        "benefits_of_simplification": [
            "Clearer mental model - 3 distinct use cases",
            "Less API surface - fewer endpoints to maintain",
            "No redundancy - unified REST API handles all runtime management",
            "Full developer control - specify exactly what you want",
            "Easier testing - fewer code paths to validate",
        ],
    }


# ️ Service removal endpoint
@app.delete("/gateway/registry/services/{service_name}", tags=["Gateway"])
async def remove_service(service_name: str):
    """Remove a service from the registry (for security purposes)"""
    try:
        # Remove from service registry
        removed_service = service_registry.remove_service(service_name)

        if removed_service:
            # Remove from global SERVICES dict
            SERVICES.pop(service_name, None)

            return {
                "success": True,
                "service_name": service_name,
                "message": f"Service {service_name} removed from registry for security purposes",
            }
        else:
            return {
                "success": False,
                "service_name": service_name,
                "message": f"Service {service_name} not found in registry",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove service: {str(e)}")


# 🎯 UNIFIED Service Management - Full Developer Control
@app.post("/gateway/registry/services/{service_name}", tags=["Gateway"])
@app.put("/gateway/registry/services/{service_name}", tags=["Gateway"])
async def manage_service(
    service_name: str,
    url: str,
    prefix: str = None,
    has_openapi: bool = True,
    health_path: str = None,
    openapi_path: str = None,
    full_control: bool = True,
):
    """
    Unified service management endpoint with FULL DEVELOPER CONTROL

    Supports both POST (add) and PUT (update) operations.
    When full_control=True (default), no automatic path generation - use exactly what you specify.

    Examples:
    - Core service: prefix="api", health_path="/api/health", openapi_path="/api/openapi.json"
    - External service: prefix="custom", health_path="/custom/status", openapi_path="/custom/docs"
    - No prefix service: prefix="", health_path="/health", openapi_path="/openapi.json"
    """
    try:
        # Use unified service management with full control
        service_config = await service_registry.add_manual_service(
            name=service_name,
            url=url,
            prefix=prefix,
            has_openapi=has_openapi,
            health_path=health_path,
            openapi_path=openapi_path,
            full_control=full_control,
        )

        # Create dynamic route for the new/updated service
        routes_created = create_service_route(service_name, service_config)

        # Determine service type
        core_services = {"api", "fairness", "classification", "regression", "data_drift", "frontend"}
        service_type = "core_service" if service_name in core_services else "external_service"

        return {
            "success": True,
            "service_name": service_name,
            "service_type": service_type,
            "control_mode": "full_control" if full_control else "smart_defaults",
            "routes_created": routes_created,
            "configuration": {
                "url": service_config.url,
                "prefix": service_config.prefix,
                "has_openapi": service_config.has_openapi,
                "health_path": service_config.health_path,
                "openapi_path": service_config.openapi_path,
                "is_manual": service_config.is_manual,
            },
            "message": f"Service '{service_name}' configured successfully with full developer control",
            "routing_status": (
                "Routes created dynamically - no restart required"
                if routes_created
                else "No routes needed for this service type"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to manage service: {str(e)}")


# DEBUG ENDPOINT REMOVED: debug_urls()
# Reason: Development-only endpoint not needed in production

# DEBUG ENDPOINT REMOVED: debug_service_specs()
# Reason: Development-only endpoint for OpenAPI spec debugging

# DEBUG ENDPOINT REMOVED: debug_endpoints()
# Reason: Development-only endpoint for endpoint counting and analysis

# DEBUG ENDPOINT REMOVED: debug_cache()
# Reason: Development-only cache statistics and debugging

# DEBUG ENDPOINT REMOVED: clear_cache()
# Reason: Development-only cache clearing functionality

# 🔥 ESSENTIAL PRODUCTION DEBUG ENDPOINTS
# These are the absolute minimum debug endpoints needed for production troubleshooting


@app.get("/gateway/debug-health", tags=["Gateway Debug"])
async def debug_health():
    """
    PRODUCTION ESSENTIAL: Detailed health tracking for SRE/DevOps monitoring

    Critical for:
    - Incident response and troubleshooting
    - Performance monitoring and capacity planning
    - Service failure analysis and diagnostics
    - SRE dashboard integration
    """
    try:
        health_summary = health_tracker.get_health_summary()

        # Calculate overall platform health metrics
        total_services = len(health_summary)
        healthy_services = sum(1 for s in health_summary.values() if s["status"] == "healthy")
        avg_response_time = sum(s["avg_response_time"] for s in health_summary.values()) / max(total_services, 1)
        total_requests = sum(s["total_requests"] for s in health_summary.values())

        return {
            "health_tracking_enabled": True,
            "platform_overview": {
                "total_services_tracked": total_services,
                "healthy_services": healthy_services,
                "unhealthy_services": total_services - healthy_services,
                "platform_health_score": f"{(healthy_services / max(total_services, 1)) * 100:.1f}%",
                "average_response_time_ms": round(avg_response_time, 2),
                "total_requests_processed": total_requests,
            },
            "service_details": health_summary,
            "last_updated": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"error": f"Failed to get health tracking data: {str(e)}", "health_tracking_enabled": False}


@app.get("/gateway/debug-cache", tags=["Gateway Debug"])
async def debug_cache():
    """
    PRODUCTION ESSENTIAL: Cache inspection for OpenAPI troubleshooting

    Critical for:
    - OpenAPI spec corruption diagnosis
    - Service integration debugging
    - Performance analysis and optimization
    - Cache health monitoring
    """
    try:
        cache_stats = openapi_cache.get_stats()

        # Analyze cache entries for production insights (using public method)
        all_keys = openapi_cache.list_keys()
        unified_specs = [k for k in all_keys if k.startswith("unified_openapi_")]
        service_specs = [k for k in all_keys if k.startswith("openapi_") and not k.startswith("unified_openapi_")]

        return {
            "cache_enabled": True,
            "cache_health": {
                "total_entries": cache_stats["total_entries"],
                "valid_entries": cache_stats["valid_entries"],
                "expired_entries": cache_stats["expired_entries"],
                "cache_efficiency": f"{(cache_stats['valid_entries'] / max(cache_stats['total_entries'], 1)) * 100:.1f}% valid",
            },
            "cache_breakdown": {
                "unified_specs": len(unified_specs),
                "individual_service_specs": len(service_specs),
                "cache_keys": all_keys[:10],  # Limit for production, using public method
            },
            "cache_config": {
                "default_ttl_seconds": openapi_cache.default_ttl,
                "memory_based": True,
                "auto_cleanup_enabled": True,
            },
            "troubleshooting": {
                "cache_issues": cache_stats["expired_entries"] > 0,
                "recommendations": [
                    "Clear cache if OpenAPI specs are corrupted",
                    "Monitor cache efficiency for performance",
                    "Check for expired entries during issues",
                ],
            },
        }

    except Exception as e:
        return {"error": f"Failed to get cache stats: {str(e)}", "cache_enabled": False}


@app.post("/gateway/debug-cache/clear", tags=["Gateway Debug"])
async def clear_cache(request: Request):
    """
    PRODUCTION ESSENTIAL: Emergency cache clearing for recovery

    Critical for:
    - Emergency recovery from corrupted OpenAPI specs
    - Service deployment cache refresh
    - Integration issue resolution
    - Incident response and system recovery

    Security: Requires admin authorization via X-Admin-Key header or internal origin
    """
    # Basic security check - require admin key or internal origin
    admin_key = os.getenv("GATEWAY_ADMIN_KEY", "")
    request_admin_key = request.headers.get("X-Admin-Key", "")
    client_host = request.client.host if request.client else ""

    # Allow if admin key matches, or if it's from localhost/internal network
    is_authorized = (
        (admin_key and request_admin_key == admin_key)
        or client_host in ["127.0.0.1", "localhost", "::1"]
        or client_host.startswith("10.")
        or client_host.startswith("192.168.")
        or client_host.startswith("172.")
    )

    if not is_authorized:
        raise HTTPException(
            status_code=403, detail="Access denied. Requires X-Admin-Key header or internal network access."
        )
    try:
        # Capture state before clearing
        old_stats = openapi_cache.get_stats()
        clear_timestamp = datetime.now()

        # Clear the cache
        openapi_cache.clear()

        # Get new state
        new_stats = openapi_cache.get_stats()

        return {
            "success": True,
            "message": "Production cache cleared successfully",
            "operation_details": {
                "cleared_at": clear_timestamp.isoformat(),
                "entries_cleared": old_stats["total_entries"],
                "valid_entries_cleared": old_stats["valid_entries"],
                "expired_entries_cleared": old_stats["expired_entries"],
            },
            "post_clear_state": {
                "total_entries": new_stats["total_entries"],
                "cache_empty": new_stats["total_entries"] == 0,
            },
            "next_steps": [
                "Next OpenAPI request will rebuild cache",
                "Monitor /debug-cache to verify cache rebuilding",
                "Check unified docs at /docs to confirm recovery",
            ],
            "recovery_note": "Cache will automatically rebuild on next API documentation request",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to clear cache: {str(e)}",
            "troubleshooting": "Contact system administrator if cache clearing fails",
        }


# 🚀 DYNAMIC ROUTE GENERATION (replacing manual route functions)
def create_service_route(service_name: str, service_config: ServiceConfig):
    """Create route for a single service"""
    if service_name == "frontend":
        # Frontend routes are handled separately - skip for individual creation
        return 0

    if not service_config.prefix:
        return 0

    prefix = service_config.prefix

    # Create closure to capture current service values
    def create_route_handler(svc_name: str, svc_prefix: str):
        # Create unique function name and operation ID
        unique_operation_id = f"route_{svc_name}_service"
        unique_function_name = f"route_{svc_name}_service"

        @app.api_route(
            f"/{svc_prefix}/{{path:path}}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            operation_id=unique_operation_id,
            include_in_schema=False,
            tags=[f"Dynamic-{svc_name.title()}"],
        )
        async def route_service(path: str, request: Request):
            # Get current URL from service registry (fixes stale URL issue)
            current_service = service_registry.get_service(svc_name)
            if not current_service:
                raise HTTPException(status_code=503, detail=f"Service {svc_name} not found")

            current_url = current_service.url
            target_url = f"{current_url}/{svc_prefix}/{path}"
            return await proxy_request(target_url, request, svc_name)

        # Set unique function name to avoid Operation ID conflicts
        route_service.__name__ = unique_function_name
        return route_service

    # Create the route function
    create_route_handler(service_name, prefix)
    return 1


def create_dynamic_routes():
    """Auto-generate routes for all services in the registry with unique Operation IDs"""
    logger.info("🔄 Creating dynamic routes from service registry...")

    routes_created = 0

    # Create routes for each service
    for service_name, service_config in service_registry.get_all_services().items():
        if service_name == "frontend":
            service_url = service_config.url

            # Root route for frontend
            @app.api_route(
                "/",
                methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
                operation_id=f"route_frontend_root",
                include_in_schema=False,
            )
            async def route_frontend_root(request: Request):
                # Get current URL from service registry (fixes stale URL issue)
                current_service = service_registry.get_service("frontend")
                current_url = current_service.url if current_service else service_url
                target_url = f"{current_url}/"
                return await proxy_request(target_url, request, "frontend")

            # Set unique function name to avoid Operation ID conflicts
            route_frontend_root.__name__ = f"route_frontend_root"

            # Catch-all route for frontend (must be last)
            @app.api_route(
                "/{path:path}",
                methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
                operation_id=f"route_frontend_catchall",
                include_in_schema=False,
            )
            async def route_frontend_catchall(path: str, request: Request):
                # Skip if it's already a service route
                service_prefixes = [s.prefix for s in service_registry.get_all_services().values() if s.prefix]
                if any(path.startswith(f"{prefix}/") for prefix in service_prefixes):
                    raise HTTPException(status_code=404, detail="Not found")
                if path.startswith(("gateway/", "docs", "redoc", "openapi.json")):
                    raise HTTPException(status_code=404, detail="Not found")

                # Get current URL from service registry (fixes stale URL issue)
                current_service = service_registry.get_service("frontend")
                current_url = current_service.url if current_service else service_url
                target_url = f"{current_url}/{path}"
                return await proxy_request(target_url, request, "frontend")

            # Set unique function name to avoid Operation ID conflicts
            route_frontend_catchall.__name__ = f"route_frontend_catchall"

            routes_created += 2

        else:  # API services with prefixes
            routes_created += create_service_route(service_name, service_config)

    logger.info(
        f"Dynamic routing complete: {routes_created} routes created for {len(service_registry.get_all_services())} services"
    )


# Initialize dynamic routes
create_dynamic_routes()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
