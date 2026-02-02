# Testing in the Classification Service

This directory contains tests for the Classification microservice. The following types of testing are currently used or recommended:

## 1. Unit Testing
- **Description:** Tests individual components (e.g., service classes, utility functions) in isolation, using mocks for dependencies.
- **Example:**
  - `test_classification_stats.py` (when using direct calls to service logic with mocked data/models)
- **Purpose:** Ensures correctness of internal logic without involving the API layer or external systems.

## 2. API Testing
- **Description:** Tests the FastAPI endpoints using `TestClient`, simulating real HTTP requests and responses.
- **Example:**
  - Previous versions of tests using FastAPI's `TestClient` and dependency overrides
- **Purpose:** Validates routing, request/response handling, authentication, and integration of dependencies at the API level.

## 3. Integration Testing
- **Description:** Tests the interaction between multiple components or services, often using real or staged dependencies (e.g., database, external APIs).
- **Example:**
  - API tests that do not mock all dependencies, or tests that use a test database
- **Purpose:** Ensures that components work together as expected in a production-like environment.

## 4. Contract Testing (Recommended)
- **Description:** Ensures that the API adheres to a defined contract (e.g. Swagger spec), and that changes do not break consumers.
- **Example:**
- **Purpose:** Prevents breaking changes and ensures compatibility with clients.

---

**Best Practice:**
- Use a combination of unit, API, and integration tests for robust coverage.
- Add contract tests if your service is consumed by external clients.

---
Pure unit testing means:
1. Test single function/module
2. No FastAPI
3. No HTTP
4. No dependencies (like Database, S3, Other APIs, Auth)
5. No Contract tests

If you ONLY do pure unit testing
You will miss:
1. Wrong endpoint paths
2. Missing request keys
3. Broken dependency wiring
4. Incorrect status codes
5. Auth failures
6. Wrong Request/Response payload structure
👉 Frontend will break in production.

Our approaches:
1. API integration tests
2. Endpoint correctness
3. Required fields
4. Status codes
5. Contract stability
6. Regression protection
7. Unit test

Better than only unittest Because it:
1. Catches breaking API changes
2. Protects frontend contracts
3. Prevents silent production bugs
4. Works perfectly with CI/CD
5. Scales with microservices

We are doing
1. Unit testing
2. API testing
3. Contract testing
4. Regression testing
5. Integration testing

Note:-  Why we do testing like Unit, API, Contract, Integration etc.?
Catch bugs early, ensure correctness, and confidently deliver reliable, maintainable software without breaking existing functionality.
