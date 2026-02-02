
# ----------- Stage 1: Build dependencies -----------
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and build tools first
RUN pip install --upgrade pip setuptools wheel

# Only copy requirements for dependency install
COPY services/what_if/requirements-dev.txt .

# Install Python packages to custom directory
RUN --mount=type=cache,target=/root/.cache/pip pip install --prefix=/install --verbose -r requirements-dev.txt

# ----------- Stage 2: Runtime image -----------
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY services/what_if/app ./services/what_if/app
COPY shared ./shared
COPY shared_migrations ./shared_migrations

# Add non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "services.what_if.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
