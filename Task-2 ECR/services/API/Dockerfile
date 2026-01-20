# ----------- Stage 1: Build dependencies -----------
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Only copy requirements for dependency install
COPY services/API/requirements*.txt .

RUN pip install --prefix=/install -r requirements.txt
# ----------- Stage 2: Runtime image -----------
FROM python:3.13-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY services/API/app ./services/API/app
COPY shared ./shared
COPY shared_migrations ./shared_migrations

EXPOSE 8000

CMD ["uvicorn", "services.API.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
