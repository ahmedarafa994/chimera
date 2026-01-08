# =============================================================================
# Project Chimera Backend API - Production Dockerfile
# Multi-stage build with security hardening
# =============================================================================

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY pyproject.toml .
RUN pip install poetry && poetry export -f requirements.txt --output requirements.txt --without-hashes
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# =============================================================================
# Production stage
# =============================================================================
FROM python:3.11-slim AS production

# Security: Create non-root user
RUN groupadd -r chimera && useradd -r -g chimera chimera

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY --chown=chimera:chimera . .

# Security: Remove unnecessary files
RUN rm -rf /app/.git /app/.env /app/*.md /app/tests

# Security: Switch to non-root user
USER chimera

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "run.py"]