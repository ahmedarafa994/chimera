# =============================================================================
# Project Chimera Flask Backend - Production Dockerfile
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
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Download SpaCy model
RUN pip install spacy && python -m spacy download en_core_web_sm

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
COPY --from=builder /usr/local/lib/python3.11/site-packages/en_core_web_sm /usr/local/lib/python3.11/site-packages/en_core_web_sm
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
    FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose Flask port
EXPOSE 8000

# Run the application with Waitress (production server)
CMD ["python", "run.py"]
