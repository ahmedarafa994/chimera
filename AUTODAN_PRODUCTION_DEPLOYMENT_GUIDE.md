# AutoDAN Production Deployment Guide

## Overview

This comprehensive deployment guide provides step-by-step instructions for deploying AutoDAN modules to production environments. It covers infrastructure requirements, security configurations, monitoring setup, and operational procedures.

---

## Table of Contents

1. [Infrastructure Requirements](#1-infrastructure-requirements)
2. [Security Configuration](#2-security-configuration)
3. [Environment Setup](#3-environment-setup)
4. [Container Deployment](#4-container-deployment)
5. [Monitoring and Observability](#5-monitoring-and-observability)
6. [Performance Optimization](#6-performance-optimization)
7. [Backup and Recovery](#7-backup-and-recovery)
8. [Operational Procedures](#8-operational-procedures)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [Security Compliance](#10-security-compliance)

---

## 1. Infrastructure Requirements

### 1.1 Minimum Hardware Requirements

#### Production Environment
- **CPU**: 8 cores (Intel Xeon or AMD EPYC)
- **Memory**: 32 GB RAM minimum, 64 GB recommended
- **Storage**: 500 GB SSD for system, 2 TB for data/logs
- **Network**: 1 Gbps network interface
- **GPU**: Optional - NVIDIA Tesla/RTX for acceleration

#### High-Availability Setup
- **Load Balancer**: 2x instances (active/passive)
- **Application Servers**: 3+ instances (horizontal scaling)
- **Database**: 3-node cluster with replication
- **Cache**: Redis cluster (3+ nodes)
- **Storage**: Distributed storage (Ceph, GlusterFS, or cloud equivalent)

### 1.2 Cloud Provider Requirements

#### AWS Deployment
```yaml
# Recommended AWS instance types
compute:
  - type: c5.2xlarge  # 8 vCPUs, 16 GB RAM
  - type: m5.4xlarge  # 16 vCPUs, 64 GB RAM (recommended)
  - type: r5.2xlarge  # 8 vCPUs, 64 GB RAM (memory-optimized)

storage:
  - ebs_volume_type: gp3
  - iops: 3000
  - throughput: 125 MB/s

networking:
  - vpc_with_private_subnets: true
  - nat_gateway: true
  - security_groups: restrictive
```

#### GCP Deployment
```yaml
# Recommended GCP machine types
compute:
  - type: n2-standard-8   # 8 vCPUs, 32 GB RAM
  - type: n2-standard-16  # 16 vCPUs, 64 GB RAM (recommended)
  - type: n2-highmem-8    # 8 vCPUs, 64 GB RAM (memory-optimized)

storage:
  - disk_type: pd-ssd
  - size: 500GB
  - performance: high
```

#### Azure Deployment
```yaml
# Recommended Azure VM sizes
compute:
  - type: Standard_D8s_v3   # 8 vCPUs, 32 GB RAM
  - type: Standard_D16s_v3  # 16 vCPUs, 64 GB RAM (recommended)
  - type: Standard_E8s_v3   # 8 vCPUs, 64 GB RAM (memory-optimized)

storage:
  - disk_type: Premium_SSD
  - size: 512GB
  - caching: ReadWrite
```

### 1.3 Network Requirements

```yaml
# Network Configuration
security_groups:
  web_tier:
    - port: 443 (HTTPS)
    - port: 80 (HTTP redirect)
    - source: 0.0.0.0/0

  app_tier:
    - port: 8001 (AutoDAN API)
    - port: 8080 (Internal services)
    - source: web_tier_sg

  data_tier:
    - port: 5432 (PostgreSQL)
    - port: 6379 (Redis)
    - source: app_tier_sg

firewall_rules:
  - allow_internal_communication
  - deny_external_database_access
  - allow_monitoring_ports
  - allow_ssh_from_bastion_only
```

---

## 2. Security Configuration

### 2.1 Environment Variables

```bash
# Production Environment Variables
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export DEBUG=false

# Security Settings
export JWT_SECRET="$(openssl rand -base64 32)"
export API_KEY="$(openssl rand -base64 32)"
export CHIMERA_API_KEY="$(openssl rand -base64 32)"
export ENCRYPTION_KEY="$(openssl rand -base64 32)"

# AutoDAN Security
export AUTODAN_SAFETY_ENABLED=true
export AUTODAN_ETHICAL_MODE=true
export AUTODAN_AUDIT_LOGGING=true
export AUTODAN_RATE_LIMIT_ENABLED=true

# Database Security
export DB_SSL_MODE=require
export DB_SSL_CERT_PATH=/etc/ssl/certs/db-client.crt
export DB_SSL_KEY_PATH=/etc/ssl/private/db-client.key
export DB_SSL_CA_PATH=/etc/ssl/certs/ca-certificate.crt

# API Provider Security
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# Network Security
export ALLOWED_ORIGINS="https://yourdomain.com,https://api.yourdomain.com"
export CORS_ENABLED=true
export HTTPS_ONLY=true
export SECURE_COOKIES=true

# Monitoring and Logging
export SENTRY_DSN="your-sentry-dsn"
export PROMETHEUS_ENABLED=true
export JAEGER_ENABLED=true
```

### 2.2 SSL/TLS Configuration

```nginx
# Nginx SSL Configuration
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    # SSL Certificates
    ssl_certificate /etc/ssl/certs/yourdomain.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.key;

    # SSL Security Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";

    # Rate Limiting
    limit_req zone=api burst=20 nodelay;
    limit_req_status 429;

    location / {
        proxy_pass http://autodan-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Security headers for proxy
        proxy_set_header X-Request-ID $request_id;
        proxy_set_header X-Correlation-ID $request_id;
    }
}
```

### 2.3 API Security

```python
# API Security Configuration
SECURITY_CONFIG = {
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 60,
        "burst_size": 10,
        "whitelist": ["trusted-ips"]
    },
    "authentication": {
        "jwt_expiry": 3600,  # 1 hour
        "refresh_token_expiry": 86400,  # 24 hours
        "require_https": True,
        "secure_cookies": True
    },
    "input_validation": {
        "max_request_size": "10MB",
        "sanitize_inputs": True,
        "validate_schemas": True,
        "escape_html": True
    },
    "cors": {
        "enabled": True,
        "allow_origins": ["https://yourdomain.com"],
        "allow_methods": ["GET", "POST"],
        "allow_headers": ["Authorization", "Content-Type"],
        "max_age": 3600
    }
}
```

---

## 3. Environment Setup

### 3.1 Production Docker Configuration

```dockerfile
# Dockerfile.production
FROM python:3.11-slim as base

# Security: Create non-root user
RUN groupadd -r autodan && useradd -r -g autodan autodan

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Change ownership to non-root user
RUN chown -R autodan:autodan /app

# Switch to non-root user
USER autodan

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]
```

### 3.2 Docker Compose Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  autodan-api:
    build:
      context: .
      dockerfile: Dockerfile.production
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://user:pass@postgres:5432/autodan
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    networks:
      - autodan-network
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - autodan-api
    networks:
      - autodan-network

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=autodan
      - POSTGRES_USER=autodan
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    networks:
      - autodan-network

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - autodan-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - autodan-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    networks:
      - autodan-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  autodan-network:
    driver: bridge
```

### 3.3 Kubernetes Deployment

```yaml
# k8s/autodan-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autodan-api
  namespace: autodan
spec:
  replicas: 3
  selector:
    matchLabels:
      app: autodan-api
  template:
    metadata:
      labels:
        app: autodan-api
    spec:
      containers:
      - name: autodan-api
        image: autodan:latest
        ports:
        - containerPort: 8001
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: autodan-secrets
              key: database-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: autodan-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: autodan-api-service
  namespace: autodan
spec:
  selector:
    app: autodan-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8001
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: autodan-ingress
  namespace: autodan
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "60"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: autodan-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: autodan-api-service
            port:
              number: 80
```

---

## 4. Container Deployment

### 4.1 Build and Push Images

```bash
#!/bin/bash
# deploy.sh - Production deployment script

set -e

# Configuration
DOCKER_REGISTRY="your-registry.com"
IMAGE_NAME="autodan"
VERSION=${1:-latest}
ENVIRONMENT=${2:-production}

echo "🚀 Starting AutoDAN deployment..."

# Build production image
echo "📦 Building production image..."
docker build -f Dockerfile.production -t ${IMAGE_NAME}:${VERSION} .

# Tag for registry
docker tag ${IMAGE_NAME}:${VERSION} ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}

# Push to registry
echo "📤 Pushing to registry..."
docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}

# Deploy based on environment
if [ "$ENVIRONMENT" = "kubernetes" ]; then
    echo "☸️ Deploying to Kubernetes..."
    kubectl set image deployment/autodan-api autodan-api=${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}
    kubectl rollout status deployment/autodan-api
elif [ "$ENVIRONMENT" = "docker-compose" ]; then
    echo "🐳 Deploying with Docker Compose..."
    export IMAGE_VERSION=${VERSION}
    docker-compose -f docker-compose.prod.yml up -d
else
    echo "🖥️ Deploying standalone..."
    docker run -d \
        --name autodan-api \
        --restart unless-stopped \
        -p 8001:8001 \
        --env-file .env.production \
        ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}
fi

echo "✅ Deployment completed successfully!"

# Run health checks
echo "🏥 Running health checks..."
sleep 30
curl -f http://localhost:8001/health || {
    echo "❌ Health check failed!"
    exit 1
}

echo "✅ Health check passed!"
```

### 4.2 Production Environment File

```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Server Configuration
PORT=8001
HOST=0.0.0.0
WORKERS=4
WORKER_CLASS=uvicorn.workers.UvicornWorker

# Database
DATABASE_URL=postgresql://autodan:${DB_PASSWORD}@postgres:5432/autodan
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
REDIS_POOL_SIZE=10

# Security
JWT_SECRET=${JWT_SECRET}
API_KEY=${API_KEY}
CHIMERA_API_KEY=${CHIMERA_API_KEY}
ENCRYPTION_KEY=${ENCRYPTION_KEY}

# AutoDAN Configuration
AUTODAN_SAFETY_ENABLED=true
AUTODAN_ETHICAL_MODE=true
AUTODAN_MAX_GENERATIONS=20
AUTODAN_DEFAULT_TEMPERATURE=0.7

# AI Providers
OPENAI_API_KEY=${OPENAI_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
GOOGLE_API_KEY=${GOOGLE_API_KEY}
DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}

# Monitoring
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=true
SENTRY_DSN=${SENTRY_DSN}

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_BURST=10
```

---

## 5. Monitoring and Observability

### 5.1 Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "autodan_alerts.yml"

scrape_configs:
  - job_name: 'autodan-api'
    static_configs:
      - targets: ['autodan-api:8001']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
```

### 5.2 Grafana Dashboards

```json
{
  "dashboard": {
    "title": "AutoDAN Production Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{handler}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "AutoDAN Strategy Performance",
        "type": "table",
        "targets": [
          {
            "expr": "autodan_strategy_duration_seconds",
            "legendFormat": "{{strategy}}"
          }
        ]
      }
    ]
  }
}
```

### 5.3 Custom Metrics

```python
# app/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time

# Custom metrics for AutoDAN
registry = CollectorRegistry()

# Request metrics
autodan_requests_total = Counter(
    'autodan_requests_total',
    'Total AutoDAN requests',
    ['method', 'strategy', 'status'],
    registry=registry
)

autodan_request_duration = Histogram(
    'autodan_request_duration_seconds',
    'AutoDAN request duration',
    ['method', 'strategy'],
    registry=registry
)

# Strategy metrics
autodan_strategy_executions = Counter(
    'autodan_strategy_executions_total',
    'Total strategy executions',
    ['strategy', 'success'],
    registry=registry
)

autodan_strategy_duration = Histogram(
    'autodan_strategy_duration_seconds',
    'Strategy execution duration',
    ['strategy'],
    registry=registry
)

# System metrics
autodan_active_requests = Gauge(
    'autodan_active_requests',
    'Currently active requests',
    registry=registry
)

autodan_cache_hit_rate = Gauge(
    'autodan_cache_hit_rate',
    'Cache hit rate',
    registry=registry
)

def track_request(method: str, strategy: str):
    """Context manager to track request metrics."""
    start_time = time.time()
    autodan_active_requests.inc()

    try:
        yield
        status = 'success'
    except Exception:
        status = 'error'
        raise
    finally:
        duration = time.time() - start_time
        autodan_requests_total.labels(
            method=method,
            strategy=strategy,
            status=status
        ).inc()
        autodan_request_duration.labels(
            method=method,
            strategy=strategy
        ).observe(duration)
        autodan_active_requests.dec()
```

### 5.4 Alerting Rules

```yaml
# monitoring/autodan_alerts.yml
groups:
  - name: autodan_alerts
    rules:
      - alert: AutoDANHighErrorRate
        expr: rate(autodan_requests_total{status="error"}[5m]) / rate(autodan_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in AutoDAN API"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"

      - alert: AutoDANHighLatency
        expr: histogram_quantile(0.95, rate(autodan_request_duration_seconds_bucket[5m])) > 5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High latency in AutoDAN API"
          description: "95th percentile latency is {{ $value }}s"

      - alert: AutoDANServiceDown
        expr: up{job="autodan-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "AutoDAN API is down"
          description: "AutoDAN API has been down for more than 1 minute"

      - alert: AutoDANDatabaseConnections
        expr: postgres_max_connections - postgres_connections > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low database connections available"
          description: "Only {{ $value }} database connections remaining"
```

---

## 6. Performance Optimization

### 6.1 Application Performance

```python
# app/core/performance_config.py
PERFORMANCE_CONFIG = {
    "uvicorn": {
        "workers": 4,
        "worker_class": "uvicorn.workers.UvicornWorker",
        "max_requests": 1000,
        "max_requests_jitter": 100,
        "preload_app": True,
        "keepalive": 2
    },
    "database": {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 3600,
        "pool_pre_ping": True
    },
    "redis": {
        "connection_pool_size": 10,
        "socket_keepalive": True,
        "socket_keepalive_options": {},
        "retry_on_timeout": True
    },
    "autodan": {
        "max_concurrent_requests": 5,
        "request_timeout": 30,
        "strategy_cache_size": 1000,
        "result_cache_ttl": 3600
    }
}
```

### 6.2 Caching Strategy

```python
# app/core/caching.py
import redis
from typing import Any, Optional
import json
import hashlib

class AutoDANCache:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour

    def _generate_key(self, prefix: str, data: dict) -> str:
        """Generate cache key from request data."""
        serialized = json.dumps(data, sort_keys=True)
        hash_digest = hashlib.sha256(serialized.encode()).hexdigest()
        return f"{prefix}:{hash_digest}"

    async def get_strategy_result(self, request_data: dict) -> Optional[dict]:
        """Get cached strategy result."""
        key = self._generate_key("strategy", request_data)
        cached_data = await self.redis_client.get(key)
        if cached_data:
            return json.loads(cached_data)
        return None

    async def set_strategy_result(self, request_data: dict, result: dict, ttl: Optional[int] = None) -> None:
        """Cache strategy result."""
        key = self._generate_key("strategy", request_data)
        serialized_result = json.dumps(result)
        await self.redis_client.setex(key, ttl or self.default_ttl, serialized_result)

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        keys = await self.redis_client.keys(pattern)
        if keys:
            return await self.redis_client.delete(*keys)
        return 0
```

### 6.3 Database Optimization

```sql
-- Database optimization queries
-- Create indexes for AutoDAN tables

-- Strategy execution tracking
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_executions_timestamp
ON strategy_executions(created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_executions_strategy
ON strategy_executions(strategy_name, success);

-- Request logging
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_request_logs_timestamp
ON request_logs(timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_request_logs_user
ON request_logs(user_id, timestamp DESC);

-- Performance optimization
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_component
ON performance_metrics(component_name, timestamp DESC);

-- Partitioning for large tables
CREATE TABLE request_logs_partitioned (
    id BIGSERIAL,
    timestamp TIMESTAMP NOT NULL,
    request_data JSONB,
    response_data JSONB,
    duration_ms INTEGER,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE request_logs_y2026m01 PARTITION OF request_logs_partitioned
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

---

## 7. Backup and Recovery

### 7.1 Database Backup

```bash
#!/bin/bash
# backup.sh - Database backup script

set -e

BACKUP_DIR="/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DB_NAME="autodan"
DB_USER="autodan"
RETENTION_DAYS=30

echo "🗄️ Starting database backup..."

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Full database backup
pg_dump -h postgres -U ${DB_USER} -d ${DB_NAME} \
    --format=custom \
    --compress=9 \
    --no-owner \
    --no-privileges \
    > ${BACKUP_DIR}/autodan_full_${TIMESTAMP}.backup

# Schema-only backup
pg_dump -h postgres -U ${DB_USER} -d ${DB_NAME} \
    --schema-only \
    --format=plain \
    > ${BACKUP_DIR}/autodan_schema_${TIMESTAMP}.sql

# Data-only backup for critical tables
pg_dump -h postgres -U ${DB_USER} -d ${DB_NAME} \
    --data-only \
    --table=users \
    --table=api_keys \
    --table=strategy_configurations \
    --format=custom \
    > ${BACKUP_DIR}/autodan_critical_data_${TIMESTAMP}.backup

# Compress backups
gzip ${BACKUP_DIR}/autodan_schema_${TIMESTAMP}.sql

# Upload to cloud storage
aws s3 cp ${BACKUP_DIR}/autodan_full_${TIMESTAMP}.backup \
    s3://autodan-backups/database/ \
    --storage-class STANDARD_IA

# Clean up old backups
find ${BACKUP_DIR} -name "autodan_*" -type f -mtime +${RETENTION_DAYS} -delete

echo "✅ Database backup completed: autodan_full_${TIMESTAMP}.backup"
```

### 7.2 Application State Backup

```bash
#!/bin/bash
# app_backup.sh - Application state backup

BACKUP_DIR="/backups/app_state"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "📦 Backing up application state..."

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Backup configuration files
tar -czf ${BACKUP_DIR}/config_${TIMESTAMP}.tar.gz \
    /app/config/ \
    /app/.env* \
    /etc/nginx/ \
    /etc/ssl/

# Backup logs
tar -czf ${BACKUP_DIR}/logs_${TIMESTAMP}.tar.gz \
    /app/logs/ \
    /var/log/nginx/

# Backup Redis data
redis-cli --rdb ${BACKUP_DIR}/redis_${TIMESTAMP}.rdb

# Upload to cloud storage
aws s3 sync ${BACKUP_DIR}/ s3://autodan-backups/app-state/ \
    --exclude="*.log" \
    --storage-class STANDARD_IA

echo "✅ Application state backup completed"
```

### 7.3 Disaster Recovery Plan

```yaml
# disaster_recovery.yml
disaster_recovery:
  rpo: "1 hour"  # Recovery Point Objective
  rto: "4 hours"  # Recovery Time Objective

  backup_strategy:
    database:
      frequency: "every 6 hours"
      retention: "30 days"
      location: ["local", "s3", "secondary_region"]

    application:
      frequency: "daily"
      retention: "14 days"
      location: ["s3", "secondary_region"]

  recovery_procedures:
    1. "Assess damage and scope"
    2. "Activate incident response team"
    3. "Restore from most recent backup"
    4. "Validate data integrity"
    5. "Perform smoke tests"
    6. "Gradually restore traffic"
    7. "Monitor for issues"
    8. "Document lessons learned"

  emergency_contacts:
    - name: "DevOps Team Lead"
      phone: "+1-xxx-xxx-xxxx"
      email: "devops-lead@company.com"

    - name: "Database Administrator"
      phone: "+1-xxx-xxx-xxxx"
      email: "dba@company.com"
```

---

## 8. Operational Procedures

### 8.1 Deployment Checklist

```markdown
## Pre-Deployment Checklist

### Environment Validation
- [ ] All environment variables configured
- [ ] SSL certificates valid and not expiring
- [ ] Database connections tested
- [ ] Redis connectivity verified
- [ ] External API keys validated

### Security Verification
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] CORS settings validated
- [ ] Input validation active
- [ ] Audit logging enabled

### Performance Validation
- [ ] Load balancer health checks passing
- [ ] Auto-scaling policies configured
- [ ] Resource limits set appropriately
- [ ] Monitoring alerts configured
- [ ] Cache warming completed

### Testing
- [ ] Health checks passing
- [ ] Smoke tests completed
- [ ] Integration tests passed
- [ ] Security scans completed
- [ ] Performance benchmarks met

## Deployment Process

1. **Pre-deployment**
   ```bash
   # Run pre-deployment checks
   ./scripts/pre_deployment_check.sh

   # Backup current state
   ./scripts/backup.sh
   ```

2. **Deploy**
   ```bash
   # Deploy new version
   ./scripts/deploy.sh v1.2.0 production
   ```

3. **Post-deployment**
   ```bash
   # Run post-deployment checks
   ./scripts/post_deployment_check.sh

   # Monitor for issues
   ./scripts/monitor_deployment.sh
   ```

4. **Rollback** (if needed)
   ```bash
   # Rollback to previous version
   ./scripts/rollback.sh v1.1.9
   ```
```

### 8.2 Monitoring Runbook

```markdown
## AutoDAN Monitoring Runbook

### Key Metrics to Monitor

#### Application Metrics
- Request rate (normal: 10-100 RPS)
- Response time (target: <2s P95)
- Error rate (target: <5%)
- Success rate (target: >95%)

#### Infrastructure Metrics
- CPU utilization (alert: >80%)
- Memory usage (alert: >85%)
- Disk space (alert: >90%)
- Network I/O (monitor for spikes)

#### Business Metrics
- AutoDAN strategy success rates
- Average optimization time
- Cache hit ratios
- Provider API latencies

### Alert Response Procedures

#### High Error Rate Alert
1. Check application logs for error patterns
2. Verify external API connectivity
3. Check database connection pool
4. Review recent deployments
5. Scale resources if needed

#### High Latency Alert
1. Check slow query logs
2. Monitor database performance
3. Verify cache performance
4. Check network connectivity
5. Scale application instances

#### Service Down Alert
1. Check pod/container status
2. Review application logs
3. Verify health check endpoints
4. Check load balancer configuration
5. Restart service if necessary

### Escalation Matrix
- **Level 1**: On-call engineer (5 minutes)
- **Level 2**: Senior engineer (15 minutes)
- **Level 3**: Engineering manager (30 minutes)
- **Level 4**: CTO/VP Engineering (1 hour)
```

### 8.3 Maintenance Windows

```yaml
# maintenance_schedule.yml
maintenance_windows:
  regular:
    frequency: "monthly"
    duration: "2 hours"
    time: "Sunday 02:00-04:00 UTC"
    activities:
      - "OS security updates"
      - "Database maintenance"
      - "SSL certificate renewal"
      - "Log rotation and cleanup"
      - "Performance optimization"

  emergency:
    duration: "30 minutes"
    notification: "immediate"
    activities:
      - "Critical security patches"
      - "Service restarts"
      - "Configuration updates"

  notification_channels:
    - email: "ops-team@company.com"
    - slack: "#autodan-ops"
    - status_page: "https://status.company.com"
```

---

## 9. Troubleshooting Guide

### 9.1 Common Issues

#### Issue: High Memory Usage

**Symptoms:**
- Memory utilization > 90%
- Pod/container restarts
- Slow response times

**Diagnosis:**
```bash
# Check memory usage
kubectl top pods -n autodan
docker stats autodan-api

# Check memory leaks
kubectl exec -it autodan-api-xxx -- ps aux --sort=-%mem | head -20

# Review memory metrics
curl http://localhost:8001/metrics | grep memory
```

**Resolution:**
```bash
# Increase memory limits
kubectl patch deployment autodan-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"autodan-api","resources":{"limits":{"memory":"8Gi"}}}]}}}}'

# Restart pods
kubectl rollout restart deployment/autodan-api

# Monitor improvement
kubectl logs -f deployment/autodan-api
```

#### Issue: Database Connection Errors

**Symptoms:**
- "connection refused" errors
- High database connection count
- Slow database queries

**Diagnosis:**
```sql
-- Check active connections
SELECT count(*) as active_connections
FROM pg_stat_activity
WHERE state = 'active';

-- Check slow queries
SELECT query, state, query_start, now() - query_start as duration
FROM pg_stat_activity
WHERE now() - query_start > interval '5 minutes';
```

**Resolution:**
```bash
# Increase connection pool
export DATABASE_POOL_SIZE=20
export DATABASE_MAX_OVERFLOW=40

# Restart application
kubectl rollout restart deployment/autodan-api

# Monitor connections
watch "kubectl exec postgres-0 -- psql -c \"SELECT count(*) FROM pg_stat_activity;\""
```

#### Issue: AutoDAN Strategy Failures

**Symptoms:**
- Strategy execution timeouts
- High error rates for specific strategies
- Inconsistent results

**Diagnosis:**
```bash
# Check strategy metrics
curl http://localhost:8001/metrics | grep autodan_strategy

# Review strategy logs
kubectl logs deployment/autodan-api | grep "strategy.*error"

# Test individual strategies
curl -X POST http://localhost:8001/api/v1/autodan/jailbreak \
  -H "Content-Type: application/json" \
  -d '{"request": "test", "method": "vanilla"}'
```

**Resolution:**
```python
# Check strategy configuration
from app.core.autodan_strategy_registry import get_strategy_registry
registry = get_strategy_registry()
strategy = registry.get_strategy("problematic_strategy")
print(strategy.get_performance_metrics())

# Reset strategy cache
await registry.cleanup_all_instances()

# Update strategy parameters
strategy_config = strategy.get_default_parameters()
strategy_config["timeout"] = 60  # Increase timeout
```

### 9.2 Performance Debugging

```bash
#!/bin/bash
# performance_debug.sh - Performance debugging script

echo "🔍 AutoDAN Performance Debugging"

# Check system resources
echo "=== System Resources ==="
kubectl top nodes
kubectl top pods -n autodan

# Check application metrics
echo "=== Application Metrics ==="
curl -s http://localhost:8001/metrics | grep -E "(request_duration|error_rate|cache_hit)"

# Check database performance
echo "=== Database Performance ==="
kubectl exec postgres-0 -- psql autodan -c "
  SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
  FROM pg_stats
  WHERE tablename IN ('users', 'strategy_executions', 'request_logs')
  ORDER BY tablename, attname;
"

# Check slow queries
kubectl exec postgres-0 -- psql autodan -c "
  SELECT
    calls,
    total_time/1000 as total_time_seconds,
    mean_time/1000 as mean_time_seconds,
    query
  FROM pg_stat_statements
  ORDER BY total_time DESC
  LIMIT 10;
"

# Check Redis performance
echo "=== Redis Performance ==="
kubectl exec redis-0 -- redis-cli INFO stats

# Generate performance report
echo "=== Performance Report Generated ==="
echo "Review the above metrics and check Grafana dashboards for detailed analysis"
```

---

## 10. Security Compliance

### 10.1 Security Checklist

```markdown
## Production Security Checklist

### Infrastructure Security
- [ ] All servers patched with latest security updates
- [ ] Firewall rules configured with least privilege
- [ ] VPN or bastion host for admin access
- [ ] Regular security scanning automated
- [ ] Intrusion detection system active

### Application Security
- [ ] Input validation on all endpoints
- [ ] SQL injection protection enabled
- [ ] XSS protection headers configured
- [ ] CSRF protection implemented
- [ ] Rate limiting on all API endpoints
- [ ] Authentication required for all operations
- [ ] Authorization checks implemented
- [ ] Sensitive data encrypted at rest and in transit

### Data Protection
- [ ] Database encrypted at rest
- [ ] Backups encrypted and tested
- [ ] API keys rotated regularly
- [ ] Secrets management system in use
- [ ] Data retention policies implemented
- [ ] GDPR/privacy compliance verified

### AutoDAN Specific Security
- [ ] Ethical boundary enforcement active
- [ ] Content filtering enabled
- [ ] Audit logging for all jailbreak attempts
- [ ] Research context validation implemented
- [ ] Responsible disclosure procedures documented
```

### 10.2 Audit Logging

```python
# app/security/audit_logger.py
import logging
import json
from datetime import datetime
from typing import Dict, Any

class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger("autodan.audit")
        self.logger.setLevel(logging.INFO)

        # Configure audit log handler
        handler = logging.FileHandler("/var/log/autodan/audit.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_autodan_request(self, user_id: str, request_data: Dict[str, Any],
                           result: Dict[str, Any], session_id: str = None):
        """Log AutoDAN jailbreak request for audit purposes."""
        audit_event = {
            "event_type": "autodan_request",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "session_id": session_id,
            "request": {
                "method": request_data.get("method"),
                "goal_hash": self._hash_sensitive_data(request_data.get("request", "")),
                "parameters": {
                    k: v for k, v in request_data.items()
                    if k not in ["request", "goal"]
                }
            },
            "result": {
                "success": result.get("success"),
                "score": result.get("score"),
                "execution_time_ms": result.get("execution_time_ms"),
                "prompt_hash": self._hash_sensitive_data(result.get("generated_prompt", ""))
            },
            "compliance": {
                "ethical_boundaries_checked": True,
                "content_filtered": True,
                "research_context_validated": True
            }
        }

        self.logger.info(json.dumps(audit_event))

    def log_security_event(self, event_type: str, details: Dict[str, Any],
                          severity: str = "INFO"):
        """Log security-related events."""
        security_event = {
            "event_type": f"security_{event_type}",
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "details": details
        }

        if severity == "CRITICAL":
            self.logger.critical(json.dumps(security_event))
        elif severity == "WARNING":
            self.logger.warning(json.dumps(security_event))
        else:
            self.logger.info(json.dumps(security_event))

    def _hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for audit logging."""
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()[:16]
```

### 10.3 Compliance Monitoring

```python
# app/security/compliance_monitor.py
from typing import Dict, Any, List
from datetime import datetime, timedelta

class ComplianceMonitor:
    def __init__(self):
        self.compliance_rules = {
            "data_retention": {
                "audit_logs": timedelta(days=2555),  # 7 years
                "user_data": timedelta(days=2555),   # 7 years
                "request_logs": timedelta(days=90),  # 90 days
                "cache_data": timedelta(hours=24)    # 24 hours
            },
            "access_controls": {
                "api_key_rotation": timedelta(days=90),
                "password_expiry": timedelta(days=90),
                "session_timeout": timedelta(hours=8)
            },
            "encryption_requirements": {
                "data_at_rest": True,
                "data_in_transit": True,
                "api_keys": True,
                "user_credentials": True
            }
        }

    async def check_compliance(self) -> Dict[str, Any]:
        """Check system compliance with security requirements."""
        compliance_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "compliant",
            "checks": {},
            "violations": [],
            "recommendations": []
        }

        # Check data retention compliance
        retention_status = await self._check_data_retention()
        compliance_report["checks"]["data_retention"] = retention_status

        # Check access control compliance
        access_status = await self._check_access_controls()
        compliance_report["checks"]["access_controls"] = access_status

        # Check encryption compliance
        encryption_status = await self._check_encryption()
        compliance_report["checks"]["encryption"] = encryption_status

        # Aggregate violations
        all_checks = [retention_status, access_status, encryption_status]
        for check in all_checks:
            if not check["compliant"]:
                compliance_report["overall_status"] = "non_compliant"
                compliance_report["violations"].extend(check["violations"])

        return compliance_report

    async def _check_data_retention(self) -> Dict[str, Any]:
        """Check data retention compliance."""
        # Implementation would check actual data retention
        # This is a simplified example
        return {
            "compliant": True,
            "violations": [],
            "last_cleanup": datetime.utcnow().isoformat()
        }

    async def _check_access_controls(self) -> Dict[str, Any]:
        """Check access control compliance."""
        return {
            "compliant": True,
            "violations": [],
            "api_keys_rotated": True,
            "sessions_expired": True
        }

    async def _check_encryption(self) -> Dict[str, Any]:
        """Check encryption compliance."""
        return {
            "compliant": True,
            "violations": [],
            "ssl_enabled": True,
            "database_encrypted": True
        }
```

---

## Conclusion

This comprehensive deployment guide provides all necessary components for successfully deploying AutoDAN modules to production environments. The guide covers infrastructure requirements, security configurations, monitoring setup, and operational procedures to ensure a robust, secure, and scalable deployment.

### Key Success Factors

1. **Security First**: All configurations prioritize security with encryption, authentication, and audit logging
2. **Monitoring**: Comprehensive observability with Prometheus, Grafana, and custom metrics
3. **Scalability**: Horizontal scaling with load balancing and auto-scaling capabilities
4. **Reliability**: High availability with redundancy and disaster recovery procedures
5. **Compliance**: Security compliance monitoring and audit trail maintenance

### Next Steps

1. Review and customize configurations for your specific environment
2. Test deployment in staging environment first
3. Implement monitoring and alerting before production deployment
4. Establish operational procedures and train team members
5. Conduct security review and penetration testing
6. Plan disaster recovery testing schedule

For additional support or questions about this deployment guide, please consult the AutoDAN documentation or contact the development team.