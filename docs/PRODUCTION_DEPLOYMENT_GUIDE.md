# Chimera Production Deployment Guide

This guide covers deploying Chimera to production environments with best practices for security, scalability, and reliability.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Database Setup](#database-setup)
- [Redis Configuration](#redis-configuration)
- [LLM Provider Configuration](#llm-provider-configuration)
- [Security Hardening](#security-hardening)
- [Monitoring & Observability](#monitoring--observability)
- [Backup & Recovery](#backup--recovery)
- [Scaling Guidelines](#scaling-guidelines)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 50 GB SSD | 100+ GB NVMe |
| Network | 100 Mbps | 1 Gbps |

### Software Requirements

- **Docker** 24.0+ with Docker Compose v2
- **Python** 3.11+ (for local development)
- **Node.js** 18+ (for frontend builds)
- **PostgreSQL** 14+ (production database)
- **Redis** 7.0+ (caching and rate limiting)

### Required API Keys

Obtain API keys from the following providers:

| Provider | Required | Environment Variable |
|----------|----------|---------------------|
| OpenAI | Yes | `OPENAI_API_KEY` |
| Google AI | Recommended | `GOOGLE_API_KEY` |
| Anthropic | Optional | `ANTHROPIC_API_KEY` |
| DeepSeek | Optional | `DEEPSEEK_API_KEY` |

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/chimera.git
cd chimera
```

### 2. Create Production Environment File

```bash
cp .env.template .env.production
```

### 3. Configure Environment Variables

Edit `.env.production` with production values:

```bash
# Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
API_WORKERS=4

# Security - GENERATE NEW VALUES
API_KEY=your-secure-api-key-here
SECRET_KEY=your-256-bit-secret-key
JWT_SECRET=your-jwt-secret-key

# Database
DATABASE_URL=postgresql://chimera:password@db:5432/chimera
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://:password@redis:6379/0
REDIS_MAX_CONNECTIONS=50

# Rate Limiting
RATE_LIMIT_CALLS=100
RATE_LIMIT_PERIOD=60

# LLM Providers
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...

# Frontend
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_WS_URL=wss://api.yourdomain.com
```

### 4. Generate Secure Secrets

```bash
# Generate API key
openssl rand -hex 32

# Generate Secret Key
openssl rand -hex 64

# Generate JWT Secret
openssl rand -hex 32
```

---

## Docker Deployment

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    build:
      context: ./backend-api
      dockerfile: Dockerfile.prod
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env.production
    ports:
      - "8001:8001"
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    environment:
      - NODE_ENV=production
    env_file:
      - .env.production
    ports:
      - "3700:3700"
    depends_on:
      - api
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3700"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    restart: unless-stopped

  db:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: chimera
      POSTGRES_USER: chimera
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U chimera"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
      - frontend
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Deploy with Docker Compose

```bash
# Build and start services
docker-compose -f docker-compose.prod.yml up -d --build

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Check service health
docker-compose -f docker-compose.prod.yml ps
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.25+)
- kubectl configured
- Helm 3.x installed

### Helm Chart Installation

```bash
# Add Chimera Helm repository (if available)
helm repo add chimera https://charts.chimera.example.com

# Install with custom values
helm install chimera chimera/chimera \
  --namespace chimera \
  --create-namespace \
  --values values-production.yaml
```

### Custom Values File

Create `values-production.yaml`:

```yaml
replicaCount:
  api: 3
  frontend: 2

image:
  api:
    repository: your-registry/chimera-api
    tag: "latest"
  frontend:
    repository: your-registry/chimera-frontend
    tag: "latest"

resources:
  api:
    limits:
      cpu: 4000m
      memory: 8Gi
    requests:
      cpu: 2000m
      memory: 4Gi
  frontend:
    limits:
      cpu: 1000m
      memory: 2Gi
    requests:
      cpu: 500m
      memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: chimera.yourdomain.com
      paths:
        - path: /api
          pathType: Prefix
          service: api
        - path: /
          pathType: Prefix
          service: frontend
  tls:
    - secretName: chimera-tls
      hosts:
        - chimera.yourdomain.com

postgresql:
  enabled: true
  auth:
    postgresPassword: your-postgres-password
    database: chimera
  primary:
    persistence:
      size: 100Gi

redis:
  enabled: true
  auth:
    password: your-redis-password
  master:
    persistence:
      size: 10Gi
```

---

## Database Setup

### PostgreSQL Configuration

1. **Create Database and User**

```sql
CREATE DATABASE chimera;
CREATE USER chimera WITH ENCRYPTED PASSWORD 'your-secure-password';
GRANT ALL PRIVILEGES ON DATABASE chimera TO chimera;
```

2. **Run Migrations**

```bash
cd backend-api
poetry run alembic upgrade head
```

3. **Optimize PostgreSQL for Production**

Add to `postgresql.conf`:

```conf
# Connection Settings
max_connections = 200
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 64MB

# Write Performance
wal_buffers = 64MB
checkpoint_completion_target = 0.9

# Query Optimization
random_page_cost = 1.1
effective_io_concurrency = 200
```

---

## Redis Configuration

### Production Redis Settings

```conf
# /etc/redis/redis.conf

# Memory
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec

# Security
requirepass your-strong-password
rename-command CONFIG ""
rename-command FLUSHALL ""

# Performance
tcp-keepalive 300
timeout 0
```

---

## LLM Provider Configuration

### Multi-Provider Setup

Configure multiple providers for failover:

```python
# backend-api/app/core/config.py
LLM_PROVIDERS = {
    "primary": {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
    "secondary": {
        "provider": "google",
        "model": "gemini-2.0-flash",
        "api_key_env": "GOOGLE_API_KEY",
    },
    "fallback": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
}
```

### Provider Health Checks

The circuit breaker automatically handles provider failures:

```python
# Automatic failover when primary fails
# See: backend-api/app/services/llm_service.py
```

---

## Security Hardening

### 1. API Security

```bash
# Ensure security middleware is enabled in production
# backend-api/app/main.py - Lines 379-427 should be uncommented
```

### 2. NGINX SSL Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name chimera.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location /api/ {
        proxy_pass http://api:8001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        proxy_pass http://frontend:3700;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Firewall Rules

```bash
# Allow only necessary ports
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw enable
```

---

## Monitoring & Observability

### Prometheus Metrics

Chimera exposes metrics at `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'chimera-api'
    static_configs:
      - targets: ['api:8001']
    metrics_path: /metrics
```

### Key Metrics to Monitor

| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| `http_requests_total` | N/A | Total request count |
| `http_request_duration_seconds` | P95 > 2s | Request latency |
| `circuit_breaker_state` | open | Provider health |
| `llm_tokens_used_total` | Budget limit | Token consumption |
| `cache_hit_rate` | < 50% | Cache efficiency |

### Grafana Dashboard

Import the Chimera dashboard from `monitoring/grafana/dashboards/chimera.json`.

---

## Backup & Recovery

### Database Backup

```bash
# Daily backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump -h db -U chimera chimera | gzip > /backups/chimera_${TIMESTAMP}.sql.gz

# Retain last 30 days
find /backups -name "chimera_*.sql.gz" -mtime +30 -delete
```

### Redis Backup

```bash
# Redis RDB snapshot
redis-cli -a ${REDIS_PASSWORD} BGSAVE
cp /var/lib/redis/dump.rdb /backups/redis_$(date +%Y%m%d).rdb
```

### Recovery Procedure

1. Stop services
2. Restore database from backup
3. Restore Redis state (optional)
4. Run migrations if needed
5. Start services
6. Verify health checks

---

## Scaling Guidelines

### Horizontal Scaling

| Component | Scale When | Max Instances |
|-----------|------------|---------------|
| API | CPU > 70% | 10 |
| Frontend | Memory > 80% | 5 |
| Workers | Queue depth > 100 | 20 |

### Vertical Scaling

| Component | Start | Scale To |
|-----------|-------|----------|
| API | 4 CPU, 8GB | 16 CPU, 32GB |
| Database | 2 CPU, 4GB | 8 CPU, 32GB |
| Redis | 1 CPU, 2GB | 4 CPU, 8GB |

---

## Troubleshooting

### Common Issues

#### 1. API Not Responding

```bash
# Check container logs
docker logs chimera-api --tail 100

# Check health endpoint
curl http://localhost:8001/health

# Check database connection
docker exec chimera-api python -c "from app.core.database import engine; print(engine.url)"
```

#### 2. High Latency

```bash
# Check Redis connection
redis-cli -a ${REDIS_PASSWORD} ping

# Check cache hit rate
curl http://localhost:8001/metrics | grep cache_hit

# Check circuit breaker state
curl http://localhost:8001/api/v1/health/providers
```

#### 3. LLM Provider Errors

```bash
# Check provider status
curl http://localhost:8001/api/v1/providers/status

# View circuit breaker state
curl http://localhost:8001/api/v1/health/circuit-breakers

# Test specific provider
curl -X POST http://localhost:8001/api/v1/test-provider \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai"}'
```

### Log Locations

| Service | Location |
|---------|----------|
| API | `/var/log/chimera/api.log` |
| Frontend | `/var/log/chimera/frontend.log` |
| NGINX | `/var/log/nginx/access.log` |

---

## Support

For production support issues:

- **Documentation**: https://docs.chimera.example.com
- **Issues**: https://github.com/your-org/chimera/issues
- **Security**: security@chimera.example.com

---

**Last Updated**: January 2026
**Version**: 1.0.0
