# Chimera Production Deployment Guide

This guide provides step-by-step instructions for deploying the Chimera AI security testing platform in production environments with enterprise-grade security, performance, and scalability.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Security Hardening](#security-hardening)
3. [Database Setup](#database-setup)
4. [Environment Configuration](#environment-configuration)
5. [Container Deployment](#container-deployment)
6. [Load Balancer & SSL](#load-balancer--ssl)
7. [Monitoring & Observability](#monitoring--observability)
8. [Backup & Recovery](#backup--recovery)
9. [Security Validation](#security-validation)
10. [Maintenance & Updates](#maintenance--updates)

## Prerequisites

### Infrastructure Requirements

**Minimum Production Specifications:**
- **CPU**: 8 cores (16 vCPUs recommended)
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 100GB SSD minimum (500GB recommended)
- **Network**: 1Gbps bandwidth
- **OS**: Ubuntu 22.04 LTS, RHEL 8+, or equivalent

**Required Services:**
- PostgreSQL 14+ (managed service recommended)
- Redis Cluster 6.2+ (managed service recommended)
- Load Balancer (AWS ALB, NGINX, or similar)
- Container Runtime (Docker 20.10+, containerd)
- Container Orchestrator (Kubernetes 1.25+ recommended)

### Software Dependencies

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Kubernetes (if using K8s)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

## Security Hardening

### 1. System Security

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install security tools
sudo apt install -y fail2ban ufw aide rkhunter

# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw --force enable

# Configure fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 2. Generate Production Secrets

```bash
# Generate secure JWT secret (64 characters)
python3 -c "import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(64))"

# Generate API keys (32 characters each)
python3 -c "import secrets; print('API_KEY=' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('CHIMERA_API_KEY=' + secrets.token_urlsafe(32))"

# Generate encryption key (64 hex characters)
python3 -c "import secrets; print('CHIMERA_ENCRYPTION_KEY=' + secrets.token_hex(32))"

# Generate database password (24 characters)
python3 -c "import secrets, string; chars = string.ascii_letters + string.digits + '!@#$%^&*'; print('DB_PASSWORD=' + ''.join(secrets.choice(chars) for _ in range(24)))"
```

### 3. SSL/TLS Certificate Setup

**Option A: Let's Encrypt (Recommended for most deployments)**
```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Generate certificates
sudo certbot certonly --standalone -d api.yourdomain.com -d app.yourdomain.com

# Set up auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

**Option B: Custom Certificate Authority**
```bash
# Generate private key
openssl genrsa -out chimera.key 4096

# Generate certificate signing request
openssl req -new -key chimera.key -out chimera.csr

# Generate self-signed certificate (for internal use)
openssl x509 -req -days 365 -in chimera.csr -signkey chimera.key -out chimera.crt

# Set proper permissions
chmod 600 chimera.key
chmod 644 chimera.crt
```

## Database Setup

### PostgreSQL Production Configuration

```sql
-- Create database and user
CREATE USER chimera_user WITH PASSWORD 'SECURE_DB_PASSWORD_FROM_SECRETS';
CREATE DATABASE chimera_prod OWNER chimera_user;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE chimera_prod TO chimera_user;
GRANT ALL ON SCHEMA public TO chimera_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO chimera_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO chimera_user;

-- Optimize PostgreSQL for production
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Reload configuration
SELECT pg_reload_conf();
```

### Redis Cluster Setup

```bash
# Redis cluster configuration (redis.conf)
cat > redis-cluster.conf << EOF
port 6379
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
requirepass SECURE_REDIS_PASSWORD_FROM_SECRETS
maxmemory 2gb
maxmemory-policy allkeys-lru
EOF

# Start Redis cluster nodes
redis-server redis-cluster.conf
```

## Environment Configuration

### 1. Copy and Configure Production Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/chimera.git
cd chimera

# Copy production environment template
cp .env.production .env

# Edit configuration with your secrets
nano .env
```

### 2. Essential Production Settings

Update `.env` with the generated secrets and your infrastructure details:

```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Security (Use generated secrets)
JWT_SECRET=YOUR_GENERATED_JWT_SECRET_64_CHARS
API_KEY=YOUR_GENERATED_API_KEY
CHIMERA_API_KEY=YOUR_GENERATED_CHIMERA_API_KEY

# Database
DATABASE_URL=postgresql://chimera_user:SECURE_DB_PASSWORD@db-host:5432/chimera_prod

# Redis
REDIS_URL=redis://:SECURE_REDIS_PASSWORD@redis-cluster:6379/0

# CORS (Update with your domains)
ALLOWED_ORIGINS=https://app.yourdomain.com,https://api.yourdomain.com

# AI Provider Keys (Add your production keys)
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GOOGLE_API_KEY=your-google-api-key
```

## Container Deployment

### Docker Compose Production Deployment

Create `docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  chimera-backend:
    build:
      context: ./backend-api
      dockerfile: Dockerfile.production
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env
    ports:
      - "8001:8001"
    volumes:
      - ./logs:/app/logs
      - /etc/ssl/certs:/etc/ssl/certs:ro
      - /etc/ssl/private:/etc/ssl/private:ro
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  chimera-frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.production
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - NEXT_TELEMETRY_DISABLED=1
    depends_on:
      - chimera-backend
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3001/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: chimera_prod
      POSTGRES_USER: chimera_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    ports:
      - "5432:5432"
    command: >
      postgres
      -c shared_buffers=256MB
      -c max_connections=200
      -c effective_cache_size=1GB

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - /etc/ssl/certs:/etc/ssl/certs:ro
      - /etc/ssl/private:/etc/ssl/private:ro
    depends_on:
      - chimera-backend
      - chimera-frontend

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    driver: bridge
```

### Kubernetes Deployment

Create `k8s/production/` directory with manifests:

```yaml
# k8s/production/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: chimera-prod
  labels:
    environment: production
    app: chimera

---
# k8s/production/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chimera-config
  namespace: chimera-prod
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "WARNING"
  LOG_FORMAT: "json"

---
# k8s/production/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: chimera-secrets
  namespace: chimera-prod
type: Opaque
stringData:
  JWT_SECRET: "YOUR_JWT_SECRET"
  API_KEY: "YOUR_API_KEY"
  DATABASE_URL: "postgresql://user:pass@host:5432/chimera_prod"
  REDIS_PASSWORD: "YOUR_REDIS_PASSWORD"

---
# k8s/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chimera-backend
  namespace: chimera-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chimera-backend
  template:
    metadata:
      labels:
        app: chimera-backend
    spec:
      containers:
      - name: chimera-backend
        image: chimera/backend:latest
        ports:
        - containerPort: 8001
        envFrom:
        - configMapRef:
            name: chimera-config
        - secretRef:
            name: chimera-secrets
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
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Load Balancer & SSL

### NGINX Configuration

Create `nginx/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream backend {
        least_conn;
        server chimera-backend:8001 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    upstream frontend {
        least_conn;
        server chimera-frontend:3000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/m;

    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # API server
    server {
        listen 443 ssl http2;
        server_name api.yourdomain.com;

        ssl_certificate /etc/ssl/certs/chimera.crt;
        ssl_certificate_key /etc/ssl/private/chimera.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_prefer_server_ciphers off;

        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /auth/ {
            limit_req zone=auth burst=5 nodelay;
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }

    # Frontend server
    server {
        listen 443 ssl http2;
        server_name app.yourdomain.com;

        ssl_certificate /etc/ssl/certs/chimera.crt;
        ssl_certificate_key /etc/ssl/private/chimera.key;

        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }

    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name api.yourdomain.com app.yourdomain.com;
        return 301 https://$server_name$request_uri;
    }
}
```

## Monitoring & Observability

### 1. Prometheus Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'chimera-backend'
    static_configs:
      - targets: ['chimera-backend:9090']
    metrics_path: /metrics
    scrape_interval: 30s

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

### 2. Grafana Dashboard

Import dashboard for Chimera monitoring:
- **Backend Performance**: Response times, throughput, error rates
- **AI Provider Health**: Provider availability, response times, costs
- **Security Metrics**: Failed authentication attempts, rate limit violations
- **Infrastructure**: CPU, memory, disk, network utilization

### 3. Alerting Rules

Create `monitoring/alert_rules.yml`:

```yaml
groups:
  - name: chimera.rules
    rules:
      - alert: ChimeraBackendDown
        expr: up{job="chimera-backend"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Chimera backend is down"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: DatabaseConnectionFailure
        expr: postgres_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL database is unreachable"
```

## Backup & Recovery

### 1. Database Backup

Create automated backup script `scripts/backup-db.sh`:

```bash
#!/bin/bash

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="chimera_prod"
DB_USER="chimera_user"

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump -h postgres -U $DB_USER -d $DB_NAME -f "$BACKUP_DIR/chimera_backup_$DATE.sql"

# Compress backup
gzip "$BACKUP_DIR/chimera_backup_$DATE.sql"

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR/chimera_backup_$DATE.sql.gz" "s3://chimera-backups/production/"

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "chimera_backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: chimera_backup_$DATE.sql.gz"
```

### 2. Backup Cron Job

```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * /opt/chimera/scripts/backup-db.sh >> /var/log/chimera-backup.log 2>&1
```

### 3. Recovery Procedure

```bash
#!/bin/bash
# Recovery script: scripts/restore-db.sh

BACKUP_FILE=$1
DB_NAME="chimera_prod"
DB_USER="chimera_user"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    exit 1
fi

# Stop application
docker-compose -f docker-compose.production.yml stop chimera-backend

# Restore database
gunzip -c $BACKUP_FILE | psql -h postgres -U $DB_USER -d $DB_NAME

# Start application
docker-compose -f docker-compose.production.yml start chimera-backend

echo "Database restored from $BACKUP_FILE"
```

## Security Validation

### 1. Security Checklist

**Pre-deployment Security Audit:**

- [ ] All default passwords changed
- [ ] SSL/TLS certificates installed and configured
- [ ] Firewall rules properly configured
- [ ] Database access restricted to application only
- [ ] API keys encrypted at rest
- [ ] Rate limiting enabled and tested
- [ ] Security headers configured in load balancer
- [ ] Backup encryption enabled
- [ ] Log retention and monitoring configured
- [ ] Vulnerability scanning completed

### 2. Security Testing

```bash
# Run security scan
docker run --rm -v $(pwd):/app clair-scanner:latest /app

# Test SSL configuration
nmap --script ssl-enum-ciphers -p 443 api.yourdomain.com

# Test rate limiting
ab -n 1000 -c 50 https://api.yourdomain.com/api/v1/health

# Vulnerability assessment
nuclei -target https://api.yourdomain.com -templates nuclei-templates/
```

### 3. Compliance Validation

**GDPR Compliance:**
- [ ] Data retention policies implemented
- [ ] User data encryption at rest and in transit
- [ ] Right to erasure functionality tested
- [ ] Data processing audit logs enabled

**SOC 2 Compliance:**
- [ ] Access controls implemented and tested
- [ ] Audit logging enabled for all administrative actions
- [ ] Incident response procedures documented
- [ ] Regular security assessments scheduled

## Maintenance & Updates

### 1. Update Procedure

```bash
#!/bin/bash
# Update script: scripts/update-chimera.sh

# Backup current deployment
./scripts/backup-db.sh

# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Rolling update (zero downtime)
docker-compose -f docker-compose.production.yml up -d --no-deps chimera-backend
docker-compose -f docker-compose.production.yml up -d --no-deps chimera-frontend

# Verify health
curl -f https://api.yourdomain.com/health || {
    echo "Health check failed, rolling back..."
    docker-compose -f docker-compose.production.yml rollback
    exit 1
}

echo "Update completed successfully"
```

### 2. Health Monitoring

Set up continuous health monitoring:

```bash
#!/bin/bash
# Health monitoring script: scripts/health-check.sh

ENDPOINTS=(
    "https://api.yourdomain.com/health"
    "https://api.yourdomain.com/health/ready"
    "https://app.yourdomain.com"
)

for endpoint in "${ENDPOINTS[@]}"; do
    if ! curl -f -s --max-time 10 "$endpoint" > /dev/null; then
        echo "ERROR: $endpoint is not responding"
        # Send alert (Slack, email, etc.)
        curl -X POST -H 'Content-type: application/json' \
             --data "{\"text\":\"🚨 Chimera Health Check Failed: $endpoint\"}" \
             "$SLACK_WEBHOOK_URL"
    fi
done
```

### 3. Performance Monitoring

Monitor key performance metrics:

```bash
# CPU and memory usage
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Database performance
docker exec postgres psql -U chimera_user -d chimera_prod -c "
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;"

# Redis performance
docker exec redis redis-cli info stats
```

## Troubleshooting

### Common Issues

**1. Database Connection Issues**
```bash
# Check database connectivity
docker exec chimera-backend python -c "
from app.core.config import settings
import psycopg2
try:
    conn = psycopg2.connect(settings.DATABASE_URL)
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
"
```

**2. AI Provider Issues**
```bash
# Test provider connectivity
curl -X POST https://api.yourdomain.com/api/v1/providers \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json"
```

**3. Performance Issues**
```bash
# Check application logs
docker logs chimera-backend --tail 100 -f

# Monitor resource usage
docker exec chimera-backend top -p 1
```

For additional support, consult the [Chimera Documentation](https://docs.chimera.ai) or contact the development team.

---

**Security Notice**: This deployment guide includes security best practices, but always conduct a thorough security audit before deploying to production. Regularly update all components and monitor for security vulnerabilities.
