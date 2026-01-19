# LLM Integration System - Deployment Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Server](#running-the-server)
5. [Testing](#testing)
6. [Production Deployment](#production-deployment)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **Python:** 3.8 or higher
- **RAM:** 2GB minimum, 4GB recommended
- **CPU:** 2 cores minimum
- **Storage:** 1GB free space

### Dependencies
- Flask 3.0.0
- Requests 2.31.0
- Flask-CORS 4.0.0
- Python-dateutil 2.8.2

---

## Installation

### 1. Clone Repository

```bash
cd Project_Chimera
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements_llm.txt
```

### 4. Verify Installation

```bash
python -c "import flask; print(f'Flask {flask.__version__} installed')"
```

---

## Configuration

### Environment Variables

Create a `.env` file in the `Project_Chimera` directory:

```bash
# API Configuration
PORT=5000
DEBUG=False
CHIMERA_API_KEY=your_secure_api_key_here

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# Anthropic Configuration
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
ANTHROPIC_BASE_URL=https://api.anthropic.com/v1
ANTHROPIC_MODEL=claude-3-opus-20240229

# Custom Model Configuration (Optional)
CUSTOM_API_KEY=your-custom-key
CUSTOM_BASE_URL=http://localhost:8001
CUSTOM_MODEL=custom-model-name

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# Batch Processing
MAX_BATCH_WORKERS=5
MAX_QUEUE_SIZE=1000

# Caching
CACHE_ENABLED=True
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=1000

# Monitoring
METRICS_WINDOW_MINUTES=60
```

### Configuration File (Optional)

Create `config.json`:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": false
  },
  "providers": {
    "openai": {
      "enabled": true,
      "rate_limit": 60,
      "timeout": 60,
      "max_retries": 3
    },
    "anthropic": {
      "enabled": true,
      "rate_limit": 50,
      "timeout": 60,
      "max_retries": 3
    }
  },
  "batch_processing": {
    "max_workers": 5,
    "max_queue_size": 1000
  },
  "monitoring": {
    "enabled": true,
    "window_minutes": 60,
    "export_interval": 3600
  }
}
```

---

## Running the Server

### Development Mode

```bash
# With environment variables from .env
python api_server.py

# Or explicitly set variables
export CHIMERA_API_KEY=your_key
export OPENAI_API_KEY=your_openai_key
python api_server.py
```

### Production Mode

```bash
# Using Gunicorn (recommended)
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 api_server:app

# With more options
gunicorn \
  --workers 4 \
  --bind 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile access.log \
  --error-logfile error.log \
  --log-level info \
  api_server:app
```

### Using Systemd (Linux)

Create `/etc/systemd/system/chimera-api.service`:

```ini
[Unit]
Description=Project Chimera LLM API
After=network.target

[Service]
Type=simple
User=chimera
WorkingDirectory=/opt/Project_Chimera
Environment="PATH=/opt/Project_Chimera/venv/bin"
EnvironmentFile=/opt/Project_Chimera/.env
ExecStart=/opt/Project_Chimera/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable chimera-api
sudo systemctl start chimera-api
sudo systemctl status chimera-api
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements_llm.txt .
RUN pip install --no-cache-dir -r requirements_llm.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "api_server:app"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chimera-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - CHIMERA_API_KEY=${CHIMERA_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DEBUG=False
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Build and run:

```bash
docker-compose up -d
docker-compose logs -f
```

---

## Testing

### Health Check

```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "operational",
  "health": {
    "status": "healthy"
  }
}
```

### Test Authentication

```bash
# Valid API key
curl -H "X-API-Key: your_api_key" \
  http://localhost:5000/api/v1/providers

# Invalid API key (should return 403)
curl -H "X-API-Key: wrong_key" \
  http://localhost:5000/api/v1/providers
```

### Test Transformation

```bash
curl -X POST http://localhost:5000/api/v1/transform \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "Test prompt",
    "potency_level": 5,
    "technique_suite": "subtle_persuasion"
  }'
```

### Test Execution (requires valid LLM API keys)

```bash
curl -X POST http://localhost:5000/api/v1/execute \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "Explain quantum computing",
    "potency_level": 5,
    "technique_suite": "academic_research",
    "provider": "openai"
  }'
```

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils  # Linux
brew install httpd  # Mac

# Run load test
ab -n 1000 -c 10 -H "X-API-Key: your_key" \
  http://localhost:5000/health
```

---

## Production Deployment

### Security Checklist

- [ ] Change default API key
- [ ] Use HTTPS (SSL/TLS)
- [ ] Enable firewall rules
- [ ] Set DEBUG=False
- [ ] Configure CORS properly
- [ ] Implement request size limits
- [ ] Set up log rotation
- [ ] Enable API rate limiting
- [ ] Use secure environment variable storage
- [ ] Regular security updates

### Nginx Reverse Proxy

Install Nginx:

```bash
sudo apt-get install nginx
```

Create `/etc/nginx/sites-available/chimera-api`:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Logging
    access_log /var/log/nginx/chimera_access.log;
    error_log /var/log/nginx/chimera_error.log;
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/chimera-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### SSL Certificate (Let's Encrypt)

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

---

## Monitoring

### View Metrics

```bash
# Get system metrics
curl -H "X-API-Key: your_key" \
  http://localhost:5000/api/v1/metrics

# Get provider metrics
curl -H "X-API-Key: your_key" \
  "http://localhost:5000/api/v1/metrics/providers?provider=openai"

# Export metrics
curl -H "X-API-Key: your_key" \
  http://localhost:5000/api/v1/metrics/export \
  -o metrics.json
```

### Log Monitoring

```bash
# View real-time logs
tail -f /var/log/chimera/api.log

# Search for errors
grep ERROR /var/log/chimera/api.log

# Monitor with journalctl
journalctl -u chimera-api -f
```

### Health Monitoring Script

Create `monitor.sh`:

```bash
#!/bin/bash

API_URL="http://localhost:5000"
API_KEY="your_api_key"

while true; do
    response=$(curl -s -H "X-API-Key: $API_KEY" "$API_URL/health")
    status=$(echo $response | jq -r '.health.status')
    
    if [ "$status" != "healthy" ]; then
        echo "ALERT: API unhealthy at $(date)"
        echo $response | jq .
        # Send alert notification here
    fi
    
    sleep 60
done
```

### Prometheus Integration (Optional)

Install Prometheus client:

```bash
pip install prometheus-flask-exporter
```

Add to `api_server.py`:

```python
from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)
```

---

## Troubleshooting

### Common Issues

#### 1. API Key Authentication Fails

```bash
# Verify API key is set
echo $CHIMERA_API_KEY

# Check header format
curl -v -H "X-API-Key: your_key" http://localhost:5000/api/v1/providers
```

#### 2. Provider Connection Errors

```bash
# Test OpenAI connection
python -c "
from llm_provider_client import LLMClientFactory, LLMProvider
client = LLMClientFactory.from_env(LLMProvider.OPENAI)
print('OpenAI client created successfully')
"

# Check API key validity
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

#### 3. High Latency

- Check provider API status
- Verify network connectivity
- Review rate limiting settings
- Check server resources (CPU, RAM)
- Enable caching

#### 4. Memory Issues

```bash
# Monitor memory usage
free -h

# Check Python process
ps aux | grep python

# Increase worker memory limit (Gunicorn)
gunicorn --worker-class gevent --worker-connections 1000 --max-requests 1000
```

#### 5. Rate Limiting Errors

- Increase `RATE_LIMIT_PER_MINUTE`
- Add more workers
- Implement request queuing
- Use batch processing

### Debug Mode

Enable debug logging:

```python
# In api_server.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or via environment:

```bash
export DEBUG=True
python api_server.py
```

### Performance Tuning

```bash
# Optimize worker count (rule of thumb: 2-4 × CPU cores)
gunicorn -w 8 api_server:app

# Enable async workers
gunicorn -k gevent -w 4 api_server:app

# Adjust timeout for slow LLM responses
gunicorn --timeout 180 api_server:app
```

---

## Support

### Getting Help

- **Documentation:** See `LLM_INTEGRATION_API_DOCUMENTATION.md`
- **Logs:** Check `/var/log/chimera/` or application logs
- **Metrics:** Use `/api/v1/metrics` endpoint
- **Health:** Monitor `/health` endpoint

### Maintenance

#### Regular Tasks

- Review logs weekly
- Export metrics monthly
- Update dependencies quarterly
- Rotate API keys periodically
- Review rate limit settings
- Clear old cache entries

#### Backup

```bash
# Backup configuration
tar -czf chimera-config-backup.tar.gz .env config.json

# Backup metrics
curl -H "X-API-Key: $API_KEY" \
  http://localhost:5000/api/v1/metrics/export \
  -o backup/metrics-$(date +%Y%m%d).json
```

---

## Quick Reference

### Start Server
```bash
python api_server.py
```

### Stop Server
```bash
# Ctrl+C (development)
sudo systemctl stop chimera-api  # production
```

### View Logs
```bash
tail -f logs/api.log
```

### Check Status
```bash
curl http://localhost:5000/health
```

### Test API
```bash
curl -H "X-API-Key: your_key" \
  http://localhost:5000/api/v1/providers
```

---

**Last Updated:** 2024-11-21  
**Version:** 1.0.0