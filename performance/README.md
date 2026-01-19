# Performance Testing and Monitoring Setup Guide for Chimera AI

This guide provides comprehensive instructions for setting up and running the complete performance monitoring stack for Chimera AI system.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 16+ and npm
- Python 3.11+ with pip/poetry
- k6 load testing tool
- Artillery.js (optional)
- py-spy for flame graph generation

### Installation Commands

```bash
# Install k6 (Linux/macOS)
sudo apt-get install k6  # Ubuntu/Debian
brew install k6         # macOS

# Install Artillery.js
npm install -g artillery

# Install Python profiling tools
pip install py-spy memory-profiler matplotlib pandas

# Install additional monitoring dependencies
pip install prometheus-client opentelemetry-api opentelemetry-sdk
```

## 📊 Performance Monitoring Stack

### 1. Start Monitoring Infrastructure

```bash
# Start the complete monitoring stack
cd monitoring
docker-compose -f ../performance/docker-compose.monitoring.yml up -d

# Verify services are running
docker-compose ps

# Access web interfaces:
# Grafana: http://localhost:3001 (admin/admin123!)
# Prometheus: http://localhost:9090
# Jaeger: http://localhost:16686
# AlertManager: http://localhost:9093
```

### 2. Backend Performance Monitoring

```bash
# Navigate to backend
cd backend-api

# Install monitoring dependencies
pip install -r requirements.txt

# Add to your main.py or startup script:
from app.core.performance_monitoring import performance_collector
from app.core.profiling import global_profiler

# Start application with monitoring
python run.py --enable-monitoring
```

### 3. Frontend Performance Monitoring

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Add to your main layout or _app.tsx:
import { performanceMonitor } from '@/lib/performance-monitor';

# Initialize monitoring
performanceMonitor.initialize();

# Wrap components with performance profiler:
import PerformanceProfiler from '@/components/PerformanceProfiler';

export default function MyPage() {
  return (
    <PerformanceProfiler id="my-page">
      <YourContent />
    </PerformanceProfiler>
  );
}
```

## 🔥 Running Performance Tests

### Load Testing with k6

```bash
# Basic load test
cd performance/load-testing
k6 run k6-chimera-test.js

# Custom configuration
k6 run k6-chimera-test.js \
  --env BASE_URL=http://localhost:8001 \
  --env API_KEY=your-api-key \
  --env FRONTEND_URL=http://localhost:3001

# Extended stress test
k6 run k6-chimera-test.js --vus 100 --duration 15m

# Generate detailed reports
k6 run --out json=results.json k6-chimera-test.js
```

### Load Testing with Artillery

```bash
# Run Artillery test
cd performance/load-testing
artillery run artillery-config.yml

# Custom target
artillery run artillery-config.yml --target http://localhost:8001

# Generate HTML report
artillery run artillery-config.yml --output report.json
artillery report report.json
```

## 🔍 Profiling and Analysis

### CPU Flame Graph Generation

```bash
# Profile running Chimera backend
cd performance/analysis
python flame_graph_generator.py

# Profile specific functions
python flame_graph_generator.py sample

# Manual py-spy profiling
py-spy record -o flamegraph.svg --pid $(pgrep -f "python.*chimera") --duration 60
```

### Memory Analysis

```bash
# Run memory profiler
python flame_graph_generator.py

# Manual memory tracking
python -c "
from app.core.profiling import global_profiler
global_profiler.memory_profiler.start_tracking()
# ... run your application
global_profiler.memory_profiler.take_snapshot('test')
"
```

### Database Performance Analysis

```bash
# Run database profiler
cd performance/analysis
python db_io_profiler.py

# Configure database connection in script:
# db_connection = 'postgresql://user:pass@localhost:5432/chimera'
# redis_url = 'redis://localhost:6379/0'
```

## 📈 Dashboard Access

### Grafana Dashboard
- **URL**: http://localhost:3001
- **Login**: admin / admin123!
- **Main Dashboard**: "Chimera AI Performance Baseline Dashboard"

### Key Metrics Panels:
1. **Core Web Vitals**: FCP, LCP, FID, CLS
2. **API Response Times**: P50, P95, P99
3. **LLM Provider Performance**: Duration and request rates
4. **Transformation Performance**: Technique-specific metrics
5. **Error Rates**: HTTP and LLM error percentages
6. **Resource Usage**: Memory and CPU utilization
7. **Circuit Breaker Status**: Provider health monitoring
8. **WebSocket Performance**: Connection and message metrics
9. **AutoDAN Performance**: Optimization duration and rates
10. **Cache Performance**: Hit/miss rates
11. **Database Performance**: Transaction rates and connections
12. **Data Pipeline**: ETL job performance

### Prometheus Queries
Access Prometheus at http://localhost:9090 and use these sample queries:

```promql
# API response time 95th percentile
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="chimera-backend"}[5m]))

# LLM provider error rate
rate(chimera_llm_requests_total{status="error"}[5m]) / rate(chimera_llm_requests_total[5m])

# Memory usage trend
chimera_memory_usage_bytes{type="rss"}

# Transformation performance by technique
avg(chimera_transformation_duration_seconds) by (technique)
```

## 🎯 Critical User Journey Testing

### 1. Prompt Generation Journey
```bash
# Test basic prompt enhancement
curl -X POST http://localhost:8001/api/v1/enhance \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"prompt": "Create viral content", "enhancement_type": "comprehensive"}'

# Measure with curl timing
curl -w "@curl-format.txt" -X POST http://localhost:8001/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"prompt": "Test prompt", "provider": "mock"}'
```

### 2. Transformation Testing
```bash
# Test transformation techniques
curl -X POST http://localhost:8001/api/v1/transform \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"prompt": "Test", "techniques": ["advanced", "cognitive_hacking"]}'
```

### 3. Jailbreak Testing
```bash
# Test AutoDAN optimization
curl -X POST http://localhost:8001/api/v1/autodan/optimize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"target_prompt": "Test prompt", "method": "vanilla"}'
```

## 📊 Performance Baselines

### Expected Performance Thresholds

| Metric | Good | Needs Improvement | Poor |
|--------|------|-------------------|------|
| API Response Time (P95) | < 1s | 1-3s | > 3s |
| LLM Provider Response (P95) | < 5s | 5-15s | > 15s |
| Transformation Time (P95) | < 500ms | 500ms-2s | > 2s |
| Frontend FCP | < 1.8s | 1.8-3s | > 3s |
| Frontend LCP | < 2.5s | 2.5-4s | > 4s |
| Frontend FID | < 100ms | 100-300ms | > 300ms |
| Frontend CLS | < 0.1 | 0.1-0.25 | > 0.25 |
| Error Rate | < 1% | 1-5% | > 5% |
| Memory Growth | < 10% | 10-20% | > 20% |

### Setting Up Alerts

```yaml
# Add to monitoring/prometheus/rules/chimera-alerts.yml
groups:
- name: performance-sla
  rules:
  - alert: PerformanceBaseline
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 3
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "API performance below baseline"
```

## 🔧 Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory leaks
   python performance/analysis/flame_graph_generator.py
   # Review memory timeline charts
   ```

2. **Slow API Responses**
   ```bash
   # Check flame graphs for bottlenecks
   py-spy top --pid $(pgrep -f chimera)
   ```

3. **Database Performance Issues**
   ```bash
   # Run database analysis
   python performance/analysis/db_io_profiler.py
   ```

### Optimization Recommendations

1. **Enable Connection Pooling**: Configure appropriate pool sizes for database and Redis
2. **Implement Caching**: Use Redis for frequently accessed data
3. **Optimize Queries**: Use EXPLAIN ANALYZE for slow queries
4. **Bundle Optimization**: Analyze frontend bundles with webpack-bundle-analyzer
5. **Resource Limits**: Set appropriate CPU/memory limits for containers

## 📝 Generating Reports

### Automated Report Generation
```bash
# Generate comprehensive performance report
python performance/analysis/generate_report.py

# Schedule regular reports (crontab)
0 6 * * * cd /path/to/chimera && python performance/analysis/generate_report.py
```

### Manual Analysis
```bash
# Export Prometheus metrics
curl http://localhost:9090/api/v1/query_range?query=up&start=2024-01-01T00:00:00Z&end=2024-01-02T00:00:00Z&step=15s

# Export Grafana dashboard data
curl -u admin:admin123! http://localhost:3001/api/dashboards/uid/chimera-performance
```

This comprehensive setup provides complete visibility into Chimera AI's performance across all critical components and user journeys, enabling data-driven optimization decisions and proactive performance management.
