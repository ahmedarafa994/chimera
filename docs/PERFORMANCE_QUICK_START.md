# Performance Analysis Quick Start Guide

## Overview

This document provides quick-start instructions for performance profiling and load testing of the Chimera platform.

## Performance Profiling

### Run Comprehensive Profile

```bash
# Profile all modules (60 seconds each)
cd backend-api
python scripts/performance_profiler.py --all --duration 60 --output-dir profiles

# Results saved to: profiles/performance_report_YYYYMMDD_HHMMSS.json
```

### Profile Specific Module

```bash
# Profile LLM service
python scripts/performance_profiler.py --module llm_service --duration 60

# Profile transformation service
python scripts/performance_profiler.py --module transformation --duration 60

# Profile AutoDAN service
python scripts/performance_profiler.py --module autodan --duration 60
```

### Profile API Endpoint

```bash
# Profile /api/v1/generate endpoint (100 requests)
python scripts/performance_profiler.py --endpoint /api/v1/generate --requests 100

# Results show:
# - Total requests
# - Average latency
# - Throughput (req/s)
# - Top 10 hotspots
```

### Memory Profiling

```bash
# Profile memory usage
pip install memory_profiler matplotlib
python scripts/performance_profiler.py --memory

# Results:
# - Peak memory usage
# - Average memory usage
# - Memory plot saved to profiles/memory_usage_YYYYMMDD_HHMMSS.png
```

## Load Testing

### Start Backend Server

```bash
# Terminal 1: Start backend
cd backend-api
python run.py
```

### Run Locust Web UI

```bash
# Terminal 2: Start Locust with web UI
cd backend-api
locust -f tests/load/locustfile.py --host http://localhost:8001

# Open browser to: http://localhost:8089
```

### Run Headless Load Test

```bash
# Basic test (100 users, 10 spawn rate, 5 minutes)
locust -f tests/load/locustfile.py \
  --host http://localhost:8001 \
  --headless \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m \
  --html reports/load_test_report.html

# Results saved to: reports/load_test_report.html
```

### Run Cache Performance Test

```bash
# Test cache effectiveness with repeated requests
locust -f tests/load/locustfile.py \
  --host http://localhost:8001 \
  --headless \
  --users 500 \
  --spawn-rate 50 \
  --run-time 3m \
  --user-class CacheTestUser \
  --html reports/cache_test_report.html
```

### Run AutoDAN Stress Test

```bash
# Test AutoDAN performance (CPU-intensive)
locust -f tests/load/locustfile.py \
  --host http://localhost:8001 \
  --headless \
  --users 20 \
  --spawn-rate 1 \
  --run-time 10m \
  --user-class AutoDANUser \
  --html reports/autodan_test_report.html
```

## Performance Targets

### Current Baseline (Estimated)

| Metric | Current Value |
|--------|---------------|
| Throughput | 10-50 req/s |
| P50 Latency | 200-500ms |
| P95 Latency | 500-1000ms |
| P99 Latency | 1000-2000ms |
| Error Rate | 5-10% |
| Cache Hit Rate | 0% |
| Bundle Size | 450KB |

### Target Performance (After Optimization)

| Metric | Target Value | Improvement |
|--------|--------------|-------------|
| Throughput | 500-1000 req/s | 10-20x |
| P50 Latency | 50-100ms | 5-10x |
| P95 Latency | 100-200ms | 5-10x |
| P99 Latency | <500ms | 4x |
| Error Rate | <1% | 5-10x |
| Cache Hit Rate | 80-90% | New |
| Bundle Size | <150KB | 70% reduction |

## Key Metrics to Monitor

### System Metrics

- **CPU Utilization**: <70% average, <90% peak
- **Memory Usage**: <4GB per instance
- **Disk I/O**: <100MB/s read, <50MB/s write
- **Network I/O**: <500Mbps in, <200Mbps out

### Application Metrics

- **Request Rate**: Monitor for sudden spikes
- **Response Time**: P50, P95, P99 percentiles
- **Error Rate**: Should be <1%
- **Cache Hit Rate**: Should be >80%
- **Circuit Breaker State**: Monitor for open circuits

### Business Metrics

- **LLM API Calls**: Per provider, per model
- **Token Usage**: Total, per provider
- **Transformation Success Rate**: Should be >95%
- **Jailbreak Success Rate**: Research metric

## Common Performance Issues

### Issue 1: High Latency

**Symptoms**: P95 latency >500ms

**Diagnosis**:
```bash
# Profile to find hotspots
python scripts/performance_profiler.py --all --duration 30

# Check top 10 hotspots in report
```

**Solutions**:
- Implement Redis cache (300-400% improvement)
- Add HTTP connection pooling (-200ms latency)
- Optimize database queries (if applicable)

### Issue 2: Low Throughput

**Symptoms**: <50 req/s despite low CPU usage

**Diagnosis**:
```bash
# Check for blocking operations
python scripts/performance_profiler.py --module llm_service --duration 60

# Look for long-running functions in hotspots
```

**Solutions**:
- Use dedicated thread pool for AutoDAN
- Implement parallel processing
- Increase concurrency limits

### Issue 3: Memory Leaks

**Symptoms**: Memory usage increases over time

**Diagnosis**:
```bash
# Memory profiling
python scripts/performance_profiler.py --memory

# Check for unbounded growth in memory plot
```

**Solutions**:
- Implement LRU cache for circuit breakers
- Add memory limits to caches
- Clean up idle connections

### Issue 4: High Error Rate

**Symptoms**: >5% error rate

**Diagnosis**:
```bash
# Check logs for common errors
tail -f logs/chimera.log | grep ERROR

# Check circuit breaker states
curl http://localhost:8001/api/v1/providers
```

**Solutions**:
- Implement provider failover
- Add retry logic with exponential backoff
- Optimize timeout values

## Performance Optimization Checklist

### Phase 1: Critical Fixes (Week 1-2)

- [ ] Implement Redis cache
- [ ] Dedicated thread pool for AutoDAN
- [ ] API client code splitting
- [ ] HTTP connection pooling
- [ ] Parallel batch processing

### Phase 2: Scalability (Week 3-4)

- [ ] Redis rate limiting
- [ ] Kubernetes deployment
- [ ] Load balancer configuration
- [ ] Z-order clustering for Delta Lake
- [ ] Parallel Airflow tasks

### Phase 3: Advanced Optimization (Week 5-8)

- [ ] Adaptive TTL cache
- [ ] Priority task queue
- [ ] ML-based provider selection
- [ ] React Query integration
- [ ] Component memoization

### Phase 4: Monitoring (Week 9-10)

- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] Alerting rules
- [ ] Distributed tracing
- [ ] Performance regression tests

## Resources

### Documentation

- [Full Performance Analysis Report](./PERFORMANCE_ANALYSIS_REPORT.md)
- [Data Pipeline Architecture](./DATA_PIPELINE_ARCHITECTURE.md)
- [API Documentation](../backend-api/docs/API.md)

### Tools

- **cProfile**: Built-in Python profiler
- **memory_profiler**: Memory usage profiling
- **Locust**: Load testing framework
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization

### Scripts

- `scripts/performance_profiler.py`: Performance profiling
- `tests/load/locustfile.py`: Load testing scenarios
- `tests/load/run_load_test.py`: Load test runner

## Support

For performance-related questions or issues:
1. Check the full performance analysis report
2. Review profiling results
3. Consult load test reports
4. Contact the performance engineering team
