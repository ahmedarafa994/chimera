# Chimera Performance Optimization Guide (PERF-008)

## Overview

This guide establishes the continuous performance optimization process for the Chimera AI system. It covers monitoring practices, optimization strategies, and performance review workflows.

## Performance Metrics Dashboard

### Grafana Dashboards

Access the performance monitoring dashboards at:
- **Performance Optimization**: http://localhost:3001/d/chimera-performance-optimization
- **Performance Baseline**: http://localhost:3001/d/chimera-performance-baseline
- **Technical Operations**: http://localhost:3001/d/chimera-technical-dashboard

### Key Metrics to Monitor

| Category | Metric | Target | Alert Threshold |
|----------|--------|--------|-----------------|
| **Cache** | L1 Hit Rate | >70% | <50% |
| **Cache** | L2 Hit Rate | >80% | <50% |
| **Cache** | Combined Hit Rate | >85% | <30% |
| **Cache** | L1 Memory Utilization | <70% | >90% |
| **Connection Pool** | Utilization | <70% | >90% |
| **Connection Pool** | Avg Wait Time | <50ms | >100ms |
| **Connection Pool** | Stale Connections | <5 | >10 |
| **Connection Pool** | Failure Rate | <1% | >1% |
| **Optimizer** | Gradient Duration | <300ms | >500ms |
| **Worker Pool** | Queue Backlog | <50 | >100 |
| **Compression** | Compression Ratio | >50% | <50% |
| **Circuit Breaker** | Trip Rate | <0.05/sec | >0.1/sec |
| **SLA** | Generate API P95 | <7s | >10s |
| **SLA** | Transform API P95 | <1.5s | >2s |
| **SLA** | Throughput | >10 RPS | <10 RPS |

## Weekly Performance Review Checklist

### 1. Review Prometheus Alerts
```bash
# Check active alerts
curl http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing")'
```

### 2. Review Cache Performance
- Check L1 and L2 cache hit rates
- Verify cache memory utilization is healthy
- Review cache health status
- Identify any cache-eviction patterns

### 3. Review Connection Pool Health
- Check pool utilization across providers
- Review wait times and identify bottlenecks
- Check for stale connections
- Review failure rates by provider

### 4. Review SLA Compliance
- Generate API P95 latency <10s?
- Transform API P95 latency <2s?
- Overall throughput >10 RPS?
- Any SLA breaches in the past week?

### 5. Review Gradient Optimizer
- Average duration trending up or down?
- Any unusual spikes in optimization time?
- Worker queue backlog healthy?

### 6. Review Compression Effectiveness
- Compression ratio >50%?
- Any response size anomalies?

### 7. Review Circuit Breaker Activity
- Any unexpected circuit breaker trips?
- Are providers recovering properly?

## Performance Regression Testing

### Running Performance Tests

```bash
# Run all performance regression tests
cd backend-api
pytest tests/test_performance_regression.py -v -m performance

# Run with baseline comparison
pytest tests/test_performance_regression.py --baseline-file tests/performance_baseline.json -v

# Update baseline after improvements
pytest tests/test_performance_regression.py --update-baseline
```

### Performance Test Thresholds

| Operation | P50 | P95 | P99 | Min RPS | Max Error Rate |
|-----------|-----|-----|-----|---------|----------------|
| Health Check | 50ms | 100ms | 200ms | 100 | - |
| List Providers | 100ms | 200ms | 500ms | 50 | - |
| Transform Prompt | 500ms | 1000ms | 2000ms | - | 2% |

## Load Testing

### Running Load Tests

```bash
# Quick smoke test (10 users, 1 minute)
python tests/load/run_load_test.py smoke

# Standard load test (100 users, 5 minutes)
python tests/load/run_load_test.py load

# Stress test (1000 users, 10 minutes)
python tests/load/run_load_test.py stress

# Soak test (50 users, 1 hour - memory leak detection)
python tests/load/run_load_test.py soak

# Custom configuration
python tests/load/run_load_test.py --users 500 --spawn-rate 25 --run-time 15m --headless --html-report reports/load-test.html
```

### Load Test Configurations

| Test Type | Users | Spawn Rate | Duration | Purpose |
|-----------|-------|------------|----------|---------|
| Smoke | 10 | 2/sec | 1 min | Quick validation |
| Load | 100 | 10/sec | 5 min | Normal capacity |
| Stress | 1000 | 50/sec | 10 min | Breakpoint testing |
| Soak | 50 | 5/sec | 1 hour | Memory leak detection |

## Performance Optimization Strategies

### Cache Optimization

1. **Monitor Cache Hit Rates**: Target >70% for L1, >80% for L2
2. **Adjust Cache TTLs**: Based on data freshness requirements
3. **Optimize Cache Keys**: Use consistent, granular cache keys
4. **Review Cache Eviction**: Monitor for thrashing patterns

### Connection Pool Optimization

1. **Adjust Pool Sizes**: Based on provider capacity and latency
2. **Tune Timeouts**: Balance between responsiveness and resource usage
3. **Monitor Pool Utilization**: Scale up if consistently >70%
4. **Review Stale Connections**: Investigate root causes

### API Latency Optimization

1. **Identify Slow Endpoints**: Use tracing to find bottlenecks
2. **Optimize Database Queries**: Add indexes, rewrite queries
3. **Enable Response Compression**: Gzip/brotli for large payloads
4. **Implement Pagination**: For large data sets

### Throughput Optimization

1. **Horizontal Scaling**: Add more backend instances
2. **Async Processing**: Offload long-running tasks
3. **CDN Caching**: Cache static responses at edge
4. **Connection Reuse**: HTTP keep-alive, connection pooling

## Incident Response

### Performance Degradation Response

1. **Identify the Symptom**: High latency? Low throughput? Errors?
2. **Check Dashboards**: Grafana for visualization, Prometheus for raw metrics
3. **Review Recent Changes**: Did any deployment correlate with the issue?
4. **Check External Dependencies**: LLM provider status, Redis, database
5. **Review Alerts**: Any firing alerts related to the issue?

### Escalation Matrix

| Severity | Response Time | Escalation |
|----------|---------------|------------|
| Critical (P0) | 15 min | Engineering lead + on-call |
| High (P1) | 1 hour | Engineering lead |
| Medium (P2) | 1 business day | Team backlog |
| Low (P3) | 1 week | Team backlog |

## Performance Baseline Updates

### When to Update Baselines

1. **After Significant Optimizations**: Performance improved by >20%
2. **After Architecture Changes**: New caching layer, connection pooling
3. **After Infrastructure Changes**: Database migration, Redis upgrade
4. **Quarterly**: Regular baseline refresh

### Update Process

```bash
# 1. Run performance tests
pytest tests/test_performance_regression.py --update-baseline -v

# 2. Review results
cat tests/performance_baseline.json | jq

# 3. Commit to version control
git add tests/performance_baseline.json
git commit -m "chore: update performance baseline (PERF-008)"
```

## Continuous Improvement

### Performance SLO (Service Level Objectives)

| SLO | Target | Measurement |
|-----|--------|-------------|
| API Availability | 99.9% | Uptime monitoring |
| API Latency (P95) | <5s | Prometheus histograms |
| Error Rate | <1% | HTTP 5xx rate |
| Cache Hit Rate | >70% | Cache metrics |

### Performance Budgets

- **Bundle Size**: Main bundle <500KB gzipped
- **API Response Time**: P95 <5s for generate, <2s for transform
- **First Contentful Paint**: <2.5s
- **Time to Interactive**: <5s

## Documentation Updates

This guide should be updated when:
- New metrics are added
- Thresholds change
- New optimization strategies are implemented
- Architecture changes affect performance

## References

- Prometheus Alerts: `monitoring/prometheus/rules/chimera-alerts.yml`
- Grafana Dashboards: `monitoring/grafana/dashboards/`
- Performance Tests: `backend-api/tests/test_performance_regression.py`
- Load Tests: `backend-api/tests/load/`
