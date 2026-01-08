# Weekly Performance Review Checklist

## Quick Health Check (5 minutes)

- [ ] **Grafana Dashboard**: No red metrics
- [ ] **Prometheus Alerts**: No firing alerts
- [ ] **API Latency**: P95 within SLA
- [ ] **Error Rate**: <1%

---

## Detailed Review (15 minutes)

### Cache Performance
- [ ] L1 Hit Rate >70%
- [ ] L2 Hit Rate >80%
- [ ] Combined Hit Rate >85%
- [ ] L1 Memory <90%
- [ ] L2 Redis healthy

**Action if failing**: Review cache key patterns, adjust TTLs

### Connection Pools
- [ ] Pool Utilization <90%
- [ ] Avg Wait Time <100ms
- [ ] Stale Connections <10
- [ ] Failure Rate <1%

**Action if failing**: Adjust pool size, investigate provider latency

### API Performance
- [ ] Generate P95 <10s
- [ ] Transform P95 <2s
- [ ] Throughput >10 RPS
- [ ] Timeout rate acceptable

**Action if failing**: Check LLM provider status, review slow traces

### Circuit Breakers
- [ ] No unexpected OPEN states
- [ ] Trip rate <0.1/sec
- [ ] Providers recovering properly

**Action if failing**: Investigate root cause, check provider health

### Gradient Optimizer
- [ ] Avg Duration <500ms
- [ ] Worker Queue <100

**Action if failing**: Check batch size, optimize prompt complexity

### Compression
- [ ] Compression Ratio >50%

**Action if failing**: Review compression configuration

---

## Monthly Deep Dive (30 minutes)

- [ ] **Performance Baseline**: Run regression tests
- [ ] **Load Testing**: Run full load test suite
- [ ] **Cost Analysis**: Review LLM provider costs
- [ ] **Capacity Planning**: Forecast growth needs
- [ ] **Optimization Backlog**: Prioritize improvements

---

## Quarterly Review (1 hour)

- [ ] **SLO Review**: Assess SLA compliance
- [ ] **Architecture Review**: Performance impact of changes
- [ ] **Baseline Update**: Refresh performance baselines
- [ ] **Tool Assessment**: Evaluate monitoring tools
- [ ] **Process Review**: Improve performance workflow

---

## Quick Commands

```bash
# Check Prometheus alerts
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing")'

# Run performance regression tests
pytest tests/test_performance_regression.py -v -m performance

# Quick smoke test
python tests/load/run_load_test.py smoke --headless

# View dashboard
# http://localhost:3001/d/chimera-performance-optimization
```

---

## Escalation

| Issue Type | Severity | Response Time | Contact |
|------------|----------|---------------|---------|
| API Down | P0 | 15 min | On-call |
| SLA Breach | P1 | 1 hour | Team Lead |
| Cache Degradation | P2 | 1 day | Engineering |
| Performance Trend | P3 | 1 week | Team Backlog |

---

## Notes

**Date**: _______________
**Reviewer**: _______________
**Action Items**:
-
-
-
