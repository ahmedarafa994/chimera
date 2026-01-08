# Performance Testing and Benchmarking Suite

This directory contains the comprehensive performance testing and benchmarking suite for the Chimera backend optimization project.

## Overview

The performance testing suite validates that all backend optimizations meet the specified performance targets:

- **API Response Time P95**: < 2s
- **LLM Provider P95**: < 10s
- **Transformation Time P95**: < 1s
- **Concurrent Request Handling**: 150/s
- **Memory Usage Optimization**: 30% reduction
- **Cache Hit Ratio**: > 85%

## Components

### Core Testing Framework

- **`performance_benchmarks.py`** - Main benchmark suite with comprehensive testing
- **`performance_utils.py`** - Utilities, data generators, and analysis tools
- **`performance_runner.py`** - Configuration management and CLI runner

### Test Integration

- **`../tests/test_performance_integration.py`** - pytest integration tests
- **Performance markers**: `@pytest.mark.performance`, `@pytest.mark.slow`, `@pytest.mark.regression`

## Quick Start

### Run Full Performance Suite

```bash
# From backend-api directory
python -m app.testing.performance_runner full

# With custom configuration
python -m app.testing.performance_runner full --config performance.json
```

### Run Quick Performance Check

```bash
python -m app.testing.performance_runner quick
```

### Run Load Testing

```bash
# Test API endpoints
python -m app.testing.performance_runner load --scenario api

# Test LLM service
python -m app.testing.performance_runner load --scenario llm
```

### Run with pytest

```bash
# Run all performance tests
pytest tests/test_performance_integration.py -m performance -v

# Run only fast performance tests (exclude slow tests)
pytest tests/test_performance_integration.py -m "performance and not slow" -v

# Run regression tests
pytest tests/test_performance_integration.py -m regression -v
```

## Configuration

### Environment Variables

```bash
# Performance targets
export PERF_API_RESPONSE_TARGET=2000      # API P95 response time (ms)
export PERF_LLM_RESPONSE_TARGET=10000     # LLM P95 response time (ms)
export PERF_TRANSFORMATION_TARGET=1000    # Transformation P95 time (ms)
export PERF_THROUGHPUT_TARGET=150         # Requests per second
export PERF_MEMORY_TARGET=30              # Memory optimization percent
export PERF_CACHE_TARGET=85               # Cache hit ratio percent

# Test configuration
export PERF_TEST_DURATION=60              # Load test duration (seconds)
export PERF_CONCURRENT_USERS=50           # Concurrent test users
export PERF_REQUEST_RATE=10               # Requests per second per user

# CI/CD configuration
export PERF_FAIL_ON_REGRESSION=true       # Fail on performance regression
export PERF_SKIP_SLOW_TESTS=false         # Skip slow tests in CI
```

### Configuration File

```json
{
  "performance_targets": {
    "api_response_p95_ms": 2000.0,
    "llm_provider_p95_ms": 10000.0,
    "transformation_p95_ms": 1000.0,
    "concurrent_requests_per_second": 150.0,
    "memory_optimization_percent": 30.0,
    "cache_hit_ratio_percent": 85.0
  },
  "performance_gates": {
    "max_response_time_ms": 2000,
    "min_throughput_rps": 150,
    "max_memory_mb": 1024,
    "min_cache_hit_ratio": 0.85,
    "max_error_rate": 0.01,
    "max_regression_percent": 10.0
  }
}
```

## Testing Scenarios

### 1. LLM Service Performance

- **Response Time Testing**: Validates LLM generation times meet P95 targets
- **Batch Performance**: Tests batch processing throughput and efficiency
- **Cache Effectiveness**: Validates cache hit ratios and speedup

### 2. Transformation Service Performance

- **Individual Transformations**: Tests single transformation performance
- **Parallel Processing**: Validates parallel transformation speedup
- **Memory Efficiency**: Monitors memory usage during transformations

### 3. Memory Optimization

- **Memory Usage Tracking**: Monitors memory consumption patterns
- **Leak Detection**: Tests for memory leaks across service operations
- **Optimization Validation**: Confirms 30% memory usage reduction

### 4. Cache Performance

- **Hit Ratio Testing**: Validates cache effectiveness across services
- **Latency Testing**: Measures cache access performance
- **Eviction Testing**: Tests cache eviction policies

### 5. Concurrent Performance

- **Load Testing**: Simulates concurrent users and requests
- **Throughput Testing**: Validates 150 RPS target under load
- **Stability Testing**: Tests system stability under sustained load

### 6. End-to-End Performance

- **Full Pipeline Testing**: Tests complete request flow
- **Integration Performance**: Validates service interaction performance
- **Real-world Scenarios**: Tests with realistic data and usage patterns

## Performance Gates

The performance gate system provides automated pass/fail criteria for CI/CD:

```bash
# Check performance gates
python -m app.testing.performance_runner quick

# Expected output:
# Performance Gate: PASS
# Gates Passed: 6/6
#   ✅ response_time: P95 response time: 1800.00ms (limit: 2000ms)
#   ✅ throughput: Throughput: 160.0 rps (minimum: 150 rps)
#   ✅ memory: Peak memory: 800.0MB (limit: 1024MB)
#   ✅ cache: Cache hit ratio: 87% (minimum: 85%)
#   ✅ error_rate: Error rate: 0.50% (maximum: 1.00%)
#   ✅ regression_check: No significant regressions detected
```

## Baseline Management

The system automatically manages performance baselines for regression detection:

```bash
# Results are saved to performance_baselines/
# Format: performance_baseline_<timestamp>.json

# Manual baseline management
python -m app.testing.performance_runner config --output performance_baseline.json
```

## CI/CD Integration

### GitHub Actions Integration

```yaml
name: Performance Tests
on: [push, pull_request]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11

      - name: Install dependencies
        run: |
          cd backend-api
          pip install -r requirements.txt
          pip install pytest pytest-asyncio

      - name: Run quick performance check
        run: |
          cd backend-api
          python -m app.testing.performance_runner quick
        env:
          PERF_SKIP_SLOW_TESTS: true
          PERF_TEST_DURATION: 30
          PERF_CONCURRENT_USERS: 10

      - name: Run performance tests
        run: |
          cd backend-api
          pytest tests/test_performance_integration.py -m "performance and not slow" -v
```

### Docker Integration

```dockerfile
# Add to Dockerfile for performance testing
COPY backend-api/app/testing/ /app/testing/
RUN pip install pytest pytest-asyncio

# Performance test entrypoint
ENTRYPOINT ["python", "-m", "app.testing.performance_runner"]
```

## Monitoring and Alerting

### Performance Metrics

The suite collects comprehensive metrics:

- Response time percentiles (P50, P90, P95, P99)
- Throughput (requests/second)
- Error rates and success rates
- Memory usage and optimization
- Cache hit ratios and latency
- Concurrent request handling
- Resource utilization (CPU, memory, network)

### Regression Detection

Automatic regression detection compares against baseline:

- **Minor Regression**: 10-25% performance decrease
- **Major Regression**: 25-50% performance decrease
- **Critical Regression**: >50% performance decrease

### Performance Reports

Results are saved in multiple formats:

- **JSON**: `performance_report_<timestamp>.json` - Detailed results
- **CSV**: `performance_summary_<timestamp>.csv` - Tabular summary
- **HTML**: `performance_dashboard.html` - Visual dashboard (future)

## Troubleshooting

### Common Issues

1. **Service Unavailable**: Ensure backend services are running on port 8001
2. **Mock Service Failures**: Performance tests use mock providers for consistency
3. **Memory Limits**: Adjust test parameters if running on resource-constrained systems
4. **Timeout Errors**: Increase timeout values in CI environments

### Debug Mode

```bash
# Run with verbose logging
python -m app.testing.performance_runner full -v

# Run individual test components
pytest tests/test_performance_integration.py::test_llm_service_response_time -v -s
```

### Performance Tuning

If targets are not met, check:

1. **Service Configurations**: Review optimized service implementations
2. **Resource Limits**: Ensure adequate CPU/memory for testing
3. **Network Latency**: Consider network conditions in distributed setups
4. **Test Parameters**: Adjust test intensity and duration

## Results Interpretation

### Success Criteria

- All performance gates pass (PASS status)
- No critical performance regressions detected
- Test success rate > 95%
- Resource usage within configured limits

### Performance Score

Overall performance score calculation:

```
Performance Score = (Gates Passed / Total Gates) × 100%

Target: > 90% for production readiness
```

## Extensions

The framework is designed for extensibility:

- **Custom Test Scenarios**: Add new test scenarios in `performance_benchmarks.py`
- **Additional Metrics**: Extend metric collection in `performance_utils.py`
- **Custom Gates**: Add new performance gates in `CICDPerformanceGate`
- **Report Formats**: Add new output formats in the runner

This comprehensive performance testing suite ensures that all backend optimizations deliver the promised performance improvements and provides ongoing monitoring for performance regressions.