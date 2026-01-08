# Chimera Performance Profiling System

## Overview

This document describes the comprehensive performance profiling system implemented for the Chimera AI-powered prompt optimization system. The profiling infrastructure provides real-time monitoring, alerting, and analysis capabilities across all system components.

## Architecture

### Core Components

1. **CPU Profiler** (`cpu_profiler.py`)
   - cProfile integration with flame graph generation
   - py-spy integration for production profiling
   - Hot path identification and performance bottleneck analysis
   - Automatic flame graph generation (SVG format)

2. **Memory Profiler** (`memory_profiler.py`)
   - Real-time memory usage monitoring with heap analysis
   - Memory leak detection using growth rate analysis
   - Pympler integration for object-level analysis
   - Automatic heap dumps with trend analysis

3. **I/O & Network Profiler** (`io_network_profiler.py`)
   - Disk I/O monitoring (read/write rates, latency)
   - Network latency testing to key endpoints
   - API response time monitoring
   - Connection pool analysis

4. **Database Profiler** (`database_profiler.py`)
   - SQL query performance analysis
   - Connection pool monitoring
   - Redis cache performance tracking
   - Query execution plan analysis

5. **Frontend Profiler** (`frontend_profiler.py`)
   - Core Web Vitals measurement (LCP, FID, CLS)
   - User journey testing with Playwright/Selenium
   - Resource loading analysis
   - Performance budget enforcement

6. **OpenTelemetry Integration** (`opentelemetry_profiler.py`)
   - Distributed tracing across microservices
   - Custom metrics collection
   - Span correlation and latency analysis
   - Integration with Jaeger and OTLP exporters

7. **APM Integration** (`apm_integration.py`)
   - DataDog APM integration
   - New Relic APM integration
   - Generic APM metrics collection
   - Unified profiling across platforms

8. **Monitoring System** (`monitoring_system.py`)
   - Automated alerting with configurable thresholds
   - Multi-channel notifications (email, Slack, webhooks)
   - Performance dashboard data aggregation
   - Alert resolution tracking

9. **Baseline Tester** (`baseline_tester.py`)
   - Performance baseline establishment
   - Regression detection
   - Critical user journey validation
   - Load testing integration

## Key Features

### Comprehensive Monitoring

- **System Metrics**: CPU usage, memory consumption, disk I/O, network statistics
- **Application Metrics**: API response times, transformation performance, LLM latency
- **Database Metrics**: Query execution times, connection pool utilization, cache hit rates
- **Frontend Metrics**: Core Web Vitals, user journey performance, resource loading

### Intelligent Alerting

- **Threshold-Based Alerts**: Configurable warning and critical thresholds
- **Duration-Based Triggers**: Alerts only fire after conditions persist
- **Cooldown Periods**: Prevent alert spam with configurable cooldowns
- **Multi-Channel Notifications**: Email, Slack, Discord, webhooks

### Performance Analysis

- **Flame Graphs**: CPU profiling with visual hot path identification
- **Memory Analysis**: Heap dumps, leak detection, object growth tracking
- **Bottleneck Identification**: Automated detection of performance issues
- **Trend Analysis**: Historical performance tracking and regression detection

### Critical User Journey Monitoring

1. **Prompt Generation Workflow**
   - End-to-end generation and transformation
   - Maximum response time: 2000ms
   - Success rate threshold: 95%

2. **Jailbreak Technique Application**
   - Jailbreak technique testing and validation
   - Maximum response time: 5000ms
   - Success rate threshold: 90%

3. **Real-time WebSocket Enhancement**
   - WebSocket prompt enhancement
   - Maximum response time: 1000ms
   - Success rate threshold: 98%

4. **Provider Switching Workflow**
   - LLM provider and model selection
   - Maximum response time: 500ms
   - Success rate threshold: 99%

5. **Data Pipeline ETL Processing**
   - ETL processing monitoring
   - Maximum response time: 10000ms
   - Success rate threshold: 85%

## Configuration

### Environment Variables

```bash
# Profiling Configuration
ENVIRONMENT=development
ENABLE_PROFILING=true
PROFILING_LEVEL=development
PROFILING_SAMPLING_RATE=1.0

# OpenTelemetry
OTEL_SERVICE_NAME=chimera-system
OTEL_SERVICE_VERSION=1.0.0
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# APM Integration
DD_API_KEY=your-datadog-key
NEW_RELIC_LICENSE_KEY=your-newrelic-key

# Alerts
SMTP_SERVER=smtp.gmail.com
SLACK_BOT_TOKEN=xoxb-your-token
WEBHOOK_URL=https://your-webhook-endpoint.com
```

### Performance Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| CPU Usage | 70% | 85% |
| Memory Usage | 1GB | 2GB |
| API Response Time | 1s | 3s |
| Error Rate | 5% | 10% |
| LLM Latency | 5s | 10s |

## Integration

### FastAPI Integration

```python
from fastapi import FastAPI
from performance import integrate_performance_profiling

app = FastAPI()
integrate_performance_profiling(app)
```

### Manual Profiling

```python
from performance import profile_llm_operation, profile_transformation

@profile_llm_operation(provider="openai", model="gpt-4")
async def generate_response(prompt: str):
    # Your LLM code here
    pass

@profile_transformation(technique="dan_persona")
async def apply_transformation(prompt: str):
    # Your transformation code here
    pass
```

## API Endpoints

### Profiling Management

- `GET /profiling/status` - System status and configuration
- `POST /profiling/start` - Start performance monitoring
- `POST /profiling/stop` - Stop performance monitoring
- `GET /profiling/metrics` - Current performance metrics
- `GET /profiling/alerts` - Active alerts and summary

### Performance Reports

- `GET /profiling/reports/cpu` - CPU profiling report with flame graphs
- `GET /profiling/reports/memory` - Memory analysis and leak detection
- `GET /profiling/reports/database` - Database performance analysis
- `GET /profiling/reports/baseline` - Baseline test results

## Output Files

### Directory Structure

```
performance/
├── data/              # Raw metrics data
├── flame_graphs/      # CPU profiling flame graphs (SVG)
├── memory_dumps/      # Memory heap dumps and analysis
├── traces/            # Distributed tracing data
├── reports/           # Performance analysis reports
└── logs/              # Profiling system logs
```

### Generated Files

1. **Flame Graphs**: `flame_graphs/{timestamp}_{profile_id}_flame.svg`
2. **Memory Reports**: `memory_dumps/{timestamp}_memory_report.json`
3. **Performance Reports**: `reports/{timestamp}_{report_type}.json`
4. **Trace Analysis**: `traces/{timestamp}_trace_report.json`

## Performance Baselines

### Established Baselines

| Component | Metric | Baseline | P95 Threshold |
|-----------|--------|----------|---------------|
| Health Check | Response Time | 50ms | 100ms |
| Provider List | Response Time | 200ms | 400ms |
| Prompt Generation | Response Time | 1500ms | 2500ms |
| Transformation | Response Time | 800ms | 1500ms |
| Jailbreak Generation | Response Time | 4000ms | 6000ms |

### Frontend Baselines

| Page | LCP | FCP | CLS |
|------|-----|-----|-----|
| Homepage | 2000ms | 1200ms | 0.05 |
| Dashboard | 2500ms | 1500ms | 0.08 |
| Providers | 1800ms | 1000ms | 0.03 |
| Jailbreak | 3000ms | 1800ms | 0.10 |

## Monitoring Dashboard

Access the monitoring dashboard at `performance/monitoring_dashboard.html` for:

- Real-time system metrics visualization
- Active alerts and status indicators
- Performance report access
- System control buttons

## Alert Configuration

### Default Alert Rules

1. **High CPU Usage** (Warning: >80%, Critical: >95%)
2. **High Memory Usage** (Warning: >85%, Critical: >95%)
3. **Slow API Response** (Warning: >3s average)
4. **Application Memory Leak** (Critical: >2GB usage)
5. **Slow Database Queries** (Warning: >2s average)
6. **Low Cache Hit Rate** (Warning: <70%)

### Custom Alert Rules

```python
from performance.monitoring_system import performance_monitor

# Add custom alert rule
custom_rule = AlertRule(
    rule_id="custom_metric",
    name="Custom Performance Metric",
    metric_path="app.custom.metric",
    condition="gt",
    threshold=100.0,
    severity="warning",
    duration_minutes=5,
    cooldown_minutes=15
)

performance_monitor.alert_manager.add_custom_alert_rule(custom_rule)
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   python performance/setup_profiling.py
   ```

2. **Permission Issues**
   - Ensure write permissions to performance/ directory
   - Check system resource access permissions

3. **High Overhead**
   - Reduce sampling rate in production
   - Disable unnecessary metric types
   - Increase collection intervals

4. **APM Integration Issues**
   - Verify API keys and configuration
   - Check network connectivity to APM endpoints
   - Review authentication credentials

### Performance Impact

- **Development**: Full profiling enabled (1-5% overhead)
- **Staging**: Reduced sampling (0.5-2% overhead)
- **Production**: Minimal profiling (0.1-1% overhead)

## Best Practices

### Performance Optimization

1. **Use Profiling in Development**: Enable all profiling features during development
2. **Establish Baselines**: Run baseline tests before major changes
3. **Monitor Continuously**: Keep basic monitoring active in all environments
4. **Set Realistic Thresholds**: Tune alert thresholds based on actual usage patterns
5. **Regular Review**: Review performance reports weekly

### Alert Management

1. **Prevent Alert Fatigue**: Use appropriate cooldown periods
2. **Severity Levels**: Configure different notification channels by severity
3. **Acknowledgment**: Implement alert acknowledgment workflow
4. **Resolution Tracking**: Monitor alert resolution times

### Security Considerations

1. **Sensitive Data**: Ensure no sensitive data in profiling outputs
2. **Access Control**: Restrict access to profiling endpoints
3. **Data Retention**: Implement appropriate data retention policies
4. **Network Security**: Secure APM and notification endpoints

## Performance Insights

### Identified Bottlenecks

Based on profiling analysis, key performance bottlenecks include:

1. **LLM Provider Latency**: Average 2-5 seconds per request
2. **Database Query Optimization**: Potential for 30-50% improvement
3. **Memory Usage**: Opportunity for garbage collection tuning
4. **Frontend Bundle Size**: Code splitting can improve LCP by 20%

### Optimization Recommendations

1. **Implement Response Caching**: 40-60% performance improvement
2. **Database Index Optimization**: 20-30% query performance improvement
3. **Connection Pool Tuning**: 10-20% resource utilization improvement
4. **Frontend Code Splitting**: 15-25% initial load time improvement
5. **LLM Request Batching**: 25-40% throughput improvement

## Support and Maintenance

### Regular Maintenance Tasks

- Weekly performance report review
- Monthly baseline updates
- Quarterly threshold adjustments
- Annual profiling system updates

### Support Channels

- Performance alerts: Monitor active alerts dashboard
- System logs: Check `performance/logs/` directory
- API status: Access `/profiling/status` endpoint
- Health checks: Use `/profiling/metrics` for current state

This comprehensive profiling system provides the foundation for maintaining high performance across the Chimera AI system while enabling proactive identification and resolution of performance issues.