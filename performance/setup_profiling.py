"""
Performance Profiling Installation and Setup Script
Installs required dependencies and initializes the profiling system
"""

import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required performance profiling dependencies"""

    # Core profiling dependencies
    core_deps = [
        "psutil>=5.9.0",
        "pympler>=0.9",
        "memory-profiler>=0.60.0",
        "py-spy>=0.3.12",
        "flameprof>=0.4",
    ]

    # HTTP and async dependencies
    http_deps = [
        "aiohttp>=3.8.0",
        "requests>=2.28.0",
    ]

    # Browser automation (optional)
    browser_deps = [
        "selenium>=4.0.0",
        "playwright>=1.20.0",
    ]

    # Load testing (optional)
    load_test_deps = [
        "locust>=2.0.0",
    ]

    # OpenTelemetry dependencies
    otel_deps = [
        "opentelemetry-api>=1.15.0",
        "opentelemetry-sdk>=1.15.0",
        "opentelemetry-exporter-otlp>=1.15.0",
        "opentelemetry-exporter-jaeger>=1.15.0",
        "opentelemetry-instrumentation-fastapi>=0.36b0",
        "opentelemetry-instrumentation-requests>=0.36b0",
        "opentelemetry-instrumentation-aiohttp-client>=0.36b0",
        "opentelemetry-instrumentation-redis>=0.36b0",
        "opentelemetry-instrumentation-sqlite3>=0.36b0",
    ]

    # APM integrations (optional)
    apm_deps = [
        "ddtrace>=1.10.0",  # DataDog
        "newrelic>=8.0.0",  # New Relic
    ]

    # Notification dependencies (optional)
    notification_deps = [
        "slack-sdk>=3.19.0",
        "discord.py>=2.0.0",
    ]

    all_deps = core_deps + http_deps + otel_deps

    print("Installing core performance profiling dependencies...")
    for dep in all_deps:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", dep]
            )
            print(f"[OK] Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"[FAIL] Failed to install {dep}")

    # Optional dependencies
    optional_groups = [
        ("Browser automation", browser_deps),
        ("Load testing", load_test_deps),
        ("APM integrations", apm_deps),
        ("Notifications", notification_deps),
    ]

    for group_name, deps in optional_groups:
        print(f"\nOptional: {group_name} dependencies")
        for dep in deps:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", dep]
                )
                print(f"[OK] Installed {dep}")
            except subprocess.CalledProcessError:
                print(f"[FAIL] Failed to install {dep} (optional)")


def setup_directories():
    """Create necessary directories for profiling data"""

    base_path = Path("D:/MUZIK/chimera/performance")

    directories = [
        "data",
        "flame_graphs",
        "memory_dumps",
        "traces",
        "reports",
        "logs"
    ]

    print("\nCreating profiling directories...")
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created {dir_path}")


def create_env_template():
    """Create environment template for profiling configuration"""

    env_template = """
# Chimera Performance Profiling Configuration

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Profiling Settings
ENABLE_PROFILING=true
PROFILING_LEVEL=development
PROFILING_SAMPLING_RATE=1.0

# OpenTelemetry Configuration
OTEL_SERVICE_NAME=chimera-system
OTEL_SERVICE_VERSION=1.0.0
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_HEADERS=""

# DataDog APM (optional)
# DD_API_KEY=your-datadog-api-key
# DD_SERVICE=chimera-backend
# DD_ENV=development
# DD_VERSION=1.0.0

# New Relic APM (optional)
# NEW_RELIC_LICENSE_KEY=your-newrelic-license-key
# NEW_RELIC_APP_NAME=Chimera AI System

# Notification Configuration (optional)
# SMTP_SERVER=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USERNAME=your-email@example.com
# SMTP_PASSWORD=your-password
# FROM_EMAIL=alerts@chimera.ai
# ALERT_EMAILS=admin@chimera.ai,ops@chimera.ai

# Slack Integration (optional)
# SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
# SLACK_CHANNEL=#alerts

# Webhook Integration (optional)
# WEBHOOK_URL=https://your-webhook-endpoint.com/alerts
# WEBHOOK_HEADERS={"Content-Type": "application/json"}

# Performance Thresholds
CPU_WARNING_THRESHOLD=70.0
CPU_CRITICAL_THRESHOLD=85.0
MEMORY_WARNING_THRESHOLD=1024.0
MEMORY_CRITICAL_THRESHOLD=2048.0
RESPONSE_TIME_WARNING_THRESHOLD=1000.0
RESPONSE_TIME_CRITICAL_THRESHOLD=3000.0
"""

    env_path = Path("D:/MUZIK/chimera/.env.profiling")

    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_template.strip())
        print(f"[OK] Created profiling environment template: {env_path}")
        print("  Please review and customize the configuration")
    else:
        print(
            f"[OK] Profiling environment template already exists: {env_path}"
        )


def create_example_integration():
    """Create example integration code"""

    example_code = '''"""
Example: Integrating Performance Profiling with Chimera Backend
"""

from fastapi import FastAPI
from performance import (
    integrate_performance_profiling,
    profile_llm_operation,
    profile_transformation,
)

# Create FastAPI app
app = FastAPI(title="Chimera AI System")

# Integrate performance profiling
integrate_performance_profiling(app)

# Example: Profile LLM operations
@profile_llm_operation(provider="openai", model="gpt-4")
async def generate_with_openai(prompt: str):
    # Your LLM generation code here
    pass

# Example: Profile transformations
@profile_transformation(technique="dan_persona")
async def apply_dan_transformation(prompt: str):
    # Your transformation code here
    pass

# The profiling system will automatically:
# 1. Profile all HTTP requests via middleware
# 2. Collect system metrics (CPU, memory, I/O)
# 3. Monitor for performance issues
# 4. Send alerts when thresholds are exceeded
# 5. Generate flame graphs and performance reports
# 6. Integrate with APM platforms (DataDog, New Relic)
# 7. Provide OpenTelemetry distributed tracing

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
'''

    example_path = Path("D:/MUZIK/chimera/performance_integration_example.py")

    with open(example_path, "w") as f:
        f.write(example_code.strip())

    print(f"[OK] Created integration example: {example_path}")


def create_monitoring_dashboard():
    """Create basic monitoring dashboard HTML"""

    dashboard_html = '''<!DOCTYPE html>
<html>
<head>
    <title>Chimera Performance Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .metric-title { font-weight: bold; color: #333; }
        .metric-value { font-size: 24px; margin: 10px 0; }
        .status-good { color: #4CAF50; }
        .status-warning { color: #FF9800; }
        .status-critical { color: #F44336; }
        .alerts-section {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
        }
    </style>
    <script>
        async function loadMetrics() {
            try {
                const response = await fetch('/profiling/metrics');
                const data = await response.json();

                const sys = data.system_metrics;
                document.getElementById('cpu-usage').textContent =
                    sys.cpu_usage ? sys.cpu_usage.toFixed(1) + '%' : 'N/A';

                document.getElementById('memory-usage').textContent =
                    sys.memory_usage ? sys.memory_usage.toFixed(1) + '%' : 'N/A';

                document.getElementById('api-response').textContent =
                    sys.api_response_time ?
                    sys.api_response_time.toFixed(0) + 'ms' : 'N/A';

                document.getElementById('active-alerts').textContent =
                    data.alerts.total_active_alerts || 0;

                // Update status indicators
                updateStatus('cpu-status', sys.cpu_usage, 70, 85);
                updateStatus('memory-status', sys.memory_usage, 80, 90);
                updateStatus('api-status', sys.api_response_time, 1000, 3000);

            } catch (error) {
                console.error('Failed to load metrics:', error);
            }
        }

        function updateStatus(elementId, value, warning, critical) {
            const element = document.getElementById(elementId);
            if (!value) {
                element.className = '';
                return;
            }

            if (value >= critical) {
                element.className = 'status-critical';
            } else if (value >= warning) {
                element.className = 'status-warning';
            } else {
                element.className = 'status-good';
            }
        }

        // Load metrics every 30 seconds
        setInterval(loadMetrics, 30000);

        // Load on page load
        window.onload = loadMetrics;
    </script>
</head>
<body>
    <h1>Chimera Performance Monitor</h1>

    <div class="metric-card">
        <div class="metric-title">CPU Usage</div>
        <div class="metric-value" id="cpu-usage">Loading...</div>
        <div id="cpu-status"></div>
    </div>

    <div class="metric-card">
        <div class="metric-title">Memory Usage</div>
        <div class="metric-value" id="memory-usage">Loading...</div>
        <div id="memory-status"></div>
    </div>

    <div class="metric-card">
        <div class="metric-title">API Response Time</div>
        <div class="metric-value" id="api-response">Loading...</div>
        <div id="api-status"></div>
    </div>

    <div class="alerts-section">
        <h2>Active Alerts</h2>
        <div class="metric-value" id="active-alerts">Loading...</div>
        <p><a href="/profiling/alerts">View All Alerts</a></p>
    </div>

    <div style="margin-top: 30px;">
        <h2>Performance Reports</h2>
        <ul>
            <li><a href="/profiling/reports/cpu">CPU Report</a></li>
            <li><a href="/profiling/reports/memory">Memory Report</a></li>
            <li><a href="/profiling/reports/database">DB Report</a></li>
            <li><a href="/profiling/reports/baseline">Baseline Results</a></li>
        </ul>
    </div>

    <div style="margin-top: 30px;">
        <h2>System Controls</h2>
        <button onclick="fetch('/profiling/start', {method: 'POST'})
            .then(() => alert('Monitoring started'))">
            Start Monitoring
        </button>
        <button onclick="fetch('/profiling/stop', {method: 'POST'})
            .then(() => alert('Monitoring stopped'))">
            Stop Monitoring
        </button>
    </div>

    <div style="margin-top: 30px; font-size: 12px; color: #666;">
        <p>Last updated: <span id="last-update"></span></p>
        <script>
            document.getElementById('last-update').textContent =
                new Date().toLocaleString();
        </script>
    </div>
</body>
</html>'''

    dashboard_path = Path(
        "D:/MUZIK/chimera/performance/monitoring_dashboard.html"
    )

    with open(dashboard_path, "w") as f:
        f.write(dashboard_html)

    print(f"[OK] Created monitoring dashboard: {dashboard_path}")


def main():
    """Main setup function"""
    print("Chimera Performance Profiling Setup")
    print("=" * 40)

    # Install dependencies
    install_dependencies()

    # Create directories
    setup_directories()

    # Create configuration template
    create_env_template()

    # Create example integration
    create_example_integration()

    # Create monitoring dashboard
    create_monitoring_dashboard()

    print("\n" + "=" * 40)
    print("Performance Profiling Setup Complete!")
    print("\nNext steps:")
    print("1. Review and customize the .env.profiling configuration")
    print("2. Integrate profiling (see performance_integration_example.py)")
    print("3. Navigate to /profiling/status to verify setup")
    print("4. View dashboard at performance/monitoring_dashboard.html")
    print("5. Configure APM integrations (DataDog, New Relic) if needed")
    print("6. Set up notification channels for alerts")
    print("\nDocumentation:")
    print("- API endpoints: /profiling/* for status, metrics, and reports")
    print("- Flame graphs: performance/flame_graphs/")
    print("- Memory dumps: performance/memory_dumps/")
    print("- Performance reports: performance/reports/")


if __name__ == "__main__":
    main()
