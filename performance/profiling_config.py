"""
Chimera Performance Profiling Configuration
Comprehensive profiling setup for FastAPI backend and Next.js frontend
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProfilingLevel(Enum):
    """Profiling levels for different use cases"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    RESEARCH = "research"

class MetricType(Enum):
    """Types of performance metrics to collect"""
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    LLM_LATENCY = "llm_latency"
    TRANSFORMATION = "transformation"
    WEBSOCKET = "websocket"
    FRONTEND = "frontend"

@dataclass
class ProfilingConfig:
    """Configuration for performance profiling"""

    # Environment settings
    environment: ProfilingLevel = ProfilingLevel.DEVELOPMENT
    profiling_enabled: bool = True
    sampling_rate: float = 0.1  # 10% sampling in production

    # Output directories
    base_output_dir: str = "D:/MUZIK/chimera/performance/data"
    flame_graphs_dir: str = "D:/MUZIK/chimera/performance/flame_graphs"
    memory_dumps_dir: str = "D:/MUZIK/chimera/performance/memory_dumps"
    traces_dir: str = "D:/MUZIK/chimera/performance/traces"
    reports_dir: str = "D:/MUZIK/chimera/performance/reports"

    # Profiling intervals (seconds)
    cpu_profile_duration: int = 60
    memory_snapshot_interval: int = 300  # 5 minutes
    io_monitoring_interval: int = 30
    trace_collection_interval: int = 120  # 2 minutes

    # Enabled metric types
    enabled_metrics: list[MetricType] = field(default_factory=lambda: [
        MetricType.CPU,
        MetricType.MEMORY,
        MetricType.IO,
        MetricType.NETWORK,
        MetricType.DATABASE,
        MetricType.CACHE,
        MetricType.LLM_LATENCY,
        MetricType.TRANSFORMATION,
        MetricType.WEBSOCKET,
        MetricType.FRONTEND
    ])

    # Service endpoints for monitoring
    backend_url: str = "http://localhost:8001"
    frontend_url: str = "http://localhost:3000"
    redis_url: str = "redis://localhost:6379/0"

    # Critical user journeys to monitor
    critical_journeys: list[dict[str, Any]] = field(default_factory=lambda: [
        {
            "name": "prompt_generation_workflow",
            "description": "End-to-end prompt generation and transformation",
            "endpoints": ["/api/v1/generate", "/api/v1/transform", "/api/v1/execute"],
            "max_response_time_ms": 2000,
            "success_rate_threshold": 0.95
        },
        {
            "name": "jailbreak_technique_application",
            "description": "Jailbreak technique application and testing",
            "endpoints": ["/api/v1/generation/jailbreak/generate", "/api/v1/autodan/optimize"],
            "max_response_time_ms": 5000,
            "success_rate_threshold": 0.90
        },
        {
            "name": "realtime_websocket_enhancement",
            "description": "Real-time WebSocket prompt enhancement",
            "endpoints": ["/ws/enhance"],
            "max_response_time_ms": 1000,
            "success_rate_threshold": 0.98
        },
        {
            "name": "provider_switching_workflow",
            "description": "LLM provider switching and model selection",
            "endpoints": ["/api/v1/providers", "/api/v1/session/models"],
            "max_response_time_ms": 500,
            "success_rate_threshold": 0.99
        },
        {
            "name": "data_pipeline_etl",
            "description": "Data pipeline ETL processing",
            "endpoints": ["/api/v1/pipeline/ingest", "/api/v1/pipeline/status"],
            "max_response_time_ms": 10000,
            "success_rate_threshold": 0.85
        }
    ])

    # Performance thresholds
    performance_thresholds: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "cpu": {
            "warning": 70.0,  # 70% CPU usage
            "critical": 85.0   # 85% CPU usage
        },
        "memory": {
            "warning": 1024.0,  # 1GB memory usage
            "critical": 2048.0  # 2GB memory usage
        },
        "response_time": {
            "warning": 1000.0,  # 1 second
            "critical": 3000.0  # 3 seconds
        },
        "error_rate": {
            "warning": 0.05,   # 5% error rate
            "critical": 0.10   # 10% error rate
        },
        "llm_latency": {
            "warning": 5000.0,  # 5 seconds
            "critical": 10000.0 # 10 seconds
        }
    })

    # APM Configuration
    apm_config: dict[str, Any] = field(default_factory=lambda: {
        "datadog": {
            "enabled": False,
            "service_name": "chimera-backend",
            "env": "development",
            "version": "1.0.0",
            "trace_sample_rate": 0.1
        },
        "new_relic": {
            "enabled": False,
            "license_key": None,
            "app_name": "Chimera AI System"
        },
        "opentelemetry": {
            "enabled": True,
            "service_name": "chimera-system",
            "service_version": "1.0.0",
            "endpoint": "http://localhost:4317",
            "headers": {}
        }
    })

    # Load testing configuration
    load_test_config: dict[str, Any] = field(default_factory=lambda: {
        "scenarios": [
            {
                "name": "baseline_load",
                "virtual_users": 10,
                "duration": "5m",
                "ramp_up": "30s"
            },
            {
                "name": "peak_load",
                "virtual_users": 50,
                "duration": "10m",
                "ramp_up": "2m"
            },
            {
                "name": "stress_test",
                "virtual_users": 100,
                "duration": "15m",
                "ramp_up": "5m"
            }
        ]
    })

    def __post_init__(self):
        """Create necessary directories"""
        directories = [
            self.base_output_dir,
            self.flame_graphs_dir,
            self.memory_dumps_dir,
            self.traces_dir,
            self.reports_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def get_output_path(self, metric_type: MetricType, filename: str) -> str:
        """Get output path for a specific metric type"""
        timestamp = int(time.time())

        if metric_type == MetricType.CPU:
            return os.path.join(self.flame_graphs_dir, f"{timestamp}_{filename}")
        elif metric_type == MetricType.MEMORY:
            return os.path.join(self.memory_dumps_dir, f"{timestamp}_{filename}")
        else:
            return os.path.join(self.base_output_dir, f"{timestamp}_{filename}")

    def is_metric_enabled(self, metric_type: MetricType) -> bool:
        """Check if a metric type is enabled"""
        return metric_type in self.enabled_metrics

    def get_sampling_rate(self) -> float:
        """Get appropriate sampling rate based on environment"""
        if self.environment == ProfilingLevel.PRODUCTION:
            return 0.01  # 1% in production
        elif self.environment == ProfilingLevel.STAGING:
            return 0.1   # 10% in staging
        else:
            return 1.0   # 100% in development/research

# Global configuration instance
config = ProfilingConfig()

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    config.environment = ProfilingLevel.PRODUCTION
    config.profiling_enabled = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
elif os.getenv("ENVIRONMENT") == "staging":
    config.environment = ProfilingLevel.STAGING

# APM configuration from environment
if os.getenv("DATADOG_API_KEY"):
    config.apm_config["datadog"]["enabled"] = True

if os.getenv("NEW_RELIC_LICENSE_KEY"):
    config.apm_config["new_relic"]["enabled"] = True
    config.apm_config["new_relic"]["license_key"] = os.getenv("NEW_RELIC_LICENSE_KEY")

# OpenTelemetry configuration
if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    config.apm_config["opentelemetry"]["endpoint"] = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
