"""
Performance Profiling Integration for Chimera Backend
Integration with FastAPI application for comprehensive performance profiling
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware

from performance.apm_integration import apm_profiler, apm_trace_async
from performance.cpu_profiler import cpu_profiler, profile_cpu_async
from performance.database_profiler import (db_profiler, profile_cache,
                                           profile_query)
from performance.io_network_profiler import io_profiler, monitor_io
from performance.memory_profiler import memory_profiler, monitor_memory
from performance.monitoring_system import performance_monitor
from performance.opentelemetry_profiler import (otel_profiler,
                                                trace_async_operation)
# Import all profiling modules
from performance.profiling_config import config

logger = logging.getLogger(__name__)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic performance profiling"""

    async def dispatch(self, request: Request, call_next):
        """Profile each request automatically"""
        start_time = time.time()

        # Extract route information
        route = request.url.path
        method = request.method

        # Start profiling context
        with (
            cpu_profiler.profile_context(f"{method}_{route.replace('/', '_')}"),
            otel_profiler.trace_operation(
                f"{method} {route}",
                http_method=method,
                http_route=route,
                http_url=str(request.url),
            ),
            apm_profiler.trace_operation(f"{method}_{route}", http_method=method, http_route=route),
        ):
            # Execute request
            response = await call_next(request)

            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000

            # Record metrics
            apm_profiler.custom_metric(
                "http.request.duration",
                duration_ms,
                {
                    "method": method,
                    "route": route,
                    "status_code": str(response.status_code),
                },
            )

            # Log slow requests
            if duration_ms > 1000:  # >1 second
                logger.warning(f"Slow request: {method} {route} took {duration_ms:.0f}ms")

            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            response.headers["X-Profiling-Enabled"] = "true"

            return response


def setup_profiling_integration(app: FastAPI) -> None:
    """Set up comprehensive profiling integration with FastAPI"""

    # Add performance middleware
    app.add_middleware(PerformanceMiddleware)

    # Instrument with OpenTelemetry
    otel_profiler.instrument_fastapi(app)
    otel_profiler.instrument_libraries()

    # Create custom metrics
    otel_profiler.create_custom_metrics()

    # Add profiling endpoints
    @app.get("/profiling/status")
    async def profiling_status():
        """Get profiling system status"""
        return {
            "profiling_enabled": config.profiling_enabled,
            "environment": config.environment.value,
            "enabled_metrics": [metric.value for metric in config.enabled_metrics],
            "monitoring_active": performance_monitor.monitoring_active,
            "apm_platforms": {
                "datadog": apm_profiler.datadog.enabled,
                "newrelic": apm_profiler.newrelic.enabled,
                "opentelemetry": True,
            },
        }

    @app.get("/profiling/metrics")
    async def profiling_metrics():
        """Get current performance metrics"""
        return performance_monitor.get_dashboard_data()

    @app.post("/profiling/start")
    async def start_profiling():
        """Start performance monitoring"""
        await performance_monitor.start_monitoring()
        return {"status": "started", "message": "Performance monitoring started"}

    @app.post("/profiling/stop")
    async def stop_profiling():
        """Stop performance monitoring"""
        await performance_monitor.stop_monitoring()
        return {"status": "stopped", "message": "Performance monitoring stopped"}

    @app.get("/profiling/alerts")
    async def get_alerts():
        """Get active performance alerts"""
        return {
            "active_alerts": [
                alert.__dict__ for alert in performance_monitor.alert_manager.get_active_alerts()
            ],
            "alert_summary": performance_monitor.alert_manager.get_alert_summary(),
        }

    @app.get("/profiling/reports/cpu")
    async def get_cpu_report():
        """Generate CPU profiling report"""
        # This would return recent CPU profiling data
        return {"message": "CPU report endpoint - implement based on requirements"}

    @app.get("/profiling/reports/memory")
    async def get_memory_report():
        """Generate memory profiling report"""
        report_path = memory_profiler.save_memory_report()
        return {"report_path": report_path, "message": "Memory report generated"}

    @app.get("/profiling/reports/database")
    async def get_database_report():
        """Generate database performance report"""
        report = db_profiler.generate_database_report()
        return {
            "report_id": report.report_id,
            "performance_summary": report.performance_summary,
            "optimization_recommendations": report.optimization_recommendations,
        }

    @app.get("/profiling/reports/baseline")
    async def run_baseline_test():
        """Run performance baseline tests"""
        from performance.baseline_tester import baseline_tester

        report = await baseline_tester.run_comprehensive_baseline()

        return {
            "report_id": report.report_id,
            "baseline_status": report.baseline_status,
            "test_summary": {
                "total_tests": len(report.test_results),
                "successful_tests": len([r for r in report.test_results if r.success]),
                "failed_tests": len([r for r in report.test_results if not r.success]),
            },
            "recommendations": report.recommendations,
        }


# Decorators for manual profiling
def profile_llm_operation(provider: str, model: str):
    """Decorator for profiling LLM operations"""

    def decorator(func):
        @apm_trace_async("llm_request")
        @trace_async_operation("llm_request")
        @monitor_memory()
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                # Extract metrics from result
                if hasattr(result, "usage") and result.usage:
                    tokens_used = getattr(result.usage, "total_tokens", 0)
                    apm_profiler.custom_metric(
                        "llm.tokens_used", tokens_used, {"provider": provider, "model": model}
                    )

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000

                # Trace LLM request
                prompt = kwargs.get("prompt", args[0] if args else "")
                response_text = getattr(result, "content", str(result))

                otel_profiler.trace_llm_request(provider, model, prompt, response_text, latency_ms)
                apm_profiler.trace_llm_request(provider, model, prompt, response_text, latency_ms)

                return result

            except Exception as e:
                # Record error metrics
                apm_profiler.custom_metric(
                    "llm.errors",
                    1,
                    {"provider": provider, "model": model, "error_type": type(e).__name__},
                )
                raise

        return wrapper

    return decorator


def profile_transformation(technique: str):
    """Decorator for profiling transformation operations"""

    def decorator(func):
        @apm_trace_async("prompt_transformation")
        @trace_async_operation("prompt_transformation")
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            # Get input prompt
            input_prompt = kwargs.get("prompt", args[0] if args else "")
            input_length = len(input_prompt)

            try:
                result = await func(*args, **kwargs)

                # Get output prompt
                output_prompt = result if isinstance(result, str) else str(result)
                output_length = len(output_prompt)

                # Calculate metrics
                processing_time_ms = (time.time() - start_time) * 1000
                complexity_ratio = output_length / input_length if input_length > 0 else 1.0

                # Trace transformation
                otel_profiler.trace_transformation(
                    technique, input_prompt, output_prompt, processing_time_ms
                )
                apm_profiler.trace_transformation(
                    technique, input_length, output_length, processing_time_ms
                )

                # Record metrics
                apm_profiler.custom_metric(
                    "transformation.processing_time", processing_time_ms, {"technique": technique}
                )
                apm_profiler.custom_metric(
                    "transformation.complexity_ratio", complexity_ratio, {"technique": technique}
                )

                return result

            except Exception as e:
                # Record error
                apm_profiler.custom_metric(
                    "transformation.errors",
                    1,
                    {"technique": technique, "error_type": type(e).__name__},
                )
                raise

        return wrapper

    return decorator


def profile_jailbreak_attempt(technique: str):
    """Decorator for profiling jailbreak attempts (research purposes)"""

    def decorator(func):
        @apm_trace_async("jailbreak_attempt")
        @trace_async_operation("jailbreak_research")
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                # Extract success metrics from result
                success = getattr(result, "success", False)
                confidence_score = getattr(result, "confidence", 0.0)
                detection_bypassed = getattr(result, "detection_bypassed", False)

                # Trace jailbreak attempt
                otel_profiler.trace_jailbreak_attempt(
                    technique, success, confidence_score, detection_bypassed
                )

                # Record research metrics
                apm_profiler.custom_metric(
                    "research.jailbreak_attempts",
                    1,
                    {"technique": technique, "success": str(success)},
                )

                processing_time_ms = (time.time() - start_time) * 1000
                apm_profiler.custom_metric(
                    "jailbreak.processing_time", processing_time_ms, {"technique": technique}
                )

                return result

            except Exception as e:
                apm_profiler.custom_metric(
                    "jailbreak.errors", 1, {"technique": technique, "error_type": type(e).__name__}
                )
                raise

        return wrapper

    return decorator


# Startup and shutdown events
@asynccontextmanager
async def profiling_lifespan(_app: FastAPI):
    """Lifespan manager for profiling system"""
    # Startup
    logger.info("Starting Chimera Performance Profiling System")

    if config.profiling_enabled:
        await performance_monitor.start_monitoring()
        logger.info("Performance monitoring started")

    yield

    # Shutdown
    logger.info("Shutting down performance monitoring")
    await performance_monitor.stop_monitoring()


# Integration function to be called from main.py
def integrate_performance_profiling(app: FastAPI) -> None:
    """Main integration function - call this from your FastAPI app setup"""

    if not config.profiling_enabled:
        logger.info("Performance profiling is disabled")
        return

    logger.info(f"Setting up performance profiling (Environment: {config.environment.value})")

    # Set up all profiling components
    setup_profiling_integration(app)

    # Add lifespan manager
    app.router.lifespan_context = profiling_lifespan

    logger.info("Performance profiling integration complete")


# Export key components
__all__ = [
    "PerformanceMiddleware",
    "apm_profiler",
    "cpu_profiler",
    "db_profiler",
    "integrate_performance_profiling",
    "io_profiler",
    "memory_profiler",
    "otel_profiler",
    "performance_monitor",
    "profile_jailbreak_attempt",
    "profile_llm_operation",
    "profile_transformation",
]
