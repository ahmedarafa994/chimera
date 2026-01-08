"""
Automated Performance Monitoring and Alerting System
Comprehensive monitoring with intelligent alerting and automated response
"""

import importlib.util as _importlib
import json
import os
import smtplib
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from email.mime.multipart import MimeMultipart
from email.mime.text import MimeText
from typing import Any

from database_profiler import db_profiler
from io_network_profiler import io_profiler
from memory_profiler import memory_profiler
from profiling_config import MetricType, config

# Webhook and notification availability
HTTP_AVAILABLE = _importlib.find_spec("aiohttp") is not None or _importlib.find_spec("requests") is not None

# Slack notifications
SLACK_AVAILABLE = _importlib.find_spec("slack_sdk") is not None

# Discord notifications
DISCORD_AVAILABLE = _importlib.find_spec("discord") is not None


@dataclass
class Alert:
    """Performance alert data structure"""
    alert_id: str
    timestamp: datetime
    severity: str  # critical, warning, info
    category: str  # cpu, memory, network, database, frontend, custom
    metric_name: str
    current_value: float
    threshold_value: float
    service: str
    description: str
    suggested_actions: list[str]
    alert_data: dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: datetime | None = None

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    metric_path: str  # e.g., "system.cpu.usage"
    condition: str  # gt, lt, eq, gte, lte
    threshold: float
    severity: str
    duration_minutes: int  # Alert after condition persists for this duration
    cooldown_minutes: int  # Don't re-alert for this period
    enabled: bool = True
    tags: dict[str, str] = None
    description: str = ""

@dataclass
class NotificationConfig:
    """Notification configuration"""
    channel_type: str  # email, slack, discord, webhook
    enabled: bool
    config: dict[str, Any]
    severity_filter: list[str]  # Which severities to notify for

class MetricsCollector:
    """Centralized metrics collection"""

    def __init__(self):
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.collection_active = False
        self.collection_thread: threading.Thread | None = None

    def start_collection(self, interval: int = 30) -> None:
        """Start metrics collection"""
        if self.collection_active:
            return

        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval,),
            daemon=True
        )
        self.collection_thread.start()
        print(f"Metrics collection started with {interval}s interval")

    def stop_collection(self) -> None:
        """Stop metrics collection"""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)

    def _collection_loop(self, interval: int) -> None:
        """Main metrics collection loop"""
        while self.collection_active:
            try:
                timestamp = datetime.now(UTC)

                # Collect system metrics
                self._collect_system_metrics(timestamp)

                # Collect application metrics
                self._collect_application_metrics(timestamp)

                # Collect database metrics
                self._collect_database_metrics(timestamp)

                time.sleep(interval)

            except Exception as e:
                print(f"Error in metrics collection: {e}")
                time.sleep(interval)

    def _collect_system_metrics(self, timestamp: datetime) -> None:
        """Collect system-level metrics"""
        try:
            import psutil

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu.usage", cpu_percent, timestamp)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system.memory.usage_percent", memory.percent, timestamp)
            self.record_metric("system.memory.available_bytes", memory.available, timestamp)

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.record_metric("system.disk.read_bytes_per_sec", disk_io.read_bytes, timestamp)
                self.record_metric("system.disk.write_bytes_per_sec", disk_io.write_bytes, timestamp)

            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                self.record_metric("system.network.bytes_sent_per_sec", net_io.bytes_sent, timestamp)
                self.record_metric("system.network.bytes_recv_per_sec", net_io.bytes_recv, timestamp)

            # Process-specific metrics
            process = psutil.Process()
            self.record_metric("process.cpu.usage", process.cpu_percent(), timestamp)
            self.record_metric("process.memory.rss_bytes", process.memory_info().rss, timestamp)
            self.record_metric("process.threads.count", process.num_threads(), timestamp)

        except Exception as e:
            print(f"Error collecting system metrics: {e}")

    def _collect_application_metrics(self, timestamp: datetime) -> None:
        """Collect application-specific metrics"""
        try:
            # Get memory profiler data
            if memory_profiler.memory_snapshots:
                latest_snapshot = memory_profiler.memory_snapshots[-1]
                memory_mb = latest_snapshot.process_memory["rss"] / 1024 / 1024
                self.record_metric("app.memory.usage_mb", memory_mb, timestamp)

            # Get I/O profiler data
            if io_profiler.api_metrics:
                recent_api_metrics = list(io_profiler.api_metrics)[-10:]
                if recent_api_metrics:
                    avg_response_time = sum(m.response_time_ms for m in recent_api_metrics if m.response_time_ms > 0) / len(recent_api_metrics)
                    self.record_metric("app.api.avg_response_time_ms", avg_response_time, timestamp)

        except Exception as e:
            print(f"Error collecting application metrics: {e}")

    def _collect_database_metrics(self, timestamp: datetime) -> None:
        """Collect database performance metrics"""
        try:
            # Database query metrics
            if db_profiler.query_metrics:
                recent_queries = list(db_profiler.query_metrics)[-10:]
                if recent_queries:
                    avg_query_time = sum(q.execution_time_ms for q in recent_queries) / len(recent_queries)
                    self.record_metric("db.query.avg_execution_time_ms", avg_query_time, timestamp)

            # Cache metrics
            cache_stats = db_profiler.analyze_cache_performance()
            if cache_stats:
                self.record_metric("cache.hit_rate", cache_stats.get("overall_hit_rate", 0), timestamp)
                self.record_metric("cache.total_operations", cache_stats.get("total_operations", 0), timestamp)

        except Exception as e:
            print(f"Error collecting database metrics: {e}")

    def record_metric(self, metric_name: str, value: float, timestamp: datetime | None = None) -> None:
        """Record a metric value"""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        metric_data = {
            "value": value,
            "timestamp": timestamp
        }

        self.metrics[metric_name].append(metric_data)

    def get_metric_values(self, metric_name: str, duration_minutes: int = 60) -> list[float]:
        """Get metric values for the specified duration"""
        cutoff_time = datetime.now(UTC) - timedelta(minutes=duration_minutes)

        if metric_name not in self.metrics:
            return []

        values = []
        for metric_data in self.metrics[metric_name]:
            if metric_data["timestamp"] >= cutoff_time:
                values.append(metric_data["value"])

        return values

    def get_latest_value(self, metric_name: str) -> float | None:
        """Get the latest value for a metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None

        return self.metrics[metric_name][-1]["value"]

class AlertManager:
    """Alert management and notification system"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.notification_configs: list[NotificationConfig] = []

        # Alert cooldown tracking
        self.alert_cooldowns: dict[str, datetime] = {}

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None

        self._load_default_alert_rules()
        self._load_notification_configs()

    def _load_default_alert_rules(self) -> None:
        """Load default alert rules for Chimera system"""
        default_rules = [
            # System alerts
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                metric_path="system.cpu.usage",
                condition="gt",
                threshold=80.0,
                severity="warning",
                duration_minutes=2,
                cooldown_minutes=10,
                description="System CPU usage is above 80%"
            ),
            AlertRule(
                rule_id="critical_cpu_usage",
                name="Critical CPU Usage",
                metric_path="system.cpu.usage",
                condition="gt",
                threshold=95.0,
                severity="critical",
                duration_minutes=1,
                cooldown_minutes=5,
                description="System CPU usage is critically high (>95%)"
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                metric_path="system.memory.usage_percent",
                condition="gt",
                threshold=85.0,
                severity="warning",
                duration_minutes=3,
                cooldown_minutes=15,
                description="System memory usage is above 85%"
            ),
            AlertRule(
                rule_id="critical_memory_usage",
                name="Critical Memory Usage",
                metric_path="system.memory.usage_percent",
                condition="gt",
                threshold=95.0,
                severity="critical",
                duration_minutes=1,
                cooldown_minutes=5,
                description="System memory usage is critically high (>95%)"
            ),

            # Application alerts
            AlertRule(
                rule_id="slow_api_response",
                name="Slow API Response Time",
                metric_path="app.api.avg_response_time_ms",
                condition="gt",
                threshold=3000.0,
                severity="warning",
                duration_minutes=3,
                cooldown_minutes=10,
                description="Average API response time is above 3 seconds"
            ),
            AlertRule(
                rule_id="app_memory_leak",
                name="Application Memory Leak",
                metric_path="app.memory.usage_mb",
                condition="gt",
                threshold=2048.0,  # 2GB
                severity="critical",
                duration_minutes=5,
                cooldown_minutes=20,
                description="Application memory usage suggests potential memory leak"
            ),

            # Database alerts
            AlertRule(
                rule_id="slow_database_queries",
                name="Slow Database Queries",
                metric_path="db.query.avg_execution_time_ms",
                condition="gt",
                threshold=2000.0,
                severity="warning",
                duration_minutes=5,
                cooldown_minutes=15,
                description="Database queries are running slowly (>2s average)"
            ),
            AlertRule(
                rule_id="low_cache_hit_rate",
                name="Low Cache Hit Rate",
                metric_path="cache.hit_rate",
                condition="lt",
                threshold=0.7,  # 70% hit rate
                severity="warning",
                duration_minutes=10,
                cooldown_minutes=30,
                description="Cache hit rate is below 70%"
            )
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    def _load_notification_configs(self) -> None:
        """Load notification configurations"""
        # Email notifications
        if os.getenv("SMTP_SERVER"):
            email_config = NotificationConfig(
                channel_type="email",
                enabled=True,
                config={
                    "smtp_server": os.getenv("SMTP_SERVER"),
                    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                    "smtp_username": os.getenv("SMTP_USERNAME"),
                    "smtp_password": os.getenv("SMTP_PASSWORD"),
                    "from_email": os.getenv("FROM_EMAIL", "alerts@chimera.ai"),
                    "to_emails": os.getenv("ALERT_EMAILS", "").split(",")
                },
                severity_filter=["critical", "warning"]
            )
            self.notification_configs.append(email_config)

        # Slack notifications
        if os.getenv("SLACK_BOT_TOKEN"):
            slack_config = NotificationConfig(
                channel_type="slack",
                enabled=SLACK_AVAILABLE,
                config={
                    "bot_token": os.getenv("SLACK_BOT_TOKEN"),
                    "channel": os.getenv("SLACK_CHANNEL", "#alerts")
                },
                severity_filter=["critical", "warning", "info"]
            )
            self.notification_configs.append(slack_config)

        # Webhook notifications
        if os.getenv("WEBHOOK_URL"):
            webhook_config = NotificationConfig(
                channel_type="webhook",
                enabled=HTTP_AVAILABLE,
                config={
                    "url": os.getenv("WEBHOOK_URL"),
                    "headers": json.loads(os.getenv("WEBHOOK_HEADERS", "{}"))
                },
                severity_filter=["critical", "warning"]
            )
            self.notification_configs.append(webhook_config)

    def start_monitoring(self, interval: int = 30) -> None:
        """Start alert monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        print(f"Alert monitoring started with {interval}s interval")

    def stop_monitoring(self) -> None:
        """Stop alert monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

    def _monitoring_loop(self, interval: int) -> None:
        """Main alert monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_alert_rules()
                self._check_alert_resolutions()
                time.sleep(interval)

            except Exception as e:
                print(f"Error in alert monitoring: {e}")
                time.sleep(interval)

    def _check_alert_rules(self) -> None:
        """Check all alert rules against current metrics"""
        current_time = datetime.now(UTC)

        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            # Check cooldown
            if rule.rule_id in self.alert_cooldowns:
                cooldown_expires = self.alert_cooldowns[rule.rule_id] + timedelta(minutes=rule.cooldown_minutes)
                if current_time < cooldown_expires:
                    continue

            # Get metric values for the duration
            values = self.metrics_collector.get_metric_values(rule.metric_path, rule.duration_minutes)

            if not values:
                continue

            # Check if condition is met for the entire duration
            condition_met = self._evaluate_condition(values, rule.condition, rule.threshold)

            if condition_met:
                # Create or update alert
                self._trigger_alert(rule, values[-1], current_time)

    def _evaluate_condition(self, values: list[float], condition: str, threshold: float) -> bool:
        """Evaluate alert condition against metric values"""
        if not values:
            return False

        # For duration-based alerts, all values in the period should meet the condition
        if condition == "gt":
            return all(v > threshold for v in values)
        elif condition == "gte":
            return all(v >= threshold for v in values)
        elif condition == "lt":
            return all(v < threshold for v in values)
        elif condition == "lte":
            return all(v <= threshold for v in values)
        elif condition == "eq":
            return all(abs(v - threshold) < 0.001 for v in values)  # Float equality with tolerance
        else:
            return False

    def _trigger_alert(self, rule: AlertRule, current_value: float, timestamp: datetime) -> None:
        """Trigger an alert"""
        alert_id = f"{rule.rule_id}_{int(timestamp.timestamp())}"

        # Check if alert already exists and is active
        existing_alert_key = f"{rule.rule_id}_active"
        if existing_alert_key in self.active_alerts:
            return  # Alert already active

        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(rule, current_value)

        alert = Alert(
            alert_id=alert_id,
            timestamp=timestamp,
            severity=rule.severity,
            category=self._get_metric_category(rule.metric_path),
            metric_name=rule.metric_path,
            current_value=current_value,
            threshold_value=rule.threshold,
            service="chimera-system",
            description=rule.description,
            suggested_actions=suggested_actions,
            alert_data={"rule_id": rule.rule_id}
        )

        # Store alert
        self.active_alerts[existing_alert_key] = alert
        self.alert_history.append(alert)

        # Set cooldown
        self.alert_cooldowns[rule.rule_id] = timestamp

        # Send notifications
        self._send_notifications(alert)

        print(f"ALERT TRIGGERED [{alert.severity.upper()}]: {alert.description} - Current: {current_value:.2f}, Threshold: {rule.threshold:.2f}")

    def _check_alert_resolutions(self) -> None:
        """Check if active alerts should be resolved"""
        current_time = datetime.now(UTC)
        resolved_alerts = []

        for alert_key, alert in self.active_alerts.items():
            if alert.resolved:
                continue

            rule = self.alert_rules.get(alert.alert_data.get("rule_id"))
            if not rule:
                continue

            # Get recent metric values
            values = self.metrics_collector.get_metric_values(rule.metric_path, 5)  # Last 5 minutes

            if values:
                # Check if condition is no longer met
                condition_met = self._evaluate_condition(values, rule.condition, rule.threshold)

                if not condition_met:
                    # Resolve alert
                    alert.resolved = True
                    alert.resolution_time = current_time
                    resolved_alerts.append(alert_key)

                    # Send resolution notification
                    self._send_resolution_notification(alert)

                    print(f"ALERT RESOLVED: {alert.description}")

        # Remove resolved alerts from active alerts
        for alert_key in resolved_alerts:
            del self.active_alerts[alert_key]

    def _generate_suggested_actions(self, rule: AlertRule, _current_value: float) -> list[str]:
        """Generate suggested actions for an alert"""
        actions = []

        if "cpu" in rule.metric_path.lower():
            actions.extend([
                "Check for high CPU processes using top or htop",
                "Review recent deployments or code changes",
                "Consider scaling horizontally if sustained",
                "Investigate potential infinite loops or inefficient algorithms"
            ])

        elif "memory" in rule.metric_path.lower():
            actions.extend([
                "Check for memory leaks using memory profiler",
                "Review recent memory-intensive operations",
                "Consider increasing available memory",
                "Investigate object retention and garbage collection"
            ])

        elif "api" in rule.metric_path.lower():
            actions.extend([
                "Review slow API endpoints",
                "Check database query performance",
                "Verify network connectivity to external services",
                "Consider implementing response caching"
            ])

        elif "database" in rule.metric_path.lower():
            actions.extend([
                "Analyze slow query logs",
                "Check database connection pool utilization",
                "Review recent database schema changes",
                "Consider query optimization and indexing"
            ])

        elif "cache" in rule.metric_path.lower():
            actions.extend([
                "Review cache key patterns and TTL values",
                "Check cache server health and connectivity",
                "Analyze cache eviction policies",
                "Consider cache warming strategies"
            ])

        # Add general actions
        actions.extend([
            "Monitor system resources and application logs",
            "Check for any recent configuration changes",
            "Verify external service dependencies"
        ])

        return actions[:5]  # Limit to top 5 actions

    def _get_metric_category(self, metric_path: str) -> str:
        """Determine metric category from metric path"""
        if "cpu" in metric_path.lower():
            return "cpu"
        elif "memory" in metric_path.lower():
            return "memory"
        elif "network" in metric_path.lower():
            return "network"
        elif "database" in metric_path.lower() or "db" in metric_path.lower():
            return "database"
        elif "api" in metric_path.lower():
            return "frontend"
        else:
            return "custom"

    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert"""
        for notification_config in self.notification_configs:
            if not notification_config.enabled or alert.severity not in notification_config.severity_filter:
                continue

            try:
                if notification_config.channel_type == "email":
                    self._send_email_notification(alert, notification_config)
                elif notification_config.channel_type == "slack":
                    self._send_slack_notification(alert, notification_config)
                elif notification_config.channel_type == "webhook":
                    self._send_webhook_notification(alert, notification_config)

            except Exception as e:
                print(f"Error sending {notification_config.channel_type} notification: {e}")

    def _send_email_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send email notification"""
        try:
            smtp_config = config.config

            msg = MimeMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = ', '.join(smtp_config['to_emails'])
            msg['Subject'] = f"[{alert.severity.upper()}] Chimera Alert: {alert.description}"

            body = f"""
Alert Details:
- Severity: {alert.severity.upper()}
- Category: {alert.category}
- Metric: {alert.metric_name}
- Current Value: {alert.current_value:.2f}
- Threshold: {alert.threshold_value:.2f}
- Service: {alert.service}
- Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Description:
{alert.description}

Suggested Actions:
{chr(10).join(f'- {action}' for action in alert.suggested_actions)}

Alert ID: {alert.alert_id}
            """

            msg.attach(MimeText(body, 'plain'))

            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
            server.starttls()
            server.login(smtp_config['smtp_username'], smtp_config['smtp_password'])
            server.send_message(msg)
            server.quit()

        except Exception as e:
            print(f"Failed to send email notification: {e}")

    def _send_slack_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send Slack notification"""
        if not SLACK_AVAILABLE:
            return

        try:
            from slack_sdk import WebClient

            client = WebClient(token=config.config['bot_token'])

            # Create Slack message
            {"critical": "#ff0000", "warning": "#ffff00", "info": "#00ff00"}.get(alert.severity, "#808080")

            blocks = [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"🚨 {alert.severity.upper()} Alert"}
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Metric:* {alert.metric_name}"},
                        {"type": "mrkdwn", "text": f"*Current:* {alert.current_value:.2f}"},
                        {"type": "mrkdwn", "text": f"*Threshold:* {alert.threshold_value:.2f}"},
                        {"type": "mrkdwn", "text": f"*Service:* {alert.service}"}
                    ]
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Description:* {alert.description}"}
                }
            ]

            client.chat_postMessage(
                channel=config.config['channel'],
                blocks=blocks,
                text=f"{alert.severity.upper()} Alert: {alert.description}"
            )

        except Exception as e:
            print(f"Failed to send Slack notification: {e}")

    def _send_webhook_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send webhook notification"""
        if not HTTP_AVAILABLE:
            return

        try:
            import requests

            webhook_data = {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity,
                "category": alert.category,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "service": alert.service,
                "description": alert.description,
                "suggested_actions": alert.suggested_actions,
            }

            response = requests.post(
                config.config['url'],
                json=webhook_data,
                headers=config.config.get('headers', {}),
                timeout=10,
            )

            response.raise_for_status()

        except Exception as e:
            print(f"Failed to send webhook notification: {e}")

    def _send_resolution_notification(self, alert: Alert) -> None:
        """Send alert resolution notification"""
        resolution_message = f"Alert resolved: {alert.description} (Duration: {(alert.resolution_time - alert.timestamp).total_seconds():.0f}s)"
        print(f"RESOLUTION: {resolution_message}")

        # Could send resolution notifications via same channels if needed

    def add_custom_alert_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule"""
        self.alert_rules[rule.rule_id] = rule

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.active_alerts.values():
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert system summary"""
        active_alerts = list(self.active_alerts.values())

        return {
            "total_active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.severity == "critical"]),
            "warning_alerts": len([a for a in active_alerts if a.severity == "warning"]),
            "info_alerts": len([a for a in active_alerts if a.severity == "info"]),
            "total_alert_rules": len(self.alert_rules),
            "enabled_alert_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "notification_channels": len([c for c in self.notification_configs if c.enabled])
        }

class PerformanceMonitor:
    """Main performance monitoring orchestrator"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.monitoring_active = False

    async def start_monitoring(self) -> None:
        """Start comprehensive performance monitoring"""
        print("Starting Chimera Performance Monitoring System...")

        # Start metrics collection
        self.metrics_collector.start_collection(interval=30)

        # Start alert monitoring
        self.alert_manager.start_monitoring(interval=30)

        # Start profilers
        if config.is_metric_enabled(MetricType.MEMORY):
            memory_profiler.start_monitoring(interval=60)

        if config.is_metric_enabled(MetricType.IO) or config.is_metric_enabled(MetricType.NETWORK):
            await io_profiler.start_monitoring(interval=60)

        self.monitoring_active = True
        print("Performance monitoring system started successfully")

    async def stop_monitoring(self) -> None:
        """Stop all monitoring"""
        print("Stopping performance monitoring...")

        self.metrics_collector.stop_collection()
        self.alert_manager.stop_monitoring()
        memory_profiler.stop_monitoring()
        await io_profiler.stop_monitoring()

        self.monitoring_active = False
        print("Performance monitoring stopped")

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get dashboard data for monitoring UI"""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "monitoring_active": self.monitoring_active,
            "system_metrics": {
                "cpu_usage": self.metrics_collector.get_latest_value("system.cpu.usage"),
                "memory_usage": self.metrics_collector.get_latest_value("system.memory.usage_percent"),
                "api_response_time": self.metrics_collector.get_latest_value("app.api.avg_response_time_ms")
            },
            "alerts": self.alert_manager.get_alert_summary(),
            "active_alerts": [asdict(alert) for alert in self.alert_manager.get_active_alerts()]
        }

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Convenience functions
async def start_performance_monitoring() -> None:
    """Start the comprehensive performance monitoring system"""
    await performance_monitor.start_monitoring()

async def stop_performance_monitoring() -> None:
    """Stop the performance monitoring system"""
    await performance_monitor.stop_monitoring()

def get_monitoring_status() -> dict[str, Any]:
    """Get current monitoring status"""
    return performance_monitor.get_dashboard_data()
