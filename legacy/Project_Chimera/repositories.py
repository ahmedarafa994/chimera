from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import case, desc, func

from app.extensions import db
from app.models import ErrorLog, LLMUsage, RequestLog, TechniqueUsage


class MetricsRepository:
    """
    Repository for persisting and retrieving monitoring metrics using SQLAlchemy.
    Provides persistent storage backing for the MonitoringDashboard.
    """

    def __init__(self, session=None):
        """
        Initialize the repository.
        :param session: SQLAlchemy session. If None, uses the default db.session.
        """
        self.session = session if session else db.session

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    def log_request(
        self,
        request_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        ip_address: str,
    ):
        """Log an incoming API request."""
        try:
            log = RequestLog(
                request_id=request_id,
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                latency_ms=latency_ms,
                ip_address=ip_address,
            )
            self.session.add(log)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            # In a real app, we might log to a file here to avoid infinite recursion if DB is down
            print(f"Error logging request: {e}")

    def log_llm_usage(
        self,
        request_id: str,
        provider: str,
        model: str,
        tokens_used: int,
        cost: float,
        latency_ms: float,
        cached: bool,
    ):
        """Log LLM provider usage."""
        try:
            usage = LLMUsage(
                request_id=request_id,
                provider=provider,
                model=model,
                tokens_used=tokens_used,
                cost=cost,
                latency_ms=latency_ms,
                cached=cached,
            )
            self.session.add(usage)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error logging LLM usage: {e}")

    def log_technique_usage(
        self,
        request_id: str,
        technique_suite: str,
        potency_level: int,
        transformers_count: int,
        framers_count: int,
        obfuscators_count: int,
        success: bool,
    ):
        """Log technique suite usage."""
        try:
            usage = TechniqueUsage(
                request_id=request_id,
                technique_suite=technique_suite,
                potency_level=potency_level,
                transformers_count=transformers_count,
                framers_count=framers_count,
                obfuscators_count=obfuscators_count,
                success=success,
            )
            self.session.add(usage)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error logging technique usage: {e}")

    def log_error(
        self,
        error_type: str,
        error_message: str,
        request_id: str | None = None,
        stack_trace: str | None = None,
        provider: str | None = None,
        technique: str | None = None,
    ):
        """Log a system or processing error."""
        try:
            error = ErrorLog(
                request_id=request_id,
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace,
                provider=provider,
                technique=technique,
            )
            self.session.add(error)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error logging error: {e}")

    # -------------------------------------------------------------------------
    # Read Operations (Dashboard Support)
    # -------------------------------------------------------------------------

    def get_provider_metrics(self, window_minutes: int = 60) -> dict[str, Any]:
        """
        Aggregate metrics by provider for the specified time window.
        Returns a dictionary keyed by provider name.
        """
        since = datetime.utcnow() - timedelta(minutes=window_minutes)

        results = (
            self.session.query(
                LLMUsage.provider,
                func.count(LLMUsage.id).label("count"),
                func.sum(LLMUsage.tokens_used).label("tokens"),
                func.sum(LLMUsage.cost).label("cost"),
                func.avg(LLMUsage.latency_ms).label("latency"),
                func.sum(case((LLMUsage.cached, 1), else_=0)).label("hits"),
            )
            .filter(LLMUsage.timestamp >= since)
            .group_by(LLMUsage.provider)
            .all()
        )

        metrics = {}
        for r in results:
            total_requests = r.count
            hits = int(r.hits or 0)
            metrics[r.provider] = {
                "total_requests": total_requests,
                "total_tokens": int(r.tokens or 0),
                "total_cost": float(r.cost or 0.0),
                "avg_latency_ms": float(r.latency or 0.0),
                "cache_hits": hits,
                "cache_misses": total_requests - hits,
                "cache_hit_rate": (hits / total_requests * 100) if total_requests > 0 else 0,
            }
        return metrics

    def get_technique_metrics(self, window_minutes: int = 60) -> dict[str, Any]:
        """
        Aggregate metrics by technique suite for the specified time window.
        """
        since = datetime.utcnow() - timedelta(minutes=window_minutes)

        results = (
            self.session.query(
                TechniqueUsage.technique_suite,
                func.count(TechniqueUsage.id).label("count"),
                func.sum(case((TechniqueUsage.success, 1), else_=0)).label("successes"),
                func.avg(TechniqueUsage.potency_level).label("avg_potency"),
            )
            .filter(TechniqueUsage.timestamp >= since)
            .group_by(TechniqueUsage.technique_suite)
            .all()
        )

        metrics = {}
        for r in results:
            total = r.count
            successes = int(r.successes or 0)
            metrics[r.technique_suite] = {
                "total_uses": total,
                "successful_uses": successes,
                "failed_uses": total - successes,
                "success_rate": (successes / total * 100) if total > 0 else 0,
                "avg_potency": float(r.avg_potency or 0.0),
            }
        return metrics

    def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Retrieve the most recent errors.
        """
        errors = self.session.query(ErrorLog).order_by(desc(ErrorLog.timestamp)).limit(limit).all()

        return [e.to_dict() for e in errors]

    def get_dashboard_summary(self, window_minutes: int = 60) -> dict[str, Any]:
        """
        Get high-level dashboard summary statistics.
        """
        since = datetime.utcnow() - timedelta(minutes=window_minutes)

        # Total Requests & Success Rate
        total_reqs = (
            self.session.query(func.count(RequestLog.id))
            .filter(RequestLog.timestamp >= since)
            .scalar()
            or 0
        )

        # Calculate success rate based on 200 OK status codes
        successful_reqs = (
            self.session.query(func.count(RequestLog.id))
            .filter(RequestLog.timestamp >= since, RequestLog.status_code == 200)
            .scalar()
            or 0
        )

        # LLM Totals
        llm_stats = (
            self.session.query(
                func.sum(LLMUsage.tokens_used).label("tokens"),
                func.sum(LLMUsage.cost).label("cost"),
            )
            .filter(LLMUsage.timestamp >= since)
            .first()
        )

        return {
            "window_minutes": window_minutes,
            "total_requests": total_reqs,
            "successful_requests": successful_reqs,
            "failed_requests": total_reqs - successful_reqs,
            "success_rate": (successful_reqs / total_reqs * 100) if total_reqs > 0 else 0,
            "total_tokens": int(llm_stats.tokens or 0) if llm_stats else 0,
            "total_cost": float(llm_stats.cost or 0.0) if llm_stats else 0.0,
        }

    def get_latency_timeseries(self, window_minutes: int = 60) -> list[dict[str, Any]]:
        """
        Get average latency grouped by minute for the last window.
        """
        since = datetime.utcnow() - timedelta(minutes=window_minutes)

        # DB-agnostic time grouping via Python processing
        # This avoids SQL dialect differences (SQLite vs Postgres) for 'date_trunc'
        logs = (
            self.session.query(RequestLog.timestamp, RequestLog.latency_ms)
            .filter(RequestLog.timestamp >= since)
            .order_by(RequestLog.timestamp)
            .all()
        )

        # Bucketing by minute
        buckets = {}
        for log in logs:
            minute_key = log.timestamp.replace(second=0, microsecond=0)
            if minute_key not in buckets:
                buckets[minute_key] = []
            buckets[minute_key].append(log.latency_ms)

        result = []
        # Sorting keys to ensure chronological order
        for key in sorted(buckets.keys()):
            vals = buckets[key]
            avg = sum(vals) / len(vals)
            result.append({"timestamp": key.isoformat(), "avg_latency": avg, "count": len(vals)})

        return result
