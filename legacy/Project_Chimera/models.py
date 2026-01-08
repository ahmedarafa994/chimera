from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class RequestLog(db.Model):
    """
    Tracks all incoming API requests for audit, debugging, and performance monitoring.
    Links to other logs via request_id.
    """

    __tablename__ = "request_logs"

    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.String(64), unique=True, index=True, nullable=False)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    endpoint = db.Column(db.String(64), index=True)
    method = db.Column(db.String(10))
    status_code = db.Column(db.Integer)
    latency_ms = db.Column(db.Float)
    ip_address = db.Column(db.String(64))

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "ip_address": self.ip_address,
        }


class LLMUsage(db.Model):
    """
    Tracks token usage, costs, and performance metrics associated with LLM provider calls.
    """

    __tablename__ = "llm_usage"

    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.String(64), index=True, nullable=False)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    provider = db.Column(db.String(64), index=True)
    model = db.Column(db.String(64))
    tokens_used = db.Column(db.Integer, default=0)
    cost = db.Column(db.Float, default=0.0)
    latency_ms = db.Column(db.Float)
    cached = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
        }


class TechniqueUsage(db.Model):
    """
    Tracks the application of prompt engineering techniques, including suites and potency.
    """

    __tablename__ = "technique_usage"

    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.String(64), index=True, nullable=False)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    technique_suite = db.Column(db.String(64), index=True)
    potency_level = db.Column(db.Integer)
    transformers_count = db.Column(db.Integer, default=0)
    framers_count = db.Column(db.Integer, default=0)
    obfuscators_count = db.Column(db.Integer, default=0)
    success = db.Column(db.Boolean, default=True)

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "technique_suite": self.technique_suite,
            "potency_level": self.potency_level,
            "transformers_count": self.transformers_count,
            "framers_count": self.framers_count,
            "obfuscators_count": self.obfuscators_count,
            "success": self.success,
        }


class ErrorLog(db.Model):
    """
    Centralized error logging table linked to specific requests and techniques.
    """

    __tablename__ = "error_logs"

    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.String(64), index=True, nullable=True)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    error_type = db.Column(db.String(100), index=True)
    error_message = db.Column(db.Text)
    stack_trace = db.Column(db.Text, nullable=True)
    provider = db.Column(db.String(64), nullable=True)
    technique = db.Column(db.String(64), nullable=True)

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "provider": self.provider,
            "technique": self.technique,
        }
