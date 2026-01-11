"""
Audit Logging Module
Provides tamper-evident audit logging for security and compliance

Features:
- Hash chain for tamper detection
- Structured audit events
- Multiple storage backends
- Query and verification capabilities
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Audit Action Definitions
# =============================================================================


class AuditAction(str, Enum):
    """Enumeration of auditable actions"""

    # Authentication events
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"
    AUTH_TOKEN_REFRESH = "auth.token_refresh"
    AUTH_MFA_CHALLENGE = "auth.mfa_challenge"
    AUTH_MFA_VERIFIED = "auth.mfa_verified"

    # API key events
    API_KEY_CREATED = "apikey.created"
    API_KEY_ROTATED = "apikey.rotated"
    API_KEY_REVOKED = "apikey.revoked"
    API_KEY_USED = "apikey.used"

    # Prompt transformation events
    PROMPT_TRANSFORM = "prompt.transform"
    PROMPT_ENHANCE = "prompt.enhance"
    PROMPT_JAILBREAK = "prompt.jailbreak"
    PROMPT_BATCH_PROCESS = "prompt.batch_process"

    # Configuration events
    CONFIG_CHANGE = "config.change"
    CONFIG_VIEW = "config.view"

    # User management events
    USER_CREATE = "user.create"
    USER_MODIFY = "user.modify"
    USER_DELETE = "user.delete"
    USER_ROLE_CHANGE = "user.role_change"
    USER_REGISTER = "user.register"
    USER_VERIFY = "user.verify"
    USER_PASSWORD_CHANGE = "user.password_change"
    USER_PROFILE_UPDATE = "user.profile_update"
    USER_INVITE = "user.invite"

    # Campaign events
    CAMPAIGN_CREATE = "campaign.create"
    CAMPAIGN_UPDATE = "campaign.update"
    CAMPAIGN_DELETE = "campaign.delete"
    CAMPAIGN_SHARE = "campaign.share"
    CAMPAIGN_UNSHARE = "campaign.unshare"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"

    # Security events
    SECURITY_RATE_LIMIT = "security.rate_limit"
    SECURITY_BLOCKED_REQUEST = "security.blocked_request"
    SECURITY_INJECTION_ATTEMPT = "security.injection_attempt"
    SECURITY_UNAUTHORIZED = "security.unauthorized"


class AuditSeverity(str, Enum):
    """Severity levels for audit events"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Audit Entry Data Class
# =============================================================================


@dataclass
class AuditEntry:
    """Represents a single audit log entry"""

    timestamp: str
    action: str
    severity: str
    user_id: str | None
    resource: str
    details: dict[str, Any]
    ip_address: str | None
    request_id: str | None
    user_agent: str | None
    previous_hash: str
    hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEntry":
        return cls(**data)


# =============================================================================
# Storage Backends
# =============================================================================


class AuditStorage(ABC):
    """Abstract base class for audit storage backends"""

    @abstractmethod
    def store(self, entry: AuditEntry) -> None:
        """Store an audit entry"""
        ...

    @abstractmethod
    def get_all(self) -> list[AuditEntry]:
        """Retrieve all audit entries"""
        ...

    @abstractmethod
    def query(
        self,
        action: AuditAction | None = None,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with filters"""
        ...


class InMemoryAuditStorage(AuditStorage):
    """In-memory audit storage (for development/testing)"""

    def __init__(self, max_entries: int = 10000):
        self._entries: list[AuditEntry] = []
        self._max_entries = max_entries

    def store(self, entry: AuditEntry) -> None:
        self._entries.append(entry)
        # Rotate old entries if limit exceeded
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

    def get_all(self) -> list[AuditEntry]:
        return list(self._entries)

    def query(
        self,
        action: AuditAction | None = None,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        results = []

        for entry in reversed(self._entries):
            if action and entry.action != action.value:
                continue
            if user_id and entry.user_id != user_id:
                continue
            if start_time:
                entry_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                if entry_time < start_time:
                    continue
            if end_time:
                entry_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                if entry_time > end_time:
                    continue

            results.append(entry)
            if len(results) >= limit:
                break

        return results


class FileAuditStorage(AuditStorage):
    """File-based audit storage with daily rotation"""

    def __init__(self, log_dir: str = "logs/audit"):
        import os

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def _get_log_file(self, date: datetime | None = None) -> str:
        if date is None:
            date = datetime.utcnow()
        filename = f"audit_{date.strftime('%Y-%m-%d')}.jsonl"
        return f"{self.log_dir}/{filename}"

    def store(self, entry: AuditEntry) -> None:
        log_file = self._get_log_file()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def get_all(self) -> list[AuditEntry]:
        import glob

        entries = []
        for log_file in sorted(glob.glob(f"{self.log_dir}/audit_*.jsonl")):
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(AuditEntry.from_dict(json.loads(line)))
        return entries

    def query(
        self,
        action: AuditAction | None = None,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        all_entries = self.get_all()
        results = []

        for entry in reversed(all_entries):
            if action and entry.action != action.value:
                continue
            if user_id and entry.user_id != user_id:
                continue
            if start_time:
                entry_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                if entry_time < start_time:
                    continue
            if end_time:
                entry_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                if entry_time > end_time:
                    continue

            results.append(entry)
            if len(results) >= limit:
                break

        return results


# =============================================================================
# Main Audit Logger
# =============================================================================


class AuditLogger:
    """
    Main audit logger with tamper-evident hash chain.

    Usage:
        audit = AuditLogger()
        audit.log(
            AuditAction.PROMPT_TRANSFORM,
            user_id="user123",
            resource="/api/v1/transform",
            details={"technique": "gptfuzz", "prompt_length": 150}
        )
    """

    def __init__(self, storage: AuditStorage = None):
        self.storage = storage or FileAuditStorage()
        self._last_hash = self._get_initial_hash()
        self._logger = logging.getLogger("chimera.audit")

    def _get_initial_hash(self) -> str:
        """Get the last hash from storage or initialize"""
        entries = self.storage.get_all()
        if entries:
            return entries[-1].hash
        return "0" * 64  # Genesis hash

    def _compute_hash(self, entry_data: dict[str, Any]) -> str:
        """Compute SHA-256 hash of entry data"""
        entry_json = json.dumps(entry_data, sort_keys=True, default=str)
        return hashlib.sha256(entry_json.encode()).hexdigest()

    def log(
        self,
        action: AuditAction,
        user_id: str | None = None,
        resource: str = "",
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        request_id: str | None = None,
        user_agent: str | None = None,
        severity: AuditSeverity = AuditSeverity.INFO,
    ) -> str:
        """
        Create a tamper-evident audit log entry.

        Args:
            action: The audit action being logged
            user_id: ID of the user performing the action
            resource: The resource being accessed/modified
            details: Additional details about the action
            ip_address: Client IP address
            request_id: Request correlation ID
            user_agent: Client user agent string
            severity: Severity level of the event

        Returns:
            Hash of the created entry
        """
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Sanitize details to remove sensitive data
        sanitized_details = self._sanitize_details(details or {})

        # Create entry without hash
        entry_data = {
            "timestamp": timestamp,
            "action": action.value,
            "severity": severity.value,
            "user_id": user_id,
            "resource": resource,
            "details": sanitized_details,
            "ip_address": ip_address,
            "request_id": request_id,
            "user_agent": user_agent,
            "previous_hash": self._last_hash,
        }

        # Compute hash
        entry_hash = self._compute_hash(entry_data)

        # Create full entry
        entry = AuditEntry(**entry_data, hash=entry_hash)

        # Store and update chain
        self.storage.store(entry)
        self._last_hash = entry_hash

        # Log to standard logger as well
        log_message = f"AUDIT [{action.value}] user={user_id} resource={resource}"
        if severity == AuditSeverity.CRITICAL:
            self._logger.critical(log_message)
        elif severity == AuditSeverity.ERROR:
            self._logger.error(log_message)
        elif severity == AuditSeverity.WARNING:
            self._logger.warning(log_message)
        else:
            self._logger.info(log_message)

        return entry_hash

    def _sanitize_details(self, details: dict[str, Any]) -> dict[str, Any]:
        """Remove sensitive data from details before logging"""
        sensitive_keys = {
            "password",
            "secret",
            "api_key",
            "token",
            "authorization",
            "credit_card",
            "ssn",
            "private_key",
        }

        sanitized = {}
        for key, value in details.items():
            lower_key = key.lower()
            if any(sk in lower_key for sk in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_details(value)
            else:
                sanitized[key] = value

        return sanitized

    def verify_chain(self) -> tuple[bool, str | None, int | None]:
        """
        Verify the integrity of the audit log chain.

        Returns:
            Tuple of (is_valid, failed_hash, failed_index)
        """
        entries = self.storage.get_all()
        previous_hash = "0" * 64  # Genesis hash

        for i, entry in enumerate(entries):
            # Reconstruct entry data without hash
            entry_data = {
                "timestamp": entry.timestamp,
                "action": entry.action,
                "severity": entry.severity,
                "user_id": entry.user_id,
                "resource": entry.resource,
                "details": entry.details,
                "ip_address": entry.ip_address,
                "request_id": entry.request_id,
                "user_agent": entry.user_agent,
                "previous_hash": previous_hash,
            }

            # Compute expected hash
            computed_hash = self._compute_hash(entry_data)

            # Verify
            if computed_hash != entry.hash:
                logger.error(f"Audit chain verification failed at entry {i}")
                return False, entry.hash, i

            # Verify previous hash link
            if entry.previous_hash != previous_hash:
                logger.error(f"Audit chain link broken at entry {i}")
                return False, entry.hash, i

            previous_hash = entry.hash

        logger.info(f"Audit chain verified: {len(entries)} entries")
        return True, None, None

    def query(
        self,
        action: AuditAction | None = None,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with filters"""
        return self.storage.query(
            action=action, user_id=user_id, start_time=start_time, end_time=end_time, limit=limit
        )

    def get_user_activity(self, user_id: str, limit: int = 50) -> list[AuditEntry]:
        """Get recent activity for a specific user"""
        return self.query(user_id=user_id, limit=limit)

    def get_security_events(self, limit: int = 100) -> list[AuditEntry]:
        """Get security-related events"""
        security_actions = [
            AuditAction.AUTH_FAILED,
            AuditAction.SECURITY_RATE_LIMIT,
            AuditAction.SECURITY_BLOCKED_REQUEST,
            AuditAction.SECURITY_INJECTION_ATTEMPT,
            AuditAction.SECURITY_UNAUTHORIZED,
        ]

        all_entries = []
        for action in security_actions:
            entries = self.query(action=action, limit=limit)
            all_entries.extend(entries)

        # Sort by timestamp and limit
        all_entries.sort(key=lambda e: e.timestamp, reverse=True)
        return all_entries[:limit]


# =============================================================================
# Singleton Instance
# =============================================================================

_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the singleton audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_log(
    action: AuditAction,
    user_id: str | None = None,
    resource: str = "",
    details: dict[str, Any] | None = None,
    **kwargs,
) -> str:
    """Convenience function to log an audit event"""
    return get_audit_logger().log(
        action=action, user_id=user_id, resource=resource, details=details, **kwargs
    )


# =============================================================================
# FastAPI Integration
# =============================================================================


def create_audit_middleware():
    """Create FastAPI middleware for automatic request auditing"""
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware

    class AuditMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Extract request info
            request_id = request.headers.get("X-Request-ID", "")
            user_agent = request.headers.get("User-Agent", "")
            ip_address = request.client.host if request.client else None

            # Store in request state for use in endpoints
            request.state.audit_context = {
                "request_id": request_id,
                "user_agent": user_agent,
                "ip_address": ip_address,
            }

            response = await call_next(request)
            return response

    return AuditMiddleware
