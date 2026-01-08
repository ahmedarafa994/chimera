"""
Audit Log Endpoints

Provides access to system audit logs for compliance and security monitoring.
"""

from contextlib import suppress
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.core.audit import AuditAction, AuditEntry, get_audit_logger
from app.core.config import settings

router = APIRouter(prefix="/audit", tags=["audit"])
security = HTTPBearer()


class AuditLogResponse(BaseModel):
    logs: list[AuditEntry]
    total: int
    verified: bool


class AuditStatsResponse(BaseModel):
    total_events: int
    events_by_severity: dict[str, int]
    events_by_action: dict[str, int]
    last_verification: str


async def verify_admin_access(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify admin access.
    In a real app, this would check roles/claims.
    For now, we check the API key against the admin key.
    """
    if not credentials or credentials.credentials != settings.CHIMERA_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key"
        )
    return credentials.credentials


@router.get(
    "/logs",
    response_model=AuditLogResponse,
    summary="Query audit logs",
    description="Retrieve audit logs with filtering capabilities.",
)
async def get_audit_logs(
    action: str | None = Query(None, description="Filter by action type"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    start_date: datetime | None = Query(None, description="Start date filter"),
    end_date: datetime | None = Query(None, description="End date filter"),
    limit: int = Query(100, ge=1, le=1000),
    severity: str | None = Query(None, description="Filter by severity"),
    _admin: str = Depends(verify_admin_access),
):
    """
    Retrieve audit logs based on filters.
    """
    logger = get_audit_logger()

    # Convert string action to Enum if provided
    audit_action = None
    if action:
        with suppress(ValueError):
            audit_action = AuditAction(action)

    logs = logger.query(
        action=audit_action, user_id=user_id, start_time=start_date, end_time=end_date, limit=limit
    )

    # Manual severity filter since query() might not support it directly in all implementations
    if severity:
        logs = [log for log in logs if log.severity == severity]

    # Verify chain integrity for the returned logs (simplified check)
    is_valid, _, _ = logger.verify_chain()

    return AuditLogResponse(logs=logs, total=len(logs), verified=is_valid)


@router.get(
    "/stats",
    response_model=AuditStatsResponse,
    summary="Get audit statistics",
    description="Retrieve summary statistics of audit events.",
)
async def get_audit_stats(_admin: str = Depends(verify_admin_access)):
    """
    Get statistical summary of audit logs.
    """
    logger = get_audit_logger()
    all_logs = (
        logger.storage.get_all()
    )  # Get all logs for stats (might be heavy in prod, use DB agg)

    severity_counts = {}
    action_counts = {}

    for log in all_logs:
        severity_counts[log.severity] = severity_counts.get(log.severity, 0) + 1
        action_counts[log.action] = action_counts.get(log.action, 0) + 1

    return AuditStatsResponse(
        total_events=len(all_logs),
        events_by_severity=severity_counts,
        events_by_action=action_counts,
        last_verification=datetime.utcnow().isoformat(),
    )


@router.post(
    "/verify",
    summary="Verify audit chain",
    description="Trigger a cryptographic verification of the audit log chain.",
)
async def verify_audit_chain(_admin: str = Depends(verify_admin_access)):
    """
    Cryptographically verify the tamper-evident hash chain.
    """
    logger = get_audit_logger()
    is_valid, failed_hash, failed_index = logger.verify_chain()

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail={
                "message": "Audit chain verification FAILED",
                "failed_hash": failed_hash,
                "failed_index": failed_index,
            },
        )

    return {
        "status": "success",
        "message": "Audit chain verified successfully",
        "timestamp": datetime.utcnow().isoformat(),
        "total_verified": len(logger.storage.get_all()),
    }
