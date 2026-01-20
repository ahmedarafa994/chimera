"""Scheduled Testing & Monitoring Endpoints.

Phase 3 enterprise feature for automation:
- Recurring adversarial test scheduling
- Alert system for behavior changes
- Defense regression monitoring
- Compliance documentation support
"""

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.database import get_db
from app.core.observability import get_logger
from app.db.models import User

logger = get_logger("chimera.api.scheduled_testing")
router = APIRouter()


# Scheduling Models
class ScheduleFrequency(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM_CRON = "custom_cron"


class AlertType(str, Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"


class MonitoringMetric(str, Enum):
    SUCCESS_RATE = "success_rate"
    FAILURE_COUNT = "failure_count"
    RESPONSE_TIME = "response_time"
    NEW_VULNERABILITIES = "new_vulnerabilities"
    REGRESSION_DETECTION = "regression_detection"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ScheduleStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    ERROR = "error"


class AlertRule(BaseModel):
    """Alert rule configuration."""

    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    metric: MonitoringMetric

    # Threshold configuration
    threshold_value: float
    comparison_operator: str = Field(..., pattern="^(>|<|>=|<=|==|!=)$")

    # Alert settings
    alert_type: AlertType
    alert_target: str  # email address, webhook URL, etc.
    severity: AlertSeverity = AlertSeverity.WARNING

    # Timing
    cooldown_minutes: int = Field(default=60, ge=5, le=1440)  # 5 minutes to 24 hours
    enabled: bool = True


class ScheduledTest(BaseModel):
    """Scheduled test configuration."""

    schedule_id: str
    name: str
    description: str

    # Test configuration (reuses CI/CD config)
    test_config: dict[str, Any]  # Same as CICDTestConfig from cicd.py

    # Schedule configuration
    frequency: ScheduleFrequency
    cron_expression: str | None = None  # For custom_cron frequency
    timezone: str = Field(default="UTC")

    # Execution settings
    max_execution_time: int = Field(default=3600, ge=300, le=7200)  # 5 minutes to 2 hours
    retry_on_failure: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=5)

    # Monitoring and alerts
    alert_rules: list[AlertRule] = Field(default_factory=list)
    enable_regression_detection: bool = Field(default=True)
    baseline_window_days: int = Field(default=7, ge=1, le=30)

    # Metadata
    created_by: str
    workspace_id: str | None = None
    created_at: datetime
    updated_at: datetime

    # Status
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    next_execution: datetime | None = None
    last_execution: datetime | None = None
    execution_count: int = 0
    failure_count: int = 0

    @validator("cron_expression")
    def validate_cron_expression(cls, v, values):
        if values.get("frequency") == ScheduleFrequency.CUSTOM_CRON and not v:
            msg = "cron_expression is required when frequency is custom_cron"
            raise ValueError(msg)
        return v


class ScheduleExecution(BaseModel):
    """Individual schedule execution record."""

    execution_id: str
    schedule_id: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    # Status and results
    status: str  # "success", "failed", "error", "timeout"
    test_execution_id: str | None = None  # Link to CI/CD execution

    # Metrics captured
    success_rate: float = 0.0
    total_tests: int = 0
    failed_tests: int = 0
    new_vulnerabilities: int = 0

    # Alerts triggered
    alerts_triggered: list[dict[str, Any]] = Field(default_factory=list)

    # Error details
    error_message: str | None = None
    retry_count: int = 0


class AlertEvent(BaseModel):
    """Alert event record."""

    alert_id: str
    schedule_id: str
    execution_id: str | None = None
    rule_id: str

    # Alert details
    metric: MonitoringMetric
    threshold_value: float
    actual_value: float
    severity: AlertSeverity

    # Message
    title: str
    message: str

    # Timing
    triggered_at: datetime
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None

    # Delivery
    alert_type: AlertType
    delivery_status: str = "pending"  # "pending", "sent", "failed"
    delivery_attempts: int = 0


class ScheduleCreate(BaseModel):
    """Request to create scheduled test."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=1000)
    test_config: dict[str, Any]
    frequency: ScheduleFrequency
    cron_expression: str | None = None
    timezone: str = Field(default="UTC")
    workspace_id: str | None = None
    alert_rules: list[AlertRule] = Field(default_factory=list)


class ScheduleUpdate(BaseModel):
    """Request to update scheduled test."""

    name: str | None = None
    description: str | None = None
    test_config: dict[str, Any] | None = None
    frequency: ScheduleFrequency | None = None
    cron_expression: str | None = None
    timezone: str | None = None
    status: ScheduleStatus | None = None
    alert_rules: list[AlertRule] | None = None


class ScheduleListResponse(BaseModel):
    """Response for schedule listing."""

    schedules: list[ScheduledTest]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class ExecutionListResponse(BaseModel):
    """Response for execution listing."""

    executions: list[ScheduleExecution]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class AlertListResponse(BaseModel):
    """Response for alert listing."""

    alerts: list[AlertEvent]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class MonitoringDashboard(BaseModel):
    """Monitoring dashboard data."""

    total_schedules: int
    active_schedules: int
    recent_executions: int
    success_rate: float

    # Recent activity
    recent_executions_list: list[ScheduleExecution]
    recent_alerts: list[AlertEvent]

    # Trends
    success_rate_trend: list[dict[str, Any]]
    execution_count_trend: list[dict[str, Any]]

    # Health status
    unhealthy_schedules: list[ScheduledTest]
    pending_alerts: int


# In-memory storage for demo (in production, would use database)
scheduled_tests: dict[str, ScheduledTest] = {}
schedule_executions: dict[str, list[ScheduleExecution]] = {}
alert_events: list[AlertEvent] = []


def calculate_next_execution(
    frequency: ScheduleFrequency,
    cron_expr: str | None = None,
) -> datetime:
    """Calculate next execution time based on frequency."""
    now = datetime.utcnow()

    if frequency == ScheduleFrequency.HOURLY:
        return now + timedelta(hours=1)
    if frequency == ScheduleFrequency.DAILY:
        return now + timedelta(days=1)
    if frequency == ScheduleFrequency.WEEKLY:
        return now + timedelta(weeks=1)
    if frequency == ScheduleFrequency.MONTHLY:
        return now + timedelta(days=30)  # Simplified monthly calculation
    if frequency == ScheduleFrequency.CUSTOM_CRON and cron_expr:
        # Simplified cron parsing - in production would use proper cron library
        return now + timedelta(hours=1)  # Fallback
    return now + timedelta(hours=1)  # Default


@router.post("/schedules", response_model=ScheduledTest, status_code=status.HTTP_201_CREATED)
async def create_scheduled_test(
    schedule_data: ScheduleCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Create a new scheduled test."""
    try:
        # Generate schedule ID
        schedule_id = (
            f"schedule_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        )

        # Calculate next execution
        next_execution = calculate_next_execution(
            schedule_data.frequency,
            schedule_data.cron_expression,
        )

        # Create scheduled test
        scheduled_test = ScheduledTest(
            schedule_id=schedule_id,
            name=schedule_data.name,
            description=schedule_data.description,
            test_config=schedule_data.test_config,
            frequency=schedule_data.frequency,
            cron_expression=schedule_data.cron_expression,
            timezone=schedule_data.timezone,
            alert_rules=schedule_data.alert_rules,
            created_by=current_user.id,
            workspace_id=schedule_data.workspace_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            next_execution=next_execution,
        )

        # Store scheduled test
        scheduled_tests[schedule_id] = scheduled_test

        logger.info(f"Created scheduled test {schedule_id} for user {current_user.id}")

        return scheduled_test

    except Exception as e:
        logger.exception(f"Failed to create scheduled test: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create scheduled test",
        )


@router.get("/schedules", response_model=ScheduleListResponse)
async def list_scheduled_tests(
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    workspace_id: Annotated[str | None, Query(description="Filter by workspace")] = None,
    status_filter: Annotated[ScheduleStatus | None, Query(description="Filter by status")] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List scheduled tests."""
    try:
        # Get accessible schedules
        accessible_schedules = []

        for schedule in scheduled_tests.values():
            # Check access permissions
            has_access = schedule.created_by == current_user.id or (
                schedule.workspace_id and schedule.workspace_id == workspace_id
            )  # TODO: Check workspace membership

            if has_access:
                accessible_schedules.append(schedule)

        # Apply filters
        filtered_schedules = accessible_schedules

        if workspace_id:
            filtered_schedules = [s for s in filtered_schedules if s.workspace_id == workspace_id]

        if status_filter:
            filtered_schedules = [s for s in filtered_schedules if s.status == status_filter]

        # Sort by updated_at descending
        filtered_schedules.sort(key=lambda x: x.updated_at, reverse=True)

        # Apply pagination
        total = len(filtered_schedules)
        offset = (page - 1) * page_size
        schedules_page = filtered_schedules[offset : offset + page_size]

        return ScheduleListResponse(
            schedules=schedules_page,
            total=total,
            page=page,
            page_size=page_size,
            has_next=offset + page_size < total,
            has_prev=page > 1,
        )

    except Exception as e:
        logger.exception(f"Failed to list scheduled tests: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve scheduled tests",
        )


@router.get("/schedules/{schedule_id}", response_model=ScheduledTest)
async def get_scheduled_test(
    schedule_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Get scheduled test details."""
    try:
        schedule = scheduled_tests.get(schedule_id)

        if not schedule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scheduled test not found",
            )

        # Check access permissions
        has_access = schedule.created_by == current_user.id or (
            schedule.workspace_id
        )  # TODO: Check workspace membership

        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this scheduled test",
            )

        return schedule

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get scheduled test {schedule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve scheduled test",
        )


@router.patch("/schedules/{schedule_id}", response_model=ScheduledTest)
async def update_scheduled_test(
    schedule_id: str,
    update_data: ScheduleUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Update scheduled test (creator only)."""
    try:
        schedule = scheduled_tests.get(schedule_id)

        if not schedule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scheduled test not found",
            )

        if schedule.created_by != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only schedule creator can update",
            )

        # Update fields
        if update_data.name is not None:
            schedule.name = update_data.name
        if update_data.description is not None:
            schedule.description = update_data.description
        if update_data.test_config is not None:
            schedule.test_config = update_data.test_config
        if update_data.frequency is not None:
            schedule.frequency = update_data.frequency
            # Recalculate next execution
            schedule.next_execution = calculate_next_execution(
                schedule.frequency,
                schedule.cron_expression,
            )
        if update_data.cron_expression is not None:
            schedule.cron_expression = update_data.cron_expression
        if update_data.timezone is not None:
            schedule.timezone = update_data.timezone
        if update_data.status is not None:
            schedule.status = update_data.status
        if update_data.alert_rules is not None:
            schedule.alert_rules = update_data.alert_rules

        schedule.updated_at = datetime.utcnow()

        logger.info(f"Updated scheduled test {schedule_id}")

        return schedule

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update scheduled test {schedule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update scheduled test",
        )


@router.post("/schedules/{schedule_id}/execute")
async def trigger_manual_execution(
    schedule_id: str,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Trigger manual execution of scheduled test."""
    try:
        schedule = scheduled_tests.get(schedule_id)

        if not schedule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scheduled test not found",
            )

        # Check access permissions
        has_access = schedule.created_by == current_user.id or (
            schedule.workspace_id
        )  # TODO: Check workspace membership

        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this scheduled test",
            )

        # Create execution record
        execution_id = f"exec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        execution = ScheduleExecution(
            execution_id=execution_id,
            schedule_id=schedule_id,
            started_at=datetime.utcnow(),
            status="running",
        )

        # Store execution
        if schedule_id not in schedule_executions:
            schedule_executions[schedule_id] = []
        schedule_executions[schedule_id].append(execution)

        # Start background execution
        background_tasks.add_task(execute_scheduled_test_background, schedule_id, execution_id)

        logger.info(f"Triggered manual execution {execution_id} for schedule {schedule_id}")

        return {"execution_id": execution_id, "message": "Test execution started"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to trigger execution for schedule {schedule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to trigger test execution",
        )


@router.get("/schedules/{schedule_id}/executions", response_model=ExecutionListResponse)
async def list_schedule_executions(
    schedule_id: str,
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List executions for a scheduled test."""
    try:
        schedule = scheduled_tests.get(schedule_id)

        if not schedule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scheduled test not found",
            )

        # Check access permissions
        has_access = schedule.created_by == current_user.id or (
            schedule.workspace_id
        )  # TODO: Check workspace membership

        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this scheduled test",
            )

        # Get executions
        executions = schedule_executions.get(schedule_id, [])

        # Sort by started_at descending
        executions.sort(key=lambda x: x.started_at, reverse=True)

        # Apply pagination
        total = len(executions)
        offset = (page - 1) * page_size
        executions_page = executions[offset : offset + page_size]

        return ExecutionListResponse(
            executions=executions_page,
            total=total,
            page=page,
            page_size=page_size,
            has_next=offset + page_size < total,
            has_prev=page > 1,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to list executions for schedule {schedule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve executions",
        )


@router.get("/alerts", response_model=AlertListResponse)
async def list_alerts(
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    severity: Annotated[AlertSeverity | None, Query(description="Filter by severity")] = None,
    unresolved_only: Annotated[bool, Query(description="Show only unresolved alerts")] = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List alert events."""
    try:
        # Get accessible alerts (based on schedule access)
        accessible_alerts = []

        for alert in alert_events:
            schedule = scheduled_tests.get(alert.schedule_id)
            if schedule and (schedule.created_by == current_user.id or schedule.workspace_id):
                accessible_alerts.append(alert)

        # Apply filters
        filtered_alerts = accessible_alerts

        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]

        if unresolved_only:
            filtered_alerts = [a for a in filtered_alerts if not a.resolved_at]

        # Sort by triggered_at descending
        filtered_alerts.sort(key=lambda x: x.triggered_at, reverse=True)

        # Apply pagination
        total = len(filtered_alerts)
        offset = (page - 1) * page_size
        alerts_page = filtered_alerts[offset : offset + page_size]

        return AlertListResponse(
            alerts=alerts_page,
            total=total,
            page=page,
            page_size=page_size,
            has_next=offset + page_size < total,
            has_prev=page > 1,
        )

    except Exception as e:
        logger.exception(f"Failed to list alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve alerts",
        )


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Acknowledge an alert."""
    try:
        # Find alert
        alert = next((a for a in alert_events if a.alert_id == alert_id), None)

        if not alert:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

        # Check access permissions
        schedule = scheduled_tests.get(alert.schedule_id)
        if not schedule or (schedule.created_by != current_user.id and not schedule.workspace_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this alert",
            )

        # Acknowledge alert
        alert.acknowledged_at = datetime.utcnow()

        logger.info(f"Acknowledged alert {alert_id}")

        return {"message": "Alert acknowledged successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to acknowledge alert {alert_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acknowledge alert",
        )


@router.get("/dashboard", response_model=MonitoringDashboard)
async def get_monitoring_dashboard(
    workspace_id: Annotated[str | None, Query(description="Filter by workspace")] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get monitoring dashboard data."""
    try:
        # Get accessible schedules
        accessible_schedules = []
        for schedule in scheduled_tests.values():
            has_access = schedule.created_by == current_user.id or (
                schedule.workspace_id and schedule.workspace_id == workspace_id
            )
            if has_access:
                accessible_schedules.append(schedule)

        # Calculate metrics
        total_schedules = len(accessible_schedules)
        active_schedules = sum(1 for s in accessible_schedules if s.status == ScheduleStatus.ACTIVE)

        # Recent executions (last 24 hours)
        recent_executions_list = []
        recent_execution_count = 0

        for schedule in accessible_schedules:
            executions = schedule_executions.get(schedule.schedule_id, [])
            recent = [
                e for e in executions if e.started_at >= datetime.utcnow() - timedelta(days=1)
            ]
            recent_executions_list.extend(recent)
            recent_execution_count += len(recent)

        # Sort recent executions by started_at descending
        recent_executions_list.sort(key=lambda x: x.started_at, reverse=True)
        recent_executions_list = recent_executions_list[:10]  # Limit to 10 most recent

        # Calculate success rate
        if recent_execution_count > 0:
            successful_executions = sum(1 for e in recent_executions_list if e.status == "success")
            success_rate = (successful_executions / recent_execution_count) * 100
        else:
            success_rate = 100.0

        # Recent alerts (last 24 hours)
        recent_alerts = [
            a
            for a in alert_events
            if a.triggered_at >= datetime.utcnow() - timedelta(days=1)
            and any(s.schedule_id == a.schedule_id for s in accessible_schedules)
        ]
        recent_alerts.sort(key=lambda x: x.triggered_at, reverse=True)
        recent_alerts = recent_alerts[:5]  # Limit to 5 most recent

        # Trends (simplified - last 7 days)
        success_rate_trend = []
        execution_count_trend = []

        for i in range(7):
            date = datetime.utcnow() - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")

            # Calculate metrics for this date
            day_executions = [
                e for e in recent_executions_list if e.started_at.date() == date.date()
            ]

            day_success_rate = 100.0
            if day_executions:
                successful = sum(1 for e in day_executions if e.status == "success")
                day_success_rate = (successful / len(day_executions)) * 100

            success_rate_trend.append({"date": date_str, "value": day_success_rate})
            execution_count_trend.append({"date": date_str, "value": len(day_executions)})

        # Unhealthy schedules
        unhealthy_schedules = [
            s
            for s in accessible_schedules
            if s.status != ScheduleStatus.ACTIVE or s.failure_count > 3
        ]

        # Pending alerts
        pending_alerts = sum(1 for a in recent_alerts if not a.acknowledged_at)

        return MonitoringDashboard(
            total_schedules=total_schedules,
            active_schedules=active_schedules,
            recent_executions=recent_execution_count,
            success_rate=success_rate,
            recent_executions_list=recent_executions_list,
            recent_alerts=recent_alerts,
            success_rate_trend=success_rate_trend,
            execution_count_trend=execution_count_trend,
            unhealthy_schedules=unhealthy_schedules,
            pending_alerts=pending_alerts,
        )

    except Exception as e:
        logger.exception(f"Failed to get monitoring dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard data",
        )


@router.delete("/schedules/{schedule_id}")
async def delete_scheduled_test(
    schedule_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Delete scheduled test (creator only)."""
    try:
        schedule = scheduled_tests.get(schedule_id)

        if not schedule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scheduled test not found",
            )

        if schedule.created_by != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only schedule creator can delete",
            )

        # Delete schedule and related data
        del scheduled_tests[schedule_id]
        schedule_executions.pop(schedule_id, None)

        # Remove related alerts
        global alert_events
        alert_events = [a for a in alert_events if a.schedule_id != schedule_id]

        logger.info(f"Deleted scheduled test {schedule_id}")

        return {"message": f"Scheduled test '{schedule.name}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete scheduled test {schedule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete scheduled test",
        )


# Background task functions
async def execute_scheduled_test_background(schedule_id: str, execution_id: str) -> None:
    """Execute scheduled test in background."""
    try:
        schedule = scheduled_tests.get(schedule_id)
        executions = schedule_executions.get(schedule_id, [])
        execution = next((e for e in executions if e.execution_id == execution_id), None)

        if not schedule or not execution:
            logger.error(f"Schedule or execution not found: {schedule_id}/{execution_id}")
            return

        # Simulate test execution (in production, would call actual CI/CD testing)
        import time

        time.sleep(2)  # Simulate execution time

        # Simulate results
        success_rate = 85 + (hash(execution_id) % 15)  # 85-99% success rate
        total_tests = 10 + (hash(schedule_id) % 10)  # 10-19 total tests
        failed_tests = max(0, total_tests - int(total_tests * success_rate / 100))

        # Update execution
        execution.completed_at = datetime.utcnow()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        execution.status = "success" if success_rate > 90 else "failed"
        execution.success_rate = success_rate
        execution.total_tests = total_tests
        execution.failed_tests = failed_tests
        execution.new_vulnerabilities = failed_tests  # Simplified

        # Update schedule statistics
        schedule.last_execution = execution.completed_at
        schedule.execution_count += 1
        if execution.status == "failed":
            schedule.failure_count += 1

        # Calculate next execution
        schedule.next_execution = calculate_next_execution(
            schedule.frequency,
            schedule.cron_expression,
        )

        # Check alert rules
        await check_alert_rules(schedule, execution)

        logger.info(f"Completed scheduled execution {execution_id} - Status: {execution.status}")

    except Exception as e:
        logger.exception(f"Failed to execute scheduled test {schedule_id}/{execution_id}: {e}")
        if execution:
            execution.status = "error"
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()


async def check_alert_rules(schedule: ScheduledTest, execution: ScheduleExecution) -> None:
    """Check alert rules and trigger alerts if needed."""
    try:
        for rule in schedule.alert_rules:
            if not rule.enabled:
                continue

            # Get metric value
            metric_value = 0.0
            if rule.metric == MonitoringMetric.SUCCESS_RATE:
                metric_value = execution.success_rate
            elif rule.metric == MonitoringMetric.FAILURE_COUNT:
                metric_value = execution.failed_tests
            elif rule.metric == MonitoringMetric.NEW_VULNERABILITIES:
                metric_value = execution.new_vulnerabilities
            # Add other metrics as needed

            # Check threshold
            should_alert = False
            if rule.comparison_operator == ">":
                should_alert = metric_value > rule.threshold_value
            elif rule.comparison_operator == "<":
                should_alert = metric_value < rule.threshold_value
            elif rule.comparison_operator == ">=":
                should_alert = metric_value >= rule.threshold_value
            elif rule.comparison_operator == "<=":
                should_alert = metric_value <= rule.threshold_value
            elif rule.comparison_operator == "==":
                should_alert = metric_value == rule.threshold_value
            elif rule.comparison_operator == "!=":
                should_alert = metric_value != rule.threshold_value

            if should_alert:
                # Create alert
                alert = AlertEvent(
                    alert_id=str(uuid.uuid4()),
                    schedule_id=schedule.schedule_id,
                    execution_id=execution.execution_id,
                    rule_id=rule.rule_id,
                    metric=rule.metric,
                    threshold_value=rule.threshold_value,
                    actual_value=metric_value,
                    severity=rule.severity,
                    title=f"Alert: {rule.name}",
                    message=f"{rule.description} - {rule.metric.value}: {metric_value} {rule.comparison_operator} {rule.threshold_value}",
                    triggered_at=datetime.utcnow(),
                    alert_type=rule.alert_type,
                )

                # Store alert
                alert_events.append(alert)

                # Add to execution alerts
                execution.alerts_triggered.append(
                    {
                        "rule_name": rule.name,
                        "metric": rule.metric.value,
                        "threshold": rule.threshold_value,
                        "actual": metric_value,
                        "severity": rule.severity.value,
                    },
                )

                logger.warning(
                    f"Alert triggered for schedule {schedule.schedule_id}: {alert.title}",
                )

    except Exception as e:
        logger.exception(f"Failed to check alert rules for schedule {schedule.schedule_id}: {e}")
