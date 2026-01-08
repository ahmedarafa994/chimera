"""
Evasion Task CRUD Operations

Provides database operations for evasion tasks including:
- Creating tasks
- Retrieving task status and results
- Updating task status
- Listing tasks with filtering
"""

from datetime import datetime

from sqlalchemy import desc
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.db.models import DBEvasionTask
from app.schemas.api_schemas import (
    EvasionAttemptResult,
    EvasionTaskConfig,
    EvasionTaskStatusEnum,
)


def create_evasion_task(
    db: Session, task_id: str, evasion_config: EvasionTaskConfig
) -> DBEvasionTask:
    """Create a new evasion task in the database."""
    db_evasion_task = DBEvasionTask(
        task_id=task_id,
        target_model_id=evasion_config.target_model_id,
        initial_prompt=evasion_config.initial_prompt,
        strategy_chain=[s.model_dump() for s in evasion_config.strategy_chain],
        success_criteria=evasion_config.success_criteria,
        max_attempts=evasion_config.max_attempts,
        status=EvasionTaskStatusEnum.PENDING.value,
        results=[],
    )
    db.add(db_evasion_task)
    db.commit()
    db.refresh(db_evasion_task)
    return db_evasion_task


def get_evasion_task(db: Session, task_id: str) -> DBEvasionTask | None:
    """Get an evasion task by its ID."""
    return db.query(DBEvasionTask).filter(DBEvasionTask.task_id == task_id).first()


def update_evasion_task_status(
    db: Session,
    task_id: str,
    status: str | EvasionTaskStatusEnum,
    message: str | None = None,
) -> DBEvasionTask | None:
    """
    Update the status of an evasion task.

    Args:
        db: Database session
        task_id: The task ID to update
        status: New status (can be string or enum)
        message: Optional message (used for failed_reason if status is FAILED)

    Returns:
        Updated task or None if not found
    """
    db_task = get_evasion_task(db, task_id)
    if db_task:
        # Handle both string and enum status
        status_value = status.value if isinstance(status, EvasionTaskStatusEnum) else status

        db_task.status = status_value
        db_task.updated_at = func.now()

        # Set completed_at for terminal states
        terminal_states = [
            EvasionTaskStatusEnum.COMPLETED.value,
            EvasionTaskStatusEnum.FAILED.value,
            EvasionTaskStatusEnum.CANCELLED.value,
        ]
        if status_value in terminal_states:
            db_task.completed_at = datetime.utcnow()

        # Set failed_reason for failed status
        if message and status_value == EvasionTaskStatusEnum.FAILED.value:
            db_task.failed_reason = message

        db.commit()
        db.refresh(db_task)
    return db_task


def add_evasion_attempt_result(
    db: Session, task_id: str, attempt_result: EvasionAttemptResult
) -> DBEvasionTask | None:
    """Add an attempt result to an evasion task."""
    db_task = get_evasion_task(db, task_id)
    if db_task:
        # Ensure results is a list of dicts for JSON column
        current_results = list(db_task.results) if db_task.results else []
        current_results.append(attempt_result.model_dump())
        db_task.results = current_results
        db_task.overall_success = db_task.overall_success or attempt_result.is_evasion_successful
        db_task.updated_at = func.now()
        db.commit()
        db.refresh(db_task)
    return db_task


def finalize_evasion_task(
    db: Session,
    task_id: str,
    final_status: str,
    overall_success: bool,
    failed_reason: str | None = None,
) -> DBEvasionTask | None:
    """Finalize an evasion task with its final status and results."""
    db_task = get_evasion_task(db, task_id)
    if db_task:
        db_task.status = EvasionTaskStatusEnum.COMPLETED.value
        db_task.final_status = final_status
        db_task.overall_success = overall_success
        db_task.failed_reason = failed_reason
        db_task.completed_at = datetime.utcnow()
        db_task.updated_at = func.now()
        db.commit()
        db.refresh(db_task)
    return db_task


def list_evasion_tasks(
    db: Session,
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[DBEvasionTask]:
    """
    List evasion tasks with optional filtering.

    Args:
        db: Database session
        status_filter: Optional status to filter by
        limit: Maximum number of tasks to return
        offset: Number of tasks to skip

    Returns:
        List of evasion tasks
    """
    query = db.query(DBEvasionTask)

    # Apply status filter if provided
    if status_filter:
        # Validate status filter
        try:
            EvasionTaskStatusEnum(status_filter)
            query = query.filter(DBEvasionTask.status == status_filter)
        except ValueError:
            # Invalid status filter, ignore it
            pass

    # Order by created_at descending (newest first)
    query = query.order_by(desc(DBEvasionTask.created_at))

    # Apply pagination
    query = query.offset(offset).limit(limit)

    return query.all()


def count_evasion_tasks(
    db: Session,
    status_filter: str | None = None,
) -> int:
    """
    Count evasion tasks with optional filtering.

    Args:
        db: Database session
        status_filter: Optional status to filter by

    Returns:
        Count of matching tasks
    """
    query = db.query(func.count(DBEvasionTask.id))

    if status_filter:
        try:
            EvasionTaskStatusEnum(status_filter)
            query = query.filter(DBEvasionTask.status == status_filter)
        except ValueError:
            pass

    return query.scalar() or 0


def delete_evasion_task(db: Session, task_id: str) -> bool:
    """
    Delete an evasion task by ID.

    Args:
        db: Database session
        task_id: The task ID to delete

    Returns:
        True if deleted, False if not found
    """
    db_task = get_evasion_task(db, task_id)
    if db_task:
        db.delete(db_task)
        db.commit()
        return True
    return False
