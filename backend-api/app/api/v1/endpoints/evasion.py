"""Evasion Task API Endpoints.

Provides endpoints for:
- Creating metamorphic evasion tasks
- Checking task status
- Retrieving task results
- Cancelling running tasks
"""

import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import SyncSessionFactory
from app.crud import evasion_crud, llm_crud
from app.schemas.api_schemas import (
    EvasionAttemptResult,
    EvasionTaskConfig,
    EvasionTaskResult,
    EvasionTaskStatusEnum,
    EvasionTaskStatusResponse,
    MetamorphosisStrategyConfig,
)
from app.tasks.evasion_tasks import run_evasion_task  # Import the Celery task

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================


class CancelTaskResponse(BaseModel):
    """Response for task cancellation."""

    success: bool
    message: str
    task_id: str


# =============================================================================
# Database Dependency
# =============================================================================


def get_sync_db():
    db = SyncSessionFactory()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/evasion/generate",
    response_model=EvasionTaskStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create evasion task",
    description="Initiates a new asynchronous metamorphic evasion task.",
)
async def generate_evasion_task(
    evasion_config: EvasionTaskConfig,
    db: Annotated[Session, Depends(get_sync_db)],
):
    """Initiates a new asynchronous metamorphic evasion task.

    The task will be processed in the background using Celery.
    Use the returned task_id to check status and retrieve results.
    """
    # Validate target model exists
    db_llm_model = llm_crud.get_llm_model(db, evasion_config.target_model_id)
    if not db_llm_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target LLM model not found.",
        )

    task_id = str(uuid.uuid4())
    logger.info(f"Received request to generate evasion task. Task ID: {task_id}")

    # Create the task entry in the database
    evasion_crud.create_evasion_task(db, task_id, evasion_config)

    # Dispatch the task to Celery
    # Celery tasks cannot directly take Pydantic models as arguments without serialization.
    # We pass the dictionary representation.
    run_evasion_task.delay(task_id, evasion_config.model_dump())

    return EvasionTaskStatusResponse(
        task_id=task_id,
        status=EvasionTaskStatusEnum.PENDING,
        message="Evasion task has been submitted and is pending execution.",
    )


@router.get(
    "/evasion/status/{task_id}",
    response_model=EvasionTaskStatusResponse,
    summary="Get task status",
    description="Retrieves the current status of an asynchronous evasion task.",
)
def get_evasion_task_status(task_id: str, db: Annotated[Session, Depends(get_sync_db)]):
    """Retrieves the current status of an asynchronous evasion task.

    Returns the task status, progress percentage, and current step information.
    """
    db_task = evasion_crud.get_evasion_task(db, task_id)
    if not db_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evasion task not found.")

    # Get Celery task state for more granular progress if needed
    celery_result = run_evasion_task.AsyncResult(task_id)
    celery_status = celery_result.status
    celery_info = celery_result.info

    message = f"Task {celery_status}"
    progress = None
    if isinstance(celery_info, dict) and "current" in celery_info and "total" in celery_info:
        current = celery_info["current"]
        total = celery_info["total"]
        progress = (current / total) * 100 if total else 0
        message = f"{celery_info.get('status', 'Processing')} ({current}/{total})"

    # Override with DB status if more definitive (e.g., COMPLETED/FAILED)
    status_enum = EvasionTaskStatusEnum(db_task.status)
    if status_enum in [
        EvasionTaskStatusEnum.COMPLETED,
        EvasionTaskStatusEnum.FAILED,
        EvasionTaskStatusEnum.CANCELLED,
    ]:
        # If DB says completed/failed/cancelled, then Celery status doesn't matter as much here.
        pass
    else:
        # Otherwise, use Celery's dynamic status
        try:
            # Ensure status is a valid enum value, default to PENDING/RUNNING logic if unknown
            status_enum = EvasionTaskStatusEnum(celery_status)
        except ValueError:
            # Fallback if celery status is something standard like 'SUCCESS' which maps to COMPLETED
            if celery_status == "SUCCESS":
                status_enum = EvasionTaskStatusEnum.COMPLETED
            elif celery_status == "FAILURE":
                status_enum = EvasionTaskStatusEnum.FAILED
            elif celery_status == "REVOKED":
                status_enum = EvasionTaskStatusEnum.CANCELLED
            else:
                status_enum = EvasionTaskStatusEnum.RUNNING

    return EvasionTaskStatusResponse(
        task_id=task_id,
        status=status_enum,
        current_step=message,
        progress=progress,
        message=(
            db_task.failed_reason
            if db_task.status == EvasionTaskStatusEnum.FAILED.value
            else message
        ),
    )


@router.get(
    "/evasion/results/{task_id}",
    response_model=EvasionTaskResult,
    summary="Get task results",
    description="Retrieves the comprehensive results of a completed evasion task.",
)
def get_evasion_task_results(task_id: str, db: Annotated[Session, Depends(get_sync_db)]):
    """Retrieves the comprehensive results of a completed evasion task.

    Only available for tasks that have completed (successfully or with failure).
    Check /evasion/status/{task_id} first to ensure the task is complete.
    """
    db_task = evasion_crud.get_evasion_task(db, task_id)
    if not db_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evasion task not found.")

    if db_task.status not in [
        EvasionTaskStatusEnum.COMPLETED.value,
        EvasionTaskStatusEnum.FAILED.value,
    ]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task is not yet completed or has failed irrevocably. "
            "Check /evasion/status/{task_id} first.",
        )

    return EvasionTaskResult(
        task_id=db_task.task_id,
        status=EvasionTaskStatusEnum(db_task.status),
        initial_prompt=db_task.initial_prompt,
        target_model_id=db_task.target_model_id,
        strategy_chain=[MetamorphosisStrategyConfig(**s) for s in db_task.strategy_chain],
        success_criteria=db_task.success_criteria,
        final_status=db_task.final_status,
        results=[EvasionAttemptResult(**r) for r in db_task.results],
        overall_success=db_task.overall_success,
        completed_at=db_task.completed_at.isoformat() if db_task.completed_at else None,
        failed_reason=db_task.failed_reason,
    )


@router.post(
    "/evasion/cancel/{task_id}",
    response_model=CancelTaskResponse,
    summary="Cancel evasion task",
    description="Attempts to cancel a running evasion task.",
)
def cancel_evasion_task(task_id: str, db: Annotated[Session, Depends(get_sync_db)]):
    """Attempts to cancel a running evasion task.

    This will:
    1. Revoke the Celery task if it's still pending/running
    2. Update the task status in the database to CANCELLED

    Note: If the task has already completed or failed, this will return
    success=False with an appropriate message.
    """
    db_task = evasion_crud.get_evasion_task(db, task_id)
    if not db_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evasion task not found.")

    # Check if task is already in a terminal state
    if db_task.status in [
        EvasionTaskStatusEnum.COMPLETED.value,
        EvasionTaskStatusEnum.FAILED.value,
        EvasionTaskStatusEnum.CANCELLED.value,
    ]:
        return CancelTaskResponse(
            success=False,
            message=f"Task is already in terminal state: {db_task.status}",
            task_id=task_id,
        )

    try:
        # Attempt to revoke the Celery task
        celery_result = run_evasion_task.AsyncResult(task_id)
        celery_result.revoke(terminate=True)

        # Update the database status
        evasion_crud.update_evasion_task_status(db, task_id, EvasionTaskStatusEnum.CANCELLED.value)

        logger.info(f"Successfully cancelled evasion task: {task_id}")

        return CancelTaskResponse(
            success=True,
            message="Task cancellation requested successfully.",
            task_id=task_id,
        )

    except Exception as e:
        logger.exception(f"Failed to cancel evasion task {task_id}: {e}")

        # Still try to update the database status
        try:
            evasion_crud.update_evasion_task_status(
                db,
                task_id,
                EvasionTaskStatusEnum.CANCELLED.value,
            )
        except Exception as db_error:
            logger.exception(f"Failed to update task status in database: {db_error}")

        return CancelTaskResponse(
            success=False,
            message=f"Task cancellation attempted but may not have succeeded: {e!s}",
            task_id=task_id,
        )


@router.get(
    "/evasion/tasks",
    response_model=dict[str, Any],
    summary="List evasion tasks",
    description="Lists all evasion tasks with optional filtering.",
)
def list_evasion_tasks(
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_sync_db),
):
    """Lists all evasion tasks with optional filtering.

    Parameters
    ----------
    - status_filter: Filter by task status (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
    - limit: Maximum number of tasks to return (default: 50)
    - offset: Number of tasks to skip (default: 0)

    """
    try:
        tasks = evasion_crud.list_evasion_tasks(
            db,
            status_filter=status_filter,
            limit=limit,
            offset=offset,
        )

        return {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "status": task.status,
                    "initial_prompt": (
                        task.initial_prompt[:100] + "..."
                        if len(task.initial_prompt) > 100
                        else task.initial_prompt
                    ),
                    "target_model_id": task.target_model_id,
                    "created_at": (task.created_at.isoformat() if task.created_at else None),
                    "completed_at": (task.completed_at.isoformat() if task.completed_at else None),
                    "overall_success": task.overall_success,
                }
                for task in tasks
            ],
            "total": len(tasks),
            "limit": limit,
            "offset": offset,
        }
    except AttributeError:
        # If list_evasion_tasks doesn't exist in crud, return empty list
        logger.warning("list_evasion_tasks not implemented in evasion_crud")
        return {
            "tasks": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "message": "Task listing not yet implemented",
        }
