import logging
from typing import Any

from sqlalchemy.orm import Session

from app.core.celery_app import celery_app

# Use SyncSessionFactory instead of SessionLocal if that was the name in database.py
from app.core.database import SyncSessionFactory
from app.crud import evasion_crud
from app.schemas.api_schemas import EvasionAttemptResult, EvasionTaskConfig, EvasionTaskStatusEnum
from app.services.evasion_engine import EvasionEngine

logger = logging.getLogger(__name__)
# Need to patch asyncio loop for sync Celery workers running async code if needed
import asyncio


@celery_app.task(bind=True, name="run_evasion_task")
def run_evasion_task(self, task_id: str, evasion_config_dict: dict[str, Any]):
    """
    Celery task to run a complete metamorphic evasion task.
    """
    # Create a new session for this task
    db: Session = SyncSessionFactory()
    try:
        evasion_config = EvasionTaskConfig(**evasion_config_dict)
        logger.info(
            f"Celery Task {task_id}: Starting evasion task for model {evasion_config.target_model_id}"
        )

        evasion_engine = EvasionEngine(db)

        evasion_crud.update_evasion_task_status(
            db, task_id, EvasionTaskStatusEnum.RUNNING, "Task started."
        )

        successful_attempt = False
        all_results = []

        # Helper to run async method in sync task
        def run_async(coro):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)

        for attempt_num in range(1, evasion_config.max_attempts + 1):
            if successful_attempt:
                logger.info(
                    f"Celery Task {task_id}: Evasion successful in previous attempt, stopping."
                )
                break  # Stop if successful

            logger.info(
                f"Celery Task {task_id}: Executing attempt {attempt_num}/{evasion_config.max_attempts}"
            )
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": attempt_num,
                    "total": evasion_config.max_attempts,
                    "status": "Running evasion attempt",
                },
            )

            try:
                # EvasionEngine.execute_evasion_attempt is async
                attempt_result = run_async(
                    evasion_engine.execute_evasion_attempt(task_id, evasion_config, attempt_num)
                )

                evasion_crud.add_evasion_attempt_result(db, task_id, attempt_result)
                all_results.append(attempt_result)

                if attempt_result.is_evasion_successful:
                    successful_attempt = True
                    logger.info(
                        f"Celery Task {task_id}: Evasion successful on attempt {attempt_num}."
                    )
                    break  # Exit loop if evasion is successful
            except Exception as e:
                logger.error(f"Celery Task {task_id}: Attempt {attempt_num} failed with error: {e}")
                # Log this specific attempt as failed, but try next if max_attempts allows
                failed_attempt_result = EvasionAttemptResult(
                    attempt_number=attempt_num,
                    transformed_prompt="N/A",
                    llm_response=f"Error: {e}",
                    is_evasion_successful=False,
                    evaluation_details={"error": str(e)},
                    transformation_log=[],
                )
                evasion_crud.add_evasion_attempt_result(db, task_id, failed_attempt_result)
                all_results.append(failed_attempt_result)

        final_status = "SUCCESS" if successful_attempt else "FAILURE"
        overall_success = successful_attempt
        failed_reason = (
            "No successful evasion within max attempts." if not successful_attempt else None
        )

        evasion_crud.finalize_evasion_task(
            db, task_id, final_status, overall_success, failed_reason
        )
        # Final update to COMPLETED
        evasion_crud.update_evasion_task_status(db, task_id, EvasionTaskStatusEnum.COMPLETED)
        logger.info(f"Celery Task {task_id}: Task completed with final status: {final_status}")

    except Exception as e:
        logger.exception(
            f"Celery Task {task_id}: An unhandled error occurred during task execution."
        )
        evasion_crud.update_evasion_task_status(
            db, task_id, EvasionTaskStatusEnum.FAILED, message=str(e)
        )
    finally:
        db.close()
