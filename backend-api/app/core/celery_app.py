import os

from celery import Celery

from app.core.config import settings

# Ensure Redis URL is set, otherwise default to localhost
redis_url = getattr(
    settings,
    "CELERY_BROKER_URL",
    os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
)
result_backend = getattr(
    settings,
    "CELERY_RESULT_BACKEND",
    os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
)

celery_app = Celery("chimera_tasks", broker=redis_url, backend=result_backend)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# Optional: Load tasks from a specific module
# celery_app.autodiscover_tasks(['app.tasks'])
