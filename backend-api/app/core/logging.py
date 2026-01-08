import logging
import sys

from app.core.config import settings


def setup_logging():
    """
    Configure logging for the application.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set lower level for third-party libraries if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger("chimera")
    logger.setLevel(log_level)
    return logger


logger = setup_logging()
