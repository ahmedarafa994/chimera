import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.errors import get_cors_headers
from app.core.unified_errors import ChimeraError

logger = logging.getLogger(__name__)


async def chimera_exception_handler(request: Request, exc: ChimeraError):
    """
    Handle unified ChimeraError exceptions.
    Returns a standardized error response structure.
    """
    logger.error(f"ChimeraError: {exc.message} (Code: {exc.error_code}, Status: {exc.status_code})")

    response_content = exc.to_dict()

    return JSONResponse(
        status_code=exc.status_code, content=response_content, headers=get_cors_headers(request)
    )
