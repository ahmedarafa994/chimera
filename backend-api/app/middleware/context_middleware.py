"""Context Propagation Middleware.

FastAPI middleware that creates and injects request context at the API boundary.
The context flows through all layers automatically via contextvars.

This middleware:
1. Extracts user_id and session_id from request
2. Fetches current provider/model selection from GlobalModelSelectionState
3. Creates immutable RequestContext
4. Runs the request within that context
5. Cleans up after request completes

Usage:
    from app.middleware.context_middleware import ContextPropagationMiddleware

    app.add_middleware(ContextPropagationMiddleware)
"""

import logging
import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.request_context import ContextManager, RequestContext
from app.services.global_model_selection_state import get_global_model_selection_state

logger = logging.getLogger(__name__)


class ContextPropagationMiddleware(BaseHTTPMiddleware):
    """Middleware that creates and propagates request context.

    For every incoming request:
    - Creates a unique request_id
    - Extracts user_id from authentication
    - Fetches current provider/model selection
    - Creates RequestContext and runs request within it
    For every incoming request:
    - Creates a unique request_id
    - Extracts user_id from authentication
    - Fetches current provider/model selection
    - Creates RequestContext and runs request within it
    """

    def __init__(self, app, excluded_paths: list[str] | None = None) -> None:
        super().__init__(app)
        self.excluded_paths = excluded_paths or []
        self.excluded_paths = [path.rstrip("/") for path in self.excluded_paths]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with context propagation.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware/endpoint handler

        Returns:
            Response from handler

        """
        start_time = time.time()
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Check for excluded paths
        path = request.url.path.rstrip("/")
        for excluded_path in self.excluded_paths:
            if path == excluded_path or path.startswith(excluded_path + "/"):
                return await call_next(request)

        try:
            # Extract user information
            user_id = self._extract_user_id(request)
            session_id = self._extract_session_id(request)

            # Get provider/model selection
            provider, model = await self._get_provider_model_selection(session_id, user_id)

            # Create request context
            context = RequestContext(
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                provider=provider,
                model=model,
                metadata={
                    "ip": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", "unknown"),
                    "path": request.url.path,
                    "method": request.method,
                },
            )

            # Log context creation
            logger.debug(
                f"Created request context: request_id={request_id}, "
                f"user_id={user_id}, provider={provider}, model={model}",
            )

            # Run request within context
            async def run_request():
                return await call_next(request)

            response = await ContextManager.run_with_context(context, run_request)

            # Add context headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Provider"] = provider
            response.headers["X-Model"] = model

            # Log request completion
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Request completed: request_id={request_id}, "
                f"user_id={user_id}, provider={provider}, model={model}, "
                f"status={response.status_code}, duration_ms={duration_ms:.2f}",
            )

            return response

        except Exception as e:
            logger.error(
                f"Context middleware error: request_id={request_id}, error={e!s}",
                exc_info=True,
            )

            # Even on error, try to add request ID to response
            response = Response(content=f"Internal server error: {e!s}", status_code=500)
            response.headers["X-Request-ID"] = request_id
            return response

    def _extract_user_id(self, request: Request) -> str:
        """Extract user ID from request.

        Tries multiple sources:
        1. request.state.user (set by auth middleware)
        2. request.headers["X-User-ID"]
        3. request.cookies["user_id"]
        4. Generates anonymous ID

        Args:
            request: FastAPI request

        Returns:
            User ID string

        """
        # Try request state (set by auth middleware)
        if hasattr(request.state, "user"):
            user = request.state.user
            if hasattr(user, "id"):
                return str(user.id)
            if isinstance(user, dict) and "id" in user:
                return str(user["id"])

        # Try header
        if "X-User-ID" in request.headers:
            return request.headers["X-User-ID"]

        # Try cookie
        if "user_id" in request.cookies:
            return request.cookies["user_id"]

        # Generate anonymous user ID (should be replaced by proper auth)
        anonymous_id = f"anonymous-{uuid.uuid4().hex[:8]}"
        logger.warning(f"No user ID found, using anonymous: {anonymous_id}")
        return anonymous_id

    def _extract_session_id(self, request: Request) -> str:
        """Extract session ID from request.

        Args:
            request: FastAPI request

        Returns:
            Session ID string

        """
        # Try header
        if "X-Session-ID" in request.headers:
            return request.headers["X-Session-ID"]

        # Try cookie
        if "session_id" in request.cookies:
            return request.cookies["session_id"]

        # Try request state
        if hasattr(request.state, "session_id"):
            return request.state.session_id

        # Generate new session ID
        return str(uuid.uuid4())

    async def _get_provider_model_selection(self, session_id: str, user_id: str) -> tuple[str, str]:
        """Get the current provider/model selection for a user/session.

        Uses the GlobalModelSelectionState service to resolve selection
        using the three-tier hierarchy: request -> session -> global.

        Args:
            session_id: Session identifier
            user_id: User identifier

        Returns:
            Tuple of (provider, model)

        """
        try:
            state = get_global_model_selection_state()
            selection = await state.get_current_selection(
                session_id=session_id,
                user_id=user_id,
                db_session=None,  # No DB session in middleware
            )
            return selection.provider, selection.model_id
        except Exception as e:
            logger.exception(
                f"Failed to get selection for session {session_id}, user {user_id}: {e}",
            )
            # Return default
            return "openai", "gpt-4o"
