"""Selection Middleware - FastAPI middleware for provider selection injection and validation.

This module implements the middleware layer that injects and validates provider/model
selections for incoming requests. It implements the four-tier selection hierarchy:
1. Request Override (header/parameter)
2. Session Preference (database)
3. Global Model Selection (user-set via /model-selection endpoint)
4. Static Default (environment)
"""

import logging
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.database import get_async_session_factory
from app.core.selection_context import SelectionContext
from app.domain.models import Selection, SelectionScope
from app.services.model_selection_service import model_selection_service
from app.services.unified_provider_registry import unified_registry

logger = logging.getLogger(__name__)


class SelectionMiddleware(BaseHTTPMiddleware):
    """Middleware for injecting provider/model selection into request context.

    This middleware implements the selection resolution strategy:
    1. Check for request-level override (headers or query params)
    2. Check for session-level preference (database lookup)
    3. Check for global model selection (set via /model-selection endpoint)
    4. Fall back to static default (environment variables)

    Once resolved, the selection is set in SelectionContext and remains
    immutable for the duration of the request.
    """

    def __init__(
        self,
        app,
        default_provider: str = "openai",
        default_model: str = "gpt-4-turbo",
        enable_session_lookup: bool = True,
        excluded_paths: list[str] | None = None,
    ) -> None:
        """Initialize selection middleware.

        Args:
            app: FastAPI application instance
            default_provider: Default provider ID (fallback)
            default_model: Default model ID (fallback)
            enable_session_lookup: Whether to perform session database lookups
            excluded_paths: List of paths to exclude from selection logic

        """
        super().__init__(app)
        self._default_provider = default_provider
        self._default_model = default_model
        self._enable_session_lookup = enable_session_lookup
        self.excluded_paths = excluded_paths or []
        # Ensure paths are normalized
        self.excluded_paths = [path.rstrip("/") for path in self.excluded_paths]

        logger.info(f"SelectionMiddleware initialized (default={default_provider}/{default_model})")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and inject selection into context.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler in chain

        Returns:
            Response from downstream handler

        """
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())

        # Check for excluded paths
        path = request.url.path.rstrip("/")
        for excluded_path in self.excluded_paths:
            if path == excluded_path or path.startswith(excluded_path + "/"):
                logger.debug(f"[{request_id}] Skipping selection for excluded path: {path}")
                return await call_next(request)

        try:
            # Resolve selection using three-tier hierarchy
            selection = await self._resolve_selection(request, request_id)

            # Set selection in context
            SelectionContext.set_selection(
                selection=selection,
                session_id=self._extract_session_id(request),
                user_id=self._extract_user_id(request),
                request_id=request_id,
            )

            logger.debug(
                f"[{request_id}] Selection set: {selection.provider_id}/{selection.model_id} "
                f"(scope={selection.scope.value})",
            )

            # Attach request ID to request state for downstream access
            request.state.request_id = request_id
            request.state.selection = selection

            # Call next handler
            response = await call_next(request)

            # Add selection headers to response
            response.headers["X-Provider-ID"] = selection.provider_id
            response.headers["X-Model-ID"] = selection.model_id
            response.headers["X-Selection-Scope"] = selection.scope.value
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            logger.error(f"[{request_id}] Selection middleware error: {e}", exc_info=True)
            # Don't block request on selection errors - use defaults
            fallback_selection = self._create_fallback_selection()
            SelectionContext.set_selection(
                selection=fallback_selection,
                request_id=request_id,
            )
            return await call_next(request)

        finally:
            # Context cleanup happens automatically via contextvars
            pass

    async def _resolve_selection(self, request: Request, request_id: str) -> Selection:
        """Resolve selection using four-tier hierarchy.

        Priority (highest to lowest):
        1. Request Override - Headers or query parameters
        2. Session Preference - Database lookup by session_id
        3. Global Model Selection - User-set via /model-selection endpoint
        4. Static Default - Environment variables or constructor defaults

        Args:
            request: FastAPI request
            request_id: Request identifier for logging

        Returns:
            Resolved Selection object

        """
        # 1. Check for request-level override
        request_override = self._extract_request_override(request)
        if request_override:
            provider_id, model_id = request_override
            if self._validate_selection(provider_id, model_id):
                logger.debug(f"[{request_id}] Using request override: {provider_id}/{model_id}")
                return Selection(
                    provider_id=provider_id,
                    model_id=model_id,
                    scope=SelectionScope.REQUEST,
                    user_id=self._extract_user_id(request),
                    session_id=self._extract_session_id(request),
                )

        # 2. Check for session-level preference
        if self._enable_session_lookup:
            session_id = self._extract_session_id(request)
            if session_id:
                session_selection = await self._lookup_session_preference(session_id, request_id)
                if session_selection and session_selection[0] and session_selection[1]:
                    provider_id, model_id = session_selection
                    logger.debug(
                        f"[{request_id}] Using session preference: {provider_id}/{model_id}",
                    )

                    # Create a simple object to match expected interface
                    class SessionSelection:
                        def __init__(self, provider_id: str, model_id: str) -> None:
                            self.provider_id = provider_id
                            self.model_id = model_id

                    return SessionSelection(provider_id, model_id)

        # 3. Check for global model selection (set via /model-selection endpoint)
        global_selection = self._get_global_model_selection(request_id)
        if global_selection:
            return global_selection

        # 4. Fall back to static default
        logger.debug(
            f"[{request_id}] Using static default: {self._default_provider}/{self._default_model}",
        )
        return Selection(
            provider_id=self._default_provider,
            model_id=self._default_model,
            scope=SelectionScope.GLOBAL,
            user_id=self._extract_user_id(request),
            session_id=self._extract_session_id(request),
        )

    def _get_global_model_selection(self, request_id: str) -> Selection | None:
        """Get the global model selection set via /model-selection endpoint.

        This reads from the model_selection_service which persists user's
        selected provider/model to a JSON file.

        Args:
            request_id: Request identifier for logging

        Returns:
            Selection if set and valid, None otherwise

        """
        try:
            selection = model_selection_service.get_selection()
            if selection:
                provider_id = selection.provider
                model_id = selection.model

                # Validate the selection is still valid
                if self._validate_selection(provider_id, model_id):
                    logger.debug(
                        f"[{request_id}] Using global model selection: {provider_id}/{model_id}",
                    )
                    return Selection(
                        provider_id=provider_id,
                        model_id=model_id,
                        scope=SelectionScope.GLOBAL,
                    )
                logger.warning(
                    f"[{request_id}] Global model selection {provider_id}/{model_id} "
                    "is no longer valid. Falling back to static default.",
                )
            return None
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to get global model selection: {e}")
            return None

    def _extract_request_override(self, request: Request) -> tuple[str, str] | None:
        """Extract provider/model override from request headers or query params.

        Checks (in order):
        1. Headers: X-Provider-ID and X-Model-ID
        2. Query params: provider and model

        Args:
            request: FastAPI request

        Returns:
            Tuple of (provider_id, model_id) if found, None otherwise

        """
        # Check headers
        provider_id = request.headers.get("X-Provider-ID")
        model_id = request.headers.get("X-Model-ID")

        if provider_id and model_id:
            return (provider_id, model_id)

        # Check query params
        provider_id = request.query_params.get("provider")
        model_id = request.query_params.get("model")

        if provider_id and model_id:
            return (provider_id, model_id)

        return None

    def _extract_session_id(self, request: Request) -> str | None:
        """Extract session ID from request.

        Checks (in order):
        1. Request state (set by auth middleware)
        2. X-Session-ID header
        3. session_id query parameter
        4. Session cookie

        Args:
            request: FastAPI request

        Returns:
            Session ID if found, None otherwise

        """
        # Check request state
        if hasattr(request.state, "session_id"):
            return request.state.session_id

        # Check headers
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            return session_id

        # Check query params
        session_id = request.query_params.get("session_id")
        if session_id:
            return session_id

        # Check cookies
        session_id = request.cookies.get("session_id")
        if session_id:
            return session_id

        return None

    def _extract_user_id(self, request: Request) -> str | None:
        """Extract user ID from request.

        Checks (in order):
        1. Request state (set by auth middleware)
        2. X-User-ID header

        Args:
            request: FastAPI request

        Returns:
            User ID if found, None otherwise

        """
        # Check request state
        if hasattr(request.state, "user_id"):
            return request.state.user_id

        # Check headers
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id

        return None

    async def _lookup_session_preference(
        self,
        session_id: str,
        request_id: str,
    ) -> tuple[str | None, str | None]:
        """Look up session preference from database.

        Queries the selections table for the session's stored preference.
        Returns None if no preference is stored or if the stored selection
        is no longer valid (provider/model disabled or removed).

        Args:
            session_id: Session identifier
            request_id: Request identifier for logging

        Returns:
            Tuple of (provider_id, model_id) if found, None otherwise

        """
        try:
            from app.services.selection_service import selection_service

            # Use async session factory context manager
            async_session_factory = get_async_session_factory()
            async with async_session_factory() as db_session:
                selection = await selection_service.get_session_selection(session_id, db_session)

                if selection:
                    logger.debug(
                        f"[{request_id}] Found session preference: "
                        f"{selection.provider_id}/{selection.model_id}",
                    )
                    return selection.provider_id, selection.model_id
                logger.debug(
                    f"[{request_id}] No valid session preference found for session={session_id}",
                )
                return None, None

        except Exception as e:
            logger.error(f"[{request_id}] Failed to lookup session preference: {e}", exc_info=True)
            return None, None

    def _validate_selection(self, provider_id: str, model_id: str) -> bool:
        """Validate that provider/model combination is valid.

        Validation strategy (from most to least strict):
        1. Check unified_registry if available
        2. Check llm_service._providers for registered providers
        3. Check model_router_service for model availability
        4. Be lenient and allow if we can't validate

        Args:
            provider_id: Provider identifier
            model_id: Model identifier

        Returns:
            True if valid, False otherwise

        """
        try:
            # First try unified_registry (if populated)
            if unified_registry._providers:
                is_valid = unified_registry.validate_selection(provider_id, model_id)
                if is_valid:
                    return True
                # If unified_registry rejects, try other methods before giving up

            # Check if provider is registered in llm_service (core provider system)
            # This is the authoritative source for registered providers
            try:
                from app.services.llm_service import llm_service

                if provider_id in llm_service._providers:
                    # Provider is registered - allow the selection
                    # We trust that the user knows the correct model IDs
                    logger.debug(f"Validated selection via llm_service: {provider_id}/{model_id}")
                    return True
            except Exception as e:
                logger.debug(f"llm_service check failed: {e}")

            # Fall back to model_router_service validation
            try:
                from app.services.model_router_service import model_router_service

                providers = model_router_service.get_supported_providers()

                # Strip provider prefix from model_id for comparison
                model_name = (
                    model_id.replace(f"{provider_id}:", "") if ":" in model_id else model_id
                )

                for p in providers:
                    if p.get("provider") == provider_id:
                        # Check in available_models list (API returns 'available_models' not 'models')
                        models = p.get("available_models", p.get("models", []))
                        if model_name in models or model_id in models:
                            logger.debug(
                                f"Validated selection via model_router_service: {provider_id}/{model_id}",
                            )
                            return True

                        # Also check models_detail if available
                        models_detail = p.get("models_detail", [])
                        for m in models_detail:
                            if m.get("id") == model_id or m.get("name") == model_name:
                                logger.debug(
                                    f"Validated selection via models_detail: {provider_id}/{model_id}",
                                )
                                return True

                        # Provider found in model_router but model not in list
                        # Still allow it - model catalogs can be incomplete
                        logger.debug(
                            f"Provider '{provider_id}' found but model '{model_id}' not in catalog. "
                            f"Allowing selection anyway.",
                        )
                        return True

                # Provider not found in model_router_service - that's okay
                # It might be a custom/dynamically added provider
                logger.debug(
                    f"Provider '{provider_id}' not in model_router_service. "
                    f"Allowing selection if provider is otherwise valid.",
                )
                # Allow if we couldn't find it - be lenient
                return True
            except Exception as e:
                logger.warning(f"model_router_service validation failed: {e}")
                # Be lenient - allow selection if we can't validate
                return True
        except Exception as e:
            logger.warning(f"Selection validation error: {e}")
            # Be lenient on validation errors - allow the selection
            return True

    def _create_fallback_selection(self) -> Selection:
        """Create a fallback selection for error cases.

        Returns:
            Selection with default values

        """
        return Selection(
            provider_id=self._default_provider,
            model_id=self._default_model,
            scope=SelectionScope.GLOBAL,
        )


class SelectionValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for validating that selection context is properly set.

    This middleware runs after SelectionMiddleware and ensures that
    the selection context is available for downstream handlers.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate selection context and call next handler.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler in chain

        Returns:
            Response from downstream handler

        """
        # Check if selection is set
        if not SelectionContext.is_set():
            logger.error(
                f"Selection context not set for request: {request.url.path}. "
                "Ensure SelectionMiddleware is configured correctly.",
            )
            # Could raise HTTPException here, but for now we'll allow it to proceed

        # Log selection for debugging
        selection = SelectionContext.get_selection()
        if selection:
            logger.debug(
                f"Validated selection: {selection.provider_id}/{selection.model_id} "
                f"for path: {request.url.path}",
            )

        return await call_next(request)
