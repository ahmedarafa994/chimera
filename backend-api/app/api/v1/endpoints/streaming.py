"""
SSE Streaming endpoint for real-time text generation.

This module provides SSE streaming for LLM text generation,
enabling real-time token-by-token responses to clients.

Now integrated with Unified Provider/Model Selection System for
consistent provider/model resolution across streaming operations.

Features:
- Unified provider/model selection with hierarchy resolution
- Selection tracing headers for debugging
- Multiple output formats (SSE, JSONL, raw)
- Stream lifecycle management (metrics, cancellation)
- Comprehensive error handling for provider failures
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.errors import AppError
from app.core.logging import logger
from app.domain.models import PromptRequest
from app.services.llm_service import LLMService, llm_service
from app.services.provider_resolution_service import (
    ProviderResolutionService,
    get_provider_resolution_service,
)
from app.services.stream_context import StreamingContext, get_streaming_context
from app.services.stream_formatters import (
    StreamFormat,
    StreamingHeadersBuilder,
    StreamResponseFormatter,
    get_stream_formatter,
)
from app.services.unified_streaming_service import (
    UnifiedStreamingService,
    get_unified_streaming_service,
)

router = APIRouter(tags=["streaming"])


# =============================================================================
# Custom Exceptions for Streaming
# =============================================================================


class StreamingError(Exception):
    """Base exception for streaming errors."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        stream_id: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.stream_id = stream_id


class ProviderNotAvailableError(StreamingError):
    """Raised when the resolved provider is not available."""

    def __init__(self, provider: str, stream_id: str | None = None):
        super().__init__(
            message=f"Provider '{provider}' is not available or not configured",
            error_code="PROVIDER_NOT_AVAILABLE",
            status_code=503,
            stream_id=stream_id,
        )
        self.provider = provider


class ModelNotFoundError(StreamingError):
    """Raised when the specified model is not found."""

    def __init__(
        self,
        model: str,
        provider: str,
        stream_id: str | None = None,
    ):
        super().__init__(
            message=f"Model '{model}' not found for provider '{provider}'",
            error_code="MODEL_NOT_FOUND",
            status_code=404,
            stream_id=stream_id,
        )
        self.model = model
        self.provider = provider


class RateLimitExceededError(StreamingError):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        provider: str,
        retry_after: int | None = None,
        stream_id: str | None = None,
    ):
        super().__init__(
            message=f"Rate limit exceeded for provider '{provider}'",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            stream_id=stream_id,
        )
        self.provider = provider
        self.retry_after = retry_after


class StreamInterruptedError(StreamingError):
    """Raised when a stream is interrupted unexpectedly."""

    def __init__(
        self,
        reason: str,
        stream_id: str | None = None,
    ):
        super().__init__(
            message=f"Stream interrupted: {reason}",
            error_code="STREAM_INTERRUPTED",
            status_code=500,
            stream_id=stream_id,
        )
        self.reason = reason


# =============================================================================
# Request/Response Models
# =============================================================================


class StreamGenerateRequest(BaseModel):
    """Request model for unified stream generation."""

    prompt: str = Field(..., description="The text prompt for generation")
    system_instruction: str | None = Field(None, description="Optional system instruction")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int | None = Field(None, ge=1, le=100000, description="Maximum tokens to generate")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling")
    stop_sequences: list[str] | None = Field(None, description="Stop sequences")
    provider: str | None = Field(None, description="Explicit provider override")
    model: str | None = Field(None, description="Explicit model override")


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str = Field(..., description="Message role (system/user/assistant)")
    content: str = Field(..., description="Message content")


class StreamChatRequest(BaseModel):
    """Request model for unified stream chat."""

    messages: list[ChatMessage] = Field(..., description="List of chat messages")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int | None = Field(None, ge=1, le=100000, description="Maximum tokens to generate")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling")
    stop_sequences: list[str] | None = Field(None, description="Stop sequences")
    provider: str | None = Field(None, description="Explicit provider override")
    model: str | None = Field(None, description="Explicit model override")


# =============================================================================
# Dependencies
# =============================================================================


def get_llm_service() -> LLMService:
    """Get LLM service instance."""
    return llm_service


async def get_streaming_service() -> UnifiedStreamingService:
    """Get unified streaming service instance."""
    return get_unified_streaming_service()


async def get_stream_context() -> StreamingContext:
    """Get streaming context manager instance."""
    return get_streaming_context()


async def get_formatter() -> StreamResponseFormatter:
    """Get stream response formatter instance."""
    return get_stream_formatter()


# =============================================================================
# Legacy Stream Generators (for backward compatibility)
# =============================================================================


async def _stream_generator(
    request: PromptRequest, service: LLMService
) -> AsyncGenerator[str, None]:
    """
    Generate SSE events from streaming response.

    Yields SSE-formatted events with JSON data containing:
    - text: The generated text chunk
    - is_final: Whether this is the final chunk
    - finish_reason: Reason for completion (if final)
    - token_count: Token count for this chunk (if available)
    """
    import json

    try:
        # Use the LLM service's streaming method
        async for chunk in service.generate_text_stream(request):
            event_data = {
                "text": chunk.text,
                "is_final": chunk.is_final,
                "finish_reason": chunk.finish_reason,
                "token_count": chunk.token_count,
            }
            yield f"data: {json.dumps(event_data)}\n\n"

            if chunk.is_final:
                break

    except AppError as e:
        logger.error(f"Streaming error: {e.message}")
        error = {"error": e.message, "status_code": e.status_code}
        yield f"data: {json.dumps(error)}\n\n"
    except NotImplementedError as e:
        logger.error(f"Streaming not supported: {e}")
        error = {"error": str(e), "status_code": 501}
        yield f"data: {json.dumps(error)}\n\n"
    except Exception as e:
        logger.error(f"Unexpected streaming error: {e}", exc_info=True)
        error = {"error": str(e), "status_code": 500}
        yield f"data: {json.dumps(error)}\n\n"


async def _raw_stream_generator(
    request: PromptRequest, service: LLMService
) -> AsyncGenerator[str, None]:
    """
    Generate raw text chunks without SSE formatting.

    Useful for direct text streaming to UI without JSON parsing overhead.
    """
    try:
        async for chunk in service.generate_text_stream(request):
            if chunk.text:
                yield chunk.text

    except AppError as e:
        logger.error(f"Raw streaming error: {e.message}")
        yield f"\n[Error: {e.message}]"
    except NotImplementedError as e:
        logger.error(f"Streaming not supported: {e}")
        yield f"\n[Error: Streaming not supported - {e}]"
    except Exception as e:
        logger.error(f"Unexpected raw streaming error: {e}", exc_info=True)
        yield f"\n[Error: {e}]"


# =============================================================================
# Unified Streaming Generators
# =============================================================================


async def _unified_stream_generator(
    request: StreamGenerateRequest,
    session_id: str | None,
    streaming_service: UnifiedStreamingService,
    formatter: StreamResponseFormatter,
    stream_format: StreamFormat = StreamFormat.SSE,
) -> AsyncGenerator[str, None]:
    """
    Generate streaming response using unified provider/model selection.

    Uses the UnifiedStreamingService to resolve provider/model from the
    selection hierarchy and maintain consistency during streaming.
    """
    stream_id = ""
    total_chunks = 0

    try:
        async for chunk in streaming_service.stream_generate(
            prompt=request.prompt,
            session_id=session_id,
            explicit_provider=request.provider,
            explicit_model=request.model,
            system_instruction=request.system_instruction,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=request.stop_sequences,
        ):
            stream_id = chunk.stream_id
            total_chunks += 1

            # Format chunk based on requested format
            formatted = formatter.format_unified_chunk(chunk, stream_format)
            yield formatted

    except asyncio.CancelledError:
        logger.info(f"Stream cancelled: {stream_id}")
        if stream_format == StreamFormat.SSE:
            error_chunk = formatter.format_sse_error(
                error="Stream cancelled by client",
                error_code="CANCELLED",
                stream_id=stream_id,
            )
            yield error_chunk.content
        raise

    except AppError as e:
        logger.error(f"Unified streaming error: {e.message}")
        if stream_format == StreamFormat.SSE:
            error_chunk = formatter.format_sse_error(
                error=e.message,
                error_code=str(e.status_code),
                stream_id=stream_id,
            )
            yield error_chunk.content

    except Exception as e:
        logger.error(f"Unexpected unified streaming error: {e}", exc_info=True)
        if stream_format == StreamFormat.SSE:
            error_chunk = formatter.format_sse_error(
                error=str(e),
                error_code="500",
                stream_id=stream_id,
            )
            yield error_chunk.content

    finally:
        # Send done event for SSE
        if stream_format == StreamFormat.SSE and total_chunks > 0:
            done_chunk = formatter.format_sse_done(
                stream_id=stream_id,
                total_chunks=total_chunks,
            )
            yield done_chunk.content


async def _unified_chat_stream_generator(
    request: StreamChatRequest,
    session_id: str | None,
    streaming_service: UnifiedStreamingService,
    formatter: StreamResponseFormatter,
    stream_format: StreamFormat = StreamFormat.SSE,
) -> AsyncGenerator[str, None]:
    """
    Generate chat streaming response using unified provider/model selection.
    """
    stream_id = ""
    total_chunks = 0

    # Convert messages to list of dicts
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    try:
        async for chunk in streaming_service.stream_chat(
            messages=messages,
            session_id=session_id,
            explicit_provider=request.provider,
            explicit_model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=request.stop_sequences,
        ):
            stream_id = chunk.stream_id
            total_chunks += 1

            formatted = formatter.format_unified_chunk(chunk, stream_format)
            yield formatted

    except asyncio.CancelledError:
        logger.info(f"Chat stream cancelled: {stream_id}")
        if stream_format == StreamFormat.SSE:
            error_chunk = formatter.format_sse_error(
                error="Stream cancelled by client",
                error_code="CANCELLED",
                stream_id=stream_id,
            )
            yield error_chunk.content
        raise

    except AppError as e:
        logger.error(f"Unified chat streaming error: {e.message}")
        if stream_format == StreamFormat.SSE:
            error_chunk = formatter.format_sse_error(
                error=e.message,
                error_code=str(e.status_code),
                stream_id=stream_id,
            )
            yield error_chunk.content

    except Exception as e:
        logger.error(f"Unexpected chat streaming error: {e}", exc_info=True)
        if stream_format == StreamFormat.SSE:
            error_chunk = formatter.format_sse_error(
                error=str(e),
                error_code="500",
                stream_id=stream_id,
            )
            yield error_chunk.content

    finally:
        if stream_format == StreamFormat.SSE and total_chunks > 0:
            done_chunk = formatter.format_sse_done(
                stream_id=stream_id,
                total_chunks=total_chunks,
            )
            yield done_chunk.content


# =============================================================================
# Legacy Endpoints (for backward compatibility)
# =============================================================================


@router.post(
    "/generate/stream",
    summary="Stream text generation (legacy)",
    description="""
    Stream text generation using Server-Sent Events (SSE).

    **Note**: This is the legacy endpoint. For unified provider/model
    selection, use `/stream/generate` instead.

    This endpoint returns a stream of JSON events, each containing:
    - `text`: The generated text chunk
    - `is_final`: Boolean indicating if this is the final chunk
    - `finish_reason`: Reason for completion (e.g., "STOP", "MAX_TOKENS")
    - `token_count`: Number of tokens in this chunk (if available)

    **Supported Providers**: Google/Gemini, OpenAI, Anthropic, DeepSeek
    """,
    responses={
        200: {
            "description": "SSE stream of generated text chunks",
            "content": {"text/event-stream": {}},
        },
        400: {"description": "Invalid request"},
        501: {"description": "Streaming not supported"},
        502: {"description": "Provider error"},
    },
)
async def stream_generate_legacy(
    request: PromptRequest,
    service: LLMService = Depends(get_llm_service),
):
    """
    Stream text generation using Server-Sent Events (SSE).
    Legacy endpoint for backward compatibility.
    """
    return StreamingResponse(
        _stream_generator(request, service),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/generate/stream/raw",
    summary="Stream raw text generation (legacy)",
    description="""
    Stream raw text chunks without SSE JSON formatting.

    **Note**: This is the legacy endpoint.

    This endpoint returns plain text chunks as they are generated,
    useful for direct display in UI without JSON parsing overhead.
    """,
    responses={
        200: {
            "description": "Stream of raw text chunks",
            "content": {"text/plain": {}},
        },
        400: {"description": "Invalid request"},
        501: {"description": "Streaming not supported"},
        502: {"description": "Provider error"},
    },
)
async def stream_generate_raw(
    request: PromptRequest,
    service: LLMService = Depends(get_llm_service),
):
    """Stream raw text chunks (no JSON wrapping). Legacy endpoint."""
    return StreamingResponse(
        _raw_stream_generator(request, service),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# New Unified Streaming Endpoints
# =============================================================================


@router.post(
    "/stream/generate",
    summary="Stream text generation with unified selection",
    description="""
    Stream text generation using current provider/model selection.

    This endpoint uses the Unified Provider/Model Selection System to
    resolve which provider and model to use based on the hierarchy:
    1. Explicit parameters (provider/model in request)
    2. Session selection (from X-Session-Id header)
    3. Global default selection

    The selection remains consistent for the entire stream duration.

    Response headers include:
    - `X-Stream-Provider`: The resolved provider
    - `X-Stream-Model`: The resolved model
    - `X-Stream-Session-Id`: The session ID (if provided)
    - `X-Stream-Started-At`: Stream start timestamp

    **Example SSE events**:
    ```
    event: chunk
    id: 0
    data: {"text":"Hello","chunk_index":0,"is_final":false,...}

    event: chunk
    id: 1
    data: {"text":" world","chunk_index":1,"is_final":false,...}

    event: done
    data: {"stream_id":"stream_abc123","total_chunks":2,...}
    ```
    """,
    responses={
        200: {
            "description": "SSE stream of generated text chunks",
            "content": {"text/event-stream": {}},
        },
        400: {"description": "Invalid request"},
        501: {"description": "Streaming not supported for resolved provider"},
        502: {"description": "Provider error"},
    },
)
async def stream_generate(
    request: StreamGenerateRequest,
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    streaming_service: UnifiedStreamingService = Depends(get_streaming_service),
    resolution_service: ProviderResolutionService = Depends(
        lambda: get_provider_resolution_service()
    ),
    formatter: StreamResponseFormatter = Depends(get_formatter),
):
    """
    Stream text generation using unified provider/model selection.
    """
    # Generate stream ID upfront for tracing
    stream_id = f"stream_{uuid.uuid4().hex[:12]}"
    datetime.utcnow()

    try:
        # Resolve provider/model before starting stream
        resolution = await resolution_service.resolve_with_metadata(
            explicit_provider=request.provider,
            explicit_model=request.model,
            session_id=x_session_id,
            user_id=x_user_id,
        )

        # Validate provider availability
        if not await _validate_provider_available(resolution.provider, streaming_service):
            raise ProviderNotAvailableError(
                provider=resolution.provider,
                stream_id=stream_id,
            )

        # Build headers with full selection tracing
        headers = StreamingHeadersBuilder.build_sse_headers(
            provider=resolution.provider,
            model=resolution.model_id,
            session_id=x_session_id,
            stream_id=stream_id,
        )

        # Add additional tracing headers
        headers["X-Stream-Resolution-Source"] = resolution.resolution_source
        headers["X-Stream-Resolution-Priority"] = str(resolution.resolution_priority)

        return StreamingResponse(
            _unified_stream_generator(
                request=request,
                session_id=x_session_id,
                streaming_service=streaming_service,
                formatter=formatter,
                stream_format=StreamFormat.SSE,
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    except ProviderNotAvailableError as e:
        logger.error(f"Provider not available: {e.provider}")
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "provider": e.provider,
                "stream_id": stream_id,
            },
        )
    except ModelNotFoundError as e:
        logger.error(f"Model not found: {e.model} for {e.provider}")
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "model": e.model,
                "provider": e.provider,
                "stream_id": stream_id,
            },
        )
    except RateLimitExceededError as e:
        logger.warning(f"Rate limit exceeded: {e.provider}")
        headers = {"Retry-After": str(e.retry_after)} if e.retry_after else {}
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "provider": e.provider,
                "retry_after": e.retry_after,
                "stream_id": stream_id,
            },
            headers=headers,
        )
    except Exception as e:
        logger.error(f"Unexpected error starting stream: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "STREAM_START_ERROR",
                "message": str(e),
                "stream_id": stream_id,
            },
        )


@router.post(
    "/stream/chat",
    summary="Stream chat completion with unified selection",
    description="""
    Stream chat completion using current provider/model selection.

    Similar to `/stream/generate` but accepts chat messages instead
    of a single prompt. Uses the Unified Provider/Model Selection System.

    **Request body**:
    ```json
    {
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    ```
    """,
    responses={
        200: {
            "description": "SSE stream of chat response chunks",
            "content": {"text/event-stream": {}},
        },
        400: {"description": "Invalid request"},
        501: {"description": "Streaming not supported"},
        502: {"description": "Provider error"},
    },
)
async def stream_chat(
    request: StreamChatRequest,
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    streaming_service: UnifiedStreamingService = Depends(get_streaming_service),
    resolution_service: ProviderResolutionService = Depends(
        lambda: get_provider_resolution_service()
    ),
    formatter: StreamResponseFormatter = Depends(get_formatter),
):
    """
    Stream chat completion using unified provider/model selection.
    """
    # Generate stream ID upfront for tracing
    stream_id = f"stream_{uuid.uuid4().hex[:12]}"
    datetime.utcnow()

    try:
        # Resolve provider/model before starting stream
        resolution = await resolution_service.resolve_with_metadata(
            explicit_provider=request.provider,
            explicit_model=request.model,
            session_id=x_session_id,
            user_id=x_user_id,
        )

        # Validate provider availability
        if not await _validate_provider_available(resolution.provider, streaming_service):
            raise ProviderNotAvailableError(
                provider=resolution.provider,
                stream_id=stream_id,
            )

        # Build headers with full selection tracing
        headers = StreamingHeadersBuilder.build_sse_headers(
            provider=resolution.provider,
            model=resolution.model_id,
            session_id=x_session_id,
            stream_id=stream_id,
        )

        # Add additional tracing headers
        headers["X-Stream-Resolution-Source"] = resolution.resolution_source
        headers["X-Stream-Resolution-Priority"] = str(resolution.resolution_priority)

        return StreamingResponse(
            _unified_chat_stream_generator(
                request=request,
                session_id=x_session_id,
                streaming_service=streaming_service,
                formatter=formatter,
                stream_format=StreamFormat.SSE,
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    except ProviderNotAvailableError as e:
        logger.error(f"Provider not available for chat: {e.provider}")
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "provider": e.provider,
                "stream_id": stream_id,
            },
        )
    except ModelNotFoundError as e:
        logger.error(f"Model not found for chat: {e.model} for {e.provider}")
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "model": e.model,
                "provider": e.provider,
                "stream_id": stream_id,
            },
        )
    except RateLimitExceededError as e:
        logger.warning(f"Rate limit exceeded for chat: {e.provider}")
        headers = {"Retry-After": str(e.retry_after)} if e.retry_after else {}
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "provider": e.provider,
                "retry_after": e.retry_after,
                "stream_id": stream_id,
            },
            headers=headers,
        )
    except Exception as e:
        logger.error(f"Unexpected error starting chat stream: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "STREAM_START_ERROR",
                "message": str(e),
                "stream_id": stream_id,
            },
        )


@router.post(
    "/stream/generate/jsonl",
    summary="Stream text generation as JSON Lines",
    description="""
    Stream text generation using JSON Lines format instead of SSE.

    Each line is a complete JSON object. Useful for clients that
    prefer line-by-line parsing over SSE event handling.
    """,
    responses={
        200: {
            "description": "JSON Lines stream",
            "content": {"application/x-ndjson": {}},
        },
        400: {"description": "Invalid request"},
    },
)
async def stream_generate_jsonl(
    request: StreamGenerateRequest,
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    streaming_service: UnifiedStreamingService = Depends(get_streaming_service),
    resolution_service: ProviderResolutionService = Depends(
        lambda: get_provider_resolution_service()
    ),
    formatter: StreamResponseFormatter = Depends(get_formatter),
):
    """Stream text generation as JSON Lines."""
    # Generate stream ID upfront for tracing
    stream_id = f"stream_{uuid.uuid4().hex[:12]}"

    try:
        # Resolve provider/model before starting stream
        resolution = await resolution_service.resolve_with_metadata(
            explicit_provider=request.provider,
            explicit_model=request.model,
            session_id=x_session_id,
            user_id=x_user_id,
        )

        # Validate provider availability
        if not await _validate_provider_available(resolution.provider, streaming_service):
            raise ProviderNotAvailableError(
                provider=resolution.provider,
                stream_id=stream_id,
            )

        # Build headers with full selection tracing
        headers = StreamingHeadersBuilder.build_jsonl_headers(
            provider=resolution.provider,
            model=resolution.model_id,
            session_id=x_session_id,
            stream_id=stream_id,
        )

        return StreamingResponse(
            _unified_stream_generator(
                request=request,
                session_id=x_session_id,
                streaming_service=streaming_service,
                formatter=formatter,
                stream_format=StreamFormat.JSONL,
            ),
            media_type="application/x-ndjson",
            headers=headers,
        )

    except ProviderNotAvailableError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "provider": e.provider,
                "stream_id": stream_id,
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error starting JSONL stream: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "STREAM_START_ERROR",
                "message": str(e),
                "stream_id": stream_id,
            },
        )


# =============================================================================
# Helper Functions
# =============================================================================


async def _validate_provider_available(
    provider: str,
    streaming_service: UnifiedStreamingService,
) -> bool:
    """
    Validate that a provider is available for streaming.

    Checks:
    - Provider plugin exists
    - Provider supports streaming
    - Provider API key is configured (via plugin)

    Returns:
        True if provider is available, False otherwise
    """
    try:
        from app.services.provider_plugins import get_plugin

        plugin = get_plugin(provider)
        if not plugin:
            logger.warning(f"Provider plugin not found: {provider}")
            return False

        if not plugin.supports_streaming():
            logger.warning(f"Provider {provider} does not support streaming")
            return False

        # Check if API key is configured
        if not plugin.is_configured():
            logger.warning(f"Provider {provider} is not configured")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating provider {provider}: {e}")
        return False


# =============================================================================
# Streaming Management Endpoints
# =============================================================================


@router.get(
    "/stream/active",
    summary="Get active streams",
    description="Get information about currently active streaming sessions.",
    responses={200: {"description": "List of active streams"}},
)
async def get_active_streams(
    streaming_service: UnifiedStreamingService = Depends(get_streaming_service),
):
    """Get all currently active streaming sessions."""
    return {
        "active_streams": streaming_service.get_active_streams(),
        "count": len(streaming_service.get_active_streams()),
    }


@router.get(
    "/stream/metrics",
    summary="Get streaming metrics",
    description="Get aggregate metrics about streaming performance.",
    responses={200: {"description": "Streaming metrics summary"}},
)
async def get_stream_metrics(
    streaming_service: UnifiedStreamingService = Depends(get_streaming_service),
):
    """Get streaming performance metrics."""
    return streaming_service.get_metrics_summary()


@router.get(
    "/stream/metrics/recent",
    summary="Get recent stream metrics",
    description="Get metrics for recent streaming sessions.",
    responses={200: {"description": "Recent stream metrics"}},
)
async def get_recent_stream_metrics(
    limit: int = 100,
    streaming_service: UnifiedStreamingService = Depends(get_streaming_service),
):
    """Get metrics for recent streams."""
    return {
        "metrics": streaming_service.get_recent_metrics(limit=limit),
        "count": min(limit, len(streaming_service.get_recent_metrics(limit))),
    }


@router.delete(
    "/stream/{stream_id}",
    summary="Cancel a stream",
    description="Request cancellation of an active stream.",
    responses={
        200: {"description": "Stream cancelled"},
        404: {"description": "Stream not found"},
    },
)
async def cancel_stream(
    stream_id: str,
    streaming_service: UnifiedStreamingService = Depends(get_streaming_service),
):
    """Cancel an active stream by ID."""
    cancelled = await streaming_service.cancel_stream(stream_id)
    if cancelled:
        return {"status": "cancelled", "stream_id": stream_id}
    raise HTTPException(
        status_code=404,
        detail=f"Stream not found: {stream_id}",
    )


# =============================================================================
# Capabilities Endpoint
# =============================================================================


@router.get(
    "/generate/stream/capabilities",
    summary="Get streaming capabilities",
    description="Get which providers support streaming.",
    responses={200: {"description": "Streaming capabilities by provider"}},
)
async def get_streaming_capabilities(
    service: LLMService = Depends(get_llm_service),
):
    """
    Get streaming capabilities for all registered providers.
    """
    providers = service.get_available_providers()
    capabilities = {}

    for provider in providers:
        capabilities[provider] = {
            "streaming_supported": service.supports_streaming(provider),
            "token_counting_supported": service.supports_token_counting(provider),
        }

    return {
        "providers": capabilities,
        "default_provider": service._default_provider,
    }
