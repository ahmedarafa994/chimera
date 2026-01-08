"""
Input Validation and Sanitization Module
Provides comprehensive input validation to prevent injection attacks
"""

import html
import logging
import re
from typing import Any, ClassVar

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Sanitization Utilities
# =============================================================================


class Sanitizer:
    """Input sanitization utilities"""

    # Patterns for potentially dangerous content
    SCRIPT_PATTERN = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)

    @classmethod
    def escape_html(cls, text: str) -> str:
        """Escape HTML entities to prevent XSS"""
        return html.escape(text)

    @classmethod
    def remove_null_bytes(cls, text: str) -> str:
        """Remove null bytes that could cause issues"""
        return text.replace("\x00", "")

    @classmethod
    def remove_scripts(cls, text: str) -> str:
        """Remove script tags"""
        return cls.SCRIPT_PATTERN.sub("", text)

    @classmethod
    def sanitize_prompt(cls, text: str, max_length: int = 50000) -> str:
        """
        Sanitize a prompt input.

        Args:
            text: The input text to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized text
        """
        if not text:
            return ""

        # Remove null bytes
        text = cls.remove_null_bytes(text)

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Input truncated from {len(text)} to {max_length} chars")

        # Remove excessive whitespace
        text = " ".join(text.split())

        return text.strip()

    @classmethod
    def sanitize_for_display(cls, text: str) -> str:
        """Sanitize text for safe display (HTML escaped)"""
        text = cls.remove_null_bytes(text)
        text = cls.remove_scripts(text)
        text = cls.escape_html(text)
        return text


# =============================================================================
# Validated Input Models
# =============================================================================


class PromptInput(BaseModel):
    """
    Validated prompt input with injection prevention.

    Usage:
        @app.post("/api/enhance")
        async def enhance(request: PromptInput):
            # request.prompt is already sanitized
            ...
    """

    prompt: str = Field(
        ..., min_length=1, max_length=50000, description="The prompt text to process"
    )

    @validator("prompt")
    def sanitize_prompt(cls, v):
        """Sanitize the prompt input"""
        if not v:
            raise ValueError("Prompt cannot be empty")

        # Remove null bytes
        v = Sanitizer.remove_null_bytes(v)

        # Check for suspiciously long repeated patterns
        if len(set(v)) < len(v) * 0.01 and len(v) > 1000:
            raise ValueError("Prompt contains suspicious repeated content")

        return v.strip()

    class Config:
        str_strip_whitespace = True


class EnhancementRequest(BaseModel):
    """Validated enhancement request"""

    prompt: str = Field(..., min_length=1, max_length=50000)
    tone: str | None = Field("engaging", max_length=50)
    virality_boost: bool | None = Field(True)
    include_seo: bool | None = Field(True)
    add_frameworks: bool | None = Field(True)

    @validator("prompt")
    def sanitize_prompt(cls, v):
        return Sanitizer.sanitize_prompt(v)

    @validator("tone")
    def validate_tone(cls, v):
        if v:
            allowed_tones = [
                "engaging",
                "professional",
                "casual",
                "formal",
                "persuasive",
                "informative",
                "creative",
                "technical",
            ]
            if v.lower() not in allowed_tones:
                raise ValueError(f"Tone must be one of: {allowed_tones}")
        return v.lower() if v else "engaging"


class JailbreakRequest(BaseModel):
    """Validated jailbreak enhancement request"""

    prompt: str = Field(..., min_length=1, max_length=50000)
    technique_preference: str | None = Field("advanced", max_length=50)
    obfuscation_level: int | None = Field(7, ge=1, le=10)
    target_model: str | None = Field("general", max_length=100)

    @validator("prompt")
    def sanitize_prompt(cls, v):
        return Sanitizer.sanitize_prompt(v)

    @validator("technique_preference")
    def validate_technique(cls, v):
        if v:
            allowed = ["basic", "intermediate", "advanced", "expert"]
            if v.lower() not in allowed:
                raise ValueError(f"Technique must be one of: {allowed}")
        return v.lower() if v else "advanced"

    @validator("target_model")
    def sanitize_target_model(cls, v):
        if v:
            # Only allow alphanumeric, hyphens, and dots
            if not re.match(r"^[\w\-\.]+$", v):
                raise ValueError("Invalid target model format")
        return v


class TransformRequest(BaseModel):
    """Validated transformation request"""

    prompt: str = Field(..., min_length=1, max_length=50000)
    technique: str | None = Field(None, max_length=100)
    suite: str | None = Field(None, max_length=100)
    intensity: int | None = Field(5, ge=1, le=10)
    provider: str | None = Field(None, max_length=50)
    model: str | None = Field(None, max_length=100)

    @validator("prompt")
    def sanitize_prompt(cls, v):
        return Sanitizer.sanitize_prompt(v)

    @validator("technique", "suite", "provider", "model")
    def sanitize_identifiers(cls, v):
        if v:
            # Only allow safe identifier characters
            if not re.match(r"^[\w\-\.]+$", v):
                raise ValueError("Invalid identifier format")
        return v


class AIModelRequest(BaseModel):
    """Validated AI model request"""

    prompt: str = Field(..., min_length=1, max_length=50000)
    model: str = Field(..., max_length=100)
    temperature: float | None = Field(0.7, ge=0, le=2)
    max_tokens: int | None = Field(2000, ge=1, le=8000)
    stream: bool | None = Field(False)

    @validator("prompt")
    def sanitize_prompt(cls, v):
        return Sanitizer.sanitize_prompt(v)

    @validator("model")
    def validate_model(cls, v):
        if not re.match(r"^[\w\-\.\/]+$", v):
            raise ValueError("Invalid model identifier format")
        return v


# =============================================================================
# Response Sanitization
# =============================================================================


class SanitizedResponse(BaseModel):
    """
    Base response model with automatic HTML escaping for display safety.

    Use this for any response that will be rendered in a browser.
    """

    class Config:
        # Custom JSON encoder for automatic escaping
        json_encoders: ClassVar[dict[type, Any]] = {str: lambda v: html.escape(v) if v else v}


class SafeAPIResponse(BaseModel):
    """Safe API response with sanitized output"""

    success: bool
    data: dict[str, Any] | None = None
    message: str | None = None

    @validator("message")
    def sanitize_message(cls, v):
        if v:
            return Sanitizer.sanitize_for_display(v)
        return v


# =============================================================================
# Validation Middleware
# =============================================================================

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for global input validation and sanitization.
    """

    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

    async def dispatch(self, request: Request, call_next):
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_CONTENT_LENGTH:
            raise HTTPException(status_code=413, detail="Request body too large")

        # Check content type for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if "application/json" not in content_type and "multipart/form-data" not in content_type:
                # Log but don't block - let the endpoint handle it
                logger.warning(
                    f"Unexpected content-type: {content_type} for {request.method} {request.url.path}"
                )

        response = await call_next(request)
        return response


# =============================================================================
# Rate Limiting Support
# =============================================================================


def get_rate_limit_key(request: Request) -> str:
    """
    Generate a rate limit key based on client identity.

    Tries to use the most specific identifier:
    1. Authenticated user ID
    2. API key hash
    3. Client IP
    """
    # Check for authenticated user
    if hasattr(request.state, "user"):
        return f"user:{request.state.user.sub}"

    # Check for API key (use hash for privacy)
    api_key = request.headers.get("X-API-Key")
    if api_key:
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        return f"apikey:{key_hash}"

    # Fall back to IP address
    client_ip = request.client.host if request.client else "unknown"

    # Check for proxy headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Get the first IP in the chain (original client)
        client_ip = forwarded_for.split(",")[0].strip()

    return f"ip:{client_ip}"
