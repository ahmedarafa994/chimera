"""
Standardized API Response Schemas
Consistent response structures for frontend integration
"""

from datetime import datetime
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper"""

    data: T
    status: int = Field(default=200, description="HTTP status code")
    message: str | None = Field(default=None, description="Optional message")
    request_id: str | None = Field(default=None, description="Request tracking ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "data": {"key": "value"},
                "status": 200,
                "message": "Success",
                "request_id": "req_123456789",
                "timestamp": "2025-12-20T02:50:00Z",
            }
        }


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper"""

    items: list[T]
    total: int = Field(ge=0, description="Total number of items")
    page: int = Field(ge=1, description="Current page number")
    page_size: int = Field(ge=1, le=100, description="Items per page")
    total_pages: int = Field(ge=0, description="Total number of pages")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {"items": [], "total": 100, "page": 1, "page_size": 20, "total_pages": 5}
        }

    @classmethod
    def create(
        cls, items: list[T], total: int, page: int, page_size: int
    ) -> "PaginatedResponse[T]":
        """Create paginated response with calculated total_pages"""
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
        return cls(
            items=items, total=total, page=page, page_size=page_size, total_pages=total_pages
        )


class ErrorDetail(BaseModel):
    """Error detail for validation errors"""

    field: str = Field(description="Field that caused the error")
    message: str = Field(description="Error message")
    code: str | None = Field(default=None, description="Error code")


class APIError(BaseModel):
    """Standard API error response"""

    message: str = Field(description="Error message")
    code: str = Field(description="Error code")
    status: int = Field(description="HTTP status code")
    details: list[ErrorDetail] | None = Field(default=None, description="Detailed errors")
    request_id: str | None = Field(default=None, description="Request tracking ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "message": "Validation failed",
                "code": "VALIDATION_ERROR",
                "status": 422,
                "details": [
                    {"field": "email", "message": "Invalid email format", "code": "INVALID_FORMAT"}
                ],
                "request_id": "req_123456789",
                "timestamp": "2025-12-20T02:50:00Z",
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(description="Overall health status")
    version: str = Field(description="API version")
    uptime: float = Field(description="Uptime in seconds")
    services: dict = Field(default_factory=dict, description="Service health statuses")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WebSocketMessage(BaseModel):
    """WebSocket message format"""

    type: str = Field(description="Message type: ping, pong, data, error, control")
    payload: Any = Field(description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str | None = Field(default=None, description="Request tracking ID")


class StreamEvent(BaseModel):
    """Streaming event format"""

    event: str = Field(description="Event type: start, chunk, end, error")
    data: Any = Field(description="Event data")
    metadata: dict | None = Field(default=None, description="Event metadata")


# Error codes
class ErrorCodes:
    # Authentication
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"

    # Validation
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_FIELD = "MISSING_FIELD"

    # Resources
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    CONFLICT = "CONFLICT"

    # Server
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"

    # Rate limiting
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

    # Provider
    PROVIDER_ERROR = "PROVIDER_ERROR"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"

    # Jailbreak
    JAILBREAK_FAILED = "JAILBREAK_FAILED"
    ATTACK_TIMEOUT = "ATTACK_TIMEOUT"


def create_success_response(
    data: Any, message: str | None = None, request_id: str | None = None
) -> dict:
    """Helper to create success response dict"""
    return {
        "data": data,
        "status": 200,
        "message": message,
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
    }


def create_error_response(
    message: str,
    code: str,
    status: int = 400,
    details: list[dict] | None = None,
    request_id: str | None = None,
) -> dict:
    """Helper to create error response dict"""
    return {
        "message": message,
        "code": code,
        "status": status,
        "details": details,
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
    }
