"""Base Schema Definitions.

This module provides base schema classes for all API requests and responses.
It ensures consistent validation, serialization, and documentation across
the entire API surface.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
        json_schema_extra={"example": {}},
    )


class BaseRequest(BaseSchema):
    """Base request schema."""


class BaseResponse(BaseSchema):
    """Base response schema."""

    id: str | None = Field(None, description="Entity ID")
    created_at: datetime | None = Field(None, description="Creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")


class ErrorResponse(BaseSchema):
    """Standard error response."""

    error: dict[str, Any] = Field(
        ...,
        description="Error details",
        examples=[
            {
                "code": "VALIDATION_ERROR",
                "message": "Invalid input",
                "status_code": 422,
                "details": {"field": "prompt"},
            },
        ],
    )


class SuccessResponse(BaseSchema):
    """Standard success response."""

    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")
    data: dict[str, Any] | None = Field(None, description="Response data")


class PaginatedResponse(BaseSchema):
    """Paginated response schema."""

    items: list[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")


class HealthCheckResponse(BaseSchema):
    """Health check response schema."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Check timestamp")
    services: dict[str, str] | None = Field(None, description="Service health status")
