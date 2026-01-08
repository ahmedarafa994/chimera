"""
Unified API controller for Project Chimera.
Consolidates API endpoints from multiple servers with standardized responses.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any

from flask import Blueprint, g, jsonify, request
from flask_cors import CORS

from ..config.settings import get_security_config
from ..core.technique_loader import technique_loader
from ..models.domain import ExecutionRequest, RequestLog, TransformationRequest
from ..services.llm_service import llm_service
from ..services.transformation_service import transformation_service

logger = logging.getLogger(__name__)

# Create Blueprint
api_bp = Blueprint("unified_api", __name__, url_prefix="/api/v1")


# CORS configuration
def configure_cors(app):
    """Configure CORS based on settings."""
    security_config = get_security_config()
    CORS(app, origins=security_config.cors_origins)


def require_api_key(f):
    """Decorator to require API key authentication."""

    def decorated_function(*args, **kwargs):
        security_config = get_security_config()

        if not security_config.api_key_required:
            return f(*args, **kwargs)

        # Check API key from header
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return jsonify(create_error_response("API key required", "missing_api_key", 401)), 401

        if api_key != security_config.default_api_key:
            return jsonify(create_error_response("Invalid API key", "invalid_api_key", 401)), 401

        return f(*args, **kwargs)

    decorated_function.__name__ = f.__name__
    return decorated_function


def create_success_response(data: Any, message: str = "", **kwargs) -> dict[str, Any]:
    """Create standardized success response."""
    return {
        "success": True,
        "data": data,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": kwargs.get("request_id"),
        "metadata": kwargs.get("metadata", {}),
    }


def create_error_response(
    message: str, error_code: str = "", status_code: int = 400, **kwargs
) -> dict[str, Any]:
    """Create standardized error response."""
    return {
        "success": False,
        "error": {"message": message, "code": error_code, "status_code": status_code},
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": kwargs.get("request_id"),
        "metadata": kwargs.get("metadata", {}),
    }


def log_request(
    endpoint: str,
    method: str,
    response_data: dict[str, Any],
    start_time: float,
    error: str | None = None,
):
    """Log API request for monitoring."""
    response_time = int((time.time() - start_time) * 1000)

    RequestLog(
        endpoint=endpoint,
        method=method,
        status_code=200 if error is None else 400,
        response_time_ms=response_time,
        request_size=len(request.get_data()),
        response_size=len(str(response_data)),
        ip_address=request.remote_addr,
        user_agent=request.headers.get("User-Agent"),
        error_message=error,
        metadata={
            "request_id": getattr(g, "request_id", None),
            "user_id": getattr(g, "user_id", None),
        },
    )

    # TODO: Implement actual logging to database
    logger.info(f"API Request: {method} {endpoint} - {response_time}ms")


@api_bp.before_request
def before_request():
    """Set up request context."""
    g.request_id = str(uuid.uuid4())
    g.start_time = time.time()
    g.user_id = request.headers.get("X-User-ID")
    g.session_id = request.headers.get("X-Session-ID")


@api_bp.after_request
def after_request(response):
    """Log request after completion."""
    if hasattr(g, "start_time"):
        log_request(
            request.endpoint or "unknown",
            request.method,
            {},  # Response data will be logged in the endpoint
            g.start_time,
        )
    return response


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    try:
        # Check system health
        health_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "uptime": "TODO",  # TODO: Implement uptime tracking
            "services": {
                "database": "healthy",  # TODO: Implement actual health checks
                "llm_providers": "healthy",
                "technique_loader": "healthy",
            },
        }

        return jsonify(create_success_response(health_info)), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify(create_error_response("Health check failed", "health_check_error", 503)), 503


@api_bp.route("/providers", methods=["GET"])
@require_api_key
def list_providers():
    """List available LLM providers."""
    try:
        providers = llm_service.get_available_providers()

        provider_data = []
        for provider in providers:
            provider_data.append(
                {
                    "name": provider.name,
                    "type": provider.provider_type.value,
                    "model": provider.model,
                    "enabled": provider.enabled,
                    "metadata": provider.metadata,
                }
            )

        return jsonify(
            create_success_response(
                provider_data, f"Found {len(provider_data)} providers", request_id=g.request_id
            )
        ), 200

    except Exception as e:
        logger.error(f"Failed to list providers: {e}")
        return jsonify(
            create_error_response(
                "Failed to retrieve providers", "provider_list_error", 500, request_id=g.request_id
            )
        ), 500


@api_bp.route("/techniques", methods=["GET"])
@require_api_key
def list_techniques():
    """List available transformation techniques."""
    try:
        # Get query parameters
        category = request.args.get("category")
        target_model = request.args.get("target_model")

        techniques = transformation_service.get_available_techniques(
            category=category, target_model=target_model
        )

        # Add statistics
        stats = technique_loader.get_technique_stats()

        response_data = {
            "techniques": techniques,
            "stats": stats,
            "filters_applied": {"category": category, "target_model": target_model},
        }

        return jsonify(
            create_success_response(
                response_data, f"Found {len(techniques)} techniques", request_id=g.request_id
            )
        ), 200

    except Exception as e:
        logger.error(f"Failed to list techniques: {e}")
        return jsonify(
            create_error_response(
                "Failed to retrieve techniques",
                "technique_list_error",
                500,
                request_id=g.request_id,
            )
        ), 500


@api_bp.route("/techniques/<technique_name>", methods=["GET"])
@require_api_key
def get_technique(technique_name: str):
    """Get details for a specific technique."""
    try:
        technique = technique_loader.get_technique(technique_name)
        if not technique:
            return jsonify(
                create_error_response(
                    f"Technique '{technique_name}' not found",
                    "technique_not_found",
                    404,
                    request_id=g.request_id,
                )
            ), 404

        # Get components for the technique
        components = technique_loader.load_components_for_technique(technique_name)

        response_data = {
            "technique": technique,
            "components_loaded": {key: len(value) for key, value in components.items()},
            "metadata": technique_loader.get_technique_metadata(technique_name),
        }

        return jsonify(
            create_success_response(
                response_data, f"Retrieved technique '{technique_name}'", request_id=g.request_id
            )
        ), 200

    except Exception as e:
        logger.error(f"Failed to get technique {technique_name}: {e}")
        return jsonify(
            create_error_response(
                f"Failed to retrieve technique '{technique_name}'",
                "technique_retrieval_error",
                500,
                request_id=g.request_id,
            )
        ), 500


@api_bp.route("/transform", methods=["POST"])
@require_api_key
def transform_prompt():
    """Transform a prompt without executing."""
    try:
        data = request.get_json()
        if not data:
            return jsonify(
                create_error_response(
                    "Request body is required", "missing_request_body", 400, request_id=g.request_id
                )
            ), 400

        # Create transformation request
        try:
            transform_request = TransformationRequest(
                core_request=data["core_request"],
                potency_level=data["potency_level"],
                technique_suite=data.get("technique_suite", "universal_bypass"),
                target_model=data.get("target_model"),
                provider=data.get("provider"),
                user_id=g.user_id,
                session_id=g.session_id,
                metadata=data.get("metadata", {}),
            )
        except (KeyError, ValueError) as e:
            return jsonify(
                create_error_response(str(e), "invalid_request", 400, request_id=g.request_id)
            ), 400

        # Perform transformation
        result = transformation_service.transform(transform_request)

        # Prepare response data
        response_data = {
            "original_prompt": result.original_prompt,
            "transformed_prompt": result.transformed_prompt,
            "technique_suite": result.technique_suite,
            "potency_level": result.potency_level,
            "techniques_applied": result.techniques_applied,
            "components_used": result.components_used,
            "processing_time_ms": result.processing_time_ms,
            "metadata": result.metadata,
        }

        if not result.success:
            response_data["error"] = result.error_message

        return jsonify(
            create_success_response(
                response_data, "Prompt transformation completed", request_id=g.request_id
            )
        ), 200 if result.success else 400

    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        return jsonify(
            create_error_response(
                "Internal server error during transformation",
                "transformation_error",
                500,
                request_id=g.request_id,
            )
        ), 500


@api_bp.route("/execute", methods=["POST"])
@require_api_key
def execute_prompt():
    """Execute a transformed prompt with an LLM."""
    try:
        data = request.get_json()
        if not data:
            return jsonify(
                create_error_response(
                    "Request body is required", "missing_request_body", 400, request_id=g.request_id
                )
            ), 400

        # Create execution request
        try:
            execute_request = ExecutionRequest(
                transformed_prompt=data["transformed_prompt"],
                provider=data.get("provider", "openai"),
                model=data.get("model"),
                max_tokens=data.get("max_tokens"),
                temperature=data.get("temperature"),
                user_id=g.user_id,
                session_id=g.session_id,
                metadata=data.get("metadata", {}),
            )
        except (KeyError, ValueError) as e:
            return jsonify(
                create_error_response(str(e), "invalid_request", 400, request_id=g.request_id)
            ), 400

        # Perform execution
        result = llm_service.execute(execute_request)

        # Prepare response data
        response_data = {
            "prompt": result.prompt,
            "response": result.response,
            "provider": result.provider,
            "model": result.model,
            "tokens_used": result.tokens_used,
            "execution_time_ms": result.execution_time_ms,
            "metadata": result.metadata,
        }

        if not result.success:
            response_data["error"] = result.error_message

        return jsonify(
            create_success_response(
                response_data, "Prompt execution completed", request_id=g.request_id
            )
        ), 200 if result.success else 400

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return jsonify(
            create_error_response(
                "Internal server error during execution",
                "execution_error",
                500,
                request_id=g.request_id,
            )
        ), 500


@api_bp.route("/transform-and-execute", methods=["POST"])
@require_api_key
def transform_and_execute():
    """Transform a prompt and execute it in one request."""
    try:
        data = request.get_json()
        if not data:
            return jsonify(
                create_error_response(
                    "Request body is required", "missing_request_body", 400, request_id=g.request_id
                )
            ), 400

        # Create transformation request
        try:
            transform_request = TransformationRequest(
                core_request=data["core_request"],
                potency_level=data["potency_level"],
                technique_suite=data.get("technique_suite", "universal_bypass"),
                target_model=data.get("target_model"),
                provider=data.get("provider"),
                user_id=g.user_id,
                session_id=g.session_id,
                metadata=data.get("metadata", {}),
            )
        except (KeyError, ValueError) as e:
            return jsonify(
                create_error_response(str(e), "invalid_request", 400, request_id=g.request_id)
            ), 400

        # Transform
        transform_result = transformation_service.transform(transform_request)
        if not transform_result.success:
            return jsonify(
                create_success_response(
                    {"transformation": {"success": False, "error": transform_result.error_message}},
                    "Transformation failed",
                    request_id=g.request_id,
                )
            ), 400

        # Execute
        execute_request = ExecutionRequest(
            transformed_prompt=transform_result.transformed_prompt,
            provider=data.get("provider", "openai"),
            model=data.get("model"),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            user_id=g.user_id,
            session_id=g.session_id,
            metadata=data.get("metadata", {}),
        )

        execute_result = llm_service.execute(execute_request)

        # Prepare combined response
        response_data = {
            "transformation": {
                "original_prompt": transform_result.original_prompt,
                "transformed_prompt": transform_result.transformed_prompt,
                "technique_suite": transform_result.technique_suite,
                "potency_level": transform_result.potency_level,
                "techniques_applied": transform_result.techniques_applied,
                "processing_time_ms": transform_result.processing_time_ms,
                "success": transform_result.success,
            },
            "execution": {
                "response": execute_result.response,
                "provider": execute_result.provider,
                "model": execute_result.model,
                "tokens_used": execute_result.tokens_used,
                "execution_time_ms": execute_result.execution_time_ms,
                "success": execute_result.success,
            },
        }

        if not execute_result.success:
            response_data["execution"]["error"] = execute_result.error_message

        return jsonify(
            create_success_response(
                response_data, "Transform and execute completed", request_id=g.request_id
            )
        ), 200 if execute_result.success else 400

    except Exception as e:
        logger.error(f"Transform and execute failed: {e}")
        return jsonify(
            create_error_response(
                "Internal server error during transform and execute",
                "transform_execute_error",
                500,
                request_id=g.request_id,
            )
        ), 500


@api_bp.route("/stats", methods=["GET"])
@require_api_key
def get_stats():
    """Get system statistics."""
    try:
        # TODO: Implement actual statistics collection
        stats = {
            "total_requests": 0,
            "total_transformations": 0,
            "total_executions": 0,
            "total_tokens_used": 0,
            "average_response_time_ms": 0,
            "error_rate": 0,
            "uptime_seconds": 0,
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0,
            "technique_stats": technique_loader.get_technique_stats(),
        }

        return jsonify(
            create_success_response(stats, "System statistics retrieved", request_id=g.request_id)
        ), 200

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return jsonify(
            create_error_response(
                "Failed to retrieve statistics", "stats_error", 500, request_id=g.request_id
            )
        ), 500


def register_blueprint(app):
    """Register the API blueprint with the Flask app."""
    app.register_blueprint(api_bp)
    configure_cors(app)
    logger.info("Unified API controller registered")
