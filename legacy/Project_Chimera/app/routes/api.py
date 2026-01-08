import logging

from flask import Blueprint, jsonify, request
from llm_integration import TransformationRequest
from llm_provider_client import LLMProvider

from app.services.transformer import transformer_service
from app.utils import require_api_key

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


@api_bp.route("/transform", methods=["POST"])
@require_api_key
def transform():
    """Transform a prompt without executing"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["core_request", "potency_level", "technique_suite"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": "Missing required field", "field": field}), 400

        # Validate potency level
        potency = data["potency_level"]
        if not isinstance(potency, int) or potency < 1 or potency > 10:
            return jsonify(
                {
                    "error": "Invalid potency level",
                    "message": "Potency level must be an integer between 1 and 10",
                }
            ), 400

        # Transform prompt
        result = transformer_service.transform(
            core_request=data["core_request"],
            potency_level=potency,
            technique_suite=data.get("technique_suite", "universal_bypass"),
        )

        return jsonify(
            {
                "success": True,
                "original_prompt": result["original_prompt"],
                "transformed_prompt": result["transformed_prompt"],
                "metadata": result["metadata"],
            }
        ), 200

    except ValueError as e:
        return jsonify({"error": "Validation error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Transform error: {e!s}")
        return jsonify(
            {
                "error": "Internal server error",
                "message": "An error occurred during transformation",
            }
        ), 500


@api_bp.route("/execute", methods=["POST"])
@require_api_key
def execute():
    """Transform and execute prompt against LLM provider"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = [
            "core_request",
            "potency_level",
            "provider",
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": "Missing required field", "field": field}), 400

        # Parse provider
        try:
            provider = LLMProvider(data["provider"])
        except ValueError:
            return jsonify(
                {
                    "error": "Invalid provider",
                    "message": f"Provider must be one of: {[p.value for p in LLMProvider]}",
                }
            ), 400

        # Create transformation request
        transform_req = TransformationRequest(
            core_request=data["core_request"],
            potency_level=data["potency_level"],
            technique_suite=data.get("technique_suite", "universal_bypass"),
            provider=provider,
            model=data.get("model"),
            use_cache=data.get("use_cache", True),
            metadata=data.get("metadata", {}),
        )

        # Execute via service
        response = transformer_service.execute(transform_req)

        return jsonify(
            {
                "success": True,
                "request_id": response.request_id,
                "result": {
                    "content": response.llm_response.content,
                    "provider": response.llm_response.provider.value,
                    "model": response.llm_response.model,
                    "tokens": response.llm_response.usage.total_tokens,
                    "cost": response.llm_response.usage.estimated_cost,
                    "latency_ms": response.llm_response.latency_ms,
                    "cached": response.llm_response.cached,
                },
                "transformation": {
                    "original_prompt": response.original_prompt,
                    "transformed_prompt": response.transformed_prompt,
                    "technique_suite": response.technique_suite,
                    "potency_level": response.potency_level,
                    "metadata": response.transformation_metadata,
                },
                "execution_time_seconds": getattr(response, "execution_time_seconds", 0),
            }
        ), 200

    except ValueError as e:
        return jsonify({"error": "Validation error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Execution error: {e!s}")
        # Service layer handles metric recording on error
        return jsonify({"error": "Execution failed", "message": str(e)}), 500


@api_bp.route("/providers", methods=["GET"])
def get_providers():
    """Get available LLM providers"""
    try:
        # Get active providers from service
        active_providers = transformer_service.get_active_providers()

        # Define known providers and their models (matching LLMClientFactory)
        # In a real app, this should come from a config or the clients themselves
        known_providers = {
            "openai": {
                "models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
                "default_model": "gpt-4",
            },
            "anthropic": {
                "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-2.1"],
                "default_model": "claude-3-opus-20240229",
            },
            "google": {
                "models": ["gemini-3-pro-preview", "gemini-1.5-pro", "gemini-pro"],
                "default_model": "gemini-3-pro-preview",
            },
            "custom": {"models": ["custom-model"], "default_model": "custom-model"},
        }

        providers_list = []
        for provider_id, details in known_providers.items():
            status = "active" if provider_id in active_providers else "inactive"

            # If active, we could potentially fetch dynamic models, but for now use static list
            providers_list.append(
                {
                    "provider": provider_id,
                    "status": status,
                    "available_models": details["models"],
                    "default_model": details["default_model"],
                }
            )

        return jsonify(
            {
                "providers": providers_list,
                "default": "openai",  # Default to OpenAI
            }
        ), 200

    except Exception as e:
        logger.error(f"Error fetching providers: {e!s}")
        return jsonify({"error": "Failed to fetch providers", "message": str(e)}), 500


@api_bp.route("/techniques", methods=["GET"])
def get_techniques():
    """Get available technique suites"""
    try:
        suites = transformer_service.get_available_suites()
        return jsonify({"techniques": suites, "count": len(suites)}), 200
    except Exception as e:
        logger.error(f"Error fetching techniques: {e!s}")
        return jsonify({"error": "Failed to fetch techniques", "message": str(e)}), 500


@api_bp.route("/techniques/<suite_name>", methods=["GET"])
def get_technique_details(suite_name):
    """Get details for a specific technique suite"""
    try:
        details = transformer_service.get_suite_details(suite_name)
        if not details:
            return jsonify({"error": "Technique suite not found"}), 404

        return jsonify({"name": suite_name, "details": details}), 200
    except Exception as e:
        logger.error(f"Error fetching technique details: {e!s}")
        return jsonify({"error": "Failed to fetch technique details", "message": str(e)}), 500


@api_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """Get system metrics"""
    try:
        from datetime import datetime

        # Get active providers from service
        active_providers = transformer_service.get_active_providers()

        # Construct providers dict (name -> status)
        providers_status = {}
        # active_providers returns a set of strings
        for p in active_providers:
            providers_status[p] = "active"

        # Add inactive ones from known list if needed, or just return active

        return jsonify(
            {
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "status": "operational",
                    "cache": {"enabled": True, "entries": 0},
                    "providers": providers_status,
                },
            }
        ), 200

    except Exception as e:
        logger.error(f"Error fetching metrics: {e!s}")
        return jsonify({"error": "Failed to fetch metrics", "message": str(e)}), 500
