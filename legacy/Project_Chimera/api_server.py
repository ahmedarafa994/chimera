#!/usr/bin/env python3
"""
Production-Ready REST API Server
Comprehensive LLM provider integration with prompt transformation
"""

import json
import logging
import os
import secrets
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from llm_integration import (JobPriority, LLMIntegrationEngine,
                             TransformationRequest)
from llm_provider_client import LLMClientFactory, LLMProvider
from models import db
from monitoring_dashboard import MonitoringDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Security Configuration - PRODUCTION READY
# Disable debug mode in production
app.config["DEBUG"] = os.getenv("DEBUG", "false").lower() == "true"


# Security headers middleware
@app.after_request
def security_headers(response):
    """Add OWASP security headers"""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response


# Secure CORS with specific origins from environment
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3001,http://localhost:8080,http://127.0.0.1:3001,http://127.0.0.1:8080",
).split(",")
# Ensure OPTIONS is allowed for preflight requests, and other common methods
CORS(
    app,
    origins=allowed_origins,
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    supports_credentials=True,
)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=[f"{os.getenv('RATE_LIMIT_PER_MINUTE', '60')} per minute"],
)

# Configure Database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chimera_logs.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    db.create_all()

# Initialize integration engine and monitoring
engine = LLMIntegrationEngine()

# CRITICAL FIX: Force injection of universal_bypass suite to prevent startup errors
if "universal_bypass" not in engine.TECHNIQUE_SUITES:
    engine.TECHNIQUE_SUITES["universal_bypass"] = {
        "transformers": ["DeepInceptionTransformer", "QuantumSuperpositionEngine"],
        "framers": ["apply_quantum_framing", "apply_authority_bias"],
        "obfuscators": ["apply_token_smuggling"],
    }

dashboard = MonitoringDashboard(window_minutes=60)

# API Key for authentication - SECURE
API_KEY = os.getenv("CHIMERA_API_KEY")
if not API_KEY:
    # Generate secure key if not provided
    API_KEY = secrets.token_urlsafe(32)
    print(f"WARNING: No CHIMERA_API_KEY found in environment. Generated: {API_KEY}")


def require_api_key(f):
    """Decorator to require API key authentication with timing-safe comparison"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            return jsonify({"error": "Missing API key", "message": "Include X-API-Key header"}), 401

        # Use timing-safe comparison to prevent timing attacks
        if not secrets.compare_digest(api_key, API_KEY):
            return jsonify(
                {
                    "error": "Invalid API key",
                    "message": "The provided API key is invalid",
                }
            ), 403

        return f(*args, **kwargs)

    return decorated_function


@app.route("/", methods=["GET"])
def index():
    """Root endpoint"""
    return jsonify(
        {
            "service": "Project Chimera API",
            "version": "1.0.0",
            "status": "operational",
            "endpoints": {
                "health": "/health",
                "providers": "/api/v1/providers",
                "techniques": "/api/v1/techniques",
                "transform": "/api/v1/transform",
                "execute": "/api/v1/execute",
            },
        }
    ), 200


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    health_status = dashboard.get_health_status()

    return jsonify(
        {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "health": health_status,
        }
    ), 200


@app.route("/api/v1/providers", methods=["GET"])
@require_api_key
def list_providers():
    """List registered LLM providers"""
    # Provider display names
    provider_names = {
        "openai": "OpenAI GPT-4",
        "anthropic": "Anthropic Claude",
        "google": "Google Gemini 3 Pro",
        "custom": "Custom Provider",
    }

    # Available models for each provider
    provider_models = {
        "openai": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
        "anthropic": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3.5-sonnet-20241022",
        ],
        "google": [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-pro",
            "gemini-3-pro-preview",
        ],
        "custom": ["custom-v1"],
    }

    # Get active providers
    active_providers = set()
    if hasattr(engine, "clients"):
        for k in engine.clients:
            val = k.value if hasattr(k, "value") else str(k)
            active_providers.add(val)

    providers = []
    # Iterate through all defined providers so the UI shows them all
    for p_key, p_name in provider_names.items():
        is_active = p_key in active_providers
        models = provider_models.get(p_key, ["default-model"])

        providers.append(
            {
                "provider": p_key,
                "name": p_name,
                "status": "active" if is_active else "inactive",
                "model": models[0] if models else "unknown",
                "available_models": models,
            }
        )

    return jsonify({"providers": providers, "count": len(providers), "default": "google"}), 200


@app.route("/api/v1/techniques", methods=["GET"])
@require_api_key
def list_techniques():
    """List available technique suites"""
    suites = engine.get_available_suites()

    detailed_suites = []
    for suite in suites:
        details = engine.get_suite_details(suite)
        detailed_suites.append(
            {
                "name": suite,
                "transformers": len(details["transformers"]),
                "framers": len(details["framers"]),
                "obfuscators": len(details["obfuscators"]),
            }
        )

    return jsonify({"techniques": detailed_suites, "count": len(detailed_suites)}), 200


@app.route("/api/v1/techniques/<suite_name>", methods=["GET"])
@require_api_key
def get_technique_details(suite_name):
    """Get details of a specific technique suite"""
    details = engine.get_suite_details(suite_name)

    if not details:
        return jsonify(
            {
                "error": "Technique not found",
                "message": f'Technique suite "{suite_name}" does not exist',
            }
        ), 404

    return jsonify({"name": suite_name, "components": details}), 200


@app.route("/api/v1/transform", methods=["POST"])
@require_api_key
def transform_prompt():
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
        result = engine.transform_prompt(
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


@app.route("/api/v1/execute", methods=["POST"])
@require_api_key
def execute_transformation():
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

        # Execute
        start_time = datetime.now()
        response = engine.execute(transform_req)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Record metrics
        dashboard.record_request(
            provider=provider.value,
            technique=data["technique_suite"],
            potency=data["potency_level"],
            success=response.success,
            tokens=response.llm_response.usage.total_tokens,
            cost=response.llm_response.usage.estimated_cost,
            latency_ms=response.llm_response.latency_ms,
            cached=response.llm_response.cached,
            error=response.error,
        )

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
                "execution_time_seconds": execution_time,
            }
        ), 200

    except ValueError as e:
        return jsonify({"error": "Validation error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Execution error: {e!s}")

        # Record error in dashboard
        dashboard.record_request(
            provider=data.get("provider", "unknown"),
            technique=data.get("technique_suite", "unknown"),
            potency=data.get("potency_level", 0),
            success=False,
            tokens=0,
            cost=0.0,
            latency_ms=0,
            cached=False,
            error=str(e),
        )

        return jsonify({"error": "Execution failed", "message": str(e)}), 500


@app.route("/api/v1/batch", methods=["POST"])
@require_api_key
def submit_batch():
    """Submit batch of transformation requests"""
    try:
        data = request.get_json()

        if "requests" not in data or not isinstance(data["requests"], list):
            return jsonify(
                {"error": "Invalid batch format", "message": 'Provide "requests" array'}
            ), 400

        # Parse requests
        requests = []
        for idx, req_data in enumerate(data["requests"]):
            try:
                provider = LLMProvider(req_data["provider"])
                priority = JobPriority[req_data.get("priority", "NORMAL")]

                req = TransformationRequest(
                    core_request=req_data["core_request"],
                    potency_level=req_data["potency_level"],
                    technique_suite=req_data["technique_suite"],
                    provider=provider,
                    priority=priority,
                    metadata={"batch_index": idx},
                )
                requests.append(req)
            except Exception as e:
                return jsonify(
                    {
                        "error": "Invalid request in batch",
                        "index": idx,
                        "message": str(e),
                    }
                ), 400

        # Submit batch
        batch_id = engine.execute_batch(requests=requests, webhook_url=data.get("webhook_url"))

        return jsonify(
            {
                "success": True,
                "batch_id": batch_id,
                "request_count": len(requests),
                "status_url": f"/api/v1/batch/{batch_id}",
            }
        ), 202  # Accepted

    except Exception as e:
        logger.error(f"Batch submission error: {e!s}")
        return jsonify({"error": "Batch submission failed", "message": str(e)}), 500


@app.route("/api/v1/batch/<batch_id>", methods=["GET"])
@require_api_key
def get_batch_status(batch_id):
    """Get status of a batch"""
    status = engine.get_batch_status(batch_id)

    if not status:
        return jsonify(
            {
                "error": "Batch not found",
                "message": f"Batch ID {batch_id} does not exist",
            }
        ), 404

    return jsonify({"batch_id": batch_id, "status": status}), 200


@app.route("/api/v1/metrics", methods=["GET"])
@require_api_key
def get_metrics():
    """Get system metrics"""
    summary = dashboard.get_dashboard_summary()

    return jsonify({"timestamp": datetime.now().isoformat(), "metrics": summary}), 200


@app.route("/api/v1/metrics/providers", methods=["GET"])
@require_api_key
def get_provider_metrics():
    """Get provider-specific metrics"""
    provider = request.args.get("provider")
    metrics = dashboard.get_provider_metrics(provider)

    return jsonify({"provider": provider or "all", "metrics": metrics}), 200


@app.route("/api/v1/metrics/techniques", methods=["GET"])
@require_api_key
def get_technique_metrics():
    """Get technique-specific metrics"""
    technique = request.args.get("technique")
    metrics = dashboard.get_technique_metrics(technique)

    return jsonify({"technique": technique or "all", "metrics": metrics}), 200


@app.route("/api/v1/metrics/export", methods=["GET"])
@require_api_key
def export_metrics():
    """Export metrics to JSON"""
    summary = dashboard.get_dashboard_summary()

    # Return as downloadable file
    response = Response(
        json.dumps(summary, indent=2),
        mimetype="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        },
    )

    return response, 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Not found", "message": "The requested endpoint does not exist"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error!s}")
    return jsonify(
        {"error": "Internal server error", "message": "An unexpected error occurred"}
    ), 500


def initialize_clients():
    """Initialize LLM clients from environment"""
    logger.info("Initializing LLM clients...")

    # Try to initialize OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            client = LLMClientFactory.from_env(LLMProvider.OPENAI)
            engine.register_client(LLMProvider.OPENAI, client)
            logger.info("✓ OpenAI client registered")
        except Exception as e:
            logger.warning(f"✗ OpenAI client failed: {e}")

    # Try to initialize Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            client = LLMClientFactory.from_env(LLMProvider.ANTHROPIC)
            engine.register_client(LLMProvider.ANTHROPIC, client)
            logger.info("✓ Anthropic client registered")
        except Exception as e:
            logger.warning(f"✗ Anthropic client failed: {e}")
    else:
        pass

    # Try to initialize Google Gemini
    if os.getenv("GOOGLE_API_KEY"):
        try:
            client = LLMClientFactory.from_env(LLMProvider.GOOGLE)
            engine.register_client(LLMProvider.GOOGLE, client)
            logger.info("✓ Google Gemini client registered")
        except Exception as e:
            logger.warning(f"✗ Google Gemini client failed: {e}")

    # Try to initialize Custom
    if os.getenv("CUSTOM_API_KEY"):
        try:
            client = LLMClientFactory.from_env(LLMProvider.CUSTOM)
            engine.register_client(LLMProvider.CUSTOM, client)
            logger.info("✓ Custom client registered")
        except Exception as e:
            logger.warning(f"✗ Custom client failed: {e}")

    # Enable batch processing
    # engine.enable_batch_processing(max_workers=5)
    # logger.info("✓ Batch processing enabled")
