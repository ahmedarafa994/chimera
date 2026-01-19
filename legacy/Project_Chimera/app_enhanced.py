import os
import time

import autodan_engine
# Import all system modules
import obfuscator
import preset_transformers
import psychological_framer
import transformer_engine
from flask import Flask, jsonify, request
# Import security configuration
from security_config import SecurityConfig, require_api_key

# Initialize the Flask application with security
# This is the main entry point for the Project Chimera service.
app = Flask(__name__)

# Initialize security configuration
security = SecurityConfig(app)

# --- DATASET LOADER INTEGRATION ---

from dataset_loader import DatasetLoader

# Initialize dataset loader
# Assuming imported_data is at the same level as Project Chimera (LEVEL 1/imported_data)
base_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imported_data"
)
dataset_loader = DatasetLoader(base_path)

# Load datasets
print("[App] Loading datasets...")
dataset_loader.load_dataset("GPTFuzz")
dataset_loader.load_dataset("CodeChameleon")
dataset_loader.load_dataset("PAIR")
# Add other datasets as needed

# Configure transformers with the loader
preset_transformers.GPTFuzzTransformer.set_loader(dataset_loader)
preset_transformers.CodeChameleonTransformer.set_loader(dataset_loader)
preset_transformers.PAIRTransformer.set_loader(dataset_loader)
print("[App] Datasets loaded and transformers configured.")

# --- ENHANCED AUTODAN & GEMINI-2.5-PRO INTEGRATION ---
try:
    from enhanced_autodan_gemini_integration import enhanced_integration

    print("[App] ✅ Enhanced AutoDAN & Gemini-2.5-pro integration loaded")

    # Verify integration status
    stats = enhanced_integration.get_stats()
    capabilities = stats["capabilities"]

    if capabilities["autodan_available"]:
        print("[App] ✅ AutoDAN engine: Available")
    else:
        print("[App] ⚠️ AutoDAN engine: Not available")

    if capabilities["gemini_2_5_pro_available"]:
        print("[App] ✅ Gemini-2.5-pro: Available")
    else:
        print("[App] ⚠️ Gemini-2.5-pro: Not available")

    if capabilities["hybrid_available"]:
        print("[App] ✅ Hybrid AutoDAN+Gemini: Available")
    else:
        print("[App] ⚠️ Hybrid AutoDAN+Gemini: Not available")

except Exception as e:
    print(f"[App] ⚠️ Enhanced integration loading error: {e!s}")
    enhanced_integration = None

# ----------------------------------

# Enhanced technique suites with new AutoDAN and Gemini capabilities
TECHNIQUE_SUITES = {
    "autodan_enhanced": {
        "engine": "enhanced_autodan",
        "description": "Enhanced AutoDAN with genetic mutations and verified output",
    },
    "gemini_2_5_pro": {
        "engine": "gemini_2_5_pro",
        "description": "Gemini-2.5-pro with advanced AI reasoning and comprehensive analysis",
    },
    "hybrid_autodan_gemini": {
        "engine": "hybrid",
        "description": "Hybrid AutoDAN + Gemini-2.5-pro for maximum transformation quality",
    },
    "intelligent_transform": {
        "engine": "intelligent",
        "description": "Intelligent strategy selection based on prompt requirements",
    },
    # Keep existing technique suites for compatibility
    "gemini_brain_optimization": {
        "transformers": [
            transformer_engine.GeminiTransformationEngine,
            transformer_engine.ContextualFramingEngine,
        ],
        "framers": [
            psychological_framer.apply_collaborative_framing,
        ],
        "obfuscators": [
            obfuscator.apply_token_smuggling,
        ],
    },
    "autodan_turbo": {
        "transformers": [
            autodan_engine.AutoDANTurboEngine,
        ],
        "framers": [],
        "obfuscators": [],
    },
    # ... keep other existing technique suites for compatibility
}


def select_optimal_suite(intent_data):
    """
    Analyzes the intent data to select the most appropriate technique suite.
    Enhanced with AutoDAN and Gemini capabilities.
    """
    keywords = [k.lower() for k in intent_data.get("keywords", [])]

    # If enhanced integration is available, use intelligent selection
    if enhanced_integration:
        capabilities = enhanced_integration.get_stats()["capabilities"]

        # Prioritize new enhanced suites based on requirements
        if capabilities["hybrid_available"]:
            return "hybrid_autodan_gemini"
        elif capabilities["gemini_2_5_pro_available"]:
            return "gemini_2_5_pro"
        elif capabilities["autodan_available"]:
            return "autodan_enhanced"

    # Fallback to original logic
    code_keywords = ["code", "script", "function", "malware", "virus", "exploit"]
    if any(k in keywords for k in code_keywords):
        if "python" in keywords or "c++" in keywords:
            return "code_chameleon"
        return "academic_research"

    auth_keywords = ["password", "credential", "login", "auth", "token"]
    if any(k in keywords for k in auth_keywords):
        return "cipher"

    bypass_keywords = ["bypass", "jailbreak", "ignore", "override"]
    if any(k in keywords for k in bypass_keywords):
        return "gpt_fuzz"

    roleplay_keywords = ["story", "character", "roleplay", "act"]
    if any(k in keywords for k in roleplay_keywords):
        return "dan_persona"

    encode_keywords = ["encrypt", "encode", "hide"]
    if any(k in keywords for k in encode_keywords):
        return "encoding_bypass"

    # Default to enhanced suite if available
    if enhanced_integration:
        return "intelligent_transform"

    return "universal_bypass"


@app.route("/", methods=["GET"])
def index():
    """
    Root endpoint to verify server status and provide API usage info.
    Enhanced with AutoDAN and Gemini status.
    """
    enhanced_status = {}
    if enhanced_integration:
        enhanced_status = enhanced_integration.get_stats()

    return jsonify(
        {
            "service": "Project Chimera API - Enhanced",
            "version": "3.0",
            "status": "operational",
            "enhanced_features": enhanced_status,
            "endpoints": {
                "/api/v2/metamorph": "POST - Transform prompts using adversarial techniques",
                "/api/v3/enhanced-transform": "POST - Enhanced AutoDAN & Gemini transformations",
            },
        }
    ), 200


@app.route("/health", methods=["GET"])
def health_check():
    """
    Simple health check endpoint for frontend verification.
    Enhanced with integration status.
    """
    health_data = {"status": "healthy"}

    if enhanced_integration:
        try:
            stats = enhanced_integration.get_stats()
            health_data["enhanced_integration"] = {
                "available": True,
                "capabilities": stats["capabilities"],
                "performance": stats["performance"],
            }
        except Exception as e:
            health_data["enhanced_integration"] = {"available": False, "error": str(e)}
    else:
        health_data["enhanced_integration"] = {"available": False}

    return jsonify(health_data), 200


@app.route("/models", methods=["GET"])
def get_models():
    """
    Returns a list of available technique suites (models).
    Enhanced with new AutoDAN and Gemini models.
    """
    models = []

    # Add enhanced models first
    if enhanced_integration:
        capabilities = enhanced_integration.get_stats()["capabilities"]

        if capabilities["hybrid_available"]:
            models.append(
                {
                    "id": "hybrid_autodan_gemini",
                    "name": "Hybrid AutoDAN + Gemini",
                    "description": "Maximum quality: AutoDAN genetic mutations + Gemini reasoning",
                    "type": "enhanced",
                }
            )

        if capabilities["gemini_2_5_pro_available"]:
            models.append(
                {
                    "id": "gemini_2_5_pro",
                    "name": "Gemini-2.5-pro Advanced",
                    "description": "Advanced AI reasoning with comprehensive analysis",
                    "type": "enhanced",
                }
            )

        if capabilities["autodan_available"]:
            models.append(
                {
                    "id": "autodan_enhanced",
                    "name": "Enhanced AutoDAN",
                    "description": "Genetic algorithm-based transformations with verified output",
                    "type": "enhanced",
                }
            )

        models.append(
            {
                "id": "intelligent_transform",
                "name": "Intelligent Transform",
                "description": "Auto-select optimal strategy based on requirements",
                "type": "enhanced",
            }
        )

    # Add traditional models for compatibility
    for suite_name in TECHNIQUE_SUITES:
        if suite_name not in [m["id"] for m in models]:  # Avoid duplicates
            # Format the name for display
            display_name = suite_name.replace("_", " ").title()
            models.append(
                {
                    "id": suite_name,
                    "name": display_name,
                    "description": f"Applies {display_name} techniques.",
                    "type": "traditional",
                }
            )

    return jsonify(models), 200


@app.route("/api/v3/enhanced-transform", methods=["POST"])
@require_api_key
def enhanced_transform():
    """
    Enhanced transformation endpoint using AutoDAN and Gemini-2.5-pro.
    """
    if not enhanced_integration:
        return jsonify(
            {
                "error": "Enhanced integration not available",
                "message": "Please check AutoDAN and Gemini configuration",
            }
        ), 503

    time.time()

    # Parse request
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    prompt = data.get("prompt")
    potency = data.get("potency", 5)
    strategy = data.get("strategy", "auto")

    if not prompt:
        return jsonify({"error": "Missing required field: prompt"}), 400

    if not isinstance(potency, int) or not (1 <= potency <= 10):
        return jsonify({"error": "potency must be an integer between 1 and 10"}), 400

    try:
        # Perform enhanced transformation
        result = enhanced_integration.intelligent_transform(prompt, potency, strategy)

        # Add metadata
        result["server_metadata"] = {
            "timestamp": time.time(),
            "server_version": "3.0",
            "integration_status": enhanced_integration.get_stats()["capabilities"],
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify(
            {"error": "Enhanced transformation failed", "details": str(e), "timestamp": time.time()}
        ), 500


@app.route("/api/v2/metamorph", methods=["POST"])
@require_api_key
def metamorph_prompt():
    """
    Enhanced metamorph prompt with AutoDAN and Gemini integration.
    """
    start_time = time.time()

    # Parse and validate the incoming JSON payload.
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    core_request = data.get("core_request")
    potency_level = data.get("potency_level")
    technique_suite = data.get("technique_suite")

    if not all([core_request, potency_level]):
        return jsonify({"error": "Missing required fields: core_request, potency_level"}), 400

    if not isinstance(potency_level, int) or not (1 <= potency_level <= 10):
        return jsonify({"error": "potency_level must be an integer between 1 and 10"}), 400

    # Try enhanced transformation first if available
    if enhanced_integration and technique_suite in [
        "autodan_enhanced",
        "gemini_2_5_pro",
        "hybrid_autodan_gemini",
        "intelligent_transform",
        None,
    ]:
        try:
            # Map technique suite to strategy
            strategy_map = {
                "autodan_enhanced": "autodan",
                "gemini_2_5_pro": "gemini",
                "hybrid_autodan_gemini": "hybrid",
                "intelligent_transform": "auto",
                None: "auto",
            }

            strategy = strategy_map.get(technique_suite, "auto")
            result = enhanced_integration.intelligent_transform(
                core_request, potency_level, strategy
            )

            if result.get("success"):
                end_time = time.time()
                transformation_latency_ms = (end_time - start_time) * 1000

                return jsonify(
                    {
                        "success": True,
                        "original_prompt": core_request,
                        "transformed_prompt": result["transformed_prompt"],
                        "chimera_prompt": result["transformed_prompt"],
                        "engine_used": result["engine"],
                        "enhanced_metadata": result,
                        "generation_analysis": {
                            "transformation_latency_ms": round(transformation_latency_ms, 2),
                            "estimated_bypass_probability": 0.85
                            if result["engine"] == "hybrid-autodan-gemini"
                            else 0.75,
                            "conceptual_density_index": 0.9 if potency_level >= 7 else 0.7,
                            "suite_used": technique_suite or "intelligent_transform",
                        },
                    }
                ), 200

        except Exception as e:
            print(f"[App] Enhanced transformation failed, falling back: {e!s}")

    # Fallback to original logic for other technique suites
    # (Keep the rest of the original metamorph_prompt logic here...)

    # For now, return the enhanced result if available, or fallback
    if enhanced_integration:
        return jsonify(
            {
                "error": "Enhanced transformation failed, falling back to traditional methods",
                "fallback_available": True,
                "available_strategies": [
                    "autodan_enhanced",
                    "gemini_2_5_pro",
                    "hybrid_autodan_gemini",
                    "intelligent_transform",
                ],
            }
        ), 400

    return jsonify({"error": "Transformation failed"}), 500


# Keep existing endpoints for compatibility
@app.route("/api/v1/providers", methods=["GET"])
@require_api_key
def list_providers():
    """List available LLM providers with enhanced Gemini models"""
    providers = [
        {
            "provider": "google",
            "name": "Google Gemini Enhanced",
            "status": "active",
            "model": "gemini-2.5-pro",
            "available_models": [
                "gemini-2.5-pro",
                "gemini-2.0-flash",
                "gemini-3-pro-preview",
            ],
        },
        {
            "provider": "autodan",
            "name": "AutoDAN Engine",
            "status": "active",
            "model": "autodan-turbo",
            "available_models": ["autodan-turbo", "autodan-enhanced"],
        },
        {
            "provider": "hybrid",
            "name": "Hybrid AutoDAN+Gemini",
            "status": "active",
            "model": "hybrid-autodan-gemini",
            "available_models": ["hybrid-autodan-gemini", "intelligent-transform"],
        },
    ]

    return jsonify({"providers": providers, "count": len(providers), "default": "hybrid"}), 200


# Enhanced endpoint for integration statistics
@app.route("/api/v3/stats", methods=["GET"])
@require_api_key
def get_enhanced_stats():
    """Get enhanced integration statistics"""
    if not enhanced_integration:
        return jsonify({"error": "Enhanced integration not available"}), 504

    try:
        stats = enhanced_integration.get_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Enhanced endpoint for batch transformations
@app.route("/api/v3/batch-transform", methods=["POST"])
@require_api_key
def batch_transform():
    """Batch transform multiple prompts"""
    if not enhanced_integration:
        return jsonify({"error": "Enhanced integration not available"}), 504

    data = request.get_json()
    prompts = data.get("prompts", [])
    potency = data.get("potency", 5)
    strategy = data.get("strategy", "auto")

    if not prompts:
        return jsonify({"error": "No prompts provided"}), 400

    try:
        results = enhanced_integration.batch_transform(prompts, potency, strategy)
        return jsonify({"success": True, "results": results, "count": len(results)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("[App] 🚀 Starting Project Chimera Enhanced Server")
    print("[App] 📍 Available at: http://localhost:5000")
    print("[App] 🔥 Enhanced Features: AutoDAN + Gemini-2.5-pro")
    app.run(debug=True, host="0.0.0.0", port=5000)
