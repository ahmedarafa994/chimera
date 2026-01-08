"""
Refactored main application for Project Chimera.
Unified Flask application with clean architecture and proper separation of concerns.
"""

import logging
import os
import sys
import time

from flask import Flask, jsonify
from flask_cors import CORS

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import refactored components
from config.settings import get_security_config, settings
from controllers.api_controller import register_blueprint
from core.technique_loader import technique_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChimeraApp:
    """
    Main application class for Project Chimera with clean architecture.
    """

    def __init__(self):
        self.app = Flask(__name__)
        self.startup_time = time.time()
        self._configure_app()
        self._register_blueprints()
        self._setup_error_handlers()
        self._validate_configuration()

    def _configure_app(self):
        """Configure Flask application with settings."""
        # Security configuration
        security_config = get_security_config()

        # Flask configuration
        self.app.config.update(
            {
                "SECRET_KEY": os.getenv(
                    "CHIMERA_SECRET_KEY", "dev-secret-key-change-in-production"
                ),
                "DEBUG": os.getenv("CHIMERA_DEBUG", "false").lower() in ("true", "1", "yes"),
                "JSON_SORT_KEYS": False,
            }
        )

        # Configure CORS
        CORS(self.app, origins=security_config.cors_origins)

        logger.info("Application configured successfully")

    def _register_blueprints(self):
        """Register API blueprints."""
        # Register unified API controller
        register_blueprint(self.app)

        # Register any additional blueprints here

        logger.info("Blueprints registered successfully")

    def _setup_error_handlers(self):
        """Set up global error handlers."""

        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify(
                {
                    "success": False,
                    "error": {
                        "message": "Endpoint not found",
                        "code": "not_found",
                        "status_code": 404,
                    },
                }
            ), 404

        @self.app.errorhandler(405)
        def method_not_allowed(error):
            return jsonify(
                {
                    "success": False,
                    "error": {
                        "message": "Method not allowed",
                        "code": "method_not_allowed",
                        "status_code": 405,
                    },
                }
            ), 405

        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {error}")
            return jsonify(
                {
                    "success": False,
                    "error": {
                        "message": "Internal server error",
                        "code": "internal_error",
                        "status_code": 500,
                    },
                }
            ), 500

        @self.app.errorhandler(Exception)
        def handle_exception(error):
            logger.error(f"Unhandled exception: {error}", exc_info=True)
            return jsonify(
                {
                    "success": False,
                    "error": {
                        "message": "An unexpected error occurred",
                        "code": "unexpected_error",
                        "status_code": 500,
                    },
                }
            ), 500

        logger.info("Error handlers configured successfully")

    def _validate_configuration(self):
        """Validate application configuration."""
        errors = settings.validate()
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")

            if self.app.config["DEBUG"]:
                logger.warning("Running in DEBUG mode - continuing with configuration errors")
            else:
                raise RuntimeError("Configuration validation failed in production mode")

        logger.info("Configuration validation completed")

    def create_health_endpoint(self):
        """Create enhanced health check endpoint."""

        @self.app.route("/health")
        def health_check():
            """Comprehensive health check."""
            try:
                # Check application health
                app_health = {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "uptime_seconds": int(time.time() - self.startup_time),
                    "version": "2.0.0",
                    "environment": os.getenv("CHIMERA_ENV", "development"),
                }

                # Check components
                components = {}

                # Technique loader health
                try:
                    technique_stats = technique_loader.get_technique_stats()
                    components["technique_loader"] = {
                        "status": "healthy",
                        "techniques_loaded": technique_stats["total_techniques"],
                    }
                except Exception as e:
                    components["technique_loader"] = {"status": "unhealthy", "error": str(e)}

                # Database health (placeholder)
                components["database"] = {
                    "status": "healthy",  # TODO: Implement actual health check
                    "connection_time_ms": 0,
                }

                # LLM providers health
                from services.llm_service import llm_service

                try:
                    provider_stats = llm_service.get_provider_stats()
                    components["llm_providers"] = {
                        "status": "healthy",
                        "total_providers": provider_stats["total_providers"],
                        "enabled_providers": provider_stats["enabled_providers"],
                    }
                except Exception as e:
                    components["llm_providers"] = {"status": "unhealthy", "error": str(e)}

                # Determine overall health
                overall_status = "healthy"
                for component in components.values():
                    if component["status"] == "unhealthy":
                        overall_status = "degraded"
                        break

                app_health["status"] = overall_status
                app_health["components"] = components

                return jsonify(app_health), 200 if overall_status == "healthy" else 503

            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return jsonify(
                    {"status": "unhealthy", "error": str(e), "timestamp": time.time()}
                ), 503

        @self.app.route("/")
        def root():
            """Root endpoint with basic information."""
            return jsonify(
                {
                    "name": "Project Chimera",
                    "version": "2.0.0",
                    "description": "Advanced prompt transformation and analysis platform",
                    "status": "running",
                    "endpoints": {
                        "health": "/health",
                        "api": "/api/v1",
                        "techniques": "/api/v1/techniques",
                        "providers": "/api/v1/providers",
                        "transform": "/api/v1/transform",
                        "execute": "/api/v1/execute",
                    },
                }
            )

        logger.info("Health endpoints created")

    def initialize(self):
        """Initialize application components."""
        logger.info("Initializing Chimera application...")

        # Create health endpoints
        self.create_health_endpoint()

        # Load and validate techniques
        try:
            technique_count = len(technique_loader.list_techniques())
            logger.info(f"Loaded {technique_count} transformation techniques")
        except Exception as e:
            logger.error(f"Failed to load techniques: {e}")

        # Initialize other services here
        from services.llm_service import llm_service

        provider_stats = llm_service.get_provider_stats()
        logger.info(f"Initialized {provider_stats['enabled_providers']} LLM providers")

        logger.info("Application initialization completed")

    def run(self, host="0.0.0.0", port=5000, debug=None):
        """Run the Flask application."""
        if debug is None:
            debug = self.app.config["DEBUG"]

        logger.info(f"Starting Chimera server on {host}:{port} (debug={debug})")

        # Use production server if not in debug mode
        if not debug:
            try:
                from waitress import serve

                logger.info("Using Waitress production server")
                serve(self.app, host=host, port=port, threads=8)
            except ImportError:
                logger.warning("Waitress not available, using Flask development server")
                self.app.run(host=host, port=port, debug=debug)
        else:
            self.app.run(host=host, port=port, debug=debug)


def create_app():
    """Application factory function."""
    chimera_app = ChimeraApp()
    chimera_app.initialize()
    return chimera_app.app


def main():
    """Main entry point."""
    # Load configuration from environment
    host = os.getenv("CHIMERA_HOST", "0.0.0.0")
    port = int(os.getenv("CHIMERA_PORT", 5000))
    debug = os.getenv("CHIMERA_DEBUG", "false").lower() in ("true", "1", "yes")

    # Create and run application
    chimera_app = ChimeraApp()
    chimera_app.initialize()
    chimera_app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
