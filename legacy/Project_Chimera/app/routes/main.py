from datetime import datetime

from flask import Blueprint, jsonify

from app.services.transformer import transformer_service

main_bp = Blueprint("main", __name__)


@main_bp.route("/", methods=["GET"])
def index():
    """Root endpoint"""
    return (
        jsonify(
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
        ),
        200,
    )


@main_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    health_status = transformer_service.get_health_status()

    return (
        jsonify(
            {
                "status": "operational",
                "timestamp": datetime.now().isoformat(),
                "health": health_status,
            }
        ),
        200,
    )
