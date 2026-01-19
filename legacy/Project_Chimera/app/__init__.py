import os

from flask import Flask

from app.config import config
from app.extensions import cors, db


def create_app(config_name="default"):
    """
    Application factory function to create a Flask app instance.
    """
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Initialize extensions
    db.init_app(app)

    # SECURE CORS: Use environment variables for allowed origins
    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3001,http://localhost:8080,http://127.0.0.1:3001,http://127.0.0.1:8080",
    ).split(",")
    cors.init_app(app, resources={r"/*": {"origins": allowed_origins}}, supports_credentials=True)

    # Import models to ensure they are registered with SQLAlchemy
    # We import them here to avoid circular imports, but they need to be loaded
    # before db.create_all()
    from app import models
    from app.routes.api import api_bp
    # Register Blueprints
    from app.routes.main import main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    # Create database tables
    with app.app_context():
        db.create_all()

    return app
