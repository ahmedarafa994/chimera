import logging
import os

from app import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ChimeraRunner")

# Create the application instance using the factory
# Defaults to 'default' (Development) config unless FLASK_CONFIG env var is set
app = create_app(os.getenv("FLASK_CONFIG", "default"))

if __name__ == "__main__":
    # Configuration matches original start_server.py settings
    host = "0.0.0.0"
    port = 8000

    logger.info(f"Starting server on http://127.0.0.1:{port} (bound to {host})")

    # Run the Flask development server
    # use_reloader=False is kept from original script to prevent double initialization issues
    app.run(host=host, port=port, debug=app.config["DEBUG"], use_reloader=False)
