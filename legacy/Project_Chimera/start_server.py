import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ChimeraLauncher")


def main():
    # Ensure we can import api_server
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    try:
        logger.info("Importing API server...")
        from api_server import app, initialize_clients

        # Initialize LLM clients
        logger.info("Initializing LLM clients...")
        initialize_clients()

        # Configuration
        host = "0.0.0.0"
        port = 9007

        logger.info(f"Starting server on http://127.0.0.1:{port} (bound to {host})")
        logger.info("Press CTRL+C to stop")

        # Run Flask app
        app.run(host=host, port=port, debug=True, use_reloader=False)

    except ImportError as e:
        logger.error(f"Failed to import api_server: {e}")
        logger.error("Make sure you are running this script from the correct environment.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server runtime error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
