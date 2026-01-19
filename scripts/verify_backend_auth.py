import asyncio
import logging
import sys

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8001"


async def verify_backend():
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Test Health
        try:
            logger.info(f"Testing {BASE_URL}/health...")
            resp = await client.get(f"{BASE_URL}/health")
            if resp.status_code == 200:
                logger.info("Health check PASSED")
            else:
                logger.error(f"Health check FAILED: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Health check ERROR: {e}")
            return False

        # 2. Test Login (Env Admin)
        try:
            logger.info("Testing Login (Env Admin)...")
            # Try default admin credentials or whatever is configured in .env
            # Assuming admin/admin or admin/change_me for dev
            # Check requirements.txt or env for hints?
            # We'll try the one from the logs: admin@chimera.local
            # But wait, env admin uses CHIMERA_ADMIN_USER which defaults to "admin"

            payload = {
                "username": "admin",
                "password": "change_me_123",  # A guess, but we mostly want to see if it responds, even with 401
            }

            resp = await client.post(f"{BASE_URL}/api/v1/auth/login", json=payload)
            logger.info(f"Login Response: {resp.status_code}")

            if resp.status_code in [200, 401, 403]:
                # If we get a response, the server is handling the request (not hanging)
                logger.info(f"Login endpoint is RESPONSIVE (Status: {resp.status_code})")
                return True
            else:
                logger.error(f"Login endpoint UNEXPECTED STATUS: {resp.status_code} {resp.text}")
                return False

        except Exception as e:
            logger.error(f"Login request ERROR: {type(e).__name__}: {e!s}")
            import traceback

            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = asyncio.run(verify_backend())
    sys.exit(0 if success else 1)
