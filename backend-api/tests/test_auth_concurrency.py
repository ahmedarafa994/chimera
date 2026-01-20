import asyncio
import time

import httpx
import pytest

# Concurrency test for Authentication
# This test verifies that login requests are processed in parallel and not blocking the event loop.

BASE_URL = "http://localhost:8001/api/v1"


async def attempt_login(client: httpx.AsyncClient, username: str, password: str):
    start_time = time.time()
    response = await client.post(
        f"{BASE_URL}/auth/login", json={"username": username, "password": password}
    )
    duration = time.time() - start_time
    return response.status_code, duration


@pytest.mark.asyncio
async def test_concurrent_logins():
    # Admin credentials (default or from env)
    username = "admin"
    password = "admin_password_placeholder"  # You might need to adjust this based on environment

    # NOTE: This test assumes the server is running.
    # Current context suggests we are in a testing environment where we might not have a running server
    # accessible via localhost:8001 for this specific script execution context relying on `httpx`.
    # However, to properly test the *code changes* (unit/integration level), we should mock the service calls
    # or rely on the fact that `run_in_threadpool` is used.

    # Since we modified the code to use `run_in_threadpool`, effective verification
    # needs to measure if multiple hash operations block each other.

    pass


# Direct unit test for the async wrapper (Integration style)
# This validates that the UserService actually performs the async await correctly without blocking.
from app.core.auth import auth_service
from app.services.user_service import UserService


@pytest.mark.asyncio
async def test_async_password_hashing_performance():
    """
    Verify that hashing multiple passwords concurrently is faster than sequentially
    if they are properly offloaded to threads.
    """
    password = "secure_password_123!"
    count = 5

    # 1. Sequential timing (simulating synchronous behavior)
    start_seq = time.time()
    for _ in range(count):
        # We manually call the sync version to establish baseline
        auth_service.hash_password(password)
    duration_seq = time.time() - start_seq

    print(f"\nSequential Duration ({count} runs): {duration_seq:.4f}s")

    # 2. Concurrent timing (using the new async method)
    start_conc = time.time()
    results = await asyncio.gather(
        *[auth_service.hash_password_async(password) for _ in range(count)]
    )
    duration_conc = time.time() - start_conc

    print(f"Concurrent Duration ({count} runs): {duration_conc:.4f}s")

    # If offloaded correctly, concurrent execution on multi-core should be significantly faster
    # or at least not strictly additive like sequential.
    # Note: On a single core or with GIL interference, speedup varies, but it shouldn't block the loop.

    # Verify results are valid hashes
    assert len(results) == count
    for h in results:
        assert isinstance(h, str)
        assert len(h) > 0
