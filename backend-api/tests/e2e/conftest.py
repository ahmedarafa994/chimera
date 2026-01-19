"""
E2E Test Configuration and Fixtures
Provides fixtures for end-to-end testing with Playwright
"""

import pytest
import requests
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="session")
def browser_context():
    """Create a browser context for E2E tests"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        yield context
        context.close()
        browser.close()


@pytest.fixture
def page(browser_context):
    """Create a new page for each test"""
    page = browser_context.new_page()
    yield page
    page.close()


@pytest.fixture(scope="session")
def api_base_url():
    """Base URL for API testing"""
    return "http://localhost:8001"


@pytest.fixture(scope="session")
def frontend_base_url():
    """Base URL for frontend testing"""
    return "http://localhost:3001"


@pytest.fixture(scope="session")
def api_available(api_base_url) -> bool:
    """Check if the backend API is reachable."""
    try:
        response = requests.get(f"{api_base_url}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
def require_api(api_available):
    """Skip E2E API tests when the backend is not running."""
    if not api_available:
        pytest.skip("API is not available")
