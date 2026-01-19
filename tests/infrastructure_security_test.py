#!/usr/bin/env python3
"""
Infrastructure Security Tests - TDD Approach
Tests for critical security vulnerabilities that must be fixed
"""

import os
import sys

import pytest

# Add Project_Chimera to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Project_Chimera"))


class TestFlaskCORSSecurity:
    """Test CORS configuration security vulnerabilities"""

    def test_api_server_cors_restricted_origins(self):
        """Test that api_server.py does NOT use wildcard CORS origins"""
        # This test should FAIL before the fix (vulnerable code exists)
        with open("Project_Chimera/api_server.py") as f:
            content = f.read()

        # Check for the vulnerable wildcard pattern
        vulnerable_patterns = [
            'CORS(app, resources={r"/*": {"origins": "*"}}',
            'origins="*"',
            'origins: ["*"]',
            'origins: ["*"]',
        ]

        for pattern in vulnerable_patterns:
            assert pattern not in content, f"VULNERABLE CORS pattern found: {pattern}"

    def test_app_init_cors_restricted_origins(self):
        """Test that app/__init__.py does NOT use wildcard CORS origins"""
        # This test should FAIL before the fix (vulnerable code exists)
        with open("Project_Chimera/app/__init__.py") as f:
            content = f.read()

        # Check for the vulnerable wildcard pattern
        vulnerable_patterns = [
            'cors.init_app(app, resources={r"/*": {"origins": "*"}}',
            'origins="*"',
            'origins: ["*"]',
        ]

        for pattern in vulnerable_patterns:
            assert pattern not in content, f"VULNERABLE CORS pattern found: {pattern}"

    def test_cors_uses_environment_variables(self):
        """Test that CORS configuration uses environment variables"""
        # This test should FAIL before the fix (no env var usage)
        with open("Project_Chimera/api_server.py") as f:
            content = f.read()

        # After fix, should use environment variables for origins
        assert (
            "os.getenv" in content or "ALLOWED_ORIGINS" in content
        ), "CORS should use environment variables for allowed origins"


class TestSecurityHeaders:
    """Test OWASP security headers implementation"""

    def test_security_headers_present(self):
        """Test that OWASP security headers are implemented"""
        # This test should FAIL before security headers are added
        with open("Project_Chimera/api_server.py") as f:
            content = f.read()

        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
        ]

        for header in required_headers:
            assert header in content, f"Missing security header: {header}"


class TestAuthenticationSecurity:
    """Test authentication security improvements"""

    def test_timing_safe_comparison(self):
        """Test that authentication uses timing-safe comparison"""
        # This test should FAIL before fix (uses simple string comparison)
        with open("Project_Chimera/api_server.py") as f:
            content = f.read()

        # Should use secrets.compare_digest() for timing attack protection
        assert (
            "secrets.compare_digest" in content
        ), "Authentication should use timing-safe comparison"

    def test_rate_limiting_implementation(self):
        """Test that rate limiting is implemented"""
        # This test should FAIL before rate limiting is added
        with open("Project_Chimera/api_server.py") as f:
            content = f.read()

        # Should have rate limiting implementation
        assert (
            "limiter" in content or "rate_limit" in content.lower()
        ), "Should implement rate limiting"


class TestDependencySecurity:
    """Test dependency security updates"""

    def test_no_hardcoded_secrets(self):
        """Test that no hardcoded secrets exist"""
        # This test should FAIL if hardcoded secrets are present
        files_to_check = [
            "Project_Chimera/api_server.py",
            "Project_Chimera/app.py",
            "Project_Chimera/app/__init__.py",
        ]

        forbidden_patterns = [
            "chimera_default_key_change_in_production",
            "sk-",  # OpenAI API key pattern (but not in variable names)
            "AIza",  # Google API key pattern
            "ANTHROPIC_API_KEY=",  # Hardcoded Anthropic key assignment
        ]

        for file_path in files_to_check:
            if os.path.exists(file_path):
                with open(file_path) as f:
                    content = f.read()
                    for pattern in forbidden_patterns:
                        assert (
                            pattern not in content
                        ), f"Hardcoded secret '{pattern}' found in {file_path}"


class TestEnvironmentSecurity:
    """Test environment-specific security configurations"""

    def test_production_debug_disabled(self):
        """Test that debug mode is disabled in production"""
        # This test should FAIL if debug mode can be enabled in production
        with open("Project_Chimera/api_server.py") as f:
            content = f.read()

        # Should check environment variable for debug mode
        assert "os.getenv" in content and (
            "DEBUG" in content or "debug" in content
        ), "Should use environment variable to control debug mode"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
