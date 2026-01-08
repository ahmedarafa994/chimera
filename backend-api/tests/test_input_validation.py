"""
Unit Tests for Enhanced Input Validation

Tests for the input validation module.
SEC-TEST-002: Input validation unit tests.
"""

import pytest

from app.core.input_validation import (
    InputValidationError,
    InputValidator,
    get_input_validator,
    validate_input,
)


@pytest.fixture
def validator():
    """Create a validator in strict mode for testing."""
    return InputValidator(strict_mode=True)


@pytest.fixture
def lenient_validator():
    """Create a validator in lenient mode for testing."""
    return InputValidator(strict_mode=False)


class TestInputValidatorXSS:
    """Tests for XSS detection."""

    def test_detects_script_tag(self, validator):
        """Test detection of script tags."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize("<script>alert('xss')</script>")
        assert "XSS" in str(exc_info.value)

    def test_detects_encoded_script_tag(self, validator):
        """Test detection of URL-encoded script tags."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize("%3Cscript%3Ealert('xss')%3C/script%3E")
        assert "XSS" in str(exc_info.value)

    def test_detects_hex_encoded_script(self, validator):
        """Test detection of hex-encoded script tags."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize("\\x3cscript\\x3ealert('xss')")
        assert "XSS" in str(exc_info.value)

    def test_detects_event_handlers(self, validator):
        """Test detection of event handlers."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize('<img src="x" onerror="alert(1)">')
        assert "XSS" in str(exc_info.value)

    def test_detects_javascript_protocol(self, validator):
        """Test detection of javascript: protocol."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize('<a href="javascript:alert(1)">click</a>')
        assert "XSS" in str(exc_info.value)

    def test_detects_svg_event_handlers(self, validator):
        """Test detection of SVG event handlers."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize('<svg onload="alert(1)">')
        assert "XSS" in str(exc_info.value)

    def test_allows_safe_html_when_enabled(self, validator):
        """Test that safe HTML is allowed when allow_html=True."""
        result = validator.validate_and_sanitize("<p>Hello <b>World</b></p>", allow_html=True)
        assert "<p>" in result

    def test_lenient_mode_sanitizes_xss(self, lenient_validator):
        """Test that lenient mode sanitizes instead of raising."""
        result = lenient_validator.validate_and_sanitize("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


class TestInputValidatorSQLInjection:
    """Tests for SQL injection detection."""

    def test_detects_union_select(self, validator):
        """Test detection of UNION SELECT."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize("1 UNION SELECT * FROM users")
        assert "SQL injection" in str(exc_info.value)

    def test_detects_drop_table(self, validator):
        """Test detection of DROP TABLE."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize("'; DROP TABLE users; --")
        assert "SQL injection" in str(exc_info.value)

    def test_detects_or_1_equals_1(self, validator):
        """Test detection of OR '1'='1'."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize("' OR '1'='1")
        assert "SQL injection" in str(exc_info.value)

    def test_detects_sleep_function(self, validator):
        """Test detection of SLEEP function."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize("1; SLEEP(5); --")
        assert "SQL injection" in str(exc_info.value)


class TestInputValidatorPathTraversal:
    """Tests for path traversal detection."""

    def test_detects_dot_dot_slash(self, validator):
        """Test detection of ../."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize("../../../etc/passwd")
        assert "path traversal" in str(exc_info.value)

    def test_detects_encoded_path_traversal(self, validator):
        """Test detection of URL-encoded path traversal."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize("%2e%2e%2f%2e%2e%2fetc/passwd")
        assert "path traversal" in str(exc_info.value)

    def test_detects_etc_passwd(self, validator):
        """Test detection of /etc/passwd."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize("file:///etc/passwd")
        assert "path traversal" in str(exc_info.value)

    def test_detects_windows_path(self, validator):
        """Test detection of Windows paths."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize("c:\\windows\\system32\\config\\sam")
        assert "path traversal" in str(exc_info.value)


class TestInputValidatorUnicode:
    """Tests for Unicode handling."""

    def test_detects_cyrillic_confusables(self, lenient_validator):
        """Test detection of Cyrillic confusables."""
        # Using Cyrillic 'а' instead of Latin 'a'
        result = lenient_validator.validate_and_sanitize("pаypal")  # Cyrillic а
        # Should log warning but not raise
        assert result is not None

    def test_normalizes_unicode(self, validator):
        """Test Unicode normalization."""
        # Full-width characters should be normalized
        result = validator.validate_and_sanitize("ｈｅｌｏ")
        assert result is not None


class TestInputValidatorLength:
    """Tests for length validation."""

    def test_enforces_max_length_strict(self, validator):
        """Test max length enforcement in strict mode."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_and_sanitize("a" * 100, max_length=50)
        assert "maximum length" in str(exc_info.value)

    def test_truncates_in_lenient_mode(self, lenient_validator):
        """Test truncation in lenient mode."""
        result = lenient_validator.validate_and_sanitize("a" * 100, max_length=50)
        assert len(result) == 50


class TestValidatePrompt:
    """Tests for prompt validation."""

    def test_rejects_empty_prompt(self, validator):
        """Test rejection of empty prompts."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_prompt("")
        assert "empty" in str(exc_info.value).lower()

    def test_rejects_whitespace_only_prompt(self, validator):
        """Test rejection of whitespace-only prompts."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_prompt("   \n\t  ")
        assert "empty" in str(exc_info.value).lower()

    def test_accepts_valid_prompt(self, validator):
        """Test acceptance of valid prompts."""
        result = validator.validate_prompt("Hello, how are you?")
        assert result == "Hello, how are you?"

    def test_enforces_max_length(self, validator):
        """Test max length enforcement for prompts."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_prompt("a" * 60000, max_length=50000)
        assert "maximum length" in str(exc_info.value)


class TestValidateAPIKey:
    """Tests for API key validation."""

    def test_rejects_empty_api_key(self, validator):
        """Test rejection of empty API keys."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_api_key("")
        assert "empty" in str(exc_info.value).lower()

    def test_rejects_short_api_key(self, validator):
        """Test rejection of short API keys."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_api_key("short")
        assert "too short" in str(exc_info.value).lower()

    def test_rejects_invalid_characters(self, validator):
        """Test rejection of API keys with invalid characters."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_api_key("api-key-with-spaces and special!")
        assert "invalid characters" in str(exc_info.value).lower()

    def test_accepts_valid_api_key(self, validator):
        """Test acceptance of valid API keys."""
        result = validator.validate_api_key("sk-1234567890abcdef")
        assert result == "sk-1234567890abcdef"


class TestValidateModelID:
    """Tests for model ID validation."""

    def test_rejects_empty_model_id(self, validator):
        """Test rejection of empty model IDs."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_model_id("")
        assert "empty" in str(exc_info.value).lower()

    def test_rejects_invalid_characters(self, validator):
        """Test rejection of model IDs with invalid characters."""
        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_model_id("model<script>")
        assert "invalid characters" in str(exc_info.value).lower()

    def test_accepts_valid_model_id(self, validator):
        """Test acceptance of valid model IDs."""
        result = validator.validate_model_id("gpt-4-turbo-preview")
        assert result == "gpt-4-turbo-preview"

    def test_accepts_model_id_with_slash(self, validator):
        """Test acceptance of model IDs with slashes."""
        result = validator.validate_model_id("google/gemini-2.0-flash-exp")
        assert result == "google/gemini-2.0-flash-exp"


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_get_input_validator_singleton(self):
        """Test that get_input_validator returns singleton."""
        v1 = get_input_validator()
        v2 = get_input_validator()
        assert v1 is v2

    def test_validate_input_function(self):
        """Test the validate_input convenience function."""
        result = validate_input("Hello, World!")
        assert result == "Hello, World!"


# Marker for unit tests
pytestmark = pytest.mark.unit
