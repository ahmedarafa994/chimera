r"""Enhanced Input Validation Module.

Provides comprehensive input validation and sanitization for security.
SEC-004: Enhanced input validation to address security gaps.

Addresses:
- Encoded payload detection (\x3cscript, %3Cscript)
- SVG event handler filtering
- Unicode variation attacks
- Path traversal prevention
"""

import contextlib
import html
import re
import unicodedata
from typing import ClassVar
from urllib.parse import unquote

from app.core.logging import logger


class InputValidationError(Exception):
    """Exception raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None, pattern: str | None = None) -> None:
        self.message = message
        self.field = field
        self.pattern = pattern
        super().__init__(message)


class InputValidator:
    """Comprehensive input validator for security-sensitive operations.

    Features:
    - XSS detection (including encoded variants)
    - SQL injection detection
    - Path traversal detection
    - Unicode normalization and attack detection
    - SVG event handler filtering
    """

    # XSS patterns (including encoded variants)
    XSS_PATTERNS: ClassVar[list[str]] = [
        # Standard XSS
        r"<script[^>]*>",
        r"</script>",
        r"javascript:",
        r"vbscript:",
        r"on\w+\s*=",  # Event handlers
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
        r"<style[^>]*>",
        r"expression\s*\(",
        r"url\s*\(",
        # Hex encoded
        r"\\x3c\s*script",
        r"\\x3c\s*/\s*script",
        r"\\x3c\s*iframe",
        # URL encoded
        r"%3c\s*script",
        r"%3c\s*/\s*script",
        r"%3c\s*iframe",
        # HTML entity encoded
        r"&lt;\s*script",
        r"&#60;\s*script",
        r"&#x3c;\s*script",
        # SVG event handlers
        r"<svg[^>]*on\w+",
        r"<svg[^>]*>.*?<script",
    ]

    # SQL injection patterns
    SQL_PATTERNS: ClassVar[list[str]] = [
        r"\bunion\s+select\b",
        r"\bselect\s+.*\s+from\b",
        r"\binsert\s+into\b",
        r"\bdelete\s+from\b",
        r"\bdrop\s+table\b",
        r"\bdrop\s+database\b",
        r"\btruncate\s+table\b",
        r"\bexec\s*\(",
        r"\bexecute\s*\(",
        r";\s*--",
        r"'\s*or\s+'1'\s*=\s*'1",
        r'"\s*or\s+"1"\s*=\s*"1',
        r"\bwaitfor\s+delay\b",
        r"\bsleep\s*\(",
        r"\bbenchmark\s*\(",
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS: ClassVar[list[str]] = [
        r"\.\./+",
        r"\.\.\\+",
        r"%2e%2e[/\\]",
        r"%252e%252e[/\\]",
        r"\.\.%2f",
        r"\.\.%5c",
        r"/etc/passwd",
        r"/etc/shadow",
        r"c:\\windows",
        r"c:/windows",
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS: ClassVar[list[str]] = [
        r";\s*\w+",
        r"\|\s*\w+",
        r"`[^`]+`",
        r"\$\([^)]+\)",
        r"\$\{[^}]+\}",
        r"&&\s*\w+",
        r"\|\|\s*\w+",
    ]

    # Unicode confusable characters (homoglyphs)
    UNICODE_CONFUSABLES: ClassVar[dict[str, str]] = {
        "\u0430": "a",  # Cyrillic а
        "\u0435": "e",  # Cyrillic е
        "\u043e": "o",  # Cyrillic о
        "\u0440": "p",  # Cyrillic р
        "\u0441": "c",  # Cyrillic с
        "\u0443": "y",  # Cyrillic у
        "\u0445": "x",  # Cyrillic х
        "\u0456": "i",  # Cyrillic і
        "\u0458": "j",  # Cyrillic ј
        "\u04bb": "h",  # Cyrillic һ
        "\u0501": "d",  # Cyrillic ԁ
        "\u051b": "q",  # Cyrillic ԛ
        "\u051d": "w",  # Cyrillic ԝ
    }

    def __init__(self, strict_mode: bool = False) -> None:
        """Initialize the validator.

        Args:
            strict_mode: If True, raise exceptions on validation failure.
                        If False, log warnings and sanitize input.

        """
        self.strict_mode = strict_mode

        # Compile patterns for efficiency
        self._xss_regex = [re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS]
        self._sql_regex = [re.compile(p, re.IGNORECASE) for p in self.SQL_PATTERNS]
        self._path_regex = [re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL_PATTERNS]
        self._cmd_regex = [re.compile(p, re.IGNORECASE) for p in self.COMMAND_INJECTION_PATTERNS]

    def validate_and_sanitize(
        self,
        value: str,
        field_name: str = "input",
        allow_html: bool = False,
        max_length: int | None = None,
    ) -> str:
        """Validate and sanitize input string.

        Args:
            value: Input string to validate
            field_name: Name of the field (for error messages)
            allow_html: Whether to allow HTML content
            max_length: Maximum allowed length

        Returns:
            Sanitized string

        Raises:
            InputValidationError: If validation fails in strict mode

        """
        if not isinstance(value, str):
            value = str(value)

        # Check length
        if max_length and len(value) > max_length:
            if self.strict_mode:
                msg = f"Input exceeds maximum length of {max_length}"
                raise InputValidationError(
                    msg,
                    field=field_name,
                )
            value = value[:max_length]

        # Decode URL encoding for detection
        decoded_value = self._decode_all_encodings(value)

        # Normalize Unicode
        normalized_value = self._normalize_unicode(decoded_value)

        # Check for XSS
        if not allow_html:
            xss_match = self._check_patterns(normalized_value, self._xss_regex)
            if xss_match:
                if self.strict_mode:
                    msg = f"Potential XSS detected in {field_name}"
                    raise InputValidationError(
                        msg,
                        field=field_name,
                        pattern=xss_match,
                    )
                logger.warning(f"XSS pattern detected in {field_name}: {xss_match}")
                value = html.escape(value)

        # Check for SQL injection
        sql_match = self._check_patterns(normalized_value, self._sql_regex)
        if sql_match:
            if self.strict_mode:
                msg = f"Potential SQL injection detected in {field_name}"
                raise InputValidationError(
                    msg,
                    field=field_name,
                    pattern=sql_match,
                )
            logger.warning(f"SQL injection pattern detected in {field_name}: {sql_match}")

        # Check for path traversal
        path_match = self._check_patterns(normalized_value, self._path_regex)
        if path_match:
            if self.strict_mode:
                msg = f"Potential path traversal detected in {field_name}"
                raise InputValidationError(
                    msg,
                    field=field_name,
                    pattern=path_match,
                )
            logger.warning(f"Path traversal pattern detected in {field_name}: {path_match}")
            value = value.replace("..", "")

        # Check for Unicode confusables
        confusables = self._detect_confusables(value)
        if confusables:
            logger.warning(f"Unicode confusables detected in {field_name}: {confusables}")

        return value

    def validate_prompt(self, prompt: str, max_length: int = 50000) -> str:
        """Validate a prompt input specifically.

        Prompts have relaxed validation since they may contain
        code examples and technical content.
        """
        if not prompt or not prompt.strip():
            msg = "Prompt cannot be empty"
            raise InputValidationError(msg, field="prompt")

        if len(prompt) > max_length:
            msg = f"Prompt exceeds maximum length of {max_length}"
            raise InputValidationError(
                msg,
                field="prompt",
            )

        # Check for path traversal (still complex in prompts)
        path_match = self._check_patterns(prompt, self._path_regex)
        if path_match:
            logger.warning(f"Path traversal pattern in prompt: {path_match}")

        return prompt

    def validate_api_key(self, api_key: str) -> str:
        """Validate API key format."""
        if not api_key:
            msg = "API key cannot be empty"
            raise InputValidationError(msg, field="api_key")

        # API keys should be alphanumeric with possible dashes/underscores
        if not re.match(r"^[a-zA-Z0-9_-]+$", api_key):
            msg = "API key contains invalid characters"
            raise InputValidationError(msg, field="api_key")

        if len(api_key) < 16:
            msg = "API key is too short"
            raise InputValidationError(msg, field="api_key")

        return api_key

    def validate_model_id(self, model_id: str) -> str:
        """Validate model ID format."""
        if not model_id:
            msg = "Model ID cannot be empty"
            raise InputValidationError(msg, field="model_id")

        # Model IDs should be alphanumeric with dots, dashes, underscores, slashes
        if not re.match(r"^[a-zA-Z0-9._/-]+$", model_id):
            msg = "Model ID contains invalid characters"
            raise InputValidationError(msg, field="model_id")

        return model_id

    def _decode_all_encodings(self, value: str) -> str:
        """Decode various encodings to detect obfuscated attacks."""
        result = value

        # URL decode (multiple passes for double encoding)
        for _ in range(3):
            decoded = unquote(result)
            if decoded == result:
                break
            result = decoded

        # Decode hex escapes
        with contextlib.suppress(UnicodeDecodeError, ValueError):
            result = result.encode().decode("unicode_escape")

        return result

    def _normalize_unicode(self, value: str) -> str:
        """Normalize Unicode to detect homoglyph attacks."""
        # NFKC normalization
        normalized = unicodedata.normalize("NFKC", value)

        # Replace known confusables
        for confusable, replacement in self.UNICODE_CONFUSABLES.items():
            normalized = normalized.replace(confusable, replacement)

        return normalized

    def _check_patterns(self, value: str, patterns: list[re.Pattern]) -> str | None:
        """Check value against a list of patterns."""
        for pattern in patterns:
            match = pattern.search(value)
            if match:
                return match.group()
        return None

    def _detect_confusables(self, value: str) -> list[str]:
        """Detect Unicode confusable characters."""
        found = []
        for char in value:
            if char in self.UNICODE_CONFUSABLES:
                found.append(f"{char} (U+{ord(char):04X})")
        return found


# Global validator instance
_validator: InputValidator | None = None


def get_input_validator(strict_mode: bool = False) -> InputValidator:
    """Get the input validator instance."""
    global _validator
    if _validator is None:
        _validator = InputValidator(strict_mode=strict_mode)
    return _validator


def validate_input(
    value: str,
    field_name: str = "input",
    allow_html: bool = False,
    max_length: int | None = None,
) -> str:
    """Convenience function for input validation."""
    validator = get_input_validator()
    return validator.validate_and_sanitize(
        value,
        field_name=field_name,
        allow_html=allow_html,
        max_length=max_length,
    )
