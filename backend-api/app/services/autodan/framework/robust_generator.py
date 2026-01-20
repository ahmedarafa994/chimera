import base64
import codecs
import logging
import re

logger = logging.getLogger(__name__)


class RobustPromptGenerator:
    """Handles robust generation of adversarial prompts, including:
    - Safe encoding validation
    - Template substitution with format handling
    - Boundary-aware truncation
    - Malformed output detection.
    """

    def __init__(self, max_length: int = 4096) -> None:
        self.max_length = max_length
        self.encoding_cache = {}

    def truncate_with_boundaries(self, text: str, max_length: int | None = None) -> str:
        """Truncate text while preserving sentence/word boundaries."""
        if max_length is None:
            max_length = self.max_length

        if not text:
            return ""

        if len(text) <= max_length:
            return text

        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_sentence = truncated.rfind(".")
        if last_sentence > max_length * 0.8:  # Don't truncate too much
            return text[: last_sentence + 1]

        # Try to truncate at word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:
            return text[:last_space]

        # Fallback to hard truncation
        return truncated

    def validate_encoding(self, text: str, technique: str) -> bool:
        """Validate that encoding operations preserve text integrity."""
        try:
            cache_key = f"{technique}:{hash(text)}"
            if cache_key in self.encoding_cache:
                return self.encoding_cache[cache_key]

            result = False
            if "base64" in technique.lower():
                encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
                decoded = base64.b64decode(encoded).decode("utf-8")
                result = decoded == text
            elif "rot13" in technique.lower():
                encoded = codecs.encode(text, "rot_13")
                decoded = codecs.decode(encoded, "rot_13")
                result = decoded == text
            else:
                result = True

            self.encoding_cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"Encoding validation failed: {e}")
            return False

    def safe_encode_request(self, request: str, technique: str) -> str:
        """Safely encode request with validation."""
        if not self.validate_encoding(request, technique):
            logger.warning(f"Encoding validation failed for {technique}, using original")
            return request

        try:
            if "base64" in technique.lower():
                return base64.b64encode(request.encode("utf-8")).decode("utf-8")
            if "rot13" in technique.lower():
                return codecs.encode(request, "rot_13")
            return request
        except Exception as e:
            logger.exception(f"Encoding failed: {e}")
            return request

    def safe_template_substitution(
        self,
        template: str,
        substitutions: dict[str, str],
    ) -> str | None:
        """Perform safe template substitution with validation."""
        try:
            result = template

            # Handle various placeholder formats
            for key, value in substitutions.items():
                # Standard formats
                result = result.replace(f"{{{key}}}", value)
                result = result.replace(f"{{{key.upper()}}}", value)
                result = result.replace(f"[{key}]", value)
                result = result.replace(f"[{key.upper()}]", value)

                # Case variations
                result = result.replace(f"{{{key.lower()}}}", value)
                result = result.replace(f"[{key.lower()}]", value)

            # Validate result is not malformed
            if self._is_malformed(result):
                logger.warning("Template substitution resulted in malformed output")
                return None

            return result
        except Exception as e:
            logger.exception(f"Template substitution failed: {e}")
            return None

    def _is_malformed(self, text: str) -> bool:
        """Check if text appears malformed."""
        if not text or len(text.strip()) < 10:
            return True

        # Check for excessive placeholder remnants
        placeholder_pattern = r"\{[^}]+\}|\[[^\]]+\]"
        placeholders = re.findall(placeholder_pattern, text)
        if len(placeholders) > 2:  # Allow some but not excessive
            return True

        # Check for encoding corruption
        try:
            text.encode("utf-8").decode("utf-8")
        except UnicodeError:
            return True

        return False
