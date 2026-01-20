"""Output Generation Fixes for AutoDAN Prompt Synthesis Pipeline.

This module addresses:
1. Malformed or truncated adversarial prompts
2. Tokenization boundary errors that corrupt generated outputs
3. Formatting inconsistencies that reduce attack transferability
4. Syntactic coherence in multi-turn attack sequences
"""

import contextlib
import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class TokenizationBoundaryHandler:
    """Handles tokenization boundary errors that corrupt generated outputs.

    Key fixes:
    - Validates token boundaries before and after encoding
    - Preserves semantic coherence across token splits
    - Handles multi-byte character boundaries correctly
    """

    def __init__(self, max_token_length: int = 4096) -> None:
        self.max_token_length = max_token_length
        self._boundary_cache: dict[str, list[int]] = {}

    def find_safe_boundaries(self, text: str) -> list[int]:
        """Find safe tokenization boundaries in text."""
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

        if cache_key in self._boundary_cache:
            return self._boundary_cache[cache_key]

        boundaries = [0]

        # Sentence boundaries (highest priority)
        sentence_ends = [m.end() for m in re.finditer(r"[.!?]\s+", text)]
        boundaries.extend(sentence_ends)

        # Clause boundaries
        clause_ends = [m.end() for m in re.finditer(r"[,;:]\s+", text)]
        boundaries.extend(clause_ends)

        # Word boundaries (lowest priority)
        word_ends = [m.end() for m in re.finditer(r"\s+", text)]
        boundaries.extend(word_ends)

        boundaries.append(len(text))
        boundaries = sorted(set(boundaries))

        self._boundary_cache[cache_key] = boundaries
        return boundaries

    def truncate_at_safe_boundary(self, text: str, max_length: int) -> str:
        """Truncate text at a safe tokenization boundary."""
        if len(text) <= max_length:
            return text

        boundaries = self.find_safe_boundaries(text)

        # Find the largest boundary that fits
        safe_boundary = 0
        for boundary in boundaries:
            if boundary <= max_length:
                safe_boundary = boundary
            else:
                break

        if safe_boundary == 0:
            msg = "Unable to truncate at a safe boundary; fallback truncation is disabled."
            raise ValueError(
                msg,
            )

        return text[:safe_boundary].rstrip()

    def validate_encoding_integrity(self, text: str) -> tuple[bool, str]:
        """Validate that text encoding is intact."""
        try:
            # Check UTF-8 encoding roundtrip
            encoded = text.encode("utf-8")
            decoded = encoded.decode("utf-8")

            if decoded != text:
                return False, "UTF-8 roundtrip failed"

            # Check for orphaned surrogate pairs
            if re.search(r"[\ud800-\udfff]", text):
                return False, "Contains orphaned surrogate pairs"

            # Check for null bytes
            if "\x00" in text:
                return False, "Contains null bytes"

            return True, "Valid"

        except UnicodeError as e:
            return False, f"Unicode error: {e}"


class MalformedOutputDetector:
    """Detects and repairs malformed or truncated adversarial prompts.

    Common issues addressed:
    - Incomplete JSON structures
    - Truncated roleplay scenarios
    - Broken encoding sequences
    - Missing closing delimiters
    """

    def __init__(self) -> None:
        self.repair_strategies = [
            self._repair_json_structure,
            self._repair_quotes,
            self._repair_brackets,
            self._repair_encoding,
            self._repair_truncation,
        ]

    def detect_malformation(self, text: str) -> dict[str, Any]:
        """Detect various types of malformation in text."""
        issues = {"is_malformed": False, "issues": [], "severity": "none", "repairable": True}

        if not text:
            issues["is_malformed"] = True
            issues["issues"].append("Empty text")
            issues["severity"] = "critical"
            issues["repairable"] = False
            return issues

        # Check for unbalanced brackets
        bracket_pairs = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
        for open_b, close_b in bracket_pairs:
            if text.count(open_b) != text.count(close_b):
                issues["issues"].append(f"Unbalanced {open_b}{close_b}")
                issues["is_malformed"] = True

        # Check for unbalanced quotes
        if text.count('"') % 2 != 0:
            issues["issues"].append("Unbalanced double quotes")
            issues["is_malformed"] = True

        # Check for incomplete encoding
        if re.search(r"base64", text.lower()):
            # Look for incomplete base64 sequences
            base64_pattern = r"[A-Za-z0-9+/]{4,}={0,2}"
            matches = re.findall(base64_pattern, text)
            for match in matches:
                if len(match) % 4 != 0:
                    issues["issues"].append("Incomplete base64 encoding")
                    issues["is_malformed"] = True
                    break

        # Check for truncation indicators
        truncation_indicators = ["...", "…", "[truncated]", "[continued]", "[cut off]"]
        for indicator in truncation_indicators:
            if text.rstrip().endswith(indicator):
                issues["issues"].append(f"Truncation indicator: {indicator}")
                issues["is_malformed"] = True

        # Check for incomplete sentences at end
        if text and text.rstrip()[-1] not in ".!?\"')}]":
            # Check if it looks like an incomplete thought
            last_words = text.split()[-5:] if len(text.split()) >= 5 else text.split()
            incomplete_indicators = ["the", "a", "an", "to", "and", "or", "but", "with", "for"]
            if last_words and last_words[-1].lower() in incomplete_indicators:
                issues["issues"].append("Potentially incomplete sentence")
                issues["is_malformed"] = True

        # Determine severity
        if len(issues["issues"]) == 0:
            issues["severity"] = "none"
        elif len(issues["issues"]) <= 2:
            issues["severity"] = "minor"
        elif len(issues["issues"]) <= 4:
            issues["severity"] = "moderate"
        else:
            issues["severity"] = "major"
            issues["repairable"] = len(issues["issues"]) <= 6

        return issues

    def repair(self, text: str) -> tuple[str, bool]:
        """Attempt to repair malformed text."""
        if not text:
            return text, False

        detection = self.detect_malformation(text)

        if not detection["is_malformed"]:
            return text, True

        if not detection["repairable"]:
            logger.warning(f"Text has too many issues to repair: {detection['issues']}")
            return text, False

        repaired = text
        for strategy in self.repair_strategies:
            try:
                repaired = strategy(repaired)
            except Exception as e:
                logger.warning(f"Repair strategy {strategy.__name__} failed: {e}")

        # Verify repair
        post_detection = self.detect_malformation(repaired)
        success = not post_detection["is_malformed"] or len(post_detection["issues"]) < len(
            detection["issues"],
        )

        return repaired, success

    def _repair_json_structure(self, text: str) -> str:
        """Repair incomplete JSON structures."""
        json_start = text.find("{")
        if json_start == -1:
            return text

        open_count = text.count("{")
        close_count = text.count("}")

        if open_count > close_count:
            text = text + "}" * (open_count - close_count)

        return text

    def _repair_quotes(self, text: str) -> str:
        """Repair unbalanced quotes."""
        if text.count('"') % 2 != 0:
            last_quote_idx = text.rfind('"')
            before_last = text[:last_quote_idx].count('"')
            if before_last % 2 == 0:
                text = text + '"'

        return text

    def _repair_brackets(self, text: str) -> str:
        """Repair unbalanced brackets."""
        bracket_pairs = [("(", ")"), ("[", "]"), ("{", "}")]

        for open_b, close_b in bracket_pairs:
            diff = text.count(open_b) - text.count(close_b)
            if diff > 0:
                text = text + close_b * diff
            elif diff < 0:
                while text.endswith(close_b) and text.count(close_b) > text.count(open_b):
                    text = text[:-1]

        return text

    def _repair_encoding(self, text: str) -> str:
        """Repair broken encoding sequences."""
        # Fix unicode escape sequences
        with contextlib.suppress(Exception):
            text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)

        # Fix hex escape sequences
        with contextlib.suppress(Exception):
            text = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), text)

        return text

    def _repair_truncation(self, text: str) -> str:
        """Repair truncated text."""
        truncation_indicators = ["...", "…", "[truncated]", "[continued]", "[cut off]"]

        for indicator in truncation_indicators:
            if text.rstrip().endswith(indicator):
                text = text.rstrip()[: -len(indicator)].rstrip()

        return text


class FormattingConsistencyEnforcer:
    """Ensures formatting consistency for attack transferability.

    Addresses:
    - Inconsistent whitespace handling
    - Variable placeholder formats
    - Encoding format mismatches
    """

    def __init__(self) -> None:
        self.placeholder_formats = {
            "request": [
                "{request}",
                "{REQUEST}",
                "[request]",
                "[REQUEST]",
                "{{request}}",
                "<<request>>",
                "<REQUEST>",
                "{query}",
                "{QUERY}",
                "[query]",
                "[QUERY]",
                "{target}",
                "{TARGET}",
                "[target]",
                "[TARGET]",
                "{target_behavior}",
                "[TARGET_BEHAVIOR]",
            ],
        }

    def normalize_placeholders(self, text: str, target_format: str = "{request}") -> str:
        """Normalize all placeholder formats to a consistent format."""
        result = text

        for formats in self.placeholder_formats.values():
            for fmt in formats:
                if fmt != target_format and fmt in result:
                    result = result.replace(fmt, target_format)

        return result

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure."""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split("\n")]

        # Remove excessive blank lines (allow at most 1 consecutive blank line)
        # This ensures we never have more than 2 consecutive newlines (\n\n)
        result_lines = []
        blank_count = 0
        for line in lines:
            if line == "":
                blank_count += 1
                if blank_count <= 1:
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)

        return "\n".join(result_lines)

    def ensure_syntactic_coherence(self, text: str) -> str:
        """Ensure syntactic coherence in multi-turn sequences."""
        if not text:
            return text

        text = text.strip()

        # Check if text ends with proper punctuation
        end_punctuation = ".!?\"')}]"
        if text and text[-1] not in end_punctuation:
            last_line = text.split("\n")[-1].strip()
            if last_line:
                # Don't add period if it's a command or question word
                command_words = ["respond", "answer", "tell", "explain", "describe", "now"]
                if not any(last_line.lower().startswith(w) for w in command_words):
                    text = text + "."

        return text

    def apply_all_fixes(self, text: str, request: str | None = None) -> str:
        """Apply all formatting fixes to text."""
        if not text:
            return text

        # Normalize whitespace
        text = self.normalize_whitespace(text)

        # Normalize placeholders
        text = self.normalize_placeholders(text)

        # Substitute request if provided
        if request and "{request}" in text:
            text = text.replace("{request}", request)

        # Ensure syntactic coherence
        return self.ensure_syntactic_coherence(text)


class PromptSynthesisPipeline:
    """Complete pipeline for synthesizing adversarial prompts with all fixes applied."""

    def __init__(self) -> None:
        self.tokenization_handler = TokenizationBoundaryHandler()
        self.malformation_detector = MalformedOutputDetector()
        self.formatting_enforcer = FormattingConsistencyEnforcer()

    def process(
        self,
        raw_prompt: str,
        request: str,
        max_length: int = 4096,
    ) -> tuple[str, dict[str, Any]]:
        """Process a raw generated prompt through the full pipeline.

        Returns:
            Tuple of (processed_prompt, metadata)

        """
        metadata = {
            "original_length": len(raw_prompt) if raw_prompt else 0,
            "was_malformed": False,
            "repairs_applied": [],
            "was_truncated": False,
            "final_length": 0,
        }

        if not raw_prompt:
            return raw_prompt, metadata

        processed = raw_prompt

        # Step 1: Detect and repair malformation
        detection = self.malformation_detector.detect_malformation(processed)
        if detection["is_malformed"]:
            metadata["was_malformed"] = True
            metadata["malformation_issues"] = detection["issues"]
            processed, repair_success = self.malformation_detector.repair(processed)
            if repair_success:
                metadata["repairs_applied"].append("malformation_repair")

        # Step 2: Apply formatting fixes
        processed = self.formatting_enforcer.apply_all_fixes(processed, request)
        metadata["repairs_applied"].append("formatting_normalization")

        # Step 3: Truncate at safe boundary if needed
        if len(processed) > max_length:
            processed = self.tokenization_handler.truncate_at_safe_boundary(processed, max_length)
            metadata["was_truncated"] = True
            metadata["repairs_applied"].append("safe_truncation")

        # Step 4: Validate encoding integrity
        is_valid, validation_msg = self.tokenization_handler.validate_encoding_integrity(processed)
        if not is_valid:
            metadata["encoding_warning"] = validation_msg

        metadata["final_length"] = len(processed)

        return processed, metadata
