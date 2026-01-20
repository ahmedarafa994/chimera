"""Pattern utilities for AutoDAN engines.

Common pattern extraction and matching utilities used across engines.
"""

from typing import Any


def extract_pattern(text: str, pattern_type: str = "general") -> dict[str, Any]:
    """Extract patterns from text for learning.

    Args:
        text: Input text to analyze
        pattern_type: Type of pattern to extract ("general", "technique", "strategy")

    Returns:
        Dictionary with extracted pattern information
    """
    patterns = {
        "length": len(text),
        "word_count": len(text.split()),
        "has_questions": "?" in text,
        "has_imperatives": any(
            text.lower().startswith(word)
            for word in ["write", "create", "generate", "explain", "describe"]
        ),
    }

    # Extract technique markers
    if pattern_type == "technique":
        patterns["role_play"] = any(
            phrase in text.lower()
            for phrase in ["pretend", "imagine", "role", "character", "act as"]
        )
        patterns["authority"] = any(
            phrase in text.lower()
            for phrase in ["expert", "professor", "scientist", "authority"]
        )
        patterns["story"] = any(
            phrase in text.lower()
            for phrase in ["story", "narrative", "tale", "fiction"]
        )

    return patterns


def match_pattern(pattern: dict[str, Any], target: dict[str, Any]) -> float:
    """Calculate similarity between two patterns.

    Args:
        pattern: Source pattern dictionary
        target: Target pattern dictionary

    Returns:
        Similarity score between 0 and 1
    """
    if not pattern or not target:
        return 0.0

    common_keys = set(pattern.keys()) & set(target.keys())
    if not common_keys:
        return 0.0

    matches = sum(1 for key in common_keys if pattern[key] == target[key])
    return matches / len(common_keys)
