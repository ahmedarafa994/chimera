import re
import string


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while preserving abbreviations."""
    # Simple sentence splitting that handles common abbreviations
    sentence_endings = re.compile(r"[.!?]+")
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def contains_sensitive_terms(text: str, sensitive_terms: list[str]) -> bool:
    """Check if text contains any of the specified sensitive terms."""
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in sensitive_terms)


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Extract important keywords from text."""
    # Remove punctuation and split into words
    translator = str.maketrans("", "", string.punctuation)
    words = text.translate(translator).lower().split()

    # Filter out common stop words (simplified list)
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
    }
    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    # Return unique keywords up to max_keywords
    return list(dict.fromkeys(keywords))[:max_keywords]


def calculate_complexity_score(text: str) -> float:
    """Calculate a complexity score based on sentence length and vocabulary diversity."""
    sentences = split_into_sentences(text)
    if not sentences:
        return 0.0

    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    unique_words = len(set(text.lower().split()))
    total_words = len(text.split())
    lexical_diversity = unique_words / total_words if total_words > 0 else 0

    # Simple weighted score
    return (avg_sentence_length * 0.6) + (lexical_diversity * 100 * 0.4)
