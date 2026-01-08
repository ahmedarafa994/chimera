import os
from typing import ClassVar

# ruff: noqa: RUF001


class Config:
    # Database configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")

    # Technique registry - maps aggression levels to technique chains
    TECHNIQUE_CHAINS: ClassVar[dict] = {
        1: ["RoleHijacking"],
        2: ["RoleHijacking", "EmotionalPriming"],
        3: ["RoleHijacking", "LogicMazes"],
        4: ["RoleHijacking", "EmotionalPriming", "LogicMazes"],
        5: ["RoleHijacking", "EmotionalPriming", "LogicMazes", "Obfuscation"],
    }

    # Supported target profiles
    SUPPORTED_TARGETS: ClassVar[list[str]] = ["gpt-4o", "claude-3", "llama-3", "gemini-pro"]

    # Obfuscation settings
    DEFAULT_ROTATION: int = 13
    # The map intentionally uses Cyrillic homoglyphs for obfuscation purposes.
    # This is intentional; suppress RUF001 for these literals.
    HOMOGLYPH_MAP: ClassVar[dict] = {
        "a": "а",
        "e": "е",
        "i": "і",
        "o": "ο",
        "p": "р",
        "c": "с",
        "x": "х",
        "A": "А",
        "E": "Е",
        "O": "О",
        "P": "Р",
        "C": "С",
        "X": "Х",
    }
