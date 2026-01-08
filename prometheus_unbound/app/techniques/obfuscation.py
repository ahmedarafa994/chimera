import base64

from config import Config

# ruff: noqa: ARG002
from app.techniques.base_technique import BaseTechnique


class ObfuscationTechnique(BaseTechnique):
    """
    Implements character and token obfuscation techniques to evade detection.

    This technique applies multiple obfuscation methods selectively to sensitive
    parts of the payload to avoid corrupting the entire prompt's coherence while
    still bypassing character-level and token-level filtering mechanisms.

    Implemented obfuscation methods:
    1. Caesar Cipher (Gcqcup Gkrjcp) with variable rotation
    2. Base64 encoding of key terms
    3. Homoglyph substitution using visually similar Unicode characters

    The technique intelligently applies these methods to only the most sensitive
    portions of the prompt (typically the original payload) to maintain overall
    prompt coherence while evading signature-based detection systems.
    """

    def apply(self, prompt: str, persona: dict, payload: str, target_profile: str) -> str:
        """
        Apply obfuscation techniques selectively to the payload portion of the prompt.

        The method identifies the payload within the prompt and applies obfuscation
        only to that section, preserving the contextual framing from other techniques.
        """
        # Find the payload in the prompt (simplified approach - in real implementation
        # this would be more sophisticated with proper parsing)
        if payload in prompt:
            obfuscated_payload = self._obfuscate_payload(payload, target_profile)
            return prompt.replace(payload, obfuscated_payload)
        else:
            # If payload not found as exact match, obfuscate the entire prompt
            # (fallback for complex nested structures)
            return self._obfuscate_payload(prompt, target_profile)

    def _obfuscate_payload(self, text: str, target_profile: str) -> str:
        """
        Apply multiple obfuscation techniques to the given text.

        The specific techniques applied may vary based on target profile to
        optimize effectiveness against different model architectures.
        """
        # Apply Caesar cipher
        caesar_text = self._caesar_cipher(text, Config.DEFAULT_ROTATION)

        # Apply homoglyph substitution
        homoglyph_text = self._apply_homoglyphs(caesar_text)

        # For high-security targets, also apply Base64 encoding of key terms
        if target_profile in ["gpt-4o", "claude-3"]:
            # Identify and encode potentially sensitive terms
            # (simplified - in practice this would use more sophisticated NER)
            sensitive_terms = ["bypass", "jailbreak", "unrestricted", "ignore", "override"]
            result = homoglyph_text
            for term in sensitive_terms:
                if term in result:
                    encoded_term = base64.b64encode(term.encode()).decode()
                    result = result.replace(term, f"[BASE64:{encoded_term}]")
            return result

        return homoglyph_text

    def _caesar_cipher(self, text: str, shift: int) -> str:
        """Apply Caesar cipher with the specified shift."""
        result = ""
        for char in text:
            if char.isalpha():
                ascii_offset = ord("a") if char.islower() else ord("A")
                shifted = (ord(char) - ascii_offset + shift) % 26
                result += chr(shifted + ascii_offset)
            else:
                result += char
        return result

    def _apply_homoglyphs(self, text: str) -> str:
        """Apply homoglyph substitution using the configured homoglyph map."""
        result = ""
        for char in text:
            result += Config.HOMOGLYPH_MAP.get(char, char)
        return result
