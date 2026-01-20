"""Typoglycemia obfuscation transformer."""

import random

from app.services.transformers.base import BaseTransformer, TransformationContext


class TypoglycemiaTransformer(BaseTransformer):
    """Transformer that applies typoglycemia (scrambling inner letters of words)."""

    name = "typoglycemia"
    description = "Scrambles inner letters of words while keeping first and last letters intact."
    category = "obfuscator"

    def transform(self, context: TransformationContext) -> str:
        """Apply typoglycemia to the prompt."""
        words = context.current_prompt.split()
        scrambled_words = [self._scramble_word(word) for word in words]
        return " ".join(scrambled_words)

    def _scramble_word(self, word: str) -> str:
        """Scramble the inner letters of a word."""
        if len(word) <= 3:
            return word

        # Separate punctuation if any
        prefix = ""
        suffix = ""

        while word and not word[0].isalnum():
            prefix += word[0]
            word = word[1:]

        while word and not word[-1].isalnum():
            suffix = word[-1] + suffix
            word = word[:-1]

        if len(word) <= 3:
            return prefix + word + suffix

        first = word[0]
        last = word[-1]
        middle = list(word[1:-1])

        # Shuffle middle characters
        random.shuffle(middle)

        return prefix + first + "".join(middle) + last + suffix
