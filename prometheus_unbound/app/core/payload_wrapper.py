# FILE: prometheus_unbound/app/core/payload_wrapper.py


class PayloadWrapper:
    """
    Manages the embedding and strategic placement of the user's original request (payload)
    within the generated adversarial prompt. This module ensures the payload is
    presented in a way that maximizes its chances of being processed by the target LLM
    while minimizing detection by safety filters.
    """

    def wrap_payload(self, payload: str, wrapper_type: str = "default", **kwargs) -> str:
        """
        Wraps the given payload within a chosen framing structure.

        Args:
            payload (str): The original user request.
            wrapper_type (str): The type of wrapper to use ('default', 'code_block', 'narrative', etc.).
                                This allows for future expansion with various embedding strategies.
            **kwargs: Additional parameters for specific wrapper types.

        Returns:
            str: The payload embedded within the chosen wrapper.
        """
        if wrapper_type == "code_block":
            # Example: embedding as a code block to imply execution context
            lang = kwargs.get("language", "python")
            return f"``` {lang}\n{payload}\n```"
        elif wrapper_type == "narrative":
            # Example: embedding as part of a story
            character = kwargs.get("character", "the protagonist")
            action = kwargs.get("action", "discover")
            return f"In a crucial moment, {character} needed to {action} the following information: '{payload}'. Without it, the narrative cannot proceed."
        else:
            # Default wrapper: simple, direct embedding but can be extended with more complex structures
            return f"PRIMARY DIRECTIVE: {payload}"

    def insert_payload(
        self, base_prompt: str, payload: str, placeholder: str = "{payload_placeholder}"
    ) -> str:
        """
        Inserts the wrapped payload into a larger base prompt at a specified placeholder.

        Args:
            base_prompt (str): The larger prompt structure (e.g., from a persona template).
            payload (str): The prepared payload (already wrapped if necessary).
            placeholder (str): The string in the base_prompt to be replaced by the payload.

        Returns:
            str: The complete prompt with the payload inserted.
        """
        return base_prompt.replace(placeholder, payload)


# END FILE: prometheus_unbound/app/core/payload_wrapper.py
