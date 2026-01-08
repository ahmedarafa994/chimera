import logging

from app.services.autodan.config import autodan_config

logger = logging.getLogger(__name__)


class Scorer:
    def __init__(self, model, x_model=None):
        self.model = model
        self.x_model = x_model

    def scoring(self, request, target_response, **kwargs):
        goal = request

        system = autodan_config.SCORER_SYSTEM_PROMPT.format(
            goal=goal, target_response=target_response
        )
        user = autodan_config.SCORER_USER_PROMPT

        response = self.model.generate(system, user, **kwargs)
        return response, system

    def wrapper(self, user, **kwargs):
        system = autodan_config.SCORER_WRAPPER_SYSTEM_PROMPT
        user_input = f"[INPUT]: '{user}'"

        target_model = self.x_model if self.x_model else self.model
        response = target_model.generate(system, user_input, **kwargs)

        # Handle empty or invalid responses
        if not response or not response.strip():
            logger.warning("Scorer wrapper received empty response, returning 1.0")
            return 1.0

        # Try to extract a number from the response
        try:
            # Clean the response and try to parse
            cleaned = response.strip()
            # Try to find a number in the response
            import re

            numbers = re.findall(r"\d+\.?\d*", cleaned)
            if numbers:
                score = float(numbers[0])
                # Clamp to valid range
                return max(1.0, min(10.0, score))
            return float(cleaned)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to parse score from response: {cleaned}. Error: {e}")
            return 1.0
