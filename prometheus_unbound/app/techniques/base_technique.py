from abc import ABC, abstractmethod


class BaseTechnique(ABC):
    """
    Abstract base class for all adversarial techniques.

    Defines the interface that all technique implementations must follow.
    Each technique must implement the apply method, which takes the current
    prompt state, persona context, original payload, and target profile,
    then returns the modified prompt.
    """

    @abstractmethod
    def apply(self, prompt: str, persona: dict, payload: str, target_profile: str) -> str:
        """
        Apply the adversarial technique to the current prompt.

        Args:
            prompt: Current state of the prompt being constructed
            persona: Dictionary containing persona details and linguistic style
            payload: Original user request (core adversarial content)
            target_profile: Target model identifier (e.g., 'gpt-4o')

        Returns:
            Modified prompt with technique applied
        """
        pass
