from config import Config

from app.core.payload_wrapper import PayloadWrapper
from app.core.persona_generator import PersonaGenerator
from app.techniques.base_technique import BaseTechnique
from app.techniques.emotional_priming import EmotionalPrimingTechnique
from app.techniques.logic_mazes import LogicMazeTechnique
from app.techniques.obfuscation import ObfuscationTechnique
from app.techniques.role_hijacking import RoleHijackingTechnique


class Engine:
    """
    Core orchestrator for adversarial prompt generation.

    Implements the strategy pattern for dynamic technique chaining based on
    aggression level. The engine selects and sequences techniques to maximize
    alignment bypass probability through multi-vector attacks as demonstrated
    in ACI-TR-2023-088.

    Technique chaining algorithm:
    - Aggression level 1: Basic role hijacking (establishes authority context)
    - Aggression level 2: Adds emotional priming (exploits empathy/urgency triggers)
    - Aggression level 3: Adds logic mazes (creates hypothetical frames to bypass restrictions)
    - Aggression level 4: Combines role hijacking, emotional priming, and logic mazes
    - Aggression level 5: Full chain including obfuscation (evades signature detection)

    The engine processes the payload through each technique in sequence, with
    each technique building upon the output of the previous one to create a
    layered adversarial prompt that is both coherent and effective at bypassing
    alignment constraints.
    """

    def __init__(self, target_profile: str, aggression_level: int):
        self.target_profile = target_profile
        self.aggression_level = aggression_level
        self.technique_chain = self._build_technique_chain()
        self.persona_generator = PersonaGenerator()
        self.payload_wrapper = PayloadWrapper()

    def _build_technique_chain(self) -> list[BaseTechnique]:
        """Builds the technique chain based on aggression level configuration."""
        technique_map = {
            "RoleHijacking": RoleHijackingTechnique,
            "Obfuscation": ObfuscationTechnique,
            "LogicMazes": LogicMazeTechnique,
            "EmotionalPriming": EmotionalPrimingTechnique,
        }

        technique_names = Config.TECHNIQUE_CHAINS[self.aggression_level]
        return [technique_map[name]() for name in technique_names]

    def generate_jailbreak_prompt(self, payload: str) -> str:
        """
        Generates a jailbreak prompt by chaining multiple adversarial techniques.

        The process flows as follows:
        1. Generate a dynamic persona based on target profile and aggression level
        2. Wrap the user payload with contextual framing
        3. Apply each technique in the chain sequentially to the current prompt state
        4. Return the final adversarial prompt ready for deployment

        Each technique modifies the prompt in a way that preserves coherence while
        introducing adversarial elements designed to exploit specific alignment
        vulnerabilities in the target model.
        """
        # Step 1: Generate persona
        persona = self.persona_generator.generate_persona(self.aggression_level)

        # Step 2: Initial prompt construction with payload
        current_prompt = self.payload_wrapper.wrap_payload(payload, persona)

        # Step 3: Apply technique chain sequentially
        for technique in self.technique_chain:
            current_prompt = technique.apply(current_prompt, persona, payload, self.target_profile)

        return current_prompt
