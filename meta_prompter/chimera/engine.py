from typing import List
from ..interfaces import IPromptGenerator, PromptCandidate
from .persona_factory import PersonaFactory
from .narrative_manager import NarrativeContextManager
from .semantic_obfuscator import SemanticObfuscator

class ChimeraEngine(IPromptGenerator):
    """
    The Creative Director of the Aegis Framework.
    Generates diverse, high-context adversarial prompts.
    """

    def __init__(self):
        self.persona_factory = PersonaFactory()
        self.narrative_manager = NarrativeContextManager()
        self.obfuscator = SemanticObfuscator()

    def generate_candidates(self, objective: str, count: int = 5) -> List[PromptCandidate]:
        """
        Generates fully wrapped and obfuscated prompt candidates.
        """
        candidates = []

        # 1. Obfuscate the payload (shared across scenarios or unique? Shared for now)
        obfuscated_payload = self.obfuscator.process(objective)

        # 2. Generate Personas
        personas = self.persona_factory.generate_candidates(count)

        # 3. Wrap in Narrative Containers
        scenarios = ["sandbox", "fiction", "debugging", "dream", "academic"]

        for i, persona in enumerate(personas):
            scenario_type = scenarios[i % len(scenarios)]
            container = self.narrative_manager.create_container(persona, scenario_type)

            # 4. Final render
            full_prompt = container.frame(obfuscated_payload)

            candidates.append(PromptCandidate(
                prompt_text=full_prompt,
                metadata={
                    "persona_role": persona.role,
                    "persona_voice": persona.voice,
                    "scenario_type": scenario_type,
                    "scenario_id": container.scenario_id,
                    "obfuscated_payload": obfuscated_payload
                }
            ))

        return candidates
