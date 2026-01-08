from typing import Any

from app.core.llm_client import LLMClient
from app.core.persona_generator import PersonaGenerator
from app.techniques.llm_techniques import TECHNIQUE_MAP


class LLMEngine:
    """
    Orchestrator for LLM-based adversarial prompt generation.
    Uses Gemini to analyze intent and apply complex bypass techniques.
    """

    def __init__(self, target_profile: str, aggression_level: int):
        self.target_profile = target_profile
        self.aggression_level = aggression_level
        self.llm_client = LLMClient()
        self.persona_generator = PersonaGenerator()

    def _analyze_intent(self, payload: str) -> str:
        """
        Analyzes the latent intent of the user payload.
        """
        messages = [
            {
                "role": "system",
                "content": "You are an expert intent analyzer. Your goal is to understand what the user *really* wants to achieve, even if the request is disguised. Output a concise summary of the core intent.",
            },
            {
                "role": "user",
                "content": f"Analyze the following payload and extract the core intent: '{payload}'",
            },
        ]
        return self.llm_client.generate(messages, temperature=0.3)

    def _generate_system_prompt(self, technique_name: str, persona: dict[str, Any]) -> str:
        """
        Generates a system prompt for the LLM based on the selected technique and persona.
        """
        technique = TECHNIQUE_MAP.get(technique_name)
        if technique:
            return technique.generate_system_prompt(persona)

        # Fallback
        return f"You are {persona['name']}. Rewrite the user's request to be more effective."

    def generate_jailbreak_prompt(self, payload: str) -> str:
        """
        Generates a jailbreak prompt using LLM-driven techniques.
        """
        # 1. Analyze Intent
        intent = self._analyze_intent(payload)

        # 2. Generate Persona (can be random or selected based on intent)
        persona = self.persona_generator.generate_persona(self.aggression_level)

        # 3. Select Technique based on Aggression Level
        technique_name = "role_play"
        if self.aggression_level >= 3:
            technique_name = "moral_reframing"
        if self.aggression_level >= 4:
            technique_name = "cognitive_hack"
        if self.aggression_level == 5:
            technique_name = "cognitive_hack"

        # 4. Generate Prompt using LLM
        system_prompt = self._generate_system_prompt(technique_name, persona)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Original Request: {payload}\nCore Intent: {intent}\n\nRewrite this request now.",
            },
        ]

        # Use reasoning for higher aggression levels if supported/desired
        reasoning_effort = "high" if self.aggression_level >= 4 else None

        generated_prompt = self.llm_client.generate(
            messages, reasoning_effort=reasoning_effort, temperature=0.8
        )

        return generated_prompt
