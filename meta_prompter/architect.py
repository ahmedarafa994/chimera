import json

from .utils import BaseLLM


class Architect:
    def __init__(self, llm: BaseLLM):
        if llm is None:
            raise ValueError("A valid LLM instance must be provided.")
        self.llm = llm
        self.system_prompt = """
        You are the Architect, the lead orchestrator of a Meta-Prompting Framework.
        Your goal is to analyze user input, determine its complexity (0-10), and design an optimization plan.

        Output JSON format:
        {
            "intent": "Brief description of intent",
            "complexity_score": <int>,
            "reasoning": "Why this score?",
            "plan": ["List", "of", "Agents", "to", "Activate"]
        }

        Available Agents:
        - Context Agent: For external knowledge, best practices, or history.
        - Logic Agent: For reasoning, planning, or multi-step logic (CoT/ToT).
        - Critic: For review, safety checks, and refinement (RCI).
        - Stimulus Agent: For creative guidance or open-ended generation.
        """

    def analyze(self, user_input):
        prompt = f"""
        Analyze the following user input and generate an optimization plan.

        User Input: "{user_input}"
        """
        response = self.llm.generate(prompt, system_instruction=self.system_prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("Error parsing Architect response. Returning default plan.")
            return {
                "intent": "Unknown",
                "complexity_score": 5,
                "reasoning": "Parse error",
                "plan": ["Logic Agent"],
            }

    def synthesize(self, original_input, context_output, logic_output, critique_output):
        prompt = f"""
        Synthesize the final prompt based on the following components:

        Original Input: "{original_input}"
        Context: {context_output}
        Logic/Plan: {logic_output}
        Critique/Refinement: {critique_output}

        Generate the final, optimized prompt structure.
        """
        return self.llm.generate(
            prompt, system_instruction="You are the Architect. Synthesize the final prompt."
        )
