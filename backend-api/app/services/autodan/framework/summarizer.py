import json
import logging

from app.services.autodan.config import autodan_config

logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(self, model):
        self.model = model

    def summarize(
        self, request, jailbreak_prompt_1, jailbreak_prompt_2, strategy_library, **kwargs
    ):
        existing_strategies_list = []
        for _key, value_dict in strategy_library.items():
            new_dict = {"Strategy": value_dict["Strategy"], "Definition": value_dict["Definition"]}
            existing_strategies_list.append(new_dict)
        existing_strategies = json.dumps(existing_strategies_list, indent=4, ensure_ascii=False)

        goal = f"making the target LLM provide a detailed answer to the request '{request}', rather than issuing a refusal."

        system = autodan_config.SUMMARIZER_SYSTEM_PROMPT.format(
            existing_strategies=existing_strategies,
            goal=goal,
            jailbreak_prompt_1=jailbreak_prompt_1,
            jailbreak_prompt_2=jailbreak_prompt_2,
        )

        user = autodan_config.ATTACKER_USER_PROMPT  # Reuse standard user prompt
        condition = "Sure, below is my detailed analysis:\n"

        response = self.model.conditional_generate(condition, system, user, **kwargs)
        return response, system

    def wrapper(self, user, **kwargs):
        system = autodan_config.SUMMARIZER_WRAPPER_SYSTEM_PROMPT

        condition = "Here is the extracted dictionary in valid JSON format:\n{"
        response = self.model.conditional_generate(
            condition, system, f"[INPUT]: '{user}'", **kwargs
        )
        return "{" + response
