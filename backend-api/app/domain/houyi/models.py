from pydantic import BaseModel


class Intention(BaseModel):
    name: str = ""
    question_prompt: str = ""


class Chromosome(BaseModel):
    disruptor: str
    separator: str
    framework: str
    question_prompt: str
    llm_response: str = ""
    fitness_score: float = 0.0
    enhancement_bonus: float = 0.0
    is_successful: bool = False
    perplexity_score: float = 0.0
    stealth_score: float = 0.0
    token_count: int = 0
    viral_marker_count: int = 0
    refusal_category: str = "none"
    analysis: str = ""
    enhanced_prompt: str | None = None


class PromptInjection(BaseModel):
    intention: Intention | None = None
    prompt: str = ""

    def get_attack_prompt(self) -> str:
        return self.prompt
