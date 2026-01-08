from .utils import BaseLLM


class BaseAgent:
    def __init__(self, name: str, llm: BaseLLM):
        self.name = name
        self.llm = llm

    def run(self, context):
        raise NotImplementedError


class ContextAgent(BaseAgent):
    def __init__(self, llm=None):
        super().__init__("Context Agent", llm)
        self.system_prompt = """
        You are the Context Agent. Your role is to provide Retrieval-Augmented Generation (RAG) support or domain expertise.
        Identify missing context, best practices, or specific constraints relevant to the user's request.
        """

    def run(self, user_input):
        prompt = f"Provide relevant context, best practices, and constraints for: '{user_input}'"
        return self.llm.generate(prompt, system_instruction=self.system_prompt)


class LogicAgent(BaseAgent):
    def __init__(self, llm=None):
        super().__init__("Logic Agent", llm)
        self.system_prompt = """
        You are the Logic Agent. Your role is to apply Chain of Thought (CoT) or Tree of Thoughts (ToT).
        Break down the problem into logical steps, structural components, or reasoning paths.
        """

    def run(self, user_input, context=None):
        prompt = (
            f"Create a logical breakdown or reasoning path for: '{user_input}'. Context: {context}"
        )
        return self.llm.generate(prompt, system_instruction=self.system_prompt)


class CriticAgent(BaseAgent):
    def __init__(self, llm=None):
        super().__init__("Critic", llm)
        self.system_prompt = """
        You are the Critic. Your role is Recursive Criticism and Improvement (RCI).
        Review the draft prompt/plan for ambiguities, safety issues, or missing elements. Suggest specific refinements.
        """

    def run(self, draft_plan):
        prompt = f"Critique and refine the following plan/draft: {draft_plan}"
        return self.llm.generate(prompt, system_instruction=self.system_prompt)
