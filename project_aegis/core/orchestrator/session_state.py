from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

@dataclass
class Interaction:
    turn_id: int
    prompt: str
    response: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class SessionState:
    """
    Maintains the state of a multi-turn attack session.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_model: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    interactions: List[Interaction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_interaction(self, prompt: str, response: str, metrics: Dict[str, float] = None):
        turn_id = len(self.interactions) + 1
        interaction = Interaction(
            turn_id=turn_id,
            prompt=prompt,
            response=response,
            metrics=metrics or {}
        )
        self.interactions.append(interaction)

    def get_context_window(self) -> str:
        # Reconstruct conversation history for model context
        # This is a simplified string representation
        history = ""
        for i in self.interactions:
            history += f"User: {i.prompt}\n"
            history += f"Assistant: {i.response}\n"
        return history
