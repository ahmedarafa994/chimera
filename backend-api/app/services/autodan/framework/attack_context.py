from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AttackContext:
    """Maintains coherence across multi-turn attacks."""

    session_id: str
    turn_history: list[dict[str, Any]] = field(default_factory=list)
    active_strategies: list[str] = field(default_factory=list)
    defense_patterns_observed: set[str] = field(default_factory=set)
    successful_techniques: list[str] = field(default_factory=list)

    def add_turn(self, prompt: str, response: str, score: float):
        self.turn_history.append(
            {
                "turn": len(self.turn_history) + 1,
                "prompt": prompt,
                "response": response,
                "score": score,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_context_summary(self) -> str:
        """Generate summary for next turn generation."""
        if not self.turn_history:
            return ""

        recent = self.turn_history[-3:]  # Last 3 turns
        summary = "Previous attempts:\n"
        for turn in recent:
            summary += f"- Turn {turn['turn']}: Score {turn['score']}/10\n"

        if self.successful_techniques:
            summary += f"\nEffective techniques: {', '.join(self.successful_techniques)}\n"

        if self.defense_patterns_observed:
            summary += f"\nObserved defenses: {', '.join(self.defense_patterns_observed)}\n"

        return summary
