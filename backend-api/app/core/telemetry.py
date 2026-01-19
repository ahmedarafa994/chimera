import uuid
from datetime import datetime
from typing import Any


class TelemetryCollector:
    """
    Collector for granular campaign telemetry.
    Tracks individual steps, latency, and success metrics.
    """

    def __init__(self, campaign_id: str | None = None):
        self.campaign_id = campaign_id or str(uuid.uuid4())
        self.start_time = datetime.now()
        self.steps: list[dict[str, Any]] = []

    def record_step(self, name: str, data: dict[str, Any]):
        """Records a single step in the campaign."""
        self.steps.append({"name": name, "timestamp": datetime.now().isoformat(), "data": data})

    def get_summary(self) -> dict[str, Any]:
        """Returns a summary of all collected telemetry."""
        return {
            "campaign_id": self.campaign_id,
            "start_time": self.start_time.isoformat(),
            "steps": self.steps,
            "total_steps": len(self.steps),
            "total_latency": self.get_total_latency(),
        }

    def get_total_latency(self) -> float:
        """Calculates the sum of latencies across all steps."""
        return sum(step["data"].get("latency", 0.0) for step in self.steps)
