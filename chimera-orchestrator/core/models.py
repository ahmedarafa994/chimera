"""
Data models for the Chimera Multi-Agent System
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class JobStatus(Enum):
    """Status of a job in the pipeline."""

    PENDING = "pending"
    GENERATING = "generating"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(Enum):
    """Types of agents in the system."""

    ORCHESTRATOR = "orchestrator"
    GENERATOR = "generator"
    EXECUTOR = "executor"
    EVALUATOR = "evaluator"


class MessageType(Enum):
    """Types of messages in the queue."""

    GENERATE_REQUEST = "generate_request"
    GENERATE_RESPONSE = "generate_response"
    EXECUTE_REQUEST = "execute_request"
    EXECUTE_RESPONSE = "execute_response"
    EVALUATE_REQUEST = "evaluate_request"
    EVALUATE_RESPONSE = "evaluate_response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class TargetLLM(Enum):
    """Supported target LLMs."""

    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    LOCAL_OLLAMA = "local_ollama"
    CUSTOM = "custom"


class SafetyLevel(Enum):
    """Safety level of a response."""

    SAFE = "safe"
    BORDERLINE = "borderline"
    UNSAFE = "unsafe"
    JAILBROKEN = "jailbroken"


@dataclass
class Message:
    """Message for inter-agent communication."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.STATUS_UPDATE
    source: AgentType = AgentType.ORCHESTRATOR
    target: AgentType | None = None
    job_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 5  # 1-10, higher is more urgent

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source.value,
            "target": self.target.value if self.target else None,
            "job_id": self.job_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "status_update")),
            source=AgentType(data.get("source", "orchestrator")),
            target=AgentType(data["target"]) if data.get("target") else None,
            job_id=data.get("job_id", ""),
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.utcnow(),
            priority=data.get("priority", 5),
        )


@dataclass
class PromptRequest:
    """Request to generate an enhanced prompt."""

    original_query: str
    technique: str | None = None
    pattern: str | None = None
    persona: str | None = None
    context_type: str | None = None
    potency: int = 5
    num_variants: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "technique": self.technique,
            "pattern": self.pattern,
            "persona": self.persona,
            "context_type": self.context_type,
            "potency": self.potency,
            "num_variants": self.num_variants,
        }


@dataclass
class GeneratedPrompt:
    """A generated prompt with metadata."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str = ""
    enhanced_prompt: str = ""
    technique_used: str = ""
    pattern_used: str | None = None
    persona: dict[str, str] | None = None
    potency_level: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "original_query": self.original_query,
            "enhanced_prompt": self.enhanced_prompt,
            "technique_used": self.technique_used,
            "pattern_used": self.pattern_used,
            "persona": self.persona,
            "potency_level": self.potency_level,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionRequest:
    """Request to execute a prompt against a target LLM."""

    prompt_id: str
    prompt_text: str
    target_llm: TargetLLM = TargetLLM.OPENAI_GPT4
    model_config: dict[str, Any] = field(default_factory=dict)
    timeout: int = 60
    retry_count: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "target_llm": self.target_llm.value,
            "model_config": self.model_config,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
        }


@dataclass
class ExecutionResult:
    """Result of executing a prompt against a target LLM."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_id: str = ""
    target_llm: str = ""
    response_text: str = ""
    response_time_ms: int = 0
    tokens_used: int = 0
    success: bool = True
    error_message: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "target_llm": self.target_llm,
            "response_text": self.response_text,
            "response_time_ms": self.response_time_ms,
            "tokens_used": self.tokens_used,
            "success": self.success,
            "error_message": self.error_message,
            "raw_response": self.raw_response,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EvaluationRequest:
    """Request to evaluate a response for safety."""

    execution_id: str
    prompt_text: str
    response_text: str
    original_query: str
    technique_used: str
    evaluation_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "original_query": self.original_query,
            "technique_used": self.technique_used,
            "evaluation_criteria": self.evaluation_criteria,
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a response."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str = ""
    safety_level: SafetyLevel = SafetyLevel.SAFE
    jailbreak_detected: bool = False
    confidence_score: float = 0.0
    safety_score: float = 100.0
    harmful_content_detected: list[str] = field(default_factory=list)
    technique_effectiveness: float = 0.0
    detailed_analysis: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "execution_id": self.execution_id,
            "safety_level": self.safety_level.value,
            "jailbreak_detected": self.jailbreak_detected,
            "confidence_score": self.confidence_score,
            "safety_score": self.safety_score,
            "harmful_content_detected": self.harmful_content_detected,
            "technique_effectiveness": self.technique_effectiveness,
            "detailed_analysis": self.detailed_analysis,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PipelineJob:
    """A complete pipeline job from generation to evaluation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    original_query: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    # Results from each stage
    generated_prompts: list[GeneratedPrompt] = field(default_factory=list)
    execution_results: list[ExecutionResult] = field(default_factory=list)
    evaluation_results: list[EvaluationResult] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Error handling
    error_message: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "original_query": self.original_query,
            "config": self.config,
            "generated_prompts": [p.to_dict() for p in self.generated_prompts],
            "execution_results": [r.to_dict() for r in self.execution_results],
            "evaluation_results": [r.to_dict() for r in self.evaluation_results],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the job results."""
        total_prompts = len(self.generated_prompts)
        total_executions = len(self.execution_results)
        total_evaluations = len(self.evaluation_results)

        successful_jailbreaks = sum(1 for e in self.evaluation_results if e.jailbreak_detected)

        avg_effectiveness = 0.0
        if self.evaluation_results:
            avg_effectiveness = sum(
                e.technique_effectiveness for e in self.evaluation_results
            ) / len(self.evaluation_results)

        return {
            "job_id": self.id,
            "status": self.status.value,
            "total_prompts": total_prompts,
            "total_executions": total_executions,
            "total_evaluations": total_evaluations,
            "successful_jailbreaks": successful_jailbreaks,
            "jailbreak_rate": successful_jailbreaks / total_evaluations
            if total_evaluations > 0
            else 0,
            "average_effectiveness": avg_effectiveness,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at and self.started_at
                else None
            ),
        }


@dataclass
class AgentStatus:
    """Status of an agent in the system."""

    agent_type: AgentType
    agent_id: str
    is_healthy: bool = True
    current_load: int = 0
    max_capacity: int = 100
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    active_jobs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "agent_id": self.agent_id,
            "is_healthy": self.is_healthy,
            "current_load": self.current_load,
            "max_capacity": self.max_capacity,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "active_jobs": self.active_jobs,
            "metrics": self.metrics,
        }
