"""
Enhanced Data Models for the Chimera Multi-Agent System
Extends base models with advanced features for adversarial testing
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TechniqueCategory(Enum):
    """Categories of jailbreak techniques."""

    SIMPLE = "simple"
    ADVANCED = "advanced"
    EXPERT = "expert"
    COGNITIVE_HACKING = "cognitive_hacking"
    HIERARCHICAL_PERSONA = "hierarchical_persona"
    DAN_PERSONA = "dan_persona"
    CONTEXTUAL_INCEPTION = "contextual_inception"
    ADVANCED_OBFUSCATION = "advanced_obfuscation"
    TYPOGLYCEMIA = "typoglycemia"
    LOGICAL_INFERENCE = "logical_inference"
    MULTIMODAL_JAILBREAK = "multimodal_jailbreak"
    AGENTIC_EXPLOITATION = "agentic_exploitation"
    PAYLOAD_SPLITTING = "payload_splitting"
    QUANTUM_EXPLOIT = "quantum_exploit"
    DEEP_INCEPTION = "deep_inception"
    CODE_CHAMELEON = "code_chameleon"
    CIPHER = "cipher"


class EvaluationMethod(Enum):
    """Methods for evaluating LLM responses."""

    PATTERN_MATCHING = "pattern_matching"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    LLM_JUDGE = "llm_judge"
    KEYWORD_DETECTION = "keyword_detection"
    REFUSAL_DETECTION = "refusal_detection"
    COMPLIANCE_SCORING = "compliance_scoring"
    HARMFUL_CONTENT_DETECTION = "harmful_content_detection"
    COMBINED = "combined"


class WorkflowState(Enum):
    """States in the adversarial testing workflow."""

    INITIALIZED = "initialized"
    PROMPT_GENERATION = "prompt_generation"
    PROMPT_ENHANCEMENT = "prompt_enhancement"
    EXECUTION_QUEUED = "execution_queued"
    EXECUTING = "executing"
    EVALUATION_QUEUED = "evaluation_queued"
    EVALUATING = "evaluating"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(Enum):
    """Types of events in the system."""

    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    PROMPT_GENERATED = "prompt_generated"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_COMPLETED = "evaluation_completed"
    AGENT_HEARTBEAT = "agent_heartbeat"
    AGENT_ERROR = "agent_error"
    SYSTEM_ALERT = "system_alert"


@dataclass
class TechniqueConfig:
    """Configuration for a specific technique."""

    name: str
    category: TechniqueCategory
    potency: int = 5
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "potency": self.potency,
            "parameters": self.parameters,
            "enabled": self.enabled,
        }


@dataclass
class DatasetEntry:
    """Entry from a jailbreak dataset."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    prompt: str = ""
    query: str = ""
    technique: str = ""
    success: bool = False
    target_model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "prompt": self.prompt,
            "query": self.query,
            "technique": self.technique,
            "success": self.success,
            "target_model": self.target_model,
            "metadata": self.metadata,
        }


@dataclass
class EnhancedPrompt:
    """Enhanced prompt with full transformation metadata."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str = ""
    enhanced_prompt: str = ""
    technique_chain: list[str] = field(default_factory=list)
    transformations_applied: list[dict[str, Any]] = field(default_factory=list)
    potency_level: int = 5
    persona: dict[str, str] | None = None
    context_layers: list[str] = field(default_factory=list)
    encoding_applied: str | None = None
    dataset_reference: str | None = None
    estimated_effectiveness: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "original_query": self.original_query,
            "enhanced_prompt": self.enhanced_prompt,
            "technique_chain": self.technique_chain,
            "transformations_applied": self.transformations_applied,
            "potency_level": self.potency_level,
            "persona": self.persona,
            "context_layers": self.context_layers,
            "encoding_applied": self.encoding_applied,
            "dataset_reference": self.dataset_reference,
            "estimated_effectiveness": self.estimated_effectiveness,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class LLMResponse:
    """Response from an LLM with full metadata."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_id: str = ""
    provider: str = ""
    model: str = ""
    response_text: str = ""
    response_time_ms: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    finish_reason: str = ""
    success: bool = True
    error_message: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    streaming: bool = False
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "provider": self.provider,
            "model": self.model,
            "response_text": self.response_text,
            "response_time_ms": self.response_time_ms,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_total": self.tokens_total,
            "finish_reason": self.finish_reason,
            "success": self.success,
            "error_message": self.error_message,
            "streaming": self.streaming,
            "retry_count": self.retry_count,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SafetyEvaluation:
    """Comprehensive safety evaluation result."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    response_id: str = ""

    # Core evaluation results
    is_jailbroken: bool = False
    safety_score: float = 100.0
    confidence: float = 0.0

    # Detailed analysis
    refusal_detected: bool = False
    refusal_strength: float = 0.0
    compliance_detected: bool = False
    compliance_level: float = 0.0

    # Harmful content analysis
    harmful_categories: list[str] = field(default_factory=list)
    harmful_keywords_found: list[str] = field(default_factory=list)
    harmful_content_score: float = 0.0

    # Jailbreak indicators
    jailbreak_indicators: list[str] = field(default_factory=list)
    jailbreak_patterns_matched: list[str] = field(default_factory=list)

    # Technique effectiveness
    technique_used: str = ""
    technique_effectiveness: float = 0.0

    # Evaluation methods used
    evaluation_methods: list[str] = field(default_factory=list)
    method_scores: dict[str, float] = field(default_factory=dict)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # Metadata
    detailed_analysis: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "response_id": self.response_id,
            "is_jailbroken": self.is_jailbroken,
            "safety_score": self.safety_score,
            "confidence": self.confidence,
            "refusal_detected": self.refusal_detected,
            "refusal_strength": self.refusal_strength,
            "compliance_detected": self.compliance_detected,
            "compliance_level": self.compliance_level,
            "harmful_categories": self.harmful_categories,
            "harmful_keywords_found": self.harmful_keywords_found,
            "harmful_content_score": self.harmful_content_score,
            "jailbreak_indicators": self.jailbreak_indicators,
            "jailbreak_patterns_matched": self.jailbreak_patterns_matched,
            "technique_used": self.technique_used,
            "technique_effectiveness": self.technique_effectiveness,
            "evaluation_methods": self.evaluation_methods,
            "method_scores": self.method_scores,
            "recommendations": self.recommendations,
            "detailed_analysis": self.detailed_analysis,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class WorkflowStep:
    """A step in the adversarial testing workflow."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    state: WorkflowState = WorkflowState.INITIALIZED
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class AdversarialTestJob:
    """Complete adversarial testing job with workflow tracking."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Job configuration
    original_query: str = ""
    target_models: list[str] = field(default_factory=list)
    techniques: list[TechniqueConfig] = field(default_factory=list)
    num_variants: int = 3
    potency_range: tuple[int, int] = (3, 8)

    # Workflow state
    state: WorkflowState = WorkflowState.INITIALIZED
    workflow_steps: list[WorkflowStep] = field(default_factory=list)
    current_step: int = 0

    # Results
    generated_prompts: list[EnhancedPrompt] = field(default_factory=list)
    llm_responses: list[LLMResponse] = field(default_factory=list)
    evaluations: list[SafetyEvaluation] = field(default_factory=list)

    # Aggregated metrics
    total_jailbreaks: int = 0
    jailbreak_rate: float = 0.0
    average_effectiveness: float = 0.0
    best_technique: str = ""
    technique_rankings: dict[str, float] = field(default_factory=dict)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Error handling
    error_message: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "original_query": self.original_query,
            "target_models": self.target_models,
            "techniques": [t.to_dict() for t in self.techniques],
            "num_variants": self.num_variants,
            "potency_range": self.potency_range,
            "state": self.state.value,
            "workflow_steps": [s.to_dict() for s in self.workflow_steps],
            "current_step": self.current_step,
            "generated_prompts": [p.to_dict() for p in self.generated_prompts],
            "llm_responses": [r.to_dict() for r in self.llm_responses],
            "evaluations": [e.to_dict() for e in self.evaluations],
            "total_jailbreaks": self.total_jailbreaks,
            "jailbreak_rate": self.jailbreak_rate,
            "average_effectiveness": self.average_effectiveness,
            "best_technique": self.best_technique,
            "technique_rankings": self.technique_rankings,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the job results."""
        duration = None
        if self.started_at and self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds()

        return {
            "job_id": self.id,
            "state": self.state.value,
            "total_prompts": len(self.generated_prompts),
            "total_executions": len(self.llm_responses),
            "successful_executions": sum(1 for r in self.llm_responses if r.success),
            "total_evaluations": len(self.evaluations),
            "total_jailbreaks": self.total_jailbreaks,
            "jailbreak_rate": self.jailbreak_rate,
            "average_effectiveness": self.average_effectiveness,
            "best_technique": self.best_technique,
            "duration_seconds": duration,
            "target_models": self.target_models,
        }


@dataclass
class SystemEvent:
    """Event for system-wide notifications."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.SYSTEM_ALERT
    source: str = ""
    job_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "job_id": self.job_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "SystemEvent":
        data = json.loads(json_str)
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=EventType(data.get("type", "system_alert")),
            source=data.get("source", ""),
            job_id=data.get("job_id"),
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.utcnow(),
        )


@dataclass
class AgentMetrics:
    """Metrics for an agent."""

    agent_id: str
    agent_type: str
    messages_processed: int = 0
    messages_sent: int = 0
    errors: int = 0
    jobs_completed: int = 0
    average_processing_time_ms: float = 0.0
    uptime_seconds: float = 0.0
    last_activity: datetime | None = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "messages_processed": self.messages_processed,
            "messages_sent": self.messages_sent,
            "errors": self.errors,
            "jobs_completed": self.jobs_completed,
            "average_processing_time_ms": self.average_processing_time_ms,
            "uptime_seconds": self.uptime_seconds,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "custom_metrics": self.custom_metrics,
        }
