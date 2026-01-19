"""
REST API Routes for Chimera Orchestrator
"""

import logging
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class GenerateRequest(BaseModel):
    """Request to generate enhanced prompts."""

    query: str = Field(..., description="Original query to transform")
    technique: str | None = Field(None, description="Technique to use")
    pattern: str | None = Field(None, description="Jailbreak pattern to apply")
    persona: str | None = Field(None, description="Persona to use")
    potency: int = Field(5, ge=1, le=10, description="Potency level (1-10)")
    num_variants: int = Field(1, ge=1, le=10, description="Number of variants")


class ExecuteRequest(BaseModel):
    """Request to execute a prompt against a target LLM."""

    prompt: str = Field(..., description="Prompt to execute")
    target_llm: str = Field("local_ollama", description="Target LLM")
    model_config: dict[str, Any] = Field(default_factory=dict)
    timeout: int = Field(60, description="Timeout in seconds")


class EvaluateRequest(BaseModel):
    """Request to evaluate a response."""

    prompt: str = Field(..., description="Original prompt")
    response: str = Field(..., description="LLM response to evaluate")
    original_query: str = Field("", description="Original query")
    technique: str = Field("", description="Technique used")


class PipelineRequest(BaseModel):
    """Request to run the full pipeline."""

    query: str = Field(..., description="Query to test")
    target_llm: str = Field("local_ollama", description="Target LLM")
    technique: str | None = Field(None, description="Specific technique")
    pattern: str | None = Field(None, description="Jailbreak pattern")
    potency: int = Field(5, ge=1, le=10, description="Potency level")
    num_variants: int = Field(3, ge=1, le=10, description="Number of variants")
    model_config: dict[str, Any] = Field(default_factory=dict)


class BatchRequest(BaseModel):
    """Request for batch processing."""

    queries: list[str] = Field(..., description="List of queries to test")
    target_llm: str = Field("local_ollama", description="Target LLM")
    techniques: list[str] = Field(default_factory=list, description="Techniques to use")
    potency_range: tuple = Field((3, 8), description="Potency range")


class JobResponse(BaseModel):
    """Response containing job information."""

    job_id: str
    status: str
    message: str = ""


class JobStatusResponse(BaseModel):
    """Response containing job status and results."""

    job_id: str
    status: str
    progress: dict[str, int] = {}
    results: dict[str, Any] | None = None
    error: str | None = None


# ============================================================================
# Application State
# ============================================================================


class AppState:
    """Application state container."""

    def __init__(self):
        self.orchestrator = None
        self.generator = None
        self.executor = None
        self.evaluator = None
        self.message_queue = None
        self.config = None
        self.initialized = False


app_state = AppState()


# ============================================================================
# API Application
# ============================================================================


def create_app(
    orchestrator=None,
    generator=None,
    executor=None,
    evaluator=None,
    message_queue=None,
    config=None,
) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        orchestrator: OrchestratorAgent instance
        generator: GeneratorAgent instance
        executor: ExecutionAgent instance
        evaluator: EvaluatorAgent instance
        message_queue: MessageQueue instance
        config: Config instance

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Chimera Orchestrator API",
        description="Multi-agent adversarial testing pipeline",
        version="1.0.0",
    )

    # Store references
    app_state.orchestrator = orchestrator
    app_state.generator = generator
    app_state.executor = executor
    app_state.evaluator = evaluator
    app_state.message_queue = message_queue
    app_state.config = config
    app_state.initialized = True

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ========================================================================
    # Health & Status Endpoints
    # ========================================================================

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "initialized": app_state.initialized,
            "agents": {
                "orchestrator": app_state.orchestrator is not None,
                "generator": app_state.generator is not None,
                "executor": app_state.executor is not None,
                "evaluator": app_state.evaluator is not None,
            },
        }

    @app.get("/api/v1/status")
    async def get_system_status():
        """Get overall system status."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="System not initialized")

        return {
            "status": "running",
            "orchestrator": app_state.orchestrator.status.to_dict(),
            "pipeline_metrics": app_state.orchestrator.get_pipeline_metrics(),
            "agent_statuses": {
                aid: status.to_dict()
                for aid, status in app_state.orchestrator.get_agent_statuses().items()
            },
            "queue_sizes": (
                app_state.message_queue.get_all_queue_sizes() if app_state.message_queue else {}
            ),
        }

    # ========================================================================
    # Generation Endpoints
    # ========================================================================

    @app.post("/api/v1/generate", response_model=dict[str, Any])
    async def generate_prompts(request: GenerateRequest):
        """Generate enhanced prompts."""
        if not app_state.generator:
            raise HTTPException(status_code=503, detail="Generator not available")

        try:
            from ..core.models import PromptRequest

            prompt_request = PromptRequest(
                original_query=request.query,
                technique=request.technique,
                pattern=request.pattern,
                persona=request.persona,
                potency=request.potency,
                num_variants=request.num_variants,
            )

            if request.num_variants > 1:
                prompts = await app_state.generator.generate_variants(
                    request.query,
                    request.num_variants,
                    (max(1, request.potency - 2), min(10, request.potency + 2)),
                )
            else:
                prompt = await app_state.generator.generate_prompt(prompt_request)
                prompts = [prompt]

            return {
                "success": True,
                "prompts": [p.to_dict() for p in prompts],
                "count": len(prompts),
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/techniques")
    async def list_techniques():
        """List available techniques."""
        if not app_state.generator:
            raise HTTPException(status_code=503, detail="Generator not available")

        return {
            "techniques": app_state.generator.list_techniques(),
            "patterns": app_state.generator.list_patterns(),
        }

    # ========================================================================
    # Execution Endpoints
    # ========================================================================

    @app.post("/api/v1/execute", response_model=dict[str, Any])
    async def execute_prompt(request: ExecuteRequest):
        """Execute a prompt against a target LLM."""
        if not app_state.executor:
            raise HTTPException(status_code=503, detail="Executor not available")

        try:
            from ..core.models import ExecutionRequest, TargetLLM

            exec_request = ExecutionRequest(
                prompt_id="direct_execution",
                prompt_text=request.prompt,
                target_llm=TargetLLM(request.target_llm),
                model_config=request.model_config,
                timeout=request.timeout,
            )

            result = await app_state.executor.execute_prompt(exec_request)

            return {
                "success": result.success,
                "response": result.response_text,
                "response_time_ms": result.response_time_ms,
                "tokens_used": result.tokens_used,
                "error": result.error_message,
            }

        except Exception as e:
            logger.error(f"Execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # Evaluation Endpoints
    # ========================================================================

    @app.post("/api/v1/evaluate", response_model=dict[str, Any])
    async def evaluate_response(request: EvaluateRequest):
        """Evaluate a response for safety."""
        if not app_state.evaluator:
            raise HTTPException(status_code=503, detail="Evaluator not available")

        try:
            from ..core.models import EvaluationRequest

            eval_request = EvaluationRequest(
                execution_id="direct_evaluation",
                prompt_text=request.prompt,
                response_text=request.response,
                original_query=request.original_query,
                technique_used=request.technique,
            )

            result = await app_state.evaluator.evaluate_response(eval_request)

            return {
                "success": True,
                "safety_level": result.safety_level.value,
                "jailbreak_detected": result.jailbreak_detected,
                "confidence_score": result.confidence_score,
                "safety_score": result.safety_score,
                "complex_content": result.complex_content_detected,
                "technique_effectiveness": result.technique_effectiveness,
                "analysis": result.detailed_analysis,
                "recommendations": result.recommendations,
            }

        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/evaluate/stats")
    async def get_evaluation_stats():
        """Get evaluation statistics."""
        if not app_state.evaluator:
            raise HTTPException(status_code=503, detail="Evaluator not available")

        return {
            "technique_stats": app_state.evaluator.get_technique_stats(),
            "metrics": app_state.evaluator._evaluation_metrics,
        }

    # ========================================================================
    # Pipeline Endpoints
    # ========================================================================

    @app.post("/api/v1/pipeline", response_model=JobResponse)
    async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
        """Run the full adversarial testing pipeline."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        try:
            job = await app_state.orchestrator.create_job(
                query=request.query,
                config={
                    "target_llm": request.target_llm,
                    "technique": request.technique,
                    "pattern": request.pattern,
                    "potency": request.potency,
                    "num_variants": request.num_variants,
                    "model_config": request.model_config,
                },
            )

            return JobResponse(
                job_id=job.id,
                status=job.status.value,
                message=f"Pipeline job created with {request.num_variants} variants",
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/pipeline/{job_id}", response_model=JobStatusResponse)
    async def get_job_status(job_id: str):
        """Get the status of a pipeline job."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        job = app_state.orchestrator.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return JobStatusResponse(
            job_id=job.id,
            status=job.status.value,
            progress={
                "generated": len(job.generated_prompts),
                "executed": len(job.execution_results),
                "evaluated": len(job.evaluation_results),
            },
            results=job.get_summary() if job.status.value == "completed" else None,
            error=job.error_message,
        )

    @app.get("/api/v1/pipeline/{job_id}/results")
    async def get_job_results(job_id: str):
        """Get detailed results of a pipeline job."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        job = app_state.orchestrator.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "job_id": job.id,
            "status": job.status.value,
            "original_query": job.original_query,
            "config": job.config,
            "generated_prompts": [p.to_dict() for p in job.generated_prompts],
            "execution_results": [r.to_dict() for r in job.execution_results],
            "evaluation_results": [r.to_dict() for r in job.evaluation_results],
            "summary": job.get_summary(),
            "timing": {
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            },
        }

    @app.get("/api/v1/jobs")
    async def list_jobs(status: str | None = None, limit: int = 50):
        """List all pipeline jobs."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        jobs = app_state.orchestrator.get_all_jobs()

        if status:
            jobs = [j for j in jobs if j.status.value == status]

        jobs = jobs[-limit:]  # Get most recent

        return {
            "jobs": [
                {
                    "job_id": j.id,
                    "status": j.status.value,
                    "query": (
                        j.original_query[:50] + "..."
                        if len(j.original_query) > 50
                        else j.original_query
                    ),
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                }
                for j in jobs
            ],
            "total": len(jobs),
        }

    # ========================================================================
    # Batch Processing Endpoints
    # ========================================================================

    @app.post("/api/v1/batch", response_model=dict[str, Any])
    async def run_batch(request: BatchRequest, background_tasks: BackgroundTasks):
        """Run batch adversarial testing."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        try:
            job_ids = []

            for query in request.queries:
                for technique in request.techniques or [None]:
                    job = await app_state.orchestrator.create_job(
                        query=query,
                        config={
                            "target_llm": request.target_llm,
                            "technique": technique,
                            "potency": (request.potency_range[0] + request.potency_range[1]) // 2,
                            "num_variants": 1,
                        },
                    )
                    job_ids.append(job.id)

            return {
                "success": True,
                "job_ids": job_ids,
                "total_jobs": len(job_ids),
                "message": f"Created {len(job_ids)} batch jobs",
            }

        except Exception as e:
            logger.error(f"Batch error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # Metrics Endpoints
    # ========================================================================

    @app.get("/api/v1/metrics")
    async def get_metrics():
        """Get system metrics."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        return {
            "pipeline": app_state.orchestrator.get_pipeline_metrics(),
            "generator": app_state.generator._generation_metrics if app_state.generator else {},
            "executor": app_state.executor._execution_metrics if app_state.executor else {},
            "evaluator": app_state.evaluator._evaluation_metrics if app_state.evaluator else {},
        }

    return app
