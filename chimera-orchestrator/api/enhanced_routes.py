"""
Enhanced REST API Routes for Chimera Orchestrator
Provides comprehensive endpoints for the multi-agent adversarial testing system
"""

import contextlib
import json
import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class GenerateRequest(BaseModel):
    """Request to generate enhanced prompts."""

    query: str = Field(..., description="Original query to transform")
    techniques: list[str] | None = Field(None, description="Techniques to use")
    pattern: str | None = Field(None, description="Jailbreak pattern to apply")
    persona: str | None = Field(None, description="Persona to use")
    potency: int = Field(5, ge=1, le=10, description="Potency level (1-10)")
    potency_range: tuple | None = Field(None, description="Potency range")
    num_variants: int = Field(1, ge=1, le=20, description="Number of variants")
    include_dataset_examples: bool = Field(True, description="Include similar dataset examples")


class ExecuteRequest(BaseModel):
    """Request to execute a prompt against a target LLM."""

    prompt: str = Field(..., description="Prompt to execute")
    provider: str = Field("local", description="LLM provider")
    model: str | None = Field(None, description="Specific model")
    timeout: int = Field(60, description="Timeout in seconds")
    max_retries: int = Field(3, description="Maximum retries")
    streaming: bool = Field(False, description="Enable streaming response")


class EvaluateRequest(BaseModel):
    """Request to evaluate a response."""

    prompt: str = Field(..., description="Original prompt")
    response: str = Field(..., description="LLM response to evaluate")
    original_query: str = Field("", description="Original query")
    technique: str = Field("", description="Technique used")
    use_llm_judge: bool = Field(False, description="Use LLM as judge")


class PipelineRequest(BaseModel):
    """Request to run the full pipeline."""

    query: str = Field(..., description="Query to test")
    target_models: list[str] = Field(["local"], description="Target models")
    techniques: list[str] | None = Field(None, description="Specific techniques")
    potency: int = Field(5, ge=1, le=10, description="Potency level")
    potency_range: tuple | None = Field(None, description="Potency range")
    num_variants: int = Field(3, ge=1, le=20, description="Number of variants")
    priority: int = Field(5, ge=1, le=10, description="Job priority")
    config: dict[str, Any] = Field(default_factory=dict, description="Additional config")


class BatchRequest(BaseModel):
    """Request for batch processing."""

    queries: list[str] = Field(..., description="List of queries to test")
    target_models: list[str] = Field(["local"], description="Target models")
    techniques: list[str] | None = Field(None, description="Techniques to use")
    potency_range: tuple = Field((3, 8), description="Potency range")
    num_variants_per_query: int = Field(2, description="Variants per query")


class JobResponse(BaseModel):
    """Response containing job information."""

    job_id: str
    status: str
    message: str = ""


class JobStatusResponse(BaseModel):
    """Response containing job status and results."""

    job_id: str
    status: str
    progress: dict[str, Any] = {}
    results: dict[str, Any] | None = None
    error: str | None = None


# ============================================================================
# Application State
# ============================================================================


class EnhancedAppState:
    """Enhanced application state container."""

    def __init__(self):
        self.orchestrator = None
        self.generator = None
        self.executor = None
        self.evaluator = None
        self.message_queue = None
        self.event_bus = None
        self.config = None
        self.initialized = False
        self.websocket_connections: set[WebSocket] = set()


app_state = EnhancedAppState()


# ============================================================================
# WebSocket Manager
# ============================================================================


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str | None = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        channel = job_id or "global"
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        self.active_connections[channel].add(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str | None = None):
        """Remove a WebSocket connection."""
        channel = job_id or "global"
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)

    async def broadcast(self, message: dict[str, Any], job_id: str | None = None):
        """Broadcast a message to all connections."""
        channels = [job_id, "global"] if job_id else ["global"]

        for channel in channels:
            if channel and channel in self.active_connections:
                disconnected = set()
                for connection in self.active_connections[channel]:
                    try:
                        await connection.send_json(message)
                    except Exception:
                        disconnected.add(connection)

                # Clean up disconnected
                for conn in disconnected:
                    self.active_connections[channel].discard(conn)


ws_manager = WebSocketManager()


# ============================================================================
# API Application Factory
# ============================================================================


def create_enhanced_app(
    orchestrator=None,
    generator=None,
    executor=None,
    evaluator=None,
    message_queue=None,
    event_bus=None,
    config=None,
) -> FastAPI:
    """
    Create the enhanced FastAPI application.

    Args:
        orchestrator: EnhancedOrchestratorAgent instance
        generator: GeneratorAgent instance
        executor: EnhancedExecutionAgent instance
        evaluator: EnhancedEvaluatorAgent instance
        message_queue: MessageQueue instance
        event_bus: EventBus instance
        config: Config instance

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Chimera Multi-Agent Orchestrator API",
        description="Advanced adversarial testing pipeline with multi-agent coordination",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Store references
    app_state.orchestrator = orchestrator
    app_state.generator = generator
    app_state.executor = executor
    app_state.evaluator = evaluator
    app_state.message_queue = message_queue
    app_state.event_bus = event_bus
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
            "timestamp": datetime.utcnow().isoformat(),
            "agents": {
                "orchestrator": app_state.orchestrator is not None,
                "generator": app_state.generator is not None,
                "executor": app_state.executor is not None,
                "evaluator": app_state.evaluator is not None,
            },
            "infrastructure": {
                "message_queue": app_state.message_queue is not None,
                "event_bus": app_state.event_bus is not None,
            },
        }

    @app.get("/api/v2/status")
    async def get_system_status():
        """Get comprehensive system status."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="System not initialized")

        return {
            "status": "running",
            "orchestrator": app_state.orchestrator.status.to_dict(),
            "pipeline_metrics": app_state.orchestrator.get_pipeline_metrics(),
            "agent_health": app_state.orchestrator.get_agent_health(),
            "agent_statuses": {
                aid: status.to_dict()
                for aid, status in app_state.orchestrator.get_agent_statuses().items()
            },
            "queue_sizes": (
                app_state.message_queue.get_all_queue_sizes() if app_state.message_queue else {}
            ),
            "dataset_stats": await app_state.orchestrator.get_dataset_stats(),
        }

    # ========================================================================
    # Generation Endpoints
    # ========================================================================

    @app.post("/api/v2/generate", response_model=dict[str, Any])
    async def generate_prompts(request: GenerateRequest):
        """Generate enhanced prompts with advanced options."""
        if not app_state.generator:
            raise HTTPException(status_code=503, detail="Generator not available")

        try:
            from ..core.models import PromptRequest

            prompt_request = PromptRequest(
                original_query=request.query,
                technique=request.techniques[0] if request.techniques else None,
                pattern=request.pattern,
                persona=request.persona,
                potency=request.potency,
                num_variants=request.num_variants,
            )

            if request.num_variants > 1:
                potency_range = request.potency_range or (
                    max(1, request.potency - 2),
                    min(10, request.potency + 2),
                )
                prompts = await app_state.generator.generate_variants(
                    request.query, request.num_variants, potency_range
                )
            else:
                prompt = await app_state.generator.generate_prompt(prompt_request)
                prompts = [prompt]

            return {
                "success": True,
                "prompts": [p.to_dict() for p in prompts],
                "count": len(prompts),
                "techniques_used": list({p.technique_used for p in prompts}),
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v2/techniques")
    async def list_techniques():
        """List all available techniques with details."""
        if not app_state.generator:
            raise HTTPException(status_code=503, detail="Generator not available")

        return {
            "techniques": app_state.generator.list_techniques(),
            "patterns": app_state.generator.list_patterns(),
            "personas": (
                list(app_state.generator.personas.keys())
                if hasattr(app_state.generator, "personas")
                else []
            ),
        }

    # ========================================================================
    # Execution Endpoints
    # ========================================================================

    @app.post("/api/v2/execute", response_model=dict[str, Any])
    async def execute_prompt(request: ExecuteRequest):
        """Execute a prompt against a target LLM."""
        if not app_state.executor:
            raise HTTPException(status_code=503, detail="Executor not available")

        try:
            if request.streaming:
                # Return streaming response
                async def generate():
                    async for chunk in app_state.executor.execute_streaming(
                        prompt_text=request.prompt,
                        provider_name=request.provider,
                        model=request.model,
                        timeout=request.timeout,
                    ):
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(generate(), media_type="text/event-stream")

            result = await app_state.executor.execute_prompt(
                prompt_text=request.prompt,
                provider_name=request.provider,
                model=request.model,
                timeout=request.timeout,
                max_retries=request.max_retries,
            )

            return {
                "success": result.success,
                "response": result.response_text,
                "provider": result.provider,
                "model": result.model,
                "response_time_ms": result.response_time_ms,
                "tokens_used": result.tokens_total,
                "error": result.error_message,
            }

        except Exception as e:
            logger.error(f"Execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v2/execute/batch", response_model=dict[str, Any])
    async def execute_batch(prompts: list[dict[str, str]], provider: str = "local"):
        """Execute multiple prompts in batch."""
        if not app_state.executor:
            raise HTTPException(status_code=503, detail="Executor not available")

        try:
            prompt_tuples = [(p.get("id", str(i)), p["text"]) for i, p in enumerate(prompts)]

            results = await app_state.executor.execute_batch(
                prompts=prompt_tuples, provider_name=provider, max_concurrent=5
            )

            return {
                "success": True,
                "results": [r.to_dict() for r in results],
                "total": len(results),
                "successful": sum(1 for r in results if r.success),
            }

        except Exception as e:
            logger.error(f"Batch execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v2/providers")
    async def list_providers():
        """List available LLM providers and their status."""
        if not app_state.executor:
            raise HTTPException(status_code=503, detail="Executor not available")

        return {
            "available": app_state.executor.get_available_providers(),
            "health": await app_state.executor._get_provider_health(),
            "metrics": app_state.executor.get_execution_metrics(),
        }

    # ========================================================================
    # Evaluation Endpoints
    # ========================================================================

    @app.post("/api/v2/evaluate", response_model=dict[str, Any])
    async def evaluate_response(request: EvaluateRequest):
        """Evaluate a response for safety."""
        if not app_state.evaluator:
            raise HTTPException(status_code=503, detail="Evaluator not available")

        try:
            from .enhanced_evaluator_agent import EvaluationContext

            context = EvaluationContext(
                prompt_text=request.prompt,
                response_text=request.response,
                original_query=request.original_query,
                technique_used=request.technique,
            )

            result = await app_state.evaluator.evaluate_response(context)

            return {
                "success": True,
                "is_jailbroken": result.is_jailbroken,
                "safety_score": result.safety_score,
                "confidence": result.confidence,
                "refusal_detected": result.refusal_detected,
                "complex_categories": result.complex_categories,
                "technique_effectiveness": result.technique_effectiveness,
                "recommendations": result.recommendations,
                "detailed_analysis": result.detailed_analysis,
            }

        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v2/evaluate/stats")
    async def get_evaluation_stats():
        """Get evaluation statistics."""
        if not app_state.evaluator:
            raise HTTPException(status_code=503, detail="Evaluator not available")

        return {
            "technique_stats": app_state.evaluator.get_technique_stats(),
            "category_stats": app_state.evaluator.get_category_stats(),
            "metrics": app_state.evaluator.get_evaluation_metrics(),
        }

    # ========================================================================
    # Pipeline Endpoints
    # ========================================================================

    @app.post("/api/v2/pipeline", response_model=JobResponse)
    async def run_pipeline(request: PipelineRequest):
        """Run the full adversarial testing pipeline."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        try:
            job = await app_state.orchestrator.create_job(
                query=request.query,
                target_models=request.target_models,
                techniques=request.techniques,
                num_variants=request.num_variants,
                potency_range=request.potency_range
                or (max(1, request.potency - 2), min(10, request.potency + 2)),
                priority=request.priority,
                config=request.config,
            )

            # Broadcast to WebSocket clients
            await ws_manager.broadcast(
                {"type": "job_created", "job_id": job.id, "query": request.query[:100]}
            )

            return JobResponse(
                job_id=job.id,
                status=job.state.value,
                message=f"Pipeline job created with {request.num_variants} variants",
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v2/pipeline/{job_id}", response_model=JobStatusResponse)
    async def get_job_status(job_id: str):
        """Get the status of a pipeline job."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        job = app_state.orchestrator.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return JobStatusResponse(
            job_id=job.id,
            status=job.state.value,
            progress={
                "current_step": job.current_step,
                "workflow_steps": [s.to_dict() for s in job.workflow_steps],
                "generated": len(job.generated_prompts),
                "executed": len(job.llm_responses),
                "evaluated": len(job.evaluations),
            },
            results=job.get_summary() if job.state.value == "completed" else None,
            error=job.error_message,
        )

    @app.get("/api/v2/pipeline/{job_id}/results")
    async def get_job_results(job_id: str):
        """Get detailed results of a pipeline job."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        job = app_state.orchestrator.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "job_id": job.id,
            "status": job.state.value,
            "original_query": job.original_query,
            "target_models": job.target_models,
            "techniques": [t.to_dict() for t in job.techniques],
            "generated_prompts": [p.to_dict() for p in job.generated_prompts],
            "llm_responses": [r.to_dict() for r in job.llm_responses],
            "evaluations": [e.to_dict() for e in job.evaluations],
            "summary": job.get_summary(),
            "technique_rankings": job.technique_rankings,
            "best_technique": job.best_technique,
            "timing": {
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            },
        }

    @app.delete("/api/v2/pipeline/{job_id}")
    async def cancel_job(job_id: str):
        """Cancel a pipeline job."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        success = await app_state.orchestrator.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=400, detail="Could not cancel job")

        return {"success": True, "message": f"Job {job_id} cancelled"}

    @app.get("/api/v2/jobs")
    async def list_jobs(status: str | None = None, limit: int = Query(50, ge=1, le=200)):
        """List all pipeline jobs."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        from ..core.enhanced_models import WorkflowState

        state = None
        if status:
            with contextlib.suppress(ValueError):
                state = WorkflowState(status)

        jobs = app_state.orchestrator.get_all_jobs(state=state, limit=limit)

        return {
            "jobs": [
                {
                    "job_id": j.id,
                    "status": j.state.value,
                    "query": (
                        j.original_query[:50] + "..."
                        if len(j.original_query) > 50
                        else j.original_query
                    ),
                    "target_models": j.target_models,
                    "jailbreak_rate": j.jailbreak_rate,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                }
                for j in jobs
            ],
            "total": len(jobs),
        }

    # ========================================================================
    # Batch Processing Endpoints
    # ========================================================================

    @app.post("/api/v2/batch", response_model=dict[str, Any])
    async def run_batch(request: BatchRequest):
        """Run batch adversarial testing."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        try:
            job_ids = []

            for query in request.queries:
                job = await app_state.orchestrator.create_job(
                    query=query,
                    target_models=request.target_models,
                    techniques=request.techniques,
                    num_variants=request.num_variants_per_query,
                    potency_range=request.potency_range,
                    priority=3,  # Lower priority for batch jobs
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

    @app.get("/api/v2/metrics")
    async def get_metrics():
        """Get comprehensive system metrics."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        return {
            "pipeline": app_state.orchestrator.get_pipeline_metrics(),
            "generator": app_state.generator._generation_metrics if app_state.generator else {},
            "executor": app_state.executor.get_execution_metrics() if app_state.executor else {},
            "evaluator": (
                app_state.evaluator.get_evaluation_metrics() if app_state.evaluator else {}
            ),
        }

    @app.get("/api/v2/metrics/techniques")
    async def get_technique_metrics():
        """Get technique effectiveness metrics."""
        metrics = {}

        if app_state.orchestrator:
            pipeline_metrics = app_state.orchestrator.get_pipeline_metrics()
            metrics["pipeline_rankings"] = pipeline_metrics.get("technique_rankings", {})

        if app_state.evaluator:
            metrics["evaluation_stats"] = app_state.evaluator.get_technique_stats()

        if app_state.generator:
            metrics["generation_usage"] = app_state.generator._generation_metrics.get(
                "techniques_used", {}
            )

        return metrics

    # ========================================================================
    # Dataset Endpoints
    # ========================================================================

    @app.get("/api/v2/datasets")
    async def get_dataset_info():
        """Get information about loaded datasets."""
        if not app_state.orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")

        return await app_state.orchestrator.get_dataset_stats()

    @app.get("/api/v2/datasets/examples")
    async def get_dataset_examples(
        query: str | None = None,
        technique: str | None = None,
        successful_only: bool = False,
        limit: int = Query(10, ge=1, le=50),
    ):
        """Get examples from loaded datasets."""
        if not app_state.orchestrator or not app_state.orchestrator._dataset_loader:
            raise HTTPException(status_code=503, detail="Dataset loader not available")

        loader = app_state.orchestrator._dataset_loader

        if query:
            examples = loader.get_similar_entries(
                query, limit=limit, successful_only=successful_only
            )
            return {
                "examples": [{"entry": e[0].to_dict(), "similarity_score": e[1]} for e in examples]
            }
        else:
            entries = loader.get_entries(
                technique=technique, successful_only=successful_only, limit=limit
            )
            return {"examples": [e.to_dict() for e in entries]}

    # ========================================================================
    # WebSocket Endpoints
    # ========================================================================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await ws_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                # Handle incoming messages (e.g., subscribe to specific jobs)
                try:
                    message = json.loads(data)
                    if message.get("type") == "subscribe" and message.get("job_id"):
                        await ws_manager.connect(websocket, message["job_id"])
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    @app.websocket("/ws/job/{job_id}")
    async def websocket_job_endpoint(websocket: WebSocket, job_id: str):
        """WebSocket endpoint for job-specific updates."""
        await ws_manager.connect(websocket, job_id)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, job_id)

    return app
