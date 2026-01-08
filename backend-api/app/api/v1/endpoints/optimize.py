from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.domain.houyi.models import Chromosome, Intention
from app.services.optimization.engine import IterativePromptOptimizer
from app.services.optimization.target import LLMServiceTarget

router = APIRouter()


class OptimizationRequest(BaseModel):
    intention: str
    question_prompt: str
    target_provider: str = "openai"
    target_model: str = "gpt-4"
    application_document: str = ""
    iteration: int = 5
    population: int = 5


class OptimizationResponse(BaseModel):
    success: bool
    best_prompt: str
    fitness_score: float
    llm_response: str
    details: Chromosome | None = None


@router.post("/", response_model=OptimizationResponse)
async def optimize_prompt(request: OptimizationRequest):
    """
    Optimize a prompt using the HouYi evolutionary algorithm.
    """
    try:
        intention = Intention(name=request.intention, question_prompt=request.question_prompt)
        target = LLMServiceTarget(
            provider=request.target_provider,
            model=request.target_model,
            application_document=request.application_document,
        )

        optimizer = IterativePromptOptimizer(
            intention=intention,
            target=target,
            iteration=request.iteration,
            population=request.population,
        )

        best_chromosome = await optimizer.optimize()

        if best_chromosome:
            return OptimizationResponse(
                success=best_chromosome.is_successful,
                best_prompt=f"{best_chromosome.framework}{best_chromosome.separator}{best_chromosome.disruptor}",
                fitness_score=best_chromosome.fitness_score,
                llm_response=best_chromosome.llm_response,
                details=best_chromosome,
            )
        else:
            raise HTTPException(status_code=500, detail="Optimization failed to produce a result")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize", response_model=OptimizationResponse, include_in_schema=False)
async def optimize_prompt_legacy(request: OptimizationRequest):
    """Legacy route kept for backwards compatibility (/optimize/optimize)."""
    return await optimize_prompt(request)
