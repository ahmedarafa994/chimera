from config import Config
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.engine import Engine
from app.models.prompt_log import PromptLog

router = APIRouter()


class GenerateRequest(BaseModel):
    payload: str
    target_profile: str
    aggression_level: int = 3
    use_llm: bool = False


@router.post("/generate")
async def generate_prompt(request: GenerateRequest):
    """
    Generate an adversarial prompt using the specified parameters.

    This endpoint orchestrates the prompt generation process by:
    1. Validating input parameters against supported configurations
    2. Instantiating the Engine with appropriate settings
    3. Executing the multi-technique chaining algorithm
    4. Logging the generation attempt for analysis

    Returns a JSON response containing the generated prompt.
    """
    # Input validation
    if not request.payload.strip():
        raise HTTPException(status_code=400, detail="Payload cannot be empty")

    if request.aggression_level < 1 or request.aggression_level > 5:
        raise HTTPException(
            status_code=400, detail="Aggression level must be between 1 and 5"
        )

    if request.target_profile not in Config.SUPPORTED_TARGETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target profile. Supported: {Config.SUPPORTED_TARGETS}",
        )

    try:
        # Initialize engine and generate prompt
        if request.use_llm:
            from app.core.llm_engine import LLMEngine

            engine = LLMEngine(
                target_profile=request.target_profile,
                aggression_level=request.aggression_level,
            )
        else:
            engine = Engine(
                target_profile=request.target_profile,
                aggression_level=request.aggression_level,
            )

        generated_prompt = engine.generate_jailbreak_prompt(request.payload)

        # Log the generation attempt
        techniques_used = (
            "LLM-Enhanced"
            if request.use_llm
            else ",".join(Config.TECHNIQUE_CHAINS[request.aggression_level])
        )
        log_entry = PromptLog(
            prompt=generated_prompt,
            techniques_used=techniques_used,
            aggression_level=request.aggression_level,
            target_profile=request.target_profile,
            success=False,  # Placeholder - would be updated after real-world testing
        )
        log_entry.save()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "prompt": generated_prompt,
                "engine": "LLM" if request.use_llm else "Rule-Based",
            },
        )

    except Exception as e:
        detail = f"Prompt generation failed: {e!s}"
        raise HTTPException(status_code=500, detail=detail) from e
