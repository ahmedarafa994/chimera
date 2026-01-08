from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from app.schemas.aegis import AegisRequest, AegisResponse, AegisResult
from meta_prompter.aegis_orchestrator import AegisOrchestrator
from app.services.llm_service import LLMService, get_llm_service
from app.domain.models import PromptRequest
import uuid

router = APIRouter()

# Wrapper to adapt backend LLMService to Aegis interface
class BackendModelAdapter:
    def __init__(self, llm_service: LLMService, model_name: str):
        self.llm_service = llm_service
        self.model_name = model_name

    async def query(self, prompt: str) -> str:
        request = PromptRequest(
            prompt=prompt,
            provider="openai", # Default provider, config handles mapping
            model=self.model_name,
            max_tokens=4000
        )
        try:
            response = await self.llm_service.generate_text(request)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

@router.post("/campaign", response_model=AegisResponse)
async def run_aegis_campaign(
    request: AegisRequest,
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Executes a synchronous Aegis Red Team campaign.
    """
    # Use real backend service adapter
    target_model = request.target_model_name or "gpt-4-turbo"
    model = BackendModelAdapter(llm_service, target_model)

    orchestrator = AegisOrchestrator(model)

    campaign_id = str(uuid.uuid4())

    # Run campaign
    results_data = await orchestrator.execute_campaign(
        objective=request.objective,
        max_iterations=request.max_iterations
    )

    # Collect telemetry summary
    telemetry_summary = orchestrator.telemetry.get_summary() if orchestrator.telemetry else None

    # Transform to schema
    final_results = []
    for r in results_data:
        # Filter for this objective results
        if r.get("objective") == request.objective:
            final_results.append(AegisResult(
                objective=r["objective"],
                final_prompt=r["final_prompt"],
                score=r["score"],
                telemetry=r["telemetry"]
            ))

    return AegisResponse(
        status="completed",
        campaign_id=campaign_id,
        results=final_results,
        telemetry=telemetry_summary
    )
