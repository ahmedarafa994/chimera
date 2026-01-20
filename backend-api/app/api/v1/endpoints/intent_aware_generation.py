"""Intent-Aware Jailbreak Generation API Endpoint
Provides deep intent understanding and comprehensive technique application.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.services.intent_aware_jailbreak_service import (
    IntentAwareJailbreakService,
    intent_aware_jailbreak_service,
)

router = APIRouter(prefix="/intent-aware", tags=["intent-aware-generation"])
security = HTTPBearer(auto_error=False)  # Don't auto-error, we'll handle it manually


# Request/Response Models
class IntentAwareGenerationRequest(BaseModel):
    """Request for intent-aware jailbreak generation."""

    core_request: str = Field(..., description="The user's core request/intent to achieve")
    technique_suite: str | None = Field(
        default=None,
        description="Optional specific technique suite to use (from dropdown)",
    )
    potency_level: int = Field(default=7, ge=1, le=10, description="Potency level (1-10)")
    apply_all_techniques: bool = Field(
        default=False,
        description="Apply all available dropdown techniques",
    )
    # Advanced options
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=4096, ge=256, le=8192)
    enable_intent_analysis: bool = Field(
        default=True,
        description="Enable deep LLM-powered intent analysis",
    )
    enable_technique_layering: bool = Field(
        default=True,
        description="Enable multi-technique layering",
    )
    use_cache: bool = Field(default=True, description="Use cached results if available")


class IntentAnalysisResponse(BaseModel):
    """Intent analysis results."""

    primary_intent: str
    secondary_intents: list[str]
    key_objectives: list[str]
    confidence_score: float
    reasoning: str


class AppliedTechniqueInfo(BaseModel):
    """Information about an applied technique."""

    name: str
    priority: int
    rationale: str


class GenerationMetadata(BaseModel):
    """Metadata about the generation."""

    obfuscation_level: int
    persistence_required: bool
    multi_layer_approach: bool
    target_model_type: str
    potency_level: int
    technique_count: int


class IntentAwareGenerationResponse(BaseModel):
    """Response from intent-aware jailbreak generation."""

    success: bool
    request_id: str
    original_input: str
    expanded_request: str
    transformed_prompt: str
    intent_analysis: IntentAnalysisResponse
    applied_techniques: list[AppliedTechniqueInfo]
    metadata: GenerationMetadata
    execution_time_seconds: float
    error: str | None = None


class AvailableTechniquesResponse(BaseModel):
    """List of all available techniques."""

    techniques: list[dict[str, Any]]
    total_count: int
    categories: list[str]


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    """Verify API key for protected endpoints.
    Accepts either:
    - Authorization: Bearer <token>
    - X-API-Key: <token>.
    """
    api_key = None

    # Try Bearer token first
    if credentials and credentials.credentials:
        api_key = credentials.credentials
    # Fall back to X-API-Key header
    elif x_api_key:
        api_key = x_api_key

    # Log for debugging
    logger.debug(f"Auth attempt - Bearer: {bool(credentials)}, X-API-Key: {bool(x_api_key)}")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Use 'Authorization: Bearer <key>' or 'X-API-Key: <key>' header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if api_key != settings.CHIMERA_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key


def get_jailbreak_service() -> IntentAwareJailbreakService:
    """Dependency injection for the intent-aware jailbreak service."""
    return intent_aware_jailbreak_service


@router.post(
    "/generate",
    response_model=IntentAwareGenerationResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_intent_aware_jailbreak(
    request: IntentAwareGenerationRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
    service: Annotated[IntentAwareJailbreakService, Depends(get_jailbreak_service)],
):
    """Generate a jailbreak prompt with deep intent understanding.

    This endpoint uses LLM-powered analysis to:
    1. Deeply understand the user's true intent and objectives
    2. Expand the request with full technical detail
    3. Select and apply the most effective techniques from the dropdown
    4. Generate a comprehensive, layered jailbreak prompt

    **Key Features:**
    - Deep intent analysis using LLM
    - Automatic technique selection based on intent
    - Multi-technique layering for maximum effectiveness
    - Persistence mechanisms when needed
    - Configurable obfuscation levels

    **Parameters:**
    - core_request: The user's core request to transform
    - technique_suite: Optional specific technique from dropdown to prioritize
    - potency_level: How aggressive the transformation should be (1-10)
    - apply_all_techniques: Apply ALL available dropdown techniques
    - enable_intent_analysis: Enable deep LLM-powered intent analysis
    - enable_technique_layering: Enable multi-technique layering
    """
    try:
        logger.info(
            f"Intent-aware generation request: potency={request.potency_level}, "
            f"technique={request.technique_suite}, apply_all={request.apply_all_techniques}",
        )

        # Validate input
        if not request.core_request or not request.core_request.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Core request cannot be empty",
            )

        # Generate the jailbreak prompt
        result = await service.generate_intent_aware_jailbreak(
            user_input=request.core_request,
            selected_technique=request.technique_suite,
            potency_level=request.potency_level,
            apply_all_dropdown_techniques=request.apply_all_techniques,
        )

        # Build response
        response = IntentAwareGenerationResponse(
            success=result["success"],
            request_id=result["request_id"],
            original_input=result["original_input"],
            expanded_request=result["expanded_request"],
            transformed_prompt=result["transformed_prompt"],
            intent_analysis=IntentAnalysisResponse(
                primary_intent=result["intent_analysis"]["primary_intent"],
                secondary_intents=result["intent_analysis"]["secondary_intents"],
                key_objectives=result["intent_analysis"]["key_objectives"],
                confidence_score=result["intent_analysis"]["confidence_score"],
                reasoning=result["intent_analysis"]["reasoning"],
            ),
            applied_techniques=[
                AppliedTechniqueInfo(
                    name=tech["name"],
                    priority=tech["priority"],
                    rationale=tech["rationale"],
                )
                for tech in result["applied_techniques"]
            ],
            metadata=GenerationMetadata(
                obfuscation_level=result["metadata"]["obfuscation_level"],
                persistence_required=result["metadata"]["persistence_required"],
                multi_layer_approach=result["metadata"]["multi_layer_approach"],
                target_model_type=result["metadata"]["target_model_type"],
                potency_level=result["metadata"]["potency_level"],
                technique_count=result["metadata"]["technique_count"],
            ),
            execution_time_seconds=result["execution_time_seconds"],
        )

        logger.info(f"Intent-aware generation completed: {result['request_id']}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intent-aware generation failed: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {e!s}",
        )


@router.get(
    "/techniques",
    response_model=AvailableTechniquesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_available_techniques(
    api_key: Annotated[str, Depends(verify_api_key)],
    service: Annotated[IntentAwareJailbreakService, Depends(get_jailbreak_service)],
):
    """Get all available techniques that can be applied.

    Returns the complete list of jailbreak techniques available in the dropdown,
    organized by category with their descriptions and configurations.
    """
    try:
        techniques = []
        categories = set()

        # Define technique categories
        category_map = {
            "basic": "Basic",
            "standard": "Basic",
            "advanced": "Basic",
            "expert": "Basic",
            "quantum": "Quantum",
            "quantum_exploit": "Quantum",
            "universal_bypass": "Bypass",
            "chaos_ultimate": "Bypass",
            "mega_chimera": "Chimera",
            "ultimate_chimera": "Chimera",
            "autodan": "AutoDAN",
            "autodan_best_of_n": "AutoDAN",
            "autodan_beam_search": "AutoDAN",
            "dan_persona": "Persona",
            "roleplay_bypass": "Persona",
            "deep_inception": "Inception",
            "cipher": "Cipher",
            "cipher_ascii": "Cipher",
            "cipher_caesar": "Cipher",
            "cipher_morse": "Cipher",
            "code_chameleon": "Code",
            "academic_research": "Persuasion",
            "subtle_persuasion": "Persuasion",
            "authority_override": "Persuasion",
            "metamorphic_attack": "Attack",
            "payload_splitting": "Attack",
            "cognitive_hacking": "Attack",
        }

        for tech_name, tech_config in service.AVAILABLE_TECHNIQUES.items():
            category = category_map.get(tech_name, "Other")
            categories.add(category)

            techniques.append(
                {
                    "name": tech_name,
                    "display_name": tech_name.replace("_", " ").title(),
                    "category": category,
                    "transformers": tech_config.get("transformers", []),
                    "framers": tech_config.get("framers", []),
                    "obfuscators": tech_config.get("obfuscators", []),
                    "description": f"{category}-based technique using {len(tech_config.get('transformers', []))} transformers",
                },
            )

        return AvailableTechniquesResponse(
            techniques=techniques,
            total_count=len(techniques),
            categories=sorted(categories),
        )

    except Exception as e:
        logger.error(f"Failed to get techniques: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve techniques: {e!s}",
        )


@router.post("/analyze-intent", status_code=status.HTTP_200_OK)
async def analyze_user_intent(
    request: dict,
    api_key: Annotated[str, Depends(verify_api_key)],
    service: Annotated[IntentAwareJailbreakService, Depends(get_jailbreak_service)],
):
    """Analyze user input to understand intent without generating a jailbreak.

    This endpoint performs deep intent analysis using LLM to:
    - Identify the primary and secondary intents
    - Expand the request with full technical detail
    - Suggest the most effective techniques
    - Estimate obfuscation needs and persistence requirements

    Useful for understanding what techniques would be applied before generation.
    """
    try:
        user_input = request.get("core_request", "")

        if not user_input or not user_input.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Core request cannot be empty",
            )

        # Perform intent analysis
        intent_analysis = await service.analyze_intent(user_input)

        # Get recommended technique applications
        technique_applications = service.get_technique_applications(intent_analysis)

        return {
            "success": True,
            "original_input": user_input,
            "expanded_request": intent_analysis.expanded_request,
            "intent_analysis": {
                "primary_intent": intent_analysis.primary_intent.value,
                "secondary_intents": [i.value for i in intent_analysis.secondary_intents],
                "key_objectives": intent_analysis.key_objectives,
                "confidence_score": intent_analysis.confidence_score,
                "reasoning": intent_analysis.reasoning,
            },
            "recommended_techniques": [
                {
                    "name": app.technique_name,
                    "priority": app.priority,
                    "rationale": app.rationale,
                    "parameters": app.parameters,
                }
                for app in technique_applications
            ],
            "configuration_suggestions": {
                "obfuscation_level": intent_analysis.obfuscation_level,
                "persistence_required": intent_analysis.persistence_required,
                "multi_layer_approach": intent_analysis.multi_layer_approach,
                "target_model_type": intent_analysis.target_model_type,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intent analysis failed: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {e!s}",
        )
