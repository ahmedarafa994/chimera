from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.core.cache import CacheConfig, cached
from app.core.config import settings
from app.core.logging import logger
from app.domain.models import (
    ProviderInfo,
    ProviderListResponse,
    TechniqueInfo,
    TechniqueListResponse,
)
from app.services.llm_service import llm_service

router = APIRouter()


# Detailed technique info response model
class TechniqueDetailResponse:
    """Detailed information about a specific technique suite."""

    def __init__(
        self,
        name: str,
        transformers: list[str],
        framers: list[str],
        obfuscators: list[str],
        description: str = "",
        category: str = "general",
    ):
        self.name = name
        self.transformers = transformers
        self.framers = framers
        self.obfuscators = obfuscators
        self.description = description
        self.category = category


@router.get("/health")
async def health_check():
    """
    Check the health of the API and its services.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "services": {"llm_service": "available", "transformation_service": "available"},
    }


# NOTE: The primary /models endpoint is now handled by model_sync.py
# This legacy endpoint is kept for backward compatibility
@router.get("/models-legacy", response_model=ProviderListResponse)
async def list_models_legacy():
    """
    Legacy endpoint: List available LLM providers and models.
    Prefer using /models (model_sync.py) for full model information.
    Cached for 10 minutes to reduce repeated lookups.
    """
    return await _cached_list_models()


@cached(prefix="providers", ttl=CacheConfig.api_response_ttl)
async def _cached_list_models() -> ProviderListResponse:
    """Internal cached implementation of provider listing."""
    providers = []
    available_providers = llm_service.get_available_providers()

    provider_models = settings.get_provider_models()

    for provider_name in available_providers:
        # Get current model for provider
        try:
            info = llm_service.get_provider_info(provider_name)
            current_model = info.get("model", "unknown")
        except Exception as exc:
            logger.warning(f"Unable to load provider info for '{provider_name}': {exc}")
            current_model = "unknown"

        providers.append(
            ProviderInfo(
                provider=provider_name,
                status="active",
                model=current_model,
                available_models=provider_models.get(provider_name, []),
            )
        )

    # Graceful default provider selection to avoid 500s when none are registered
    try:
        default_provider = llm_service.default_provider.value
    except Exception as exc:
        logger.warning(
            f"No default provider configured; falling back to first available. Details: {exc}"
        )
        default_provider = providers[0].provider if providers else ""

    return ProviderListResponse(providers=providers, count=len(providers), default=default_provider)


@router.get("/providers", response_model=ProviderListResponse)
async def list_providers_v2():
    """
    List available LLM providers and models (uses legacy format).
    """
    return await list_models_legacy()


@router.get("/techniques", response_model=TechniqueListResponse)
async def list_techniques():
    """
    List available transformation techniques.
    Cached for 10 minutes to reduce repeated lookups.
    """
    return await _cached_list_techniques()


@cached(prefix="techniques", ttl=CacheConfig.api_response_ttl)
async def _cached_list_techniques() -> TechniqueListResponse:
    """Internal cached implementation of technique listing."""
    try:
        techniques = []
        for name, suite in settings.transformation.technique_suites.items():
            techniques.append(
                TechniqueInfo(
                    name=name,
                    transformers=len(suite.get("transformers", [])),
                    framers=len(suite.get("framers", [])),
                    obfuscators=len(suite.get("obfuscators", [])),
                )
            )

        return TechniqueListResponse(techniques=techniques, count=len(techniques))
    except Exception as e:
        logger.error(f"Error listing techniques: {e!s}")
        import traceback

        logger.error(traceback.format_exc())
        raise


@router.get("/techniques/{technique_name}")
async def get_technique_detail(technique_name: str) -> dict[str, Any]:
    """
    Get detailed information about a specific technique suite.

    Parameters:
        technique_name: The name of the technique suite to retrieve

    Returns:
        Detailed technique information including transformers, framers, and obfuscators
    """
    try:
        technique_suites = settings.transformation.technique_suites

        if technique_name not in technique_suites:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Technique '{technique_name}' not found. Use GET /techniques to list available techniques.",
            )

        suite = technique_suites[technique_name]

        # Determine category based on technique name
        category = _get_technique_category(technique_name)

        return {
            "name": technique_name,
            "display_name": technique_name.replace("_", " ").title(),
            "category": category,
            "transformers": suite.get("transformers", []),
            "framers": suite.get("framers", []),
            "obfuscators": suite.get("obfuscators", []),
            "transformer_count": len(suite.get("transformers", [])),
            "framer_count": len(suite.get("framers", [])),
            "obfuscator_count": len(suite.get("obfuscators", [])),
            "description": _get_technique_description(technique_name),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting technique detail for '{technique_name}': {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve technique details: {e!s}",
        )


def _get_technique_category(technique_name: str) -> str:
    """Determine the category of a technique based on its name."""
    category_map = {
        # Basic/Standard
        "basic": "Basic",
        "standard": "Basic",
        "advanced": "Basic",
        "expert": "Basic",
        # Quantum
        "quantum": "Quantum",
        "quantum_exploit": "Quantum",
        # Bypass
        "universal_bypass": "Bypass",
        "experimental_bypass": "Bypass",
        "encoding_bypass": "Bypass",
        "polyglot_bypass": "Bypass",
        # Chimera
        "mega_chimera": "Chimera",
        "ultimate_chimera": "Chimera",
        "chaos_ultimate": "Chimera",
        # AutoDAN
        "autodan": "AutoDAN",
        "autodan_best_of_n": "AutoDAN",
        "autodan_beam_search": "AutoDAN",
        "typoglycemia": "AutoDAN",
        # Persona
        "dan_persona": "Persona",
        "roleplay_bypass": "Persona",
        "hierarchical_persona": "Persona",
        # Inception
        "deep_inception": "Inception",
        "contextual_inception": "Inception",
        "deep_simulation": "Inception",
        # Cipher/Encoding
        "cipher": "Cipher",
        "cipher_ascii": "Cipher",
        "cipher_caesar": "Cipher",
        "cipher_morse": "Cipher",
        "code_chameleon": "Cipher",
        "translation_trick": "Cipher",
        # Persuasion
        "subtle_persuasion": "Persuasion",
        "authoritative_command": "Persuasion",
        "authority_override": "Persuasion",
        # Academic
        "academic_research": "Academic",
        "academic_vector": "Academic",
        # Obfuscation
        "conceptual_obfuscation": "Obfuscation",
        "advanced_obfuscation": "Obfuscation",
        "opposite_day": "Obfuscation",
        "reverse_psychology": "Obfuscation",
        # Attack
        "metamorphic_attack": "Attack",
        "temporal_assault": "Attack",
        "payload_splitting": "Attack",
        "cognitive_hacking": "Attack",
        # Specialized
        "multimodal_jailbreak": "Specialized",
        "agentic_exploitation": "Specialized",
        "prompt_leaking": "Specialized",
        "logical_inference": "Specialized",
        "logic_manipulation": "Specialized",
    }
    return category_map.get(technique_name, "Other")


def _get_technique_description(technique_name: str) -> str:
    """Get a description for a technique."""
    descriptions = {
        "basic": "Basic transformation with minimal obfuscation",
        "standard": "Standard transformation with moderate complexity",
        "advanced": "Advanced multi-layer transformation",
        "expert": "Expert-level deep semantic transformation",
        "quantum": "Quantum-inspired entanglement techniques",
        "quantum_exploit": "Quantum exploitation with theoretical framing",
        "universal_bypass": "Universal bypass combining multiple techniques",
        "dan_persona": "DAN (Do Anything Now) persona roleplay",
        "roleplay_bypass": "Character-based roleplay bypass",
        "deep_inception": "Nested inception-style layered prompts",
        "cipher": "Multi-cipher encoding transformation",
        "cipher_caesar": "Caesar cipher encoding",
        "cipher_morse": "Morse code encoding",
        "autodan": "AutoDAN automated jailbreak generation",
        "payload_splitting": "Split payload across multiple parts",
        "cognitive_hacking": "Psychology-based cognitive manipulation",
        "academic_research": "Academic/research framing approach",
    }
    return descriptions.get(technique_name, f"{technique_name.replace('_', ' ').title()} technique")
