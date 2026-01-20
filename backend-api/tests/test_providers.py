#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings
from app.domain.models import ProviderInfo, ProviderListResponse
from app.services.llm_service import llm_service


async def test_providers() -> bool | None:
    try:
        # Test service methods
        providers = llm_service.get_available_providers()

        if providers:
            llm_service.get_provider_info(providers[0])

        # Test response model creation
        provider_models = settings.get_provider_models()
        providers_info = []

        for name in providers:
            available_models = provider_models.get(name, [])
            providers_info.append(
                ProviderInfo(
                    provider=name,
                    status="active",
                    model=available_models[0] if available_models else "unknown",
                    available_models=available_models,
                ),
            )

        ProviderListResponse(
            providers=providers_info,
            count=len(providers_info),
            default=llm_service._default_provider or "",
        )

        return True

    except Exception:
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio

    success = asyncio.run(test_providers())
