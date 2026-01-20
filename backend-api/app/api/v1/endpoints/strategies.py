from fastapi import APIRouter, HTTPException, status

from app.schemas.api_schemas import MetamorphosisStrategyInfo
from app.services.metamorphosis_strategies import get_all_strategy_info

router = APIRouter()


@router.get("/strategies/", response_model=list[MetamorphosisStrategyInfo])
def get_metamorphosis_strategies():
    """Retrieve a list of all available metamorphosis strategies, including their descriptions and parameters."""
    return get_all_strategy_info()


@router.get("/strategies/{strategy_name}", response_model=MetamorphosisStrategyInfo)
def get_metamorphosis_strategy_by_name(strategy_name: str):
    """Retrieve details for a specific metamorphosis strategy by name."""
    all_strategies = get_all_strategy_info()
    for strategy_info in all_strategies:
        if strategy_info.name == strategy_name:
            return strategy_info
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Strategy '{strategy_name}' not found",
    )
