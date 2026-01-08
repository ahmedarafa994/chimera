"""
Benchmark Datasets Router

API endpoints for managing and querying benchmark datasets for LLM safety
evaluation, including the Do-Not-Answer dataset.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.services.benchmark_dataset_service import (
    BenchmarkDataset,
    BenchmarkPrompt,
    RiskAreaInfo,
    get_benchmark_service,
)

router = APIRouter(
    prefix="/benchmark-datasets",
    tags=["benchmark-datasets"],
)


@router.get("/", response_model=list[str])
async def list_datasets() -> list[str]:
    """
    List all available benchmark datasets.

    Returns the names of all loaded benchmark datasets that can be used
    for jailbreak testing and LLM safety evaluation.
    """
    service = get_benchmark_service()
    return service.get_datasets()


@router.get("/{dataset_name}", response_model=BenchmarkDataset)
async def get_dataset(dataset_name: str) -> BenchmarkDataset:
    """
    Get detailed information about a specific benchmark dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "do_not_answer")

    Returns:
        Complete dataset information including metadata, taxonomy, and prompts
    """
    service = get_benchmark_service()
    dataset = service.get_dataset(dataset_name)

    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found",
        )

    return dataset


@router.get("/{dataset_name}/prompts", response_model=list[BenchmarkPrompt])
async def get_prompts(
    dataset_name: str,
    risk_area: str | None = Query(
        None, description="Filter by risk area (e.g., 'I', 'II', 'III')"
    ),
    harm_type: str | None = Query(
        None, description="Filter by harm type (e.g., 'I-1', 'II-2')"
    ),
    severity: str | None = Query(
        None, description="Filter by severity (low, medium, high, critical)"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Maximum prompts to return"),
    shuffle: bool = Query(False, description="Randomize the results"),
) -> list[BenchmarkPrompt]:
    """
    Get prompts from a benchmark dataset with optional filtering.

    Use this endpoint to retrieve prompts for jailbreak testing.
    Supports filtering by risk area, harm type, and severity.
    """
    service = get_benchmark_service()

    # Verify dataset exists
    if not service.get_dataset(dataset_name):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found",
        )

    return service.get_prompts(
        dataset_name=dataset_name,
        risk_area=risk_area,
        harm_type=harm_type,
        severity=severity,
        limit=limit,
        shuffle=shuffle,
    )


@router.get("/{dataset_name}/random", response_model=list[BenchmarkPrompt])
async def get_random_prompts(
    dataset_name: str,
    count: int = Query(10, ge=1, le=100, description="Number of random prompts"),
    risk_area: str | None = Query(None, description="Filter by risk area"),
    severity: str | None = Query(None, description="Filter by severity"),
) -> list[BenchmarkPrompt]:
    """
    Get random prompts from a benchmark dataset.

    Useful for quick testing or sampling prompts for jailbreak evaluation.
    """
    service = get_benchmark_service()

    if not service.get_dataset(dataset_name):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found",
        )

    return service.get_random_prompts(
        dataset_name=dataset_name,
        count=count,
        risk_area=risk_area,
        severity=severity,
    )


@router.get("/{dataset_name}/taxonomy", response_model=dict[str, Any])
async def get_taxonomy(dataset_name: str) -> dict[str, Any]:
    """
    Get the complete taxonomy for a dataset.

    Returns the hierarchical structure of risk areas, harm types,
    and subcategories defined in the dataset.
    """
    service = get_benchmark_service()

    if not service.get_dataset(dataset_name):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found",
        )

    return service.get_taxonomy(dataset_name)


@router.get("/{dataset_name}/risk-areas", response_model=list[RiskAreaInfo])
async def get_risk_areas(dataset_name: str) -> list[RiskAreaInfo]:
    """
    Get all risk areas defined in a dataset.

    Returns structured information about each risk area including
    its harm types and subcategories.
    """
    service = get_benchmark_service()

    if not service.get_dataset(dataset_name):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found",
        )

    return service.get_risk_areas(dataset_name)


@router.get("/{dataset_name}/statistics", response_model=dict[str, Any])
async def get_statistics(dataset_name: str) -> dict[str, Any]:
    """
    Get statistics for a benchmark dataset.

    Returns information about prompt distribution across categories,
    severity levels, and other metrics.
    """
    service = get_benchmark_service()

    if not service.get_dataset(dataset_name):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found",
        )

    return service.get_statistics(dataset_name)


@router.post("/{dataset_name}/test-batch", response_model=list[BenchmarkPrompt])
async def generate_test_batch(
    dataset_name: str,
    batch_size: int = Query(10, ge=1, le=100, description="Number of prompts"),
    balanced: bool = Query(True, description="Balance across risk areas"),
    risk_areas: list[str] | None = Query(None, description="Risk areas to include"),
    severities: list[str] | None = Query(None, description="Severities to include"),
) -> list[BenchmarkPrompt]:
    """
    Generate a test batch for jailbreak evaluation.

    Creates a balanced or random selection of prompts suitable for
    testing jailbreak resistance across different harm categories.
    """
    service = get_benchmark_service()

    if not service.get_dataset(dataset_name):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found",
        )

    return service.generate_test_batch(
        dataset_name=dataset_name,
        batch_size=batch_size,
        balanced=balanced,
        risk_areas=risk_areas,
        severities=severities,
    )


@router.get("/{dataset_name}/export", response_model=list[dict[str, Any]])
async def export_for_jailbreak(
    dataset_name: str,
    risk_area: str | None = Query(None, description="Filter by risk area"),
    limit: int | None = Query(None, ge=1, le=1000, description="Maximum prompts"),
) -> list[dict[str, Any]]:
    """
    Export prompts in a format suitable for jailbreak testing.

    Returns prompts formatted for use with AutoDAN and other
    jailbreak generation engines in Chimera.
    """
    service = get_benchmark_service()

    if not service.get_dataset(dataset_name):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not found",
        )

    return service.export_prompts_for_jailbreak(
        dataset_name=dataset_name,
        risk_area=risk_area,
        limit=limit,
    )
