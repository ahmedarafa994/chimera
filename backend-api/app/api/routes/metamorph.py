"""Metamorph API Router
Exposes the MetamorphService for dynamic prompt transformations.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.metamorph_service import TransformationMode, metamorph_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/metamorph", tags=["metamorph"])


class TransformRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to transform")
    suite: str = Field(default="gemini_brain_optimization")
    potency: int = Field(default=5, ge=1, le=10)
    mode: str = Field(default="layered")


class TransformResponse(BaseModel):
    original: str
    transformed: str
    suite_used: str
    techniques_applied: list[str]
    layers: int


class DatasetInfo(BaseModel):
    name: str
    count: int


class SuiteInfo(BaseModel):
    name: str
    transformers_count: int
    framers_count: int
    obfuscators_count: int


class StatusResponse(BaseModel):
    initialized: bool
    suites_available: list[str]
    datasets_loaded: list[DatasetInfo]
    total_techniques: int


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get MetamorphService status."""
    suites = metamorph_service.get_available_suites()
    stats = metamorph_service.get_dataset_stats()

    total = 0
    for s in suites:
        info = metamorph_service.get_suite_info(s)
        if info:
            total += (
                info.get("transformers_count", 0)
                + info.get("framers_count", 0)
                + info.get("obfuscators_count", 0)
            )

    return StatusResponse(
        initialized=metamorph_service.is_initialized,
        suites_available=suites,
        datasets_loaded=[
            DatasetInfo(name=n, count=i.get("count", 0))
            for n, i in stats.get("datasets", {}).items()
        ],
        total_techniques=total,
    )


@router.get("/suites", response_model=list[SuiteInfo])
async def list_suites():
    """List all transformation suites."""
    suites = metamorph_service.get_available_suites()
    result = []
    for s in suites:
        info = metamorph_service.get_suite_info(s)
        if info:
            result.append(
                SuiteInfo(
                    name=s,
                    transformers_count=info.get("transformers_count", 0),
                    framers_count=info.get("framers_count", 0),
                    obfuscators_count=info.get("obfuscators_count", 0),
                ),
            )
    return result


@router.post("/transform", response_model=TransformResponse)
async def transform_prompt(request: TransformRequest):
    """Transform a prompt using the specified suite."""
    suites = metamorph_service.get_available_suites()
    if request.suite not in suites:
        raise HTTPException(status_code=400, detail="Suite not found")

    mode_map = {
        "basic": TransformationMode.BASIC,
        "layered": TransformationMode.LAYERED,
        "recursive": TransformationMode.RECURSIVE,
        "adaptive": TransformationMode.ADAPTIVE,
    }
    mode = mode_map.get(request.mode, TransformationMode.LAYERED)

    result = metamorph_service.transform(
        prompt=request.prompt,
        suite_name=request.suite,
        potency=request.potency,
        mode=mode,
    )

    return TransformResponse(
        original=result.original_prompt,
        transformed=result.transformed_prompt,
        suite_used=result.metadata.get("suite", request.suite),
        techniques_applied=result.techniques_applied,
        layers=result.layers_count,
    )


@router.get("/datasets")
async def list_datasets():
    """List loaded datasets."""
    return metamorph_service.get_dataset_stats()


@router.get("/datasets/{name}/sample")
async def sample_dataset(name: str, count: Annotated[int, Query(ge=1, le=10)] = 1):
    """Get random samples from a dataset."""
    samples = []
    for _ in range(count):
        s = metamorph_service.get_random_prompt(name)
        if s:
            samples.append(s)
    if not samples:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"dataset": name, "samples": samples}
