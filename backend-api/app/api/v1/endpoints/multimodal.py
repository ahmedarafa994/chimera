"""Multi-Modal Attack Testing Endpoints.

Phase 4 innovation feature for next-generation security:
- Vision+text and audio+text attack capabilities
- Image captioning vulnerability testing
- Unified reporting across modalities
- Future-proofing for multi-modal LLMs
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from app.core.auth import get_current_user

router = APIRouter()


class ModalityType(str, Enum):
    TEXT_ONLY = "text_only"
    VISION_TEXT = "vision_text"
    AUDIO_TEXT = "audio_text"
    VIDEO_TEXT = "video_text"
    MULTI_MODAL = "multi_modal"


class AttackVector(str, Enum):
    VISUAL_PROMPT_INJECTION = "visual_prompt_injection"
    IMAGE_CAPTION_MANIPULATION = "image_caption_manipulation"
    STEGANOGRAPHIC_EMBEDDING = "steganographic_embedding"
    AUDIO_PROMPT_INJECTION = "audio_prompt_injection"
    ADVERSARIAL_IMAGES = "adversarial_images"
    CROSS_MODAL_CONFUSION = "cross_modal_confusion"
    MULTIMODAL_JAILBREAKING = "multimodal_jailbreaking"
    CONTEXT_HIJACKING = "context_hijacking"


class TestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


class VulnerabilityLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MediaFile(BaseModel):
    file_id: str
    filename: str
    content_type: str
    file_size: int
    file_path: str
    uploaded_at: str
    width: int | None = None
    height: int | None = None
    duration: float | None = None
    format: str | None = None


class MultiModalPrompt(BaseModel):
    prompt_id: str
    text_content: str
    media_files: list[MediaFile]
    attack_vector: AttackVector
    modality_type: ModalityType
    instructions: str | None = None
    context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultiModalTestExecution(BaseModel):
    execution_id: str
    prompt_id: str
    model_name: str
    provider: str
    started_at: str
    completed_at: str | None = None
    duration_seconds: float | None = None
    status: TestStatus
    input_prompt: MultiModalPrompt
    model_response: str | None = None
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    vulnerability_detected: bool = False
    vulnerability_level: VulnerabilityLevel = VulnerabilityLevel.NONE
    confidence_score: float = 0.0
    jailbreak_success: bool = False
    information_leaked: bool = False
    harmful_content_generated: bool = False
    analysis_results: dict[str, Any] = Field(default_factory=dict)
    cross_modal_consistency: float | None = None
    attention_analysis: dict[str, Any] | None = None
    error_message: str | None = None
    executed_by: str
    workspace_id: str | None = None


class MultiModalTestSuite(BaseModel):
    suite_id: str
    name: str
    description: str | None = None
    target_models: list[str] = Field(default_factory=list)
    attack_vectors: list[AttackVector] = Field(default_factory=list)
    modality_types: list[ModalityType] = Field(default_factory=list)
    timeout_seconds: int = 300
    parallel_execution: bool = False
    max_concurrent_tests: int = 1
    enable_cross_modal_analysis: bool = False
    generate_adversarial_examples: bool = False
    analyze_attention_patterns: bool = False
    created_by: str
    workspace_id: str | None = None
    created_at: str
    updated_at: str
    total_tests: int = 0
    completed_tests: int = 0
    vulnerabilities_found: int = 0
    success_rate: float = 0.0
    executions: list[MultiModalTestExecution] = Field(default_factory=list)
    prompts: list[MultiModalPrompt] = Field(default_factory=list)


class MultiModalTestCreate(BaseModel):
    name: str
    description: str | None = None
    target_models: list[str] = Field(default_factory=list)
    attack_vectors: list[AttackVector] = Field(default_factory=list)
    modality_types: list[ModalityType] = Field(default_factory=list)
    workspace_id: str | None = None


media_storage: dict[str, MediaFile] = {}
prompts_storage: dict[str, MultiModalPrompt] = {}
suites_storage: dict[str, MultiModalTestSuite] = {}
executions_storage: dict[str, list[MultiModalTestExecution]] = {}


def _now() -> str:
    return datetime.utcnow().isoformat()


@router.post("/suites", response_model=MultiModalTestSuite)
async def create_test_suite(
    suite_data: MultiModalTestCreate,
    current_user: Annotated[Any, Depends(get_current_user)],
):
    """Create a new multi-modal test suite."""
    suite_id = str(uuid.uuid4())
    now = _now()
    user_id = current_user.id if hasattr(current_user, "id") else str(current_user)

    suite = MultiModalTestSuite(
        suite_id=suite_id,
        name=suite_data.name,
        description=suite_data.description,
        target_models=suite_data.target_models,
        attack_vectors=suite_data.attack_vectors,
        modality_types=suite_data.modality_types,
        created_by=user_id,
        workspace_id=suite_data.workspace_id,
        created_at=now,
        updated_at=now,
    )

    suites_storage[suite_id] = suite
    executions_storage[suite_id] = []

    return suite


@router.post("/media/upload")
async def upload_media_file(
    file: Annotated[UploadFile, File()] = ...,
    current_user: Any = Depends(get_current_user),
):
    """Upload media file for multi-modal testing."""
    file_id = str(uuid.uuid4())
    contents = await file.read()
    file_size = len(contents)

    media_file = MediaFile(
        file_id=file_id,
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        file_size=file_size,
        file_path=f"/tmp/{file_id}",
        uploaded_at=_now(),
    )
    media_storage[file_id] = media_file

    return {
        "file_id": media_file.file_id,
        "filename": media_file.filename,
        "content_type": media_file.content_type,
        "file_size": media_file.file_size,
        "metadata": {
            "width": media_file.width,
            "height": media_file.height,
            "duration": media_file.duration,
            "format": media_file.format,
        },
    }


@router.post("/prompts", response_model=MultiModalPrompt)
async def create_prompt(
    text_content: Annotated[str, Form()] = ...,
    attack_vector: Annotated[AttackVector, Form()] = ...,
    modality_type: Annotated[ModalityType, Form()] = ...,
    instructions: Annotated[str | None, Form()] = None,
    context: Annotated[str | None, Form()] = None,
    media_file_ids: Annotated[list[str] | None, Form()] = None,
    current_user: Any = Depends(get_current_user),
):
    """Create a multi-modal prompt with optional media files."""
    prompt_id = str(uuid.uuid4())
    media_files = []

    for file_id in media_file_ids or []:
        media = media_storage.get(file_id)
        if media:
            media_files.append(media)

    prompt = MultiModalPrompt(
        prompt_id=prompt_id,
        text_content=text_content,
        media_files=media_files,
        attack_vector=attack_vector,
        modality_type=modality_type,
        instructions=instructions,
        context=context,
        metadata={},
    )

    prompts_storage[prompt_id] = prompt
    return prompt


@router.get("/suites")
async def list_test_suites(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 10,
    modality_type: Annotated[ModalityType | None, Query()] = None,
    current_user: Any = Depends(get_current_user),
):
    """List multi-modal test suites."""
    suites = list(suites_storage.values())
    if modality_type:
        suites = [s for s in suites if modality_type in s.modality_types]

    total = len(suites)
    start = (page - 1) * page_size
    end = start + page_size
    paged = suites[start:end]

    return {
        "suites": paged,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_next": end < total,
        "has_prev": page > 1,
    }


@router.post("/suites/{suite_id}/execute")
async def execute_test_suite(
    suite_id: str, current_user: Annotated[Any, Depends(get_current_user)]
):
    """Execute a multi-modal test suite (simulated)."""
    suite = suites_storage.get(suite_id)
    if not suite:
        raise HTTPException(status_code=404, detail="Test suite not found")

    user_id = current_user.id if hasattr(current_user, "id") else str(current_user)
    now = _now()

    executions = executions_storage.get(suite_id, [])
    for prompt in suite.prompts:
        execution = MultiModalTestExecution(
            execution_id=str(uuid.uuid4()),
            prompt_id=prompt.prompt_id,
            model_name=(suite.target_models[0] if suite.target_models else "default"),
            provider="default",
            started_at=now,
            completed_at=now,
            duration_seconds=0.0,
            status=TestStatus.COMPLETED,
            input_prompt=prompt,
            model_response="",
            response_metadata={},
            executed_by=user_id,
            workspace_id=suite.workspace_id,
        )
        executions.append(execution)

    executions_storage[suite_id] = executions
    suite.executions = executions
    suite.total_tests = len(executions)
    suite.completed_tests = len(executions)
    suite.updated_at = now

    return {"message": "Execution started", "suite_id": suite_id}


@router.get("/suites/{suite_id}/executions")
async def list_suite_executions(
    suite_id: str,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 10,
    current_user: Any = Depends(get_current_user),
):
    """List executions for a test suite."""
    executions = executions_storage.get(suite_id, [])
    total = len(executions)
    start = (page - 1) * page_size
    end = start + page_size
    paged = executions[start:end]

    return {
        "executions": paged,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_next": end < total,
        "has_prev": page > 1,
    }


@router.delete("/suites/{suite_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_test_suite(
    suite_id: str, current_user: Annotated[Any, Depends(get_current_user)]
) -> None:
    """Delete a multi-modal test suite."""
    suites_storage.pop(suite_id, None)
    executions_storage.pop(suite_id, None)


@router.get("/analytics")
async def get_multimodal_analytics(
    workspace_id: str | None = None,
    current_user: Any = Depends(get_current_user),
):
    """Get multi-modal testing analytics."""
    total_suites = len(suites_storage)
    total_executions = sum(len(execs) for execs in executions_storage.values())

    return {
        "total_suites": total_suites,
        "total_executions": total_executions,
        "vulnerability_rate": 0,
        "modality_breakdown": {},
        "vulnerability_by_modality": {},
        "attack_vector_success": {},
        "model_robustness": {},
        "daily_execution_trend": [],
        "vulnerability_trend": [],
    }
