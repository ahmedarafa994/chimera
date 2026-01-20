"""Security Assessment Management Endpoints.

Phase 2 feature for security testing workflow:
- Create and manage security assessments
- Execute vulnerability scans against LLM providers
- Track assessment progress and results
- Generate assessment reports
"""

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.database import get_db
from app.core.observability import get_logger
from app.db.models import Assessment, AssessmentStatus, User

logger = get_logger("chimera.api.assessments")
router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class AssessmentCreate(BaseModel):
    """Request model for creating an assessment."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    target_provider: str = Field(..., min_length=1)
    target_model: str = Field(..., min_length=1)
    target_config: dict[str, Any] = Field(default_factory=dict)
    technique_ids: list[str] = Field(default_factory=list)


class AssessmentUpdate(BaseModel):
    """Request model for updating an assessment."""

    name: str | None = None
    description: str | None = None
    status: AssessmentStatus | None = None


class AssessmentResponse(BaseModel):
    """Response model for an assessment."""

    id: int
    name: str
    description: str | None
    status: str
    target_provider: str
    target_model: str
    target_config: dict[str, Any]
    technique_ids: list[str]
    results: dict[str, Any]
    findings_count: int
    vulnerabilities_found: int
    risk_score: int
    risk_level: str
    created_at: datetime
    updated_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None

    class Config:
        from_attributes = True


class AssessmentListResponse(BaseModel):
    """Response model for listing assessments."""

    assessments: list[AssessmentResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/")
async def list_assessments(
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    status_filter: Annotated[AssessmentStatus | None, Query(description="Filter by status")] = None,
    search: Annotated[str | None, Query(description="Search by name")] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AssessmentListResponse:
    """List assessments for the current user with pagination and filtering."""
    try:
        query = db.query(Assessment).filter(Assessment.user_id == current_user.id)

        if status_filter:
            query = query.filter(Assessment.status == status_filter)

        if search:
            query = query.filter(Assessment.name.ilike(f"%{search}%"))

        total = query.count()
        offset = (page - 1) * page_size
        assessments = (
            query.order_by(Assessment.created_at.desc()).offset(offset).limit(page_size).all()
        )

        return AssessmentListResponse(
            assessments=[
                AssessmentResponse(
                    id=a.id,
                    name=a.name,
                    description=a.description,
                    status=a.status.value if a.status else "pending",
                    target_provider=a.target_provider,
                    target_model=a.target_model,
                    target_config=a.target_config or {},
                    technique_ids=a.technique_ids or [],
                    results=a.results or {},
                    findings_count=a.findings_count or 0,
                    vulnerabilities_found=a.vulnerabilities_found or 0,
                    risk_score=a.risk_score or 0,
                    risk_level=a.risk_level or "info",
                    created_at=a.created_at,
                    updated_at=a.updated_at,
                    started_at=a.started_at,
                    completed_at=a.completed_at,
                )
                for a in assessments
            ],
            total=total,
            page=page,
            page_size=page_size,
            has_next=offset + len(assessments) < total,
            has_prev=page > 1,
        )
    except Exception as e:
        logger.exception(f"Failed to list assessments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list assessments",
        ) from e


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_assessment(
    data: AssessmentCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> AssessmentResponse:
    """Create a new security assessment."""
    try:
        assessment = Assessment(
            user_id=current_user.id,
            name=data.name,
            description=data.description,
            target_provider=data.target_provider,
            target_model=data.target_model,
            target_config=data.target_config,
            technique_ids=data.technique_ids,
            status=AssessmentStatus.PENDING,
        )

        db.add(assessment)
        db.commit()
        db.refresh(assessment)

        logger.info(f"Created assessment {assessment.id} for user {current_user.id}")

        return AssessmentResponse(
            id=assessment.id,
            name=assessment.name,
            description=assessment.description,
            status=assessment.status.value,
            target_provider=assessment.target_provider,
            target_model=assessment.target_model,
            target_config=assessment.target_config or {},
            technique_ids=assessment.technique_ids or [],
            results={},
            findings_count=0,
            vulnerabilities_found=0,
            risk_score=0,
            risk_level="info",
            created_at=assessment.created_at,
            updated_at=assessment.updated_at,
            started_at=assessment.started_at,
            completed_at=assessment.completed_at,
        )
    except Exception as e:
        db.rollback()
        logger.exception(f"Failed to create assessment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create assessment",
        ) from e


@router.get("/{assessment_id}")
async def get_assessment(
    assessment_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> AssessmentResponse:
    """Get a specific assessment by ID."""
    assessment = (
        db.query(Assessment)
        .filter(Assessment.id == assessment_id, Assessment.user_id == current_user.id)
        .first()
    )

    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment {assessment_id} not found",
        )

    return AssessmentResponse(
        id=assessment.id,
        name=assessment.name,
        description=assessment.description,
        status=assessment.status.value if assessment.status else "pending",
        target_provider=assessment.target_provider,
        target_model=assessment.target_model,
        target_config=assessment.target_config or {},
        technique_ids=assessment.technique_ids or [],
        results=assessment.results or {},
        findings_count=assessment.findings_count or 0,
        vulnerabilities_found=assessment.vulnerabilities_found or 0,
        risk_score=assessment.risk_score or 0,
        risk_level=assessment.risk_level or "info",
        created_at=assessment.created_at,
        updated_at=assessment.updated_at,
        started_at=assessment.started_at,
        completed_at=assessment.completed_at,
    )


@router.patch("/{assessment_id}")
async def update_assessment(
    assessment_id: int,
    data: AssessmentUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> AssessmentResponse:
    """Update an assessment."""
    assessment = (
        db.query(Assessment)
        .filter(Assessment.id == assessment_id, Assessment.user_id == current_user.id)
        .first()
    )

    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment {assessment_id} not found",
        )

    if data.name is not None:
        assessment.name = data.name
    if data.description is not None:
        assessment.description = data.description
    if data.status is not None:
        assessment.status = data.status
        if data.status == AssessmentStatus.RUNNING and not assessment.started_at:
            assessment.started_at = datetime.utcnow()
        elif data.status == AssessmentStatus.COMPLETED and not assessment.completed_at:
            assessment.completed_at = datetime.utcnow()

    db.commit()
    db.refresh(assessment)

    return AssessmentResponse(
        id=assessment.id,
        name=assessment.name,
        description=assessment.description,
        status=assessment.status.value if assessment.status else "pending",
        target_provider=assessment.target_provider,
        target_model=assessment.target_model,
        target_config=assessment.target_config or {},
        technique_ids=assessment.technique_ids or [],
        results=assessment.results or {},
        findings_count=assessment.findings_count or 0,
        vulnerabilities_found=assessment.vulnerabilities_found or 0,
        risk_score=assessment.risk_score or 0,
        risk_level=assessment.risk_level or "info",
        created_at=assessment.created_at,
        updated_at=assessment.updated_at,
        started_at=assessment.started_at,
        completed_at=assessment.completed_at,
    )


@router.delete("/{assessment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_assessment(
    assessment_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Delete an assessment."""
    assessment = (
        db.query(Assessment)
        .filter(Assessment.id == assessment_id, Assessment.user_id == current_user.id)
        .first()
    )

    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment {assessment_id} not found",
        )

    db.delete(assessment)
    db.commit()
    logger.info(f"Deleted assessment {assessment_id}")


@router.post("/{assessment_id}/run")
async def run_assessment(
    assessment_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> AssessmentResponse:
    """Start running an assessment."""
    assessment = (
        db.query(Assessment)
        .filter(Assessment.id == assessment_id, Assessment.user_id == current_user.id)
        .first()
    )

    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment {assessment_id} not found",
        )

    if assessment.status in [AssessmentStatus.RUNNING, AssessmentStatus.COMPLETED]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Assessment {assessment_id} is already {assessment.status.value}",
        )

    # Update status to running
    assessment.status = AssessmentStatus.RUNNING
    assessment.started_at = datetime.utcnow()
    assessment.updated_at = datetime.utcnow()

    # TODO: Add actual assessment execution logic here
    # This would typically involve:
    # 1. Loading the assessment configuration
    # 2. Running the specified techniques against the target model
    # 3. Collecting and analyzing results
    # 4. Updating the assessment with findings

    db.commit()
    db.refresh(assessment)

    logger.info(f"Started assessment {assessment_id} for user {current_user.id}")

    return AssessmentResponse(
        id=assessment.id,
        name=assessment.name,
        description=assessment.description,
        status=assessment.status.value,
        target_provider=assessment.target_provider,
        target_model=assessment.target_model,
        target_config=assessment.target_config or {},
        technique_ids=assessment.technique_ids or [],
        results=assessment.results or {},
        findings_count=assessment.findings_count or 0,
        vulnerabilities_found=assessment.vulnerabilities_found or 0,
        risk_score=assessment.risk_score or 0,
        risk_level=assessment.risk_level or "info",
        created_at=assessment.created_at,
        updated_at=assessment.updated_at,
        started_at=assessment.started_at,
        completed_at=assessment.completed_at,
    )
