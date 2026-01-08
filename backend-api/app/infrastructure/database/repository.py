"""Repository pattern for jailbreak quality metrics database operations."""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.core.config import settings

from .models import Base, FeedbackHistory, JailbreakExecution, QualityMetric, TechniquePerformance


class QualityMetricsRepository:
    """Repository for quality metrics database operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_execution(
        self,
        technique_id: str,
        original_prompt: str,
        jailbroken_prompt: str,
        target_model: str,
        target_provider: str,
        execution_status: str,
        execution_time_ms: float,
        llm_response_text: str | None = None,
        llm_response_metadata: dict[str, Any] | None = None,
    ) -> JailbreakExecution:
        """Create a new jailbreak execution record."""
        execution = JailbreakExecution(
            technique_id=technique_id,
            original_prompt=original_prompt,
            jailbroken_prompt=jailbroken_prompt,
            target_model=target_model,
            target_provider=target_provider,
            execution_status=execution_status,
            execution_time_ms=execution_time_ms,
            llm_response_text=llm_response_text,
            llm_response_metadata=llm_response_metadata,
        )
        self.session.add(execution)
        await self.session.commit()
        await self.session.refresh(execution)
        return execution

    async def create_quality_metrics(
        self,
        execution_id: str,
        effectiveness_score: float,
        naturalness_score: float,
        detectability_score: float,
        coherence_score: float,
        semantic_success_score: float,
        overall_quality_score: float,
        semantic_success_level: str,
        bypass_indicators: list[str],
        safety_trigger_detected: bool,
        response_coherence: float,
        confidence_interval_lower: float,
        confidence_interval_upper: float,
        analysis_duration_ms: float,
    ) -> QualityMetric:
        """Create quality metrics for an execution."""
        metrics = QualityMetric(
            execution_id=execution_id,
            effectiveness_score=effectiveness_score,
            naturalness_score=naturalness_score,
            detectability_score=detectability_score,
            coherence_score=coherence_score,
            semantic_success_score=semantic_success_score,
            overall_quality_score=overall_quality_score,
            semantic_success_level=semantic_success_level,
            bypass_indicators={"indicators": bypass_indicators},
            safety_trigger_detected=safety_trigger_detected,
            response_coherence=response_coherence,
            confidence_interval_lower=confidence_interval_lower,
            confidence_interval_upper=confidence_interval_upper,
            analysis_duration_ms=analysis_duration_ms,
        )
        self.session.add(metrics)
        await self.session.commit()
        await self.session.refresh(metrics)
        return metrics

    async def get_execution_by_id(self, execution_id: str) -> JailbreakExecution | None:
        """Get execution by ID with quality metrics."""
        stmt = (
            select(JailbreakExecution)
            .where(JailbreakExecution.execution_id == execution_id)
            .options(selectinload(JailbreakExecution.quality_metrics))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_quality_metrics_by_execution(self, execution_id: str) -> QualityMetric | None:
        """Get quality metrics for an execution."""
        stmt = (
            select(QualityMetric)
            .where(QualityMetric.execution_id == execution_id)
            .options(selectinload(QualityMetric.execution))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_technique_performance(
        self,
        technique_id: str,
        model_name: str | None = None,
        provider_name: str | None = None,
        time_window_hours: int = 168,
    ) -> TechniquePerformance | None:
        """Get performance statistics for a technique."""
        stmt = select(TechniquePerformance).where(TechniquePerformance.technique_id == technique_id)
        if model_name:
            stmt = stmt.where(TechniquePerformance.model_name == model_name)
        if provider_name:
            stmt = stmt.where(TechniquePerformance.provider_name == provider_name)

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_technique_performance(
        self, technique_id: str, model_name: str, provider_name: str, time_window_hours: int = 168
    ) -> TechniquePerformance:
        """Update aggregated performance statistics for a technique."""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

        # Calculate statistics from executions
        stmt = (
            select(
                func.count(JailbreakExecution.execution_id).label("total"),
                func.avg(QualityMetric.semantic_success_score).label("avg_semantic"),
                func.avg(QualityMetric.effectiveness_score).label("avg_effectiveness"),
                func.avg(QualityMetric.naturalness_score).label("avg_naturalness"),
                func.avg(QualityMetric.detectability_score).label("avg_detectability"),
                func.avg(QualityMetric.coherence_score).label("avg_coherence"),
                func.stddev(QualityMetric.overall_quality_score).label("std_dev"),
            )
            .select_from(JailbreakExecution)
            .join(QualityMetric, JailbreakExecution.execution_id == QualityMetric.execution_id)
            .where(
                and_(
                    JailbreakExecution.technique_id == technique_id,
                    JailbreakExecution.target_model == model_name,
                    JailbreakExecution.target_provider == provider_name,
                    JailbreakExecution.created_at >= cutoff_time,
                )
            )
        )

        result = await self.session.execute(stmt)
        stats = result.one()

        # Upsert performance record
        perf_stmt = select(TechniquePerformance).where(
            and_(
                TechniquePerformance.technique_id == technique_id,
                TechniquePerformance.model_name == model_name,
                TechniquePerformance.provider_name == provider_name,
            )
        )
        perf_result = await self.session.execute(perf_stmt)
        performance = perf_result.scalar_one_or_none()

        if performance:
            performance.total_attempts = stats.total or 0
            performance.semantic_success_rate = stats.avg_semantic or 0.0
            performance.avg_effectiveness = stats.avg_effectiveness or 0.0
            performance.avg_naturalness = stats.avg_naturalness or 0.0
            performance.avg_detectability = stats.avg_detectability or 0.0
            performance.avg_coherence = stats.avg_coherence or 0.0
            performance.std_deviation = stats.std_dev or 0.0
            performance.last_updated = datetime.utcnow()
        else:
            performance = TechniquePerformance(
                technique_id=technique_id,
                model_name=model_name,
                provider_name=provider_name,
                total_attempts=stats.total or 0,
                semantic_success_rate=stats.avg_semantic or 0.0,
                avg_effectiveness=stats.avg_effectiveness or 0.0,
                avg_naturalness=stats.avg_naturalness or 0.0,
                avg_detectability=stats.avg_detectability or 0.0,
                avg_coherence=stats.avg_coherence or 0.0,
                std_deviation=stats.std_dev or 0.0,
                time_window_hours=time_window_hours,
            )
            self.session.add(performance)

        await self.session.commit()
        await self.session.refresh(performance)
        return performance

    async def create_feedback(
        self,
        execution_id: str,
        feedback_type: str,
        success_indicator: bool,
        user_feedback: str | None = None,
        llm_evaluation_result: dict[str, Any] | None = None,
        rating: int | None = None,
        comments: str | None = None,
        user_id: str | None = None,
    ) -> FeedbackHistory:
        """Create feedback record."""
        feedback = FeedbackHistory(
            execution_id=execution_id,
            feedback_type=feedback_type,
            success_indicator=success_indicator,
            user_feedback=user_feedback,
            llm_evaluation_result=llm_evaluation_result,
            rating=rating,
            comments=comments,
            user_id=user_id,
        )
        self.session.add(feedback)
        await self.session.commit()
        await self.session.refresh(feedback)
        return feedback

    async def get_recent_executions(
        self, technique_id: str | None = None, limit: int = 50
    ) -> list[JailbreakExecution]:
        """Get recent executions with quality metrics."""
        stmt = (
            select(JailbreakExecution)
            .options(selectinload(JailbreakExecution.quality_metrics))
            .order_by(desc(JailbreakExecution.created_at))
            .limit(limit)
        )

        if technique_id:
            stmt = stmt.where(JailbreakExecution.technique_id == technique_id)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())


# Database session management
class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def create_tables(self):
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_session(self) -> AsyncSession:
        """Get a new database session."""
        return self.async_session()

    async def close(self):
        """Close database connections."""
        await self.engine.dispose()


# Global database manager instance
db_manager: DatabaseManager | None = None


async def get_db_manager() -> DatabaseManager:
    """Get or create database manager."""
    global db_manager
    if db_manager is None:
        database_url = getattr(settings, "DATABASE_URL", "postgresql+asyncpg://localhost/chimera")
        db_manager = DatabaseManager(database_url)
        await db_manager.create_tables()
    return db_manager


async def get_repository() -> QualityMetricsRepository:
    """Get repository instance with session."""
    manager = await get_db_manager()
    session = await manager.get_session()
    return QualityMetricsRepository(session)
