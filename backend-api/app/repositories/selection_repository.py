"""
Selection Repository - Database access layer for provider/model selections.

This module provides database operations for persisting and retrieving
user and session selections, implementing the Session Preference tier
of the three-tier selection hierarchy.
"""

import logging
from datetime import datetime

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Selection, SelectionScope
from app.infrastructure.database.models import SelectionModel

logger = logging.getLogger(__name__)


class SelectionRepository:
    """
    Repository for selection database operations.

    Provides methods for creating, retrieving, updating, and deleting
    user and session selections from the database.
    """

    def __init__(self, session: AsyncSession):
        """
        Initialize selection repository.

        Args:
            session: SQLAlchemy async session
        """
        self._session = session

    async def get_by_session_id(self, session_id: str) -> Selection | None:
        """
        Get selection for a session.

        Args:
            session_id: Session identifier

        Returns:
            Selection if found, None otherwise
        """
        try:
            stmt = select(SelectionModel).where(SelectionModel.session_id == session_id)
            result = await self._session.execute(stmt)
            model = result.scalar_one_or_none()

            if model:
                logger.debug(
                    f"Found selection for session {session_id}: {model.provider_id}/{model.model_id}"
                )
                return Selection(
                    provider_id=model.provider_id,
                    model_id=model.model_id,
                    scope=SelectionScope.SESSION,
                    session_id=model.session_id,
                    user_id=model.user_id,
                )

            logger.debug(f"No selection found for session {session_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to get selection for session {session_id}: {e}", exc_info=True)
            return None

    async def get_by_user_id(self, user_id: str) -> Selection | None:
        """
        Get default selection for a user.

        Args:
            user_id: User identifier

        Returns:
            Selection if found, None otherwise
        """
        try:
            stmt = select(SelectionModel).where(
                and_(SelectionModel.user_id == user_id, SelectionModel.session_id.is_(None))
            )
            result = await self._session.execute(stmt)
            model = result.scalar_one_or_none()

            if model:
                logger.debug(
                    f"Found selection for user {user_id}: {model.provider_id}/{model.model_id}"
                )
                return Selection(
                    provider_id=model.provider_id,
                    model_id=model.model_id,
                    scope=SelectionScope.SESSION,
                    user_id=model.user_id,
                )

            logger.debug(f"No selection found for user {user_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to get selection for user {user_id}: {e}", exc_info=True)
            return None

    async def create_or_update(
        self,
        provider_id: str,
        model_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> Selection:
        """
        Create or update a selection.

        If a selection already exists for the session_id or user_id,
        it will be updated. Otherwise, a new selection is created.

        Args:
            provider_id: Provider identifier
            model_id: Model identifier
            session_id: Optional session identifier
            user_id: Optional user identifier

        Returns:
            Created or updated Selection

        Raises:
            ValueError: If neither session_id nor user_id is provided
        """
        if not session_id and not user_id:
            raise ValueError("Either session_id or user_id must be provided")

        try:
            # Check if selection exists
            if session_id:
                stmt = select(SelectionModel).where(SelectionModel.session_id == session_id)
            else:
                stmt = select(SelectionModel).where(
                    and_(SelectionModel.user_id == user_id, SelectionModel.session_id.is_(None))
                )

            result = await self._session.execute(stmt)
            model = result.scalar_one_or_none()

            if model:
                # Update existing
                model.provider_id = provider_id
                model.model_id = model_id
                model.updated_at = datetime.utcnow()
                logger.info(f"Updated selection for session={session_id}, user={user_id}")
            else:
                # Create new
                model = SelectionModel(
                    provider_id=provider_id,
                    model_id=model_id,
                    session_id=session_id,
                    user_id=user_id,
                )
                self._session.add(model)
                logger.info(f"Created selection for session={session_id}, user={user_id}")

            await self._session.commit()
            await self._session.refresh(model)

            return Selection(
                provider_id=model.provider_id,
                model_id=model.model_id,
                scope=SelectionScope.SESSION,
                session_id=model.session_id,
                user_id=model.user_id,
            )

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to create/update selection: {e}", exc_info=True)
            raise

    async def delete_by_session_id(self, session_id: str) -> bool:
        """
        Delete selection for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        try:
            stmt = select(SelectionModel).where(SelectionModel.session_id == session_id)
            result = await self._session.execute(stmt)
            model = result.scalar_one_or_none()

            if model:
                await self._session.delete(model)
                await self._session.commit()
                logger.info(f"Deleted selection for session {session_id}")
                return True

            logger.debug(f"No selection found to delete for session {session_id}")
            return False

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to delete selection for session {session_id}: {e}", exc_info=True)
            return False

    async def delete_by_user_id(self, user_id: str) -> bool:
        """
        Delete default selection for a user.

        Args:
            user_id: User identifier

        Returns:
            True if deleted, False if not found
        """
        try:
            stmt = select(SelectionModel).where(
                and_(SelectionModel.user_id == user_id, SelectionModel.session_id.is_(None))
            )
            result = await self._session.execute(stmt)
            model = result.scalar_one_or_none()

            if model:
                await self._session.delete(model)
                await self._session.commit()
                logger.info(f"Deleted selection for user {user_id}")
                return True

            logger.debug(f"No selection found to delete for user {user_id}")
            return False

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to delete selection for user {user_id}: {e}", exc_info=True)
            return False


# Singleton instance getter
def get_selection_repository(session: AsyncSession) -> SelectionRepository:
    """
    Get selection repository instance.

    Args:
        session: SQLAlchemy async session

    Returns:
        SelectionRepository instance
    """
    return SelectionRepository(session)
