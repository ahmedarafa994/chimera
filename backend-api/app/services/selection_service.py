"""
Selection Service - Business logic for provider/model selection management.

This module provides high-level operations for managing user and session
selections, including validation, caching, and coordination with the
unified provider registry.
"""

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Selection, SelectionScope
from app.repositories.selection_repository import get_selection_repository
from app.services.unified_provider_registry import unified_registry

logger = logging.getLogger(__name__)


class SelectionService:
    """
    Service layer for selection management.

    Provides business logic for creating, retrieving, updating, and deleting
    selections with validation and coordination with the provider registry.
    """

    async def get_session_selection(
        self,
        session_id: str,
        session: AsyncSession,
    ) -> Optional[Selection]:
        """
        Get selection for a session.

        Args:
            session_id: Session identifier
            session: Database session

        Returns:
            Selection if found and valid, None otherwise
        """
        try:
            repository = get_selection_repository(session)
            selection = await repository.get_by_session_id(session_id)

            if selection:
                # Validate selection is still valid
                if self._validate_selection(selection):
                    logger.debug(f"Retrieved valid session selection: {selection.provider_id}/{selection.model_id}")
                    return selection
                else:
                    logger.warning(
                        f"Session selection {selection.provider_id}/{selection.model_id} is no longer valid. "
                        "Deleting from database."
                    )
                    await repository.delete_by_session_id(session_id)
                    return None

            return None

        except Exception as e:
            logger.error(f"Failed to get session selection: {e}", exc_info=True)
            return None

    async def get_user_selection(
        self,
        user_id: str,
        session: AsyncSession,
    ) -> Optional[Selection]:
        """
        Get default selection for a user.

        Args:
            user_id: User identifier
            session: Database session

        Returns:
            Selection if found and valid, None otherwise
        """
        try:
            repository = get_selection_repository(session)
            selection = await repository.get_by_user_id(user_id)

            if selection:
                # Validate selection is still valid
                if self._validate_selection(selection):
                    logger.debug(f"Retrieved valid user selection: {selection.provider_id}/{selection.model_id}")
                    return selection
                else:
                    logger.warning(
                        f"User selection {selection.provider_id}/{selection.model_id} is no longer valid. "
                        "Deleting from database."
                    )
                    await repository.delete_by_user_id(user_id)
                    return None

            return None

        except Exception as e:
            logger.error(f"Failed to get user selection: {e}", exc_info=True)
            return None

    async def set_session_selection(
        self,
        session_id: str,
        provider_id: str,
        model_id: str,
        session: AsyncSession,
        user_id: Optional[str] = None,
    ) -> Selection:
        """
        Set selection for a session.

        Args:
            session_id: Session identifier
            provider_id: Provider identifier
            model_id: Model identifier
            session: Database session
            user_id: Optional user identifier

        Returns:
            Created/updated Selection

        Raises:
            ValueError: If provider/model combination is invalid
        """
        # Validate selection
        if not unified_registry.validate_selection(provider_id, model_id):
            raise ValueError(
                f"Invalid provider/model combination: {provider_id}/{model_id}. "
                "Provider or model may not be registered or enabled."
            )

        try:
            repository = get_selection_repository(session)
            selection = await repository.create_or_update(
                provider_id=provider_id,
                model_id=model_id,
                session_id=session_id,
                user_id=user_id,
            )

            logger.info(f"Set session selection: {provider_id}/{model_id} for session {session_id}")
            return selection

        except Exception as e:
            logger.error(f"Failed to set session selection: {e}", exc_info=True)
            raise

    async def set_user_selection(
        self,
        user_id: str,
        provider_id: str,
        model_id: str,
        session: AsyncSession,
    ) -> Selection:
        """
        Set default selection for a user.

        Args:
            user_id: User identifier
            provider_id: Provider identifier
            model_id: Model identifier
            session: Database session

        Returns:
            Created/updated Selection

        Raises:
            ValueError: If provider/model combination is invalid
        """
        # Validate selection
        if not unified_registry.validate_selection(provider_id, model_id):
            raise ValueError(
                f"Invalid provider/model combination: {provider_id}/{model_id}. "
                "Provider or model may not be registered or enabled."
            )

        try:
            repository = get_selection_repository(session)
            selection = await repository.create_or_update(
                provider_id=provider_id,
                model_id=model_id,
                user_id=user_id,
            )

            logger.info(f"Set user selection: {provider_id}/{model_id} for user {user_id}")
            return selection

        except Exception as e:
            logger.error(f"Failed to set user selection: {e}", exc_info=True)
            raise

    async def delete_session_selection(
        self,
        session_id: str,
        session: AsyncSession,
    ) -> bool:
        """
        Delete selection for a session.

        Args:
            session_id: Session identifier
            session: Database session

        Returns:
            True if deleted, False if not found
        """
        try:
            repository = get_selection_repository(session)
            deleted = await repository.delete_by_session_id(session_id)

            if deleted:
                logger.info(f"Deleted session selection for {session_id}")
            else:
                logger.debug(f"No session selection to delete for {session_id}")

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete session selection: {e}", exc_info=True)
            return False

    async def delete_user_selection(
        self,
        user_id: str,
        session: AsyncSession,
    ) -> bool:
        """
        Delete default selection for a user.

        Args:
            user_id: User identifier
            session: Database session

        Returns:
            True if deleted, False if not found
        """
        try:
            repository = get_selection_repository(session)
            deleted = await repository.delete_by_user_id(user_id)

            if deleted:
                logger.info(f"Deleted user selection for {user_id}")
            else:
                logger.debug(f"No user selection to delete for {user_id}")

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete user selection: {e}", exc_info=True)
            return False

    def _validate_selection(self, selection: Selection) -> bool:
        """
        Validate that a selection is still valid.

        Checks that the provider and model are still registered and enabled.

        Args:
            selection: Selection to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            return unified_registry.validate_selection(
                selection.provider_id,
                selection.model_id
            )
        except Exception as e:
            logger.warning(f"Selection validation failed: {e}")
            return False


# Singleton instance
selection_service = SelectionService()
