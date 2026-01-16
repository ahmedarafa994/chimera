"""
Repository Layer

This module provides the repository pattern for data access abstraction.
All database operations should go through repositories to maintain clean
separation between business logic and data access.
"""

from app.repositories.base_repository import BaseRepository
from app.repositories.user_repository import (
    EmailAlreadyExistsError,
    InvalidTokenError,
    UserAlreadyExistsError,
    UserNotFoundError,
    UserRepository,
    UsernameAlreadyExistsError,
    get_user_repository,
)
from app.repositories.prompt_library_repository import (
    DuplicateEntityError,
    EntityNotFoundError,
    PromptLibraryRepository,
    RepositoryError,
    ValidationError,
    get_prompt_library_repository,
)

__all__ = [
    # Base repository
    "BaseRepository",
    # User repository
    "UserRepository",
    "get_user_repository",
    # User repository exceptions
    "UserNotFoundError",
    "UserAlreadyExistsError",
    "EmailAlreadyExistsError",
    "UsernameAlreadyExistsError",
    "InvalidTokenError",
    # Prompt Library repository
    "PromptLibraryRepository",
    "get_prompt_library_repository",
    "RepositoryError",
    "EntityNotFoundError",
    "DuplicateEntityError",
    "ValidationError",
]
