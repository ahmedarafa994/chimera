"""
Repository Layer

This module provides the repository pattern for data access abstraction.
All database operations should go through repositories to maintain clean
separation between business logic and data access.
"""

from app.repositories.base_repository import BaseRepository
from app.repositories.prompt_library_repository import (
    DuplicateEntityError,
    EntityNotFoundError,
    PromptLibraryRepository,
    RepositoryError,
    ValidationError,
    get_prompt_library_repository,
)
from app.repositories.user_repository import (
    EmailAlreadyExistsError,
    InvalidTokenError,
    UserAlreadyExistsError,
    UsernameAlreadyExistsError,
    UserNotFoundError,
    UserRepository,
    get_user_repository,
)

__all__ = [
    # Base repository
    "BaseRepository",
    "DuplicateEntityError",
    "EmailAlreadyExistsError",
    "EntityNotFoundError",
    "InvalidTokenError",
    # Prompt Library repository
    "PromptLibraryRepository",
    "RepositoryError",
    "UserAlreadyExistsError",
    # User repository exceptions
    "UserNotFoundError",
    # User repository
    "UserRepository",
    "UsernameAlreadyExistsError",
    "ValidationError",
    "get_prompt_library_repository",
    "get_user_repository",
]
