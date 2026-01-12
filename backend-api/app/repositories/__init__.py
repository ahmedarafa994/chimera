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

__all__ = [
    "BaseRepository",
    "PromptLibraryRepository",
    "get_prompt_library_repository",
    "RepositoryError",
    "EntityNotFoundError",
    "DuplicateEntityError",
    "ValidationError",
]
