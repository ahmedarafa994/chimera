"""
Repository Layer

This module provides the repository pattern for data access abstraction.
All database operations should go through repositories to maintain clean
separation between business logic and data access.
"""

from app.repositories.base_repository import BaseRepository

__all__ = ["BaseRepository"]
