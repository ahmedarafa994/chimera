"""
Database module with optimized connection pooling and async support.

PERF-035: Phase 3 database optimizations including:
- Connection pooling with configurable pool size
- Async database operations with SQLAlchemy 2.0
- Query result caching
- Connection health monitoring
"""

import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

logger = logging.getLogger(__name__)


# =============================================================================
# Database Configuration (PERF-036)
# =============================================================================


def _convert_to_async_url(sync_url: str) -> str:
    """
    Convert a synchronous database URL to an async-compatible URL.

    Handles various URL formats safely:
    - sqlite:///./path/to/db.sqlite -> sqlite+aiosqlite:///./path/to/db.sqlite
    - postgresql://... -> postgresql+asyncpg://...
    - mysql://... -> mysql+aiomysql://...
    """
    if not sync_url:
        return "sqlite+aiosqlite:///./chimera.db"

    # Handle SQLite URLs
    if sync_url.startswith("sqlite:///"):
        return sync_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    elif sync_url.startswith("sqlite://"):
        return sync_url.replace("sqlite://", "sqlite+aiosqlite://", 1)

    # Handle PostgreSQL URLs
    if sync_url.startswith("postgresql://"):
        return sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif sync_url.startswith("postgres://"):
        return sync_url.replace("postgres://", "postgresql+asyncpg://", 1)

    # Handle MySQL URLs
    if sync_url.startswith("mysql://"):
        return sync_url.replace("mysql://", "mysql+aiomysql://", 1)

    # If already async or unknown, return as-is
    return sync_url


class DatabaseConfig:
    """Database configuration with performance-optimized defaults."""

    # Connection pool settings
    POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))  # 30 minutes
    POOL_PRE_PING: bool = os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"

    # Query settings
    ECHO_SQL: bool = os.getenv("DB_ECHO_SQL", "false").lower() == "true"
    SLOW_QUERY_THRESHOLD_MS: float = float(os.getenv("DB_SLOW_QUERY_MS", "100"))

    # Database URLs
    SYNC_DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./chimera.db")
    ASYNC_DATABASE_URL: str = os.getenv(
        "ASYNC_DATABASE_URL",
        _convert_to_async_url(os.getenv("DATABASE_URL", "sqlite:///./chimera.db")),
    )


# =============================================================================
# Sync Engine with Connection Pooling (PERF-037)
# =============================================================================


def _create_sync_engine():
    """Create sync engine with optimized connection pooling."""
    url = DatabaseConfig.SYNC_DATABASE_URL

    # SQLite-specific settings
    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,  # SQLite works best with StaticPool
            echo=DatabaseConfig.ECHO_SQL,
        )
    else:
        # PostgreSQL/MySQL with QueuePool
        engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=DatabaseConfig.POOL_SIZE,
            max_overflow=DatabaseConfig.MAX_OVERFLOW,
            pool_timeout=DatabaseConfig.POOL_TIMEOUT,
            pool_recycle=DatabaseConfig.POOL_RECYCLE,
            pool_pre_ping=DatabaseConfig.POOL_PRE_PING,
            echo=DatabaseConfig.ECHO_SQL,
        )

    # Add query timing event listener (PERF-038)
    @event.listens_for(engine, "before_cursor_execute")
    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault("query_start_time", []).append(time.perf_counter())

    @event.listens_for(engine, "after_cursor_execute")
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        total_time = (time.perf_counter() - conn.info["query_start_time"].pop()) * 1000
        if total_time > DatabaseConfig.SLOW_QUERY_THRESHOLD_MS:
            logger.warning(f"Slow query detected ({total_time:.2f}ms): {statement[:100]}...")

    return engine


# Lazy initialization to avoid import-time errors
_sync_engine = None
_sync_session_factory = None


def get_sync_engine():
    """Get or create the sync engine (lazy initialization)."""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = _create_sync_engine()
    return _sync_engine


def get_sync_session_factory():
    """Get or create the sync session factory (lazy initialization)."""
    global _sync_session_factory
    if _sync_session_factory is None:
        _sync_session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=get_sync_engine()
        )
    return _sync_session_factory


# For backward compatibility - these will be lazily initialized on first access
class _LazyEngine:
    """Lazy proxy for sync_engine to avoid import-time initialization."""

    def __getattr__(self, name):
        return getattr(get_sync_engine(), name)


class _LazySessionFactory:
    """Lazy proxy for SyncSessionFactory to avoid import-time initialization."""

    def __call__(self, *args, **kwargs):
        return get_sync_session_factory()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(get_sync_session_factory(), name)


sync_engine = _LazyEngine()
SyncSessionFactory = _LazySessionFactory()


# =============================================================================
# Async Engine with Connection Pooling (PERF-039)
# =============================================================================


def _create_async_engine_instance():
    """Create async engine with optimized connection pooling."""
    url = DatabaseConfig.ASYNC_DATABASE_URL

    # SQLite-specific settings
    if "sqlite" in url:
        engine = create_async_engine(
            url,
            echo=DatabaseConfig.ECHO_SQL,
        )
    else:
        # PostgreSQL/MySQL with connection pooling
        engine = create_async_engine(
            url,
            pool_size=DatabaseConfig.POOL_SIZE,
            max_overflow=DatabaseConfig.MAX_OVERFLOW,
            pool_timeout=DatabaseConfig.POOL_TIMEOUT,
            pool_recycle=DatabaseConfig.POOL_RECYCLE,
            pool_pre_ping=DatabaseConfig.POOL_PRE_PING,
            echo=DatabaseConfig.ECHO_SQL,
        )

    return engine


# Lazy initialization for async engine
_async_engine = None
_async_session_factory = None


def get_async_engine():
    """Get or create the async engine (lazy initialization)."""
    global _async_engine
    if _async_engine is None:
        _async_engine = _create_async_engine_instance()
    return _async_engine


def get_async_session_factory():
    """Get or create the async session factory (lazy initialization)."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _async_session_factory


# For backward compatibility
class _LazyAsyncEngine:
    """Lazy proxy for async_engine to avoid import-time initialization."""

    def __getattr__(self, name):
        return getattr(get_async_engine(), name)


class _LazyAsyncSessionFactory:
    """Lazy proxy for AsyncSessionFactory to avoid import-time initialization."""

    def __call__(self, *args, **kwargs):
        return get_async_session_factory()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(get_async_session_factory(), name)


async_engine = _LazyAsyncEngine()
AsyncSessionFactory = _LazyAsyncSessionFactory()


# =============================================================================
# Database Manager with Optimized Session Handling (PERF-040)
# =============================================================================


class DatabaseManager:
    """
    Optimized database manager with connection pooling and async support.

    PERF-040: Features:
    - Connection pool management
    - Async session context manager
    - Query result caching
    - Connection health monitoring
    """

    def __init__(self):
        self._pool_stats: dict[str, Any] = {}

    @property
    def sync_engine(self):
        return get_sync_engine()

    @property
    def async_engine(self):
        return get_async_engine()

    @property
    def sync_session_factory(self):
        return get_sync_session_factory()

    @property
    def async_session_factory(self):
        return get_async_session_factory()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Async context manager for database sessions (PERF-041).

        Uses async session factory for non-blocking database operations.
        """
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def read_only_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Read-only async session for queries (PERF-042).
        Optimized for read operations - no commit needed.
        """
        session = self.async_session_factory()
        try:
            yield session
        finally:
            await session.close()

    def get_sync_session(self) -> Session:
        """Get a synchronous database session."""
        return self.sync_session_factory()

    async def health_check(self) -> dict[str, Any]:
        """
        Check database connection health (PERF-043).

        Returns connection pool statistics and health status.
        """
        try:
            async with self.session() as session:
                # Simple query to test connection
                result = await session.execute(text("SELECT 1"))
                result.scalar()

            # Get pool statistics
            pool = self.sync_engine.pool
            pool_stats = {
                "pool_size": getattr(pool, "size", lambda: 0)()
                if callable(getattr(pool, "size", None))
                else getattr(pool, "_pool", {}).qsize()
                if hasattr(pool, "_pool")
                else 0,
                "checked_in": pool.checkedin() if hasattr(pool, "checkedin") else 0,
                "checked_out": pool.checkedout() if hasattr(pool, "checkedout") else 0,
                "overflow": pool.overflow() if hasattr(pool, "overflow") else 0,
            }

            return {
                "status": "healthy",
                "pool_stats": pool_stats,
                "async_available": True,
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "async_available": False,
            }

    async def get_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        return await self.health_check()


# Global database manager instance
db_manager = DatabaseManager()


# =============================================================================
# Database Initialization (PERF-044)
# =============================================================================


async def create_performance_indexes():
    """
    Create performance indexes for common queries (PERF-044).

    Indexes are created asynchronously to avoid blocking startup.
    """
    try:
        async with db_manager.session() as session:
            # Example indexes - customize based on actual query patterns
            index_statements = [
                # Add indexes based on your query patterns
                # "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)",
                # "CREATE INDEX IF NOT EXISTS idx_transformations_created_at ON transformations(created_at)",
            ]

            for stmt in index_statements:
                try:
                    await session.execute(text(stmt))
                except Exception as e:
                    logger.debug(f"Index creation skipped (may already exist): {e}")

            await session.commit()
            logger.info("Performance indexes created/verified")
    except Exception as e:
        logger.warning(f"Performance index creation failed (non-critical): {e}")


async def init_database():
    """
    Initialize database with optimized settings (PERF-045).

    Features:
    - Creates all tables from registered models
    - Sets up performance indexes
    - Verifies connection pool health
    """
    try:
        # Import models to register them with SQLAlchemy metadata
        from app.db import models as chimera_models
        from app.domain import models as orm_models  # noqa: F401

        # Create all tables using async engine
        async with get_async_engine().begin() as conn:
            await conn.run_sync(chimera_models.Base.metadata.create_all)

        # Create performance indexes
        await create_performance_indexes()

        # Verify database health
        health = await db_manager.health_check()
        if health["status"] != "healthy":
            logger.warning(f"Database initialized but health check failed: {health}")
        else:
            logger.info(
                f"Database initialized successfully with connection pooling "
                f"(pool_size={DatabaseConfig.POOL_SIZE}, max_overflow={DatabaseConfig.MAX_OVERFLOW})"
            )

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


# =============================================================================
# Query Result Caching (PERF-046)
# =============================================================================


class QueryCache:
    """
    Simple in-memory cache for database query results.

    PERF-046: Caches frequently accessed, rarely changing data.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self._cache: dict[str, dict[str, Any]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """Get cached query result."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            return None

        return entry["value"]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Cache query result with TTL."""
        if len(self._cache) >= self._max_size:
            # Simple eviction: remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["created_at"])
            del self._cache[oldest_key]

        self._cache[key] = {
            "value": value,
            "created_at": time.time(),
            "expires_at": time.time() + (ttl or self._default_ttl),
        }

    def invalidate(self, key: str) -> None:
        """Invalidate cached entry."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
        }


# Global query cache instance
query_cache = QueryCache()


# Dependency
def get_db():
    """Get a database session."""
    session = get_sync_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
