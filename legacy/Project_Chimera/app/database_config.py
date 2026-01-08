"""
Optimized database configuration with connection pooling and performance enhancements.
"""

import logging
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

# Configure logging
logger = logging.getLogger(__name__)


class OptimizedDatabase:
    """Optimized database configuration with connection pooling."""

    def __init__(self, app=None):
        self.app = app
        self.engine = None
        self.SessionLocal = None
        if app:
            self.init_app(app)

    def init_app(self, app):
        """Initialize database with optimized configuration."""
        # Database configuration
        database_url = os.getenv("DATABASE_URL", "sqlite:///chimera_logs.db")

        # Enhanced engine configuration with connection pooling
        if database_url.startswith("sqlite"):
            # SQLite-specific optimizations
            engine_kwargs = {
                "poolclass": QueuePool,
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                "echo": app.config.get("SQL_DEBUG", False),
                "connect_args": {"check_same_thread": False, "timeout": 20},
            }
        else:
            # PostgreSQL/MySQL optimizations
            engine_kwargs = {
                "poolclass": QueuePool,
                "pool_size": 20,
                "max_overflow": 30,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                "echo": app.config.get("SQL_DEBUG", False),
                "pool_pre_ping": True,
            }

        self.engine = create_engine(database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Configure Flask-SQLAlchemy with optimized settings
        app.config["SQLALCHEMY_DATABASE_URI"] = database_url
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = engine_kwargs
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
        app.config["SQLALCHEMY_RECORD_QUERIES"] = app.config.get("SQL_DEBUG", False)
        app.config["SQLALCHEMY_POOL_SIZE"] = 20
        app.config["SQLALCHEMY_MAX_OVERFLOW"] = 30

        # Initialize Flask-SQLAlchemy
        from app.extensions import db

        db.init_app(app)

        # Create tables with optimized indexes
        with app.app_context():
            self.create_optimized_indexes()

    def create_optimized_indexes(self):
        """Create performance-optimized indexes."""
        try:
            # Import models here to avoid circular imports

            # Create composite indexes for common query patterns
            indexes = [
                # RequestLog indexes
                "CREATE INDEX IF NOT EXISTS idx_request_logs_timestamp_endpoint ON request_logs (timestamp DESC, endpoint)",
                "CREATE INDEX IF NOT EXISTS idx_request_logs_request_id_timestamp ON request_logs (request_id, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_request_logs_status_timestamp ON request_logs (status_code, timestamp DESC)",
                # LLMUsage indexes
                "CREATE INDEX IF NOT EXISTS idx_llm_usage_timestamp_provider ON llm_usage (timestamp DESC, provider)",
                "CREATE INDEX IF NOT EXISTS idx_llm_usage_provider_timestamp ON llm_usage (provider, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_llm_usage_cached_tokens ON llm_usage (cached, tokens_used)",
                # TechniqueUsage indexes
                "CREATE INDEX IF NOT EXISTS idx_technique_usage_suite_timestamp ON technique_usage (technique_suite, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_technique_usage_potency_timestamp ON technique_usage (potency_level, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_technique_usage_success_timestamp ON technique_usage (success, timestamp DESC)",
                # ErrorLog indexes
                "CREATE INDEX IF NOT EXISTS idx_error_logs_timestamp_type ON error_logs (timestamp DESC, error_type)",
                "CREATE INDEX IF NOT EXISTS idx_error_logs_provider_timestamp ON error_logs (provider, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_error_logs_request_timestamp ON error_logs (request_id, timestamp DESC)",
            ]

            for index_sql in indexes:
                try:
                    self.engine.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

            logger.info("Database indexes optimized successfully")

        except Exception as e:
            logger.error(f"Error creating database indexes: {e}")

    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()

    def health_check(self):
        """Database health check."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Singleton instance
optimized_db = OptimizedDatabase()
