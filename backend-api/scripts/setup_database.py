#!/usr/bin/env python3
"""
Database Setup Script for Chimera Platform

This script initializes the database schema, runs migrations,
and optionally seeds initial data for development.

Usage:
    python scripts/setup_database.py [--seed] [--reset]

Options:
    --seed      Seed the database with sample data
    --reset     Drop all tables and recreate (WARNING: destructive)
    --check     Check database connection and current migration status
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError

from app.core.config import settings
from app.db.models import Base as DBBase
from app.infrastructure.database.models import Base as InfraBase


def check_database_connection():
    """Check if database is accessible."""
    print("🔍 Checking database connection...")
    print(f"   Database URL: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else settings.DATABASE_URL}")

    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")
        return engine
    except OperationalError as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)


def check_migration_status():
    """Check current Alembic migration status."""
    print("\n📋 Checking migration status...")
    import subprocess

    try:
        result = subprocess.run(
            ["alembic", "current"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to check migration status: {e}")
        return False


def get_existing_tables(engine):
    """Get list of existing tables in database."""
    inspector = inspect(engine)
    return inspector.get_table_names()


def run_migrations():
    """Run Alembic migrations to latest version."""
    print("\n🔄 Running database migrations...")
    import subprocess

    try:
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print("✅ Migrations completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Migration failed: {e}")
        print(e.stderr)
        return False


def reset_database(engine):
    """Drop all tables and recreate (WARNING: destructive)."""
    print("\n⚠️  RESETTING DATABASE (dropping all tables)...")

    # Drop all tables
    DBBase.metadata.drop_all(bind=engine)
    InfraBase.metadata.drop_all(bind=engine)

    # Drop alembic version table
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        conn.commit()

    print("✅ All tables dropped")


def seed_database(engine):
    """Seed database with initial data for development."""
    print("\n🌱 Seeding database with sample data...")

    from passlib.context import CryptContext
    from sqlalchemy.orm import Session

    from app.db.models import User, UserRole

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    with Session(engine) as session:
        # Check if admin user exists
        existing_admin = session.query(User).filter_by(email="admin@chimera.local").first()

        if existing_admin:
            print("   ℹ️  Admin user already exists, skipping seed")
            return

        # Create admin user
        admin_user = User(
            email="admin@chimera.local",
            username="admin",
            hashed_password=pwd_context.hash("admin123"),
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True
        )
        session.add(admin_user)

        # Create researcher user
        researcher_user = User(
            email="researcher@chimera.local",
            username="researcher",
            hashed_password=pwd_context.hash("researcher123"),
            role=UserRole.RESEARCHER,
            is_active=True,
            is_verified=True
        )
        session.add(researcher_user)

        # Create viewer user
        viewer_user = User(
            email="viewer@chimera.local",
            username="viewer",
            hashed_password=pwd_context.hash("viewer123"),
            role=UserRole.VIEWER,
            is_active=True,
            is_verified=True
        )
        session.add(viewer_user)

        session.commit()

        print("✅ Sample users created:")
        print("   👤 admin@chimera.local / admin123 (Admin)")
        print("   👤 researcher@chimera.local / researcher123 (Researcher)")
        print("   👤 viewer@chimera.local / viewer123 (Viewer)")


def print_schema_summary(engine):
    """Print summary of database schema."""
    print("\n📊 Database Schema Summary:")

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    print(f"   Total tables: {len(tables)}")
    print("\n   Tables:")
    for table in sorted(tables):
        columns = inspector.get_columns(table)
        indexes = inspector.get_indexes(table)
        print(f"      • {table} ({len(columns)} columns, {len(indexes)} indexes)")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Chimera database")
    parser.add_argument("--seed", action="store_true", help="Seed database with sample data")
    parser.add_argument("--reset", action="store_true", help="Reset database (drop all tables)")
    parser.add_argument("--check", action="store_true", help="Check database status only")

    args = parser.parse_args()

    print("=" * 60)
    print("🗄️  Chimera Database Setup")
    print("=" * 60)

    # Check database connection
    engine = check_database_connection()

    # Check migration status
    check_migration_status()

    if args.check:
        print_schema_summary(engine)
        return

    # Reset database if requested
    if args.reset:
        confirm = input("\n⚠️  Are you sure you want to reset the database? (yes/no): ")
        if confirm.lower() != "yes":
            print("❌ Reset cancelled")
            return
        reset_database(engine)

    # Run migrations
    if not run_migrations():
        print("\n❌ Setup failed due to migration errors")
        sys.exit(1)

    # Seed database if requested
    if args.seed:
        seed_database(engine)

    # Print schema summary
    print_schema_summary(engine)

    print("\n" + "=" * 60)
    print("✅ Database setup completed successfully!")
    print("=" * 60)

    if args.seed:
        print("\n💡 You can now login with:")
        print("   Email: admin@chimera.local")
        print("   Password: admin123")


if __name__ == "__main__":
    main()
