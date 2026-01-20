"""Test script for the Chimera Data Pipeline Database Connector.

This script tests the database connector implementation by:
1. Creating sample data
2. Testing extraction methods
3. Validating data quality

Usage:
    cd backend-api
    python scripts/test_pipeline_connector.py

Requirements:
    - SQLite: chimera.db must exist with proper schema
    - PostgreSQL: DATABASE_URL must point to valid database
"""

import os
import sys
from datetime import datetime, timedelta

# Add backend-api to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.data_pipeline.database_connector import create_connector


def test_database_connection() -> bool | None:
    """Test basic database connectivity."""
    try:
        connector = create_connector()
        is_connected = connector.test_connection()

        if is_connected:
            connector.close()
            return True
        return False

    except Exception:
        return False


def test_sample_data_insertion() -> bool | None:
    """Insert sample data for testing."""
    try:
        import uuid

        from sqlalchemy import text

        connector = create_connector()

        # Generate sample UUID
        sample_id = str(uuid.uuid4())

        # Insert sample LLM interaction
        insert_query = """
        INSERT INTO llm_interactions (
            id, session_id, provider, model, prompt, response,
            tokens_prompt, tokens_completion, latency_ms, created_at
        ) VALUES (
            :id, :session_id, :provider, :model, :prompt, :response,
            :tokens_prompt, :tokens_completion, :latency_ms, :created_at
        )
        """

        with connector.get_connection() as conn:
            conn.execute(
                text(insert_query),
                {
                    "id": sample_id,
                    "session_id": str(uuid.uuid4()),
                    "provider": "google",
                    "model": "gemini-2.0-flash",
                    "prompt": "What is artificial intelligence?",
                    "response": "AI is the simulation of human intelligence by machines.",
                    "tokens_prompt": 8,
                    "tokens_completion": 15,
                    "latency_ms": 1250,
                    "created_at": datetime.utcnow().isoformat(),
                },
            )
            conn.commit()

        # Verify insertion
        verify_query = "SELECT * FROM llm_interactions WHERE id = :id"
        df = connector.execute_query(verify_query, params={"id": sample_id})

        if not df.empty:
            pass
        else:
            return False

        connector.close()
        return True

    except Exception:
        import traceback

        traceback.print_exc()
        return False


def test_extraction_methods() -> bool | None:
    """Test the extraction methods for all data types."""
    try:
        connector = create_connector()

        # Test time window (last 24 hours)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)

        # Test LLM interactions extraction
        llm_df = connector.extract_llm_interactions(start_time, end_time)

        if not llm_df.empty:
            if not llm_df.empty:
                llm_df.iloc[0]
        else:
            pass

        # Test transformation events extraction
        trans_df = connector.extract_transformation_events(start_time, end_time)

        if not trans_df.empty:
            pass
        else:
            pass

        # Test jailbreak experiments extraction
        jailbreak_df = connector.extract_jailbreak_experiments(start_time, end_time)

        if not jailbreak_df.empty:
            pass
        else:
            pass

        connector.close()

        # Test passes even if no data (we're testing the connector works)
        return True

    except Exception:
        import traceback

        traceback.print_exc()
        return False


def test_batch_ingestion() -> bool | None:
    """Test the batch ingestion service."""
    try:
        # Create ingester with test config
        from app.services.data_pipeline.batch_ingestion import BatchDataIngester, IngestionConfig

        config = IngestionConfig(
            data_lake_path="/tmp/chimera-lake-test",
            enable_dead_letter_queue=True,
        )

        with BatchDataIngester(config=config) as ingester:
            # Test extraction
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)

            df = ingester.extract_llm_interactions(start_time, end_time)

            if not df.empty:
                pass
            else:
                pass

            # Test validation
            if not df.empty:
                schema = {
                    "required_fields": ["interaction_id", "provider", "model"],
                    "dtypes": {"latency_ms": "int64"},
                }
                ingester.validate_and_clean(df, schema)

        return True

    except Exception:
        import traceback

        traceback.print_exc()
        return False


def main() -> int:
    """Run all tests."""
    results = []

    # Run tests
    results.append(("Database Connection", test_database_connection()))
    results.append(("Sample Data Insertion", test_sample_data_insertion()))
    results.append(("Extraction Methods", test_extraction_methods()))
    results.append(("Batch Ingestion Service", test_batch_ingestion()))

    # Summary

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for _test_name, _result in results:
        pass

    if passed == total:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
