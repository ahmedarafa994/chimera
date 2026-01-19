"""
Test script for the Chimera Data Pipeline Database Connector

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

from app.core.config import settings
from app.services.data_pipeline.database_connector import create_connector


def test_database_connection():
    """Test basic database connectivity."""
    print("\n" + "=" * 60)
    print("TEST 1: Database Connection")
    print("=" * 60)

    try:
        connector = create_connector()
        is_connected = connector.test_connection()

        if is_connected:
            print("✅ Database connection successful")
            connector.close()
            return True
        else:
            print("❌ Database connection failed")
            return False

    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False


def test_sample_data_insertion():
    """Insert sample data for testing."""
    print("\n" + "=" * 60)
    print("TEST 2: Sample Data Insertion")
    print("=" * 60)

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

        print(f"✅ Inserted sample record with ID: {sample_id}")

        # Verify insertion
        verify_query = "SELECT * FROM llm_interactions WHERE id = :id"
        df = connector.execute_query(verify_query, params={"id": sample_id})

        if not df.empty:
            print("✅ Verified sample record retrieval")
            print(f"   Provider: {df['provider'].iloc[0]}")
            print(f"   Model: {df['model'].iloc[0]}")
            print(
                f"   Tokens: {df['tokens_total'].iloc[0] if 'tokens_total' in df.columns else 'N/A'}"
            )
        else:
            print("❌ Failed to retrieve sample record")
            return False

        connector.close()
        return True

    except Exception as e:
        print(f"❌ Sample data insertion failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_extraction_methods():
    """Test the extraction methods for all data types."""
    print("\n" + "=" * 60)
    print("TEST 3: Data Extraction Methods")
    print("=" * 60)

    try:
        connector = create_connector()

        # Test time window (last 24 hours)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)

        # Test LLM interactions extraction
        print("\n--- Testing LLM Interactions Extraction ---")
        llm_df = connector.extract_llm_interactions(start_time, end_time)

        if not llm_df.empty:
            print(f"✅ Extracted {len(llm_df)} LLM interactions")
            print(f"   Columns: {list(llm_df.columns)}")
            if not llm_df.empty:
                sample = llm_df.iloc[0]
                print(f"   Sample Provider: {sample.get('provider', 'N/A')}")
                print(f"   Sample Model: {sample.get('model', 'N/A')}")
        else:
            print("⚠️  No LLM interactions found (this is OK if database is empty)")

        # Test transformation events extraction
        print("\n--- Testing Transformation Events Extraction ---")
        trans_df = connector.extract_transformation_events(start_time, end_time)

        if not trans_df.empty:
            print(f"✅ Extracted {len(trans_df)} transformation events")
        else:
            print("⚠️  No transformation events found (this is OK if database is empty)")

        # Test jailbreak experiments extraction
        print("\n--- Testing Jailbreak Experiments Extraction ---")
        jailbreak_df = connector.extract_jailbreak_experiments(start_time, end_time)

        if not jailbreak_df.empty:
            print(f"✅ Extracted {len(jailbreak_df)} jailbreak experiments")
        else:
            print("⚠️  No jailbreak experiments found (this is OK if database is empty)")

        connector.close()

        # Test passes even if no data (we're testing the connector works)
        return True

    except Exception as e:
        print(f"❌ Extraction methods test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_batch_ingestion():
    """Test the batch ingestion service."""
    print("\n" + "=" * 60)
    print("TEST 4: Batch Ingestion Service")
    print("=" * 60)

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

            print("\n--- Testing BatchDataIngester.extract_llm_interactions() ---")
            df = ingester.extract_llm_interactions(start_time, end_time)

            if not df.empty:
                print(f"✅ Batch ingester extracted {len(df)} records")
            else:
                print("⚠️  No records extracted (OK if database is empty)")

            # Test validation
            if not df.empty:
                schema = {
                    "required_fields": ["interaction_id", "provider", "model"],
                    "dtypes": {"latency_ms": "int64"},
                }
                df_clean = ingester.validate_and_clean(df, schema)
                print(f"✅ Validation completed, {len(df_clean)} records passed")

        print("✅ Batch ingestion service test completed")
        return True

    except Exception as e:
        print(f"❌ Batch ingestion test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Chimera Data Pipeline - Database Connector Test Suite")
    print("=" * 60)
    print(f"Database URL: {settings.DATABASE_URL}")

    results = []

    # Run tests
    results.append(("Database Connection", test_database_connection()))
    results.append(("Sample Data Insertion", test_sample_data_insertion()))
    results.append(("Extraction Methods", test_extraction_methods()))
    results.append(("Batch Ingestion Service", test_batch_ingestion()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
