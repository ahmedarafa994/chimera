import os
import sys

# Setup path
sys.path.append(os.path.join(os.getcwd(), "backend-api"))

try:
    from app.services.unified_transformation_engine import (
        UnifiedTransformationEngine,
    )

    print("SUCCESS: UnifiedTransformationEngine imported successfully.")

    # Try to initialize
    engine = UnifiedTransformationEngine()
    print("SUCCESS: UnifiedTransformationEngine initialized successfully.")

    # Verify methods exist
    print(f"Has Quantum: {hasattr(engine, '_orchestrate_quantum_superposition')}")
    print(f"Has Evolutionary: {hasattr(engine, '_orchestrate_evolutionary_pipeline')}")

except Exception as e:
    print(f"FAILURE: {e}")
