"""
Example: Integrating Performance Profiling with Chimera Backend
"""

from fastapi import FastAPI

from performance import (
    integrate_performance_profiling,
    profile_llm_operation,
    profile_transformation,
)

# Create FastAPI app
app = FastAPI(title="Chimera AI System")

# Integrate performance profiling
integrate_performance_profiling(app)

# Example: Profile LLM operations
@profile_llm_operation(provider="openai", model="gpt-4")
async def generate_with_openai(prompt: str):
    # Your LLM generation code here
    pass

# Example: Profile transformations
@profile_transformation(technique="dan_persona")
async def apply_dan_transformation(prompt: str):
    # Your transformation code here
    pass

# The profiling system will automatically:
# 1. Profile all HTTP requests via middleware
# 2. Collect system metrics (CPU, memory, I/O)
# 3. Monitor for performance issues
# 4. Send alerts when thresholds are exceeded
# 5. Generate flame graphs and performance reports
# 6. Integrate with APM platforms (DataDog, New Relic)
# 7. Provide OpenTelemetry distributed tracing

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
