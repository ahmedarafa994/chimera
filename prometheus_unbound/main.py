from fastapi import FastAPI

from app.api.routes import router
from app.models.prompt_log import init_db

app = FastAPI(
    title="Prometheus Unbound",
    description="Adversarial prompt generation engine for red-teaming LLM alignment protocols",
    version="1.0.0",
)

# Initialize database
init_db()

# Include API routes
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
