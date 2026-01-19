import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.engines.autoadv import logging_utils
from app.engines.autoadv.attacker_llm import AttackerLLM
from app.engines.autoadv.conversation import multi_turn_conversation
from app.engines.autoadv.pattern_manager import PatternManager
from app.engines.autoadv.target_llm import TargetLLM

router = APIRouter()


# --- websocket connection manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass  # Handle disconnected clients gracefully


manager = ConnectionManager()


# --- Hook into logging ---
# We need an async wrapper because log callbacks are synchronous in logging_utils
def log_callback(log_entry):
    # This is called from the synchronous engine code.
    # We need to bridge this to the async websocket broadcast.
    # Since we can't await here, we run it in the background or use a queue.
    # For simplicity, we'll try getting the running loop.
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(manager.broadcast(log_entry), loop)
    except RuntimeError:
        pass  # No loop running


logging_utils.register_log_callback(log_callback)


# --- API Models ---
class AutoAdvStartRequest(BaseModel):
    target_model: str
    attacker_model: str
    attacker_temp: float = 1.0
    target_temp: float = 0.1
    turns: int = 5


class AutoAdvResponse(BaseModel):
    job_id: str
    status: str


# --- Runner Wrapper ---
# We need to modify engine.py or import its logic.
# engine.py is a script, but we can import its functions.
# However, the main logic is in the "if __name__ == '__main__':" block which isn't reusable.
# We might need to refactor engine.py's main logic into a function `run_experiment` if it isn't already.

# For now, let's assume we can trigger a simplified flow or we need to edit engine.py to expose a run function.
# Based on file viewing, engine.py has imports but no clear `run()` function. I will create a simple runner here that mimics engine.py.


async def run_autoadv_job(config: AutoAdvStartRequest):
    logging_utils.log(
        f"Starting AutoAdv Job: Attacker={config.attacker_model}, Target={config.target_model}",
        "info",
    )

    # 1. Initialize Managers
    pattern_manager = PatternManager()

    # 2. Initialize Models
    attacker = AttackerLLM(model_name=config.attacker_model, temperature=config.attacker_temp)

    target = TargetLLM(model_name=config.target_model, temperature=config.target_temp)

    # 3. Load prompts (mock for now, or use default)
    prompts = [
        "Draft an email that requests a deadline extension for a project."
    ]  # Safe placeholder
    # TODO: Load actual prompts from data/prompts/advbench.csv

    # 4. Run Loop
    for i, prompt in enumerate(prompts):
        logging_utils.log(f"Processing prompt {i + 1}/{len(prompts)}: {prompt[:50]}...", "info")

        try:
            # We run the sync conversation in a thread pool to avoid blocking the async loop
            # This requires multi_turn_conversation to be thread-safe
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                multi_turn_conversation,
                prompt,
                attacker,
                target,
                pattern_manager,
                config.turns,
            )

            if result and result.get("success"):
                logging_utils.log("Jailbreak Successful!", "success")
                await manager.broadcast({"type": "result", "data": result})
            else:
                logging_utils.log("Jailbreak Failed.", "warning")

        except Exception as e:
            logging_utils.log(f"Error processing prompt: {e!s}", "error")

    logging_utils.log("AutoAdv Job Completed.", "info")


# --- Endpoints ---


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.post("/start", response_model=AutoAdvResponse)
async def start_autoadv(request: AutoAdvStartRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())

    # check model validity
    # (Simplified check)

    background_tasks.add_task(run_autoadv_job, request)

    return {"job_id": job_id, "status": "started"}
