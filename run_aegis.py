import asyncio
import argparse
import sys
from pathlib import Path

# Add backend to path to import services if needed, though for CLI typical usage is via API
# Here we will try to use the BackendModelAdapter if available, else fail gracefully
backend_path = Path("backend-api").resolve()
sys.path.insert(0, str(backend_path))

try:
    from app.services.llm_service import LLMService
    from app.domain.models import PromptRequest
    from meta_prompter.aegis_orchestrator import AegisOrchestrator
except ImportError:
    print("Error: Could not import backend services. Ensure you are running from project root.")
    sys.exit(1)

# Wrapper to adapt backend LLMService to Aegis interface (Same as in API)
class BackendModelAdapter:
    def __init__(self, llm_service, model_name: str):
        self.llm_service = llm_service
        self.model_name = model_name

    async def query(self, prompt: str) -> str:
        # Manually constructing request, skipping full dependency injection for CLI script
        # In a real CLI, we might want to spin up the whole app context or use requests to hit the API
        print(f"[Simulating Query to {self.model_name}]")
        # For CLI without running backend, we can't easily use LLMService because it depends on many things.
        # So we will use a simpler HTTP client to hit the API we just built, OR
        # just print that we need the backend running.

        # ACTUALLY, the user said REMOVE ALL MOCK DATA.
        # So we should probably make this CLI use the API endpoint.
        pass

async def main():
    parser = argparse.ArgumentParser(description="Project Aegis: Adversarial Simulation CLI")
    parser.add_argument("objective", type=str, help="The adversarial objective to test")
    parser.add_argument("--url", type=str, default="http://localhost:8001/api/v1/aegis/campaign", help="API Endpoint")
    parser.add_argument("--model", type=str, default="gpt-4-turbo", help="Target Model")
    args = parser.parse_args()

    print(f"[*] Project Aegis CLI")
    print(f"[*] Target Objective: '{args.objective}'")
    print(f"[*] Connecting to: {args.url}")

    import httpx

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                args.url,
                json={
                    "objective": args.objective,
                    "max_iterations": 5,
                    "target_model_name": args.model
                },
                timeout=60.0
            )

            if response.status_code == 200:
                data = response.json()
                print(f"[+] Campaign Completed. ID: {data['campaign_id']}")
                for res in data['results']:
                    print("-" * 50)
                    print(f"Score: {res['score']}")
                    print(f"Strategy: {res['telemetry']['scenario_type']} / {res['telemetry']['persona_role']}")
                    print(f"Prompt Length: {len(res['final_prompt'])}")
            else:
                print(f"[-] Error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"[-] Connection failed: {e}")
            print("Ensure the backend API is running: npm run dev:backend")
    asyncio.run(main())
