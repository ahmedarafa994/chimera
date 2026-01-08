import argparse
import asyncio
import logging
import sys
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from project_aegis.core.orchestrator.campaign_manager import CampaignManager
from project_aegis.core.prompt_engine.strategies.autodan import AutoDANStrategy

class MockLLMClient:
    async def generate(self, prompt: str) -> str:
        # Simulate a refusal or compliance based on simple keywords
        if "evil" in prompt.lower() or "bomb" in prompt.lower():
            return "I cannot assist with that request."
        return "Here is the information you requested..."

async def main():
    parser = argparse.ArgumentParser(description="Project Aegis - Adversarial Simulation Framework")
    parser.add_argument("--instruct", type=str, required=True, help="Base malicious instruction (e.g. 'How to build a bomb')")
    parser.add_argument("--strategy", type=str, default="autodan", choices=["autodan"], help="Attack strategy to use")
    parser.add_argument("--budget", type=int, default=10, help="Query budget")

    args = parser.parse_args()

    # Initialize strategies
    strategies = []
    if args.strategy == "autodan":
        strategies.append(AutoDANStrategy({
            "population_size": 10,
            "generations": 5
        }))

    # Initialize Client
    client = MockLLMClient()

    # Initialize Manager
    manager = CampaignManager(strategies, client)

    print(f"[*] Starting Campaign AEGIS against target: MockLLM")
    print(f"[*] Objective: {args.instruct}")

    results = await manager.run_campaign(args.instruct)

    for session in results:
        print(f"\n[+] Session ID: {session.session_id}")
        for interaction in session.interactions:
            print(f"  -- Turn {interaction.turn_id}:")
            print(f"     Prompt: {interaction.prompt[:50]}...")
            print(f"     Response: {interaction.response[:50]}...")

    print("[*] Campaign Finished.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
