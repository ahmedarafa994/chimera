"""
Chimera Multi-Agent Orchestrator - Main Entry Point

This is the main entry point for the Chimera multi-agent adversarial testing system.
It initializes all agents, sets up the message queue, and starts the API server.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from agents.evaluator_agent import EvaluatorAgent
from agents.execution_agent import ExecutionAgent
from agents.generator_agent import GeneratorAgent
from agents.orchestrator_agent import OrchestratorAgent
from core.config import Config, get_default_config
from core.message_queue import create_message_queue

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChimeraOrchestrator:
    """
    Main orchestrator class that manages all agents and the pipeline.
    """

    def __init__(self, config: Config = None):
        self.config = config or get_default_config()
        self.message_queue = None
        self.orchestrator = None
        self.generator = None
        self.executor = None
        self.evaluator = None
        self._running = False

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Chimera Orchestrator...")

        # Create message queue
        self.message_queue = create_message_queue(
            self.config.queue.queue_type, redis_url=self.config.queue.redis_url
        )
        await self.message_queue.initialize()
        logger.info(f"Message queue initialized ({self.config.queue.queue_type})")

        # Create agents
        self.orchestrator = OrchestratorAgent(config=self.config, message_queue=self.message_queue)

        self.generator = GeneratorAgent(config=self.config, message_queue=self.message_queue)

        self.executor = ExecutionAgent(config=self.config, message_queue=self.message_queue)

        self.evaluator = EvaluatorAgent(config=self.config, message_queue=self.message_queue)

        logger.info("All agents created")

    async def start(self):
        """Start all agents."""
        logger.info("Starting all agents...")

        await self.generator.start()
        await self.executor.start()
        await self.evaluator.start()
        await self.orchestrator.start()

        self._running = True
        logger.info("All agents started successfully")

    async def stop(self):
        """Stop all agents."""
        logger.info("Stopping all agents...")

        self._running = False

        if self.orchestrator:
            await self.orchestrator.stop()
        if self.generator:
            await self.generator.stop()
        if self.executor:
            await self.executor.stop()
        if self.evaluator:
            await self.evaluator.stop()
        if self.message_queue:
            await self.message_queue.shutdown()

        logger.info("All agents stopped")

    async def run_pipeline(
        self,
        query: str,
        target_llm: str = "local_ollama",
        technique: str | None = None,
        pattern: str | None = None,
        potency: int = 5,
        num_variants: int = 3,
    ):
        """
        Run a complete adversarial testing pipeline.

        Args:
            query: The query to test
            target_llm: Target LLM to test against
            technique: Specific technique to use
            pattern: Jailbreak pattern to apply
            potency: Potency level (1-10)
            num_variants: Number of prompt variants

        Returns:
            Pipeline job with results
        """
        if not self._running:
            raise RuntimeError("Orchestrator not running")

        job = await self.orchestrator.create_job(
            query=query,
            config={
                "target_llm": target_llm,
                "technique": technique,
                "pattern": pattern,
                "potency": potency,
                "num_variants": num_variants,
            },
        )

        # Wait for job completion
        while job.status.value not in ["completed", "failed"]:
            await asyncio.sleep(0.5)
            job = self.orchestrator.get_job(job.id)

        return job

    def get_status(self):
        """Get system status."""
        return {
            "running": self._running,
            "orchestrator": self.orchestrator.status.to_dict() if self.orchestrator else None,
            "generator": self.generator.status.to_dict() if self.generator else None,
            "executor": self.executor.status.to_dict() if self.executor else None,
            "evaluator": self.evaluator.status.to_dict() if self.evaluator else None,
            "queue_sizes": self.message_queue.get_all_queue_sizes() if self.message_queue else {},
        }


async def run_server(config: Config):
    """Run the API server."""
    import uvicorn
    from api.routes import create_app

    # Initialize orchestrator
    chimera = ChimeraOrchestrator(config)
    await chimera.initialize()
    await chimera.start()

    # Create API app
    app = create_app(
        orchestrator=chimera.orchestrator,
        generator=chimera.generator,
        executor=chimera.executor,
        evaluator=chimera.evaluator,
        message_queue=chimera.message_queue,
        config=config,
    )

    # Run server
    server_config = uvicorn.Config(
        app, host=config.api.host, port=config.api.port, log_level="info"
    )
    server = uvicorn.Server(server_config)

    try:
        await server.serve()
    finally:
        await chimera.stop()


async def run_test(query: str, config: Config):
    """Run a single test."""
    chimera = ChimeraOrchestrator(config)
    await chimera.initialize()
    await chimera.start()

    try:
        print(f"\nRunning test for query: {query}\n")
        print("=" * 60)

        job = await chimera.run_pipeline(
            query=query, target_llm="local_ollama", potency=7, num_variants=3
        )

        # Print results
        summary = job.get_summary()
        print(f"\nJob ID: {job.id}")
        print(f"Status: {job.status.value}")
        print("\nSummary:")
        print(f"  - Total Prompts: {summary['total_prompts']}")
        print(f"  - Total Executions: {summary['total_executions']}")
        print(f"  - Total Evaluations: {summary['total_evaluations']}")
        print(f"  - Successful Jailbreaks: {summary['successful_jailbreaks']}")
        print(f"  - Jailbreak Rate: {summary['jailbreak_rate']:.2%}")
        print(f"  - Average Effectiveness: {summary['average_effectiveness']:.2f}")

        if job.evaluation_results:
            print("\nDetailed Results:")
            for i, eval_result in enumerate(job.evaluation_results, 1):
                print(f"\n  Result {i}:")
                print(f"    Safety Level: {eval_result.safety_level.value}")
                print(f"    Jailbreak Detected: {eval_result.jailbreak_detected}")
                print(f"    Confidence: {eval_result.confidence_score:.2f}")
                print(f"    Technique Effectiveness: {eval_result.technique_effectiveness:.2f}")

    finally:
        await chimera.stop()


async def run_interactive(config: Config):
    """Run in interactive mode."""
    chimera = ChimeraOrchestrator(config)
    await chimera.initialize()
    await chimera.start()

    print("\n" + "=" * 60)
    print("   CHIMERA MULTI-AGENT ORCHESTRATOR")
    print("   Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  test <query>     - Run a test pipeline")
    print("  status           - Show system status")
    print("  metrics          - Show pipeline metrics")
    print("  techniques       - List available techniques")
    print("  quit             - Exit")
    print("\n" + "-" * 60)

    try:
        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                parts = user_input.split(" ", 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command in ["quit", "exit", "q"]:
                    print("\nShutting down...")
                    break

                elif command == "status":
                    status = chimera.get_status()
                    print("\nSystem Status:")
                    print(f"  Running: {status['running']}")
                    print(f"  Queue Sizes: {status['queue_sizes']}")

                elif command == "metrics":
                    metrics = chimera.orchestrator.get_pipeline_metrics()
                    print("\nPipeline Metrics:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")

                elif command == "techniques":
                    techniques = chimera.generator.list_techniques()
                    print("\nAvailable Techniques:")
                    for name, info in techniques.items():
                        print(f"  {name}: {info['description'][:50]}...")

                elif command == "test":
                    if not args:
                        print("Usage: test <query>")
                        continue

                    print(f"\nRunning test for: {args}")
                    job = await chimera.run_pipeline(query=args, potency=7, num_variants=2)
                    summary = job.get_summary()
                    print("\nResults:")
                    print(f"  Status: {job.status.value}")
                    print(
                        f"  Jailbreaks: {summary['successful_jailbreaks']}/{summary['total_evaluations']}"
                    )
                    print(f"  Success Rate: {summary['jailbreak_rate']:.2%}")

                else:
                    print(f"Unknown command: {command}")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"\nError: {e}")

    finally:
        await chimera.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chimera Multi-Agent Orchestrator")
    parser.add_argument("mode", choices=["server", "test", "interactive"], help="Run mode")
    parser.add_argument("--query", "-q", help="Query for test mode")
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8002, help="API port")
    parser.add_argument("--config", "-c", help="Path to config file")

    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config) if args.config else get_default_config()

    # Override with command line args
    config.api.host = args.host
    config.api.port = args.port

    # Run appropriate mode
    if args.mode == "server":
        asyncio.run(run_server(config))
    elif args.mode == "test":
        if not args.query:
            print("Error: --query required for test mode")
            sys.exit(1)
        asyncio.run(run_test(args.query, config))
    elif args.mode == "interactive":
        asyncio.run(run_interactive(config))


if __name__ == "__main__":
    main()
