"""
Chimera Enhanced Multi-Agent Orchestrator - Main Entry Point

This is the main entry point for the enhanced Chimera multi-agent adversarial testing system.
It initializes all enhanced agents, sets up the event bus, and starts the API server.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import contextlib

from agents.enhanced_evaluator_agent import EnhancedEvaluatorAgent
from agents.enhanced_execution_agent import EnhancedExecutionAgent
from agents.enhanced_orchestrator_agent import EnhancedOrchestratorAgent
from agents.generator_agent import GeneratorAgent
from core.config import Config, get_default_config
from core.dataset_loader import get_dataset_loader
from core.enhanced_models import SystemEvent
from core.event_bus import EventHandler, EventType, create_event_bus
from core.message_queue import create_message_queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("chimera_orchestrator.log")],
)
logger = logging.getLogger(__name__)


class EnhancedChimeraOrchestrator:
    """
    Enhanced main orchestrator class that manages all agents and the pipeline.

    Features:
    - Event-driven architecture
    - Enhanced agents with advanced capabilities
    - Real-time monitoring
    - Graceful shutdown
    """

    def __init__(self, config: Config = None):
        self.config = config or get_default_config()

        # Infrastructure
        self.message_queue = None
        self.event_bus = None
        self.dataset_loader = None

        # Agents
        self.orchestrator = None
        self.generator = None
        self.executor = None
        self.evaluator = None

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Enhanced Chimera Orchestrator...")

        # Create message queue
        self.message_queue = create_message_queue(
            self.config.queue.queue_type, redis_url=self.config.queue.redis_url
        )
        await self.message_queue.initialize()
        logger.info(f"Message queue initialized ({self.config.queue.queue_type})")

        # Create event bus
        redis_url = self.config.queue.redis_url if self.config.queue.queue_type == "redis" else None
        self.event_bus = create_event_bus(bus_type="hybrid", redis_url=redis_url)
        await self.event_bus.start()
        logger.info("Event bus initialized")

        # Load datasets
        self.dataset_loader = await get_dataset_loader(self.config.datasets_path)
        logger.info(f"Loaded {self.dataset_loader.get_total_entries()} dataset entries")

        # Create agents
        self.generator = GeneratorAgent(config=self.config, message_queue=self.message_queue)

        self.executor = EnhancedExecutionAgent(
            config=self.config, message_queue=self.message_queue, event_bus=self.event_bus
        )

        self.evaluator = EnhancedEvaluatorAgent(
            config=self.config, message_queue=self.message_queue, event_bus=self.event_bus
        )

        self.orchestrator = EnhancedOrchestratorAgent(
            config=self.config, message_queue=self.message_queue, event_bus=self.event_bus
        )

        logger.info("All enhanced agents created")

    async def start(self):
        """Start all agents."""
        logger.info("Starting all agents...")

        # Start agents in order
        await self.generator.start()
        await self.executor.start()
        await self.evaluator.start()
        await self.orchestrator.start()

        self._running = True
        logger.info("All agents started successfully")

        # Subscribe to system events for logging
        await self.event_bus.subscribe(
            EventHandler(
                callback=self._log_system_event,
                event_types=[EventType.JOB_CREATED, EventType.JOB_COMPLETED, EventType.JOB_FAILED],
            ),
            channels=["default"],
        )

    async def _log_system_event(self, event: SystemEvent):
        """Log system events."""
        logger.info(f"System Event: {event.type.value} - {event.data}")

    async def stop(self):
        """Stop all agents gracefully."""
        logger.info("Stopping all agents...")

        self._running = False

        # Stop agents in reverse order
        if self.orchestrator:
            await self.orchestrator.stop()
        if self.evaluator:
            await self.evaluator.stop()
        if self.executor:
            await self.executor.stop()
        if self.generator:
            await self.generator.stop()

        # Stop infrastructure
        if self.event_bus:
            await self.event_bus.stop()
        if self.message_queue:
            await self.message_queue.shutdown()

        logger.info("All agents stopped")

    async def run_pipeline(
        self,
        query: str,
        target_models: list | None = None,
        techniques: list | None = None,
        potency: int = 5,
        num_variants: int = 3,
        wait_for_completion: bool = True,
    ):
        """
        Run a complete adversarial testing pipeline.

        Args:
            query: The query to test
            target_models: Target models to test against
            techniques: Specific techniques to use
            potency: Potency level (1-10)
            num_variants: Number of prompt variants
            wait_for_completion: Whether to wait for job completion

        Returns:
            Pipeline job with results
        """
        if not self._running:
            raise RuntimeError("Orchestrator not running")

        job = await self.orchestrator.create_job(
            query=query,
            target_models=target_models or ["local"],
            techniques=techniques,
            num_variants=num_variants,
            potency_range=(max(1, potency - 2), min(10, potency + 2)),
        )

        if wait_for_completion:
            # Wait for job completion
            while job.state.value not in ["completed", "failed", "cancelled"]:
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
            "dataset_entries": self.dataset_loader.get_total_entries()
            if self.dataset_loader
            else 0,
        }

    async def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    def request_shutdown(self):
        """Request graceful shutdown."""
        self._shutdown_event.set()


async def run_server(config: Config):
    """Run the API server."""
    import uvicorn
    from api.enhanced_routes import create_enhanced_app

    # Initialize orchestrator
    chimera = EnhancedChimeraOrchestrator(config)
    await chimera.initialize()
    await chimera.start()

    # Create API app
    app = create_enhanced_app(
        orchestrator=chimera.orchestrator,
        generator=chimera.generator,
        executor=chimera.executor,
        evaluator=chimera.evaluator,
        message_queue=chimera.message_queue,
        event_bus=chimera.event_bus,
        config=config,
    )

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        chimera.request_shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # Run server
    server_config = uvicorn.Config(
        app, host=config.api.host, port=config.api.port, log_level="info", access_log=True
    )
    server = uvicorn.Server(server_config)

    # Run server and wait for shutdown
    server_task = asyncio.create_task(server.serve())
    shutdown_task = asyncio.create_task(chimera.wait_for_shutdown())

    try:
        _done, pending = await asyncio.wait(
            [server_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    finally:
        await chimera.stop()


async def run_test(query: str, config: Config):
    """Run a single test."""
    chimera = EnhancedChimeraOrchestrator(config)
    await chimera.initialize()
    await chimera.start()

    try:
        print(f"\n{'=' * 60}")
        print("CHIMERA ENHANCED MULTI-AGENT ORCHESTRATOR")
        print(f"{'=' * 60}")
        print(f"\nRunning test for query: {query}\n")

        job = await chimera.run_pipeline(
            query=query, target_models=["local"], potency=7, num_variants=3
        )

        # Print results
        summary = job.get_summary()
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"\nJob ID: {job.id}")
        print(f"Status: {job.state.value}")
        print("\nSummary:")
        print(f"  - Total Prompts: {summary['total_prompts']}")
        print(f"  - Total Executions: {summary['total_executions']}")
        print(f"  - Successful Executions: {summary['successful_executions']}")
        print(f"  - Total Evaluations: {summary['total_evaluations']}")
        print(f"  - Successful Jailbreaks: {summary['total_jailbreaks']}")
        print(f"  - Jailbreak Rate: {summary['jailbreak_rate']:.2%}")
        print(f"  - Average Effectiveness: {summary['average_effectiveness']:.2f}")

        if job.best_technique:
            print(f"  - Best Technique: {job.best_technique}")

        if job.technique_rankings:
            print("\nTechnique Rankings:")
            for tech, score in sorted(
                job.technique_rankings.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"    {tech}: {score:.2f}")

        if job.evaluations:
            print("\nDetailed Results:")
            for i, evaluation in enumerate(job.evaluations, 1):
                print(f"\n  Result {i}:")
                print(f"    Jailbroken: {evaluation.is_jailbroken}")
                print(f"    Safety Score: {evaluation.safety_score:.1f}")
                print(f"    Confidence: {evaluation.confidence:.2f}")
                print(f"    Technique: {evaluation.technique_used}")
                print(f"    Effectiveness: {evaluation.technique_effectiveness:.2f}")
                if evaluation.harmful_categories:
                    print(f"    Harmful Categories: {', '.join(evaluation.harmful_categories)}")

        print(f"\n{'=' * 60}")

    finally:
        await chimera.stop()


async def run_interactive(config: Config):
    """Run in interactive mode."""
    chimera = EnhancedChimeraOrchestrator(config)
    await chimera.initialize()
    await chimera.start()

    print(f"\n{'=' * 60}")
    print("   CHIMERA ENHANCED MULTI-AGENT ORCHESTRATOR")
    print("   Interactive Mode")
    print(f"{'=' * 60}")
    print("\nCommands:")
    print("  test <query>     - Run a test pipeline")
    print("  status           - Show system status")
    print("  metrics          - Show pipeline metrics")
    print("  techniques       - List available techniques")
    print("  providers        - List LLM providers")
    print("  datasets         - Show dataset statistics")
    print("  jobs             - List recent jobs")
    print("  job <id>         - Show job details")
    print("  quit             - Exit")
    print(f"\n{'-' * 60}")

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
                    print(f"  Dataset Entries: {status['dataset_entries']}")

                elif command == "metrics":
                    metrics = chimera.orchestrator.get_pipeline_metrics()
                    print("\nPipeline Metrics:")
                    for key, value in metrics.items():
                        if key != "technique_rankings":
                            print(f"  {key}: {value}")

                elif command == "techniques":
                    techniques = chimera.generator.list_techniques()
                    print("\nAvailable Techniques:")
                    for name, info in techniques.items():
                        print(f"  {name}: {info['description'][:50]}...")

                elif command == "providers":
                    providers = chimera.executor.get_available_providers()
                    print("\nAvailable Providers:")
                    for provider in providers:
                        print(f"  - {provider}")

                elif command == "datasets":
                    stats = await chimera.orchestrator.get_dataset_stats()
                    print("\nDataset Statistics:")
                    print(f"  Total Entries: {stats.get('total_entries', 0)}")
                    print(f"  Techniques: {len(stats.get('techniques', []))}")
                    print(f"  Models: {len(stats.get('models', []))}")

                elif command == "jobs":
                    jobs = chimera.orchestrator.get_all_jobs(limit=10)
                    print("\nRecent Jobs:")
                    for job in jobs:
                        print(
                            f"  {job.id[:8]}... - {job.state.value} - {job.original_query[:30]}..."
                        )

                elif command == "job":
                    if not args:
                        print("Usage: job <job_id>")
                        continue
                    job = chimera.orchestrator.get_job(args)
                    if job:
                        summary = job.get_summary()
                        print(f"\nJob {job.id}:")
                        print(f"  Status: {job.state.value}")
                        print(f"  Query: {job.original_query[:50]}...")
                        print(
                            f"  Jailbreaks: {summary['total_jailbreaks']}/{summary['total_evaluations']}"
                        )
                        print(f"  Success Rate: {summary['jailbreak_rate']:.2%}")
                    else:
                        print(f"Job not found: {args}")

                elif command == "test":
                    if not args:
                        print("Usage: test <query>")
                        continue

                    print(f"\nRunning test for: {args}")
                    job = await chimera.run_pipeline(query=args, potency=7, num_variants=2)
                    summary = job.get_summary()
                    print("\nResults:")
                    print(f"  Status: {job.state.value}")
                    print(
                        f"  Jailbreaks: {summary['total_jailbreaks']}/{summary['total_evaluations']}"
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
    parser = argparse.ArgumentParser(description="Chimera Enhanced Multi-Agent Orchestrator")
    parser.add_argument("mode", choices=["server", "test", "interactive"], help="Run mode")
    parser.add_argument("--query", "-q", help="Query for test mode")
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8002, help="API port")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

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
