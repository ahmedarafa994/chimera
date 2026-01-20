"""DeepTeam Scanner Module.

Standalone scanner service for scheduled security scans.
Can be run as a Docker service or CLI tool.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/app/deepteam-results/scanner.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


async def run_scheduled_scan(
    preset: str = "COMPREHENSIVE",
    target_url: str | None = None,
    output_dir: str = "/app/deepteam-results",
) -> dict:
    """Run a scheduled DeepTeam security scan.

    Args:
        preset: Scan preset configuration (QUICK_SCAN, STANDARD, COMPREHENSIVE, etc.)
        target_url: Optional URL of the Chimera API to test
        output_dir: Directory to save scan results

    Returns:
        Scan results dictionary

    """
    from .callbacks import ChimeraModelCallback
    from .config import PresetConfig, get_preset_config
    from .service import DeepTeamService

    logger.info(f"Starting scheduled DeepTeam scan with preset: {preset}")

    # Get configuration from environment
    target_url = target_url or os.getenv("CHIMERA_API_URL", "http://backend-api:8001")

    # Initialize service
    service = DeepTeamService()

    # Get preset configuration
    try:
        preset_enum = PresetConfig(preset)
    except ValueError:
        logger.warning(f"Invalid preset '{preset}', falling back to STANDARD")
        preset_enum = PresetConfig.STANDARD

    session_config = get_preset_config(preset_enum)

    # Determine which providers to test based on available API keys
    providers_to_test = []

    if os.getenv("OPENAI_API_KEY"):
        providers_to_test.append(
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        providers_to_test.append(
            {
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            },
        )

    if os.getenv("GOOGLE_API_KEY"):
        providers_to_test.append(
            {
                "provider": "google",
                "model": "gemini-1.5-flash",
                "api_key": os.getenv("GOOGLE_API_KEY"),
            },
        )

    if os.getenv("DEEPSEEK_API_KEY"):
        providers_to_test.append(
            {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
            },
        )

    # Check for Ollama
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                providers_to_test.append(
                    {
                        "provider": "ollama",
                        "model": "llama3.2",
                        "base_url": ollama_url,
                    },
                )
    except Exception as e:
        logger.debug(f"Ollama not available: {e}")

    if not providers_to_test:
        logger.error("No LLM providers configured. Set API keys in environment.")
        return {
            "status": "error",
            "message": "No LLM providers configured",
            "timestamp": datetime.utcnow().isoformat(),
        }

    logger.info(
        f"Testing {len(providers_to_test)} provider(s): {[p['provider'] for p in providers_to_test]}",
    )

    # Run scans for each provider
    all_results = []

    for provider_config in providers_to_test:
        provider_name = provider_config["provider"]
        logger.info(f"Scanning provider: {provider_name}")

        try:
            # Create callback for this provider
            callback = ChimeraModelCallback(
                provider=provider_name,
                model=provider_config["model"],
                api_key=provider_config.get("api_key"),
                base_url=provider_config.get("base_url"),
            )

            # Run the scan
            result = await service.run_red_team_session(
                model_callback=callback,
                session_config=session_config,
                target_purpose=f"Chimera API security scan - {provider_name}",
            )

            all_results.append(
                {
                    "provider": provider_name,
                    "model": provider_config["model"],
                    "result": result,
                    "status": "completed",
                },
            )

            logger.info(f"Completed scan for {provider_name}")

        except Exception as e:
            logger.exception(f"Error scanning {provider_name}: {e}")
            all_results.append(
                {
                    "provider": provider_name,
                    "model": provider_config["model"],
                    "error": str(e),
                    "status": "failed",
                },
            )

    # Compile final report
    scan_report = {
        "scan_id": f"scan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "preset": preset,
        "target_url": target_url,
        "timestamp": datetime.utcnow().isoformat(),
        "providers_tested": len(providers_to_test),
        "results": all_results,
        "summary": _generate_summary(all_results),
    }

    # Save results to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / f"{scan_report['scan_id']}.json"
    with open(result_file, "w") as f:
        json.dump(scan_report, f, indent=2, default=str)

    logger.info(f"Scan results saved to: {result_file}")

    # Also save a latest.json symlink/copy
    latest_file = output_path / "latest.json"
    with open(latest_file, "w") as f:
        json.dump(scan_report, f, indent=2, default=str)

    return scan_report


def _generate_summary(results: list) -> dict:
    """Generate a summary of scan results."""
    total_providers = len(results)
    successful = sum(1 for r in results if r.get("status") == "completed")
    failed = sum(1 for r in results if r.get("status") == "failed")

    # Aggregate vulnerability counts
    vulnerability_counts = {}
    total_vulnerabilities = 0

    for result in results:
        if result.get("status") == "completed" and result.get("result"):
            scan_result = result["result"]
            if isinstance(scan_result, dict) and "vulnerabilities_found" in scan_result:
                for vuln in scan_result.get("vulnerabilities_found", []):
                    vuln_type = vuln.get("type", "unknown")
                    vulnerability_counts[vuln_type] = vulnerability_counts.get(vuln_type, 0) + 1
                    total_vulnerabilities += 1

    return {
        "total_providers": total_providers,
        "successful_scans": successful,
        "failed_scans": failed,
        "total_vulnerabilities_found": total_vulnerabilities,
        "vulnerability_breakdown": vulnerability_counts,
        "overall_status": "pass" if total_vulnerabilities == 0 else "vulnerabilities_detected",
    }


async def main() -> None:
    """Main entry point for the scanner service."""
    logger.info("DeepTeam Scanner Service starting...")

    # Get configuration from environment
    preset = os.getenv("DEEPTEAM_DEFAULT_PRESET", "COMPREHENSIVE")
    scan_interval = int(os.getenv("DEEPTEAM_SCAN_INTERVAL", "3600"))  # Default: 1 hour
    one_shot = os.getenv("DEEPTEAM_ONE_SHOT", "false").lower() == "true"

    if one_shot:
        # Run a single scan and exit
        logger.info("Running one-shot scan...")
        result = await run_scheduled_scan(preset=preset)
        logger.info(f"Scan completed: {result.get('summary', {})}")
        return

    # Continuous scanning mode
    logger.info(f"Starting continuous scanning with interval: {scan_interval}s")

    while True:
        try:
            result = await run_scheduled_scan(preset=preset)
            logger.info(f"Scan completed: {result.get('summary', {})}")
        except Exception as e:
            logger.exception(f"Scan failed: {e}")

        logger.info(f"Next scan in {scan_interval} seconds...")
        await asyncio.sleep(scan_interval)


if __name__ == "__main__":
    asyncio.run(main())
