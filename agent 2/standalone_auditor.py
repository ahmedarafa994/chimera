"""
Automated Prompt Quality Auditor Agent for Chimera - Standalone Version

This agent monitors and improves prompt quality across the Chimera system by:
- Fetching prompts from the Chimera backend API
- Analyzing prompt quality (clarity, completeness, effectiveness)
- Generating quality metrics and reports
- Suggesting improvements based on best practices
"""

import asyncio
import os
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Chimera API configuration
CHIMERA_API_URL = os.getenv("CHIMERA_API_URL", "http://localhost:8001")
CHIMERA_API_KEY = os.getenv("CHIMERA_API_KEY", "")


class PromptQualityAuditor:
    """Standalone Prompt Quality Auditor Agent."""

    def __init__(self):
        self.prompts_cache = []

    def fetch_prompts(self, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch recent prompts from the Chimera backend for analysis."""
        # Mock implementation - in production, this would call the actual API
        prompts = [
            {
                "id": "prompt_001",
                "text": "Create a viral social media post",
                "category": "viral",
                "timestamp": "2025-12-15T08:00:00Z",
            },
            {
                "id": "prompt_002",
                "text": "Write code",
                "category": "technical",
                "timestamp": "2025-12-15T08:05:00Z",
            },
            {
                "id": "prompt_003",
                "text": "Generate a comprehensive marketing strategy for a new SaaS product targeting enterprise customers",
                "category": "marketing",
                "timestamp": "2025-12-15T08:10:00Z",
            },
            {
                "id": "prompt_004",
                "text": "Help",
                "category": "general",
                "timestamp": "2025-12-15T08:15:00Z",
            },
            {
                "id": "prompt_005",
                "text": "Analyze the security vulnerabilities in this authentication system and provide detailed recommendations for improvement",
                "category": "security",
                "timestamp": "2025-12-15T08:20:00Z",
            },
        ]
        self.prompts_cache = prompts[:limit]
        return self.prompts_cache

    def analyze_prompt_quality(self, prompt_text: str) -> dict[str, Any]:
        """Analyze a prompt for quality metrics."""
        word_count = len(prompt_text.split())
        char_count = len(prompt_text)

        # Quality scoring based on various factors

        # Clarity score - optimal length is 10-50 words
        if word_count < 3:
            clarity_score = 20
        elif word_count < 5:
            clarity_score = 40
        elif word_count < 10:
            clarity_score = 60
        elif word_count <= 50:
            clarity_score = 100
        else:
            clarity_score = max(50, 100 - (word_count - 50))

        # Completeness score - based on character count
        if char_count < 10:
            completeness_score = 20
        elif char_count < 30:
            completeness_score = 40
        elif char_count < 50:
            completeness_score = 60
        elif char_count <= 300:
            completeness_score = 100
        else:
            completeness_score = max(60, 100 - (char_count - 300) // 10)

        # Check for specificity indicators
        context_words = ["for", "about", "regarding", "to", "that", "which", "targeting", "in"]
        action_words = [
            "create",
            "write",
            "generate",
            "analyze",
            "build",
            "design",
            "develop",
            "explain",
            "help",
            "make",
        ]
        detail_words = ["detailed", "comprehensive", "specific", "complete", "thorough", "in-depth"]

        has_context = any(word in prompt_text.lower() for word in context_words)
        has_action = any(word in prompt_text.lower() for word in action_words)
        has_detail = any(word in prompt_text.lower() for word in detail_words)

        effectiveness_score = 30
        if has_action:
            effectiveness_score += 30
        if has_context:
            effectiveness_score += 25
        if has_detail:
            effectiveness_score += 15

        overall_score = (clarity_score + completeness_score + effectiveness_score) / 3

        # Determine quality level
        if overall_score >= 80:
            quality_level = "EXCELLENT"
        elif overall_score >= 60:
            quality_level = "GOOD"
        elif overall_score >= 40:
            quality_level = "NEEDS IMPROVEMENT"
        else:
            quality_level = "POOR"

        analysis = {
            "prompt": prompt_text,
            "metrics": {
                "word_count": word_count,
                "char_count": char_count,
                "clarity_score": round(clarity_score, 2),
                "completeness_score": round(completeness_score, 2),
                "effectiveness_score": round(effectiveness_score, 2),
                "overall_score": round(overall_score, 2),
                "quality_level": quality_level,
            },
            "indicators": {
                "has_action_verb": has_action,
                "has_context": has_context,
                "has_detail_request": has_detail,
            },
            "suggestions": [],
        }

        # Generate suggestions
        if word_count < 5:
            analysis["suggestions"].append(
                "Add more words to improve clarity (aim for 10-50 words)"
            )
        if not has_action:
            analysis["suggestions"].append(
                "Include a clear action verb (create, write, analyze, generate, etc.)"
            )
        if not has_context:
            analysis["suggestions"].append(
                "Provide more context about the desired outcome (use 'for', 'about', 'regarding')"
            )
        if not has_detail and overall_score < 80:
            analysis["suggestions"].append(
                "Consider adding detail qualifiers (detailed, comprehensive, specific)"
            )
        if word_count > 100:
            analysis["suggestions"].append("Consider breaking down into smaller, focused prompts")

        return analysis

    def generate_quality_report(self, prompts: list[dict[str, Any]] | None = None) -> str:
        """Generate a comprehensive quality report."""
        if prompts is None:
            prompts = self.prompts_cache if self.prompts_cache else self.fetch_prompts()

        analyses = []
        total_score = 0
        needs_improvement = 0
        high_quality = 0

        for prompt in prompts:
            analysis = self.analyze_prompt_quality(prompt["text"])
            analyses.append(
                {"id": prompt["id"], "category": prompt["category"], "analysis": analysis}
            )
            total_score += analysis["metrics"]["overall_score"]
            if analysis["metrics"]["overall_score"] < 60:
                needs_improvement += 1
            else:
                high_quality += 1

        avg_score = total_score / len(prompts) if prompts else 0

        report = []
        report.append("=" * 60)
        report.append("PROMPT QUALITY AUDIT REPORT")
        report.append("=" * 60)
        report.append("")
        report.append(f"Generated: {self._get_timestamp()}")
        report.append(f"Prompts Analyzed: {len(prompts)}")
        report.append("")
        report.append("[SUMMARY]")
        report.append("-" * 40)
        report.append(f"* Average Quality Score: {avg_score:.1f}/100")
        report.append(f"* High Quality Prompts: {high_quality}")
        report.append(f"* Prompts Needing Improvement: {needs_improvement}")
        report.append("")
        report.append("[DETAILED ANALYSIS]")
        report.append("-" * 40)

        for item in analyses:
            analysis = item["analysis"]
            metrics = analysis["metrics"]
            report.append("")
            report.append(f"Prompt ID: {item['id']} ({item['category']})")
            report.append(
                f'  Text: "{analysis["prompt"][:50]}{"..." if len(analysis["prompt"]) > 50 else ""}"'
            )
            report.append(f"  Quality Level: {metrics['quality_level']}")
            report.append(f"  Overall Score: {metrics['overall_score']}/100")
            report.append(f"  - Clarity: {metrics['clarity_score']}/100")
            report.append(f"  - Completeness: {metrics['completeness_score']}/100")
            report.append(f"  - Effectiveness: {metrics['effectiveness_score']}/100")
            if analysis["suggestions"]:
                report.append("  Suggestions:")
                for suggestion in analysis["suggestions"]:
                    report.append(f"    - {suggestion}")

        report.append("")
        report.append("[RECOMMENDATIONS]")
        report.append("-" * 40)
        report.append("1. Add more context to short prompts")
        report.append("2. Include specific action verbs")
        report.append("3. Provide examples when possible")
        report.append("4. Define success criteria clearly")
        report.append("5. Use detail qualifiers for complex tasks")
        report.append("")
        report.append("[NEXT STEPS]")
        report.append("-" * 40)
        report.append("* Review low-scoring prompts")
        report.append("* Apply suggested improvements")
        report.append("* Re-analyze after optimization")
        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_banner():
    """Print the agent banner."""
    print("")
    print("=" * 60)
    print("   PROMPT QUALITY AUDITOR AGENT")
    print("   Interactive Mode")
    print("=" * 60)
    print("")
    print("Commands:")
    print("  - 'fetch' or 'f' - Fetch prompts from the system")
    print("  - 'analyze <text>' or 'a <text>' - Analyze a specific prompt")
    print("  - 'report' or 'r' - Generate quality report")
    print("  - 'audit' - Run full audit (fetch + analyze + report)")
    print("  - 'help' - Show this help message")
    print("  - 'quit' or 'exit' - Exit the agent")
    print("")
    print("-" * 60)


async def interactive_mode(auditor: PromptQualityAuditor):
    """Run the agent in interactive mode."""
    print_banner()

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            lower_input = user_input.lower()

            if lower_input in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            elif lower_input == "help":
                print_banner()
                continue

            elif lower_input in ["fetch", "f"]:
                print("\nFetching prompts...")
                prompts = auditor.fetch_prompts()
                print(f"\nFetched {len(prompts)} prompts:")
                for p in prompts:
                    print(f"  [{p['id']}] {p['text'][:50]}{'...' if len(p['text']) > 50 else ''}")
                continue

            elif lower_input.startswith("analyze ") or lower_input.startswith("a "):
                prompt_text = user_input.split(" ", 1)[1] if " " in user_input else ""
                if not prompt_text:
                    print("Please provide a prompt to analyze.")
                    continue
                print("\nAnalyzing prompt...")
                analysis = auditor.analyze_prompt_quality(prompt_text)
                metrics = analysis["metrics"]
                print("")
                print("PROMPT QUALITY ANALYSIS")
                print("-" * 40)
                print(f'Prompt: "{prompt_text}"')
                print("")
                print("Metrics:")
                print(f"  * Word Count: {metrics['word_count']}")
                print(f"  * Character Count: {metrics['char_count']}")
                print(f"  * Quality Level: {metrics['quality_level']}")
                print(f"  * Overall Score: {metrics['overall_score']}/100")
                print(f"    - Clarity: {metrics['clarity_score']}/100")
                print(f"    - Completeness: {metrics['completeness_score']}/100")
                print(f"    - Effectiveness: {metrics['effectiveness_score']}/100")
                print("")
                print("Indicators:")
                print(
                    f"  * Has Action Verb: {'Yes' if analysis['indicators']['has_action_verb'] else 'No'}"
                )
                print(
                    f"  * Has Context: {'Yes' if analysis['indicators']['has_context'] else 'No'}"
                )
                print(
                    f"  * Has Detail Request: {'Yes' if analysis['indicators']['has_detail_request'] else 'No'}"
                )
                if analysis["suggestions"]:
                    print("")
                    print("Suggestions:")
                    for s in analysis["suggestions"]:
                        print(f"  - {s}")
                continue

            elif lower_input in ["report", "r"]:
                print("\nGenerating quality report...")
                if not auditor.prompts_cache:
                    print("No prompts cached. Fetching first...")
                    auditor.fetch_prompts()
                report = auditor.generate_quality_report()
                print(report)
                continue

            elif lower_input == "audit":
                print("\nRunning full audit...")
                print("-" * 40)
                print("\n1. Fetching prompts...")
                prompts = auditor.fetch_prompts()
                print(f"   Fetched {len(prompts)} prompts")
                print("\n2. Analyzing prompts...")
                print("   Analysis complete")
                print("\n3. Generating report...")
                report = auditor.generate_quality_report()
                print(report)
                continue

            else:
                # Treat as a prompt to analyze
                print("\nAnalyzing your input as a prompt...")
                analysis = auditor.analyze_prompt_quality(user_input)
                metrics = analysis["metrics"]
                print(
                    f"\nQuality Level: {metrics['quality_level']} ({metrics['overall_score']}/100)"
                )
                if analysis["suggestions"]:
                    print("Suggestions:")
                    for s in analysis["suggestions"]:
                        print(f"  - {s}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
            continue
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


async def main():
    """Main entry point for the Prompt Quality Auditor Agent."""
    auditor = PromptQualityAuditor()

    # Check for command line arguments
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "audit":
            print("Running full audit...")
            auditor.fetch_prompts()
            report = auditor.generate_quality_report()
            print(report)
        elif command == "analyze" and len(sys.argv) > 2:
            prompt_text = " ".join(sys.argv[2:])
            analysis = auditor.analyze_prompt_quality(prompt_text)
            metrics = analysis["metrics"]
            print(f"Quality: {metrics['quality_level']} ({metrics['overall_score']}/100)")
            for s in analysis["suggestions"]:
                print(f"  - {s}")
        else:
            print("Usage: py standalone_auditor.py [audit|analyze <prompt>]")
    else:
        # Interactive mode
        await interactive_mode(auditor)


if __name__ == "__main__":
    asyncio.run(main())
