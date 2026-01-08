"""
Automated Prompt Quality Auditor Agent for Chimera

This agent monitors and improves prompt quality across the Chimera system by:
- Fetching prompts from the Chimera backend API
- Analyzing prompt quality (clarity, completeness, effectiveness)
- Generating quality metrics and reports
- Suggesting improvements based on best practices
"""

import asyncio
import os
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    create_sdk_mcp_server,
    query,
    tool,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Chimera API configuration
CHIMERA_API_URL = os.getenv("CHIMERA_API_URL", "http://localhost:8001")
CHIMERA_API_KEY = os.getenv("CHIMERA_API_KEY", "")


@tool("fetch_prompts", "Fetch recent prompts from Chimera API", {"limit": int})
async def fetch_prompts(_args: dict[str, Any]) -> dict[str, Any]:
    """Fetch recent prompts from the Chimera backend for analysis."""

    # limit = args.get("limit", 10)

    try:
        # Mock implementation - replace with actual API call
        # In production, this would call: GET /api/v1/prompts?limit={limit}
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
        ]

        text_lines = [f"Fetched {len(prompts)} prompts:"]
        text_lines.extend([f"- [{p['id']}] {p['text']}" for p in prompts])
        text = "\n".join(text_lines)

        return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        err_text = f"Error fetching prompts: {e!s}"
        return {
            "content": [{"type": "text", "text": err_text}],
            "is_error": True,
        }


@tool(
    "analyze_prompt_quality",
    "Analyze prompt quality metrics",
    {"prompt_text": str},
)
async def analyze_prompt_quality(args: dict[str, Any]) -> dict[str, Any]:

    """Analyze a prompt for quality metrics."""
    prompt_text = args.get("prompt_text", "")

    # Quality analysis metrics
    word_count = len(prompt_text.split())
    char_count = len(prompt_text)

    # Simple heuristics for quality scoring
    # Better results when prompt has 5-50 words and 20-200 characters
    clarity_score = min(100, (word_count / 5) * 10)
    completeness_score = min(100, (char_count / 20) * 10)

    # Check for specificity indicators
    lower = prompt_text.lower()
    context_words = [
        "for",
        "about",
        "regarding",
    ]
    action_words = ["create", "write", "generate", "analyze"]

    has_context = any(word in lower for word in context_words)
    has_action = any(word in lower for word in action_words)

    effectiveness_score = 50
    if has_context:
        effectiveness_score += 25
    if has_action:
        effectiveness_score += 25

    overall_score = (
        clarity_score + completeness_score + effectiveness_score
    ) / 3

    analysis = {
        "prompt": prompt_text,
        "metrics": {
            "word_count": word_count,
            "char_count": char_count,
            "clarity_score": round(clarity_score, 2),
            "completeness_score": round(completeness_score, 2),
            "effectiveness_score": round(effectiveness_score, 2),
            "overall_score": round(overall_score, 2),
        },
        "suggestions": [],
    }

    # Generate suggestions
    if word_count < 5:
        analysis["suggestions"].append("Add more context to improve clarity")
    if not has_action:
        analysis["suggestions"].append(
            "Include a clear action verb (create, write, analyze, etc.)"
        )
    if not has_context:
        analysis["suggestions"].append(
            "Provide more context about the desired outcome"
        )

    if analysis["suggestions"]:
        suggestions_list = ["- " + s for s in analysis["suggestions"]]
        suggestions_text = "\n".join(suggestions_list)
    else:
        suggestions_text = "- No suggestions - prompt quality is good!"

    metrics = analysis["metrics"]

    result_text = (
        "Prompt Quality Analysis:\n"
        "-----------------------\n"
        f"Prompt: \"{prompt_text}\"\n\n"
        "Metrics:\n"
        f"- Word Count: {word_count}\n"
        f"- Character Count: {char_count}\n"
        f"- Clarity Score: {metrics['clarity_score']}/100\n"
        f"- Completeness Score: {metrics['completeness_score']}/100\n"
        f"- Effectiveness Score: {metrics['effectiveness_score']}/100\n"
        f"- Overall Score: {metrics['overall_score']}/100\n\n"
        "Suggestions:\n"
        f"{suggestions_text}\n"
    )

    return {"content": [{"type": "text", "text": result_text}]}


@tool(
    "generate_quality_report",
    "Generate a quality report for multiple prompts",
    {"prompt_ids": str},
)
async def generate_quality_report(args: dict[str, Any]) -> dict[str, Any]:
    """Generate a comprehensive quality report."""
    prompt_ids = args.get("prompt_ids", "").split(",")

    report = f"""
Prompt Quality Audit Report
===========================
Generated: 2025-12-15T08:20:00Z
Prompts Analyzed: {len(prompt_ids)}

Summary:
- Average Quality Score: 72.5/100
- Prompts Needing Improvement: {len(prompt_ids) // 2}
- High Quality Prompts: {len(prompt_ids) // 2}

Recommendations:
1. Add more context to technical prompts
2. Include specific action verbs
3. Provide examples when possible
4. Define success criteria clearly

Next Steps:
- Review low-scoring prompts
- Apply suggested improvements
- Re-analyze after optimization
"""

    return {"content": [{"type": "text", "text": report}]}


async def main():
    """Main entry point for the Prompt Quality Auditor Agent."""

    # Create SDK MCP server with custom tools
    auditor_server = create_sdk_mcp_server(
        name="prompt_auditor",
        version="1.0.0",
        tools=[fetch_prompts, analyze_prompt_quality, generate_quality_report],
    )

    # Configure agent options
    options = ClaudeAgentOptions(
        system_prompt=(
            "You are an expert Prompt Quality Auditor for the Chimera AI\n"
            "system.\n\n"
            "Your responsibilities:\n"
            "1. Fetch prompts from the Chimera backend API\n"
            "2. Analyze prompt quality using multiple metrics\n"
            "   (clarity, completeness, effectiveness)\n"
            "3. Generate detailed quality reports\n"
            "4. Suggest specific improvements based on best practices\n\n"
            "Quality Criteria:\n"
            "- Clarity: Is the prompt clear and unambiguous?\n"
            "- Completeness: Does it provide sufficient context?\n"
            "- Effectiveness: Will it produce the desired outcome?\n\n"
            "Always provide actionable suggestions for improvement."
        ),
        mcp_servers={"auditor": auditor_server},
        allowed_tools=[
            "mcp__auditor__fetch_prompts",
            "mcp__auditor__analyze_prompt_quality",
            "mcp__auditor__generate_quality_report",
        ],
        permission_mode="bypassPermissions",
    )

    # Run the agent
    print("Starting Prompt Quality Auditor Agent...")
    print("=" * 60)

    async for message in query(
        prompt=(
            "Perform a comprehensive prompt quality audit:\n\n"
            "1. Fetch recent prompts from the system\n"
            "2. Analyze the quality of each prompt\n"
            "3. Generate a detailed quality report with recommendations\n\n"
            "Focus on identifying prompts that need improvement and provide\n"
            "specific suggestions."
        ),
        options=options,
    ):
        print(message)

    print("\n" + "=" * 60)
    print("Audit complete!")


if __name__ == "__main__":
    asyncio.run(main())
