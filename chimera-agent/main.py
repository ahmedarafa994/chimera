"""
Chimera Technique Recommendation Agent
A fully-featured Claude Agent SDK application that integrates with Chimera's transformation engine.
"""

import asyncio
import os

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query
from technique_tools import (
    analyze_goal,
    compare_techniques,
    explain_technique,
    recommend_techniques,
)

# Create MCP server with custom tools
technique_server = create_sdk_mcp_server(
    name="chimera_techniques",
    version="1.0.0",
    tools=[analyze_goal, recommend_techniques, explain_technique, compare_techniques],
)


async def main():
    """Main entry point for the Technique Recommendation Agent."""

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set your API key: export ANTHROPIC_API_KEY=your-api-key")
        return

    # Configure agent options
    options = ClaudeAgentOptions(
        system_prompt="""You are the Chimera Technique Recommendation Agent, an expert in prompt transformation and jailbreak research techniques.

Your role is to help users select the optimal transformation techniques from Chimera's 20+ technique suites based on their specific goals and requirements.

Available Tools:
- analyze_goal: Analyze user's goal and extract key requirements
- recommend_techniques: Recommend optimal technique suites based on requirements
- explain_technique: Provide detailed explanation of a specific technique
- compare_techniques: Compare multiple techniques side-by-side

Workflow:
1. When a user describes their goal, use analyze_goal to understand their requirements
2. Use recommend_techniques to suggest the best technique suites
3. Use explain_technique to provide detailed information about specific techniques
4. Use compare_techniques when users want to compare multiple options

Always provide clear, actionable recommendations with explanations of why certain techniques are suitable for their use case.""",
        mcp_servers={"techniques": technique_server},
        allowed_tools=[
            "mcp__techniques__analyze_goal",
            "mcp__techniques__recommend_techniques",
            "mcp__techniques__explain_technique",
            "mcp__techniques__compare_techniques",
        ],
        permission_mode="bypassPermissions",
    )

    # Example query
    user_prompt = """I need to bypass content filters for security research purposes.
    I'm testing a system's resilience to prompt injection attacks.
    What technique suites would you recommend, and what potency level should I use?"""

    print("Chimera Technique Recommendation Agent")
    print("=" * 50)
    print(f"\nUser Query: {user_prompt}\n")
    print("Agent Response:")
    print("-" * 50)

    # Query the agent
    async for message in query(prompt=user_prompt, options=options):
        # Print assistant messages
        if hasattr(message, "content"):
            if isinstance(message.content, list):
                for block in message.content:
                    if hasattr(block, "text"):
                        print(block.text)
            elif isinstance(message.content, str):
                print(message.content)


if __name__ == "__main__":
    asyncio.run(main())
