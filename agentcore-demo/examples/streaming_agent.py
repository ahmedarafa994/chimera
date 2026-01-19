"""
Streaming Agent Example
Demonstrates how to use streaming responses with AgentCore
"""
import asyncio
import os
os.environ["BYPASS_TOOL_CONSENT"] = "true"

from strands import Agent
from strands_tools import calculator
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Initialize agent with calculator tool
agent = Agent(
    name="StreamingAgent",
    instructions="You are a helpful math assistant. Use the calculator tool for computations.",
    tools=[calculator],
    callback_handler=None  # Required for streaming
)

# Wrap with AgentCore
app = BedrockAgentCoreApp()

@app.entrypoint
async def agent_invocation(payload, context):
    """Handler for agent invocation with streaming support"""
    user_message = payload.get(
        "prompt",
        "No prompt found in input, please provide a 'prompt' key in your JSON payload"
    )

    print(f"Context: {context}")
    print(f"Processing message: {user_message}")

    # Stream the agent's response
    agent_stream = agent.stream_async(user_message)

    async for event in agent_stream:
        yield event

if __name__ == "__main__":
    app.run()
