"""
Simple AgentCore Demo Agent
A basic Strands agent wrapped for AgentCore deployment
"""
import os
os.environ["BYPASS_TOOL_CONSENT"] = "true"

from strands import Agent
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Create a simple agent
agent = Agent(
    name="DemoAgent",
    instructions="You are a helpful assistant. Answer questions clearly and concisely."
)

# Wrap with AgentCore
app = BedrockAgentCoreApp()

@app.entrypoint
def agent_invocation(payload, context):
    """Handler for agent invocation"""
    user_message = payload.get(
        "prompt",
        "No prompt found in input, please provide a 'prompt' key in your JSON payload"
    )

    # Invoke the agent
    result = agent(user_message)

    # Log for debugging
    print(f"User message: {user_message}")
    print(f"Agent response: {result.message}")

    return {"result": result.message}

if __name__ == "__main__":
    app.run()
