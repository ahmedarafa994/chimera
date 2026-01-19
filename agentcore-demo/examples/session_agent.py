"""
Session Management Example
Demonstrates how to maintain conversation state across requests
"""
import os
os.environ["BYPASS_TOOL_CONSENT"] = "true"

from strands import Agent
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.context import RequestContext

# Initialize agent
agent = Agent(
    name="SessionAgent",
    instructions="You are a helpful assistant that remembers conversation context."
)

# Wrap with AgentCore
app = BedrockAgentCoreApp()

# Simple in-memory session storage (use database in production)
sessions = {}

@app.entrypoint
def agent_invocation(payload, context: RequestContext):
    """Handler with session management"""
    session_id = context.session_id or "default"
    user_message = payload.get("prompt", "Hello")

    # Initialize session if new
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "count": 0
        }

    # Add message to session history
    sessions[session_id]["messages"].append(user_message)
    sessions[session_id]["count"] += 1

    # Build context from session history
    conversation_context = "\n".join([
        f"User: {msg}" for msg in sessions[session_id]["messages"][-5:]  # Last 5 messages
    ])

    # Invoke agent with context
    full_prompt = f"Conversation history:\n{conversation_context}\n\nCurrent message: {user_message}"
    result = agent(full_prompt)

    return {
        "result": result.message,
        "session_id": session_id,
        "message_count": sessions[session_id]["count"]
    }

if __name__ == "__main__":
    app.run()
