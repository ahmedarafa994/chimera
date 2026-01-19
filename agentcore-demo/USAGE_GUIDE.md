# AWS AgentCore Power - Complete Usage Guide

## Overview

This guide demonstrates how to use the aws-agentcore Kiro Power for building, testing, and deploying AI agents.

## What You Have

### MCP Tools Available

The power provides 5 MCP tools through the `agentcore-mcp-server`:

1. **search_agentcore_docs** - Search documentation with ranked results
2. **fetch_agentcore_doc** - Get full documentation pages
3. **manage_agentcore_runtime** - Deployment requirements and CLI commands
4. **manage_agentcore_memory** - Memory resource management
5. **manage_agentcore_gateway** - Gateway configuration and deployment

### Demo Project Structure

```
agentcore-demo/
├── src/
│   └── main.py                    # Basic agent
├── examples/
│   ├── streaming_agent.py         # Streaming responses
│   ├── session_agent.py           # Session management
│   └── README.md                  # Examples documentation
├── .bedrock_agentcore.yaml        # Configuration
├── requirements.txt               # Dependencies
├── README.md                      # Main documentation
├── QUICK_START.md                 # Quick reference
└── USAGE_GUIDE.md                 # This file
```

## Using the Power's MCP Tools

### Example 1: Search Documentation

Ask Kiro:

```
"Search AgentCore docs for streaming responses"
```

Kiro will use:

```javascript
kiroPowers("aws-agentcore", "agentcore-mcp-server", "search_agentcore_docs", {
  "query": "streaming responses",
  "k": 5
})
```

### Example 2: Get Full Documentation

Ask Kiro:

```
"Fetch the complete AgentCore Memory integration guide"
```

Kiro will:

1. Search for memory integration docs
2. Fetch the full content using the URL from search results

### Example 3: Get CLI Commands

Ask Kiro:

```
"How do I create an AgentCore Memory resource?"
```

Kiro will use:

```javascript
kiroPowers("aws-agentcore", "agentcore-mcp-server", "manage_agentcore_memory", {})
```

This returns complete CLI command reference for memory operations.

### Example 4: Get Deployment Info

Ask Kiro:

```
"What are the requirements for deploying to AgentCore Runtime?"
```

Kiro will use:

```javascript
kiroPowers("aws-agentcore", "agentcore-mcp-server", "manage_agentcore_runtime", {})
```

## Development Workflow

### 1. Local Development

```bash
# Navigate to demo
cd agentcore-demo

# Install dependencies
pip install -r requirements.txt

# Start dev server
agentcore dev

# In another terminal, test
agentcore invoke --dev "Hello, what can you do?"
```

### 2. Try Examples

```bash
# Streaming example
agentcore configure --entrypoint examples.streaming_agent:app
agentcore dev
agentcore invoke --dev "Calculate 25 * 47 + 123"

# Session management example
agentcore configure --entrypoint examples.session_agent:app
agentcore dev
agentcore invoke --dev "Hello, my name is Alice" --session-id user1
agentcore invoke --dev "What is my name?" --session-id user1
```

### 3. Deploy to AWS

```bash
# Configure for deployment
agentcore configure --entrypoint src.main:app

# Deploy
agentcore launch

# Test deployed agent
agentcore invoke '{"prompt": "Hello from AWS!"}'

# Check status
agentcore status

# Stop session
agentcore stop-session

# Destroy resources
agentcore destroy --dry-run  # Preview first
agentcore destroy            # Actually destroy
```

## Common Use Cases

### Use Case 1: Building a New Agent

1. Ask Kiro: "Search AgentCore docs for Strands agent examples"
2. Review the examples
3. Modify `src/main.py` with your agent logic
4. Test locally with `agentcore dev`
5. Deploy with `agentcore launch`

### Use Case 2: Adding Memory

1. Ask Kiro: "How do I create AgentCore Memory?"
2. Follow the CLI commands provided
3. Update your agent code to use memory
4. Test and deploy

### Use Case 3: Adding Tools

1. Ask Kiro: "Search AgentCore docs for tool integration"
2. Import tools from `strands_tools`
3. Add to your agent's tools list
4. Test locally

### Use Case 4: Streaming Responses

1. Check `examples/streaming_agent.py`
2. Use `agent.stream_async()` for streaming
3. Yield events in your entrypoint
4. Test with `agentcore invoke --dev`

## Tips & Best Practices

### When to Use Each MCP Tool

- **search_agentcore_docs**: When you need to find information on a topic
- **fetch_agentcore_doc**: When you need complete documentation for a specific page
- **manage_agentcore_runtime**: When deploying or configuring agents
- **manage_agentcore_memory**: When working with persistent memory
- **manage_agentcore_gateway**: When exposing MCP tools to your agent

### Development Tips

1. **Always test locally first** with `agentcore dev`
2. **Use session IDs** for conversation continuity
3. **Check status** after deployment with `agentcore status`
4. **Preview destruction** with `--dry-run` before destroying resources
5. **Use examples** as templates for your own agents

### Asking Kiro for Help

Good questions:

- "Search AgentCore docs for [topic]"
- "How do I [specific task]?"
- "Show me the CLI commands for [feature]"
- "Fetch the documentation for [specific guide]"

Kiro will automatically use the appropriate MCP tools to help you.

## Troubleshooting

### Dev Server Won't Start

Ask Kiro: "Search AgentCore docs for troubleshooting dev server"

Common fixes:

- Check `.bedrock_agentcore.yaml` configuration
- Verify entrypoint path is correct
- Ensure dependencies are installed

### Deployment Fails

Ask Kiro: "What are the deployment requirements for AgentCore?"

Kiro will provide the complete checklist using `manage_agentcore_runtime`.

### Memory Integration Issues

Ask Kiro: "How do I configure AgentCore Memory?"

Kiro will provide CLI commands and configuration guidance.

## Next Steps

1. **Explore examples**: Try streaming and session management
2. **Add capabilities**: Memory, Gateway, tools
3. **Deploy to AWS**: Use `agentcore launch`
4. **Ask Kiro**: Use the power's MCP tools for any questions

## Resources

- [Main README](README.md)
- [Quick Start](QUICK_START.md)
- [Examples](examples/README.md)
- [AgentCore Documentation](https://aws.github.io/bedrock-agentcore-starter-toolkit/)
- [Power Summary](../AGENTCORE_POWER_SUMMARY.md)

---

**Remember**: The aws-agentcore power is always available. Just ask Kiro naturally, and it will use the appropriate MCP tools to help you!
