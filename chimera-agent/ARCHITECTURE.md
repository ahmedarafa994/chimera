# Chimera Agent Architecture

## Overview

The Chimera Technique Recommendation Agent provides two implementations:

1. **Standalone Version** (`standalone_agent.py`) - Direct execution without dependencies
2. **Agent SDK Version** (`main.py`) - Integration with Claude Code CLI

## Why Two Versions?

### The ProcessTransport Issue

The Claude Agent SDK uses `ProcessTransport` for communication between agents and the Claude Code CLI. This architecture means:

- **Agent SDK agents cannot run as standalone scripts**
- They must be invoked through the Claude Code CLI infrastructure
- Running `python main.py` directly causes `CLIConnectionError: ProcessTransport is not ready for writing`

### Solution: Standalone Implementation

The `standalone_agent.py` provides:
- Direct execution with `python standalone_agent.py`
- Same core functionality (technique analysis and recommendations)
- Uses Anthropic API directly instead of Agent SDK
- No dependency on Claude Code CLI

## Architecture Comparison

### Standalone Version
```
User → standalone_agent.py → Anthropic API → Response
```

**Pros:**
- Simple execution model
- No external dependencies
- Easy to integrate with other systems
- Can be deployed as API/service

**Cons:**
- No MCP server integration
- No Claude Code CLI features
- Manual tool implementation

### Agent SDK Version
```
User → Claude Code CLI → ProcessTransport → main.py (Agent SDK) → MCP Tools → Response
```

**Pros:**
- Full MCP server support
- Claude Code CLI integration
- Advanced tool orchestration
- Permission management

**Cons:**
- Requires Claude Code CLI
- Cannot run standalone
- More complex setup

## When to Use Each

### Use Standalone Version When:
- Running as a standalone script
- Integrating with external systems
- Deploying as an API service
- Testing locally without CLI

### Use Agent SDK Version When:
- Building Claude Code plugins
- Need MCP server integration
- Want CLI-managed execution
- Require advanced tool orchestration

## Migration Path

To convert standalone to Agent SDK:
1. Extract core logic into shared modules
2. Create Agent SDK wrapper with ProcessTransport
3. Register with Claude Code CLI
4. Define MCP tools for external integration

To convert Agent SDK to standalone:
1. Replace `query()` with direct Anthropic API calls
2. Implement tool logic directly in agent class
3. Remove ProcessTransport dependencies
4. Add standalone entry point

## Technical Details

### ProcessTransport
- Bidirectional communication channel
- Managed by Claude Code CLI
- Handles message serialization
- Provides lifecycle management

### MCP Tools
- Model Context Protocol integration
- Exposes agent capabilities as tools
- Enables tool chaining and composition
- Requires Agent SDK infrastructure

### Direct API Calls
- Uses `anthropic` Python package
- Simpler request/response model
- No transport layer abstraction
- Direct control over API parameters

## Recommendations

For the Chimera project:
- **Use standalone version** for direct execution and testing
- **Use Agent SDK version** only if integrating with Claude Code CLI plugins
- Consider creating a FastAPI wrapper around standalone version for HTTP API access
- Keep both implementations in sync for core technique logic
