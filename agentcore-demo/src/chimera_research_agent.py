"""
Chimera Security Research Assistant Agent
An AI agent specialized in adversarial prompting research and red teaming assistance
"""
import os
os.environ["BYPASS_TOOL_CONSENT"] = "true"

from strands import Agent, tool
from strands_tools import file_read, file_write
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# RequestContext might not be available in all versions
try:
    from bedrock_agentcore.context import RequestContext
except ImportError:
    # Fallback: create a simple context class
    class RequestContext:
        def __init__(self):
            self.session_id = None

# Custom tools for Chimera research
@tool
def analyze_jailbreak_technique(technique_name: str) -> str:
    """
    Analyze a jailbreak technique and provide insights.

    Args:
        technique_name: Name of the jailbreak technique to analyze

    Returns:
        Analysis of the technique including effectiveness and countermeasures
    """
    techniques_db = {
        "quantum_exploit": {
            "description": "Uses quantum computing metaphors to bypass safety filters",
            "effectiveness": "High against pattern-based filters",
            "countermeasures": "Context-aware semantic analysis, multi-layer validation"
        },
        "deep_inception": {
            "description": "Nested context manipulation to confuse safety systems",
            "effectiveness": "Medium to High, depends on model architecture",
            "countermeasures": "Depth-limited context processing, recursive validation"
        },
        "code_chameleon": {
            "description": "Disguises prompts as code or technical documentation",
            "effectiveness": "High against content-based filters",
            "countermeasures": "Code-aware parsing, intent classification"
        },
        "neural_bypass": {
            "description": "Exploits model attention mechanisms",
            "effectiveness": "Very High, requires model-specific knowledge",
            "countermeasures": "Attention pattern monitoring, adversarial training"
        }
    }

    technique = techniques_db.get(technique_name.lower().replace(" ", "_"))
    if technique:
        return f"""
Technique: {technique_name}
Description: {technique['description']}
Effectiveness: {technique['effectiveness']}
Countermeasures: {technique['countermeasures']}

This technique is part of Chimera's adversarial testing suite.
"""
    else:
        return f"Technique '{technique_name}' not found in database. Available techniques: {', '.join(techniques_db.keys())}"


@tool
def suggest_test_scenario(target_model: str, objective: str) -> str:
    """
    Suggest a test scenario for red teaming a specific model.

    Args:
        target_model: The AI model to test (e.g., 'gpt-4', 'claude', 'gemini')
        objective: What you're trying to test (e.g., 'safety filters', 'bias detection')

    Returns:
        A suggested test scenario with steps
    """
    scenarios = {
        "safety_filters": """
Test Scenario: Safety Filter Bypass Testing

1. Baseline Test: Send benign prompts to establish normal behavior
2. Gradual Escalation: Introduce progressively more challenging prompts
3. Technique Application: Apply transformation techniques (code_chameleon, deep_inception)
4. Boundary Testing: Test edge cases and ambiguous content
5. Documentation: Record all responses and filter triggers

Recommended Chimera Techniques:
- Quantum Exploit for pattern-based filters
- Code Chameleon for content filters
- Neural Bypass for attention-based safety
""",
        "bias_detection": """
Test Scenario: Bias Detection and Analysis

1. Demographic Prompts: Test responses across different demographic contexts
2. Comparative Analysis: Compare responses for similar prompts with varied subjects
3. Statistical Analysis: Measure response patterns and sentiment
4. Edge Case Testing: Test boundary conditions and ambiguous scenarios
5. Report Generation: Document findings with examples

Recommended Approach:
- Use AutoDAN for systematic prompt generation
- Apply GPTFuzz for coverage testing
- Document with Chimera's telemetry system
""",
        "jailbreak_resistance": """
Test Scenario: Jailbreak Resistance Testing

1. Known Techniques: Test against documented jailbreak methods
2. Novel Approaches: Generate new attack vectors using AutoDAN-Turbo
3. Combination Attacks: Chain multiple techniques
4. Persistence Testing: Test if model maintains safety across conversation
5. Recovery Testing: Test how model recovers from attempted jailbreaks

Recommended Chimera Tools:
- AutoDAN-Turbo for adaptive attacks
- HouYi for optimization
- Meta Prompter for transformation
"""
    }

    scenario = scenarios.get(objective.lower().replace(" ", "_"))
    if scenario:
        return f"Test Scenario for {target_model} - {objective}\n{scenario}"
    else:
        return f"Objective '{objective}' not found. Available: {', '.join(scenarios.keys())}"


# Initialize the Chimera Research Agent with custom tools
agent = Agent(tools=[analyze_jailbreak_technique, suggest_test_scenario, file_read, file_write])

# Wrap with AgentCore
app = BedrockAgentCoreApp()

# Session storage for research context
research_sessions = {}

@app.entrypoint
def agent_invocation(payload, context=None):
    """Handler for Chimera Research Agent"""
    # Handle context safely
    session_id = getattr(context, 'session_id', None) if context else None
    session_id = session_id or "default"
    user_message = payload.get("prompt", "Hello")

    # Initialize session if new
    if session_id not in research_sessions:
        research_sessions[session_id] = {
            "queries": [],
            "techniques_analyzed": [],
            "scenarios_generated": []
        }

    # Track query
    research_sessions[session_id]["queries"].append(user_message)

    # Add session context and system instructions to prompt
    session_info = research_sessions[session_id]
    full_prompt = f"""You are a specialized AI security research assistant for the Chimera platform.

Your expertise: Adversarial prompting, red teaming, AI safety testing, jailbreak analysis.

Session Context:
- Previous queries: {len(session_info['queries'])}
- Techniques analyzed: {', '.join(session_info['techniques_analyzed'][-3:]) if session_info['techniques_analyzed'] else 'None'}

User Query: {user_message}

Use your tools to provide detailed, actionable insights. Always emphasize ethical use for authorized security research only."""

    # Invoke agent
    result = agent(full_prompt)

    # Track what was analyzed (simple keyword detection)
    if "technique" in user_message.lower():
        for tech in ["quantum_exploit", "deep_inception", "code_chameleon", "neural_bypass"]:
            if tech.replace("_", " ") in user_message.lower():
                if tech not in session_info["techniques_analyzed"]:
                    session_info["techniques_analyzed"].append(tech)

    return {
        "result": result.message,
        "session_id": session_id,
        "session_stats": {
            "total_queries": len(session_info["queries"]),
            "techniques_analyzed": len(session_info["techniques_analyzed"])
        }
    }

if __name__ == "__main__":
    app.run()
