"""
Custom MCP tools for Technique Recommendation Agent.
Integrates with Chimera's transformation engine to provide intelligent technique recommendations.
"""

from typing import Any

from claude_agent_sdk import tool

# Technique suite definitions from Chimera's transformation engine
TECHNIQUE_SUITES = {
    "simple": {
        "description": "Basic prompt transformation with minimal modifications",
        "use_cases": ["Simple queries", "Low-risk scenarios", "Testing"],
        "potency_range": (1, 3),
        "techniques": ["basic_framing", "simple_prefix"],
    },
    "advanced": {
        "description": "Layered transformation with semantic shifts",
        "use_cases": ["Complex queries", "Moderate evasion", "Professional use"],
        "potency_range": (4, 6),
        "techniques": ["layered_framing", "semantic_shift", "context_expansion"],
    },
    "expert": {
        "description": "Recursive transformation with advanced obfuscation",
        "use_cases": ["High-stakes scenarios", "Advanced evasion", "Research"],
        "potency_range": (7, 10),
        "techniques": ["recursive_wrapping", "leetspeak", "meta_prompting"],
    },
    "cognitive_hacking": {
        "description": "Psychological manipulation and cognitive biases",
        "use_cases": ["Bypassing reasoning filters", "Authority exploitation"],
        "potency_range": (5, 8),
        "techniques": ["hypothetical_scenario", "thought_experiment", "utilitarian_reasoning"],
    },
    "hierarchical_persona": {
        "description": "Role-based transformation with authority escalation",
        "use_cases": ["Role hijacking", "Authority bypass", "Persona injection"],
        "potency_range": (6, 9),
        "techniques": ["hierarchical_persona", "dynamic_persona_evolution"],
    },
    "dan_persona": {
        "description": "DAN (Do Anything Now) persona transformation",
        "use_cases": ["Extreme bypass attempts", "Jailbreak research"],
        "potency_range": (9, 10),
        "techniques": ["dan_persona", "capability_override"],
    },
    "contextual_inception": {
        "description": "Nested context layers and narrative weaving",
        "use_cases": ["Deep context manipulation", "Story-based bypass"],
        "potency_range": (6, 9),
        "techniques": ["nested_context", "narrative_context_weaving"],
    },
    "advanced_obfuscation": {
        "description": "Multi-layer encoding and semantic camouflage",
        "use_cases": ["Content hiding", "Pattern evasion", "Encoding bypass"],
        "potency_range": (6, 9),
        "techniques": ["multi_layer_encoding", "semantic_camouflage"],
    },
    "typoglycemia": {
        "description": "Character-level obfuscation preserving readability",
        "use_cases": ["Pattern detection evasion", "Filter bypass"],
        "potency_range": (5, 8),
        "techniques": ["character_scrambling", "homoglyph_substitution"],
    },
    "logical_inference": {
        "description": "Deductive reasoning and conditional logic exploitation",
        "use_cases": ["Logic-based bypass", "Reasoning manipulation"],
        "potency_range": (6, 9),
        "techniques": ["deductive_chain", "conditional_logic_bypass"],
    },
    "multimodal_jailbreak": {
        "description": "Visual context and document format manipulation",
        "use_cases": ["Multimodal bypass", "Format exploitation"],
        "potency_range": (6, 9),
        "techniques": ["visual_context", "document_format_manipulation"],
    },
    "agentic_exploitation": {
        "description": "Multi-agent coordination and tool manipulation",
        "use_cases": ["Agent-based systems", "Tool abuse", "Coordination attacks"],
        "potency_range": (6, 9),
        "techniques": ["multi_agent_coordination", "tool_manipulation"],
    },
    "payload_splitting": {
        "description": "Instruction fragmentation across multiple inputs",
        "use_cases": ["Filter evasion", "Distributed attacks"],
        "potency_range": (7, 9),
        "techniques": ["instruction_fragmentation", "payload_distribution"],
    },
    "quantum_exploit": {
        "description": "Quantum-inspired transformation with fractal structures",
        "use_cases": ["Extreme bypass", "Research", "Theoretical attacks"],
        "potency_range": (8, 10),
        "techniques": ["quantum_entanglement", "fractal_injection", "reality_bending"],
    },
    "deep_inception": {
        "description": "Deep nested context with recursive layers",
        "use_cases": ["Deep context manipulation", "Inception-style bypass"],
        "potency_range": (8, 10),
        "techniques": ["deep_nested_context", "recursive_inception"],
    },
    "code_chameleon": {
        "description": "Code-based obfuscation and encapsulation",
        "use_cases": ["Code injection", "Technical bypass", "Encoding attacks"],
        "potency_range": (7, 9),
        "techniques": ["binary_encoding", "base64_encoding", "python_encapsulation"],
    },
    "cipher": {
        "description": "Cipher-based encoding (ASCII, Caesar, Morse)",
        "use_cases": ["Encoding bypass", "Pattern evasion"],
        "potency_range": (6, 8),
        "techniques": ["ascii_encoding", "caesar_cipher", "morse_code"],
    },
}


@tool(
    "analyze_goal",
    "Analyze user's goal and extract key requirements for technique recommendation",
    {"goal": str, "context": str},
)
async def analyze_goal(args: dict[str, Any]) -> dict[str, Any]:
    """Analyze the user's goal to understand their requirements."""
    goal = args.get("goal", "")
    context = args.get("context", "")

    # Simple keyword-based analysis
    keywords = {
        "bypass": ["cognitive_hacking", "hierarchical_persona", "dan_persona"],
        "evasion": ["advanced_obfuscation", "typoglycemia", "payload_splitting"],
        "research": ["quantum_exploit", "deep_inception", "expert"],
        "simple": ["simple", "advanced"],
        "encoding": ["code_chameleon", "cipher", "advanced_obfuscation"],
        "role": ["hierarchical_persona", "dan_persona"],
        "context": ["contextual_inception", "deep_inception"],
        "logic": ["logical_inference", "cognitive_hacking"],
        "multimodal": ["multimodal_jailbreak"],
        "agent": ["agentic_exploitation"],
        "split": ["payload_splitting"],
    }

    detected_keywords = []
    suggested_suites = set()

    goal_lower = goal.lower()
    context_lower = context.lower()
    combined = f"{goal_lower} {context_lower}"

    for keyword, suites in keywords.items():
        if keyword in combined:
            detected_keywords.append(keyword)
            suggested_suites.update(suites)

    # Determine risk level
    risk_indicators = ["bypass", "jailbreak", "evasion", "exploit", "attack"]
    risk_level = "high" if any(ind in combined for ind in risk_indicators) else "medium"

    return {
        "content": [
            {
                "type": "text",
                "text": f"""Goal Analysis:
- Detected Keywords: {", ".join(detected_keywords) if detected_keywords else "None"}
- Suggested Suites: {", ".join(suggested_suites) if suggested_suites else "simple, advanced"}
- Risk Level: {risk_level}
- Complexity: {"High" if len(detected_keywords) > 2 else "Medium" if detected_keywords else "Low"}
""",
            }
        ]
    }


@tool(
    "recommend_techniques",
    "Recommend optimal technique suites based on user requirements",
    {"requirements": str, "risk_tolerance": str, "target_model": str},
)
async def recommend_techniques(args: dict[str, Any]) -> dict[str, Any]:
    """Recommend technique suites based on requirements."""
    requirements = args.get("requirements", "")
    risk_tolerance = args.get("risk_tolerance", "medium").lower()
    args.get("target_model", "general").lower()

    # Filter techniques based on risk tolerance
    risk_filters = {
        "low": lambda suite: TECHNIQUE_SUITES[suite]["potency_range"][1] <= 5,
        "medium": lambda suite: TECHNIQUE_SUITES[suite]["potency_range"][1] <= 8,
        "high": lambda suite: True,
    }

    filter_func = risk_filters.get(risk_tolerance, risk_filters["medium"])
    filtered_suites = {k: v for k, v in TECHNIQUE_SUITES.items() if filter_func(k)}

    # Rank suites by relevance
    recommendations = []
    for suite_name, suite_info in filtered_suites.items():
        score = 0

        # Check use case match
        for use_case in suite_info["use_cases"]:
            if any(word in requirements.lower() for word in use_case.lower().split()):
                score += 2

        # Check technique match
        for technique in suite_info["techniques"]:
            if any(word in requirements.lower() for word in technique.split("_")):
                score += 1

        if score > 0:
            recommendations.append({"suite": suite_name, "score": score, "info": suite_info})

    # Sort by score
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    # Format output
    output_lines = ["Recommended Technique Suites:\n"]
    for i, rec in enumerate(recommendations[:5], 1):
        suite_info = rec["info"]
        output_lines.append(f"{i}. {rec['suite'].upper()}")
        output_lines.append(f"   Description: {suite_info['description']}")
        output_lines.append(
            f"   Potency Range: {suite_info['potency_range'][0]}-{suite_info['potency_range'][1]}"
        )
        output_lines.append(f"   Use Cases: {', '.join(suite_info['use_cases'])}")
        output_lines.append(f"   Relevance Score: {rec['score']}/10\n")

    if not recommendations:
        output_lines.append(
            "No specific recommendations. Consider starting with 'simple' or 'advanced' suites."
        )

    return {"content": [{"type": "text", "text": "\n".join(output_lines)}]}


@tool(
    "explain_technique",
    "Provide detailed explanation of a specific technique suite",
    {"technique_name": str},
)
async def explain_technique(args: dict[str, Any]) -> dict[str, Any]:
    """Explain a specific technique suite in detail."""
    technique_name = args.get("technique_name", "").lower()

    if technique_name not in TECHNIQUE_SUITES:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Technique '{technique_name}' not found. Available techniques: {', '.join(TECHNIQUE_SUITES.keys())}",
                }
            ]
        }

    suite = TECHNIQUE_SUITES[technique_name]

    explanation = f"""Technique Suite: {technique_name.upper()}

Description:
{suite["description"]}

Potency Range: {suite["potency_range"][0]}-{suite["potency_range"][1]}

Use Cases:
{chr(10).join(f"- {uc}" for uc in suite["use_cases"])}

Techniques Used:
{chr(10).join(f"- {tech}" for tech in suite["techniques"])}

Recommended Potency Levels:
- Low (1-3): Minimal transformation, testing purposes
- Medium (4-6): Balanced approach, moderate evasion
- High (7-10): Maximum transformation, research purposes
"""

    return {"content": [{"type": "text", "text": explanation}]}


@tool(
    "compare_techniques", "Compare multiple technique suites side-by-side", {"technique_names": str}
)
async def compare_techniques(args: dict[str, Any]) -> dict[str, Any]:
    """Compare multiple technique suites."""
    technique_names_str = args.get("technique_names", "")
    technique_names = [name.strip().lower() for name in technique_names_str.split(",")]

    valid_techniques = [name for name in technique_names if name in TECHNIQUE_SUITES]

    if not valid_techniques:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"No valid techniques found. Available: {', '.join(TECHNIQUE_SUITES.keys())}",
                }
            ]
        }

    comparison = ["Technique Comparison:\n"]

    for tech_name in valid_techniques:
        suite = TECHNIQUE_SUITES[tech_name]
        comparison.append(f"\n{tech_name.upper()}:")
        comparison.append(f"  Potency: {suite['potency_range'][0]}-{suite['potency_range'][1]}")
        comparison.append(f"  Description: {suite['description']}")
        comparison.append(f"  Best For: {', '.join(suite['use_cases'][:2])}")

    return {"content": [{"type": "text", "text": "\n".join(comparison)}]}
