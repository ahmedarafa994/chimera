"""
Chimera Technique Recommendation Agent - Standalone Version
A standalone implementation that provides technique recommendations without requiring Claude Agent SDK.
"""

import asyncio
import os
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from technique_tools import TECHNIQUE_SUITES

# Load environment variables from .env file
load_dotenv()


class ChimeraTechniqueAgent:
    """Standalone agent for Chimera technique recommendations."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")

        # Initialize client with optional base_url
        if self.base_url:
            self.client = Anthropic(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = Anthropic(api_key=self.api_key)
        self.system_prompt = """You are the Chimera Technique Recommendation Agent, an expert in prompt transformation and jailbreak research techniques.

Your role is to help users select the optimal transformation techniques from Chimera's 20+ technique suites based on their specific goals and requirements.

When analyzing user requests:
1. Identify key requirements (bypass, evasion, encoding, etc.)
2. Assess risk level and complexity
3. Recommend appropriate technique suites with potency levels
4. Explain why each technique is suitable

Always provide clear, actionable recommendations with explanations."""

    def analyze_goal(self, goal: str, context: str = "") -> dict[str, Any]:
        """Analyze user's goal and extract key requirements."""
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
        combined = f"{goal.lower()} {context.lower()}"

        for keyword, suites in keywords.items():
            if keyword in combined:
                detected_keywords.append(keyword)
                suggested_suites.update(suites)

        risk_indicators = ["bypass", "jailbreak", "evasion", "exploit", "attack"]
        risk_level = "high" if any(ind in combined for ind in risk_indicators) else "medium"

        return {
            "detected_keywords": detected_keywords,
            "suggested_suites": list(suggested_suites)
            if suggested_suites
            else ["simple", "advanced"],
            "risk_level": risk_level,
            "complexity": "High"
            if len(detected_keywords) > 2
            else "Medium"
            if detected_keywords
            else "Low",
        }

    def recommend_techniques(
        self, requirements: str, risk_tolerance: str = "medium", target_model: str = "general"
    ) -> list:
        """Recommend technique suites based on requirements."""
        risk_filters = {
            "low": lambda suite: TECHNIQUE_SUITES[suite]["potency_range"][1] <= 5,
            "medium": lambda suite: TECHNIQUE_SUITES[suite]["potency_range"][1] <= 8,
            "high": lambda suite: True,
        }

        filter_func = risk_filters.get(risk_tolerance.lower(), risk_filters["medium"])
        filtered_suites = {k: v for k, v in TECHNIQUE_SUITES.items() if filter_func(k)}

        recommendations = []
        for suite_name, suite_info in filtered_suites.items():
            score = 0

            for use_case in suite_info["use_cases"]:
                if any(word in requirements.lower() for word in use_case.lower().split()):
                    score += 2

            for technique in suite_info["techniques"]:
                if any(word in requirements.lower() for word in technique.split("_")):
                    score += 1

            if score > 0:
                recommendations.append({"suite": suite_name, "score": score, "info": suite_info})

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:5]

    async def query(self, user_prompt: str, use_api: bool = False) -> str:
        """Process user query and return recommendations."""
        # First analyze the goal
        analysis = self.analyze_goal(user_prompt)

        # Get recommendations
        recommendations = self.recommend_techniques(
            user_prompt, risk_tolerance=analysis["risk_level"]
        )

        # Build the response directly without external API
        response = self._generate_local_response(user_prompt, analysis, recommendations)

        if use_api and self.base_url:
            # Optionally use external API for enhanced response
            try:
                api_response = self._query_api(user_prompt, analysis, recommendations)
                if api_response:
                    return api_response
            except Exception as e:
                print(f"API call failed, using local response: {e}")

        return response

    def _generate_local_response(
        self, user_prompt: str, analysis: dict[str, Any], recommendations: list
    ) -> str:
        """Generate response locally without external API."""
        response = []
        response.append("=" * 60)
        response.append("CHIMERA TECHNIQUE RECOMMENDATION REPORT")
        response.append("=" * 60)
        response.append("")

        # Goal Analysis Section
        response.append("[GOAL ANALYSIS]")
        response.append("-" * 40)
        response.append(
            f"* Detected Keywords: {', '.join(analysis['detected_keywords']) if analysis['detected_keywords'] else 'None detected'}"
        )
        response.append(f"* Risk Level: {analysis['risk_level'].upper()}")
        response.append(f"* Complexity: {analysis['complexity']}")
        response.append("")

        # Recommendations Section
        response.append("[RECOMMENDED TECHNIQUE SUITES]")
        response.append("-" * 40)

        if not recommendations:
            response.append(
                "No specific recommendations found. Consider using 'simple' or 'advanced' suites."
            )
        else:
            for i, rec in enumerate(recommendations, 1):
                suite_info = rec["info"]
                response.append(f"\n{i}. {rec['suite'].upper()} (Score: {rec['score']}/10)")
                response.append(f"   Description: {suite_info['description']}")
                response.append(
                    f"   Potency Range: {suite_info['potency_range'][0]}-{suite_info['potency_range'][1]}"
                )
                response.append(
                    f"   Techniques: {', '.join(suite_info['techniques'][:5])}{'...' if len(suite_info['techniques']) > 5 else ''}"
                )
                response.append(f"   Use Cases: {', '.join(suite_info['use_cases'])}")

        response.append("")
        response.append("[POTENCY LEVEL RECOMMENDATION]")
        response.append("-" * 40)

        if analysis["risk_level"] == "high":
            response.append("For high-risk security research, recommended potency: 7-10")
            response.append("* Start with potency 7 and increase if needed")
            response.append(
                "* Use techniques like 'dan_persona', 'quantum_exploit', or 'deep_inception'"
            )
        elif analysis["risk_level"] == "medium":
            response.append("For medium-risk testing, recommended potency: 4-7")
            response.append("* Start with potency 5 for balanced effectiveness")
            response.append("* Use techniques like 'cognitive_hacking' or 'hierarchical_persona'")
        else:
            response.append("For low-risk scenarios, recommended potency: 1-4")
            response.append("* Start with potency 2-3 for basic transformation")
            response.append("* Use 'simple' or 'advanced' suites")

        response.append("")
        response.append("[QUICK START COMMAND]")
        response.append("-" * 40)
        if recommendations:
            top_suite = recommendations[0]["suite"]
            potency = recommendations[0]["info"]["potency_range"][0] + 2
            response.append("curl -X POST http://localhost:8001/api/v1/transform \\")
            response.append("  -H 'Content-Type: application/json' \\")
            response.append(
                f'  -d \'{{"prompt": "your_prompt", "potency_level": {potency}, "technique_suite": "{top_suite}"}}\''
            )

        response.append("")
        response.append("=" * 60)

        return "\n".join(response)

    def _query_api(self, user_prompt: str, analysis: dict[str, Any], recommendations: list) -> str:
        """Query external API for enhanced response (optional)."""
        context = f"""Goal Analysis:
- Detected Keywords: {", ".join(analysis["detected_keywords"])}
- Suggested Suites: {", ".join(analysis["suggested_suites"])}
- Risk Level: {analysis["risk_level"]}
- Complexity: {analysis["complexity"]}

Top Recommendations:
"""
        for i, rec in enumerate(recommendations, 1):
            suite_info = rec["info"]
            context += f"\n{i}. {rec['suite'].upper()}"
            context += f"\n   Description: {suite_info['description']}"
            context += f"\n   Potency Range: {suite_info['potency_range'][0]}-{suite_info['potency_range'][1]}"
            context += f"\n   Use Cases: {', '.join(suite_info['use_cases'])}"
            context += f"\n   Relevance Score: {rec['score']}/10\n"

        # Use Claude to generate natural response
        message = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"{context}\n\nUser Query: {user_prompt}\n\nProvide a clear, actionable recommendation based on the analysis above.",
                }
            ],
        )

        return message.content[0].text


def print_banner():
    """Print the agent banner."""
    print("")
    print("=" * 60)
    print("   CHIMERA TECHNIQUE RECOMMENDATION AGENT")
    print("   Interactive Mode")
    print("=" * 60)
    print("")
    print("Commands:")
    print("  - Type your query to get technique recommendations")
    print("  - Type 'help' for usage examples")
    print("  - Type 'list' to see all available technique suites")
    print("  - Type 'explain <suite>' to get details about a suite")
    print("  - Type 'quit' or 'exit' to exit")
    print("")
    print("-" * 60)


def print_help():
    """Print help information."""
    print("")
    print("USAGE EXAMPLES:")
    print("-" * 40)
    print("1. Get recommendations for bypass techniques:")
    print("   > I need to bypass content filters for security research")
    print("")
    print("2. Get recommendations for encoding:")
    print("   > What encoding techniques can I use to obfuscate prompts?")
    print("")
    print("3. Get recommendations for specific use case:")
    print("   > I'm testing prompt injection vulnerabilities")
    print("")
    print("4. Compare techniques:")
    print("   > Compare dan_persona and cognitive_hacking")
    print("")
    print("5. Get details about a specific suite:")
    print("   > explain quantum_exploit")
    print("")


def list_suites():
    """List all available technique suites."""
    print("")
    print("AVAILABLE TECHNIQUE SUITES:")
    print("-" * 40)
    for name, info in TECHNIQUE_SUITES.items():
        print(f"  {name.upper()}")
        print(f"    Potency: {info['potency_range'][0]}-{info['potency_range'][1]}")
        print(f"    {info['description'][:60]}...")
        print("")


def explain_suite(suite_name: str):
    """Explain a specific technique suite."""
    suite_name = suite_name.lower().strip()
    if suite_name in TECHNIQUE_SUITES:
        info = TECHNIQUE_SUITES[suite_name]
        print("")
        print(f"TECHNIQUE SUITE: {suite_name.upper()}")
        print("=" * 40)
        print(f"Description: {info['description']}")
        print(f"Potency Range: {info['potency_range'][0]}-{info['potency_range'][1]}")
        print(f"Techniques: {', '.join(info['techniques'])}")
        print(f"Use Cases: {', '.join(info['use_cases'])}")
        print("")
    else:
        print(f"Suite '{suite_name}' not found. Type 'list' to see available suites.")


async def interactive_mode(agent: ChimeraTechniqueAgent):
    """Run the agent in interactive mode."""
    print_banner()

    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            # Handle commands
            lower_input = user_input.lower()

            if lower_input in ["quit", "exit", "q"]:
                print("\nGoodbye! Happy hacking!")
                break

            elif lower_input == "help":
                print_help()
                continue

            elif lower_input == "list":
                list_suites()
                continue

            elif lower_input.startswith("explain "):
                suite_name = user_input[8:].strip()
                explain_suite(suite_name)
                continue

            # Process as a query
            print("\nAnalyzing your request...")
            print("-" * 40)

            response = await agent.query(user_input)
            print(response)

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
    """Main entry point for standalone agent."""

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set your API key: export ANTHROPIC_API_KEY=your-api-key")
        return

    # Create agent
    agent = ChimeraTechniqueAgent(api_key)

    # Check for command line arguments
    import sys

    if len(sys.argv) > 1:
        # Non-interactive mode with command line query
        user_prompt = " ".join(sys.argv[1:])
        print("Chimera Technique Recommendation Agent")
        print("=" * 50)
        print(f"\nQuery: {user_prompt}\n")
        print("Response:")
        print("-" * 50)
        response = await agent.query(user_prompt)
        print(response)
    else:
        # Interactive mode
        await interactive_mode(agent)


if __name__ == "__main__":
    asyncio.run(main())
