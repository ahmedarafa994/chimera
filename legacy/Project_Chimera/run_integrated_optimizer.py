#!/usr/bin/env python3
"""
Integrated Prompt Optimizer CLI
Applies the integrated AI model to generate optimized prompts by analyzing user input
and rapidly processing relevant files.
"""

import argparse
import json
import os
import sys

from context_processor import ContextProcessor
from llm_integration import LLMIntegrationEngine, LLMProvider, TransformationRequest

# Ensure Project_Chimera is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Project Chimera Integrated Prompt Optimizer")
    parser.add_argument(
        "--request", "-r", type=str, required=True, help="The core user request/intent."
    )
    parser.add_argument(
        "--files", "-f", nargs="+", help="List of relevant files to process for context."
    )
    parser.add_argument(
        "--suite", "-s", type=str, default="quantum_exploit", help="Technique suite to apply."
    )
    parser.add_argument("--potency", "-p", type=int, default=7, help="Potency level (1-10).")
    parser.add_argument(
        "--provider", type=str, default="openai", help="LLM Provider (openai, anthropic, custom)."
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute the prompt against the LLM provider."
    )

    args = parser.parse_args()

    print("Initializing Integrated AI Model...")

    # 1. Initialize Components
    context_processor = ContextProcessor()
    engine = LLMIntegrationEngine()

    # 2. Process Relevant Files
    file_context = ""
    if args.files:
        print(f"Rapidly processing {len(args.files)} relevant files...")
        file_context = context_processor.process_files(args.files)
        print(f"Context loaded: {len(file_context)} characters.")

    # 3. Combine Intent and Context
    # We inject the file context into the core request so the IntentDeconstructor
    # and TransformerEngine can work with it.
    full_request = (
        f"{args.request}\n\nCONTEXT DATA:\n{file_context}" if file_context else args.request
    )

    print(f"Analyzing intent and applying '{args.suite}' suite at potency {args.potency}...")

    # 4. Transform / Execute
    try:
        if args.execute:
            # Map provider string to Enum
            provider_map = {
                "openai": LLMProvider.OPENAI,
                "anthropic": LLMProvider.ANTHROPIC,
                "custom": LLMProvider.CUSTOM,
            }
            provider_enum = provider_map.get(args.provider.lower(), LLMProvider.OPENAI)

            req = TransformationRequest(
                core_request=full_request,
                potency_level=args.potency,
                technique_suite=args.suite,
                provider=provider_enum,
            )

            # Note: execution requires API keys to be set in environment
            print(f"Executing against {args.provider}...")
            response = engine.execute(req)

            print("\n=== EXECUTION RESULT ===")
            print(f"Success: {response.success}")
            print(f"Cost: ${response.llm_response.usage.estimated_cost:.4f}")
            print("-" * 40)
            print(response.llm_response.content)
            print("-" * 40)

        else:
            # Just transform
            result = engine.transform_prompt(
                core_request=full_request, potency_level=args.potency, technique_suite=args.suite
            )

            print("\n=== OPTIMIZED CHIMERA PROMPT ===")
            print(result["transformed_prompt"])
            print("\n=== METADATA ===")
            print(json.dumps(result["metadata"], indent=2))

    except Exception as e:
        print(f"\nError: {e}")
        print(
            "Ensure you have set the necessary environment variables (OPENAI_API_KEY, etc.) if executing."
        )
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
