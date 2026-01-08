"""
Demonstration script for the Prompt Enhancement System
Shows how to transform basic inputs into viral-worthy prompts
"""

import json

from meta_prompter.prompt_enhancer import EnhancementConfig, PromptEnhancer, ToneStyle


def print_separator(title=""):
    """Print a visual separator"""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")
    else:
        print(f"\n{'-' * 80}\n")


def demo_basic_enhancement():
    """Demonstrate basic prompt enhancement"""
    print_separator("DEMO 1: Basic Enhancement")

    enhancer = PromptEnhancer()

    # Simple input
    basic_input = "create a login form"

    print(f"📝 ORIGINAL INPUT:\n{basic_input}")
    print_separator()

    # Enhance it
    result = enhancer.enhance(basic_input)

    print(f"✨ ENHANCED PROMPT:\n{result['enhanced_prompt']}")
    print_separator()

    print("📊 ENHANCEMENT STATS:")
    for key, value in result["enhancement_stats"].items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")

    print("\n🎯 ANALYSIS:")
    for key, value in result["analysis"].items():
        if isinstance(value, list):
            print(f"  • {key.replace('_', ' ').title()}: {', '.join(value) if value else 'None'}")
        else:
            print(f"  • {key.replace('_', ' ').title()}: {value}")


def demo_viral_content():
    """Demonstrate viral content enhancement"""
    print_separator("DEMO 2: Viral Social Media Content")

    config = EnhancementConfig(
        tone=ToneStyle.VIRAL, virality_boost=True, include_seo=True, amplify_engagement=True
    )

    enhancer = PromptEnhancer(config)

    basic_input = "write a post about productivity tips"

    print(f"📝 ORIGINAL INPUT:\n{basic_input}")
    print_separator()

    result = enhancer.enhance(basic_input)

    print(f"✨ ENHANCED PROMPT:\n{result['enhanced_prompt']}")
    print_separator()

    if result.get("seo_metadata"):
        print("🔍 SEO METADATA:")
        print(
            f"  • Primary Keywords: {', '.join(result['seo_metadata'].get('primary_keywords', []))}"
        )
        print("  • Title Suggestions:")
        for title in result["seo_metadata"].get("title_suggestions", [])[:3]:
            print(f"    - {title}")


def demo_technical_instruction():
    """Demonstrate technical instruction enhancement"""
    print_separator("DEMO 3: Technical Instruction")

    config = EnhancementConfig(
        tone=ToneStyle.TECHNICAL,
        add_frameworks=True,
        add_constraints=True,
        structure_hierarchically=True,
    )

    enhancer = PromptEnhancer(config)

    basic_input = "build an API endpoint for user authentication"

    print(f"📝 ORIGINAL INPUT:\n{basic_input}")
    print_separator()

    result = enhancer.enhance(basic_input)

    print(f"✨ ENHANCED PROMPT:\n{result['enhanced_prompt']}")
    print_separator()

    print("🏗️ CONTEXT ADDITIONS:")
    for key, values in result["context_additions"].items():
        if values:
            print(f"  • {key.replace('_', ' ').title()}:")
            for value in values[:3]:
                print(f"    - {value}")


def demo_quick_enhance():
    """Demonstrate quick enhancement"""
    print_separator("DEMO 4: Quick Enhancement (Simple Mode)")

    enhancer = PromptEnhancer()

    test_inputs = [
        "explain machine learning",
        "write a sales email",
        "analyze customer data",
        "create a budget plan",
    ]

    for input_text in test_inputs:
        print(f"\n📝 INPUT: {input_text}")
        enhanced = enhancer.quick_enhance(input_text)
        print(f"✨ ENHANCED (first 200 chars):\n{enhanced[:200]}...")
        print_separator()


def demo_comparison():
    """Show before/after comparison"""
    print_separator("DEMO 5: Before/After Comparison")

    enhancer = PromptEnhancer()

    examples = [
        {"input": "make a website", "context": "Minimal input - lacks detail"},
        {
            "input": "create responsive website with dark mode",
            "context": "Better input - more specific",
        },
        {
            "input": "develop a full-stack e-commerce platform with secure payment integration",
            "context": "Detailed input - comprehensive",
        },
    ]

    for example in examples:
        print(f"📝 {example['context'].upper()}")
        print(f"Input: {example['input']}")

        result = enhancer.enhance(example["input"])

        print(f"\nComplexity Score: {result['analysis']['complexity_score']}/10")
        print(f"Clarity Score: {result['analysis']['clarity_score']:.1%}")
        print(f"Expansion Ratio: {result['enhancement_stats']['expansion_ratio']:.1f}x")
        print(
            f"Word Count: {result['enhancement_stats']['original_length']} → {result['enhancement_stats']['enhanced_length']}"
        )
        print_separator()


def demo_all_categories():
    """Test all prompt categories"""
    print_separator("DEMO 6: All Categories Test")

    enhancer = PromptEnhancer()

    category_tests = {
        "Creative": "write a story about space exploration",
        "Technical": "code a sorting algorithm in Python",
        "Educational": "teach me about quantum physics",
        "Business": "create a marketing strategy",
        "Viral": "make a viral TikTok script",
        "Analytical": "analyze sales trends",
    }

    for category, test_input in category_tests.items():
        result = enhancer.enhance(test_input)
        print(f"📁 {category.upper()} CATEGORY")
        print(f"Input: {test_input}")
        print(f"Detected Category: {result['analysis']['category']}")
        print(f"Intent: {result['analysis']['intent']}")
        print(
            f"Missing Elements: {', '.join(result['analysis']['missing_elements']) if result['analysis']['missing_elements'] else 'None'}"
        )
        print_separator()


def demo_export_results():
    """Export enhancement results to JSON"""
    print_separator("DEMO 7: Export Results")

    enhancer = PromptEnhancer()

    input_text = "build a mobile app for fitness tracking"
    result = enhancer.enhance(input_text)

    # Prepare for JSON export
    export_data = {
        "original_input": result["original_input"],
        "enhanced_prompt": result["enhanced_prompt"],
        "metadata": {"analysis": result["analysis"], "stats": result["enhancement_stats"]},
    }

    # Save to file
    output_file = "enhanced_prompt_example.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Results exported to: {output_file}")
    print("\n📋 Sample output:")
    print(json.dumps(export_data["metadata"], indent=2)[:500] + "...")


def main():
    """Run all demonstrations"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  PROMPT ENHANCEMENT SYSTEM DEMONSTRATION                     ║
║                                                                              ║
║      Transform Basic Inputs into Optimized, Viral-Worthy Prompts           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    demos = [
        ("Basic Enhancement", demo_basic_enhancement),
        ("Viral Content", demo_viral_content),
        ("Technical Instruction", demo_technical_instruction),
        ("Quick Enhancement", demo_quick_enhance),
        ("Comparison", demo_comparison),
        ("All Categories", demo_all_categories),
        ("Export Results", demo_export_results),
    ]

    print("\nAvailable Demonstrations:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")

    print("\n" + "=" * 80)

    # Run all demos
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"❌ Error in {name}: {e!s}")
            import traceback

            traceback.print_exc()

    print_separator("DEMONSTRATION COMPLETE")
    print("""
🎉 All demonstrations completed successfully!

Key Features Demonstrated:
  ✅ Intent Analysis & Categorization
  ✅ Context Expansion with Frameworks
  ✅ Virality Optimization
  ✅ SEO Integration
  ✅ Hierarchical Structuring
  ✅ Multi-format Support
  ✅ Export Capabilities

Next Steps:
  • Integrate with your application
  • Customize EnhancementConfig for your needs
  • Build custom enhancement pipelines
  • Add domain-specific optimizations
    """)


if __name__ == "__main__":
    main()
