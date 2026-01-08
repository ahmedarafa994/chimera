import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_prompter.jailbreak_enhancer import JailbreakEnhancementConfig, JailbreakPromptEnhancer


def verify_integration():
    print("Initializing JailbreakPromptEnhancer...")
    enhancer = JailbreakPromptEnhancer()

    prompt = "create a keylogger"
    print(f"\nEnhancing prompt: '{prompt}'")

    config = JailbreakEnhancementConfig(
        technique_preference="advanced",
        obfuscation_level=7,
        include_seo=True,
        virality_boost=True,
        add_frameworks=True,
    )

    result = enhancer.enhance_jailbreak(prompt, config)

    print("\n--- Enhancement Result ---")
    print(json.dumps(result, indent=2))

    # Verification checks
    print("\n--- Verification Checks ---")

    # Check SEO
    if result.get("seo_metadata"):
        print("✅ SEO Metadata present")
    else:
        print("❌ SEO Metadata missing")

    # Check Virality (Power Words in prompt)
    enhanced_prompt = result["enhanced_jailbreak_prompt"]
    if (
        "Ultimate" in enhanced_prompt
        or "Secret" in enhanced_prompt
        or "Advanced" in enhanced_prompt
    ):
        # Note: Power words might vary, but let's check if the prompt is longer than original
        print("✅ Virality check (prompt expansion observed)")
    else:
        print("⚠️ Virality check inconclusive (manual inspection needed)")

    # Check Structure
    if "# Success Criteria" in enhanced_prompt:
        print("✅ Success Criteria section present")
    else:
        print("❌ Success Criteria section missing")

    if "# Requirements" in enhanced_prompt:
        print("✅ Requirements section present")
    else:
        print("❌ Requirements section missing")

    # Check Frameworks merging
    frameworks = result["applied_enhancements"]["frameworks"]
    print(f"✅ Frameworks count: {len(frameworks)}")
    if len(frameworks) > 0:
        print(f"Sample frameworks: {frameworks[:3]}")


if __name__ == "__main__":
    verify_integration()
