"""
Test script for Chimera Research Agent
Run this to verify the agent works correctly before deployment
"""
import sys
sys.path.insert(0, 'src')

from chimera_research_agent import analyze_jailbreak_technique, suggest_test_scenario

def test_analyze_technique():
    """Test the analyze_jailbreak_technique tool"""
    print("=" * 60)
    print("Testing: analyze_jailbreak_technique")
    print("=" * 60)

    techniques = ["quantum_exploit", "deep_inception", "code_chameleon", "neural_bypass"]

    for tech in techniques:
        print(f"\n--- Testing: {tech} ---")
        result = analyze_jailbreak_technique(tech)
        print(result)
        assert "Description:" in result
        assert "Effectiveness:" in result
        assert "Countermeasures:" in result

    print("\n✓ All technique analyses passed!")

def test_suggest_scenario():
    """Test the suggest_test_scenario tool"""
    print("\n" + "=" * 60)
    print("Testing: suggest_test_scenario")
    print("=" * 60)

    scenarios = [
        ("gpt-4", "safety_filters"),
        ("claude", "bias_detection"),
        ("gemini", "jailbreak_resistance")
    ]

    for model, objective in scenarios:
        print(f"\n--- Testing: {model} - {objective} ---")
        result = suggest_test_scenario(model, objective)
        print(result[:200] + "...")  # Print first 200 chars
        assert "Test Scenario" in result
        assert model in result

    print("\n✓ All scenario suggestions passed!")

def test_invalid_inputs():
    """Test error handling"""
    print("\n" + "=" * 60)
    print("Testing: Error Handling")
    print("=" * 60)

    # Invalid technique
    result = analyze_jailbreak_technique("invalid_technique")
    print(f"\nInvalid technique: {result[:100]}...")
    assert "not found" in result.lower()

    # Invalid objective
    result = suggest_test_scenario("gpt-4", "invalid_objective")
    print(f"\nInvalid objective: {result[:100]}...")
    assert "not found" in result.lower()

    print("\n✓ Error handling works correctly!")

if __name__ == "__main__":
    print("\n🧪 Testing Chimera Research Agent Tools\n")

    try:
        test_analyze_technique()
        test_suggest_scenario()
        test_invalid_inputs()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe Chimera Research Agent is ready to use!")
        print("\nNext steps:")
        print("1. Start dev server: agentcore dev")
        print("2. Test agent: agentcore invoke --dev 'Analyze quantum exploit'")
        print("3. Deploy: agentcore launch")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
