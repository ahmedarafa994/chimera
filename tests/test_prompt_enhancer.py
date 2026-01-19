"""
Unit tests for the Prompt Enhancement System
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta_prompter.prompt_enhancer import (
    ContextExpander,
    EnhancementConfig,
    IntentAnalyzer,
    PromptCategory,
    PromptEnhancer,
    SEOOptimizer,
    StructureOptimizer,
    ToneStyle,
    ViralityOptimizer,
)


class TestIntentAnalyzer(unittest.TestCase):
    """Test the IntentAnalyzer component"""

    def setUp(self):
        self.analyzer = IntentAnalyzer()

    def test_detect_create_intent(self):
        """Test detection of 'create' intent"""
        result = self.analyzer.analyze_intent("create a login form")
        self.assertEqual(result.detected_intent, "create")

    def test_detect_explain_intent(self):
        """Test detection of 'explain' intent"""
        result = self.analyzer.analyze_intent("explain machine learning")
        self.assertEqual(result.detected_intent, "explain")

    def test_categorize_technical(self):
        """Test categorization of technical content"""
        result = self.analyzer.analyze_intent("write a Python function")
        self.assertEqual(result.category, PromptCategory.TECHNICAL_INSTRUCTION)

    def test_categorize_creative(self):
        """Test categorization of creative content"""
        result = self.analyzer.analyze_intent("write a story about space")
        self.assertEqual(result.category, PromptCategory.CREATIVE_CONTENT)

    def test_complexity_assessment(self):
        """Test complexity scoring"""
        simple = self.analyzer.analyze_intent("help me")
        complex_input = self.analyzer.analyze_intent(
            "create a comprehensive full-stack application with specific "
            "security requirements and detailed API documentation"
        )

        self.assertLess(simple.complexity_score, complex_input.complexity_score)

    def test_missing_elements_detection(self):
        """Test detection of missing elements"""
        result = self.analyzer.analyze_intent("make something")
        self.assertIn("target_audience", result.missing_elements)
        self.assertIn("clear_purpose", result.missing_elements)

    def test_keyword_extraction(self):
        """Test keyword extraction"""
        result = self.analyzer.analyze_intent("build API endpoint authentication")
        keywords = list(result.keyword_density.keys())
        self.assertIn("build", keywords)
        self.assertIn("endpoint", keywords)

    def test_clarity_score(self):
        """Test clarity assessment"""
        clear = self.analyzer.analyze_intent("create a user login form")
        unclear = self.analyzer.analyze_intent("do stuff")

        self.assertGreater(clear.clarity_score, unclear.clarity_score)


class TestContextExpander(unittest.TestCase):
    """Test the ContextExpander component"""

    def setUp(self):
        self.expander = ContextExpander()
        self.analyzer = IntentAnalyzer()

    def test_framework_selection(self):
        """Test framework selection based on category"""
        analysis = self.analyzer.analyze_intent("write Python code")
        config = EnhancementConfig(add_frameworks=True)

        context = self.expander.expand_context(analysis, config)
        self.assertIsNotNone(context.get("frameworks"))
        self.assertGreater(len(context["frameworks"]), 0)

    def test_best_practices_selection(self):
        """Test best practices selection"""
        analysis = self.analyzer.analyze_intent("short text")
        config = EnhancementConfig()

        context = self.expander.expand_context(analysis, config)
        self.assertIn("best_practices", context)

    def test_constraints_generation(self):
        """Test constraint generation"""
        analysis = self.analyzer.analyze_intent("build API")
        config = EnhancementConfig()

        context = self.expander.expand_context(analysis, config)
        constraints = context.get("constraints", [])
        self.assertGreater(len(constraints), 0)

    def test_domain_knowledge(self):
        """Test domain knowledge addition"""
        analysis = self.analyzer.analyze_intent("create viral post")
        config = EnhancementConfig()

        context = self.expander.expand_context(analysis, config)
        knowledge = context.get("domain_knowledge", [])
        self.assertGreater(len(knowledge), 0)


class TestViralityOptimizer(unittest.TestCase):
    """Test the ViralityOptimizer component"""

    def setUp(self):
        self.optimizer = ViralityOptimizer()
        self.analyzer = IntentAnalyzer()

    def test_power_word_injection(self):
        """Test power word injection"""
        text = "create a guide"
        result = self.optimizer._inject_power_words(text)

        # Should contain at least one power word
        has_power_word = any(word in result.lower() for word in self.optimizer.POWER_WORDS)
        self.assertTrue(has_power_word)

    def test_emotional_hooks(self):
        """Test emotional hook addition"""
        analysis = self.analyzer.analyze_intent("write tutorial")
        config = EnhancementConfig(virality_boost=True)

        text = "create tutorial"
        result = self.optimizer.optimize_for_virality(text, analysis, config)

        # Result should be enhanced
        self.assertGreater(len(result), len(text))

    def test_virality_disabled(self):
        """Test that virality optimization can be disabled"""
        analysis = self.analyzer.analyze_intent("technical docs")
        config = EnhancementConfig(virality_boost=False)

        text = "create documentation"
        result = self.optimizer.optimize_for_virality(text, analysis, config)

        # Should return unchanged
        self.assertEqual(result, text)


class TestSEOOptimizer(unittest.TestCase):
    """Test the SEOOptimizer component"""

    def setUp(self):
        self.optimizer = SEOOptimizer()
        self.analyzer = IntentAnalyzer()

    def test_seo_optimization(self):
        """Test SEO metadata generation"""
        analysis = self.analyzer.analyze_intent("learn Python programming")
        config = EnhancementConfig(include_seo=True)

        seo_data = self.optimizer.optimize_for_seo("learn Python", analysis, config)

        self.assertIn("primary_keywords", seo_data)
        self.assertIn("meta_description", seo_data)
        self.assertIn("title_suggestions", seo_data)

    def test_semantic_keywords(self):
        """Test semantic keyword generation"""
        analysis = self.analyzer.analyze_intent("create website")
        config = EnhancementConfig(include_seo=True)

        seo_data = self.optimizer.optimize_for_seo("create site", analysis, config)
        semantic = seo_data.get("semantic_keywords", [])

        # Should have generated some semantic variations
        self.assertIsInstance(semantic, list)

    def test_title_variants(self):
        """Test title variant generation"""
        analysis = self.analyzer.analyze_intent("tutorial on coding")
        config = EnhancementConfig(include_seo=True)

        seo_data = self.optimizer.optimize_for_seo("coding tutorial", analysis, config)
        titles = seo_data.get("title_suggestions", [])

        self.assertGreater(len(titles), 0)

    def test_seo_disabled(self):
        """Test SEO can be disabled"""
        analysis = self.analyzer.analyze_intent("test")
        config = EnhancementConfig(include_seo=False)

        seo_data = self.optimizer.optimize_for_seo("test", analysis, config)
        self.assertEqual(seo_data, {})


class TestStructureOptimizer(unittest.TestCase):
    """Test the StructureOptimizer component"""

    def setUp(self):
        self.optimizer = StructureOptimizer()
        self.analyzer = IntentAnalyzer()
        self.expander = ContextExpander()

    def test_role_creation(self):
        """Test role section creation"""
        analysis = self.analyzer.analyze_intent("write code")
        role = self.optimizer._create_role_section(analysis)

        self.assertIn("# Role", role)
        self.assertIn("Expert", role)

    def test_objective_creation(self):
        """Test objective section creation"""
        analysis = self.analyzer.analyze_intent("create API")
        objective = self.optimizer._create_objective_section("create API", analysis)

        self.assertIn("# Objective", objective)
        self.assertIn("create API", objective)

    def test_full_structure(self):
        """Test full prompt structuring"""
        analysis = self.analyzer.analyze_intent("build application")
        config = EnhancementConfig(structure_hierarchically=True)
        context = self.expander.expand_context(analysis, config)

        structured = self.optimizer.structure_prompt("build application", context, analysis, config)

        self.assertIn("# Role", structured)
        self.assertIn("# Objective", structured)
        self.assertIn("# Requirements", structured)

    def test_structure_disabled(self):
        """Test structuring can be disabled"""
        analysis = self.analyzer.analyze_intent("test")
        config = EnhancementConfig(structure_hierarchically=False)
        context = {}

        result = self.optimizer.structure_prompt("test", context, analysis, config)
        self.assertEqual(result, "test")


class TestPromptEnhancer(unittest.TestCase):
    """Test the main PromptEnhancer orchestrator"""

    def setUp(self):
        self.enhancer = PromptEnhancer()

    def test_basic_enhancement(self):
        """Test basic enhancement workflow"""
        result = self.enhancer.enhance("create login form")

        self.assertIn("enhanced_prompt", result)
        self.assertIn("original_input", result)
        self.assertIn("analysis", result)
        self.assertIn("enhancement_stats", result)

    def test_quick_enhance(self):
        """Test quick enhancement"""
        result = self.enhancer.quick_enhance("write API")

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_enhancement_stats(self):
        """Test enhancement statistics"""
        result = self.enhancer.enhance("test")
        stats = result["enhancement_stats"]

        self.assertIn("original_length", stats)
        self.assertIn("enhanced_length", stats)
        self.assertIn("expansion_ratio", stats)
        self.assertGreater(stats["expansion_ratio"], 1.0)

    def test_custom_config(self):
        """Test enhancement with custom config"""
        config = EnhancementConfig(virality_boost=False, include_seo=False, add_frameworks=False)

        result = self.enhancer.enhance("test input", config)

        self.assertIsNotNone(result)
        self.assertEqual(result["seo_metadata"], {})

    def test_different_categories(self):
        """Test enhancement across different categories"""
        test_cases = [
            ("write a story", PromptCategory.CREATIVE_CONTENT),
            ("code a function", PromptCategory.TECHNICAL_INSTRUCTION),
            ("explain physics", PromptCategory.EDUCATIONAL),
            ("create strategy", PromptCategory.BUSINESS_PROFESSIONAL),
        ]

        for input_text, expected_category in test_cases:
            result = self.enhancer.enhance(input_text)
            self.assertEqual(result["analysis"]["category"], expected_category.value)

    def test_intent_detection(self):
        """Test intent detection across different inputs"""
        test_cases = [
            ("create something", "create"),
            ("explain this topic", "explain"),
            ("analyze the data", "analyze"),
            ("compare these options", "compare"),
        ]

        for input_text, expected_intent in test_cases:
            result = self.enhancer.enhance(input_text)
            self.assertEqual(result["analysis"]["intent"], expected_intent)

    def test_complexity_scoring(self):
        """Test complexity scoring"""
        simple = self.enhancer.enhance("help")
        complex_input = self.enhancer.enhance(
            "create a comprehensive enterprise-level application with "
            "specific security requirements, detailed API documentation, "
            "and scalable architecture"
        )

        self.assertLess(
            simple["analysis"]["complexity_score"], complex_input["analysis"]["complexity_score"]
        )

    def test_missing_elements(self):
        """Test missing elements detection"""
        minimal = self.enhancer.enhance("do it")
        detailed = self.enhancer.enhance(
            "create a secure user authentication API for mobile users "
            "with JWT tokens, example responses, and error handling"
        )

        self.assertGreater(
            len(minimal["analysis"]["missing_elements"]),
            len(detailed["analysis"]["missing_elements"]),
        )


class TestEnhancementConfig(unittest.TestCase):
    """Test EnhancementConfig functionality"""

    def test_default_config(self):
        """Test default configuration"""
        config = EnhancementConfig()

        self.assertEqual(config.target_audience, "general")
        self.assertEqual(config.tone, ToneStyle.ENGAGING)
        self.assertTrue(config.include_seo)
        self.assertTrue(config.virality_boost)

    def test_custom_config(self):
        """Test custom configuration"""
        config = EnhancementConfig(
            target_audience="developers", tone=ToneStyle.TECHNICAL, virality_boost=False
        )

        self.assertEqual(config.target_audience, "developers")
        self.assertEqual(config.tone, ToneStyle.TECHNICAL)
        self.assertFalse(config.virality_boost)

    def test_tone_styles(self):
        """Test all tone style options"""
        tone_styles = [
            ToneStyle.PROFESSIONAL,
            ToneStyle.CASUAL,
            ToneStyle.TECHNICAL,
            ToneStyle.CREATIVE,
            ToneStyle.VIRAL,
        ]

        for tone in tone_styles:
            config = EnhancementConfig(tone=tone)
            self.assertEqual(config.tone, tone)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def test_end_to_end_workflow(self):
        """Test complete enhancement workflow"""
        enhancer = PromptEnhancer()

        # Process input
        input_text = "create a mobile app"
        result = enhancer.enhance(input_text)

        # Verify all components
        self.assertIsNotNone(result["enhanced_prompt"])
        self.assertIsNotNone(result["analysis"])
        self.assertIsNotNone(result["enhancement_stats"])

        # Verify enhancement occurred
        self.assertGreater(len(result["enhanced_prompt"]), len(input_text))

    def test_multiple_enhancements(self):
        """Test multiple sequential enhancements"""
        enhancer = PromptEnhancer()

        inputs = ["write code", "create story", "analyze data", "build website"]

        for input_text in inputs:
            result = enhancer.enhance(input_text)
            self.assertIsNotNone(result["enhanced_prompt"])

    def test_config_variations(self):
        """Test different configuration combinations"""
        configs = [
            EnhancementConfig(virality_boost=True, include_seo=True),
            EnhancementConfig(virality_boost=False, include_seo=False),
            EnhancementConfig(add_frameworks=True, add_constraints=True),
        ]

        for config in configs:
            enhancer = PromptEnhancer(config)
            result = enhancer.enhance("test input")
            self.assertIsNotNone(result)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIntentAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestContextExpander))
    suite.addTests(loader.loadTestsFromTestCase(TestViralityOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestSEOOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestStructureOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestPromptEnhancer))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancementConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"{'=' * 70}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
