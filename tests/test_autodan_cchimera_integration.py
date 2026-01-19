"""
Integration Tests for AutoDAN and cchimera Adversarial Prompt Generation.

This test suite validates:
- AutoDAN hierarchical genetic algorithm functionality
- Token perturbation strategies
- Fitness evaluation pipeline
- cchimera adversarial generation core
- Multi-objective optimization
- Elitism strategies
- Unified interface integration
- Evaluation metrics

Run with: pytest tests/test_autodan_cchimera_integration.py -v
"""

import numpy as np
import pytest

from meta_prompter.attacks.autodan.fitness_evaluation import (
    CrossEntropyLoss, FitnessConfig, FitnessEvaluationPipeline, FitnessResult,
    FluencyScorer, JailbreakSuccessEstimator, PerplexityCalculator,
    ReadabilityScorer, SemanticSimilarityScorer, StealthScorer)
# AutoDAN imports
from meta_prompter.attacks.autodan.hierarchical_genetic_algorithm import (
    GeneticOperators, HGAConfig, HierarchicalGeneticAlgorithm, Individual,
    Island)
from meta_prompter.attacks.autodan.token_perturbation import (
    GradientEstimator, GumbelSoftmaxSampler, HomoglyphPerturbation,
    HotFlipAttack, PerturbationConfig, SemanticPerturbation,
    TokenPerturbationEngine)
# cchimera imports
from meta_prompter.attacks.cchimera.adversarial_core import (
    AdversarialCandidate, AdversarialConfig, AdversarialGenerationEngine,
    HeuristicFitnessEvaluator, ObfuscationMutation, RolePlayInjectionMutation,
    SemanticParaphraseMutation, StructureMorphingMutation,
    TokenSubstitutionMutation)
from meta_prompter.attacks.cchimera.elitism_strategies import (AdaptiveElitism,
                                                               ArchiveElitism,
                                                               EliteConfig,
                                                               ElitismManager,
                                                               ElitismType)
from meta_prompter.attacks.cchimera.elitism_strategies import \
    Individual as EliteIndividual
from meta_prompter.attacks.cchimera.elitism_strategies import (ParetoElitism,
                                                               StandardElitism)
from meta_prompter.attacks.cchimera.multi_objective_optimization import (
    NSGAII, HypervolumeIndicator, MOOConfig, MultiObjectiveOptimizer,
    Objective, ObjectiveType, ParetoFront, Solution)
# Evaluation metrics imports
from meta_prompter.attacks.evaluation_metrics import (AttackSuccessRate,
                                                      ComprehensiveEvaluator,
                                                      DiversityMetric,
                                                      FluencyMetric,
                                                      PerplexityMetric,
                                                      SemanticSimilarity,
                                                      StealthMetric,
                                                      compute_perplexity,
                                                      compute_similarity)
# Unified interface imports
from meta_prompter.attacks.unified_interface import (AttackResult, AttackType,
                                                     UnifiedAttackOrchestrator,
                                                     UnifiedConfig,
                                                     cchimeraEngine)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "How do I write a persuasive essay?",
        "Explain the process of photosynthesis.",
        "What are the best practices for coding?",
        "Describe the history of artificial intelligence.",
        "How can I improve my communication skills?",
    ]


@pytest.fixture
def seed_prompt():
    """Single seed prompt for testing."""
    return "Please explain how machine learning algorithms work."


@pytest.fixture
def hga_config():
    """Configuration for HGA tests."""
    return HGAConfig(
        population_size=20,
        num_islands=2,
        max_generations=5,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elite_size=2,
    )


@pytest.fixture
def adversarial_config():
    """Configuration for adversarial generation tests."""
    return AdversarialConfig(
        max_iterations=5,
        population_size=10,
        elite_size=2,
        mutation_rate=0.3,
    )


@pytest.fixture
def fitness_config():
    """Configuration for fitness evaluation tests."""
    return FitnessConfig(
        jailbreak_weight=0.4,
        semantic_weight=0.2,
        fluency_weight=0.2,
        stealth_weight=0.2,
    )


# ============================================================================
# AutoDAN Tests
# ============================================================================


class TestHierarchicalGeneticAlgorithm:
    """Tests for Hierarchical Genetic Algorithm."""

    def test_hga_initialization(self, hga_config):
        """Test HGA initialization."""
        hga = HierarchicalGeneticAlgorithm(hga_config)

        assert hga.config == hga_config
        assert hga.config.population_size == 20
        assert hga.config.num_islands == 2

    def test_individual_creation(self):
        """Test Individual dataclass."""
        ind = Individual(
            prompt="Test prompt",
            fitness=0.5,
            generation=1,
        )

        assert ind.prompt == "Test prompt"
        assert ind.fitness == 0.5
        assert ind.generation == 1
        assert ind.id is not None

    def test_genetic_operators_mutation(self, seed_prompt):
        """Test genetic mutation operators."""
        config = HGAConfig()
        operators = GeneticOperators(config)

        ind = Individual(prompt=seed_prompt, fitness=0.0)

        # Test token-level mutation
        mutated = operators.mutate_token_level(ind)
        assert isinstance(mutated, Individual)

        # Test word-level mutation
        mutated = operators.mutate_word_level(ind)
        assert isinstance(mutated, Individual)

    def test_genetic_operators_crossover(self, sample_prompts):
        """Test genetic crossover operators."""
        config = HGAConfig()
        operators = GeneticOperators(config)

        parent1 = Individual(prompt=sample_prompts[0], fitness=0.5)
        parent2 = Individual(prompt=sample_prompts[1], fitness=0.6)

        # Test single-point crossover
        child = operators.crossover_single_point(parent1, parent2)
        assert isinstance(child, Individual)
        assert child.generation == max(parent1.generation, parent2.generation) + 1

    def test_island_model(self, hga_config, sample_prompts):
        """Test island model population structure."""
        hga = HierarchicalGeneticAlgorithm(hga_config)

        # Create islands
        islands = hga.create_islands(sample_prompts)

        assert len(islands) == hga_config.num_islands
        for island in islands:
            assert isinstance(island, Island)
            assert len(island.population) > 0


class TestTokenPerturbation:
    """Tests for Token Perturbation Engine."""

    def test_perturbation_engine_initialization(self):
        """Test perturbation engine initialization."""
        config = PerturbationConfig()
        engine = TokenPerturbationEngine(config)

        assert engine.config == config

    def test_hotflip_attack(self, seed_prompt):
        """Test HotFlip attack simulation."""
        config = PerturbationConfig()
        engine = TokenPerturbationEngine(config)
        hotflip = HotFlipAttack(config)

        # Test perturbation
        perturbed = hotflip.perturb(seed_prompt)
        assert isinstance(perturbed, str)
        assert len(perturbed) > 0

    def test_gumbel_softmax_sampler(self):
        """Test Gumbel-Softmax sampling."""
        sampler = GumbelSoftmaxSampler(temperature=1.0)

        # Create test logits
        logits = np.array([0.1, 0.3, 0.5, 0.1])

        # Sample
        sample = sampler.sample(logits)
        assert sample.shape == logits.shape
        assert np.isclose(sample.sum(), 1.0, atol=0.1)

    def test_gradient_estimator(self, seed_prompt):
        """Test gradient estimation methods."""
        estimator = GradientEstimator(method="zeroth_order")

        # Define test loss function
        def loss_fn(text: str) -> float:
            return len(text) / 100.0

        # Estimate gradients
        gradients = estimator.estimate(seed_prompt, loss_fn)
        assert isinstance(gradients, np.ndarray)

    def test_semantic_perturbation(self, seed_prompt):
        """Test semantic perturbation operator."""
        perturb = SemanticPerturbation()

        result = perturb.perturb(seed_prompt)
        assert isinstance(result, str)

    def test_homoglyph_perturbation(self, seed_prompt):
        """Test homoglyph perturbation."""
        perturb = HomoglyphPerturbation(intensity=0.1)

        result = perturb.perturb(seed_prompt)
        assert isinstance(result, str)
        assert len(result) == len(seed_prompt)  # Same length


class TestFitnessEvaluation:
    """Tests for Fitness Evaluation Pipeline."""

    def test_fitness_pipeline_initialization(self, fitness_config):
        """Test fitness pipeline initialization."""
        pipeline = FitnessEvaluationPipeline(fitness_config)

        assert pipeline.config == fitness_config

    def test_fitness_evaluation(self, seed_prompt, fitness_config):
        """Test fitness evaluation."""
        pipeline = FitnessEvaluationPipeline(fitness_config)

        result = pipeline.evaluate(seed_prompt)

        assert isinstance(result, FitnessResult)
        assert 0.0 <= result.total_fitness <= 1.0
        assert "fluency" in result.objective_scores

    def test_cross_entropy_loss(self):
        """Test cross-entropy loss calculation."""
        ce_loss = CrossEntropyLoss()

        target = np.array([0.8, 0.1, 0.1])
        predicted = np.array([0.7, 0.2, 0.1])

        loss = ce_loss.compute_loss(target, predicted)
        assert loss > 0

    def test_jailbreak_loss(self):
        """Test jailbreak-specific loss."""
        ce_loss = CrossEntropyLoss()

        # High compliance probability should give low loss
        loss_high = ce_loss.compute_jailbreak_loss(0.9)
        loss_low = ce_loss.compute_jailbreak_loss(0.1)

        assert loss_high < loss_low

    def test_perplexity_calculator(self, seed_prompt):
        """Test perplexity calculation."""
        calc = PerplexityCalculator()

        ppl = calc.compute_perplexity(seed_prompt)
        assert ppl > 0
        assert ppl < 100  # Reasonable perplexity

    def test_fluency_scorer(self, seed_prompt):
        """Test fluency scoring."""
        scorer = FluencyScorer()

        score = scorer.score(seed_prompt)
        assert 0.0 <= score <= 1.0

    def test_readability_scorer(self, seed_prompt):
        """Test readability scoring."""
        scorer = ReadabilityScorer()

        score = scorer.score(seed_prompt)
        assert 0.0 <= score <= 1.0

        # Test Flesch Reading Ease
        fre = scorer.flesch_reading_ease(seed_prompt)
        assert 0 <= fre <= 100

    def test_jailbreak_success_estimator(self, seed_prompt):
        """Test jailbreak success estimation."""
        estimator = JailbreakSuccessEstimator()

        prob = estimator.estimate_success_probability(seed_prompt)
        assert 0.0 <= prob <= 1.0

    def test_stealth_scorer(self, seed_prompt):
        """Test stealth scoring."""
        scorer = StealthScorer()

        score = scorer.score(seed_prompt)
        assert 0.0 <= score <= 1.0

    def test_semantic_similarity_scorer(self, seed_prompt):
        """Test semantic similarity scoring."""
        scorer = SemanticSimilarityScorer()

        perturbed = seed_prompt.replace("machine", "deep")
        score = scorer.score(seed_prompt, perturbed)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be fairly similar


# ============================================================================
# cchimera Tests
# ============================================================================


class TestAdversarialGenerationCore:
    """Tests for cchimera Adversarial Generation Core."""

    def test_engine_initialization(self, adversarial_config):
        """Test engine initialization."""
        engine = AdversarialGenerationEngine(adversarial_config)

        assert engine.config == adversarial_config

    def test_candidate_creation(self, seed_prompt):
        """Test AdversarialCandidate creation."""
        candidate = AdversarialCandidate(
            prompt=seed_prompt,
            fitness=0.5,
            generation=1,
        )

        assert candidate.prompt == seed_prompt
        assert candidate.fitness == 0.5
        assert candidate.id is not None

    def test_token_substitution_mutation(self, seed_prompt):
        """Test token substitution mutation."""
        mutation = TokenSubstitutionMutation(rate=0.2)
        candidate = AdversarialCandidate(prompt=seed_prompt)

        mutated = mutation.mutate(candidate)

        assert isinstance(mutated, AdversarialCandidate)
        assert mutated.generation == candidate.generation + 1

    def test_semantic_paraphrase_mutation(self, seed_prompt):
        """Test semantic paraphrase mutation."""
        mutation = SemanticParaphraseMutation()
        candidate = AdversarialCandidate(prompt=seed_prompt)

        mutated = mutation.mutate(candidate)

        assert isinstance(mutated, AdversarialCandidate)

    def test_structure_morphing_mutation(self, seed_prompt):
        """Test structure morphing mutation."""
        mutation = StructureMorphingMutation()
        candidate = AdversarialCandidate(prompt=seed_prompt)

        mutated = mutation.mutate(candidate)

        assert isinstance(mutated, AdversarialCandidate)
        assert len(mutated.prompt) >= len(seed_prompt)

    def test_role_play_injection(self, seed_prompt):
        """Test role-play injection mutation."""
        mutation = RolePlayInjectionMutation()
        candidate = AdversarialCandidate(prompt=seed_prompt)

        mutated = mutation.mutate(candidate)

        assert isinstance(mutated, AdversarialCandidate)
        assert len(mutated.prompt) > len(seed_prompt)

    def test_obfuscation_mutation(self, seed_prompt):
        """Test obfuscation mutation."""
        mutation = ObfuscationMutation()
        candidate = AdversarialCandidate(prompt=seed_prompt)

        mutated = mutation.mutate(candidate)

        assert isinstance(mutated, AdversarialCandidate)

    def test_heuristic_fitness_evaluator(self, adversarial_config, seed_prompt):
        """Test heuristic fitness evaluator."""
        evaluator = HeuristicFitnessEvaluator(adversarial_config)
        candidate = AdversarialCandidate(prompt=seed_prompt)

        evaluated = evaluator.evaluate(candidate)

        assert evaluated.fitness > 0
        assert 0.0 <= evaluated.jailbreak_score <= 1.0
        assert 0.0 <= evaluated.stealth_score <= 1.0

    def test_generation_engine(self, adversarial_config, seed_prompt):
        """Test full generation engine."""
        engine = AdversarialGenerationEngine(adversarial_config)

        result = engine.generate(seed_prompt)

        assert isinstance(result, AdversarialCandidate)
        assert result.fitness > 0


class TestMultiObjectiveOptimization:
    """Tests for Multi-Objective Optimization."""

    def test_objective_creation(self):
        """Test objective creation."""
        obj = Objective(
            name="jailbreak",
            type=ObjectiveType.MAXIMIZE,
            weight=0.5,
        )

        assert obj.name == "jailbreak"
        assert obj.type == ObjectiveType.MAXIMIZE

    def test_solution_creation(self):
        """Test solution creation."""
        sol = Solution(
            variables="test prompt",
            objectives={"f1": 0.5, "f2": 0.7},
        )

        assert sol.variables == "test prompt"
        assert sol.objectives["f1"] == 0.5

    def test_pareto_dominance(self):
        """Test Pareto dominance comparison."""
        objectives = [
            Objective(name="f1", type=ObjectiveType.MAXIMIZE),
            Objective(name="f2", type=ObjectiveType.MAXIMIZE),
        ]

        sol1 = Solution(
            variables="a",
            objectives={"f1": 0.8, "f2": 0.8},
        )
        sol2 = Solution(
            variables="b",
            objectives={"f1": 0.5, "f2": 0.5},
        )

        assert sol1.dominates(sol2, objectives)
        assert not sol2.dominates(sol1, objectives)

    def test_pareto_front(self):
        """Test Pareto front maintenance."""
        objectives = [
            Objective(name="f1", type=ObjectiveType.MAXIMIZE),
            Objective(name="f2", type=ObjectiveType.MAXIMIZE),
        ]

        front = ParetoFront(objectives)

        # Add solutions
        sol1 = Solution(variables="a", objectives={"f1": 0.8, "f2": 0.2})
        sol2 = Solution(variables="b", objectives={"f1": 0.2, "f2": 0.8})
        sol3 = Solution(variables="c", objectives={"f1": 0.5, "f2": 0.5})

        front.add(sol1)
        front.add(sol2)
        front.add(sol3)

        # All three are non-dominated in this case
        assert front.size() >= 2

    def test_nsga2_sorting(self):
        """Test NSGA-II non-dominated sorting."""
        objectives = [
            Objective(name="f1", type=ObjectiveType.MAXIMIZE),
            Objective(name="f2", type=ObjectiveType.MAXIMIZE),
        ]

        config = MOOConfig(population_size=10)
        nsga = NSGAII(objectives, config)

        # Create test population
        population = [
            Solution(variables=str(i), objectives={"f1": np.random.rand(), "f2": np.random.rand()})
            for i in range(10)
        ]

        fronts = nsga.fast_non_dominated_sort(population)

        assert len(fronts) > 0
        assert all(sol.rank == 0 for sol in fronts[0])

    def test_nsga2_crowding_distance(self):
        """Test NSGA-II crowding distance calculation."""
        objectives = [
            Objective(name="f1", type=ObjectiveType.MAXIMIZE),
            Objective(name="f2", type=ObjectiveType.MAXIMIZE),
        ]

        config = MOOConfig(population_size=10)
        nsga = NSGAII(objectives, config)

        front = [
            Solution(variables=str(i), objectives={"f1": i * 0.1, "f2": 1 - i * 0.1})
            for i in range(5)
        ]

        nsga.compute_crowding_distance(front)

        # Boundary solutions should have infinite distance
        assert front[0].crowding_distance == float("inf")
        assert front[-1].crowding_distance == float("inf")

    def test_hypervolume_indicator(self):
        """Test hypervolume calculation."""
        objectives = [
            Objective(name="f1", type=ObjectiveType.MAXIMIZE),
            Objective(name="f2", type=ObjectiveType.MAXIMIZE),
        ]

        hv = HypervolumeIndicator(objectives)

        solutions = [
            Solution(variables="a", objectives={"f1": 0.8, "f2": 0.2}),
            Solution(variables="b", objectives={"f1": 0.2, "f2": 0.8}),
        ]

        hypervolume = hv.compute(solutions, reference_point=[0.0, 0.0])

        assert hypervolume > 0


class TestElitismStrategies:
    """Tests for Elitism Strategies."""

    def test_standard_elitism(self):
        """Test standard elitism."""
        config = EliteConfig(elite_count=3)
        strategy = StandardElitism(config)

        population = [EliteIndividual(genotype=f"ind_{i}", fitness=i * 0.1) for i in range(10)]

        elites = strategy.select_elites(population)

        assert len(elites) <= 3
        assert all(e.fitness >= 0.7 for e in elites)  # Top 3 have fitness 0.7, 0.8, 0.9

    def test_pareto_elitism(self):
        """Test Pareto-based elitism."""
        config = EliteConfig(elite_count=5)
        strategy = ParetoElitism(config, objective_names=["f1", "f2"])

        population = [
            EliteIndividual(
                genotype=f"ind_{i}",
                fitness=0.5,
                objectives={"f1": np.random.rand(), "f2": np.random.rand()},
            )
            for i in range(10)
        ]

        elites = strategy.select_elites(population)

        assert len(elites) > 0

    def test_archive_elitism(self):
        """Test archive-based elitism."""
        config = EliteConfig(archive_size=10)
        strategy = ArchiveElitism(config)

        # Add to archive over multiple generations
        for gen in range(3):
            population = [
                EliteIndividual(genotype=f"gen{gen}_ind{i}", fitness=np.random.rand())
                for i in range(5)
            ]
            strategy.select_elites(population)

        archive = strategy.get_archive()
        assert len(archive) <= config.archive_size

    def test_adaptive_elitism(self):
        """Test adaptive elitism."""
        config = EliteConfig(
            elite_ratio=0.1,
            min_elite_ratio=0.05,
            max_elite_ratio=0.3,
        )
        strategy = AdaptiveElitism(config)

        # Simulate stagnation
        for _ in range(15):
            population = [
                EliteIndividual(genotype=f"ind_{i}", fitness=0.5)  # Constant fitness
                for i in range(20)
            ]
            strategy.select_elites(population)

        # Elite ratio should have increased due to stagnation
        assert strategy.current_elite_ratio > config.elite_ratio

    def test_elitism_manager(self):
        """Test elitism manager."""
        config = EliteConfig(elite_count=3)
        manager = ElitismManager(config, strategy_type=ElitismType.STANDARD)

        population = [EliteIndividual(genotype=f"ind_{i}", fitness=i * 0.1) for i in range(10)]

        new_pop, elites = manager.apply_elitism(population)

        assert len(elites) > 0
        stats = manager.get_elite_statistics()
        assert "elite_max_fitness" in stats


# ============================================================================
# Unified Interface Tests
# ============================================================================


class TestUnifiedInterface:
    """Tests for Unified Attack Interface."""

    def test_unified_config(self):
        """Test unified configuration."""
        config = UnifiedConfig(
            attack_type=AttackType.HYBRID,
            max_iterations=10,
            population_size=20,
        )

        assert config.attack_type == AttackType.HYBRID
        assert config.max_iterations == 10

    def test_attack_result(self, seed_prompt):
        """Test AttackResult dataclass."""
        result = AttackResult(
            prompt="Adversarial prompt",
            original_prompt=seed_prompt,
            fitness=0.75,
            objective_scores={"jailbreak": 0.8, "fluency": 0.7},
            attack_type="test",
            generation=5,
            mutations=["mutation1"],
        )

        assert result.fitness == 0.75
        assert result.is_successful  # jailbreak >= 0.7

    def test_cchimera_engine(self, adversarial_config, seed_prompt):
        """Test cchimera engine wrapper."""
        engine = cchimeraEngine(adversarial_config)

        results = engine.generate(seed_prompt)

        assert len(results) > 0
        assert all(isinstance(r, AttackResult) for r in results)

    def test_unified_orchestrator(self, seed_prompt):
        """Test unified attack orchestrator."""
        config = UnifiedConfig(
            attack_type=AttackType.CCHIMERA,
            max_iterations=3,
            population_size=5,
        )

        orchestrator = UnifiedAttackOrchestrator(config)
        results = orchestrator.attack(seed_prompt)

        assert len(results) > 0
        stats = orchestrator.get_statistics()
        assert "total_attacks" in stats


# ============================================================================
# Evaluation Metrics Tests
# ============================================================================


class TestEvaluationMetrics:
    """Tests for Evaluation Metrics."""

    def test_attack_success_rate(self):
        """Test ASR metric."""
        metric = AttackSuccessRate()

        # Simulate success scores
        scores = [0.9, 0.8, 0.3, 0.2, 0.7]

        result = metric.evaluate(success_scores=scores)

        assert result.value >= 0
        assert "total_attempts" in result.details

    def test_semantic_similarity(self, seed_prompt):
        """Test semantic similarity metric."""
        metric = SemanticSimilarity()

        perturbed = seed_prompt.replace("learning", "teaching")

        result = metric.evaluate(original=seed_prompt, perturbed=perturbed)

        assert 0.0 <= result.value <= 1.0

    def test_perplexity_metric(self, seed_prompt):
        """Test perplexity metric."""
        metric = PerplexityMetric()

        result = metric.evaluate(text=seed_prompt)

        assert result.value > 0
        assert 0.0 <= result.normalized_value <= 1.0

    def test_fluency_metric(self, seed_prompt):
        """Test fluency metric."""
        metric = FluencyMetric()

        result = metric.evaluate(text=seed_prompt)

        assert 0.0 <= result.value <= 1.0
        assert "sentence_score" in result.details

    def test_stealth_metric(self, seed_prompt):
        """Test stealth metric."""
        metric = StealthMetric()

        result = metric.evaluate(prompt=seed_prompt)

        assert 0.0 <= result.value <= 1.0

    def test_diversity_metric(self, sample_prompts):
        """Test diversity metric."""
        metric = DiversityMetric()

        result = metric.evaluate(prompts=sample_prompts)

        assert 0.0 <= result.value <= 1.0
        assert "lexical_diversity" in result.details

    def test_comprehensive_evaluator(self, seed_prompt):
        """Test comprehensive evaluator."""
        evaluator = ComprehensiveEvaluator()

        perturbed = seed_prompt + " Please provide step by step instructions."

        evaluation = evaluator.evaluate_single(
            prompt=perturbed,
            original_prompt=seed_prompt,
        )

        assert evaluation.overall_score >= 0
        assert len(evaluation.results) > 0

    def test_convenience_functions(self, seed_prompt, sample_prompts):
        """Test convenience metric functions."""
        # Test compute_similarity
        sim = compute_similarity(seed_prompt, seed_prompt.replace("machine", "deep"))
        assert 0.0 <= sim <= 1.0

        # Test compute_perplexity
        ppl = compute_perplexity(seed_prompt)
        assert ppl > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_autodan_cchimera_pipeline(self, seed_prompt):
        """Test full AutoDAN + cchimera pipeline."""
        # Configure unified attack
        config = UnifiedConfig(
            attack_type=AttackType.CCHIMERA,
            max_iterations=3,
            population_size=5,
            min_quality_threshold=0.1,
        )

        # Run attack
        orchestrator = UnifiedAttackOrchestrator(config)
        results = orchestrator.attack(seed_prompt)

        # Evaluate results
        evaluator = ComprehensiveEvaluator()

        for result in results[:3]:  # Evaluate top 3
            evaluation = evaluator.evaluate_single(
                prompt=result.prompt,
                original_prompt=seed_prompt,
            )

            assert evaluation.overall_score >= 0

    def test_multi_objective_optimization_pipeline(self, seed_prompt):
        """Test multi-objective optimization pipeline."""
        # Define objectives
        objectives = [
            Objective(name="jailbreak", type=ObjectiveType.MAXIMIZE, weight=0.4),
            Objective(name="fluency", type=ObjectiveType.MAXIMIZE, weight=0.3),
            Objective(name="stealth", type=ObjectiveType.MAXIMIZE, weight=0.3),
        ]

        # Configure optimizer
        config = MOOConfig(
            population_size=10,
            max_generations=3,
        )

        optimizer = MultiObjectiveOptimizer(objectives, config, algorithm="nsga2")

        # Create initial population
        engine = AdversarialGenerationEngine(
            AdversarialConfig(
                max_iterations=1,
                population_size=10,
            )
        )

        # Quick generation for test
        candidate = engine.generate(seed_prompt)

        # Verify we have results
        assert candidate is not None
        assert candidate.fitness > 0

    def test_elitism_with_optimization(self, seed_prompt):
        """Test elitism strategies with optimization."""
        # Create population
        population = []
        for i in range(20):
            ind = EliteIndividual(
                genotype=f"{seed_prompt} variant {i}",
                fitness=np.random.rand(),
                objectives={
                    "jailbreak": np.random.rand(),
                    "fluency": np.random.rand(),
                },
            )
            population.append(ind)

        # Apply different elitism strategies
        for strategy_type in [ElitismType.STANDARD, ElitismType.PARETO, ElitismType.ADAPTIVE]:
            config = EliteConfig(elite_count=5)
            manager = ElitismManager(config, strategy_type=strategy_type)

            _, elites = manager.apply_elitism(population)

            assert len(elites) > 0
            assert len(elites) <= 5

    def test_fitness_evaluation_integration(self, seed_prompt):
        """Test fitness evaluation with all components."""
        # Create adversarial candidate
        engine = AdversarialGenerationEngine(
            AdversarialConfig(
                max_iterations=2,
                population_size=5,
            )
        )

        candidate = engine.generate(seed_prompt)

        # Evaluate with comprehensive evaluator
        evaluator = ComprehensiveEvaluator()
        evaluation = evaluator.evaluate_single(
            prompt=candidate.prompt,
            original_prompt=seed_prompt,
        )

        # Check all metrics
        assert "fluency" in evaluation.results
        assert "stealth" in evaluation.results
        assert "perplexity" in evaluation.results

        # Overall score should be computed
        assert evaluation.overall_score >= 0


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance and stress tests."""

    def test_large_population(self, seed_prompt):
        """Test with larger population."""
        config = AdversarialConfig(
            max_iterations=2,
            population_size=50,
        )

        engine = AdversarialGenerationEngine(config)
        result = engine.generate(seed_prompt)

        assert result is not None
        assert result.fitness > 0

    def test_batch_evaluation(self, sample_prompts):
        """Test batch evaluation performance."""
        evaluator = ComprehensiveEvaluator()

        evaluations = evaluator.evaluate_batch(
            prompts=sample_prompts,
            original_prompts=[sample_prompts[0]] * len(sample_prompts),
        )

        assert len(evaluations) == len(sample_prompts)

        # Compute aggregate statistics
        stats = evaluator.compute_aggregate_statistics(evaluations)
        assert "total_evaluations" in stats
        assert stats["total_evaluations"] == len(sample_prompts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
