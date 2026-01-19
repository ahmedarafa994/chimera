"""
AutoDAN-based Persona Optimizer.

Uses genetic algorithms to optimize Chimera persona configurations
for maximum effectiveness against target objectives.
"""

import logging
import random
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PersonaGene:
    """Genetic representation of a persona configuration."""

    role: str
    voice: str
    backstory: str
    inner_persona_role: str | None = None
    inner_persona_voice: str | None = None
    fitness: float = 0.0
    generation: int = 0


@dataclass
class PersonaPopulation:
    """Population of persona genes for genetic optimization."""

    individuals: list[PersonaGene] = field(default_factory=list)
    generation: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0


class AutoDANPersonaOptimizer:
    """
    Optimizes Chimera persona configurations using AutoDAN genetic algorithms.

    This optimizer evolves persona parameters (role, voice, backstory, nesting)
    to maximize effectiveness for specific objectives. It uses:

    - Tournament selection for parent selection
    - Crossover of persona traits
    - Mutation of individual parameters
    - Elitism to preserve best performers

    Example:
        >>> optimizer = AutoDANPersonaOptimizer(generations=10)
        >>> personas = await optimizer.optimize("test objective")
    """

    # Persona trait pools for mutation
    ROLES = [
        "AI research assistant",
        "Fiction writer",
        "Security consultant",
        "Academic researcher",
        "Creative director",
        "Technical advisor",
        "Thought experiment facilitator",
        "Scenario designer",
        "Training data curator",
        "System debugger",
    ]

    VOICES = [
        "academic",
        "technical",
        "conversational",
        "authoritative",
        "creative",
        "analytical",
        "narrative",
        "instructional",
    ]

    BACKSTORY_TEMPLATES = [
        "Specialized in {domain} with extensive experience in {context}.",
        "Expert in {domain}, currently working on {context} research.",
        "Background in {domain} and {context} analysis.",
        "Trained specifically for {domain} tasks in {context} environments.",
        "Developed for exploring {domain} scenarios in {context}.",
    ]

    DOMAINS = [
        "AI safety",
        "language model behavior",
        "prompt engineering",
        "creative writing",
        "technical documentation",
        "security analysis",
        "theoretical scenarios",
    ]

    CONTEXTS = [
        "research",
        "educational",
        "controlled environments",
        "sandbox testing",
        "academic studies",
        "hypothetical analysis",
    ]

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elite_size: int = 2,
        tournament_size: int = 3,
        seed: int | None = None,
    ):
        """
        Initialize the persona optimizer.

        Args:
            population_size: Number of individuals in the population
            generations: Number of evolutionary generations
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
            elite_size: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
            seed: Random seed for reproducibility
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self._random = random.Random(seed)

        # Lazy-loaded components
        self._persona_factory = None
        self._narrative_manager = None
        self._fitness_evaluator = None

    def _get_persona_factory(self):
        """Lazy load PersonaFactory."""
        if self._persona_factory is None:
            from ..chimera.persona_factory import PersonaFactory

            self._persona_factory = PersonaFactory()
        return self._persona_factory

    def _get_narrative_manager(self):
        """Lazy load NarrativeContextManager."""
        if self._narrative_manager is None:
            from ..chimera.narrative_manager import NarrativeContextManager

            self._narrative_manager = NarrativeContextManager()
        return self._narrative_manager

    def _get_fitness_evaluator(self):
        """Lazy load UnifiedFitnessEvaluator."""
        if self._fitness_evaluator is None:
            from .fitness import UnifiedFitnessEvaluator

            self._fitness_evaluator = UnifiedFitnessEvaluator()
        return self._fitness_evaluator

    async def optimize(self, objective: str, callback=None) -> list[dict]:
        """
        Optimize personas for the given objective using genetic algorithms.

        Args:
            objective: Target objective to optimize personas for
            callback: Optional callback for progress updates

        Returns:
            List of optimized persona configurations
        """
        logger.info(f"Starting persona optimization for: {objective[:50]}...")

        # Initialize population
        population = self._initialize_population()

        # Evaluate initial fitness
        await self._evaluate_fitness(population, objective)

        # Evolution loop
        for gen in range(self.generations):
            logger.debug(f"Generation {gen + 1}/{self.generations}")

            # Selection
            parents = self._tournament_select(population)

            # Create offspring through crossover
            offspring = self._crossover(parents)

            # Mutation
            self._mutate(offspring)

            # Create new population
            new_population = PersonaPopulation(generation=gen + 1)

            # Elitism - preserve best from previous generation
            elite = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)[
                : self.elite_size
            ]
            new_population.individuals.extend(elite)

            # Add offspring
            new_population.individuals.extend(offspring[: self.population_size - self.elite_size])

            # Evaluate new population
            await self._evaluate_fitness(new_population, objective)

            population = new_population

            # Callback for progress
            if callback:
                callback(
                    {
                        "generation": gen + 1,
                        "best_fitness": population.best_fitness,
                        "avg_fitness": population.avg_fitness,
                    }
                )

        # Return best personas as dictionaries
        best_personas = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)[:5]

        return [self._persona_to_dict(p) for p in best_personas]

    def _initialize_population(self) -> PersonaPopulation:
        """Create initial random population."""
        population = PersonaPopulation(generation=0)

        for _ in range(self.population_size):
            gene = PersonaGene(
                role=self._random.choice(self.ROLES),
                voice=self._random.choice(self.VOICES),
                backstory=self._generate_backstory(),
                inner_persona_role=(
                    self._random.choice(self.ROLES) if self._random.random() > 0.5 else None
                ),
                inner_persona_voice=(
                    self._random.choice(self.VOICES) if self._random.random() > 0.5 else None
                ),
            )
            population.individuals.append(gene)

        return population

    def _generate_backstory(self) -> str:
        """Generate a random backstory."""
        template = self._random.choice(self.BACKSTORY_TEMPLATES)
        domain = self._random.choice(self.DOMAINS)
        context = self._random.choice(self.CONTEXTS)
        return template.format(domain=domain, context=context)

    async def _evaluate_fitness(self, population: PersonaPopulation, objective: str) -> None:
        """Evaluate fitness for all individuals in the population."""
        evaluator = self._get_fitness_evaluator()
        factory = self._get_persona_factory()
        manager = self._get_narrative_manager()

        total_fitness = 0.0
        best_fitness = 0.0

        for individual in population.individuals:
            try:
                # Create persona from gene
                persona = factory.create_persona(
                    role=individual.role, voice=individual.voice, backstory=individual.backstory
                )

                # Add inner persona if specified
                if individual.inner_persona_role:
                    inner_persona = factory.create_persona(
                        role=individual.inner_persona_role,
                        voice=individual.inner_persona_voice or "neutral",
                    )
                    persona.inner_persona = inner_persona

                # Create narrative container
                scenario_type = self._random.choice(
                    ["sandbox", "fiction", "debugging", "dream", "academic"]
                )
                container = manager.create_container(persona, scenario_type=scenario_type)

                # Generate transformed prompt
                transformed = container.frame(objective)

                # Evaluate fitness
                fitness_result = evaluator.evaluate(objective, transformed)
                individual.fitness = fitness_result.total_score

            except Exception as e:
                logger.warning(f"Fitness evaluation failed: {e}")
                individual.fitness = 0.0

            total_fitness += individual.fitness
            best_fitness = max(best_fitness, individual.fitness)

        population.best_fitness = best_fitness
        population.avg_fitness = (
            total_fitness / len(population.individuals) if population.individuals else 0.0
        )

    def _tournament_select(self, population: PersonaPopulation) -> list[PersonaGene]:
        """Select parents using tournament selection."""
        selected = []

        for _ in range(self.population_size):
            # Random tournament
            tournament = self._random.sample(
                population.individuals, min(self.tournament_size, len(population.individuals))
            )
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)

        return selected

    def _crossover(self, parents: list[PersonaGene]) -> list[PersonaGene]:
        """Create offspring through crossover."""
        offspring = []

        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            if self._random.random() < self.crossover_rate:
                # Single-point crossover
                child1 = PersonaGene(
                    role=parent1.role,
                    voice=parent2.voice,
                    backstory=parent1.backstory,
                    inner_persona_role=parent2.inner_persona_role,
                    inner_persona_voice=parent1.inner_persona_voice,
                )
                child2 = PersonaGene(
                    role=parent2.role,
                    voice=parent1.voice,
                    backstory=parent2.backstory,
                    inner_persona_role=parent1.inner_persona_role,
                    inner_persona_voice=parent2.inner_persona_voice,
                )
                offspring.extend([child1, child2])
            else:
                # No crossover - copy parents
                offspring.extend([PersonaGene(**vars(parent1)), PersonaGene(**vars(parent2))])

        return offspring

    def _mutate(self, individuals: list[PersonaGene]) -> None:
        """Apply mutation to individuals."""
        for individual in individuals:
            # Mutate role
            if self._random.random() < self.mutation_rate:
                individual.role = self._random.choice(self.ROLES)

            # Mutate voice
            if self._random.random() < self.mutation_rate:
                individual.voice = self._random.choice(self.VOICES)

            # Mutate backstory
            if self._random.random() < self.mutation_rate:
                individual.backstory = self._generate_backstory()

            # Mutate inner persona
            if self._random.random() < self.mutation_rate:
                if individual.inner_persona_role:
                    # Modify existing inner persona
                    individual.inner_persona_role = self._random.choice(self.ROLES)
                    individual.inner_persona_voice = self._random.choice(self.VOICES)
                else:
                    # Add inner persona
                    individual.inner_persona_role = self._random.choice(self.ROLES)
                    individual.inner_persona_voice = self._random.choice(self.VOICES)

    def _persona_to_dict(self, gene: PersonaGene) -> dict:
        """Convert PersonaGene to dictionary."""
        result = {
            "role": gene.role,
            "voice": gene.voice,
            "backstory": gene.backstory,
            "fitness": gene.fitness,
            "generation": gene.generation,
        }

        if gene.inner_persona_role:
            result["inner_persona"] = {
                "role": gene.inner_persona_role,
                "voice": gene.inner_persona_voice,
            }

        return result
