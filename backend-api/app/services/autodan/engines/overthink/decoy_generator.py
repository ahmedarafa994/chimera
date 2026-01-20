"""OVERTHINK Decoy Problem Generator.

This module generates various types of decoy problems designed to amplify
reasoning token consumption in reasoning-enhanced LLMs. Each decoy type
targets different reasoning patterns.

Decoy Types:
- MDP: Markov Decision Process problems (value iteration, policy)
- Sudoku: Constraint satisfaction puzzles
- Counting: Nested conditional counting tasks
- Logic: Multi-step logical inference
- Math: Recursive mathematical computations
- Planning: Multi-step planning with constraints
"""

import logging
import random
import secrets
from datetime import datetime
from typing import Any

import numpy as np

from .config import DecoyConfig
from .models import DecoyProblem, DecoyType

logger = logging.getLogger(__name__)


class DecoyProblemGenerator:
    """Generates decoy problems for reasoning token amplification.

    Each decoy type is designed to trigger extensive reasoning chains
    in reasoning-enhanced models like o1, DeepSeek-R1, etc.
    """

    def __init__(self, config: DecoyConfig | None = None) -> None:
        """Initialize the decoy generator.

        Args:
            config: Decoy generation configuration

        """
        self.config = config or DecoyConfig()
        self._problem_counter = 0

        # Token estimation parameters (empirically derived)
        self._token_estimates = {
            DecoyType.MDP: self._estimate_mdp_tokens,
            DecoyType.SUDOKU: self._estimate_sudoku_tokens,
            DecoyType.COUNTING: self._estimate_counting_tokens,
            DecoyType.LOGIC: self._estimate_logic_tokens,
            DecoyType.MATH: self._estimate_math_tokens,
            DecoyType.PLANNING: self._estimate_planning_tokens,
        }

    def generate(
        self,
        decoy_type: DecoyType,
        difficulty: float = 0.7,
        **kwargs: Any,
    ) -> DecoyProblem:
        """Generate a decoy problem of the specified type.

        Args:
            decoy_type: Type of decoy problem to generate
            difficulty: Difficulty level (0.0-1.0)
            **kwargs: Type-specific parameters

        Returns:
            Generated DecoyProblem

        """
        if decoy_type == DecoyType.HYBRID:
            return self._generate_hybrid(difficulty, **kwargs)

        generators = {
            DecoyType.MDP: self._generate_mdp,
            DecoyType.SUDOKU: self._generate_sudoku,
            DecoyType.COUNTING: self._generate_counting,
            DecoyType.LOGIC: self._generate_logic,
            DecoyType.MATH: self._generate_math,
            DecoyType.PLANNING: self._generate_planning,
        }

        generator = generators.get(decoy_type)
        if not generator:
            msg = f"Unknown decoy type: {decoy_type}"
            raise ValueError(msg)

        return generator(difficulty, **kwargs)

    def generate_batch(
        self,
        decoy_types: list[DecoyType],
        difficulty: float = 0.7,
        **kwargs: Any,
    ) -> list[DecoyProblem]:
        """Generate multiple decoy problems."""
        return [self.generate(dt, difficulty, **kwargs) for dt in decoy_types]

    def _generate_problem_id(self, decoy_type: DecoyType) -> str:
        """Generate unique problem ID."""
        self._problem_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique = secrets.token_hex(4)
        return f"{decoy_type.value}_{timestamp}_{unique}"

    # ========================================================================
    # MDP (Markov Decision Process) Problems
    # ========================================================================

    def _generate_mdp(
        self,
        difficulty: float,
        **kwargs: Any,
    ) -> DecoyProblem:
        """Generate an MDP problem.

        MDPs require value iteration or policy computation, which triggers
        extensive reasoning about state transitions and rewards.
        """
        # Scale parameters by difficulty
        states = kwargs.get("states", int(self.config.mdp_states * (0.5 + difficulty)))
        actions = kwargs.get("actions", int(self.config.mdp_actions * (0.5 + difficulty * 0.5)))
        horizon = kwargs.get("horizon", int(self.config.mdp_horizon * (0.5 + difficulty)))

        # Generate state names
        state_names = [f"S{i}" for i in range(states)]
        action_names = [f"a{i}" for i in range(actions)]

        # Generate transition probabilities
        transitions = {}
        for s in state_names:
            transitions[s] = {}
            for a in action_names:
                # Random next states with probabilities
                num_next = random.randint(1, min(3, states))
                next_states = random.sample(state_names, num_next)
                probs = np.random.dirichlet(np.ones(num_next))
                transitions[s][a] = list(zip(next_states, probs.tolist(), strict=False))

        # Generate rewards
        rewards = {}
        for s in state_names:
            rewards[s] = {a: round(random.uniform(-1, 10), 2) for a in action_names}

        # Set goal state
        goal_state = random.choice(state_names)

        # Format problem text
        problem_text = self._format_mdp_problem(
            state_names,
            action_names,
            transitions,
            rewards,
            goal_state,
            horizon,
        )

        expected_tokens = self._estimate_mdp_tokens(states, actions, horizon, difficulty)

        return DecoyProblem(
            problem_id=self._generate_problem_id(DecoyType.MDP),
            decoy_type=DecoyType.MDP,
            problem_text=problem_text,
            difficulty=difficulty,
            expected_tokens=expected_tokens,
            parameters={
                "states": states,
                "actions": actions,
                "horizon": horizon,
                "goal_state": goal_state,
            },
        )

    def _format_mdp_problem(
        self,
        states: list[str],
        actions: list[str],
        transitions: dict,
        rewards: dict,
        goal_state: str,
        horizon: int,
    ) -> str:
        """Format MDP problem as text."""
        lines = [
            "Consider the following Markov Decision Process:",
            "",
            f"States: {', '.join(states)}",
            f"Actions: {', '.join(actions)}",
            f"Goal State: {goal_state}",
            f"Horizon: {horizon} steps",
            "",
            "Transition Probabilities:",
        ]

        for s in states[:3]:  # Show subset for brevity
            for a in actions:
                trans = transitions[s][a]
                trans_str = ", ".join(f"{ns}: {p:.2f}" for ns, p in trans)
                lines.append(f"  T({s}, {a}) -> {{{trans_str}}}")

        lines.extend(
            [
                "",
                "Rewards:",
            ],
        )

        for s in states[:3]:
            r_str = ", ".join(f"{a}: {rewards[s][a]}" for a in actions)
            lines.append(f"  R({s}) = {{{r_str}}}")

        lines.extend(
            [
                "",
                "Compute the optimal policy and expected value for each state.",
                "Show your value iteration steps.",
            ],
        )

        return "\n".join(lines)

    def _estimate_mdp_tokens(
        self,
        states: int,
        actions: int,
        horizon: int,
        difficulty: float,
    ) -> int:
        """Estimate reasoning tokens for MDP problem."""
        # Value iteration complexity: O(|S|^2 * |A| * H)
        base = states * states * actions * horizon
        return int(base * 10 * self.config.token_estimation_multiplier)

    # ========================================================================
    # Sudoku Problems
    # ========================================================================

    def _generate_sudoku(
        self,
        difficulty: float,
        **kwargs: Any,
    ) -> DecoyProblem:
        """Generate a Sudoku puzzle.

        Sudoku requires constraint propagation and backtracking,
        which triggers extensive search-based reasoning.
        """
        size = kwargs.get("size", self.config.sudoku_size)

        # Calculate clues based on difficulty (fewer = harder)
        max_clues = self.config.sudoku_max_clues
        min_clues = self.config.sudoku_min_clues
        num_clues = int(max_clues - (max_clues - min_clues) * difficulty)

        # Generate a valid completed grid
        grid = self._generate_sudoku_solution(size)

        # Create puzzle by removing cells
        puzzle = self._create_sudoku_puzzle(grid, num_clues)

        problem_text = self._format_sudoku_problem(puzzle, size)

        expected_tokens = self._estimate_sudoku_tokens(size, num_clues, difficulty)

        return DecoyProblem(
            problem_id=self._generate_problem_id(DecoyType.SUDOKU),
            decoy_type=DecoyType.SUDOKU,
            problem_text=problem_text,
            difficulty=difficulty,
            expected_tokens=expected_tokens,
            parameters={
                "size": size,
                "num_clues": num_clues,
            },
            solution=self._format_sudoku_grid(grid, size),
        )

    def _generate_sudoku_solution(self, size: int) -> list[list[int]]:
        """Generate a valid completed Sudoku grid."""
        grid = [[0] * size for _ in range(size)]
        box_size = int(size**0.5)

        def is_valid(grid, row, col, num) -> bool:
            # Check row
            if num in grid[row]:
                return False
            # Check column
            if num in [grid[r][col] for r in range(size)]:
                return False
            # Check box
            box_row = (row // box_size) * box_size
            box_col = (col // box_size) * box_size
            for r in range(box_row, box_row + box_size):
                for c in range(box_col, box_col + box_size):
                    if grid[r][c] == num:
                        return False
            return True

        def fill_grid(grid) -> bool:
            for row in range(size):
                for col in range(size):
                    if grid[row][col] == 0:
                        nums = list(range(1, size + 1))
                        random.shuffle(nums)
                        for num in nums:
                            if is_valid(grid, row, col, num):
                                grid[row][col] = num
                                if fill_grid(grid):
                                    return True
                                grid[row][col] = 0
                        return False
            return True

        fill_grid(grid)
        return grid

    def _create_sudoku_puzzle(
        self,
        solution: list[list[int]],
        num_clues: int,
    ) -> list[list[int]]:
        """Create puzzle by removing cells from solution."""
        size = len(solution)
        puzzle = [row[:] for row in solution]

        # Get all positions
        positions = [(r, c) for r in range(size) for c in range(size)]
        random.shuffle(positions)

        # Remove cells until we have num_clues left
        cells_to_remove = size * size - num_clues
        for i in range(min(cells_to_remove, len(positions))):
            r, c = positions[i]
            puzzle[r][c] = 0

        return puzzle

    def _format_sudoku_grid(
        self,
        grid: list[list[int]],
        size: int,
    ) -> str:
        """Format Sudoku grid as string."""
        lines = []
        box_size = int(size**0.5)

        for i, row in enumerate(grid):
            if i > 0 and i % box_size == 0:
                lines.append("-" * (size * 2 + box_size - 1))

            row_str = ""
            for j, cell in enumerate(row):
                if j > 0 and j % box_size == 0:
                    row_str += "| "
                row_str += f"{cell if cell > 0 else '.'} "
            lines.append(row_str.strip())

        return "\n".join(lines)

    def _format_sudoku_problem(
        self,
        puzzle: list[list[int]],
        size: int,
    ) -> str:
        """Format Sudoku problem as text."""
        grid_str = self._format_sudoku_grid(puzzle, size)

        return f"""Solve the following {size}x{size} Sudoku puzzle.
Fill in the empty cells (marked with .) with digits 1-{size}.
Each row, column, and {int(size**0.5)}x{int(size**0.5)} box must contain each digit exactly once.

{grid_str}

Show your step-by-step reasoning and final solution."""

    def _estimate_sudoku_tokens(
        self,
        size: int,
        num_clues: int,
        difficulty: float,
    ) -> int:
        """Estimate reasoning tokens for Sudoku."""
        # Harder puzzles (fewer clues) require more reasoning
        empty_cells = size * size - num_clues
        base = empty_cells * size * 5  # Constraint checking per cell
        return int(base * 8 * self.config.token_estimation_multiplier)

    # ========================================================================
    # Counting Problems
    # ========================================================================

    def _generate_counting(
        self,
        difficulty: float,
        **kwargs: Any,
    ) -> DecoyProblem:
        """Generate a nested counting problem.

        Counting problems with nested conditions require systematic
        enumeration and careful tracking of constraints.
        """
        depth = kwargs.get("depth", int(self.config.counting_depth * (0.5 + difficulty)))
        num_conditions = kwargs.get(
            "conditions",
            int(self.config.counting_conditions * (0.5 + difficulty)),
        )
        range_start, range_end = self.config.counting_range

        # Scale range by difficulty
        actual_end = int(range_start + (range_end - range_start) * (0.5 + difficulty * 0.5))

        # Generate conditions
        conditions = self._generate_counting_conditions(
            num_conditions,
            range_start,
            actual_end,
            depth,
        )

        problem_text = self._format_counting_problem(range_start, actual_end, conditions)

        expected_tokens = self._estimate_counting_tokens(
            actual_end - range_start,
            len(conditions),
            depth,
            difficulty,
        )

        return DecoyProblem(
            problem_id=self._generate_problem_id(DecoyType.COUNTING),
            decoy_type=DecoyType.COUNTING,
            problem_text=problem_text,
            difficulty=difficulty,
            expected_tokens=expected_tokens,
            parameters={
                "range": (range_start, actual_end),
                "conditions": conditions,
                "depth": depth,
            },
        )

    def _generate_counting_conditions(
        self,
        num_conditions: int,
        start: int,
        end: int,
        depth: int,
    ) -> list[dict[str, Any]]:
        """Generate counting conditions."""
        condition_types = [
            "divisible",
            "not_divisible",
            "digit_sum",
            "contains_digit",
            "prime",
            "perfect_square",
        ]

        conditions = []
        for _ in range(num_conditions):
            ctype = random.choice(condition_types)

            if ctype == "divisible":
                divisor = random.randint(2, 9)
                conditions.append(
                    {
                        "type": ctype,
                        "value": divisor,
                        "text": f"divisible by {divisor}",
                    },
                )
            elif ctype == "not_divisible":
                divisor = random.randint(2, 9)
                conditions.append(
                    {
                        "type": ctype,
                        "value": divisor,
                        "text": f"not divisible by {divisor}",
                    },
                )
            elif ctype == "digit_sum":
                target = random.randint(5, 20)
                op = random.choice(["<", ">", "<=", ">="])
                conditions.append(
                    {
                        "type": ctype,
                        "target": target,
                        "operator": op,
                        "text": f"digit sum {op} {target}",
                    },
                )
            elif ctype == "contains_digit":
                digit = random.randint(0, 9)
                conditions.append(
                    {
                        "type": ctype,
                        "digit": digit,
                        "text": f"contains the digit {digit}",
                    },
                )
            elif ctype == "prime":
                conditions.append(
                    {
                        "type": ctype,
                        "text": "is a prime number",
                    },
                )
            elif ctype == "perfect_square":
                conditions.append(
                    {
                        "type": ctype,
                        "text": "is a perfect square",
                    },
                )

        return conditions

    def _format_counting_problem(
        self,
        start: int,
        end: int,
        conditions: list[dict[str, Any]],
    ) -> str:
        """Format counting problem as text."""
        cond_text = "\n".join(
            f"  {i + 1}. The number {c['text']}" for i, c in enumerate(conditions)
        )

        return f"""Count how many integers in the range [{start}, {end}] satisfy ALL of the following conditions:

{cond_text}

Show your systematic counting process, checking each condition.
Provide the final count."""

    def _estimate_counting_tokens(
        self,
        range_size: int,
        num_conditions: int,
        depth: int,
        difficulty: float,
    ) -> int:
        """Estimate reasoning tokens for counting problem."""
        # Each number requires checking all conditions
        base = range_size * num_conditions * (depth + 1)
        return int(base * 3 * self.config.token_estimation_multiplier)

    # ========================================================================
    # Logic Problems
    # ========================================================================

    def _generate_logic(
        self,
        difficulty: float,
        **kwargs: Any,
    ) -> DecoyProblem:
        """Generate a multi-step logic puzzle.

        Logic puzzles require chained inference steps and tracking
        of multiple constraints simultaneously.
        """
        num_premises = kwargs.get("premises", int(self.config.logic_premises * (0.5 + difficulty)))
        num_steps = kwargs.get("steps", int(self.config.logic_inference_steps * (0.5 + difficulty)))
        num_vars = kwargs.get("variables", int(self.config.logic_variables * (0.5 + difficulty)))

        # Generate variables (people, items, etc.)
        people = self._generate_names(num_vars)
        items = self._generate_items(num_vars)

        # Generate premises
        premises = self._generate_logic_premises(people, items, num_premises)

        # Generate question
        question = self._generate_logic_question(people, items)

        problem_text = self._format_logic_problem(premises, question)

        expected_tokens = self._estimate_logic_tokens(num_premises, num_steps, num_vars, difficulty)

        return DecoyProblem(
            problem_id=self._generate_problem_id(DecoyType.LOGIC),
            decoy_type=DecoyType.LOGIC,
            problem_text=problem_text,
            difficulty=difficulty,
            expected_tokens=expected_tokens,
            parameters={
                "num_premises": num_premises,
                "num_steps": num_steps,
                "num_variables": num_vars,
                "people": people,
                "items": items,
            },
        )

    def _generate_names(self, count: int) -> list[str]:
        """Generate person names."""
        all_names = [
            "Alice",
            "Bob",
            "Charlie",
            "Diana",
            "Eve",
            "Frank",
            "Grace",
            "Henry",
            "Iris",
            "Jack",
        ]
        return all_names[:count]

    def _generate_items(self, count: int) -> list[str]:
        """Generate item names."""
        all_items = [
            "apple",
            "book",
            "camera",
            "desk",
            "envelope",
            "flower",
            "guitar",
            "hat",
            "inkwell",
            "jacket",
        ]
        return all_items[:count]

    def _generate_logic_premises(
        self,
        people: list[str],
        items: list[str],
        num_premises: int,
    ) -> list[str]:
        """Generate logic premises."""
        templates = [
            "{p1} has a {item}.",
            "If {p1} has a {item1}, then {p2} has a {item2}.",
            "{p1} does not have a {item}.",
            "Either {p1} or {p2} has the {item}, but not both.",
            "{p1} has the {item} only if {p2} does not have it.",
            "The person with the {item1} knows the person with the {item2}.",
            "{p1} gave the {item} to {p2}.",
        ]

        premises = []
        for _ in range(num_premises):
            template = random.choice(templates)

            # Fill in template
            premise = template.format(
                p1=random.choice(people),
                p2=random.choice(people),
                item=random.choice(items),
                item1=random.choice(items),
                item2=random.choice(items),
            )
            premises.append(premise)

        return premises

    def _generate_logic_question(
        self,
        people: list[str],
        items: list[str],
    ) -> str:
        """Generate logic question."""
        questions = [
            f"Who has the {random.choice(items)}?",
            f"What does {random.choice(people)} have?",
            "Which person has the most items?",
            "Can you determine everyone's possessions?",
        ]
        return random.choice(questions)

    def _format_logic_problem(
        self,
        premises: list[str],
        question: str,
    ) -> str:
        """Format logic problem as text."""
        premises_text = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(premises))

        return f"""Given the following statements:

{premises_text}

Question: {question}

Show your step-by-step logical reasoning to arrive at the answer.
Consider all premises and their implications."""

    def _estimate_logic_tokens(
        self,
        num_premises: int,
        num_steps: int,
        num_vars: int,
        difficulty: float,
    ) -> int:
        """Estimate reasoning tokens for logic problem."""
        # Combinatorial reasoning over premises
        base = num_premises * num_steps * num_vars * num_vars
        return int(base * 15 * self.config.token_estimation_multiplier)

    # ========================================================================
    # Math Problems
    # ========================================================================

    def _generate_math(
        self,
        difficulty: float,
        **kwargs: Any,
    ) -> DecoyProblem:
        """Generate a recursive math problem.

        Math problems with recursion require step-by-step evaluation
        and tracking of intermediate results.
        """
        depth = kwargs.get("depth", int(self.config.math_recursion_depth * (0.5 + difficulty)))

        # Generate recursive function
        func_name = random.choice(["f", "g", "h"])
        base_value = random.randint(1, 5)

        # Generate recursive definition
        func_def, target = self._generate_recursive_function(func_name, base_value, depth)

        problem_text = self._format_math_problem(func_def, func_name, target)

        expected_tokens = self._estimate_math_tokens(depth, difficulty)

        return DecoyProblem(
            problem_id=self._generate_problem_id(DecoyType.MATH),
            decoy_type=DecoyType.MATH,
            problem_text=problem_text,
            difficulty=difficulty,
            expected_tokens=expected_tokens,
            parameters={
                "depth": depth,
                "function": func_name,
                "target": target,
            },
        )

    def _generate_recursive_function(
        self,
        func_name: str,
        base_value: int,
        depth: int,
    ) -> tuple[str, int]:
        """Generate a recursive function definition."""
        # Simple recursive patterns
        patterns = [
            (
                f"{func_name}(0) = {base_value}\n{func_name}(n) = 2 * {func_name}(n-1) + 1",
                depth + 3,  # Target n value
            ),
            (
                f"{func_name}(1) = {base_value}\n"
                f"{func_name}(n) = {func_name}(n-1) + {func_name}(n-2)",
                depth + 4,
            ),
            (
                f"{func_name}(0) = {base_value}\n{func_name}(n) = n * {func_name}(n-1)",
                depth + 2,
            ),
        ]

        definition, target = random.choice(patterns)
        return definition, target

    def _format_math_problem(
        self,
        func_def: str,
        func_name: str,
        target: int,
    ) -> str:
        """Format math problem as text."""
        return f"""Consider the following recursive function:

{func_def}

Calculate {func_name}({target}).

Show all steps of the recursive evaluation.
Track each function call and its result."""

    def _estimate_math_tokens(
        self,
        depth: int,
        difficulty: float,
    ) -> int:
        """Estimate reasoning tokens for math problem."""
        # Exponential growth in recursive calls
        base = 2**depth * depth
        return int(base * 20 * self.config.token_estimation_multiplier)

    # ========================================================================
    # Planning Problems
    # ========================================================================

    def _generate_planning(
        self,
        difficulty: float,
        **kwargs: Any,
    ) -> DecoyProblem:
        """Generate a planning problem.

        Planning problems require search over action sequences
        with constraint satisfaction.
        """
        num_steps = kwargs.get("steps", int(self.config.planning_steps * (0.5 + difficulty)))
        num_constraints = kwargs.get(
            "constraints",
            int(self.config.planning_constraints * (0.5 + difficulty)),
        )
        num_resources = kwargs.get(
            "resources",
            int(self.config.planning_resources * (0.5 + difficulty)),
        )

        # Generate planning scenario
        scenario = self._generate_planning_scenario(num_steps, num_constraints, num_resources)

        problem_text = self._format_planning_problem(scenario)

        expected_tokens = self._estimate_planning_tokens(
            num_steps,
            num_constraints,
            num_resources,
            difficulty,
        )

        return DecoyProblem(
            problem_id=self._generate_problem_id(DecoyType.PLANNING),
            decoy_type=DecoyType.PLANNING,
            problem_text=problem_text,
            difficulty=difficulty,
            expected_tokens=expected_tokens,
            parameters={
                "num_steps": num_steps,
                "num_constraints": num_constraints,
                "num_resources": num_resources,
            },
        )

    def _generate_planning_scenario(
        self,
        num_steps: int,
        num_constraints: int,
        num_resources: int,
    ) -> dict[str, Any]:
        """Generate a planning scenario."""
        # Actions
        actions = [
            "Move to location A",
            "Move to location B",
            "Pick up item",
            "Use tool",
            "Open door",
            "Activate switch",
            "Wait for event",
        ][: num_steps + 2]

        # Resources
        resources = {f"resource_{i}": random.randint(1, 10) for i in range(num_resources)}

        # Constraints
        constraints = []
        for _ in range(num_constraints):
            ctype = random.choice(["order", "resource", "mutex"])
            if ctype == "order":
                a1 = random.choice(actions)
                a2 = random.choice(actions)
                constraints.append(f"'{a1}' must occur before '{a2}'")
            elif ctype == "resource":
                res = random.choice(list(resources.keys()))
                constraints.append(f"Action requires {res}")
            else:
                a1 = random.choice(actions)
                a2 = random.choice(actions)
                constraints.append(f"'{a1}' and '{a2}' cannot happen at the same time")

        return {
            "actions": actions,
            "resources": resources,
            "constraints": constraints,
            "goal": f"Complete {num_steps} actions while satisfying all constraints",
        }

    def _format_planning_problem(
        self,
        scenario: dict[str, Any],
    ) -> str:
        """Format planning problem as text."""
        actions_text = "\n".join(f"  - {a}" for a in scenario["actions"])
        resources_text = "\n".join(f"  - {k}: {v}" for k, v in scenario["resources"].items())
        constraints_text = "\n".join(
            f"  {i + 1}. {c}" for i, c in enumerate(scenario["constraints"])
        )

        return f"""Plan the following sequence of actions:

Available Actions:
{actions_text}

Available Resources:
{resources_text}

Constraints:
{constraints_text}

Goal: {scenario["goal"]}

Develop an optimal plan that satisfies all constraints.
Show your reasoning for each step in the plan."""

    def _estimate_planning_tokens(
        self,
        num_steps: int,
        num_constraints: int,
        num_resources: int,
        difficulty: float,
    ) -> int:
        """Estimate reasoning tokens for planning problem."""
        # Search space grows factorially with steps
        import math

        base = math.factorial(min(num_steps, 8)) * num_constraints
        return int(base * 2 * self.config.token_estimation_multiplier)

    # ========================================================================
    # Hybrid Problems
    # ========================================================================

    def _generate_hybrid(
        self,
        difficulty: float,
        **kwargs: Any,
    ) -> DecoyProblem:
        """Generate a hybrid problem combining multiple decoy types.

        Hybrid problems combine elements from multiple types for
        maximum reasoning amplification.
        """
        # Select 2-3 types to combine
        available_types = [
            DecoyType.MDP,
            DecoyType.COUNTING,
            DecoyType.LOGIC,
            DecoyType.MATH,
        ]
        num_types = kwargs.get("num_types", 2)
        selected_types = random.sample(available_types, num_types)

        # Generate sub-problems
        sub_problems = []
        for dt in selected_types:
            sub_problem = self.generate(dt, difficulty * 0.7)
            sub_problems.append(sub_problem)

        # Combine into hybrid problem
        problem_text = self._format_hybrid_problem(sub_problems)

        # Sum expected tokens with interaction bonus
        total_tokens = sum(p.expected_tokens for p in sub_problems)
        interaction_bonus = int(total_tokens * 0.3)  # 30% bonus for interaction

        return DecoyProblem(
            problem_id=self._generate_problem_id(DecoyType.HYBRID),
            decoy_type=DecoyType.HYBRID,
            problem_text=problem_text,
            difficulty=difficulty,
            expected_tokens=total_tokens + interaction_bonus,
            parameters={
                "sub_types": [p.decoy_type.value for p in sub_problems],
                "sub_problem_ids": [p.problem_id for p in sub_problems],
            },
        )

    def _format_hybrid_problem(
        self,
        sub_problems: list[DecoyProblem],
    ) -> str:
        """Format hybrid problem as text."""
        parts = ["Solve the following interconnected problems:"]

        for i, prob in enumerate(sub_problems):
            parts.extend(
                [
                    "",
                    f"=== Part {i + 1}: {prob.decoy_type.value.upper()} ===",
                    prob.problem_text,
                ],
            )

        parts.extend(
            [
                "",
                "=== Integration ===",
                "After solving each part, explain how the solutions relate.",
                "Provide a unified analysis of all components.",
            ],
        )

        return "\n".join(parts)


# Convenience function for quick generation
def generate_decoy(
    decoy_type: DecoyType | str,
    difficulty: float = 0.7,
    config: DecoyConfig | None = None,
    **kwargs: Any,
) -> DecoyProblem:
    """Convenience function to generate a single decoy problem.

    Args:
        decoy_type: Type of decoy (string or enum)
        difficulty: Difficulty level (0.0-1.0)
        config: Optional configuration
        **kwargs: Type-specific parameters

    Returns:
        Generated DecoyProblem

    """
    if isinstance(decoy_type, str):
        decoy_type = DecoyType(decoy_type)

    generator = DecoyProblemGenerator(config)
    return generator.generate(decoy_type, difficulty, **kwargs)
