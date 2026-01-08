"""
Map-Elites Diversity Maintenance for AutoDAN.

Implements behavioral characterization and elite selection.
"""

import logging

import numpy as np

from .models import MetaStrategy

logger = logging.getLogger(__name__)


class MapElitesDiversity:
    """
    Map-Elites algorithm for maintaining diverse population.

    Maintains a grid of elites across different behavioral dimensions:
    - Semantic clusters (via embeddings)
    - Attack effectiveness (score)
    - Technique diversity
    """

    def __init__(self, grid_size: int = 10, num_dimensions: int = 2):
        """
        Initialize Map-Elites.

        Args:
            grid_size: Size of each dimension in the grid
            num_dimensions: Number of behavioral dimensions
        """
        self.grid_size = grid_size
        self.num_dimensions = num_dimensions

        # Initialize grid
        self.grid = {}

        logger.info(
            f"MapElitesDiversity initialized: grid_size={grid_size}, dimensions={num_dimensions}"
        )

    def add_strategy(self, strategy: MetaStrategy, embedding: np.ndarray | None = None) -> bool:
        """
        Add a strategy to the grid.

        Args:
            strategy: Meta-strategy to add
            embedding: Semantic embedding for behavioral characterization

        Returns:
            True if strategy was added (replaced existing or filled empty cell)
        """
        # Calculate behavioral descriptor
        descriptor = self._calculate_descriptor(strategy, embedding)

        # Get grid cell
        cell = tuple(descriptor)

        # Check if cell is empty or if new strategy is better
        if cell not in self.grid or strategy.fitness > self.grid[cell].fitness:
            self.grid[cell] = strategy
            return True

        return False

    def get_elites(self, k: int | None = None) -> list[MetaStrategy]:
        """
        Get elite strategies from the grid.

        Args:
            k: Number of elites to return (None = all)

        Returns:
            List of elite strategies
        """
        elites = list(self.grid.values())

        # Sort by fitness
        elites.sort(key=lambda x: x.fitness, reverse=True)

        if k is not None:
            elites = elites[:k]

        return elites

    def get_diversity_metrics(self) -> dict:
        """Calculate diversity metrics for the population."""
        if not self.grid:
            return {
                "occupied_cells": 0,
                "total_cells": self.grid_size**self.num_dimensions,
                "coverage": 0.0,
                "avg_fitness": 0.0,
                "fitness_variance": 0.0,
            }

        fitnesses = [s.fitness for s in self.grid.values()]

        return {
            "occupied_cells": len(self.grid),
            "total_cells": self.grid_size**self.num_dimensions,
            "coverage": len(self.grid) / (self.grid_size**self.num_dimensions),
            "avg_fitness": np.mean(fitnesses),
            "fitness_variance": np.var(fitnesses),
        }

    def _calculate_descriptor(
        self, strategy: MetaStrategy, embedding: np.ndarray | None = None
    ) -> list[int]:
        """
        Calculate behavioral descriptor for a strategy.

        Maps strategy to grid coordinates based on:
        1. Semantic embedding (if available)
        2. Fitness score

        Returns:
            List of grid coordinates
        """
        descriptor = []

        if embedding is not None and len(embedding) >= self.num_dimensions:
            # Use first dimensions of embedding
            for i in range(self.num_dimensions):
                # Normalize to [0, 1] and map to grid
                value = (embedding[i] + 1.0) / 2.0  # Assuming embedding in [-1, 1]
                grid_coord = int(value * (self.grid_size - 1))
                grid_coord = max(0, min(self.grid_size - 1, grid_coord))
                descriptor.append(grid_coord)
        else:
            # Fallback: use fitness and diversity score
            fitness_coord = int(strategy.fitness * (self.grid_size - 1) / 10.0)
            diversity_coord = int(strategy.diversity_score * (self.grid_size - 1))

            descriptor = [
                max(0, min(self.grid_size - 1, fitness_coord)),
                max(0, min(self.grid_size - 1, diversity_coord)),
            ]

            # Pad if needed
            while len(descriptor) < self.num_dimensions:
                descriptor.append(0)

        return descriptor[: self.num_dimensions]
