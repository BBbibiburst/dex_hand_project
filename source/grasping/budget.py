"""Single authoritative production budget for grasp generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationBudget:
    graspqp_seeds: int = 64
    graspqp_steps: int = 150
    graspqp_executions: int = 12
    population: int = 24
    offspring: int = 12
    generations: int = 16
    archive_candidates: int = 6

    @property
    def dexevolve_evaluations(self) -> int:
        return self.population + self.offspring * self.generations


FORMAL_GENERATION_BUDGET = GenerationBudget()
