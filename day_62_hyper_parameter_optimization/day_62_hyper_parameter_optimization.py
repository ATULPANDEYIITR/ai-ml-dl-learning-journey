from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from math import log10
from random import Random
from statistics import mean, stdev
from typing import Callable, Iterable, Any


class SearchStrategy(Enum):
    GRID = "grid"
    RANDOM = "random"


@dataclass(frozen=True)
class SearchSpace:
    learning_rates: tuple[float, ...]
    max_depths: tuple[int, ...]
    estimators: tuple[int, ...]

    def validate(self) -> None:
        if not self.learning_rates or not self.max_depths or not self.estimators:
            raise ValueError("Every search-space dimension must contain at least one value.")

        if any(rate <= 0 for rate in self.learning_rates):
            raise ValueError("Learning rates must be positive.")

        if any(depth < 1 for depth in self.max_depths):
            raise ValueError("Maximum depth must be at least 1.")

        if any(count < 1 for count in self.estimators):
            raise ValueError("Estimator counts must be positive.")

    @property
    def size(self) -> int:
        return (
            len(self.learning_rates)
            * len(self.max_depths)
            * len(self.estimators)
        )


@dataclass(frozen=True)
class Configuration:
    learning_rate: float
    max_depth: int
    estimators: int

    def key(self) -> tuple[float, int, int]:
        return (self.learning_rate, self.max_depth, self.estimators)


@dataclass
class TrialResult:
    configuration: Configuration
    score: float
    duration_units: float
    strategy: SearchStrategy


@dataclass
class OptimizationResult:
    strategy: SearchStrategy
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def best(self) -> TrialResult:
        if not self.trials:
            raise RuntimeError("No trials were executed.")
        return max(self.trials, key=lambda trial: trial.score)

    @property
    def average_score(self) -> float:
        return mean(trial.score for trial in self.trials)


class SyntheticModelEvaluator:
    """
    Deterministic evaluator used to make the optimization workflow executable
    without external ML libraries.

    The score represents validation quality. It has a hidden optimum so that
    search strategies can be compared objectively.
    """

    def __init__(self, seed: int = 42) -> None:
        self.random = Random(seed)

    def evaluate(self, configuration: Configuration) -> tuple[float, float]:
        if configuration.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if configuration.max_depth <= 0:
            raise ValueError("Depth must be positive.")
        if configuration.estimators <= 0:
            raise ValueError("Estimator count must be positive.")

        # Hidden validation-quality surface.
        lr_penalty = abs(log10(configuration.learning_rate) - log10(0.05))
        depth_penalty = abs(configuration.max_depth - 6) * 0.018
        estimator_penalty = abs(configuration.estimators - 200) / 10000

        score = (
            0.96
            - lr_penalty * 0.09
            - depth_penalty
            - estimator_penalty
        )

        # Small deterministic noise simulates measurement variability.
        noise = self.random.uniform(-0.003, 0.003)
        score = max(0.0, min(1.0, score + noise))

        duration = (
            0.5
            + configuration.max_depth * 0.08
            + configuration.estimators * 0.004
        )

        return score, duration


class HyperparameterOptimizer:
    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: SyntheticModelEvaluator,
        seed: int = 42,
    ) -> None:
        search_space.validate()
        self.space = search_space
        self.evaluator = evaluator
        self.random = Random(seed)

    def generate_grid(self) -> list[Configuration]:
        return [
            Configuration(rate, depth, estimators)
            for rate, depth, estimators in product(
                self.space.learning_rates,
                self.space.max_depths,
                self.space.estimators,
            )
        ]

    def generate_random(self, trials: int) -> list[Configuration]:
        if trials < 1:
            raise ValueError("The number of random trials must be positive.")

        configurations: set[tuple[float, int, int]] = set()

        while len(configurations) < min(trials, self.space.size):
            configuration = Configuration(
                learning_rate=self.random.choice(self.space.learning_rates),
                max_depth=self.random.choice(self.space.max_depths),
                estimators=self.random.choice(self.space.estimators),
            )
            configurations.add(configuration.key())

        return [
            Configuration(rate, depth, estimators)
            for rate, depth, estimators in configurations
        ]

    def run(
        self,
        strategy: SearchStrategy,
        random_trials: int | None = None,
    ) -> OptimizationResult:
        if strategy is SearchStrategy.GRID:
            configurations = self.generate_grid()
        elif strategy is SearchStrategy.RANDOM:
            if random_trials is None:
                raise ValueError("random_trials is required for random search.")
            configurations = self.generate_random(random_trials)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        results: list[TrialResult] = []

        for configuration in configurations:
            try:
                score, duration = self.evaluator.evaluate(configuration)
                results.append(
                    TrialResult(
                        configuration=configuration,
                        score=score,
                        duration_units=duration,
                        strategy=strategy,
                    )
                )
            except ValueError as exc:
                print(f"Rejected configuration {configuration}: {exc}")

        return OptimizationResult(strategy=strategy, trials=results)


def summarize(result: OptimizationResult) -> None:
    best = result.best
    scores = [trial.score for trial in result.trials]

    print(f"\n{result.strategy.value.upper()} SEARCH")
    print("-" * 55)
    print(f"Trials:          {len(result.trials)}")
    print(f"Average score:   {result.average_score:.4f}")
    print(
        f"Score deviation: "
        f"{stdev(scores):.4f}" if len(scores) > 1 else "Score deviation: 0.0000"
    )
    print(f"Best score:      {best.score:.4f}")
    print(f"Learning rate:   {best.configuration.learning_rate}")
    print(f"Max depth:       {best.configuration.max_depth}")
    print(f"Estimators:      {best.configuration.estimators}")
    print(f"Duration units:  {best.duration_units:.2f}")


def validate_search_space(space: SearchSpace) -> None:
    space.validate()

    duplicate_rates = len(space.learning_rates) != len(set(space.learning_rates))
    duplicate_depths = len(space.max_depths) != len(set(space.max_depths))
    duplicate_estimators = len(space.estimators) != len(set(space.estimators))

    if duplicate_rates or duplicate_depths or duplicate_estimators:
        raise ValueError("Search-space dimensions must not contain duplicates.")


def run_assertions() -> None:
    space = SearchSpace(
        learning_rates=(0.005, 0.01, 0.05, 0.1, 0.2),
        max_depths=(3, 6, 9),
        estimators=(50, 100, 200, 400),
    )

    validate_search_space(space)

    assert space.size == 60

    optimizer = HyperparameterOptimizer(
        space,
        SyntheticModelEvaluator(seed=7),
        seed=7,
    )

    grid = optimizer.run(SearchStrategy.GRID)
    random_result = optimizer.run(SearchStrategy.RANDOM, random_trials=15)

    assert len(grid.trials) == 60
    assert len(random_result.trials) == 15
    assert 0.0 <= grid.best.score <= 1.0
    assert 0.0 <= random_result.best.score <= 1.0

    empty_space_error = False
    try:
        SearchSpace((), (3,), (100,)).validate()
    except ValueError:
        empty_space_error = True

    assert empty_space_error


def main() -> None:
    run_assertions()

    search_space = SearchSpace(
        learning_rates=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
        max_depths=(2, 4, 6, 8, 10),
        estimators=(50, 100, 200, 400),
    )

    optimizer = HyperparameterOptimizer(
        search_space,
        SyntheticModelEvaluator(seed=42),
        seed=42,
    )

    grid_result = optimizer.run(SearchStrategy.GRID)
    random_result = optimizer.run(SearchStrategy.RANDOM, random_trials=20)

    summarize(grid_result)
    summarize(random_result)

    print("\nCOMPARISON")
    print("-" * 55)
    print(f"Grid configurations available:   {search_space.size}")
    print(f"Random configurations evaluated: {len(random_result.trials)}")
    print(
        "Grid found a better configuration: "
        f"{grid_result.best.score >= random_result.best.score}"
    )

    # A production optimizer should persist trial metadata so that failed
    # evaluations, duplicate configurations, and expensive runs can be audited.
    print("\nTop three grid configurations:")
    for rank, trial in enumerate(
        sorted(grid_result.trials, key=lambda item: item.score, reverse=True)[:3],
        start=1,
    ):
        print(
            f"{rank}. lr={trial.configuration.learning_rate}, "
            f"depth={trial.configuration.max_depth}, "
            f"estimators={trial.configuration.estimators}, "
            f"score={trial.score:.4f}"
        )


if __name__ == "__main__":
    main()
