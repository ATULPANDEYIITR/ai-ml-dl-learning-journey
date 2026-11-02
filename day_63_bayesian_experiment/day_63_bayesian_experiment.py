#!/usr/bin/env python3
"""
Bayesian Optimization: surrogate model, acquisition function, and optimization strategy.

Self-contained implementation using only the Python standard library.
The example optimizes a noisy two-variable black-box objective with:
- Gaussian Process regression using an RBF kernel
- Expected Improvement acquisition
- random initialization
- grid-based acquisition optimization
- observation history and diagnostics
- input/output validation
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


class OptimizationDirection(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class Bounds:
    low: float
    high: float

    def validate(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("Bounds must be finite.")
        if self.low >= self.high:
            raise ValueError("Lower bound must be smaller than upper bound.")

    def clip(self, value: float) -> float:
        return min(self.high, max(self.low, value))


@dataclass(frozen=True)
class Observation:
    x: tuple[float, ...]
    y: float


class GaussianProcess:
    """Small GP regressor using Cholesky decomposition."""

    def __init__(
        self,
        length_scale: float = 0.8,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-5,
    ) -> None:
        if length_scale <= 0 or signal_variance <= 0 or noise_variance < 0:
            raise ValueError("Invalid Gaussian Process hyperparameters.")
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.x_train: list[tuple[float, ...]] = []
        self.alpha: list[float] = []
        self.cholesky: list[list[float]] = []

    def kernel(self, a: Sequence[float], b: Sequence[float]) -> float:
        squared_distance = sum((x - y) ** 2 for x, y in zip(a, b))
        return self.signal_variance * math.exp(
            -0.5 * squared_distance / (self.length_scale**2)
        )

    @staticmethod
    def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
        n = len(matrix)
        lower = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                value = matrix[i][j]
                for k in range(j):
                    value -= lower[i][k] * lower[j][k]

                if i == j:
                    if value <= 1e-12:
                        raise ValueError("Kernel matrix is not positive definite.")
                    lower[i][j] = math.sqrt(value)
                else:
                    lower[i][j] = value / lower[j][j]

        return lower

    @staticmethod
    def _forward_substitution(
        lower: list[list[float]], vector: list[float]
    ) -> list[float]:
        result = []
        for i in range(len(vector)):
            value = vector[i]
            for j in range(i):
                value -= lower[i][j] * result[j]
            result.append(value / lower[i][i])
        return result

    @staticmethod
    def _backward_substitution(
        lower: list[list[float]], vector: list[float]
    ) -> list[float]:
        result = [0.0] * len(vector)
        for i in range(len(vector) - 1, -1, -1):
            value = vector[i]
            for j in range(i + 1, len(vector)):
                value -= lower[j][i] * result[j]
            result[i] = value / lower[i][i]
        return result

    def fit(self, observations: Sequence[Observation]) -> None:
        if not observations:
            raise ValueError("At least one observation is required.")

        self.x_train = [obs.x for obs in observations]
        y = [obs.y for obs in observations]
        n = len(y)

        matrix = [
            [
                self.kernel(self.x_train[i], self.x_train[j])
                + (self.noise_variance if i == j else 0.0)
                + (1e-9 if i == j else 0.0)
                for j in range(n)
            ]
            for i in range(n)
        ]

        self.cholesky = self._cholesky(matrix)
        intermediate = self._forward_substitution(self.cholesky, y)
        self.alpha = self._backward_substitution(self.cholesky, intermediate)

    def predict(self, x: Sequence[float]) -> tuple[float, float]:
        if not self.x_train:
            raise RuntimeError("Gaussian Process has not been fitted.")

        k_star = [self.kernel(x, train_x) for train_x in self.x_train]
        mean = sum(k * a for k, a in zip(k_star, self.alpha))

        solved = self._forward_substitution(self.cholesky, k_star)
        variance = self.kernel(x, x) - sum(value * value for value in solved)
        variance = max(0.0, variance)

        return mean, math.sqrt(variance)


class AcquisitionFunction:
    """Expected Improvement for a maximization problem."""

    @staticmethod
    def _normal_pdf(z: float) -> float:
        return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def _normal_cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def expected_improvement(
        self,
        mean: float,
        standard_deviation: float,
        best_value: float,
        exploration: float = 0.01,
    ) -> float:
        if standard_deviation <= 1e-12:
            return 0.0

        improvement = mean - best_value - exploration
        z = improvement / standard_deviation

        return improvement * self._normal_cdf(z) + (
            standard_deviation * self._normal_pdf(z)
        )


class BayesianOptimizer:
    def __init__(
        self,
        bounds: Sequence[Bounds],
        objective,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        seed: int = 42,
    ) -> None:
        if not bounds:
            raise ValueError("At least one dimension is required.")

        for bound in bounds:
            bound.validate()

        self.bounds = list(bounds)
        self.objective = objective
        self.direction = direction
        self.rng = random.Random(seed)
        self.observations: list[Observation] = []
        self.gp = GaussianProcess()
        self.acquisition = AcquisitionFunction()

    def _transform_y(self, y: float) -> float:
        return y if self.direction is OptimizationDirection.MAXIMIZE else -y

    def _evaluate(self, x: tuple[float, ...]) -> Observation:
        y = float(self.objective(x))
        if not math.isfinite(y):
            raise ValueError(f"Objective returned a non-finite value for {x}.")
        observation = Observation(x=x, y=y)
        self.observations.append(observation)
        return observation

    def initialize(self, count: int) -> None:
        if count < 1:
            raise ValueError("Initialization count must be positive.")

        for _ in range(count):
            point = tuple(
                self.rng.uniform(bound.low, bound.high) for bound in self.bounds
            )
            self._evaluate(point)

    def _best_transformed_value(self) -> float:
        return max(self._transform_y(obs.y) for obs in self.observations)

    def _candidate_grid(self, points_per_dimension: int) -> Iterable[tuple[float, ...]]:
        if points_per_dimension < 2:
            raise ValueError("Grid must contain at least two points per dimension.")

        axes = []
        for bound in self.bounds:
            step = (bound.high - bound.low) / (points_per_dimension - 1)
            axes.append(
                [bound.low + i * step for i in range(points_per_dimension)]
            )

        def build(prefix: tuple[float, ...], dimension: int):
            if dimension == len(axes):
                yield prefix
                return
            for value in axes[dimension]:
                yield from build(prefix + (value,), dimension + 1)

        return build((), 0)

    def _next_point(self, points_per_dimension: int = 15) -> tuple[float, ...]:
        self.gp.fit(
            [
                Observation(obs.x, self._transform_y(obs.y))
                for obs in self.observations
            ]
        )

        best = self._best_transformed_value()
        best_point = None
        best_acquisition = -1.0

        observed = {obs.x for obs in self.observations}

        for candidate in self._candidate_grid(points_per_dimension):
            if candidate in observed:
                continue

            mean, std = self.gp.predict(candidate)
            score = self.acquisition.expected_improvement(mean, std, best)

            if score > best_acquisition:
                best_acquisition = score
                best_point = candidate

        if best_point is None:
            raise RuntimeError("No unexplored candidate remains.")

        return best_point

    def optimize(
        self,
        iterations: int,
        initial_points: int = 5,
        points_per_dimension: int = 15,
    ) -> Observation:
        if iterations < 1:
            raise ValueError("Iterations must be positive.")

        self.initialize(initial_points)

        for _ in range(iterations):
            point = self._next_point(points_per_dimension)
            self._evaluate(point)

        if self.direction is OptimizationDirection.MAXIMIZE:
            return max(self.observations, key=lambda obs: obs.y)

        return min(self.observations, key=lambda obs: obs.y)

    def save_history(self, path: str | Path) -> None:
        target = Path(path)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["iteration", "x1", "x2", "objective"])
            for index, observation in enumerate(self.observations, start=1):
                writer.writerow([index, *observation.x, observation.y])


def noisy_objective(x: Sequence[float]) -> float:
    """
    Black-box objective.

    The deterministic component has two competing peaks. The small noise
    term represents measurement or evaluation variability.
    """
    if len(x) != 2:
        raise ValueError("This objective requires exactly two variables.")

    x1, x2 = x
    deterministic = (
        1.8 * math.exp(-((x1 - 1.5) ** 2 + (x2 - 2.0) ** 2) / 1.4)
        + 1.25 * math.exp(-((x1 + 2.0) ** 2 + (x2 + 1.0) ** 2) / 0.8)
        - 0.08 * (x1**2 + x2**2)
    )

    # Fixed pseudo-noise keeps this executable and reproducible.
    pseudo_noise = 0.015 * math.sin(17.0 * x1 + 11.0 * x2)
    return deterministic + pseudo_noise


def validate_result(result: Observation, bounds: Sequence[Bounds]) -> None:
    assert len(result.x) == len(bounds)
    for value, bound in zip(result.x, bounds):
        assert bound.low <= value <= bound.high
    assert math.isfinite(result.y)


def main() -> None:
    bounds = [
        Bounds(-4.0, 4.0),
        Bounds(-4.0, 4.0),
    ]

    optimizer = BayesianOptimizer(
        bounds=bounds,
        objective=noisy_objective,
        direction=OptimizationDirection.MAXIMIZE,
        seed=42,
    )

    best = optimizer.optimize(
        iterations=18,
        initial_points=6,
        points_per_dimension=17,
    )

    validate_result(best, bounds)

    print("Bayesian Optimization")
    print("=====================")
    print(f"Observations: {len(optimizer.observations)}")
    print(f"Best x: ({best.x[0]:.6f}, {best.x[1]:.6f})")
    print(f"Best objective: {best.y:.6f}")

    transformed = [
        optimizer._transform_y(obs.y) for obs in optimizer.observations
    ]
    print(f"Objective range: {min(transformed):.6f} to {max(transformed):.6f}")
    print(
        f"Mean observed objective: "
        f"{statistics.fmean(obs.y for obs in optimizer.observations):.6f}"
    )

    history_file = Path("bayesian_optimization_history.csv")
    optimizer.save_history(history_file)
    print(f"History written to: {history_file.resolve()}")

    # A deliberately invalid configuration demonstrates validation behavior.
    try:
        Bounds(2.0, 2.0).validate()
    except ValueError as exc:
        print(f"Validation check: {exc}")


if __name__ == "__main__":
    main()
