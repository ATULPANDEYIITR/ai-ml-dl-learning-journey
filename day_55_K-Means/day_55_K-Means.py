#!/usr/bin/env python3
"""K-Means clustering with centroid updates, initialization, inertia, and K selection."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


Point = tuple[float, ...]


class Initialization(str, Enum):
    RANDOM = "random"
    KMEANS_PLUS_PLUS = "kmeans++"


class ClusteringError(ValueError):
    """Raised when clustering inputs or configuration are invalid."""


def validate_points(points: Sequence[Sequence[float]]) -> list[Point]:
    if not points:
        raise ClusteringError("At least one data point is required.")

    dimension = len(points[0])
    if dimension == 0:
        raise ClusteringError("Points must contain at least one feature.")

    result: list[Point] = []
    for index, point in enumerate(points):
        if len(point) != dimension:
            raise ClusteringError(f"Point {index} has inconsistent dimensions.")
        values = tuple(float(value) for value in point)
        if not all(math.isfinite(value) for value in values):
            raise ClusteringError(f"Point {index} contains a non-finite value.")
        result.append(values)

    return result


def squared_distance(a: Point, b: Point) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


@dataclass(frozen=True)
class ClusteringResult:
    centroids: tuple[Point, ...]
    labels: tuple[int, ...]
    inertia: float
    iterations: int
    converged: bool

    def cluster_sizes(self) -> tuple[int, ...]:
        return tuple(self.labels.count(i) for i in range(len(self.centroids)))


class KMeans:
    def __init__(
        self,
        k: int,
        *,
        initialization: Initialization = Initialization.KMEANS_PLUS_PLUS,
        max_iterations: int = 300,
        tolerance: float = 1e-6,
        random_state: int = 42,
        n_init: int = 5,
    ) -> None:
        if k < 1:
            raise ClusteringError("K must be at least 1.")
        if max_iterations < 1 or n_init < 1:
            raise ClusteringError("Iteration and restart counts must be positive.")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ClusteringError("Tolerance must be finite and non-negative.")

        self.k = k
        self.initialization = initialization
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.n_init = n_init
        self.result_: ClusteringResult | None = None

    def _initialize(self, points: list[Point], rng: random.Random) -> list[Point]:
        if self.k > len(points):
            raise ClusteringError("K cannot exceed the number of data points.")

        if self.initialization == Initialization.RANDOM:
            return [points[i] for i in rng.sample(range(len(points)), self.k)]

        # K-Means++ spreads initial centers according to squared distance.
        centers = [points[rng.randrange(len(points))]]

        while len(centers) < self.k:
            weights = [
                min(squared_distance(point, center) for center in centers)
                for point in points
            ]
            total = sum(weights)

            if total == 0:
                remaining = [p for p in points if p not in centers]
                centers.append(rng.choice(remaining or points))
                continue

            threshold = rng.random() * total
            cumulative = 0.0
            selected = points[-1]

            for point, weight in zip(points, weights):
                cumulative += weight
                if cumulative >= threshold:
                    selected = point
                    break

            centers.append(selected)

        return centers

    def _fit_once(
        self, points: list[Point], rng: random.Random
    ) -> ClusteringResult:
        centers = self._initialize(points, rng)
        labels = [-1] * len(points)
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            new_labels = [
                min(
                    range(self.k),
                    key=lambda cluster: squared_distance(point, centers[cluster]),
                )
                for point in points
            ]

            sums = [[0.0] * len(points[0]) for _ in range(self.k)]
            counts = [0] * self.k

            for point, label in zip(points, new_labels):
                counts[label] += 1
                for feature, value in enumerate(point):
                    sums[label][feature] += value

            new_centers: list[Point] = []

            for cluster in range(self.k):
                if counts[cluster]:
                    new_centers.append(
                        tuple(value / counts[cluster] for value in sums[cluster])
                    )
                else:
                    # Empty clusters are reseeded with a point far from its
                    # nearest current center.
                    candidate = max(
                        points,
                        key=lambda p: min(
                            squared_distance(p, center) for center in centers
                        ),
                    )
                    new_centers.append(candidate)

            shift = max(
                squared_distance(old, new)
                for old, new in zip(centers, new_centers)
            )

            stable_labels = new_labels == labels
            centers = new_centers
            labels = new_labels

            if stable_labels or shift <= self.tolerance**2:
                converged = True
                break

        # Reassign after the final centroid update so labels match the centers.
        labels = [
            min(
                range(self.k),
                key=lambda cluster: squared_distance(point, centers[cluster]),
            )
            for point in points
        ]
        inertia = sum(
            squared_distance(point, centers[label])
            for point, label in zip(points, labels)
        )

        return ClusteringResult(
            centroids=tuple(centers),
            labels=tuple(labels),
            inertia=inertia,
            iterations=iteration,
            converged=converged,
        )

    def fit(self, raw_points: Sequence[Sequence[float]]) -> ClusteringResult:
        points = validate_points(raw_points)
        if self.k > len(points):
            raise ClusteringError("K cannot exceed the number of data points.")

        best: ClusteringResult | None = None
        master_rng = random.Random(self.random_state)

        for _ in range(self.n_init):
            rng = random.Random(master_rng.getrandbits(64))
            candidate = self._fit_once(points, rng)
            if best is None or candidate.inertia < best.inertia:
                best = candidate

        assert best is not None
        self.result_ = best
        return best

    def predict(self, raw_points: Sequence[Sequence[float]]) -> list[int]:
        if self.result_ is None:
            raise RuntimeError("Call fit before predict.")

        points = validate_points(raw_points)
        dimension = len(self.result_.centroids[0])
        if any(len(point) != dimension for point in points):
            raise ClusteringError("Prediction dimensions do not match training data.")

        return [
            min(
                range(self.k),
                key=lambda cluster: squared_distance(
                    point, self.result_.centroids[cluster]
                ),
            )
            for point in points
        ]


def choose_k(
    raw_points: Sequence[Sequence[float]],
    candidates: Sequence[int],
    *,
    random_state: int = 42,
) -> list[tuple[int, float]]:
    """Return K and inertia for an elbow-plot inspection."""
    points = validate_points(raw_points)
    if not candidates:
        raise ClusteringError("At least one candidate K is required.")
    if any(k < 1 or k > len(points) for k in candidates):
        raise ClusteringError("Each candidate K must be between 1 and sample count.")

    results = []
    for k in candidates:
        model = KMeans(k, random_state=random_state, n_init=5)
        result = model.fit(points)
        results.append((k, result.inertia))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Synthetic customer segments: low, medium, and high spending/activity.
    points = [
        (1.0, 1.2), (1.3, 0.8), (0.7, 1.1), (1.2, 1.5),
        (5.0, 5.2), (5.4, 4.8), (4.7, 5.1), (5.2, 5.5),
        (9.0, 1.0), (9.4, 1.4), (8.6, 0.7), (9.2, 0.6),
    ]

    model = KMeans(args.k, random_state=args.seed, n_init=10)
    result = model.fit(points)

    print(f"K: {args.k}")
    print(f"Inertia: {result.inertia:.4f}")
    print(f"Iterations: {result.iterations}")
    print(f"Converged: {result.converged}")
    print(f"Cluster sizes: {result.cluster_sizes()}")

    for index, centroid in enumerate(result.centroids):
        print(f"Centroid {index}: {centroid}")

    print("\nElbow candidates:")
    for k, inertia in choose_k(points, range(1, min(6, len(points)) + 1),
                               random_state=args.seed):
        print(f"K={k}: inertia={inertia:.4f}")

    assert len(result.labels) == len(points)
    assert math.isfinite(result.inertia)
    assert all(0 <= label < args.k for label in result.labels)


if __name__ == "__main__":
    main()
