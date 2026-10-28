#!/usr/bin/env python3
"""
Advanced Dimensionality Reduction: t-SNE, UMAP-style graph embedding,
and manifold-learning utilities.

Self-contained implementation using only the Python standard library.
The implementation is intended for small educational and diagnostic datasets.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


Vector = list[float]
Matrix = list[Vector]


class ReductionMethod(str, Enum):
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"


@dataclass
class ReductionConfig:
    dimensions: int = 2
    perplexity: float = 10.0
    neighbors: int = 10
    learning_rate: float = 100.0
    iterations: int = 500
    seed: int = 42
    min_dist: float = 0.1


@dataclass
class ReductionResult:
    method: ReductionMethod
    embedding: Matrix
    stress: float
    iterations: int


def validate_matrix(X: Matrix) -> tuple[int, int]:
    if not X:
        raise ValueError("Dataset cannot be empty.")
    if not all(row for row in X):
        raise ValueError("Every observation must contain at least one feature.")

    width = len(X[0])
    if width < 2:
        raise ValueError("At least two input dimensions are required.")

    if any(len(row) != width for row in X):
        raise ValueError("All observations must have the same number of features.")

    for row in X:
        for value in row:
            if not math.isfinite(value):
                raise ValueError("All feature values must be finite.")

    return len(X), width


def squared_distance(a: Vector, b: Vector) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def euclidean_distance(a: Vector, b: Vector) -> float:
    return math.sqrt(squared_distance(a, b))


def standardize(X: Matrix) -> Matrix:
    n, d = validate_matrix(X)
    means = [sum(row[j] for row in X) / n for j in range(d)]

    scales = []
    for j in range(d):
        variance = sum((row[j] - means[j]) ** 2 for row in X) / max(1, n - 1)
        scales.append(math.sqrt(variance) or 1.0)

    return [
        [(row[j] - means[j]) / scales[j] for j in range(d)]
        for row in X
    ]


def covariance_matrix(X: Matrix) -> Matrix:
    X = standardize(X)
    n, d = validate_matrix(X)
    return [
        [
            sum(X[k][i] * X[k][j] for k in range(n)) / max(1, n - 1)
            for j in range(d)
        ]
        for i in range(d)
    ]


def mat_vec(A: Matrix, v: Vector) -> Vector:
    return [sum(a * b for a, b in zip(row, v)) for row in A]


def normalize(v: Vector) -> Vector:
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-15:
        raise ValueError("Cannot normalize a near-zero vector.")
    return [x / norm for x in v]


def power_iteration(
    A: Matrix,
    iterations: int = 200,
    seed: int = 42,
) -> tuple[float, Vector]:
    rng = random.Random(seed)
    v = normalize([rng.uniform(-1, 1) for _ in A])

    for _ in range(iterations):
        candidate = normalize(mat_vec(A, v))
        if max(abs(a - b) for a, b in zip(candidate, v)) < 1e-10:
            v = candidate
            break
        v = candidate

    Av = mat_vec(A, v)
    eigenvalue = sum(a * b for a, b in zip(v, Av))
    return eigenvalue, v


def deflate(A: Matrix, eigenvalue: float, vector: Vector) -> Matrix:
    return [
        [
            A[i][j] - eigenvalue * vector[i] * vector[j]
            for j in range(len(A))
        ]
        for i in range(len(A))
    ]


def pca(X: Matrix, dimensions: int = 2, seed: int = 42) -> Matrix:
    X = standardize(X)
    n, d = validate_matrix(X)

    if not 1 <= dimensions <= d:
        raise ValueError("PCA dimensions must be between 1 and input dimensions.")

    covariance = covariance_matrix(X)
    components = []

    for component in range(dimensions):
        eigenvalue, vector = power_iteration(
            covariance,
            seed=seed + component,
        )
        components.append(vector)
        covariance = deflate(covariance, eigenvalue, vector)

    return [
        [sum(row[j] * component[j] for j in range(d))
         for component in components]
        for row in X
    ]


def pairwise_distances(X: Matrix) -> Matrix:
    n, _ = validate_matrix(X)
    distances = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            distance = squared_distance(X[i], X[j])
            distances[i][j] = distance
            distances[j][i] = distance

    return distances


def binary_search_perplexity(
    distances: Vector,
    target_perplexity: float,
    max_iterations: int = 50,
) -> Vector:
    target_entropy = math.log(target_perplexity)
    beta = 1.0
    low = -math.inf
    high = math.inf

    for _ in range(max_iterations):
        probabilities = [
            math.exp(-beta * distance)
            if distance > 0 else 0.0
            for distance in distances
        ]
        total = sum(probabilities)

        if total <= 1e-300:
            beta *= 0.5
            continue

        probabilities = [p / total for p in probabilities]
        entropy = -sum(
            p * math.log(max(p, 1e-300))
            for p in probabilities
            if p > 0
        )

        difference = entropy - target_entropy
        if abs(difference) < 1e-5:
            return probabilities

        if difference > 0:
            low = beta
            beta = beta * 2 if math.isinf(high) else (beta + high) / 2
        else:
            high = beta
            beta = beta / 2 if math.isinf(low) else (beta + low) / 2

    return probabilities


def conditional_probabilities(
    distances: Matrix,
    perplexity: float,
) -> Matrix:
    n = len(distances)
    result = [[0.0] * n for _ in range(n)]

    for i in range(n):
        row = [
            distances[i][j] if i != j else 0.0
            for j in range(n)
        ]
        probabilities = binary_search_perplexity(
            row,
            min(perplexity, max(2.0, n - 1)),
        )
        result[i] = probabilities

    return result


def symmetrize(probabilities: Matrix) -> Matrix:
    n = len(probabilities)
    return [
        [
            (probabilities[i][j] + probabilities[j][i]) / (2.0 * n)
            for j in range(n)
        ]
        for i in range(n)
    ]


def tsne(
    X: Matrix,
    config: ReductionConfig,
) -> Matrix:
    """
    Small t-SNE implementation using exact pairwise probabilities.

    Complexity is O(n^2 * iterations), so this implementation is intended
    for small datasets rather than large production workloads.
    """
    X = standardize(X)
    n, _ = validate_matrix(X)

    if config.dimensions != 2:
        raise ValueError("This t-SNE implementation supports exactly 2 output dimensions.")
    if config.perplexity <= 1 or config.perplexity >= n:
        raise ValueError("Perplexity must be greater than 1 and smaller than sample count.")

    distances = pairwise_distances(X)
    conditional = conditional_probabilities(distances, config.perplexity)
    probabilities = symmetrize(conditional)

    rng = random.Random(config.seed)
    Y = [[rng.gauss(0, 1e-3), rng.gauss(0, 1e-3)] for _ in range(n)]
    velocity = [[0.0, 0.0] for _ in range(n)]

    for iteration in range(config.iterations):
        q_num = [[0.0] * n for _ in range(n)]
        total_q = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                distance = squared_distance(Y[i], Y[j])
                value = 1.0 / (1.0 + distance)
                q_num[i][j] = value
                q_num[j][i] = value
                total_q += 2.0 * value

        gradients = [[0.0, 0.0] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                q = q_num[i][j] / max(total_q, 1e-12)
                multiplier = 4.0 * (probabilities[i][j] - q) * q_num[i][j]

                gradients[i][0] += multiplier * (Y[i][0] - Y[j][0])
                gradients[i][1] += multiplier * (Y[i][1] - Y[j][1])

        momentum = 0.5 if iteration < 100 else 0.8

        for i in range(n):
            for dimension in range(2):
                velocity[i][dimension] = (
                    momentum * velocity[i][dimension]
                    - config.learning_rate * gradients[i][dimension]
                )
                Y[i][dimension] += velocity[i][dimension]

        mean_x = sum(row[0] for row in Y) / n
        mean_y = sum(row[1] for row in Y) / n

        for row in Y:
            row[0] -= mean_x
            row[1] -= mean_y

    return Y


def nearest_neighbors(X: Matrix, k: int) -> list[list[int]]:
    n, _ = validate_matrix(X)
    if not 1 <= k < n:
        raise ValueError("Neighbor count must be between 1 and n-1.")

    distances = pairwise_distances(X)
    neighbors = []

    for i in range(n):
        ordered = sorted(
            ((distances[i][j], j) for j in range(n) if i != j),
            key=lambda item: item[0],
        )
        neighbors.append([j for _, j in ordered[:k]])

    return neighbors


def local_connectivity_weights(
    X: Matrix,
    neighbors: list[list[int]],
) -> Matrix:
    n, _ = validate_matrix(X)
    weights = [[0.0] * n for _ in range(n)]

    for i in range(n):
        neighbor_distances = [
            (euclidean_distance(X[i], X[j]), j)
            for j in neighbors[i]
        ]

        positive = [d for d, _ in neighbor_distances if d > 0]
        sigma = statistics.median(positive) if positive else 1.0
        sigma = max(sigma, 1e-9)

        for distance, j in neighbor_distances:
            if distance == 0:
                weight = 1.0
            else:
                weight = math.exp(-max(0.0, distance - sigma) / sigma)
            weights[i][j] = weight

    return weights


def umap_style_embedding(
    X: Matrix,
    config: ReductionConfig,
) -> Matrix:
    """
    Compact UMAP-style approximation.

    It constructs a weighted k-nearest-neighbor graph and optimizes a
    low-dimensional layout using attractive and repulsive forces.
    This is intentionally not a binary-compatible implementation of
    the reference UMAP library.
    """
    X = standardize(X)
    n, _ = validate_matrix(X)

    if config.dimensions != 2:
        raise ValueError("This UMAP-style implementation supports 2D output.")
    if not 1 <= config.neighbors < n:
        raise ValueError("neighbors must be between 1 and n-1.")
    if not 0 < config.min_dist:
        raise ValueError("min_dist must be positive.")

    neighbors = nearest_neighbors(X, config.neighbors)
    weights = local_connectivity_weights(X, neighbors)

    rng = random.Random(config.seed)
    Y = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)]

    for iteration in range(config.iterations):
        gradients = [[0.0, 0.0] for _ in range(n)]

        for i in range(n):
            for j in neighbors[i]:
                dx = Y[i][0] - Y[j][0]
                dy = Y[i][1] - Y[j][1]
                distance = math.sqrt(dx * dx + dy * dy) + 1e-9

                desired = config.min_dist
                attraction = weights[i][j] * (distance - desired)

                gradients[i][0] += attraction * dx / distance
                gradients[i][1] += attraction * dy / distance

        # Sample non-neighbors to approximate repulsive forces.
        samples = min(n * 3, 1000)
        for _ in range(samples):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j or j in neighbors[i]:
                continue

            dx = Y[i][0] - Y[j][0]
            dy = Y[i][1] - Y[j][1]
            distance_sq = dx * dx + dy * dy + 1e-6
            repulsion = 0.01 / distance_sq

            gradients[i][0] += repulsion * dx
            gradients[i][1] += repulsion * dy

        learning_rate = config.learning_rate / (1.0 + iteration / 100.0)

        for i in range(n):
            Y[i][0] -= learning_rate * 0.01 * gradients[i][0]
            Y[i][1] -= learning_rate * 0.01 * gradients[i][1]

    return Y


def trustworthiness(
    original: Matrix,
    embedding: Matrix,
    k: int = 5,
) -> float:
    n, _ = validate_matrix(original)
    if len(embedding) != n:
        raise ValueError("Embedding and source dataset must have equal row counts.")
    if not 1 <= k < n:
        raise ValueError("k must be between 1 and n-1.")

    original_distances = pairwise_distances(original)
    embedded_distances = pairwise_distances(embedding)

    original_neighbors = []
    embedded_neighbors = []

    for i in range(n):
        original_neighbors.append({
            j for _, j in sorted(
                (original_distances[i][j], j)
                for j in range(n) if i != j
            )[:k]
        })

        embedded_neighbors.append({
            j for _, j in sorted(
                (embedded_distances[i][j], j)
                for j in range(n) if i != j
            )[:k]
        })

    rank_penalty = 0.0

    for i in range(n):
        original_order = [
            j for _, j in sorted(
                (original_distances[i][j], j)
                for j in range(n) if i != j
            )
        ]
        rank = {j: position + 1 for position, j in enumerate(original_order)}

        for j in embedded_neighbors[i] - original_neighbors[i]:
            rank_penalty += rank[j] - k

    normalization = n * k * (2 * n - 3 * k - 1)
    if normalization <= 0:
        return 1.0

    return 1.0 - (2.0 * rank_penalty) / normalization


def embedding_stress(original: Matrix, embedding: Matrix) -> float:
    if len(original) != len(embedding):
        raise ValueError("Input and embedding must contain equal row counts.")

    n = len(original)
    if n < 2:
        return 0.0

    source_distances = pairwise_distances(original)
    target_distances = pairwise_distances(embedding)

    numerator = 0.0
    denominator = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            source = math.sqrt(source_distances[i][j])
            target = math.sqrt(target_distances[i][j])
            numerator += (source - target) ** 2
            denominator += source ** 2

    return math.sqrt(numerator / max(denominator, 1e-12))


def generate_manifold_dataset(
    samples_per_curve: int = 30,
    seed: int = 42,
) -> Matrix:
    rng = random.Random(seed)
    data = []

    for i in range(samples_per_curve):
        t = 4.0 * math.pi * i / max(1, samples_per_curve - 1)
        noise = rng.gauss(0, 0.05)
        data.append([
            t * math.cos(t) + noise,
            t * math.sin(t) + noise,
            t / (4.0 * math.pi) + noise,
            math.sin(t / 2.0) + noise,
        ])

    for i in range(samples_per_curve):
        t = 4.0 * math.pi * i / max(1, samples_per_curve - 1)
        noise = rng.gauss(0, 0.05)
        data.append([
            t * math.cos(t + math.pi) + noise,
            t * math.sin(t + math.pi) + noise,
            t / (4.0 * math.pi) + 1.5 + noise,
            math.cos(t / 2.0) + noise,
        ])

    return data


def save_embedding(
    path: Path,
    original: Matrix,
    embedding: Matrix,
) -> None:
    if len(original) != len(embedding):
        raise ValueError("Cannot save mismatched matrices.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "source_0", "source_1", "x", "y"])

        for index, (source, point) in enumerate(zip(original, embedding)):
            writer.writerow([
                index,
                source[0],
                source[1],
                point[0],
                point[1],
            ])


def run_experiment(method: ReductionMethod, data: Matrix, config: ReductionConfig) -> ReductionResult:
    if method is ReductionMethod.PCA:
        embedding = pca(data, config.dimensions, config.seed)
        iterations = 0
    elif method is ReductionMethod.TSNE:
        embedding = tsne(data, config)
        iterations = config.iterations
    elif method is ReductionMethod.UMAP:
        embedding = umap_style_embedding(data, config)
        iterations = config.iterations
    else:
        raise ValueError(f"Unsupported reduction method: {method}")

    return ReductionResult(
        method=method,
        embedding=embedding,
        stress=embedding_stress(data, embedding),
        iterations=iterations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PCA, t-SNE, and a compact UMAP-style manifold embedding."
    )
    parser.add_argument(
        "--method",
        choices=[method.value for method in ReductionMethod],
        default="tsne",
    )
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--perplexity", type=float, default=8.0)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("embedding.csv"))

    args = parser.parse_args()

    if args.samples < 5:
        raise ValueError("--samples must be at least 5.")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive.")

    data = generate_manifold_dataset(args.samples, args.seed)

    config = ReductionConfig(
        dimensions=2,
        perplexity=args.perplexity,
        neighbors=min(args.neighbors, len(data) - 1),
        learning_rate=args.learning_rate,
        iterations=args.iterations,
        seed=args.seed,
    )

    method = ReductionMethod(args.method)
    result = run_experiment(method, data, config)

    score = trustworthiness(data, result.embedding, k=min(5, len(data) - 1))
    save_embedding(args.output, data, result.embedding)

    print(f"method={result.method.value}")
    print(f"samples={len(data)}")
    print(f"iterations={result.iterations}")
    print(f"stress={result.stress:.6f}")
    print(f"trustworthiness@5={score:.6f}")
    print(f"output={args.output}")

    assert len(result.embedding) == len(data)
    assert all(len(row) == 2 for row in result.embedding)
    assert all(math.isfinite(value) for row in result.embedding for value in row)


if __name__ == "__main__":
    main()
