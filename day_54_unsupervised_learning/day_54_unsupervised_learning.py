from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from statistics import mean
from typing import Iterable


class ClusterMethod(str, Enum):
    K_MEANS = "k-means"


class DistanceMetric(str, Enum):
    EUCLIDEAN = "euclidean"


@dataclass(frozen=True)
class Point:
    identifier: str
    features: tuple[float, ...]


@dataclass
class Cluster:
    identifier: int
    centroid: list[float]
    members: list[Point] = field(default_factory=list)


@dataclass(frozen=True)
class Anomaly:
    point_id: str
    distance: float
    threshold: float


class ValidationError(ValueError):
    pass


def validate_points(points: Iterable[Point]) -> list[Point]:
    materialized = list(points)

    if not materialized:
        raise ValidationError("At least one observation is required.")

    dimensions = len(materialized[0].features)
    if dimensions == 0:
        raise ValidationError("Each observation needs at least one feature.")

    identifiers: set[str] = set()

    for point in materialized:
        if not point.identifier:
            raise ValidationError("Observation identifiers cannot be empty.")
        if point.identifier in identifiers:
            raise ValidationError(f"Duplicate observation: {point.identifier}")
        if len(point.features) != dimensions:
            raise ValidationError("All observations must have equal dimensionality.")
        if any(not isinstance(value, (int, float)) for value in point.features):
            raise ValidationError("Features must be numeric.")
        identifiers.add(point.identifier)

    return materialized


def euclidean(a: Iterable[float], b: Iterable[float]) -> float:
    values_a = list(a)
    values_b = list(b)

    if len(values_a) != len(values_b):
        raise ValidationError("Distance vectors must have equal dimensions.")

    return sqrt(sum((x - y) ** 2 for x, y in zip(values_a, values_b)))


def standardize(points: list[Point]) -> list[Point]:
    """
    Standardization prevents a high-scale feature from dominating distance.
    Constant features receive a zero standardized value.
    """
    dimensions = len(points[0].features)
    columns = [[point.features[i] for point in points] for i in range(dimensions)]

    means = [mean(column) for column in columns]
    stds = []

    for column, column_mean in zip(columns, means):
        variance = sum((value - column_mean) ** 2 for value in column) / len(column)
        stds.append(sqrt(variance))

    result = []
    for point in points:
        values = tuple(
            0.0 if std == 0 else (value - column_mean) / std
            for value, column_mean, std in zip(point.features, means, stds)
        )
        result.append(Point(point.identifier, values))

    return result


class KMeans:
    def __init__(self, k: int, max_iterations: int = 100, tolerance: float = 1e-6):
        if k < 1:
            raise ValidationError("k must be positive.")
        if max_iterations < 1:
            raise ValidationError("max_iterations must be positive.")
        if tolerance < 0:
            raise ValidationError("tolerance cannot be negative.")

        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.clusters: list[Cluster] = []

    def fit(self, points: list[Point]) -> list[Cluster]:
        points = validate_points(points)

        if self.k > len(points):
            raise ValidationError("k cannot exceed the number of observations.")

        dimensions = len(points[0].features)

        # Deterministic initialization makes this learning artifact reproducible.
        centroids = [list(points[i].features) for i in range(self.k)]

        for _ in range(self.max_iterations):
            assignments: list[list[Point]] = [[] for _ in range(self.k)]

            for point in points:
                distances = [euclidean(point.features, centroid) for centroid in centroids]
                assignments[distances.index(min(distances))].append(point)

            new_centroids: list[list[float]] = []

            for index, members in enumerate(assignments):
                if not members:
                    # Empty clusters retain their previous centroid rather than
                    # silently changing k or creating an invalid state.
                    new_centroids.append(centroids[index])
                    continue

                new_centroids.append([
                    sum(point.features[d] for point in members) / len(members)
                    for d in range(dimensions)
                ])

            movement = max(
                euclidean(old, new)
                for old, new in zip(centroids, new_centroids)
            )

            centroids = new_centroids

            if movement <= self.tolerance:
                break

        self.clusters = [
            Cluster(index, centroids[index], assignments[index])
            for index in range(self.k)
        ]

        return self.clusters

    def inertia(self) -> float:
        if not self.clusters:
            raise ValidationError("Fit the model before calculating inertia.")

        return sum(
            euclidean(point.features, cluster.centroid) ** 2
            for cluster in self.clusters
            for point in cluster.members
        )


class PCA2D:
    """
    Small dependency-free principal-component approximation.

    The implementation uses power iteration to estimate the first principal
    direction, removes that component, and estimates the second direction.
    It is intended for small educational datasets rather than large-scale
    numerical workloads.
    """

    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        magnitude = sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            raise ValidationError("Cannot normalize a zero vector.")
        return [value / magnitude for value in vector]

    @staticmethod
    def _covariance(points: list[Point]) -> list[list[float]]:
        dimensions = len(points[0].features)
        means = [
            sum(point.features[d] for point in points) / len(points)
            for d in range(dimensions)
        ]

        covariance = [[0.0] * dimensions for _ in range(dimensions)]
        denominator = max(1, len(points) - 1)

        for point in points:
            centered = [point.features[d] - means[d] for d in range(dimensions)]
            for i in range(dimensions):
                for j in range(dimensions):
                    covariance[i][j] += centered[i] * centered[j] / denominator

        return covariance

    @staticmethod
    def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
        return [
            sum(matrix[row][column] * vector[column] for column in range(len(vector)))
            for row in range(len(matrix))
        ]

    def _power_iteration(self, matrix: list[list[float]]) -> list[float]:
        vector = self._normalize([1.0 + index for index in range(len(matrix))])

        for _ in range(self.iterations):
            vector = self._normalize(self._matvec(matrix, vector))

        return vector

    def fit_transform(self, points: list[Point]) -> list[tuple[str, float, float]]:
        points = validate_points(points)
        if len(points[0].features) < 2:
            raise ValidationError("Dimensionality reduction needs at least two features.")

        covariance = self._covariance(points)
        first = self._power_iteration(covariance)

        # Deflate the covariance matrix to obtain a second direction.
        eigenvalue = sum(
            first[i] * self._matvec(covariance, first)[i]
            for i in range(len(first))
        )

        second_matrix = [
            [
                covariance[i][j] - eigenvalue * first[i] * first[j]
                for j in range(len(first))
            ]
            for i in range(len(first))
        ]

        second = self._power_iteration(second_matrix)

        means = [
            sum(point.features[d] for point in points) / len(points[0].features)
            for d in range(len(points[0].features))
        ]

        transformed = []
        for point in points:
            centered = [value - means[index] for index, value in enumerate(point.features)]
            x = sum(a * b for a, b in zip(centered, first))
            y = sum(a * b for a, b in zip(centered, second))
            transformed.append((point.identifier, x, y))

        return transformed


def detect_anomalies(
    points: list[Point],
    threshold_multiplier: float = 2.5,
) -> list[Anomaly]:
    """
    Uses distance from the global centroid and mean-plus-multiplier*population
    standard deviation as a simple unsupervised anomaly rule.
    """
    points = validate_points(points)
    if threshold_multiplier < 0:
        raise ValidationError("threshold_multiplier cannot be negative.")

    dimensions = len(points[0].features)
    centroid = [
        sum(point.features[d] for point in points) / len(points)
        for d in range(dimensions)
    ]

    distances = [
        euclidean(point.features, centroid)
        for point in points
    ]

    average_distance = mean(distances)
    std = sqrt(
        sum((distance - average_distance) ** 2 for distance in distances)
        / len(distances)
    )
    threshold = average_distance + threshold_multiplier * std

    return [
        Anomaly(point.identifier, distance, threshold)
        for point, distance in zip(points, distances)
        if distance > threshold
    ]


def print_clusters(clusters: list[Cluster]) -> None:
    for cluster in clusters:
        members = ", ".join(point.identifier for point in cluster.members)
        centroid = ", ".join(f"{value:.3f}" for value in cluster.centroid)
        print(f"Cluster {cluster.identifier}: [{members}] centroid=({centroid})")


def main() -> None:
    raw_points = [
        Point("customer-01", (1.0, 1.2, 0.9)),
        Point("customer-02", (1.1, 0.8, 1.0)),
        Point("customer-03", (0.9, 1.0, 1.2)),
        Point("customer-04", (7.8, 8.1, 7.5)),
        Point("customer-05", (8.2, 7.9, 8.4)),
        Point("customer-06", (7.5, 8.3, 8.0)),
        Point("customer-07", (4.2, 4.0, 4.1)),
        Point("customer-08", (4.5, 4.3, 4.2)),
        Point("customer-09", (4.0, 4.4, 4.5)),
        Point("customer-10", (25.0, 1.0, 0.5)),
    ]

    print("UNSUPERVISED LEARNING WORKFLOW")
    print("=" * 32)

    normalized = standardize(raw_points)

    clustering = KMeans(k=3, max_iterations=100, tolerance=1e-7)
    clusters = clustering.fit(normalized)

    print("\nClusters")
    print("-" * 32)
    print_clusters(clusters)
    print(f"Inertia: {clustering.inertia():.4f}")

    print("\nTwo-dimensional representation")
    print("-" * 32)
    projection = PCA2D().fit_transform(normalized)
    for identifier, x, y in projection:
        print(f"{identifier:12} PC1={x:8.3f} PC2={y:8.3f}")

    print("\nAnomalies")
    print("-" * 32)
    anomalies = detect_anomalies(normalized, threshold_multiplier=2.0)
    if not anomalies:
        print("No observations exceeded the anomaly threshold.")
    else:
        for anomaly in anomalies:
            print(
                f"{anomaly.point_id}: distance={anomaly.distance:.3f}, "
                f"threshold={anomaly.threshold:.3f}"
            )

    # Assertions act as lightweight executable quality checks.
    assert len(clusters) == 3
    assert sum(len(cluster.members) for cluster in clusters) == len(raw_points)
    assert len(projection) == len(raw_points)


if __name__ == "__main__":
    main()
