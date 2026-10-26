from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from pathlib import Path
from typing import Iterable, Sequence
import json
import random
import statistics


class ClusterMethod(Enum):
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"


@dataclass(frozen=True)
class Point:
    point_id: int
    x: float
    y: float
    label: str = ""

    def distance(self, other: "Point") -> float:
        return hypot(self.x - other.x, self.y - other.y)


@dataclass
class Cluster:
    cluster_id: int
    point_ids: list[int]
    children: tuple[int, int] | None = None


@dataclass
class DBSCANResult:
    labels: dict[int, int]
    core_points: set[int]
    noise_points: set[int]


class ClusteringError(ValueError):
    pass


def validate_points(points: Sequence[Point]) -> None:
    if not points:
        raise ClusteringError("At least one point is required.")

    ids = [p.point_id for p in points]
    if len(ids) != len(set(ids)):
        raise ClusteringError("Point IDs must be unique.")

    for point in points:
        if not all(map(lambda v: isinstance(v, (int, float)), (point.x, point.y))):
            raise ClusteringError(f"Invalid coordinates for point {point.point_id}.")
        if not all(abs(v) < 1e12 for v in (point.x, point.y)):
            raise ClusteringError(f"Coordinates are outside the supported range.")


def pairwise_distances(points: Sequence[Point]) -> dict[tuple[int, int], float]:
    validate_points(points)
    result: dict[tuple[int, int], float] = {}

    for i, first in enumerate(points):
        for second in points[i + 1:]:
            result[(first.point_id, second.point_id)] = first.distance(second)

    return result


def distance_by_id(a: Point, b: Point) -> float:
    return a.distance(b)


def average_linkage(left: Sequence[Point], right: Sequence[Point]) -> float:
    distances = [
        distance_by_id(a, b)
        for a in left
        for b in right
    ]
    return statistics.fmean(distances)


def complete_linkage(left: Sequence[Point], right: Sequence[Point]) -> float:
    return max(distance_by_id(a, b) for a in left for b in right)


def single_linkage(left: Sequence[Point], right: Sequence[Point]) -> float:
    return min(distance_by_id(a, b) for a in left for b in right)


def hierarchical_clustering(
    points: Sequence[Point],
    target_clusters: int = 2,
    linkage: str = "average",
) -> list[list[Point]]:
    """
    Agglomerative hierarchical clustering.

    The implementation starts with one cluster per point and repeatedly merges
    the two closest clusters. This gives O(n^3) worst-case behavior with the
    direct linkage calculation, which is suitable for an educational,
    inspectable implementation but not for very large datasets.
    """
    validate_points(points)

    if not 1 <= target_clusters <= len(points):
        raise ClusteringError("target_clusters must be between 1 and the number of points.")

    linkage_functions = {
        "single": single_linkage,
        "complete": complete_linkage,
        "average": average_linkage,
    }

    if linkage not in linkage_functions:
        raise ClusteringError("linkage must be single, complete, or average.")

    clusters: list[list[Point]] = [[point] for point in points]
    metric = linkage_functions[linkage]

    while len(clusters) > target_clusters:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = metric(clusters[i], clusters[j])
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        assert best_pair is not None
        left, right = best_pair

        clusters[left].extend(clusters[right])
        del clusters[right]

    return clusters


def dbscan(
    points: Sequence[Point],
    eps: float,
    min_samples: int,
) -> DBSCANResult:
    """
    Density-Based Spatial Clustering of Applications with Noise.

    A point is core when its eps-neighborhood contains at least min_samples
    points, including itself. Density-connected points are expanded into the
    same cluster. Points that never become density-connected are noise.
    """
    validate_points(points)

    if eps <= 0:
        raise ClusteringError("eps must be positive.")
    if min_samples < 1:
        raise ClusteringError("min_samples must be at least 1.")

    labels: dict[int, int] = {}
    point_by_id = {p.point_id: p for p in points}
    core_points: set[int] = set()
    noise_points: set[int] = set()

    def neighbors(point: Point) -> list[int]:
        return [
            candidate.point_id
            for candidate in points
            if point.distance(candidate) <= eps
        ]

    neighborhoods = {
        point.point_id: neighbors(point)
        for point in points
    }

    for point_id, ids in neighborhoods.items():
        if len(ids) >= min_samples:
            core_points.add(point_id)

    cluster_id = 0
    visited: set[int] = set()

    for point in points:
        if point.point_id in visited:
            continue

        visited.add(point.point_id)
        point_neighbors = neighborhoods[point.point_id]

        if len(point_neighbors) < min_samples:
            labels[point.point_id] = -1
            noise_points.add(point.point_id)
            continue

        labels[point.point_id] = cluster_id
        queue = list(point_neighbors)

        while queue:
            candidate_id = queue.pop(0)

            if candidate_id not in visited:
                visited.add(candidate_id)
                candidate_neighbors = neighborhoods[candidate_id]

                if len(candidate_neighbors) >= min_samples:
                    for neighbor_id in candidate_neighbors:
                        if neighbor_id not in queue:
                            queue.append(neighbor_id)

            if candidate_id not in labels or labels[candidate_id] == -1:
                labels[candidate_id] = cluster_id
                noise_points.discard(candidate_id)

        cluster_id += 1

    for point_id, label in labels.items():
        if label == -1:
            noise_points.add(point_id)

    # Materialize the mapping to prevent accidental use of invalid IDs.
    labels = {
        point_id: label
        for point_id, label in labels.items()
        if point_id in point_by_id
    }

    return DBSCANResult(labels, core_points, noise_points)


def silhouette_score(points: Sequence[Point], labels: dict[int, int]) -> float:
    """
    Computes the mean silhouette coefficient for non-noise points.

    Noise labels (-1) are ignored because DBSCAN noise is not a conventional
    cluster and forcing it into the metric can distort model comparison.
    """
    validate_points(points)
    grouped: dict[int, list[Point]] = {}

    for point in points:
        label = labels.get(point.point_id, -1)
        if label != -1:
            grouped.setdefault(label, []).append(point)

    if len(grouped) < 2:
        return 0.0

    scores: list[float] = []

    for point in points:
        label = labels.get(point.point_id, -1)
        if label == -1:
            continue

        own_cluster = grouped[label]

        if len(own_cluster) == 1:
            a = 0.0
        else:
            a = statistics.fmean(
                point.distance(other)
                for other in own_cluster
                if other.point_id != point.point_id
            )

        b = min(
            statistics.fmean(point.distance(other) for other in cluster)
            for other_label, cluster in grouped.items()
            if other_label != label
        )

        denominator = max(a, b)
        scores.append(0.0 if denominator == 0 else (b - a) / denominator)

    return statistics.fmean(scores) if scores else 0.0


def save_report(
    path: str | Path,
    points: Sequence[Point],
    hierarchical: list[list[Point]],
    dbscan_result: DBSCANResult,
) -> None:
    payload = {
        "point_count": len(points),
        "hierarchical": [
            [point.point_id for point in cluster]
            for cluster in hierarchical
        ],
        "dbscan": {
            "labels": dbscan_result.labels,
            "core_points": sorted(dbscan_result.core_points),
            "noise_points": sorted(dbscan_result.noise_points),
        },
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def build_demo_dataset() -> list[Point]:
    random.seed(7)

    points: list[Point] = []

    for i in range(12):
        points.append(
            Point(
                point_id=i,
                x=10 + random.uniform(-0.8, 0.8),
                y=10 + random.uniform(-0.8, 0.8),
                label="group-a",
            )
        )

    for i in range(12, 24):
        points.append(
            Point(
                point_id=i,
                x=20 + random.uniform(-0.8, 0.8),
                y=20 + random.uniform(-0.8, 0.8),
                label="group-b",
            )
        )

    # A sparse bridge and two isolated observations demonstrate why density
    # based clustering can differ from hierarchical clustering.
    points.extend(
        [
            Point(24, 15.0, 15.0, "bridge"),
            Point(25, 30.0, 3.0, "outlier"),
        ]
    )

    return points


def print_clusters(title: str, clusters: Iterable[Iterable[Point]]) -> None:
    print(f"\n{title}")
    for index, cluster in enumerate(clusters):
        ids = sorted(point.point_id for point in cluster)
        print(f"  Cluster {index}: {ids}")


def run_self_tests() -> None:
    points = [
        Point(1, 0, 0),
        Point(2, 0.1, 0.1),
        Point(3, 0.2, 0),
        Point(4, 10, 10),
        Point(5, 10.1, 10),
    ]

    hierarchical = hierarchical_clustering(points, 2)
    assert len(hierarchical) == 2

    result = dbscan(points, eps=0.35, min_samples=2)
    assert result.labels[1] >= 0
    assert result.labels[4] >= 0
    assert result.labels[1] == result.labels[2]
    assert result.labels[4] == result.labels[5]

    try:
        dbscan(points, eps=0, min_samples=2)
        raise AssertionError("Invalid eps was accepted.")
    except ClusteringError:
        pass

    duplicate_points = [Point(1, 0, 0), Point(1, 1, 1)]
    try:
        hierarchical_clustering(duplicate_points)
        raise AssertionError("Duplicate IDs were accepted.")
    except ClusteringError:
        pass


def main() -> None:
    run_self_tests()

    points = build_demo_dataset()

    print("Advanced Clustering Case Study")
    print(f"Points: {len(points)}")

    distances = pairwise_distances(points)
    print(f"Pairwise distances calculated: {len(distances)}")

    hierarchical = hierarchical_clustering(
        points,
        target_clusters=3,
        linkage="average",
    )
    print_clusters("Agglomerative Hierarchical Clustering", hierarchical)

    hierarchical_labels = {}
    for cluster_id, cluster in enumerate(hierarchical):
        for point in cluster:
            hierarchical_labels[point.point_id] = cluster_id

    print(
        "Hierarchical silhouette:",
        round(silhouette_score(points, hierarchical_labels), 4),
    )

    density_result = dbscan(
        points,
        eps=1.5,
        min_samples=4,
    )

    print("\nDBSCAN")
    grouped: dict[int, list[int]] = {}

    for point_id, cluster_id in density_result.labels.items():
        grouped.setdefault(cluster_id, []).append(point_id)

    for cluster_id in sorted(grouped):
        if cluster_id == -1:
            print(f"  Noise: {sorted(grouped[cluster_id])}")
        else:
            print(f"  Cluster {cluster_id}: {sorted(grouped[cluster_id])}")

    print("Core points:", sorted(density_result.core_points))
    print("Noise points:", sorted(density_result.noise_points))
    print(
        "DBSCAN silhouette:",
        round(silhouette_score(points, density_result.labels), 4),
    )

    save_report(
        "clustering_report.json",
        points,
        hierarchical,
        density_result,
    )

    print("\nReport written to clustering_report.json")


if __name__ == "__main__":
    main()
