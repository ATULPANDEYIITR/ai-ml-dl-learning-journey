#!/usr/bin/env python3
"""
Anomaly Detection Learning and Simulation
Algorithms:
- Isolation Forest
- Local Outlier Factor
- One-Class SVM

The implementation uses only the Python standard library. It provides:
- feature validation and standardization
- synthetic repository-operation telemetry
- three anomaly-detection algorithms implemented from scratch
- score normalization and thresholding
- precision/recall evaluation against known synthetic anomalies
- disagreement analysis between detectors
- model diagnostics and production-oriented validation

The algorithms are intentionally compact learning implementations rather than
drop-in replacements for optimized numerical libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random
import statistics
from typing import Iterable, Sequence


class DetectionError(Exception):
    """Raised when detector input or configuration is invalid."""


class DetectorName(Enum):
    ISOLATION_FOREST = "Isolation Forest"
    LOCAL_OUTLIER_FACTOR = "Local Outlier Factor"
    ONE_CLASS_SVM = "One-Class SVM"


@dataclass(frozen=True)
class Event:
    event_id: int
    features: tuple[float, ...]
    is_anomaly: bool
    description: str


@dataclass
class DetectionResult:
    detector: DetectorName
    scores: list[float]
    predictions: list[bool]
    threshold: float


def validate_matrix(rows: Sequence[Sequence[float]], min_rows: int = 2) -> None:
    if len(rows) < min_rows:
        raise DetectionError(f"At least {min_rows} rows are required.")
    width = len(rows[0])
    if width == 0:
        raise DetectionError("At least one feature is required.")
    for row in rows:
        if len(row) != width:
            raise DetectionError("All observations must have the same feature count.")
        if any(not math.isfinite(float(value)) for value in row):
            raise DetectionError("Features must contain only finite numeric values.")


class StandardScaler:
    """Feature-wise standardization without external dependencies."""

    def __init__(self) -> None:
        self.means: list[float] = []
        self.stds: list[float] = []

    def fit(self, rows: Sequence[Sequence[float]]) -> "StandardScaler":
        validate_matrix(rows)
        columns = list(zip(*rows))
        self.means = [statistics.fmean(column) for column in columns]
        self.stds = []
        for column, mean in zip(columns, self.means):
            variance = statistics.fmean((x - mean) ** 2 for x in column)
            self.stds.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        validate_matrix(rows)
        if not self.means:
            raise DetectionError("Scaler must be fitted before transform.")
        if len(rows[0]) != len(self.means):
            raise DetectionError("Feature count differs from fitted scaler.")
        return [
            [(float(value) - mean) / std for value, mean, std in zip(row, self.means, self.stds)]
            for row in rows
        ]

    def fit_transform(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return self.fit(rows).transform(rows)


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def quantile(values: Sequence[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise DetectionError("Invalid quantile request.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


class LocalOutlierFactor:
    """
    Compact LOF implementation.

    LOF compares an observation's local density with the densities of its
    nearest neighbors. Values above approximately 1 indicate lower local
    density than neighboring observations.
    """

    def __init__(self, n_neighbors: int = 8) -> None:
        if n_neighbors < 2:
            raise DetectionError("LOF requires at least two neighbors.")
        self.n_neighbors = n_neighbors
        self.training: list[list[float]] = []
        self.k_distance: list[float] = []
        self.lrd: list[float] = []

    def _neighbors(self, index: int) -> list[tuple[int, float]]:
        distances = [
            (j, euclidean(self.training[index], row))
            for j, row in enumerate(self.training)
            if j != index
        ]
        distances.sort(key=lambda item: item[1])
        return distances[: min(self.n_neighbors, len(distances))]

    def fit(self, rows: Sequence[Sequence[float]]) -> "LocalOutlierFactor":
        validate_matrix(rows)
        if len(rows) <= self.n_neighbors:
            raise DetectionError("LOF requires more observations than n_neighbors.")
        self.training = [list(row) for row in rows]

        neighbor_lists = [self._neighbors(i) for i in range(len(rows))]
        self.k_distance = [
            neighbors[-1][1] if neighbors else 0.0
            for neighbors in neighbor_lists
        ]

        self.lrd = []
        for i, neighbors in enumerate(neighbor_lists):
            reachability_sum = 0.0
            for j, distance in neighbors:
                reachability_sum += max(self.k_distance[j], distance)
            self.lrd.append(
                len(neighbors) / reachability_sum
                if reachability_sum > 1e-12
                else float("inf")
            )
        return self

    def score_samples(self, rows: Sequence[Sequence[float]]) -> list[float]:
        validate_matrix(rows)
        if not self.training:
            raise DetectionError("LOF must be fitted before scoring.")
        scores: list[float] = []

        for row in rows:
            distances = [
                (i, euclidean(row, train_row))
                for i, train_row in enumerate(self.training)
            ]
            distances.sort(key=lambda item: item[1])
            neighbors = distances[: min(self.n_neighbors, len(distances))]

            reachability_sum = 0.0
            neighbor_lrd_sum = 0.0
            for j, distance in neighbors:
                reachability_sum += max(self.k_distance[j], distance)
                neighbor_lrd_sum += self.lrd[j]

            if not neighbors or reachability_sum <= 1e-12:
                scores.append(1.0)
                continue

            lrd_query = len(neighbors) / reachability_sum
            average_neighbor_lrd = neighbor_lrd_sum / len(neighbors)
            score = average_neighbor_lrd / max(lrd_query, 1e-12)
            scores.append(score)

        return scores

    def predict(self, rows: Sequence[Sequence[float]], threshold: float = 1.5) -> DetectionResult:
        scores = self.score_samples(rows)
        predictions = [score >= threshold for score in scores]
        return DetectionResult(
            DetectorName.LOCAL_OUTLIER_FACTOR, scores, predictions, threshold
        )


@dataclass
class IsolationNode:
    feature: int | None = None
    split: float | None = None
    left: "IsolationNode | None" = None
    right: "IsolationNode | None" = None
    size: int = 0


def harmonic_number(n: int) -> float:
    if n <= 1:
        return 0.0
    return sum(1.0 / i for i in range(1, n))


def expected_path_length(n: int) -> float:
    if n <= 1:
        return 0.0
    return 2.0 * harmonic_number(n - 1) - (2.0 * (n - 1) / n)


class IsolationTree:
    def __init__(self, max_depth: int, rng: random.Random) -> None:
        self.max_depth = max_depth
        self.rng = rng
        self.root: IsolationNode | None = None

    def fit(self, rows: Sequence[Sequence[float]]) -> None:
        if not rows:
            raise DetectionError("Cannot train an isolation tree on empty data.")
        self.root = self._build(list(rows), 0)

    def _build(self, rows: list[Sequence[float]], depth: int) -> IsolationNode:
        node = IsolationNode(size=len(rows))

        if (
            depth >= self.max_depth
            or len(rows) <= 1
            or all(
                max(row[column] for row in rows) - min(row[column] for row in rows)
                <= 1e-12
                for column in range(len(rows[0]))
            )
        ):
            return node

        feature = self.rng.randrange(len(rows[0]))
        minimum = min(row[feature] for row in rows)
        maximum = max(row[feature] for row in rows)

        if minimum >= maximum:
            return node

        split = self.rng.uniform(minimum, maximum)
        left = [row for row in rows if row[feature] < split]
        right = [row for row in rows if row[feature] >= split]

        if not left or not right:
            return node

        node.feature = feature
        node.split = split
        node.left = self._build(left, depth + 1)
        node.right = self._build(right, depth + 1)
        return node

    def path_length(self, row: Sequence[float]) -> float:
        if self.root is None:
            raise DetectionError("Isolation tree is not fitted.")

        node = self.root
        depth = 0

        while node.feature is not None and node.left is not None and node.right is not None:
            depth += 1
            node = node.left if row[node.feature] < node.split else node.right

        return depth + expected_path_length(node.size)


class IsolationForest:
    """Random partitioning ensemble for detecting isolated observations."""

    def __init__(
        self,
        n_trees: int = 80,
        sample_size: int = 64,
        contamination: float = 0.08,
        seed: int = 42,
    ) -> None:
        if n_trees < 1:
            raise DetectionError("n_trees must be positive.")
        if sample_size < 2:
            raise DetectionError("sample_size must be at least two.")
        if not 0.0 < contamination < 0.5:
            raise DetectionError("contamination must be between 0 and 0.5.")
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.contamination = contamination
        self.seed = seed
        self.trees: list[IsolationTree] = []
        self._normalizer = 0.0

    def fit(self, rows: Sequence[Sequence[float]]) -> "IsolationForest":
        validate_matrix(rows)
        if len(rows) < 4:
            raise DetectionError("Isolation Forest requires at least four observations.")

        rng = random.Random(self.seed)
        actual_size = min(self.sample_size, len(rows))
        depth = math.ceil(math.log2(actual_size))
        self._normalizer = expected_path_length(actual_size)
        self.trees = []

        for _ in range(self.n_trees):
            sample = rng.sample(list(rows), actual_size)
            tree = IsolationTree(depth, rng)
            tree.fit(sample)
            self.trees.append(tree)

        return self

    def score_samples(self, rows: Sequence[Sequence[float]]) -> list[float]:
        validate_matrix(rows)
        if not self.trees:
            raise DetectionError("Isolation Forest must be fitted before scoring.")
        if self._normalizer <= 0:
            raise DetectionError("Invalid forest normalization constant.")

        scores = []
        for row in rows:
            average_path = statistics.fmean(
                tree.path_length(row) for tree in self.trees
            )
            # Higher values mean stronger evidence of isolation/anomaly.
            scores.append(2.0 ** (-average_path / self._normalizer))
        return scores

    def predict(
        self,
        rows: Sequence[Sequence[float]],
        threshold: float | None = None,
    ) -> DetectionResult:
        scores = self.score_samples(rows)
        if threshold is None:
            threshold = quantile(scores, 1.0 - self.contamination)
        predictions = [score >= threshold for score in scores]
        return DetectionResult(
            DetectorName.ISOLATION_FOREST, scores, predictions, threshold
        )


class OneClassSVM:
    """
    Compact kernel one-class classifier.

    This implementation uses a radial-basis similarity matrix and an iterative
    center estimation strategy. It is intended as an executable educational
    model of the decision boundary rather than a replacement for a numerical
    optimization library.
    """

    def __init__(
        self,
        gamma: float = 0.5,
        nu: float = 0.1,
        iterations: int = 40,
    ) -> None:
        if gamma <= 0:
            raise DetectionError("gamma must be positive.")
        if not 0.0 < nu <= 0.5:
            raise DetectionError("nu must be in (0, 0.5].")
        self.gamma = gamma
        self.nu = nu
        self.iterations = iterations
        self.center: list[float] = []
        self.scale = 1.0
        self.training: list[list[float]] = []
        self.threshold = 0.0

    def _kernel(self, a: Sequence[float], b: Sequence[float]) -> float:
        distance_squared = sum((x - y) ** 2 for x, y in zip(a, b))
        return math.exp(-self.gamma * distance_squared)

    def fit(self, rows: Sequence[Sequence[float]]) -> "OneClassSVM":
        validate_matrix(rows)
        self.training = [list(row) for row in rows]
        dimension = len(rows[0])

        # Robust center from coordinate medians.
        self.center = [
            statistics.median(row[column] for row in rows)
            for column in range(dimension)
        ]

        distances = [
            math.sqrt(sum((value - center) ** 2 for value, center in zip(row, self.center)))
            for row in rows
        ]
        self.scale = max(statistics.median(distances), 1e-6)

        # Calibrate the decision threshold from training similarity.
        similarities = []
        for row in rows:
            similarity = statistics.fmean(
                self._kernel(row, other) for other in rows
            )
            similarities.append(similarity)

        self.threshold = quantile(similarities, self.nu)
        return self

    def score_samples(self, rows: Sequence[Sequence[float]]) -> list[float]:
        validate_matrix(rows)
        if not self.training:
            raise DetectionError("One-Class SVM must be fitted before scoring.")

        scores = []
        for row in rows:
            similarity = statistics.fmean(
                self._kernel(row, other) for other in self.training
            )
            # Convert normality similarity into anomaly score.
            scores.append(1.0 - similarity)
        return scores

    def predict(
        self,
        rows: Sequence[Sequence[float]],
        threshold: float | None = None,
    ) -> DetectionResult:
        raw_scores = self.score_samples(rows)

        if threshold is None:
            normality = [1.0 - score for score in raw_scores]
            similarity_cutoff = quantile(normality, self.nu)
            threshold = 1.0 - similarity_cutoff

        predictions = [score >= threshold for score in raw_scores]
        return DetectionResult(
            DetectorName.ONE_CLASS_SVM, raw_scores, predictions, threshold
        )


def generate_events(seed: int = 7) -> list[Event]:
    rng = random.Random(seed)
    events: list[Event] = []

    # Features:
    # [changed_files, additions, deletions, review_latency_hours,
    #  failed_checks, files_touched_security_area]
    for event_id in range(1, 181):
        features = (
            max(1.0, rng.gauss(8, 3)),
            max(1.0, rng.gauss(120, 45)),
            max(0.0, rng.gauss(55, 20)),
            max(0.2, rng.gauss(18, 7)),
            max(0.0, rng.gauss(0.4, 0.8)),
            max(0.0, rng.gauss(1.0, 0.7)),
        )
        events.append(
            Event(event_id, tuple(features), False, "normal repository activity")
        )

    anomaly_profiles = [
        (181, (85, 2400, 1700, 1.0, 12, 11), "large security-sensitive change"),
        (182, (1, 2, 1, 220, 0, 0), "abnormally dormant review"),
        (183, (48, 1500, 120, 2, 15, 8), "rapid high-risk change"),
        (184, (3, 15, 4, 0.2, 14, 1), "high failure-rate automation"),
        (185, (75, 3100, 20, 3, 0, 12), "large asymmetric change"),
        (186, (2, 5, 2, 280, 1, 0), "extreme review latency"),
    ]

    for event_id, values, description in anomaly_profiles:
        events.append(Event(event_id, tuple(float(v) for v in values), True, description))

    rng.shuffle(events)
    return events


def confusion_metrics(actual: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float]:
    if len(actual) != len(predicted):
        raise DetectionError("Actual and predicted arrays must have equal length.")

    tp = sum(a and p for a, p in zip(actual, predicted))
    tn = sum(not a and not p for a, p in zip(actual, predicted))
    fp = sum(not a and p for a, p in zip(actual, predicted))
    fn = sum(a and not p for a, p in zip(actual, predicted))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (tp + tn) / len(actual)

    return {
        "TP": float(tp),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def print_result(result: DetectionResult, events: Sequence[Event]) -> None:
    actual = [event.is_anomaly for event in events]
    metrics = confusion_metrics(actual, result.predictions)

    print(f"\n{result.detector.value}")
    print("-" * 72)
    print(f"Threshold: {result.threshold:.4f}")
    print(
        "Precision={precision:.3f} Recall={recall:.3f} "
        "F1={f1:.3f} Accuracy={accuracy:.3f}".format(**metrics)
    )

    ranked = sorted(
        zip(events, result.scores),
        key=lambda item: item[1],
        reverse=True,
    )

    print("\nTop anomaly candidates:")
    for event, score in ranked[:8]:
        label = "KNOWN-ANOMALY" if event.is_anomaly else "NORMAL"
        print(
            f"  event={event.event_id:3d} score={score:.4f} "
            f"{label:14s} {event.description}"
        )


def disagreement_analysis(results: Sequence[DetectionResult], events: Sequence[Event]) -> None:
    print("\nDetector disagreement")
    print("-" * 72)

    for index, event in enumerate(events):
        decisions = [result.predictions[index] for result in results]
        if len(set(decisions)) > 1:
            states = ", ".join(
                f"{result.detector.name}={'ANOMALY' if result.predictions[index] else 'NORMAL'}"
                for result in results
            )
            print(f"event={event.event_id:3d} actual={event.is_anomaly} | {states}")


def main() -> None:
    events = generate_events()
    raw_features = [event.features for event in events]

    # Scaling matters especially for distance-based methods such as LOF and
    # kernel-based methods such as the RBF similarity used by One-Class SVM.
    scaler = StandardScaler()
    features = scaler.fit_transform(raw_features)

    forest = IsolationForest(
        n_trees=100,
        sample_size=64,
        contamination=0.05,
        seed=42,
    )
    lof = LocalOutlierFactor(n_neighbors=10)
    svm = OneClassSVM(gamma=0.18, nu=0.08)

    forest_result = forest.fit(features).predict(features)
    lof_result = lof.fit(features).predict(features, threshold=1.45)
    svm_result = svm.fit(features).predict(features)

    results = [forest_result, lof_result, svm_result]

    print("ANOMALY DETECTION CASE STUDY")
    print("=" * 72)
    print(f"Observations: {len(events)}")
    print(f"Known anomalies: {sum(event.is_anomaly for event in events)}")
    print("Features: changes, additions, deletions, latency, failed checks, security scope")

    for result in results:
        print_result(result, events)

    disagreement_analysis(results, events)

    # A consensus alert requires at least two detectors to agree. This reduces
    # dependence on one model's inductive bias but can miss subtle anomalies.
    consensus = [
        sum(result.predictions[index] for result in results) >= 2
        for index in range(len(events))
    ]
    consensus_metrics = confusion_metrics(
        [event.is_anomaly for event in events],
        consensus,
    )

    print("\nTwo-of-three consensus")
    print("-" * 72)
    print(
        "Precision={precision:.3f} Recall={recall:.3f} "
        "F1={f1:.3f} Accuracy={accuracy:.3f}".format(**consensus_metrics)
    )

    # Basic operational assertions provide executable regression checks.
    assert len(results) == 3
    assert all(len(result.scores) == len(events) for result in results)
    assert all(math.isfinite(score) for result in results for score in result.scores)
    assert all(0.0 <= metrics <= 1.0 for metrics in [
        consensus_metrics["precision"],
        consensus_metrics["recall"],
        consensus_metrics["f1"],
        consensus_metrics["accuracy"],
    ])

    print("\nValidation checks passed.")
    print("\nProduction considerations:")
    print("  - Train detectors on representative normal data.")
    print("  - Keep feature scaling parameters versioned with the detector.")
    print("  - Tune thresholds against operational false-positive cost.")
    print("  - Treat anomaly scores as investigation signals, not proof of abuse.")
    print("  - Log model version, feature schema, threshold, and decision timestamp.")
    print("  - Restrict sensitive repository telemetry to authorized operators.")


if __name__ == "__main__":
    main()
