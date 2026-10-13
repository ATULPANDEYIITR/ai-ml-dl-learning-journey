"""
k_nearest_neighbors.py

A self-contained implementation of k-Nearest Neighbors (k-NN) for
classification and regression.

Demonstrates:
- Euclidean, Manhattan, Minkowski, Chebyshev, and cosine distance.
- Feature scaling and distance sensitivity.
- Neighborhood selection and deterministic tie-breaking.
- Classification, probability estimation, and regression.
- Uniform and distance-weighted voting.
- Missing-value, dimensionality, and parameter validation.
- Train/test evaluation, confusion matrices, and regression metrics.
- Cross-validation and hyperparameter selection.
- Computational complexity and practical deployment considerations.

Run with Python 3.10 or later. No external packages are required.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


Vector = Sequence[float]
DistanceFunction = Callable[[Vector, Vector], float]


def validate_vector(vector: Vector, name: str = "vector") -> None:
    """Reject empty, nonnumeric, nonfinite, or inconsistent feature vectors."""
    if not vector:
        raise ValueError(f"{name} must not be empty.")

    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must contain numeric values.")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must contain finite values.")


def validate_dataset(
    X: Sequence[Vector],
    y: Sequence[Any] | None = None,
) -> int:
    """Validate feature dimensions and optional target alignment."""
    if not X:
        raise ValueError("The dataset must contain at least one sample.")

    expected_dimensions = len(X[0])
    if expected_dimensions == 0:
        raise ValueError("Samples must contain at least one feature.")

    for index, sample in enumerate(X):
        validate_vector(sample, f"X[{index}]")
        if len(sample) != expected_dimensions:
            raise ValueError("All samples must have the same feature count.")

    if y is not None:
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if any(target is None for target in y):
            raise ValueError("Targets must not be None.")

    return expected_dimensions


def euclidean_distance(a: Vector, b: Vector) -> float:
    """Straight-line distance; large coordinate differences are penalized."""
    validate_vector(a, "a")
    validate_vector(b, "b")
    if len(a) != len(b):
        raise ValueError("Distance operands must have equal dimensions.")
    return math.sqrt(math.fsum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def manhattan_distance(a: Vector, b: Vector) -> float:
    """Sum of absolute coordinate differences."""
    validate_vector(a, "a")
    validate_vector(b, "b")
    if len(a) != len(b):
        raise ValueError("Distance operands must have equal dimensions.")
    return math.fsum(abs(float(x) - float(y)) for x, y in zip(a, b))


def minkowski_distance(a: Vector, b: Vector, p: float = 3.0) -> float:
    """Generalized L-p distance; p=1 is Manhattan and p=2 is Euclidean."""
    if not math.isfinite(p) or p < 1:
        raise ValueError("Minkowski p must be finite and at least 1.")
    validate_vector(a, "a")
    validate_vector(b, "b")
    if len(a) != len(b):
        raise ValueError("Distance operands must have equal dimensions.")
    return math.fsum(
        abs(float(x) - float(y)) ** p for x, y in zip(a, b)
    ) ** (1.0 / p)


def chebyshev_distance(a: Vector, b: Vector) -> float:
    """Largest absolute coordinate difference."""
    validate_vector(a, "a")
    validate_vector(b, "b")
    if len(a) != len(b):
        raise ValueError("Distance operands must have equal dimensions.")
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def cosine_distance(a: Vector, b: Vector) -> float:
    """
    One minus cosine similarity.

    A zero vector has undefined cosine direction and is rejected rather
    than silently assigned an arbitrary distance.
    """
    validate_vector(a, "a")
    validate_vector(b, "b")
    if len(a) != len(b):
        raise ValueError("Distance operands must have equal dimensions.")

    dot = math.fsum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(math.fsum(float(x) ** 2 for x in a))
    norm_b = math.sqrt(math.fsum(float(y) ** 2 for y in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine distance is undefined for zero vectors.")

    similarity = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return 1.0 - similarity


DISTANCES: dict[str, DistanceFunction] = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "chebyshev": chebyshev_distance,
    "cosine": cosine_distance,
}


class StandardScaler:
    """Scale each feature using training-set means and population deviations."""

    def __init__(self) -> None:
        self.means: list[float] = []
        self.scales: list[float] = []
        self.fitted = False

    def fit(self, X: Sequence[Vector]) -> StandardScaler:
        dimensions = validate_dataset(X)
        self.means = []
        self.scales = []

        for column in range(dimensions):
            values = [float(row[column]) for row in X]
            mean = statistics.fmean(values)
            variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
            deviation = math.sqrt(variance)

            self.means.append(mean)
            # Constant features cannot be standardized by division by zero.
            self.scales.append(deviation if deviation > 0 else 1.0)

        self.fitted = True
        return self

    def transform(self, X: Sequence[Vector]) -> list[list[float]]:
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transformation.")

        dimensions = validate_dataset(X)
        if dimensions != len(self.means):
            raise ValueError("Feature count differs from the fitted scaler.")

        return [
            [
                (float(value) - self.means[column]) / self.scales[column]
                for column, value in enumerate(row)
            ]
            for row in X
        ]

    def fit_transform(self, X: Sequence[Vector]) -> list[list[float]]:
        return self.fit(X).transform(X)


@dataclass(frozen=True)
class Neighbor:
    index: int
    distance: float
    target: Any


class KNearestNeighbors:
    """
    Brute-force k-NN estimator supporting classification and regression.

    Ties in classification are resolved by the total neighbor weight, then
    by the sum of distances for each label, then by the label's string form.
    Exact zero-distance neighbors exclude nonzero-distance neighbors from
    weighted predictions, preventing arbitrary near-duplicate influence.
    """

    def __init__(
        self,
        k: int = 3,
        task: str = "classification",
        metric: str = "euclidean",
        weights: str = "uniform",
        minkowski_p: float = 3.0,
        distance_epsilon: float = 1e-12,
    ) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression.")
        if metric not in DISTANCES and metric != "minkowski":
            raise ValueError(f"Unsupported metric: {metric}")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be uniform or distance.")
        if not math.isfinite(distance_epsilon) or distance_epsilon <= 0:
            raise ValueError("distance_epsilon must be positive and finite.")
        if not math.isfinite(minkowski_p) or minkowski_p < 1:
            raise ValueError("minkowski_p must be finite and at least 1.")

        self.k = k
        self.task = task
        self.metric = metric
        self.weights = weights
        self.minkowski_p = minkowski_p
        self.distance_epsilon = distance_epsilon
        self.X_train: list[list[float]] = []
        self.y_train: list[Any] = []
        self.n_features = 0
        self.classes: list[Any] = []
        self.fitted = False

    def _distance(self, a: Vector, b: Vector) -> float:
        if self.metric == "minkowski":
            result = minkowski_distance(a, b, self.minkowski_p)
        else:
            result = DISTANCES[self.metric](a, b)

        if not math.isfinite(result) or result < 0:
            raise ValueError("Distance function returned an invalid distance.")
        return result

    def fit(
        self,
        X: Sequence[Vector],
        y: Sequence[Any],
    ) -> KNearestNeighbors:
        self.n_features = validate_dataset(X, y)

        if self.k > len(X):
            raise ValueError("k cannot exceed the training sample count.")

        if self.task == "classification":
            try:
                unique_classes = set(y)
            except TypeError as exc:
                raise TypeError("Classification labels must be hashable.") from exc
            self.classes = sorted(unique_classes, key=lambda label: str(label))
        else:
            numeric_targets: list[float] = []
            for target in y:
                if isinstance(target, bool) or not isinstance(target, (int, float)):
                    raise TypeError("Regression targets must be numeric.")
                numeric_target = float(target)
                if not math.isfinite(numeric_target):
                    raise ValueError("Regression targets must be finite.")
                numeric_targets.append(numeric_target)
            self.y_train = numeric_targets

        self.X_train = [[float(value) for value in row] for row in X]
        if self.task == "classification":
            self.y_train = list(y)

        self.fitted = True
        return self

    def kneighbors(self, query: Vector, k: int | None = None) -> list[Neighbor]:
        if not self.fitted:
            raise RuntimeError("Fit the estimator before querying neighbors.")

        validate_vector(query, "query")
        if len(query) != self.n_features:
            raise ValueError("Query feature count does not match training data.")

        neighbor_count = self.k if k is None else k
        if (
            isinstance(neighbor_count, bool)
            or not isinstance(neighbor_count, int)
            or not 1 <= neighbor_count <= len(self.X_train)
        ):
            raise ValueError("Requested neighbor count is outside valid bounds.")

        distances = [
            Neighbor(index, self._distance(query, sample), self.y_train[index])
            for index, sample in enumerate(self.X_train)
        ]

        # Sorting by original index gives reproducible results for equal distances.
        distances.sort(key=lambda neighbor: (neighbor.distance, neighbor.index))
        return distances[:neighbor_count]

    def _neighbor_weight(self, distance: float) -> float:
        if self.weights == "uniform":
            return 1.0
        if distance == 0:
            return 1.0
        return 1.0 / max(distance, self.distance_epsilon)

    def predict_proba(self, X: Sequence[Vector]) -> list[dict[Any, float]]:
        if self.task != "classification":
            raise RuntimeError("predict_proba is available for classification.")
        validate_dataset(X)

        results: list[dict[Any, float]] = []

        for query in X:
            neighbors = self.kneighbors(query)
            exact_matches = [n for n in neighbors if n.distance == 0]

            # If a query exactly matches training records, use those records
            # alone in weighted voting to avoid influence from unrelated points.
            if self.weights == "distance" and exact_matches:
                neighbors = exact_matches

            totals: dict[Any, float] = defaultdict(float)
            for neighbor in neighbors:
                totals[neighbor.target] += self._neighbor_weight(neighbor.distance)

            denominator = math.fsum(totals.values())
            results.append({
                label: totals.get(label, 0.0) / denominator
                for label in self.classes
            })

        return results

    def predict(self, X: Sequence[Vector]) -> list[Any]:
        validate_dataset(X)

        if self.task == "classification":
            probabilities = self.predict_proba(X)
            predictions: list[Any] = []

            for query, distribution in zip(X, probabilities):
                neighbors = self.kneighbors(query)
                exact_matches = [n for n in neighbors if n.distance == 0]
                if self.weights == "distance" and exact_matches:
                    neighbors = exact_matches

                totals: dict[Any, float] = defaultdict(float)
                distance_totals: dict[Any, float] = defaultdict(float)

                for neighbor in neighbors:
                    totals[neighbor.target] += self._neighbor_weight(neighbor.distance)
                    distance_totals[neighbor.target] += neighbor.distance

                # A stable tie-break avoids dependence on set iteration order.
                winner = min(
                    self.classes,
                    key=lambda label: (
                        -totals.get(label, 0.0),
                        distance_totals.get(label, 0.0),
                        str(label),
                    ),
                )
                predictions.append(winner)

            return predictions

        predictions = []
        for query in X:
            neighbors = self.kneighbors(query)
            exact_matches = [n for n in neighbors if n.distance == 0]
            if self.weights == "distance" and exact_matches:
                neighbors = exact_matches

            numerator = math.fsum(
                self._neighbor_weight(n.distance) * float(n.target)
                for n in neighbors
            )
            denominator = math.fsum(
                self._neighbor_weight(n.distance) for n in neighbors
            )
            predictions.append(numerator / denominator)

        return predictions


def accuracy_score(actual: Sequence[Any], predicted: Sequence[Any]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted labels must be equally sized and nonempty.")
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def confusion_matrix(
    actual: Sequence[Any],
    predicted: Sequence[Any],
) -> dict[Any, dict[Any, int]]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted labels must be equally sized and nonempty.")

    labels = sorted(set(actual) | set(predicted), key=lambda label: str(label))
    matrix = {actual_label: {predicted_label: 0 for predicted_label in labels}
              for actual_label in labels}

    for true_label, predicted_label in zip(actual, predicted):
        matrix[true_label][predicted_label] += 1
    return matrix


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted values must be equally sized and nonempty.")
    return math.fsum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def root_mean_squared_error(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted values must be equally sized and nonempty.")
    return math.sqrt(
        math.fsum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
    )


def train_test_split(
    X: Sequence[Vector],
    y: Sequence[Any],
    test_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[list[float]], list[list[float]], list[Any], list[Any]]:
    """Deterministic split for demonstrations; stratification is not automatic."""
    validate_dataset(X, y)
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    indices = list(range(len(X)))
    random.Random(seed).shuffle(indices)
    test_count = max(1, min(len(X) - 1, round(len(X) * test_fraction)))
    test_indices = set(indices[:test_count])
    train_indices = [i for i in indices if i not in test_indices]
    test_order = indices[:test_count]

    return (
        [[float(v) for v in X[i]] for i in train_indices],
        [[float(v) for v in X[i]] for i in test_order],
        [y[i] for i in train_indices],
        [y[i] for i in test_order],
    )


def cross_validate_accuracy(
    X: Sequence[Vector],
    y: Sequence[Any],
    k: int,
    folds: int = 5,
    metric: str = "euclidean",
    weights: str = "uniform",
    scale: bool = True,
) -> float:
    """
    Evaluate k using deterministic folds.

    Scaling is fitted separately inside each fold to avoid leaking validation
    information into training preprocessing.
    """
    validate_dataset(X, y)
    if not 2 <= folds <= len(X):
        raise ValueError("folds must be between 2 and the sample count.")

    indices = list(range(len(X)))
    random.Random(73).shuffle(indices)
    fold_indices = [indices[i::folds] for i in range(folds)]
    scores: list[float] = []

    for validation_indices in fold_indices:
        validation_set = set(validation_indices)
        training_indices = [i for i in indices if i not in validation_set]

        X_train = [X[i] for i in training_indices]
        y_train = [y[i] for i in training_indices]
        X_validation = [X[i] for i in validation_indices]
        y_validation = [y[i] for i in validation_indices]

        if k > len(X_train):
            raise ValueError("k exceeds a fold's training sample count.")

        if scale:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_validation = scaler.transform(X_validation)

        model = KNearestNeighbors(k=k, metric=metric, weights=weights)
        model.fit(X_train, y_train)
        scores.append(accuracy_score(y_validation, model.predict(X_validation)))

    return statistics.fmean(scores)


def demonstration() -> None:
    print("Distance metric comparison")
    point_a = [2.0, 3.0]
    point_b = [5.0, 7.0]
    print(f"Euclidean: {euclidean_distance(point_a, point_b):.3f}")
    print(f"Manhattan: {manhattan_distance(point_a, point_b):.3f}")
    print(f"Minkowski p=3: {minkowski_distance(point_a, point_b, 3):.3f}")
    print(f"Chebyshev: {chebyshev_distance(point_a, point_b):.3f}")
    print(f"Cosine: {cosine_distance(point_a, point_b):.3f}")

    # The two features use different units. Scaling prevents annual income
    # from overwhelming an age feature solely because its numeric range is larger.
    X = [
        [22, 28000], [24, 32000], [27, 35000], [29, 41000],
        [35, 52000], [38, 57000], [42, 63000], [45, 71000],
        [48, 76000], [52, 85000], [55, 91000], [58, 98000],
    ]
    y = [
        "entry", "entry", "entry", "entry",
        "professional", "professional", "professional", "professional",
        "senior", "senior", "senior", "senior",
    ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=17)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nClassification comparison")
    for metric in ("euclidean", "manhattan", "cosine"):
        model = KNearestNeighbors(
            k=3,
            task="classification",
            metric=metric,
            weights="distance",
        ).fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        print(f"\nMetric: {metric}")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
        print("Confusion matrix:", confusion_matrix(y_test, predictions))
        print("Query probabilities:", model.predict_proba([[0.0, 0.0]])[0])

    print("\nNeighborhood inspection")
    model = KNearestNeighbors(k=3, weights="distance")
    model.fit(X_train_scaled, y_train)
    query = scaler.transform([[36, 54000]])[0]
    for neighbor in model.kneighbors(query):
        print(
            f"Training index={neighbor.index}, "
            f"distance={neighbor.distance:.4f}, "
            f"label={neighbor.target}"
        )

    # Regression estimates a continuous value instead of selecting a class.
    area_and_age = [
        [45, 8], [55, 10], [60, 5], [70, 12],
        [80, 7], [90, 4], [100, 9], [110, 6],
        [120, 3], [130, 8], [140, 5], [150, 2],
    ]
    prices = [
        32, 39, 44, 49, 55, 63, 69, 74, 82, 88, 95, 104,
    ]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        area_and_age, prices, seed=21
    )
    regression_scaler = StandardScaler().fit(Xr_train)
    Xr_train_scaled = regression_scaler.transform(Xr_train)
    Xr_test_scaled = regression_scaler.transform(Xr_test)

    regressor = KNearestNeighbors(
        k=3,
        task="regression",
        metric="euclidean",
        weights="distance",
    ).fit(Xr_train_scaled, yr_train)

    price_predictions = regressor.predict(Xr_test_scaled)
    print("\nRegression evaluation")
    print("Actual prices:", yr_test)
    print("Predicted prices:", [round(value, 2) for value in price_predictions])
    print(f"MAE: {mean_absolute_error(yr_test, price_predictions):.3f}")
    print(f"RMSE: {root_mean_squared_error(yr_test, price_predictions):.3f}")

    print("\nCross-validation for neighborhood selection")
    for k in (1, 3, 5):
        score = cross_validate_accuracy(
            X,
            y,
            k=k,
            folds=4,
            metric="euclidean",
            weights="distance",
            scale=True,
        )
        print(f"k={k}: mean validation accuracy={score:.3f}")


def run_edge_case_checks() -> None:
    """Executable checks for important correctness and validation guarantees."""
    assert math.isclose(euclidean_distance([0, 0], [3, 4]), 5.0)
    assert math.isclose(manhattan_distance([0, 0], [3, 4]), 7.0)
    assert math.isclose(minkowski_distance([0, 0], [3, 4], 1), 7.0)
    assert math.isclose(chebyshev_distance([0, 0], [3, 4]), 4.0)

    classifier = KNearestNeighbors(k=3, weights="distance")
    classifier.fit([[0], [0], [10]], ["approved", "rejected", "rejected"])
    assert classifier.predict([[0]]) == ["approved"]
    # Exact duplicate neighbors are handled without dividing by zero.
    assert math.isclose(sum(classifier.predict_proba([[0]])[0].values()), 1.0)

    regressor = KNearestNeighbors(k=2, task="regression")
    regressor.fit([[0], [2]], [10, 20])
    assert math.isclose(regressor.predict([[1]])[0], 15.0)

    try:
        KNearestNeighbors(k=4).fit([[0], [1], [2]], ["a", "b", "c"])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid k should be rejected.")

    try:
        euclidean_distance([1], [1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched dimensions should be rejected.")

    try:
        cosine_distance([0, 0], [1, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("Zero-vector cosine distance should be rejected.")

    print("Edge-case checks passed.")


if __name__ == "__main__":
    run_edge_case_checks()
    demonstration()
