"""
Feature Selection: Filter, Wrapper, Embedded Methods and RFE

A self-contained feature-selection laboratory using Python's standard
library. Implements:
1. Variance threshold filtering
2. Pearson correlation filtering
3. Mutual information ranking
4. Sequential forward selection
5. L1-regularized embedded selection
6. Recursive Feature Elimination (RFE)

The example uses a deterministic synthetic binary-classification dataset.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence


class SelectionError(Exception):
    """Raised when feature-selection input or configuration is invalid."""


class SelectionMethod(Enum):
    FILTER = "filter"
    WRAPPER = "wrapper"
    EMBEDDED = "embedded"
    RFE = "recursive_feature_elimination"


@dataclass(frozen=True)
class Feature:
    name: str
    values: tuple[float, ...]


@dataclass
class SelectionResult:
    method: str
    selected_features: list[str]
    scores: dict[str, float]
    details: dict


class Dataset:
    def __init__(
        self,
        feature_names: Sequence[str],
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
    ):
        if not feature_names or not rows:
            raise SelectionError("Features and rows cannot be empty.")

        if len(rows) != len(labels):
            raise SelectionError("Rows and labels must have equal lengths.")

        if len(set(feature_names)) != len(feature_names):
            raise SelectionError("Feature names must be unique.")

        width = len(feature_names)

        if any(len(row) != width for row in rows):
            raise SelectionError("All rows must have the same feature count.")

        if len(set(labels)) < 2:
            raise SelectionError("Classification requires at least two classes.")

        if any(label not in (0, 1) for label in labels):
            raise SelectionError("This implementation requires binary labels 0 and 1.")

        self.feature_names = list(feature_names)
        self.rows = [
            [float(value) for value in row]
            for row in rows
        ]
        self.labels = list(labels)

        if any(not math.isfinite(value) for row in self.rows for value in row):
            raise SelectionError("Feature values must be finite.")

        self._validate_variation()

    def _validate_variation(self):
        if len(self.rows) < 3:
            raise SelectionError("At least three observations are required.")

    @property
    def feature_count(self):
        return len(self.feature_names)

    @property
    def sample_count(self):
        return len(self.rows)

    def column(self, index: int) -> list[float]:
        if not 0 <= index < self.feature_count:
            raise SelectionError("Invalid feature index.")
        return [row[index] for row in self.rows]

    def subset(self, indices: Sequence[int]) -> list[list[float]]:
        if not indices:
            raise SelectionError("At least one feature must be selected.")
        return [[row[i] for i in indices] for row in self.rows]


def mean(values):
    return statistics.fmean(values)


def variance(values):
    if len(values) < 2:
        return 0.0
    return statistics.variance(values)


def pearson_correlation(x, y):
    mx, my = mean(x), mean(y)

    numerator = sum(
        (a - mx) * (b - my)
        for a, b in zip(x, y)
    )

    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in x)
        * sum((b - my) ** 2 for b in y)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def normalize(values):
    center = mean(values)
    scale = math.sqrt(variance(values))

    if scale < 1e-12:
        return [0.0] * len(values)

    return [(value - center) / scale for value in values]


def mutual_information(x, y, bins=5):
    """
    Estimate mutual information using discrete quantile-like bins.
    The result is an approximate ranking score, not an exact
    continuous-variable information estimate.
    """
    if bins < 2:
        raise SelectionError("At least two bins are required.")

    ordered = sorted(x)
    edges = [
        ordered[min(len(ordered) - 1, i * len(ordered) // bins)]
        for i in range(1, bins)
    ]

    x_categories = [
        sum(value >= edge for edge in edges)
        for value in x
    ]

    n = len(x)
    joint = {}
    counts_x = {}
    counts_y = {}

    for a, b in zip(x_categories, y):
        joint[(a, b)] = joint.get((a, b), 0) + 1
        counts_x[a] = counts_x.get(a, 0) + 1
        counts_y[b] = counts_y.get(b, 0) + 1

    result = 0.0

    for (a, b), count in joint.items():
        pxy = count / n
        px = counts_x[a] / n
        py = counts_y[b] / n

        result += pxy * math.log2(pxy / (px * py))

    return max(0.0, result)


def anova_f_score(x, y):
    """One-way ANOVA F statistic for binary classification."""
    classes = sorted(set(y))
    overall = mean(x)

    grouped = {
        cls: [value for value, label in zip(x, y) if label == cls]
        for cls in classes
    }

    between = sum(
        len(values) * (mean(values) - overall) ** 2
        for values in grouped.values()
    )

    within = sum(
        sum((value - mean(values)) ** 2 for value in values)
        for values in grouped.values()
    )

    numerator_df = len(classes) - 1
    denominator_df = len(x) - len(classes)

    if denominator_df <= 0 or within < 1e-12:
        return 0.0 if within >= 1e-12 else float("inf")

    return (between / numerator_df) / (within / denominator_df)


def filter_selection(
    dataset: Dataset,
    top_k=3,
    variance_threshold=1e-8,
    correlation_threshold=0.95,
):
    """
    Filter selection is independent of a predictive model.
    Variance and target association rank features.
    Redundant features are removed using pairwise correlation.
    """
    if not 1 <= top_k <= dataset.feature_count:
        raise SelectionError("top_k is outside the feature-count range.")

    if not 0 <= correlation_threshold <= 1:
        raise SelectionError("Invalid correlation threshold.")

    candidates = []

    for i, name in enumerate(dataset.feature_names):
        values = dataset.column(i)
        feature_variance = variance(values)

        if feature_variance <= variance_threshold:
            continue

        score = anova_f_score(values, dataset.labels)

        candidates.append({
            "index": i,
            "name": name,
            "variance": feature_variance,
            "anova_f": score,
            "mutual_information": mutual_information(
                values, dataset.labels
            ),
        })

    candidates.sort(
        key=lambda item: (
            -item["mutual_information"],
            -item["anova_f"],
            item["name"],
        )
    )

    selected = []

    for candidate in candidates:
        index = candidate["index"]

        redundant = any(
            abs(
                pearson_correlation(
                    dataset.column(index),
                    dataset.column(previous),
                )
            ) >= correlation_threshold
            for previous in selected
        )

        if not redundant:
            selected.append(index)

        if len(selected) >= top_k:
            break

    if not selected:
        raise SelectionError("Filter selection found no eligible features.")

    return SelectionResult(
        method=SelectionMethod.FILTER.value,
        selected_features=[
            dataset.feature_names[i] for i in selected
        ],
        scores={
            item["name"]: item["mutual_information"]
            for item in candidates
        },
        details={
            "ranking": candidates,
            "variance_threshold": variance_threshold,
            "correlation_threshold": correlation_threshold,
        },
    )


def sigmoid(value):
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def logistic_predict(rows, weights, bias):
    return [
        1 if sigmoid(
            bias + sum(a * b for a, b in zip(row, weights))
        ) >= 0.5 else 0
        for row in rows
    ]


def accuracy(actual, predicted):
    if len(actual) != len(predicted) or not actual:
        raise SelectionError("Invalid accuracy inputs.")

    return sum(
        a == b for a, b in zip(actual, predicted)
    ) / len(actual)


def fit_logistic(
    rows,
    labels,
    learning_rate=0.05,
    epochs=500,
    l1=0.0,
):
    """
    Batch gradient descent logistic regression.
    L1 regularization is applied to coefficients, not the intercept.
    """
    if not rows or len(rows) != len(labels):
        raise SelectionError("Invalid training data.")

    width = len(rows[0])

    if width == 0 or any(len(row) != width for row in rows):
        raise SelectionError("Inconsistent training dimensions.")

    if learning_rate <= 0 or epochs < 1 or l1 < 0:
        raise SelectionError("Invalid training configuration.")

    normalized = [
        normalize([row[j] for row in rows])
        for j in range(width)
    ]

    training = [
        [normalized[j][i] for j in range(width)]
        for i in range(len(rows))
    ]

    weights = [0.0] * width
    bias = 0.0
    n = len(rows)

    for _ in range(epochs):
        gradient = [0.0] * width
        bias_gradient = 0.0

        for row, label in zip(training, labels):
            probability = sigmoid(
                bias + sum(a * b for a, b in zip(row, weights))
            )

            error = probability - label

            bias_gradient += error

            for j in range(width):
                gradient[j] += error * row[j]

        bias -= learning_rate * bias_gradient / n

        for j in range(width):
            regularization = l1 * (
                1 if weights[j] > 0
                else -1 if weights[j] < 0
                else 0
            )

            weights[j] -= learning_rate * (
                gradient[j] / n + regularization
            )

    return weights, bias


def train_test_split(dataset, test_ratio=0.25, seed=42):
    if not 0 < test_ratio < 1:
        raise SelectionError("test_ratio must be between 0 and 1.")

    indices = list(range(dataset.sample_count))
    random.Random(seed).shuffle(indices)

    split = int(len(indices) * (1 - test_ratio))
    split = max(1, min(len(indices) - 1, split))

    train_indices = indices[:split]
    test_indices = indices[split:]

    return (
        [dataset.rows[i] for i in train_indices],
        [dataset.labels[i] for i in train_indices],
        [dataset.rows[i] for i in test_indices],
        [dataset.labels[i] for i in test_indices],
    )


def evaluate_subset(dataset, indices):
    """
    Wrapper evaluation uses a predictive model.
    Each candidate subset is evaluated on a held-out validation split.
    """
    train_x, train_y, test_x, test_y = train_test_split(
        dataset, seed=42
    )

    selected_train = [[row[i] for i in indices] for row in train_x]
    selected_test = [[row[i] for i in indices] for row in test_x]

    weights, bias = fit_logistic(
        selected_train,
        train_y,
        learning_rate=0.05,
        epochs=350,
    )

    return accuracy(
        test_y,
        logistic_predict(selected_test, weights, bias),
    )


def sequential_forward_selection(dataset, max_features=3):
    """
    Greedy wrapper method:
    Add the feature producing the highest validation accuracy
    until the requested subset size is reached.
    """
    if not 1 <= max_features <= dataset.feature_count:
        raise SelectionError("Invalid maximum feature count.")

    selected = []
    remaining = set(range(dataset.feature_count))
    history = []

    while remaining and len(selected) < max_features:
        candidates = []

        for index in sorted(remaining):
            subset = selected + [index]
            score = evaluate_subset(dataset, subset)
            candidates.append((score, index))

        score, best = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )

        selected.append(best)
        remaining.remove(best)

        history.append({
            "feature": dataset.feature_names[best],
            "validation_accuracy": score,
        })

    return SelectionResult(
        method=SelectionMethod.WRAPPER.value,
        selected_features=[
            dataset.feature_names[i] for i in selected
        ],
        scores={
            item["feature"]: item["validation_accuracy"]
            for item in history
        },
        details={"selection_history": history},
    )


def embedded_selection(dataset, l1=0.08, threshold=0.015):
    """
    Embedded method:
    L1 logistic regression performs feature selection during training.
    Coefficients are compared after standardization.
    """
    weights, bias = fit_logistic(
        dataset.rows,
        dataset.labels,
        learning_rate=0.04,
        epochs=1000,
        l1=l1,
    )

    scores = {
        name: abs(weight)
        for name, weight in zip(dataset.feature_names, weights)
    }

    selected = [
        name for name, score in scores.items()
        if score > threshold
    ]

    if not selected:
        selected = [
            max(scores, key=scores.get)
        ]

    return SelectionResult(
        method=SelectionMethod.EMBEDDED.value,
        selected_features=selected,
        scores=scores,
        details={
            "l1_strength": l1,
            "coefficient_threshold": threshold,
            "intercept": bias,
        },
    )


def recursive_feature_elimination(dataset, target_count=3):
    """
    RFE repeatedly fits a model and removes the feature with
    the smallest absolute standardized coefficient.
    """
    if not 1 <= target_count <= dataset.feature_count:
        raise SelectionError("Invalid RFE target count.")

    active = list(range(dataset.feature_count))
    history = []

    while len(active) > target_count:
        rows = dataset.subset(active)

        weights, _ = fit_logistic(
            rows,
            dataset.labels,
            learning_rate=0.04,
            epochs=700,
            l1=0.0,
        )

        weakest_position = min(
            range(len(active)),
            key=lambda position: (
                abs(weights[position]),
                dataset.feature_names[active[position]],
            ),
        )

        removed = active.pop(weakest_position)

        history.append({
            "removed_feature": dataset.feature_names[removed],
            "absolute_coefficient": abs(weights[weakest_position]),
            "remaining_count": len(active),
        })

    final_weights, _ = fit_logistic(
        dataset.subset(active),
        dataset.labels,
        learning_rate=0.04,
        epochs=700,
    )

    return SelectionResult(
        method=SelectionMethod.RFE.value,
        selected_features=[
            dataset.feature_names[i] for i in active
        ],
        scores={
            dataset.feature_names[index]: abs(weight)
            for index, weight in zip(active, final_weights)
        },
        details={
            "elimination_history": history,
            "target_count": target_count,
        },
    )


def create_dataset(seed=42, samples=240):
    """
    Features:
    - age: weak predictive signal
    - income: predictive signal
    - account_balance: predictive signal
    - transaction_count: noise
    - duplicated_income: redundant signal
    - constant_feature: zero variance
    """
    rng = random.Random(seed)

    names = [
        "age",
        "income",
        "account_balance",
        "transaction_count",
        "duplicated_income",
        "constant_feature",
    ]

    rows = []
    labels = []

    for _ in range(samples):
        age = rng.uniform(20, 70)
        income = rng.uniform(20000, 150000)
        balance = income * 0.45 + rng.gauss(0, 10000)
        transactions = rng.uniform(0, 100)
        duplicate = income + rng.gauss(0, 100)

        probability = sigmoid(
            -2.0
            + 0.035 * (age - 45)
            + 0.000025 * (income - 85000)
            + 0.000035 * (balance - 40000)
        )

        label = int(rng.random() < probability)

        rows.append([
            age,
            income,
            balance,
            transactions,
            duplicate,
            10.0,
        ])

        labels.append(label)

    return Dataset(names, rows, labels)


def export_results(results, output_directory="feature_selection_output"):
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)

    report = {
        "results": [
            asdict(result)
            for result in results
        ]
    }

    report_path = directory / "selection_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    with (directory / "selected_features.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["method", "feature"])

        for result in results:
            for feature in result.selected_features:
                writer.writerow([result.method, feature])

    return report_path


def run_tests():
    dataset = create_dataset(samples=100)

    assert dataset.feature_count == 6
    assert dataset.sample_count == 100
    assert variance(dataset.column(5)) == 0.0

    assert -1 <= pearson_correlation(
        dataset.column(0), dataset.column(1)
    ) <= 1

    assert mutual_information(
        dataset.column(1), dataset.labels
    ) >= 0

    assert accuracy([1, 0], [1, 0]) == 1.0

    try:
        Dataset(["x"], [[float("nan")]], [0])
        raise AssertionError("Non-finite values were accepted.")
    except SelectionError:
        pass

    try:
        Dataset(["x"], [[1], [2]], [0, 1])
        raise AssertionError("Insufficient observations were accepted.")
    except SelectionError:
        pass


def main():
    run_tests()

    dataset = create_dataset()

    results = [
        filter_selection(dataset, top_k=3),
        sequential_forward_selection(dataset, max_features=3),
        embedded_selection(dataset, l1=0.08),
        recursive_feature_elimination(dataset, target_count=3),
    ]

    print("=" * 65)
    print("FEATURE SELECTION LABORATORY")
    print("=" * 65)

    print(
        f"Observations: {dataset.sample_count} | "
        f"Original features: {dataset.feature_count}"
    )

    for result in results:
        print(f"\nMethod: {result.method}")
        print("Selected:", ", ".join(result.selected_features))
        print("Scores:", json.dumps(result.scores, indent=2))

    report = export_results(results)

    print(f"\nReports saved to: {report.resolve()}")
    print("\nAll validation tests passed.")


if __name__ == "__main__":
    main()
