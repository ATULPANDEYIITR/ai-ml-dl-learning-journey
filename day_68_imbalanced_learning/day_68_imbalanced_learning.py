#!/usr/bin/env python3
"""
Imbalanced Learning
Class imbalance, random resampling, SMOTE, and class weighting.

Uses only the Python standard library.
Implements:
- Synthetic binary classification data
- Stratified train/test splitting
- Random oversampling and undersampling
- SMOTE with nearest-neighbor interpolation
- Class-weighted logistic regression
- Confusion matrix and classification metrics
- Reproducible experiments and validation

Run:
    python imbalanced_learning.py
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Strategy(Enum):
    BASELINE = "baseline"
    RANDOM_OVERSAMPLING = "random_oversampling"
    RANDOM_UNDERSAMPLING = "random_undersampling"
    SMOTE = "smote"
    CLASS_WEIGHTING = "class_weighting"


@dataclass(frozen=True)
class Sample:
    features: tuple[float, ...]
    label: int

    def __post_init__(self):
        if not self.features:
            raise ValueError("Features cannot be empty.")
        if self.label not in (0, 1):
            raise ValueError("Only binary labels 0 and 1 are supported.")
        if not all(math.isfinite(x) for x in self.features):
            raise ValueError("Features must be finite.")


@dataclass
class Metrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def precision(self):
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self):
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def specificity(self):
        denominator = self.true_negative + self.false_positive
        return self.true_negative / denominator if denominator else 0.0

    @property
    def f1(self):
        denominator = self.precision + self.recall
        return (
            2 * self.precision * self.recall / denominator
            if denominator else 0.0
        )

    @property
    def balanced_accuracy(self):
        return (self.recall + self.specificity) / 2

    @property
    def accuracy(self):
        total = (
            self.true_positive + self.true_negative
            + self.false_positive + self.false_negative
        )
        return (
            (self.true_positive + self.true_negative) / total
            if total else 0.0
        )


def validate_dataset(data: Sequence[Sample]) -> None:
    if not data:
        raise ValueError("Dataset cannot be empty.")

    dimensions = len(data[0].features)

    if any(len(sample.features) != dimensions for sample in data):
        raise ValueError("Inconsistent feature dimensions.")

    counts = Counter(sample.label for sample in data)

    if len(counts) != 2:
        raise ValueError("Both classes must be represented.")


def class_distribution(data):
    counts = Counter(sample.label for sample in data)
    return {
        "majority": max(counts.values()),
        "minority": min(counts.values()),
        "ratio": max(counts.values()) / min(counts.values()),
        "counts": dict(sorted(counts.items())),
    }


def generate_dataset(
    majority_count=240,
    minority_count=30,
    seed=42
):
    """Generate overlapping but distinguishable binary classes."""
    rng = random.Random(seed)
    samples = []

    for _ in range(majority_count):
        x1 = rng.gauss(0.0, 1.2)
        x2 = rng.gauss(0.0, 1.2)
        samples.append(Sample((x1, x2), 0))

    for _ in range(minority_count):
        x1 = rng.gauss(2.0, 1.0)
        x2 = rng.gauss(2.0, 1.0)
        samples.append(Sample((x1, x2), 1))

    rng.shuffle(samples)
    return samples


def stratified_split(data, test_fraction=0.25, seed=42):
    """Split each class independently to preserve its distribution."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    rng = random.Random(seed)
    groups = {0: [], 1: []}

    for sample in data:
        groups[sample.label].append(sample)

    train, test = [], []

    for label, group in groups.items():
        rng.shuffle(group)

        if len(group) < 2:
            raise ValueError(f"Class {label} needs at least two samples.")

        split_index = max(
            1,
            min(len(group) - 1, int(len(group) * test_fraction))
        )

        test.extend(group[:split_index])
        train.extend(group[split_index:])

    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


def random_oversample(data, seed=42):
    rng = random.Random(seed)
    groups = {
        label: [sample for sample in data if sample.label == label]
        for label in (0, 1)
    }

    target = max(len(group) for group in groups.values())

    result = list(data)

    for group in groups.values():
        while len(group) < target:
            result.append(rng.choice(group))

    rng.shuffle(result)
    return result


def random_undersample(data, seed=42):
    rng = random.Random(seed)
    groups = {
        label: [sample for sample in data if sample.label == label]
        for label in (0, 1)
    }

    target = min(len(group) for group in groups.values())

    result = []

    for group in groups.values():
        result.extend(rng.sample(group, target))

    rng.shuffle(result)
    return result


def euclidean_distance(a, b):
    return math.sqrt(
        sum((x - y) ** 2 for x, y in zip(a, b))
    )


def smote(data, neighbors=5, seed=42):
    """
    Synthetic Minority Over-sampling Technique.

    Selects a minority sample, finds its nearest minority neighbors,
    and generates a point along the line between two samples.
    """
    rng = random.Random(seed)

    groups = {
        label: [sample for sample in data if sample.label == label]
        for label in (0, 1)
    }

    minority = groups[1]
    majority = groups[0]

    if len(minority) < 2:
        raise ValueError("SMOTE requires at least two minority samples.")

    if not majority:
        raise ValueError("Majority class cannot be empty.")

    k = min(neighbors, len(minority) - 1)
    required = len(majority) - len(minority)

    result = list(data)

    for _ in range(required):
        source = rng.choice(minority)

        candidates = sorted(
            (
                euclidean_distance(source.features, candidate.features),
                candidate
            )
            for candidate in minority
            if candidate is not source
        )

        selected = rng.choice(candidates[:k])[1]
        interpolation = rng.random()

        features = tuple(
            source_value
            + interpolation * (neighbor_value - source_value)
            for source_value, neighbor_value in zip(
                source.features, selected.features
            )
        )

        result.append(Sample(features, 1))

    rng.shuffle(result)
    return result


class LogisticRegression:
    """Binary logistic regression using batch gradient descent."""

    def __init__(
        self,
        learning_rate=0.08,
        epochs=1200,
        regularization=0.001
    ):
        if learning_rate <= 0 or epochs <= 0:
            raise ValueError("Invalid training parameters.")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.weights = []
        self.bias = 0.0

    @staticmethod
    def sigmoid(value):
        value = max(-500.0, min(500.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def fit(self, data, class_weights=None):
        validate_dataset(data)

        dimensions = len(data[0].features)
        self.weights = [0.0] * dimensions
        self.bias = 0.0

        if class_weights is None:
            class_weights = {0: 1.0, 1: 1.0}

        if any(
            not math.isfinite(weight) or weight <= 0
            for weight in class_weights.values()
        ):
            raise ValueError("Class weights must be finite and positive.")

        total_weight = sum(
            class_weights[sample.label] for sample in data
        )

        for _ in range(self.epochs):
            gradients = [0.0] * dimensions
            bias_gradient = 0.0

            for sample in data:
                score = sum(
                    weight * feature
                    for weight, feature in zip(
                        self.weights, sample.features
                    )
                ) + self.bias

                probability = self.sigmoid(score)
                error = probability - sample.label
                sample_weight = class_weights[sample.label]

                for j, feature in enumerate(sample.features):
                    gradients[j] += (
                        sample_weight * error * feature
                    )

                bias_gradient += sample_weight * error

            for j in range(dimensions):
                gradient = (
                    gradients[j] / total_weight
                    + self.regularization * self.weights[j]
                )
                self.weights[j] -= self.learning_rate * gradient

            self.bias -= (
                self.learning_rate * bias_gradient / total_weight
            )

        return self

    def predict_probability(self, features):
        if len(features) != len(self.weights):
            raise ValueError("Feature dimension mismatch.")

        score = sum(
            weight * feature
            for weight, feature in zip(self.weights, features)
        ) + self.bias

        return self.sigmoid(score)

    def predict(self, features, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")

        return int(self.predict_probability(features) >= threshold)


def calculate_metrics(actual, predicted):
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Prediction and target lengths must match.")

    if any(value not in (0, 1) for value in actual + predicted):
        raise ValueError("Labels must be binary.")

    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))

    return Metrics(tp, tn, fp, fn)


def calculate_class_weights(data):
    counts = Counter(sample.label for sample in data)

    if len(counts) != 2:
        raise ValueError("Both classes are required.")

    total = len(data)

    return {
        label: total / (2 * count)
        for label, count in counts.items()
    }


def run_experiment(name, train, test, weights=None):
    model = LogisticRegression().fit(train, weights)

    actual = [sample.label for sample in test]
    predicted = [
        model.predict(sample.features)
        for sample in test
    ]

    return calculate_metrics(actual, predicted)


def main():
    data = generate_dataset()
    validate_dataset(data)

    train, test = stratified_split(data)

    print("=" * 72)
    print("IMBALANCED LEARNING EXPERIMENT")
    print("=" * 72)

    print("\nOriginal dataset:", class_distribution(data))
    print("Training dataset:", class_distribution(train))
    print("Testing dataset:", class_distribution(test))

    experiments = {
        Strategy.BASELINE.value: (train, None),
        Strategy.RANDOM_OVERSAMPLING.value: (
            random_oversample(train), None
        ),
        Strategy.RANDOM_UNDERSAMPLING.value: (
            random_undersample(train), None
        ),
        Strategy.SMOTE.value: (smote(train), None),
        Strategy.CLASS_WEIGHTING.value: (
            train, calculate_class_weights(train)
        ),
    }

    print("\nMODEL EVALUATION")
    print("-" * 72)
    print(
        f"{'Strategy':27} {'Precision':>10} {'Recall':>10} "
        f"{'F1':>10} {'Balanced Acc.':>14}"
    )

    for name, (training_data, weights) in experiments.items():
        metrics = run_experiment(
            name, training_data, test, weights
        )

        print(
            f"{name:27} "
            f"{metrics.precision:10.3f} "
            f"{metrics.recall:10.3f} "
            f"{metrics.f1:10.3f} "
            f"{metrics.balanced_accuracy:14.3f}"
        )

        print(
            f"  TP={metrics.true_positive}, "
            f"TN={metrics.true_negative}, "
            f"FP={metrics.false_positive}, "
            f"FN={metrics.false_negative}"
        )

    print("\nCLASS WEIGHTS")
    print(calculate_class_weights(train))

    print("\nSMOTE DISTRIBUTION")
    print(class_distribution(smote(train)))

    print("\nVALIDATION TESTS")
    assert len(random_oversample(train)) >= len(train)
    assert len(random_undersample(train)) <= len(train)
    assert len(smote(train)) >= len(train)

    print("All validation tests passed.")

    print("\nPRODUCTION CONTROLS")
    print("- Split data before applying resampling.")
    print("- Never resample validation or test datasets.")
    print("- Fit preprocessing only on training data.")
    print("- Tune thresholds using validation data.")
    print("- Monitor minority-class recall and precision.")
    print("- Check synthetic samples for unrealistic combinations.")
    print("- Use patient-level or group-level splitting when required.")
    print("- Track dataset versions, seeds, and class distributions.")


if __name__ == "__main__":
    main()
