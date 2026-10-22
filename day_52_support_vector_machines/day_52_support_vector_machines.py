#!/usr/bin/env python3
"""
Support Vector Machines: margins, hyperplanes, support vectors, and soft margins.

Self-contained implementation using only the Python standard library.
The script trains a linear soft-margin SVM with subgradient descent,
reports the learned hyperplane and support vectors, evaluates new samples,
and demonstrates the effect of the regularization parameter C.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random
from typing import Iterable, Sequence


Vector = tuple[float, ...]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension.")
    return sum(x * y for x, y in zip(a, b))


def norm(vector: Sequence[float]) -> float:
    return sqrt(dot(vector, vector))


@dataclass(frozen=True)
class Sample:
    features: Vector
    label: int

    def __post_init__(self) -> None:
        if self.label not in (-1, 1):
            raise ValueError("SVM labels must be -1 or +1.")
        if not self.features:
            raise ValueError("Feature vectors cannot be empty.")
        if not all(isinstance(x, (int, float)) for x in self.features):
            raise TypeError("Features must be numeric.")


class LinearSVM:
    """
    Primal soft-margin objective:

        1/2 ||w||^2 + C * sum(max(0, 1 - y_i(w.x_i + b)))

    The implementation uses a subgradient update.  C controls the
    penalty for margin violations:

      small C -> stronger regularization and a wider tolerance for violations
      large C -> stronger pressure to classify training observations correctly
    """

    def __init__(
        self,
        C: float = 1.0,
        learning_rate: float = 0.01,
        epochs: int = 5000,
    ) -> None:
        if C <= 0:
            raise ValueError("C must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] = []
        self.bias = 0.0
        self._dimension = 0

    def fit(self, samples: Sequence[Sample]) -> None:
        if not samples:
            raise ValueError("Training data cannot be empty.")

        dimension = len(samples[0].features)
        if any(len(sample.features) != dimension for sample in samples):
            raise ValueError("All feature vectors must have the same dimension.")

        self._dimension = dimension
        self.weights = [0.0] * dimension
        self.bias = 0.0

        # A deterministic order makes the demonstration reproducible.
        for _ in range(self.epochs):
            for sample in samples:
                margin = sample.label * (
                    dot(self.weights, sample.features) + self.bias
                )

                if margin >= 1.0:
                    # Correctly classified outside the margin:
                    # only the L2 regularization gradient remains.
                    gradient_w = self.weights[:]
                    gradient_b = 0.0
                else:
                    # Inside the margin or misclassified:
                    # hinge-loss subgradient contributes -y*x and -y.
                    gradient_w = [
                        w - self.C * sample.label * x
                        for w, x in zip(self.weights, sample.features)
                    ]
                    gradient_b = -self.C * sample.label

                self.weights = [
                    w - self.learning_rate * gradient
                    for w, gradient in zip(self.weights, gradient_w)
                ]
                self.bias -= self.learning_rate * gradient_b

    def decision_function(self, features: Sequence[float]) -> float:
        self._validate_dimension(features)
        return dot(self.weights, features) + self.bias

    def predict(self, features: Sequence[float]) -> int:
        return 1 if self.decision_function(features) >= 0 else -1

    def geometric_margin(self, features: Sequence[float], label: int) -> float:
        if label not in (-1, 1):
            raise ValueError("Label must be -1 or +1.")
        denominator = norm(self.weights)
        if denominator == 0:
            raise ZeroDivisionError("Hyperplane has zero normal vector.")
        return label * self.decision_function(features) / denominator

    def functional_margin(self, sample: Sample) -> float:
        return sample.label * self.decision_function(sample.features)

    def hinge_loss(self, sample: Sample) -> float:
        return max(0.0, 1.0 - self.functional_margin(sample))

    def objective(self, samples: Iterable[Sample]) -> float:
        samples = list(samples)
        if not samples:
            return 0.0
        regularization = 0.5 * dot(self.weights, self.weights)
        loss = self.C * sum(self.hinge_loss(sample) for sample in samples)
        return regularization + loss

    def support_vectors(
        self,
        samples: Sequence[Sample],
        tolerance: float = 0.05,
    ) -> list[Sample]:
        """
        Training points with functional margin approximately <= 1
        are the points that influence the soft-margin boundary.
        """
        if tolerance < 0:
            raise ValueError("tolerance cannot be negative.")

        return [
            sample
            for sample in samples
            if self.functional_margin(sample) <= 1.0 + tolerance
        ]

    def _validate_dimension(self, features: Sequence[float]) -> None:
        if len(features) != self._dimension:
            raise ValueError(
                f"Expected {self._dimension} features, got {len(features)}."
            )

    def describe(self, samples: Sequence[Sample]) -> str:
        w_norm = norm(self.weights)
        if w_norm == 0:
            margin_width = float("inf")
        else:
            # Distance between the two canonical margin hyperplanes.
            margin_width = 2.0 / w_norm

        support = self.support_vectors(samples)

        lines = [
            f"weights = {[round(x, 4) for x in self.weights]}",
            f"bias = {self.bias:.4f}",
            f"||w|| = {w_norm:.4f}",
            f"margin width = {margin_width:.4f}",
            f"objective = {self.objective(samples):.4f}",
            f"support-vector candidates = {len(support)}",
        ]

        return "\n".join(lines)


def make_dataset() -> list[Sample]:
    """
    Two-dimensional repository-security classification scenario.

    x1 = number of security defects found
    x2 = test-coverage ratio

    Label +1 means "merge candidate is acceptable".
    Label -1 means "merge candidate requires remediation".

    The classes are deliberately not perfectly separable so that the
    soft-margin behavior has something meaningful to optimize.
    """
    raw = [
        (0.4, 0.95, 1),
        (0.7, 0.91, 1),
        (1.0, 0.88, 1),
        (1.2, 0.84, 1),
        (1.5, 0.80, 1),
        (1.8, 0.77, 1),
        (2.0, 0.73, 1),
        (2.4, 0.70, -1),
        (2.8, 0.65, -1),
        (3.0, 0.60, -1),
        (3.4, 0.56, -1),
        (3.8, 0.52, -1),
        (4.1, 0.48, -1),
        (4.5, 0.44, -1),
        (2.1, 0.68, -1),
        (1.7, 0.74, -1),  # deliberate boundary/noisy observation
    ]
    return [Sample((x1, x2), label) for x1, x2, label in raw]


def standardize(samples: Sequence[Sample]) -> tuple[list[Sample], Vector, Vector]:
    """Standardize features so unequal scales do not distort the hyperplane."""
    dimension = len(samples[0].features)
    means = tuple(
        sum(sample.features[j] for sample in samples) / len(samples)
        for j in range(dimension)
    )

    stds = []
    for j in range(dimension):
        variance = sum(
            (sample.features[j] - means[j]) ** 2 for sample in samples
        ) / len(samples)
        std = sqrt(variance)
        stds.append(std if std > 1e-12 else 1.0)

    transformed = [
        Sample(
            tuple(
                (value - means[j]) / stds[j]
                for j, value in enumerate(sample.features)
            ),
            sample.label,
        )
        for sample in samples
    ]

    return transformed, means, tuple(stds)


def print_predictions(model: LinearSVM, samples: Sequence[Sample]) -> None:
    print("\nTraining predictions")
    print("-" * 72)
    print("features          actual  predicted  score     functional_margin")
    for sample in samples:
        score = model.decision_function(sample.features)
        functional = model.functional_margin(sample)
        print(
            f"{str(tuple(round(x, 2) for x in sample.features)):18}"
            f"{sample.label:>7}"
            f"{model.predict(sample.features):>11}"
            f"{score:>9.3f}"
            f"{functional:>19.3f}"
        )


def accuracy(model: LinearSVM, samples: Sequence[Sample]) -> float:
    correct = sum(
        model.predict(sample.features) == sample.label for sample in samples
    )
    return correct / len(samples)


def compare_regularization(samples: Sequence[Sample]) -> None:
    print("\nSoft-margin regularization comparison")
    print("-" * 72)

    for C in (0.05, 0.25, 1.0, 5.0, 20.0):
        model = LinearSVM(C=C, learning_rate=0.002, epochs=4000)
        model.fit(samples)
        support_count = len(model.support_vectors(samples))
        print(
            f"C={C:>5.2f}  "
            f"accuracy={accuracy(model, samples):.3f}  "
            f"support_vectors={support_count:>2}  "
            f"objective={model.objective(samples):.4f}"
        )


def main() -> None:
    raw_samples = make_dataset()
    samples, means, stds = standardize(raw_samples)

    print("Support Vector Machine demonstration")
    print("=" * 72)
    print("Feature normalization")
    print(f"means = {[round(x, 4) for x in means]}")
    print(f"stds  = {[round(x, 4) for x in stds]}")

    model = LinearSVM(C=1.0, learning_rate=0.002, epochs=5000)
    model.fit(samples)

    print("\nLearned soft-margin hyperplane")
    print("-" * 72)
    print(model.describe(samples))

    support = model.support_vectors(samples)
    print("\nSupport-vector candidates")
    for index, sample in enumerate(support, start=1):
        print(
            f"{index:>2}. x={tuple(round(v, 3) for v in sample.features)}, "
            f"y={sample.label}, "
            f"functional_margin={model.functional_margin(sample):.4f}, "
            f"hinge_loss={model.hinge_loss(sample):.4f}"
        )

    print_predictions(model, samples)

    print("\nNew observations")
    print("-" * 72)

    # New points are transformed using the training-set statistics.
    new_raw = [
        (1.1, 0.86),
        (2.9, 0.62),
        (4.3, 0.50),
    ]

    for x1, x2 in new_raw:
        normalized = tuple(
            (value - means[j]) / stds[j]
            for j, value in enumerate((x1, x2))
        )
        score = model.decision_function(normalized)
        prediction = model.predict(normalized)
        print(
            f"raw=({x1:.2f}, {x2:.2f})  "
            f"normalized={tuple(round(v, 3) for v in normalized)}  "
            f"prediction={prediction:+d}  score={score:.4f}"
        )

    compare_regularization(samples)

    print("\nProduction considerations")
    print("-" * 72)
    print("* Scale features using statistics learned only from training data.")
    print("* Tune C and optimization parameters with validation data.")
    print("* Inspect support vectors because boundary observations strongly affect")
    print("  a linear SVM.")
    print("* Monitor class imbalance and choose evaluation metrics accordingly.")
    print("* Avoid treating an SVM score as a calibrated probability.")
    print("* Persist the scaler and model parameters together for reproducibility.")


if __name__ == "__main__":
    main()
