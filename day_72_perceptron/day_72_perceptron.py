#!/usr/bin/env python3
"""
Perceptron Learning Rule and Linear Separability

Implements:
- Binary classification using the perceptron learning rule.
- Online weight and bias updates.
- Step activation function.
- Linear separability testing through convergence.
- Training history and mistake tracking.
- Prediction and evaluation metrics.
- Feature normalization.
- Model persistence using JSON.
- Deterministic experiments and unit assertions.

Python 3.10+
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


class PerceptronError(Exception):
    """Base exception for perceptron operations."""


class ValidationError(PerceptronError):
    """Raised when training data or parameters are invalid."""


class TrainingError(PerceptronError):
    """Raised when training cannot proceed."""


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.1
    max_epochs: int = 100
    tolerance: float = 0.0
    shuffle: bool = False
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValidationError("Learning rate must be positive and finite.")
        if self.max_epochs < 1:
            raise ValidationError("Maximum epochs must be positive.")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValidationError("Tolerance must be non-negative.")


@dataclass
class EpochResult:
    epoch: int
    mistakes: int
    weights: list[float]
    bias: float


@dataclass
class Evaluation:
    accuracy: float
    correct: int
    incorrect: int
    confusion_matrix: dict[str, int]


def validate_dataset(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> int:
    if not features or len(features) != len(labels):
        raise ValidationError("Features and labels must have equal non-zero lengths.")

    dimensions = len(features[0])

    if dimensions == 0:
        raise ValidationError("Each sample must contain at least one feature.")

    for sample in features:
        if len(sample) != dimensions:
            raise ValidationError("All samples must have equal dimensions.")
        if not all(math.isfinite(float(value)) for value in sample):
            raise ValidationError("Features must contain finite numbers.")

    if any(label not in (0, 1) for label in labels):
        raise ValidationError("Labels must be binary: 0 or 1.")

    if len(set(labels)) != 2:
        raise ValidationError("Both binary classes must be represented.")

    return dimensions


def step_activation(value: float) -> int:
    """Classify zero as class 1."""
    return 1 if value >= 0.0 else 0


class StandardScaler:
    """Standardize features using training-set statistics."""

    def __init__(self) -> None:
        self.means: list[float] = []
        self.scales: list[float] = []

    def fit(self, features: Sequence[Sequence[float]]) -> None:
        if not features:
            raise ValidationError("Cannot fit scaler on empty data.")

        dimensions = len(features[0])

        if dimensions == 0 or any(len(row) != dimensions for row in features):
            raise ValidationError("Inconsistent feature dimensions.")

        self.means = [
            sum(row[j] for row in features) / len(features)
            for j in range(dimensions)
        ]

        self.scales = []

        for j in range(dimensions):
            variance = sum(
                (row[j] - self.means[j]) ** 2 for row in features
            ) / len(features)

            standard_deviation = math.sqrt(variance)
            self.scales.append(
                standard_deviation if standard_deviation > 1e-12 else 1.0
            )

    def transform(self, features: Sequence[Sequence[float]]) -> list[list[float]]:
        if not self.means:
            raise TrainingError("Scaler has not been fitted.")

        transformed = []

        for row in features:
            if len(row) != len(self.means):
                raise ValidationError("Feature dimension mismatch.")

            transformed.append([
                (float(value) - self.means[j]) / self.scales[j]
                for j, value in enumerate(row)
            ])

        return transformed

    def fit_transform(
        self,
        features: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        self.fit(features)
        return self.transform(features)


class Perceptron:
    """
    Binary linear classifier.

    Prediction:
        y = 1 if w.x + b >= 0 else 0

    Learning rule:
        w[i] = w[i] + learning_rate * (target - prediction) * x[i]
        b = b + learning_rate * (target - prediction)
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.weights: list[float] = []
        self.bias = 0.0
        self.history: list[EpochResult] = []
        self.converged = False
        self.epochs_trained = 0
        self.n_features = 0

    def net_input(self, sample: Sequence[float]) -> float:
        if len(sample) != self.n_features:
            raise ValidationError("Prediction feature dimension mismatch.")

        return sum(
            weight * float(value)
            for weight, value in zip(self.weights, sample)
        ) + self.bias

    def predict(self, sample: Sequence[float]) -> int:
        return step_activation(self.net_input(sample))

    def train(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
    ) -> Perceptron:
        dimensions = validate_dataset(features, labels)

        self.n_features = dimensions
        self.weights = [0.0] * dimensions
        self.bias = 0.0
        self.history.clear()
        self.converged = False
        self.epochs_trained = 0

        samples = [
            ([float(value) for value in row], int(label))
            for row, label in zip(features, labels)
        ]

        rng = random.Random(self.config.random_seed)
        learning_rate = self.config.learning_rate

        for epoch in range(1, self.config.max_epochs + 1):
            if self.config.shuffle:
                rng.shuffle(samples)

            mistakes = 0

            for sample, target in samples:
                prediction = self.predict(sample)
                error = target - prediction

                if error != 0:
                    mistakes += 1

                    # Online update: every observation changes the next decision.
                    self.weights = [
                        weight + learning_rate * error * value
                        for weight, value in zip(self.weights, sample)
                    ]

                    self.bias += learning_rate * error

            self.epochs_trained = epoch

            self.history.append(
                EpochResult(
                    epoch=epoch,
                    mistakes=mistakes,
                    weights=self.weights.copy(),
                    bias=self.bias,
                )
            )

            if mistakes == 0:
                self.converged = True
                break

        return self

    def evaluate(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
    ) -> Evaluation:
        validate_dataset(features, labels)

        tp = tn = fp = fn = 0

        for sample, target in zip(features, labels):
            prediction = self.predict(sample)

            if prediction == 1 and target == 1:
                tp += 1
            elif prediction == 0 and target == 0:
                tn += 1
            elif prediction == 1:
                fp += 1
            else:
                fn += 1

        total = len(labels)
        correct = tp + tn

        return Evaluation(
            accuracy=correct / total,
            correct=correct,
            incorrect=total - correct,
            confusion_matrix={
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
            },
        )

    def save(self, path: str | Path) -> None:
        if not self.weights:
            raise TrainingError("Cannot save an untrained model.")

        payload = {
            "weights": self.weights,
            "bias": self.bias,
            "n_features": self.n_features,
            "config": asdict(self.config),
            "converged": self.converged,
            "epochs_trained": self.epochs_trained,
        }

        destination = Path(path)
        temporary = destination.with_suffix(destination.suffix + ".tmp")

        try:
            temporary.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            temporary.replace(destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise PerceptronError(f"Model persistence failed: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> Perceptron:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))

            model = cls(TrainingConfig(**payload["config"]))
            weights = payload["weights"]
            dimensions = payload["n_features"]
            bias = payload["bias"]

            if (
                not isinstance(weights, list)
                or len(weights) != dimensions
                or dimensions < 1
                or not all(math.isfinite(float(w)) for w in weights)
                or not math.isfinite(float(bias))
            ):
                raise ValidationError("Invalid saved model parameters.")

            model.weights = [float(w) for w in weights]
            model.bias = float(bias)
            model.n_features = dimensions
            model.converged = bool(payload["converged"])
            model.epochs_trained = int(payload["epochs_trained"])

            return model

        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PerceptronError(f"Could not load model: {exc}") from exc


def demonstrate_and_gate() -> None:
    features = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    labels = [0, 0, 0, 1]

    model = Perceptron(
        TrainingConfig(
            learning_rate=0.1,
            max_epochs=100,
            shuffle=False,
        )
    )

    model.train(features, labels)

    print("\nAND gate training")
    print("-" * 55)

    for result in model.history:
        print(
            f"Epoch {result.epoch:3d} | "
            f"Mistakes: {result.mistakes} | "
            f"Weights: {result.weights} | "
            f"Bias: {result.bias:.3f}"
        )

    print("\nPredictions")

    for sample, target in zip(features, labels):
        print(
            f"Input={sample}, "
            f"Expected={target}, "
            f"Predicted={model.predict(sample)}"
        )

    evaluation = model.evaluate(features, labels)

    print(f"\nConverged: {model.converged}")
    print(f"Epochs: {model.epochs_trained}")
    print(f"Accuracy: {evaluation.accuracy:.2%}")
    print(f"Confusion matrix: {evaluation.confusion_matrix}")


def demonstrate_xor_limitation() -> None:
    features = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    labels = [0, 1, 1, 0]

    model = Perceptron(
        TrainingConfig(learning_rate=0.1, max_epochs=100)
    )
    model.train(features, labels)

    print("\nXOR linear-separability experiment")
    print("-" * 55)
    print(f"Converged: {model.converged}")
    print(f"Epochs executed: {model.epochs_trained}")
    print(f"Final epoch mistakes: {model.history[-1].mistakes}")

    # Failure to converge within a finite budget is evidence, not
    # a mathematical proof of non-separability.
    assert not model.converged


def run_tests() -> None:
    model = Perceptron().train(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [0, 0, 0, 1],
    )

    assert model.converged
    assert [model.predict(x) for x in [
        [0, 0], [0, 1], [1, 0], [1, 1]
    ]] == [0, 0, 0, 1]

    scaler = StandardScaler()
    transformed = scaler.fit_transform([
        [10, 100],
        [20, 200],
        [30, 300],
    ])

    assert len(transformed) == 3
    assert abs(sum(row[0] for row in transformed)) < 1e-10

    try:
        validate_dataset([[1, 2]], [2])
        raise AssertionError("Invalid labels should be rejected.")
    except ValidationError:
        pass

    print("\nAll assertions passed.")


def main() -> None:
    demonstrate_and_gate()
    demonstrate_xor_limitation()
    run_tests()


if __name__ == "__main__":
    main()
