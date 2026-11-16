#!/usr/bin/env python3
"""
Gradient Descent: Batch, Stochastic, and Mini-Batch Optimization

A self-contained linear regression implementation comparing three
gradient descent strategies using Python's standard library.

Features:
- Dataset generation and preprocessing
- Linear regression using gradient descent
- Batch, stochastic, and mini-batch optimization
- Learning-rate validation and convergence monitoring
- Gradient clipping
- L2 regularization
- Reproducible stochastic sampling
- Training metrics and model evaluation
- Numerical gradient verification
- CSV export
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence


class OptimizerType(Enum):
    BATCH = "batch"
    STOCHASTIC = "stochastic"
    MINI_BATCH = "mini_batch"


@dataclass(frozen=True)
class Sample:
    features: tuple[float, ...]
    target: float

    def validate(self) -> None:
        if not self.features:
            raise ValueError("A sample must contain features.")

        if not all(math.isfinite(x) for x in self.features):
            raise ValueError("Features must be finite.")

        if not math.isfinite(self.target):
            raise ValueError("Target must be finite.")


@dataclass
class TrainingConfig:
    optimizer: OptimizerType
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 16
    regularization: float = 0.0
    gradient_clip: float = 10.0
    tolerance: float = 1e-10
    patience: int = 15
    shuffle: bool = True
    seed: int = 42

    def validate(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite.")

        if self.epochs <= 0:
            raise ValueError("Epochs must be positive.")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if self.regularization < 0 or not math.isfinite(self.regularization):
            raise ValueError("Regularization must be nonnegative.")

        if self.gradient_clip <= 0 or not math.isfinite(self.gradient_clip):
            raise ValueError("Gradient clipping must be positive.")

        if self.tolerance < 0 or not math.isfinite(self.tolerance):
            raise ValueError("Tolerance must be nonnegative.")

        if self.patience <= 0:
            raise ValueError("Patience must be positive.")


@dataclass
class TrainingResult:
    optimizer: str
    weights: list[float]
    bias: float
    history: list[dict[str, float]] = field(default_factory=list)
    epochs_completed: int = 0
    converged: bool = False

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("Feature dimension mismatch.")

        return sum(w * x for w, x in zip(self.weights, features)) + self.bias


class Dataset:
    @staticmethod
    def generate(
        count: int = 200,
        dimensions: int = 2,
        noise: float = 0.5,
        seed: int = 42,
    ) -> list[Sample]:
        if count < 2 or dimensions < 1:
            raise ValueError("Invalid dataset dimensions.")

        if noise < 0 or not math.isfinite(noise):
            raise ValueError("Noise must be nonnegative.")

        rng = random.Random(seed)
        true_weights = [2.5, -1.7][:dimensions]

        while len(true_weights) < dimensions:
            true_weights.append(0.8 * len(true_weights))

        true_bias = 4.0
        samples = []

        for _ in range(count):
            features = tuple(
                rng.uniform(-5.0, 5.0)
                for _ in range(dimensions)
            )

            target = (
                sum(w * x for w, x in zip(true_weights, features))
                + true_bias
                + rng.gauss(0, noise)
            )

            samples.append(Sample(features, target))

        return samples


class LinearRegression:
    def __init__(self, dimensions: int):
        if dimensions <= 0:
            raise ValueError("Dimensions must be positive.")

        self.weights = [0.0] * dimensions
        self.bias = 0.0

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("Feature dimension mismatch.")

        return sum(
            w * x for w, x in zip(self.weights, features)
        ) + self.bias

    def loss(
        self,
        samples: Sequence[Sample],
        regularization: float = 0.0,
    ) -> float:
        if not samples:
            raise ValueError("Dataset cannot be empty.")

        mse = sum(
            (self.predict(sample.features) - sample.target) ** 2
            for sample in samples
        ) / len(samples)

        penalty = regularization * sum(
            weight * weight for weight in self.weights
        )

        return mse / 2.0 + penalty / 2.0

    def compute_gradients(
        self,
        samples: Sequence[Sample],
        regularization: float,
    ) -> tuple[list[float], float]:
        if not samples:
            raise ValueError("Gradient requires at least one sample.")

        gradients = [0.0] * len(self.weights)
        bias_gradient = 0.0

        for sample in samples:
            error = self.predict(sample.features) - sample.target

            for index, value in enumerate(sample.features):
                gradients[index] += error * value

            bias_gradient += error

        count = len(samples)

        gradients = [
            gradient / count + regularization * self.weights[index]
            for index, gradient in enumerate(gradients)
        ]

        bias_gradient /= count

        return gradients, bias_gradient

    def update(
        self,
        gradients: Sequence[float],
        bias_gradient: float,
        learning_rate: float,
        gradient_clip: float,
    ) -> float:
        # Global norm clipping limits unusually large updates.
        norm = math.sqrt(
            sum(g * g for g in gradients) + bias_gradient ** 2
        )

        scale = min(1.0, gradient_clip / max(norm, 1e-15))

        clipped = [g * scale for g in gradients]
        clipped_bias = bias_gradient * scale

        for index in range(len(self.weights)):
            self.weights[index] -= learning_rate * clipped[index]

        self.bias -= learning_rate * clipped_bias

        return norm


class GradientDescentTrainer:
    def __init__(self, config: TrainingConfig):
        config.validate()
        self.config = config
        self.rng = random.Random(config.seed)

    def fit(
        self,
        samples: Sequence[Sample],
        model: LinearRegression | None = None,
    ) -> TrainingResult:
        if not samples:
            raise ValueError("Training dataset cannot be empty.")

        for sample in samples:
            sample.validate()

        dimensions = len(samples[0].features)

        if any(len(s.features) != dimensions for s in samples):
            raise ValueError("Inconsistent feature dimensions.")

        model = model or LinearRegression(dimensions)

        if len(model.weights) != dimensions:
            raise ValueError("Model dimensions do not match dataset.")

        history = []
        best_loss = float("inf")
        stale_epochs = 0
        converged = False

        for epoch in range(1, self.config.epochs + 1):
            indices = list(range(len(samples)))

            if self.config.shuffle:
                self.rng.shuffle(indices)

            if self.config.optimizer == OptimizerType.BATCH:
                batches = [indices]

            elif self.config.optimizer == OptimizerType.STOCHASTIC:
                batches = [[index] for index in indices]

            else:
                size = self.config.batch_size
                batches = [
                    indices[start:start + size]
                    for start in range(0, len(indices), size)
                ]

            gradient_norms = []

            for batch_indices in batches:
                batch = [samples[index] for index in batch_indices]

                gradients, bias_gradient = model.compute_gradients(
                    batch,
                    self.config.regularization,
                )

                norm = model.update(
                    gradients,
                    bias_gradient,
                    self.config.learning_rate,
                    self.config.gradient_clip,
                )

                gradient_norms.append(norm)

            epoch_loss = model.loss(
                samples,
                self.config.regularization,
            )

            if not math.isfinite(epoch_loss):
                raise FloatingPointError(
                    f"Non-finite loss detected at epoch {epoch}."
                )

            history.append({
                "epoch": float(epoch),
                "loss": epoch_loss,
                "gradient_norm": statistics.mean(gradient_norms),
            })

            if best_loss - epoch_loss > self.config.tolerance:
                best_loss = epoch_loss
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= self.config.patience:
                converged = True
                break

        return TrainingResult(
            optimizer=self.config.optimizer.value,
            weights=model.weights.copy(),
            bias=model.bias,
            history=history,
            epochs_completed=len(history),
            converged=converged,
        )


def mean_squared_error(
    model: LinearRegression | TrainingResult,
    samples: Sequence[Sample],
) -> float:
    if not samples:
        raise ValueError("Evaluation dataset cannot be empty.")

    predictions = [model.predict(s.features) for s in samples]

    return sum(
        (prediction - sample.target) ** 2
        for prediction, sample in zip(predictions, samples)
    ) / len(samples)


def r_squared(
    model: LinearRegression | TrainingResult,
    samples: Sequence[Sample],
) -> float:
    if not samples:
        raise ValueError("Evaluation dataset cannot be empty.")

    mean_target = statistics.mean(s.target for s in samples)

    total = sum(
        (s.target - mean_target) ** 2 for s in samples
    )

    residual = sum(
        (model.predict(s.features) - s.target) ** 2
        for s in samples
    )

    if total == 0:
        return 1.0 if residual == 0 else 0.0

    return 1.0 - residual / total


def numerical_gradient_check() -> None:
    """Compare analytical and finite-difference gradients."""
    samples = [
        Sample((1.0, 2.0), 5.0),
        Sample((3.0, -1.0), 4.0),
    ]

    model = LinearRegression(2)
    model.weights = [0.5, -0.2]
    model.bias = 0.3

    analytical, analytical_bias = model.compute_gradients(
        samples, 0.0
    )

    epsilon = 1e-6

    for index in range(2):
        original = model.weights[index]

        model.weights[index] = original + epsilon
        upper = model.loss(samples)

        model.weights[index] = original - epsilon
        lower = model.loss(samples)

        model.weights[index] = original

        numerical = (upper - lower) / (2 * epsilon)

        assert abs(numerical - analytical[index]) < 1e-5

    original_bias = model.bias

    model.bias = original_bias + epsilon
    upper = model.loss(samples)

    model.bias = original_bias - epsilon
    lower = model.loss(samples)

    model.bias = original_bias

    numerical_bias = (upper - lower) / (2 * epsilon)

    assert abs(numerical_bias - analytical_bias) < 1e-5


def export_history(
    results: Sequence[TrainingResult],
    filename: str = "gradient_descent_results.csv",
) -> None:
    path = Path(filename)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "optimizer",
            "epoch",
            "training_loss",
            "gradient_norm",
        ])

        for result in results:
            for entry in result.history:
                writer.writerow([
                    result.optimizer,
                    int(entry["epoch"]),
                    entry["loss"],
                    entry["gradient_norm"],
                ])


def main() -> None:
    print("=" * 72)
    print("GRADIENT DESCENT: OPTIMIZATION STRATEGY COMPARISON")
    print("=" * 72)

    training_data = Dataset.generate(
        count=240,
        dimensions=2,
        noise=0.5,
        seed=42,
    )

    test_data = Dataset.generate(
        count=80,
        dimensions=2,
        noise=0.5,
        seed=99,
    )

    numerical_gradient_check()
    print("\nNumerical gradient verification: PASSED")

    configurations = [
        TrainingConfig(
            optimizer=OptimizerType.BATCH,
            learning_rate=0.01,
            epochs=150,
            patience=25,
        ),
        TrainingConfig(
            optimizer=OptimizerType.STOCHASTIC,
            learning_rate=0.003,
            epochs=150,
            patience=25,
        ),
        TrainingConfig(
            optimizer=OptimizerType.MINI_BATCH,
            learning_rate=0.01,
            epochs=150,
            batch_size=16,
            patience=25,
        ),
    ]

    results = []

    for config in configurations:
        model = LinearRegression(dimensions=2)

        trainer = GradientDescentTrainer(config)
        result = trainer.fit(training_data, model)

        results.append(result)

        mse = mean_squared_error(result, test_data)
        score = r_squared(result, test_data)

        print(f"\nOptimizer: {result.optimizer.upper()}")
        print(f"Epochs completed: {result.epochs_completed}")
        print(f"Converged by patience: {result.converged}")
        print(f"Weights: {[round(w, 4) for w in result.weights]}")
        print(f"Bias: {result.bias:.4f}")
        print(f"Training loss: {result.history[-1]['loss']:.6f}")
        print(f"Test MSE: {mse:.6f}")
        print(f"Test R-squared: {score:.6f}")

    export_history(results)

    print("\nResults exported to gradient_descent_results.csv")

    # Demonstrate validation failure handling.
    try:
        TrainingConfig(
            optimizer=OptimizerType.MINI_BATCH,
            learning_rate=-0.01,
        ).validate()
    except ValueError as error:
        print(f"Validation test: PASSED ({error})")

    print("\nOptimization comparison completed.")


if __name__ == "__main__":
    main()
