#!/usr/bin/env python3
"""
Batch Normalization: Internal Normalization and Training vs Inference.

Implements:
- Feature-wise batch statistics.
- Normalization using mean and variance.
- Learnable scale (gamma) and shift (beta).
- Running mean and variance.
- Training and inference modes.
- Backpropagation through batch normalization.
- Numerical validation and edge-case handling.
- A reproducible training/inference demonstration.

No external dependencies.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence


class Mode(Enum):
    TRAINING = "training"
    INFERENCE = "inference"


class BatchNormalizationError(Exception):
    """Base exception for batch normalization failures."""


class ShapeError(BatchNormalizationError):
    """Raised when input dimensions are inconsistent."""


class NumericalError(BatchNormalizationError):
    """Raised when invalid numerical values are encountered."""


@dataclass
class BatchStatistics:
    mean: list[float]
    variance: list[float]
    sample_count: int


@dataclass
class ForwardCache:
    normalized: list[list[float]]
    inverse_std: list[float]
    mode: Mode
    sample_count: int
    input_shape: tuple[int, int]


@dataclass
class BatchNormalization:
    """
    Feature-wise Batch Normalization.

    Training:
        Uses current mini-batch statistics.
        Updates running statistics.

    Inference:
        Uses stored running statistics.
        Does not update model state.

    Momentum convention:
        running = momentum * running
                  + (1 - momentum) * batch_statistic
    """

    features: int
    epsilon: float = 1e-5
    momentum: float = 0.9

    gamma: list[float] = field(init=False)
    beta: list[float] = field(init=False)

    running_mean: list[float] = field(init=False)
    running_variance: list[float] = field(init=False)

    batches_seen: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.features <= 0:
            raise ValueError("features must be positive")

        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

        if not math.isfinite(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")

        self.gamma = [1.0] * self.features
        self.beta = [0.0] * self.features
        self.running_mean = [0.0] * self.features
        self.running_variance = [1.0] * self.features

    def _validate(self, values: Sequence[Sequence[float]]) -> None:
        if not values:
            raise ShapeError("Input batch cannot be empty")

        for row in values:
            if len(row) != self.features:
                raise ShapeError(
                    f"Expected {self.features} features per row"
                )

            for value in row:
                if not math.isfinite(float(value)):
                    raise NumericalError("Input contains non-finite values")

    def calculate_statistics(
        self,
        values: Sequence[Sequence[float]]
    ) -> BatchStatistics:
        """Calculate population mean and variance per feature."""

        self._validate(values)

        count = len(values)

        mean = [
            sum(float(row[j]) for row in values) / count
            for j in range(self.features)
        ]

        variance = [
            sum(
                (float(row[j]) - mean[j]) ** 2
                for row in values
            ) / count
            for j in range(self.features)
        ]

        return BatchStatistics(mean, variance, count)

    def _update_running_statistics(
        self,
        statistics: BatchStatistics
    ) -> None:
        for j in range(self.features):
            self.running_mean[j] = (
                self.momentum * self.running_mean[j]
                + (1.0 - self.momentum) * statistics.mean[j]
            )

            self.running_variance[j] = (
                self.momentum * self.running_variance[j]
                + (1.0 - self.momentum) * statistics.variance[j]
            )

        self.batches_seen += 1

    def forward(
        self,
        values: Sequence[Sequence[float]],
        mode: Mode = Mode.TRAINING
    ) -> tuple[list[list[float]], ForwardCache]:
        """
        Apply normalization and affine transformation.

        y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
        """

        self._validate(values)

        if not isinstance(mode, Mode):
            raise ValueError("mode must be a Mode enum value")

        count = len(values)

        if mode is Mode.TRAINING:
            statistics = self.calculate_statistics(values)
            mean = statistics.mean
            variance = statistics.variance
            self._update_running_statistics(statistics)
        else:
            if self.batches_seen == 0:
                raise BatchNormalizationError(
                    "Inference requires initialized running statistics"
                )

            mean = self.running_mean
            variance = self.running_variance

        inverse_std = [
            1.0 / math.sqrt(variance[j] + self.epsilon)
            for j in range(self.features)
        ]

        normalized = []
        output = []

        for row in values:
            normalized_row = []
            output_row = []

            for j, value in enumerate(row):
                normalized_value = (
                    (float(value) - mean[j]) * inverse_std[j]
                )

                normalized_row.append(normalized_value)

                output_value = (
                    self.gamma[j] * normalized_value + self.beta[j]
                )

                if not math.isfinite(output_value):
                    raise NumericalError("Non-finite normalization output")

                output_row.append(output_value)

            normalized.append(normalized_row)
            output.append(output_row)

        cache = ForwardCache(
            normalized=normalized,
            inverse_std=inverse_std,
            mode=mode,
            sample_count=count,
            input_shape=(count, self.features)
        )

        return output, cache

    def backward(
        self,
        upstream_gradient: Sequence[Sequence[float]],
        cache: ForwardCache
    ) -> tuple[list[list[float]], list[float], list[float]]:
        """
        Compute gradients for input, gamma and beta.

        For each feature:

        dBeta = sum(dY)
        dGamma = sum(dY * Xhat)

        dX = gamma * inv_std / N *
             (N*dY - sum(dY) - Xhat*sum(dY*Xhat))
        """

        if cache.mode is not Mode.TRAINING:
            raise BatchNormalizationError(
                "Training gradients require a training forward pass"
            )

        if len(upstream_gradient) != cache.sample_count:
            raise ShapeError("Gradient batch size does not match input")

        for row in upstream_gradient:
            if len(row) != self.features:
                raise ShapeError("Gradient feature count mismatch")

            if not all(math.isfinite(float(v)) for v in row):
                raise NumericalError("Gradient contains non-finite values")

        count = cache.sample_count

        d_gamma = [0.0] * self.features
        d_beta = [0.0] * self.features

        for j in range(self.features):
            for i in range(count):
                gradient = float(upstream_gradient[i][j])

                d_gamma[j] += (
                    gradient * cache.normalized[i][j]
                )

                d_beta[j] += gradient

        d_input = [
            [0.0] * self.features
            for _ in range(count)
        ]

        for j in range(self.features):
            gradient_sum = d_beta[j]
            normalized_gradient_sum = d_gamma[j]

            for i in range(count):
                gradient = float(upstream_gradient[i][j])

                d_input[i][j] = (
                    self.gamma[j]
                    * cache.inverse_std[j]
                    / count
                    * (
                        count * gradient
                        - gradient_sum
                        - cache.normalized[i][j]
                        * normalized_gradient_sum
                    )
                )

        return d_input, d_gamma, d_beta

    def update_parameters(
        self,
        d_gamma: Sequence[float],
        d_beta: Sequence[float],
        learning_rate: float
    ) -> None:
        """Apply a gradient descent update to affine parameters."""

        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite")

        if len(d_gamma) != self.features or len(d_beta) != self.features:
            raise ShapeError("Parameter gradient dimensions mismatch")

        for j in range(self.features):
            if not math.isfinite(d_gamma[j]) or not math.isfinite(d_beta[j]):
                raise NumericalError("Invalid parameter gradient")

        for j in range(self.features):
            self.gamma[j] -= learning_rate * d_gamma[j]
            self.beta[j] -= learning_rate * d_beta[j]

    def reset_running_statistics(self) -> None:
        """Reset inference statistics before a new calibration process."""

        self.running_mean = [0.0] * self.features
        self.running_variance = [1.0] * self.features
        self.batches_seen = 0

    def save(self, path: str | Path) -> None:
        """Persist learned affine parameters and inference statistics."""

        data = {
            "features": self.features,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "gamma": self.gamma,
            "beta": self.beta,
            "running_mean": self.running_mean,
            "running_variance": self.running_variance,
            "batches_seen": self.batches_seen
        }

        destination = Path(path)

        with destination.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path: str | Path) -> BatchNormalization:
        """Restore a previously trained normalization layer."""

        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)

        layer = cls(
            features=data["features"],
            epsilon=data["epsilon"],
            momentum=data["momentum"]
        )

        for name in (
            "gamma",
            "beta",
            "running_mean",
            "running_variance"
        ):
            values = data[name]

            if len(values) != layer.features:
                raise ShapeError(f"Invalid serialized field: {name}")

            if not all(math.isfinite(float(v)) for v in values):
                raise NumericalError(f"Invalid values in {name}")

            setattr(layer, name, list(values))

        if not isinstance(data["batches_seen"], int):
            raise ValueError("Invalid batches_seen")

        if data["batches_seen"] < 0:
            raise ValueError("Negative batches_seen")

        if any(v < 0 for v in layer.running_variance):
            raise NumericalError("Running variance cannot be negative")

        layer.batches_seen = data["batches_seen"]

        return layer


def demonstrate() -> None:
    print("=" * 65)
    print("BATCH NORMALIZATION")
    print("=" * 65)

    layer = BatchNormalization(
        features=3,
        epsilon=1e-5,
        momentum=0.9
    )

    batch = [
        [2.0, 4.0, 6.0],
        [4.0, 8.0, 10.0],
        [6.0, 12.0, 14.0],
        [8.0, 16.0, 18.0]
    ]

    print("\n1. Training forward pass")

    training_output, cache = layer.forward(
        batch,
        Mode.TRAINING
    )

    for row in training_output:
        print([round(v, 5) for v in row])

    print("\nRunning mean:", layer.running_mean)
    print("Running variance:", layer.running_variance)

    print("\n2. Backpropagation")

    upstream_gradient = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ]

    d_input, d_gamma, d_beta = layer.backward(
        upstream_gradient,
        cache
    )

    print("Input gradients:", d_input)
    print("Gamma gradients:", d_gamma)
    print("Beta gradients:", d_beta)

    layer.update_parameters(
        d_gamma,
        d_beta,
        learning_rate=0.01
    )

    print("\nUpdated gamma:", layer.gamma)
    print("Updated beta:", layer.beta)

    print("\n3. Inference forward pass")

    inference_output, _ = layer.forward(
        [[5.0, 10.0, 12.0]],
        Mode.INFERENCE
    )

    print("Inference output:", inference_output)

    print("\n4. State persistence")

    path = Path("batch_normalization_state.json")
    layer.save(path)

    restored = BatchNormalization.load(path)

    assert restored.gamma == layer.gamma
    assert restored.running_mean == layer.running_mean

    print("Model state successfully restored")

    print("\n5. Edge-case validation")

    try:
        layer.forward([], Mode.TRAINING)
    except ShapeError as error:
        print("Empty batch rejected:", error)

    try:
        layer.forward([[1.0, 2.0]], Mode.TRAINING)
    except ShapeError as error:
        print("Incorrect feature count rejected:", error)

    try:
        layer.forward([[1.0, float("nan"), 3.0]], Mode.TRAINING)
    except NumericalError as error:
        print("Invalid numerical input rejected:", error)

    print("\n6. Constant-feature behavior")

    constant_layer = BatchNormalization(features=1)

    result, _ = constant_layer.forward(
        [[5.0], [5.0], [5.0]],
        Mode.TRAINING
    )

    assert all(abs(row[0]) < 1e-8 for row in result)

    print("Constant feature normalized:", result)

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    demonstrate()
