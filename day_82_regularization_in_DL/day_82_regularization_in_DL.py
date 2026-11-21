#!/usr/bin/env python3
"""
Regularization in Deep Learning
Techniques: Dropout, Weight Decay, Early Stopping, Data Augmentation.

A self-contained neural-network implementation using only Python's
standard library. Demonstrates forward propagation, backpropagation,
regularization, validation monitoring, and model selection.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence


class RegularizationType(Enum):
    NONE = "none"
    DROPOUT = "dropout"
    WEIGHT_DECAY = "weight_decay"
    COMBINED = "combined"


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.04
    weight_decay: float = 0.001
    dropout_rate: float = 0.20
    epochs: int = 150
    patience: int = 15
    min_delta: float = 1e-5
    batch_size: int = 16
    augmentation_noise: float = 0.0
    seed: int = 42

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValidationError("Learning rate must be positive.")
        if self.weight_decay < 0:
            raise ValidationError("Weight decay cannot be negative.")
        if not 0 <= self.dropout_rate < 1:
            raise ValidationError("Dropout rate must be in [0, 1).")
        if self.epochs <= 0 or self.patience <= 0:
            raise ValidationError("Epochs and patience must be positive.")
        if self.batch_size <= 0:
            raise ValidationError("Batch size must be positive.")
        if self.augmentation_noise < 0:
            raise ValidationError("Augmentation noise cannot be negative.")


class Dataset:
    """Stores and validates numerical training examples."""

    def __init__(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int]
    ):
        if not features or len(features) != len(labels):
            raise ValidationError("Invalid dataset dimensions.")

        dimension = len(features[0])

        if dimension == 0:
            raise ValidationError("Feature vectors cannot be empty.")

        if any(len(row) != dimension for row in features):
            raise ValidationError("Inconsistent feature dimensions.")

        if any(not all(math.isfinite(v) for v in row) for row in features):
            raise ValidationError("Features must be finite.")

        if any(label not in (0, 1) for label in labels):
            raise ValidationError("Labels must be binary.")

        self.features = [list(map(float, row)) for row in features]
        self.labels = list(labels)
        self.dimension = dimension

    def __len__(self):
        return len(self.labels)

    def split(self, validation_ratio=0.25, seed=42):
        if not 0 < validation_ratio < 1:
            raise ValidationError("Invalid validation ratio.")

        indices = list(range(len(self)))
        random.Random(seed).shuffle(indices)

        split_point = int(len(self) * (1 - validation_ratio))

        if split_point == 0 or split_point == len(self):
            raise ValidationError("Insufficient samples for splitting.")

        def create_dataset(selected):
            return Dataset(
                [self.features[i] for i in selected],
                [self.labels[i] for i in selected]
            )

        return (
            create_dataset(indices[:split_point]),
            create_dataset(indices[split_point:])
        )


class DataAugmenter:
    """Applies label-preserving Gaussian noise to numerical features."""

    def __init__(self, noise_std=0.05, seed=42):
        if noise_std < 0:
            raise ValidationError("Noise standard deviation cannot be negative.")
        self.noise_std = noise_std
        self.random = random.Random(seed)

    def transform(self, features):
        return [
            value + self.random.gauss(0, self.noise_std)
            for value in features
        ]


class NeuralNetwork:
    """A two-layer binary classifier with optional hidden-layer dropout."""

    def __init__(self, input_size, hidden_size=8, seed=42):
        if input_size <= 0 or hidden_size <= 0:
            raise ValidationError("Network dimensions must be positive.")

        rng = random.Random(seed)

        self.w1 = [
            [rng.uniform(-0.5, 0.5) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.b1 = [0.0] * hidden_size

        self.w2 = [rng.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    @staticmethod
    def sigmoid(value):
        value = max(-500, min(500, value))
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def relu(value):
        return max(0.0, value)

    def forward(self, features, training=False, dropout_rate=0.0, rng=None):
        hidden = [
            self.relu(sum(w * x for w, x in zip(weights, features)) + bias)
            for weights, bias in zip(self.w1, self.b1)
        ]

        mask = [1.0] * len(hidden)

        # Inverted dropout preserves the expected activation during training.
        if training and dropout_rate > 0:
            if rng is None:
                raise ValueError("A random generator is required for dropout.")

            scale = 1.0 / (1.0 - dropout_rate)

            mask = [
                scale if rng.random() >= dropout_rate else 0.0
                for _ in hidden
            ]

        dropped_hidden = [
            activation * multiplier
            for activation, multiplier in zip(hidden, mask)
        ]

        logit = sum(
            w * h for w, h in zip(self.w2, dropped_hidden)
        ) + self.b2

        probability = self.sigmoid(logit)

        return probability, (features, hidden, dropped_hidden, mask)

    def predict(self, features):
        probability, _ = self.forward(features)
        return int(probability >= 0.5)

    def train_batch(
        self,
        batch,
        learning_rate,
        dropout_rate,
        weight_decay,
        rng
    ):
        hidden_count = len(self.w1)
        input_count = len(self.w1[0])

        dw1 = [[0.0] * input_count for _ in range(hidden_count)]
        db1 = [0.0] * hidden_count
        dw2 = [0.0] * hidden_count
        db2 = 0.0

        for features, label in batch:
            probability, cache = self.forward(
                features,
                training=True,
                dropout_rate=dropout_rate,
                rng=rng
            )

            original, hidden, dropped, mask = cache

            # Binary cross-entropy derivative with sigmoid output.
            output_error = probability - label

            for j in range(hidden_count):
                dw2[j] += output_error * dropped[j]

            db2 += output_error

            for j in range(hidden_count):
                if hidden[j] <= 0:
                    continue

                hidden_error = (
                    output_error * self.w2[j] * mask[j]
                )

                for k in range(input_count):
                    dw1[j][k] += hidden_error * original[k]

                db1[j] += hidden_error

        count = len(batch)

        if count == 0:
            raise ValidationError("Cannot train on an empty batch.")

        # L2 weight decay is applied to weights, not biases.
        for j in range(hidden_count):
            for k in range(input_count):
                gradient = dw1[j][k] / count
                gradient += weight_decay * self.w1[j][k]
                self.w1[j][k] -= learning_rate * gradient

            self.b1[j] -= learning_rate * db1[j] / count

            self.w2[j] -= learning_rate * (
                dw2[j] / count + weight_decay * self.w2[j]
            )

        self.b2 -= learning_rate * db2 / count

    def loss(self, dataset):
        epsilon = 1e-12
        total = 0.0

        for features, label in zip(dataset.features, dataset.labels):
            probability, _ = self.forward(features)

            total -= (
                label * math.log(max(epsilon, probability))
                + (1 - label) * math.log(max(epsilon, 1 - probability))
            )

        return total / len(dataset)

    def accuracy(self, dataset):
        correct = sum(
            self.predict(features) == label
            for features, label in zip(dataset.features, dataset.labels)
        )
        return correct / len(dataset)

    def regularized_loss(self, dataset, weight_decay):
        penalty = weight_decay * (
            sum(w * w for row in self.w1 for w in row)
            + sum(w * w for w in self.w2)
        ) / 2

        return self.loss(dataset) + penalty


@dataclass
class TrainingResult:
    model: NeuralNetwork
    best_epoch: int
    stopped_epoch: int
    best_validation_loss: float
    training_loss: float
    validation_accuracy: float
    history: list[dict]


class Trainer:
    """Coordinates optimization, augmentation and early stopping."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def fit(self, training_data, validation_data):
        if training_data.dimension != validation_data.dimension:
            raise ValidationError("Feature dimensions do not match.")

        if len(training_data) == 0 or len(validation_data) == 0:
            raise ValidationError("Training and validation sets are required.")

        config = self.config

        model = NeuralNetwork(
            training_data.dimension,
            hidden_size=8,
            seed=config.seed
        )

        rng = random.Random(config.seed)
        augmenter = DataAugmenter(
            config.augmentation_noise,
            config.seed + 1
        )

        best_loss = float("inf")
        best_model = copy.deepcopy(model)
        best_epoch = 0
        stale_epochs = 0
        history = []

        for epoch in range(1, config.epochs + 1):
            indices = list(range(len(training_data)))
            rng.shuffle(indices)

            for start in range(0, len(indices), config.batch_size):
                batch_indices = indices[
                    start:start + config.batch_size
                ]

                batch = []

                for index in batch_indices:
                    features = training_data.features[index]

                    if config.augmentation_noise > 0:
                        features = augmenter.transform(features)

                    batch.append(
                        (features, training_data.labels[index])
                    )

                model.train_batch(
                    batch,
                    config.learning_rate,
                    config.dropout_rate,
                    config.weight_decay,
                    rng
                )

            train_loss = model.loss(training_data)
            validation_loss = model.loss(validation_data)

            history.append({
                "epoch": epoch,
                "training_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": model.accuracy(validation_data)
            })

            if validation_loss < best_loss - config.min_delta:
                best_loss = validation_loss
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= config.patience:
                break

        return TrainingResult(
            model=best_model,
            best_epoch=best_epoch,
            stopped_epoch=len(history),
            best_validation_loss=best_loss,
            training_loss=best_model.loss(training_data),
            validation_accuracy=best_model.accuracy(validation_data),
            history=history
        )


def generate_dataset(samples=240, seed=42):
    """Generates a noisy nonlinear binary classification dataset."""

    if samples < 20:
        raise ValidationError("At least 20 samples are required.")

    rng = random.Random(seed)
    features = []
    labels = []

    for _ in range(samples):
        x = rng.uniform(-2.5, 2.5)
        y = rng.uniform(-2.5, 2.5)

        # Nonlinear boundary with controlled label noise.
        score = x * x + y * y - 2.0
        label = int(score < 0)

        if rng.random() < 0.04:
            label = 1 - label

        features.append([x, y])
        labels.append(label)

    return Dataset(features, labels)


def run_experiment(name, config, training_data, validation_data):
    print(f"\nExperiment: {name}")

    result = Trainer(config).fit(training_data, validation_data)

    print(f"Best epoch: {result.best_epoch}")
    print(f"Stopped epoch: {result.stopped_epoch}")
    print(f"Training loss: {result.training_loss:.5f}")
    print(f"Validation loss: {result.best_validation_loss:.5f}")
    print(f"Validation accuracy: {result.validation_accuracy:.2%}")

    return result


def main():
    dataset = generate_dataset()
    training_data, validation_data = dataset.split()

    experiments = {
        "Baseline": TrainingConfig(
            dropout_rate=0,
            weight_decay=0,
            augmentation_noise=0
        ),
        "Dropout": TrainingConfig(
            dropout_rate=0.25,
            weight_decay=0
        ),
        "Weight Decay": TrainingConfig(
            dropout_rate=0,
            weight_decay=0.01
        ),
        "Combined Regularization": TrainingConfig(
            dropout_rate=0.20,
            weight_decay=0.005,
            augmentation_noise=0.04,
            patience=18
        )
    }

    results = {}

    for name, config in experiments.items():
        results[name] = run_experiment(
            name,
            config,
            training_data,
            validation_data
        )

    print("\nExperiment Comparison")
    print("-" * 75)
    print(f"{'Method':30} {'Best Epoch':12} {'Val Loss':12} {'Accuracy':12}")

    for name, result in results.items():
        print(
            f"{name:30} "
            f"{result.best_epoch:<12} "
            f"{result.best_validation_loss:<12.5f} "
            f"{result.validation_accuracy:.2%}"
        )

    best_name = min(
        results,
        key=lambda name: results[name].best_validation_loss
    )

    best = results[best_name]

    print(f"\nSelected experiment: {best_name}")
    print("Sample predictions:")

    for features in [[0.2, 0.3], [2.0, 2.0], [-0.5, 0.8]]:
        print(features, "->", best.model.predict(features))

    # Basic integrity assertions.
    assert 0 <= best.validation_accuracy <= 1
    assert best.best_epoch > 0
    assert best.stopped_epoch >= best.best_epoch

    print("\nAll validation assertions passed.")


if __name__ == "__main__":
    main()
