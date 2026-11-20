#!/usr/bin/env python3
"""
Deep Learning Training: Epochs, Batches, Learning Rate, and Checkpoints.

A self-contained training simulator using mini-batch stochastic gradient
descent for linear regression. Includes validation, learning-rate scheduling,
early stopping, checkpoint persistence, and training recovery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class TrainingError(Exception):
    """Base exception for training failures."""


class ConfigurationError(TrainingError):
    """Invalid training configuration."""


class CheckpointError(TrainingError):
    """Checkpoint persistence or recovery failure."""


class TrainingState(str, Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    EARLY_STOPPED = "early_stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 0.05
    min_learning_rate: float = 0.00001
    lr_decay: float = 0.98
    patience: int = 8
    min_delta: float = 1e-8
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 5

    def validate(self) -> None:
        if self.epochs < 1:
            raise ConfigurationError("Epochs must be positive.")
        if self.batch_size < 1:
            raise ConfigurationError("Batch size must be positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ConfigurationError("Learning rate must be positive and finite.")
        if not 0 < self.lr_decay <= 1:
            raise ConfigurationError("LR decay must be in (0, 1].")
        if self.min_learning_rate <= 0:
            raise ConfigurationError("Minimum learning rate must be positive.")
        if self.patience < 1:
            raise ConfigurationError("Patience must be positive.")
        if self.checkpoint_every < 1:
            raise ConfigurationError("Checkpoint interval must be positive.")


@dataclass
class TrainingMetrics:
    epoch: int
    training_loss: float
    validation_loss: float
    learning_rate: float
    batches: int


@dataclass
class Model:
    weights: list[float]
    bias: float

    @classmethod
    def initialize(cls, feature_count: int, seed: int) -> Model:
        rng = random.Random(seed)
        return cls(
            weights=[rng.uniform(-0.1, 0.1) for _ in range(feature_count)],
            bias=0.0,
        )

    def predict(self, features: list[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("Feature count does not match model.")
        return sum(w * x for w, x in zip(self.weights, features)) + self.bias


class Dataset:
    def __init__(
        self,
        features: list[list[float]],
        targets: list[float],
    ):
        if not features or len(features) != len(targets):
            raise ValueError("Dataset must contain matching nonempty features and targets.")

        feature_count = len(features[0])

        if feature_count == 0:
            raise ValueError("At least one feature is required.")

        for row in features:
            if len(row) != feature_count:
                raise ValueError("Inconsistent feature dimensions.")
            if not all(math.isfinite(value) for value in row):
                raise ValueError("Features must be finite.")

        if not all(math.isfinite(value) for value in targets):
            raise ValueError("Targets must be finite.")

        self.features = [row[:] for row in features]
        self.targets = targets[:]
        self.feature_count = feature_count

    def __len__(self) -> int:
        return len(self.targets)


class CheckpointManager:
    """Persists model parameters and training progress atomically."""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: Model,
        epoch: int,
        learning_rate: float,
        best_loss: float,
        best_epoch: int,
    ) -> Path:
        if epoch < 0 or not math.isfinite(learning_rate):
            raise CheckpointError("Invalid checkpoint state.")

        payload = {
            "format_version": 1,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "model": asdict(model),
        }

        destination = self.directory / f"epoch_{epoch:05d}.json"
        temporary_path: Optional[str] = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                json.dump(payload, temporary, allow_nan=False)
                temporary.flush()
                os.fsync(temporary.fileno())

            os.replace(temporary_path, destination)
            return destination

        except (OSError, ValueError, TypeError) as exc:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise CheckpointError(f"Checkpoint save failed: {exc}") from exc

    def load(self, path: str) -> dict:
        checkpoint_path = Path(path)

        try:
            if checkpoint_path.resolve().parent != self.directory.resolve():
                raise CheckpointError("Checkpoint must belong to the configured directory.")

            with checkpoint_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if data.get("format_version") != 1:
                raise CheckpointError("Unsupported checkpoint format.")

            model_data = data["model"]
            weights = model_data["weights"]
            bias = model_data["bias"]

            if not isinstance(weights, list) or not weights:
                raise CheckpointError("Invalid model weights.")

            if not all(isinstance(w, (int, float)) and math.isfinite(w) for w in weights):
                raise CheckpointError("Non-finite model weights.")

            if not isinstance(bias, (int, float)) or not math.isfinite(bias):
                raise CheckpointError("Invalid model bias.")

            if data["epoch"] < 0 or data["best_epoch"] < 0:
                raise CheckpointError("Invalid checkpoint epoch.")

            if not math.isfinite(data["learning_rate"]) or data["learning_rate"] <= 0:
                raise CheckpointError("Invalid checkpoint learning rate.")

            if not math.isfinite(data["best_loss"]):
                raise CheckpointError("Invalid best loss.")

            return data

        except CheckpointError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CheckpointError(f"Checkpoint loading failed: {exc}") from exc


class LearningRateScheduler:
    def __init__(self, initial_rate: float, decay: float, minimum: float):
        self.initial_rate = initial_rate
        self.decay = decay
        self.minimum = minimum

    def get_rate(self, epoch: int) -> float:
        return max(self.minimum, self.initial_rate * self.decay ** (epoch - 1))


class Trainer:
    def __init__(
        self,
        model: Model,
        training_data: Dataset,
        validation_data: Dataset,
        config: TrainingConfig,
    ):
        config.validate()

        if training_data.feature_count != validation_data.feature_count:
            raise ConfigurationError("Training and validation feature counts differ.")

        self.model = model
        self.training_data = training_data
        self.validation_data = validation_data
        self.config = config
        self.scheduler = LearningRateScheduler(
            config.learning_rate,
            config.lr_decay,
            config.min_learning_rate,
        )
        self.checkpoints = CheckpointManager(config.checkpoint_dir)

        self.state = TrainingState.INITIALIZED
        self.history: list[TrainingMetrics] = []
        self.best_loss = math.inf
        self.best_epoch = 0
        self.best_model: Optional[Model] = None
        self.stale_epochs = 0

    def calculate_loss(self, dataset: Dataset) -> float:
        total = 0.0

        for features, target in zip(dataset.features, dataset.targets):
            error = self.model.predict(features) - target
            total += error * error

        return total / len(dataset)

    def train_batch(self, indices: list[int], learning_rate: float) -> None:
        count = len(indices)
        weight_gradients = [0.0] * len(self.model.weights)
        bias_gradient = 0.0

        # Accumulate gradients before updating parameters.
        for index in indices:
            features = self.training_data.features[index]
            target = self.training_data.targets[index]

            error = self.model.predict(features) - target

            for j, value in enumerate(features):
                weight_gradients[j] += 2.0 * error * value / count

            bias_gradient += 2.0 * error / count

        self.model.weights = [
            weight - learning_rate * gradient
            for weight, gradient in zip(self.model.weights, weight_gradients)
        ]

        self.model.bias -= learning_rate * bias_gradient

        if not all(math.isfinite(w) for w in self.model.weights):
            raise TrainingError("Non-finite model weights detected.")

        if not math.isfinite(self.model.bias):
            raise TrainingError("Non-finite model bias detected.")

    def restore(self, path: str) -> None:
        data = self.checkpoints.load(path)

        if len(data["model"]["weights"]) != self.training_data.feature_count:
            raise CheckpointError("Checkpoint feature count mismatch.")

        self.model = Model(
            weights=data["model"]["weights"],
            bias=data["model"]["bias"],
        )

        self.best_loss = data["best_loss"]
        self.best_epoch = data["best_epoch"]

        self.best_model = Model(
            weights=self.model.weights[:],
            bias=self.model.bias,
        )

        self.start_epoch = data["epoch"] + 1

    def fit(self) -> list[TrainingMetrics]:
        self.state = TrainingState.RUNNING
        start_epoch = getattr(self, "start_epoch", 1)
        rng = random.Random(self.config.seed + start_epoch)

        try:
            for epoch in range(start_epoch, self.config.epochs + 1):
                learning_rate = self.scheduler.get_rate(epoch)
                indices = list(range(len(self.training_data)))
                rng.shuffle(indices)

                batch_count = 0

                for start in range(0, len(indices), self.config.batch_size):
                    batch_indices = indices[
                        start:start + self.config.batch_size
                    ]

                    self.train_batch(batch_indices, learning_rate)
                    batch_count += 1

                training_loss = self.calculate_loss(self.training_data)
                validation_loss = self.calculate_loss(self.validation_data)

                metrics = TrainingMetrics(
                    epoch=epoch,
                    training_loss=training_loss,
                    validation_loss=validation_loss,
                    learning_rate=learning_rate,
                    batches=batch_count,
                )

                self.history.append(metrics)

                improved = validation_loss < self.best_loss - self.config.min_delta

                if improved:
                    self.best_loss = validation_loss
                    self.best_epoch = epoch
                    self.best_model = Model(
                        weights=self.model.weights[:],
                        bias=self.model.bias,
                    )
                    self.stale_epochs = 0
                else:
                    self.stale_epochs += 1

                print(
                    f"Epoch {epoch:03d}/{self.config.epochs} | "
                    f"batches={batch_count} | "
                    f"lr={learning_rate:.6f} | "
                    f"train_loss={training_loss:.6f} | "
                    f"val_loss={validation_loss:.6f}"
                )

                if epoch % self.config.checkpoint_every == 0 or improved:
                    self.checkpoints.save(
                        self.model,
                        epoch,
                        learning_rate,
                        self.best_loss,
                        self.best_epoch,
                    )

                if self.stale_epochs >= self.config.patience:
                    self.state = TrainingState.EARLY_STOPPED
                    print("Early stopping triggered.")
                    break

            if self.best_model is not None:
                self.model = self.best_model

            if self.state != TrainingState.EARLY_STOPPED:
                self.state = TrainingState.COMPLETED

            return self.history

        except Exception:
            self.state = TrainingState.FAILED
            raise


def generate_dataset(
    sample_count: int,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Generate deterministic synthetic regression data."""
    rng = random.Random(seed)

    features = []
    targets = []

    for _ in range(sample_count):
        x1 = rng.uniform(-2.0, 2.0)
        x2 = rng.uniform(-2.0, 2.0)
        noise = rng.gauss(0.0, 0.05)

        features.append([x1, x2])
        targets.append(3.0 * x1 - 2.0 * x2 + 1.5 + noise)

    split = int(sample_count * 0.8)

    if split == 0 or split == sample_count:
        raise ValueError("Dataset split produced an empty partition.")

    return (
        Dataset(features[:split], targets[:split]),
        Dataset(features[split:], targets[split:]),
    )


def run_tests() -> None:
    config = TrainingConfig(epochs=2, batch_size=4)
    config.validate()

    dataset = Dataset([[1.0], [2.0]], [2.0, 4.0])
    model = Model([1.0], 0.0)

    assert abs(model.predict([2.0]) - 2.0) < 1e-9
    assert abs(0.0 - dataset.targets[0]) < 3.0

    scheduler = LearningRateScheduler(0.1, 0.9, 0.001)
    assert scheduler.get_rate(1) == 0.1
    assert scheduler.get_rate(2) < scheduler.get_rate(1)

    try:
        Dataset([[1.0], [1.0, 2.0]], [1.0, 2.0])
        raise AssertionError("Invalid dataset was accepted.")
    except ValueError:
        pass

    print("All unit assertions passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mini-batch deep learning training simulator."
    )

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--tests", action="store_true")

    args = parser.parse_args()

    if args.tests:
        run_tests()
        return

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    config.validate()

    training_data, validation_data = generate_dataset(200, config.seed)

    model = Model.initialize(training_data.feature_count, config.seed)

    trainer = Trainer(
        model,
        training_data,
        validation_data,
        config,
    )

    if args.resume:
        trainer.restore(args.resume)

    trainer.fit()

    print("\nTraining results")
    print("----------------")
    print("State:", trainer.state.value)
    print("Best epoch:", trainer.best_epoch)
    print("Best validation loss:", f"{trainer.best_loss:.6f}")
    print("Final model weights:", trainer.model.weights)
    print("Final model bias:", round(trainer.model.bias, 6))


if __name__ == "__main__":
    main()
