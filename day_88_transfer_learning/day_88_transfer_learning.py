#!/usr/bin/env python3
"""
TRANSFER LEARNING
Feature Extraction, Fine-Tuning, and Pretrained Models

A self-contained implementation using Python's standard library.

Demonstrates:
1. Pretrained neural network parameters.
2. Frozen feature extraction.
3. Training a new classification head.
4. Selective layer unfreezing.
5. Fine-tuning with a smaller learning rate.
6. Evaluation and model comparison.
7. Model persistence and validation.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------
# Configuration and domain types
# ---------------------------------------------------------

class TransferStrategy(Enum):
    FEATURE_EXTRACTION = "feature_extraction"
    FINE_TUNING = "fine_tuning"


@dataclass
class TrainingConfig:
    epochs: int = 150
    learning_rate: float = 0.05
    fine_tune_learning_rate: float = 0.005
    train_ratio: float = 0.8
    seed: int = 42

    def validate(self):
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if self.fine_tune_learning_rate <= 0:
            raise ValueError("Fine-tuning learning rate must be positive.")
        if not 0 < self.train_ratio < 1:
            raise ValueError("Train ratio must be between zero and one.")


@dataclass
class Evaluation:
    accuracy: float
    loss: float
    samples: int


# ---------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------

def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def binary_cross_entropy(probability: float, target: int) -> float:
    probability = min(max(probability, 1e-12), 1 - 1e-12)
    return -(
        target * math.log(probability)
        + (1 - target) * math.log(1 - probability)
    )


def dot(weights: Sequence[float], inputs: Sequence[float]) -> float:
    if len(weights) != len(inputs):
        raise ValueError("Dimension mismatch.")
    return sum(w * x for w, x in zip(weights, inputs))


# ---------------------------------------------------------
# Dataset management
# ---------------------------------------------------------

class Dataset:
    def __init__(self, features, labels):
        if not features or len(features) != len(labels):
            raise ValueError("Features and labels must have equal nonzero lengths.")

        dimension = len(features[0])

        if dimension == 0:
            raise ValueError("Feature vectors cannot be empty.")

        if any(len(row) != dimension for row in features):
            raise ValueError("Inconsistent feature dimensions.")

        if any(not all(math.isfinite(float(x)) for x in row) for row in features):
            raise ValueError("Features must contain finite numeric values.")

        if any(label not in (0, 1) for label in labels):
            raise ValueError("Labels must be binary: 0 or 1.")

        self.features = [list(map(float, row)) for row in features]
        self.labels = list(labels)
        self.dimension = dimension

    def split(self, ratio=0.8, seed=42):
        if not 0 < ratio < 1:
            raise ValueError("Invalid split ratio.")

        indices = list(range(len(self.labels)))
        random.Random(seed).shuffle(indices)

        boundary = int(len(indices) * ratio)

        if boundary == 0 or boundary == len(indices):
            raise ValueError("Insufficient samples for train/test split.")

        train_indices = indices[:boundary]
        test_indices = indices[boundary:]

        def subset(selected):
            return Dataset(
                [self.features[i] for i in selected],
                [self.labels[i] for i in selected]
            )

        return subset(train_indices), subset(test_indices)


# ---------------------------------------------------------
# Pretrained feature extractor
# ---------------------------------------------------------

class PretrainedBackbone:
    """
    A small pretrained neural network.

    The weights represent previously learned source-domain
    parameters. They are transferred to a target task.

    Architecture:
        Input -> Dense(3, 2) -> Tanh -> Features
    """

    def __init__(self):
        self.weights = [
            [0.8, -0.4, 0.5],
            [-0.3, 0.9, 0.6]
        ]

        self.biases = [0.1, -0.1]
        self.trainable = False

    def extract(self, inputs):
        if len(inputs) != len(self.weights[0]):
            raise ValueError("Input dimensions do not match backbone.")

        return [
            math.tanh(dot(weights, inputs) + bias)
            for weights, bias in zip(self.weights, self.biases)
        ]

    def set_trainable(self, enabled: bool):
        self.trainable = enabled

    def parameters(self):
        return {
            "weights": self.weights,
            "biases": self.biases
        }


# ---------------------------------------------------------
# Target classification head
# ---------------------------------------------------------

class ClassificationHead:
    def __init__(self, input_dimension, seed=42):
        generator = random.Random(seed)

        self.weights = [
            generator.uniform(-0.1, 0.1)
            for _ in range(input_dimension)
        ]

        self.bias = 0.0

    def predict_probability(self, features):
        return sigmoid(dot(self.weights, features) + self.bias)

    def predict(self, features):
        return int(self.predict_probability(features) >= 0.5)


# ---------------------------------------------------------
# Transfer learning model
# ---------------------------------------------------------

class TransferLearningModel:

    def __init__(self, backbone, head):
        self.backbone = backbone
        self.head = head
        self.trained = False

    def predict_probability(self, inputs):
        features = self.backbone.extract(inputs)
        return self.head.predict_probability(features)

    def predict(self, inputs):
        return int(self.predict_probability(inputs) >= 0.5)

    def evaluate(self, dataset):
        correct = 0
        total_loss = 0.0

        for inputs, target in zip(
            dataset.features, dataset.labels
        ):
            probability = self.predict_probability(inputs)

            prediction = int(probability >= 0.5)

            correct += int(prediction == target)
            total_loss += binary_cross_entropy(probability, target)

        count = len(dataset.labels)

        return Evaluation(
            accuracy=correct / count,
            loss=total_loss / count,
            samples=count
        )

    def train(
        self,
        dataset,
        epochs,
        learning_rate,
        strategy
    ):
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("Invalid training configuration.")

        self.backbone.set_trainable(
            strategy == TransferStrategy.FINE_TUNING
        )

        for epoch in range(epochs):
            total_loss = 0.0

            # Randomized order reduces dependence on sample ordering.
            indices = list(range(len(dataset.labels)))
            random.shuffle(indices)

            for index in indices:
                inputs = dataset.features[index]
                target = dataset.labels[index]

                backbone = self.backbone
                head = self.head

                hidden = backbone.extract(inputs)
                probability = head.predict_probability(hidden)

                error = probability - target
                total_loss += binary_cross_entropy(
                    probability, target
                )

                # Gradients for the classification head.
                head_gradients = [
                    error * value for value in hidden
                ]

                head_bias_gradient = error

                # Compute backbone gradients using the original
                # classification-head weights.
                if backbone.trainable:
                    old_head_weights = head.weights.copy()

                    for neuron in range(len(hidden)):
                        derivative = (
                            1.0 - hidden[neuron] ** 2
                        )

                        hidden_gradient = (
                            error * old_head_weights[neuron]
                            * derivative
                        )

                        for feature_index in range(len(inputs)):
                            gradient = (
                                hidden_gradient * inputs[feature_index]
                            )

                            backbone.weights[neuron][feature_index] -= (
                                learning_rate * gradient
                            )

                        backbone.biases[neuron] -= (
                            learning_rate * hidden_gradient
                        )

                for i, gradient in enumerate(head_gradients):
                    head.weights[i] -= learning_rate * gradient

                head.bias -= learning_rate * head_bias_gradient

            if (epoch + 1) % 50 == 0:
                average_loss = total_loss / len(dataset.labels)
                print(
                    f"Epoch {epoch + 1:3d} | "
                    f"Training loss: {average_loss:.5f}"
                )

        self.trained = True

    def save(self, filepath):
        if not self.trained:
            raise RuntimeError("Cannot save an untrained model.")

        payload = {
            "backbone": self.backbone.parameters(),
            "head": {
                "weights": self.head.weights,
                "bias": self.head.bias
            }
        }

        path = Path(filepath)
        temporary_path = path.with_suffix(path.suffix + ".tmp")

        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8"
            )

            temporary_path.replace(path)

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(cls, filepath):
        payload = json.loads(
            Path(filepath).read_text(encoding="utf-8")
        )

        backbone = PretrainedBackbone()
        backbone.weights = payload["backbone"]["weights"]
        backbone.biases = payload["backbone"]["biases"]

        head = ClassificationHead(
            len(payload["head"]["weights"])
        )

        head.weights = payload["head"]["weights"]
        head.bias = payload["head"]["bias"]

        model = cls(backbone, head)
        model.trained = True

        return model


# ---------------------------------------------------------
# Synthetic target-domain dataset
# ---------------------------------------------------------

def create_target_dataset(samples=200, seed=42):
    generator = random.Random(seed)

    features = []
    labels = []

    for _ in range(samples):
        x1 = generator.uniform(-1, 1)
        x2 = generator.uniform(-1, 1)
        x3 = generator.uniform(-1, 1)

        # A target classification task with a nonlinear boundary.
        score = 1.2 * x1 - 0.9 * x2 + 0.7 * x3

        label = int(
            score + generator.gauss(0, 0.15) > 0
        )

        features.append([x1, x2, x3])
        labels.append(label)

    return Dataset(features, labels)


# ---------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------

def run_experiment():
    config = TrainingConfig()
    config.validate()

    dataset = create_target_dataset()

    train_data, test_data = dataset.split(
        ratio=config.train_ratio,
        seed=config.seed
    )

    # Transfer the pretrained backbone.
    backbone = PretrainedBackbone()

    head = ClassificationHead(
        input_dimension=2,
        seed=config.seed
    )

    model = TransferLearningModel(backbone, head)

    print("\nTRANSFER LEARNING EXPERIMENT")
    print("=" * 55)

    print("\nStrategy 1: Feature Extraction")
    print("Backbone parameters remain frozen.")

    model.train(
        train_data,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        strategy=TransferStrategy.FEATURE_EXTRACTION
    )

    feature_result = model.evaluate(test_data)

    print(
        f"Feature extraction accuracy: "
        f"{feature_result.accuracy:.2%}"
    )

    print("\nStrategy 2: Fine-Tuning")
    print("Backbone parameters are updated with a smaller learning rate.")

    model.train(
        train_data,
        epochs=config.epochs,
        learning_rate=config.fine_tune_learning_rate,
        strategy=TransferStrategy.FINE_TUNING
    )

    fine_tuned_result = model.evaluate(test_data)

    print(
        f"Fine-tuned accuracy: "
        f"{fine_tuned_result.accuracy:.2%}"
    )

    print("\nMODEL COMPARISON")
    print("-" * 55)

    print(
        f"Feature extraction loss: "
        f"{feature_result.loss:.5f}"
    )

    print(
        f"Fine-tuning loss: "
        f"{fine_tuned_result.loss:.5f}"
    )

    model.save("transfer_learning_model.json")

    restored_model = TransferLearningModel.load(
        "transfer_learning_model.json"
    )

    print("\nSaved and restored model successfully.")

    sample = [0.7, -0.4, 0.2]

    print(
        "Sample prediction:",
        restored_model.predict(sample)
    )

    print(
        "Sample probability:",
        round(restored_model.predict_probability(sample), 4)
    )

    return model


# ---------------------------------------------------------
# Tests and edge cases
# ---------------------------------------------------------

def run_tests():
    print("\nRUNNING VALIDATION TESTS")

    dataset = create_target_dataset(50)

    assert len(dataset.features) == 50

    backbone = PretrainedBackbone()

    sample = [0.2, 0.4, -0.1]
    extracted = backbone.extract(sample)

    assert len(extracted) == 2
    assert all(-1 <= value <= 1 for value in extracted)

    original_weights = json.dumps(backbone.weights)

    backbone.set_trainable(False)
    backbone.extract(sample)

    assert json.dumps(backbone.weights) == original_weights

    try:
        Dataset([[1, 2], [3]], [0, 1])
        raise AssertionError("Invalid dimensions were accepted.")
    except ValueError:
        pass

    try:
        TrainingConfig(epochs=0).validate()
        raise AssertionError("Invalid epoch count was accepted.")
    except ValueError:
        pass

    print("All validation tests passed.")


if __name__ == "__main__":
    run_tests()
    run_experiment()
