#!/usr/bin/env python3
"""
Forward Propagation in Artificial Neural Networks
=================================================

Demonstrates:
- Input, hidden, and output layers
- Neurons, weights, and biases
- Matrix-based forward computation
- Activation functions
- Multiple dense layers
- Batch inference
- Dimension validation
- Numerical stability
- Model serialization
- Unit testing

No external dependencies required.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence


class Activation(Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    SOFTMAX = "softmax"


class NetworkError(Exception):
    """Base exception for neural network failures."""


class DimensionError(NetworkError):
    """Raised when matrix dimensions are incompatible."""


class NumericalError(NetworkError):
    """Raised when computation produces invalid values."""


def validate_vector(values: Sequence[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} cannot be empty.")

    for value in values:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must contain numeric values.")

        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite value.")


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))

    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def softmax(values: Sequence[float]) -> list[float]:
    """Stable softmax using maximum subtraction."""
    maximum = max(values)
    exponentials = [math.exp(x - maximum) for x in values]
    denominator = math.fsum(exponentials)

    if denominator == 0 or not math.isfinite(denominator):
        raise NumericalError("Invalid softmax denominator.")

    return [x / denominator for x in exponentials]


def activate(
    values: Sequence[float],
    activation: Activation
) -> list[float]:

    if activation == Activation.LINEAR:
        result = list(values)

    elif activation == Activation.SIGMOID:
        result = [sigmoid(x) for x in values]

    elif activation == Activation.RELU:
        result = [max(0.0, x) for x in values]

    elif activation == Activation.TANH:
        result = [math.tanh(x) for x in values]

    elif activation == Activation.SOFTMAX:
        result = softmax(values)

    else:
        raise ValueError("Unsupported activation function.")

    if not all(math.isfinite(x) for x in result):
        raise NumericalError("Activation produced invalid values.")

    return result


@dataclass
class DenseLayer:
    """
    Fully connected layer.

    Weight orientation:
        weights[output_neuron][input_neuron]

    Computation:
        z = W * a + b
        a_next = activation(z)
    """

    weights: list[list[float]]
    biases: list[float]
    activation: Activation = Activation.RELU
    name: str = "dense"

    input_size: int = field(init=False)
    output_size: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("Weights cannot be empty.")

        validate_vector(self.biases, "Biases")

        self.output_size = len(self.weights)
        self.input_size = len(self.weights[0])

        if self.input_size == 0:
            raise DimensionError("Weight rows cannot be empty.")

        if len(self.biases) != self.output_size:
            raise DimensionError(
                "Number of biases must equal output neurons."
            )

        for row in self.weights:
            validate_vector(row, "Weight row")

            if len(row) != self.input_size:
                raise DimensionError(
                    "All weight rows must have equal length."
                )

    def forward(
        self,
        inputs: Sequence[float]
    ) -> tuple[list[float], list[float]]:

        validate_vector(inputs, "Layer input")

        if len(inputs) != self.input_size:
            raise DimensionError(
                f"{self.name}: expected {self.input_size} inputs, "
                f"received {len(inputs)}."
            )

        # Weighted sums: z_j = sum(W_ji * x_i) + b_j
        weighted_sums = []

        for weights, bias in zip(self.weights, self.biases):
            value = math.fsum(
                weight * input_value
                for weight, input_value in zip(weights, inputs)
            ) + bias

            if not math.isfinite(value):
                raise NumericalError(
                    f"Non-finite weighted sum in {self.name}."
                )

            weighted_sums.append(value)

        outputs = activate(weighted_sums, self.activation)

        return weighted_sums, outputs


class NeuralNetwork:
    """Sequential fully connected neural network."""

    def __init__(self, layers: list[DenseLayer]) -> None:
        if not layers:
            raise ValueError("Network requires at least one layer.")

        for index in range(1, len(layers)):
            previous = layers[index - 1]
            current = layers[index]

            if previous.output_size != current.input_size:
                raise DimensionError(
                    f"Layer {index} expects {current.input_size} inputs, "
                    f"but previous layer produces {previous.output_size}."
                )

        self.layers = layers

    @property
    def input_size(self) -> int:
        return self.layers[0].input_size

    @property
    def output_size(self) -> int:
        return self.layers[-1].output_size

    def forward(
        self,
        inputs: Sequence[float],
        return_trace: bool = False
    ):
        validate_vector(inputs, "Network input")

        if len(inputs) != self.input_size:
            raise DimensionError(
                f"Network expects {self.input_size} features."
            )

        current = list(inputs)
        trace = []

        for index, layer in enumerate(self.layers, start=1):
            weighted_sums, outputs = layer.forward(current)

            trace.append({
                "layer": index,
                "name": layer.name,
                "activation": layer.activation.value,
                "weighted_sums": weighted_sums,
                "outputs": outputs
            })

            current = outputs

        if return_trace:
            return current, trace

        return current

    def predict_batch(
        self,
        samples: Sequence[Sequence[float]]
    ) -> list[list[float]]:

        if not samples:
            raise ValueError("Batch cannot be empty.")

        return [self.forward(sample) for sample in samples]

    def summary(self) -> None:
        print("\nNETWORK ARCHITECTURE")
        print("-" * 55)

        for index, layer in enumerate(self.layers, start=1):
            print(
                f"Layer {index}: {layer.name} | "
                f"{layer.input_size} -> {layer.output_size} | "
                f"Activation: {layer.activation.value}"
            )

        print("-" * 55)
        print(f"Input features: {self.input_size}")
        print(f"Output features: {self.output_size}")

        total_parameters = sum(
            layer.input_size * layer.output_size
            + layer.output_size
            for layer in self.layers
        )

        print(f"Trainable parameters: {total_parameters}")

    def save(self, filename: str) -> None:
        """Save architecture and parameters as JSON."""
        data = {
            "layers": [
                {
                    "name": layer.name,
                    "weights": layer.weights,
                    "biases": layer.biases,
                    "activation": layer.activation.value
                }
                for layer in self.layers
            ]
        }

        Path(filename).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, filename: str) -> NeuralNetwork:
        data = json.loads(
            Path(filename).read_text(encoding="utf-8")
        )

        layers = []

        for item in data["layers"]:
            layers.append(
                DenseLayer(
                    weights=item["weights"],
                    biases=item["biases"],
                    activation=Activation(item["activation"]),
                    name=item["name"]
                )
            )

        return cls(layers)


def create_demo_network() -> NeuralNetwork:
    """
    Architecture:
        Input: 3 features
        Hidden: 4 neurons, ReLU
        Output: 2 neurons, Softmax
    """

    random.seed(42)

    def initialize_weights(outputs: int, inputs: int):
        limit = math.sqrt(6.0 / (inputs + outputs))

        return [
            [
                random.uniform(-limit, limit)
                for _ in range(inputs)
            ]
            for _ in range(outputs)
        ]

    hidden = DenseLayer(
        weights=initialize_weights(4, 3),
        biases=[0.1, -0.1, 0.05, 0.0],
        activation=Activation.RELU,
        name="Hidden Layer"
    )

    output = DenseLayer(
        weights=initialize_weights(2, 4),
        biases=[0.05, -0.05],
        activation=Activation.SOFTMAX,
        name="Output Layer"
    )

    return NeuralNetwork([hidden, output])


def demonstrate_manual_computation() -> None:
    print("\nMANUAL MATRIX COMPUTATION")
    print("-" * 55)

    inputs = [1.0, 2.0]

    weights = [
        [0.5, -0.2],
        [0.3, 0.8]
    ]

    biases = [0.1, -0.1]

    layer = DenseLayer(
        weights=weights,
        biases=biases,
        activation=Activation.RELU,
        name="Manual Layer"
    )

    z, outputs = layer.forward(inputs)

    print("Input:", inputs)
    print("Weights:", weights)
    print("Biases:", biases)
    print("Weighted sums:", z)
    print("ReLU outputs:", outputs)


def run_tests(network: NeuralNetwork) -> None:
    """Basic correctness and edge-case tests."""

    # Linear activation identity.
    linear = DenseLayer(
        weights=[[2.0, 3.0]],
        biases=[1.0],
        activation=Activation.LINEAR
    )

    assert linear.forward([4.0, 5.0])[1] == [24.0]

    # ReLU behavior.
    relu = DenseLayer(
        weights=[[1.0]],
        biases=[-2.0],
        activation=Activation.RELU
    )

    assert relu.forward([1.0])[1] == [0.0]

    # Softmax probability distribution.
    probabilities = softmax([1.0, 2.0, 3.0])

    assert abs(sum(probabilities) - 1.0) < 1e-10

    # Sigmoid numerical stability.
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) < 1e-10

    # Batch prediction.
    batch = network.predict_batch([
        [0.2, 0.5, 0.8],
        [0.7, 0.1, 0.4]
    ])

    assert len(batch) == 2

    # Invalid dimensions.
    try:
        network.forward([1.0])
        raise AssertionError("Dimension validation failed.")
    except DimensionError:
        pass

    # Non-finite input.
    try:
        network.forward([float("nan"), 1.0, 2.0])
        raise AssertionError("Finite-value validation failed.")
    except ValueError:
        pass

    print("\nAll tests passed.")


def main() -> None:
    print("=" * 65)
    print("FORWARD PROPAGATION: NEURAL NETWORK COMPUTATION")
    print("=" * 65)

    demonstrate_manual_computation()

    network = create_demo_network()
    network.summary()

    sample = [0.2, 0.5, 0.8]

    prediction, trace = network.forward(
        sample,
        return_trace=True
    )

    print("\nFORWARD PROPAGATION TRACE")
    print("-" * 55)
    print("Input:", sample)

    for item in trace:
        print(f"\n{item['name']}")
        print("Weighted sums:", item["weighted_sums"])
        print("Activated outputs:", item["outputs"])

    print("\nFINAL OUTPUT")
    print(prediction)
    print("Probability sum:", math.fsum(prediction))

    print("\nBATCH INFERENCE")
    batch = network.predict_batch([
        [0.2, 0.5, 0.8],
        [0.7, 0.1, 0.4],
        [0.9, 0.3, 0.2]
    ])

    for index, result in enumerate(batch, start=1):
        print(f"Sample {index}: {result}")

    run_tests(network)

    # Demonstrate serialization and restoration.
    filename = "forward_propagation_model.json"

    network.save(filename)
    restored = NeuralNetwork.load(filename)

    original_output = network.forward(sample)
    restored_output = restored.forward(sample)

    assert original_output == restored_output

    Path(filename).unlink(missing_ok=True)

    print("\nModel serialization test passed.")
    print("\nForward propagation demonstration completed.")


if __name__ == "__main__":
    try:
        main()
    except (NetworkError, ValueError, TypeError, OSError) as error:
        print(f"Execution failed: {error}")
        raise SystemExit(1)
