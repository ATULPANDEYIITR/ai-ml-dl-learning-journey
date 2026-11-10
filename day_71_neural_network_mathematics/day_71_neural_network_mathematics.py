"""
Neural Network Mathematics
Topics: Neurons, weighted sums, bias, activation functions,
forward propagation, derivatives, backpropagation, and gradient descent.

Python 3.10+
Standard library only.
"""

from __future__ import annotations

import json
import math
import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger("neural_network")


# ============================================================
# 1. Activation Functions
# ============================================================

class Activation(Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"


def activate(value: float, function: Activation) -> float:
    if function == Activation.LINEAR:
        return value

    if function == Activation.SIGMOID:
        # Stable sigmoid avoids exponential overflow.
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-min(value, 709)))
        exp_value = math.exp(max(value, -709))
        return exp_value / (1.0 + exp_value)

    if function == Activation.TANH:
        return math.tanh(value)

    if function == Activation.RELU:
        return max(0.0, value)

    if function == Activation.LEAKY_RELU:
        return value if value > 0 else 0.01 * value

    raise ValueError(f"Unsupported activation: {function}")


def activation_derivative(
    value: float,
    function: Activation
) -> float:
    """Derivative evaluated at the pre-activation weighted sum."""

    if function == Activation.LINEAR:
        return 1.0

    if function == Activation.SIGMOID:
        output = activate(value, function)
        return output * (1.0 - output)

    if function == Activation.TANH:
        output = math.tanh(value)
        return 1.0 - output * output

    if function == Activation.RELU:
        return 1.0 if value > 0 else 0.0

    if function == Activation.LEAKY_RELU:
        return 1.0 if value > 0 else 0.01

    raise ValueError(f"Unsupported activation: {function}")


# ============================================================
# 2. Mathematical Validation
# ============================================================

def validate_vector(
    values: Sequence[float],
    expected_length: int | None = None
) -> None:
    if not values:
        raise ValueError("Input vector cannot be empty.")

    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"Expected {expected_length} values, received {len(values)}."
        )

    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("All values must be finite numbers.")


def weighted_sum(
    inputs: Sequence[float],
    weights: Sequence[float],
    bias: float
) -> float:
    validate_vector(inputs)
    validate_vector(weights, len(inputs))

    if not math.isfinite(bias):
        raise ValueError("Bias must be finite.")

    return math.fsum(
        x * w for x, w in zip(inputs, weights)
    ) + bias


# ============================================================
# 3. Individual Artificial Neuron
# ============================================================

@dataclass
class Neuron:
    weights: list[float]
    bias: float = 0.0
    activation: Activation = Activation.SIGMOID
    last_inputs: list[float] = field(default_factory=list)
    last_sum: float = 0.0
    last_output: float = 0.0

    def __post_init__(self):
        validate_vector(self.weights)

        if not math.isfinite(self.bias):
            raise ValueError("Bias must be finite.")

        self.weights = [float(w) for w in self.weights]

    def forward(self, inputs: Sequence[float]) -> float:
        validate_vector(inputs, len(self.weights))

        self.last_inputs = [float(x) for x in inputs]
        self.last_sum = weighted_sum(
            self.last_inputs,
            self.weights,
            self.bias
        )

        self.last_output = activate(
            self.last_sum,
            self.activation
        )

        return self.last_output

    def display(self) -> None:
        print(f"Weights: {self.weights}")
        print(f"Bias: {self.bias:.6f}")
        print(f"Weighted sum: {self.last_sum:.6f}")
        print(f"Activation: {self.activation.value}")
        print(f"Output: {self.last_output:.6f}")


# ============================================================
# 4. Loss Functions
# ============================================================

def mean_squared_error(
    predictions: Sequence[float],
    targets: Sequence[float]
) -> float:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Prediction and target sizes must match.")

    return math.fsum(
        (p - t) ** 2
        for p, t in zip(predictions, targets)
    ) / len(targets)


def mse_derivative(
    prediction: float,
    target: float
) -> float:
    return 2.0 * (prediction - target)


# ============================================================
# 5. Feedforward Neural Network
# ============================================================

class NeuralNetwork:
    def __init__(
        self,
        architecture: Sequence[int],
        hidden_activation: Activation = Activation.TANH,
        output_activation: Activation = Activation.SIGMOID,
        seed: int = 42
    ):
        if len(architecture) < 2:
            raise ValueError("Network requires input and output layers.")

        if any(
            not isinstance(size, int) or size <= 0
            for size in architecture
        ):
            raise ValueError("Layer sizes must be positive integers.")

        self.architecture = list(architecture)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        rng = random.Random(seed)

        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []

        # Xavier initialization for tanh/sigmoid-style networks.
        for fan_in, fan_out in zip(
            architecture[:-1],
            architecture[1:]
        ):
            limit = math.sqrt(6.0 / (fan_in + fan_out))

            layer_weights = [
                [rng.uniform(-limit, limit) for _ in range(fan_in)]
                for _ in range(fan_out)
            ]

            self.weights.append(layer_weights)
            self.biases.append([0.0] * fan_out)

        self.last_activations: list[list[float]] = []
        self.last_z_values: list[list[float]] = []

    def forward(self, inputs: Sequence[float]) -> list[float]:
        validate_vector(inputs, self.architecture[0])

        current = [float(x) for x in inputs]

        self.last_activations = [current.copy()]
        self.last_z_values = []

        for layer_index, layer_weights in enumerate(self.weights):
            activation_function = (
                self.output_activation
                if layer_index == len(self.weights) - 1
                else self.hidden_activation
            )

            next_layer = []
            z_layer = []

            for neuron_index, neuron_weights in enumerate(layer_weights):
                z = weighted_sum(
                    current,
                    neuron_weights,
                    self.biases[layer_index][neuron_index]
                )

                z_layer.append(z)
                next_layer.append(activate(z, activation_function))

            current = next_layer
            self.last_z_values.append(z_layer)
            self.last_activations.append(current.copy())

        return current

    # ========================================================
    # 6. Backpropagation and Gradient Descent
    # ========================================================

    def train_sample(
        self,
        inputs: Sequence[float],
        targets: Sequence[float],
        learning_rate: float
    ) -> float:
        if learning_rate <= 0 or not math.isfinite(learning_rate):
            raise ValueError("Learning rate must be positive and finite.")

        predictions = self.forward(inputs)
        validate_vector(targets, len(predictions))

        targets = [float(t) for t in targets]
        layer_count = len(self.weights)

        deltas: list[list[float]] = [
            [] for _ in range(layer_count)
        ]

        # Output-layer gradients.
        last_index = layer_count - 1
        deltas[last_index] = []

        for prediction, target, z in zip(
            predictions,
            targets,
            self.last_z_values[last_index]
        ):
            error_gradient = mse_derivative(prediction, target)

            derivative = activation_derivative(
                z,
                self.output_activation
            )

            deltas[last_index].append(
                error_gradient * derivative
            )

        # Hidden-layer gradients.
        for layer_index in range(layer_count - 2, -1, -1):
            current_deltas = []

            for neuron_index, z in enumerate(
                self.last_z_values[layer_index]
            ):
                propagated_error = math.fsum(
                    self.weights[layer_index + 1][next_index][neuron_index]
                    * deltas[layer_index + 1][next_index]
                    for next_index in range(
                        len(deltas[layer_index + 1])
                    )
                )

                derivative = activation_derivative(
                    z,
                    self.hidden_activation
                )

                current_deltas.append(
                    propagated_error * derivative
                )

            deltas[layer_index] = current_deltas

        # Apply gradients after all deltas have been calculated.
        # This avoids using partially updated weights.
        for layer_index in range(layer_count):
            previous_activations = self.last_activations[layer_index]

            for neuron_index, delta in enumerate(
                deltas[layer_index]
            ):
                for weight_index, input_value in enumerate(
                    previous_activations
                ):
                    gradient = delta * input_value

                    self.weights[layer_index][neuron_index][weight_index] -= (
                        learning_rate * gradient
                    )

                self.biases[layer_index][neuron_index] -= (
                    learning_rate * delta
                )

        return mean_squared_error(predictions, targets)

    def train(
        self,
        dataset: Sequence[tuple[Sequence[float], Sequence[float]]],
        epochs: int = 5000,
        learning_rate: float = 0.5,
        shuffle: bool = True,
        seed: int = 42
    ) -> list[float]:
        if not dataset:
            raise ValueError("Training dataset cannot be empty.")

        if epochs <= 0:
            raise ValueError("Epochs must be positive.")

        for inputs, targets in dataset:
            validate_vector(inputs, self.architecture[0])
            validate_vector(targets, self.architecture[-1])

        rng = random.Random(seed)
        history = []

        for epoch in range(epochs):
            samples = list(dataset)

            if shuffle:
                rng.shuffle(samples)

            for inputs, targets in samples:
                self.train_sample(inputs, targets, learning_rate)

            epoch_loss = math.fsum(
                mean_squared_error(
                    self.forward(inputs),
                    targets
                )
                for inputs, targets in dataset
            ) / len(dataset)

            history.append(epoch_loss)

            if epoch % 1000 == 0:
                logger.info(
                    "Epoch %d | Loss: %.8f",
                    epoch,
                    epoch_loss
                )

            if epoch_loss < 0.001:
                logger.info("Training converged at epoch %d.", epoch)
                break

        return history

    # ========================================================
    # 7. Model Persistence
    # ========================================================

    def save(self, filename: str) -> None:
        data = {
            "architecture": self.architecture,
            "hidden_activation": self.hidden_activation.value,
            "output_activation": self.output_activation.value,
            "weights": self.weights,
            "biases": self.biases
        }

        path = Path(filename)
        temporary = path.with_suffix(path.suffix + ".tmp")

        temporary.write_text(
            json.dumps(data, indent=2, allow_nan=False),
            encoding="utf-8"
        )

        temporary.replace(path)

    @classmethod
    def load(cls, filename: str) -> NeuralNetwork:
        data = json.loads(
            Path(filename).read_text(encoding="utf-8")
        )

        model = cls(
            architecture=data["architecture"],
            hidden_activation=Activation(data["hidden_activation"]),
            output_activation=Activation(data["output_activation"])
        )

        weights = data["weights"]
        biases = data["biases"]

        if len(weights) != len(model.weights):
            raise ValueError("Invalid model weight structure.")

        for index, (actual, expected) in enumerate(
            zip(weights, model.weights)
        ):
            if len(actual) != len(expected):
                raise ValueError("Invalid neuron count.")

            for row, expected_row in zip(actual, expected):
                validate_vector(row, len(expected_row))

        if len(biases) != len(model.biases):
            raise ValueError("Invalid bias structure.")

        for actual, expected in zip(biases, model.biases):
            validate_vector(actual, len(expected))

        model.weights = weights
        model.biases = biases

        return model


# ============================================================
# 8. Numerical Gradient Verification
# ============================================================

def numerical_derivative(
    function: Callable[[float], float],
    value: float,
    epsilon: float = 1e-6
) -> float:
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")

    return (
        function(value + epsilon)
        - function(value - epsilon)
    ) / (2.0 * epsilon)


def verify_sigmoid_derivative() -> bool:
    value = 0.7

    numerical = numerical_derivative(
        lambda x: activate(x, Activation.SIGMOID),
        value
    )

    analytical = activation_derivative(
        value,
        Activation.SIGMOID
    )

    return math.isclose(
        numerical,
        analytical,
        rel_tol=1e-5,
        abs_tol=1e-7
    )


# ============================================================
# 9. Demonstration
# ============================================================

def demonstrate_single_neuron() -> None:
    print("\n--- Individual Neuron ---")

    neuron = Neuron(
        weights=[0.5, -0.3, 0.8],
        bias=0.2,
        activation=Activation.SIGMOID
    )

    result = neuron.forward([1.0, 2.0, 0.5])

    neuron.display()

    print(f"Final neuron output: {result:.6f}")


def demonstrate_activations() -> None:
    print("\n--- Activation Functions ---")

    value = 1.5

    for function in Activation:
        output = activate(value, function)
        derivative = activation_derivative(value, function)

        print(
            f"{function.value:12} "
            f"output={output:.6f} "
            f"derivative={derivative:.6f}"
        )


def demonstrate_xor() -> None:
    print("\n--- XOR Neural Network Training ---")

    dataset = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0])
    ]

    network = NeuralNetwork(
        architecture=[2, 4, 1],
        hidden_activation=Activation.TANH,
        output_activation=Activation.SIGMOID,
        seed=42
    )

    history = network.train(
        dataset,
        epochs=10000,
        learning_rate=0.5
    )

    print(f"Final training loss: {history[-1]:.8f}")

    for inputs, targets in dataset:
        prediction = network.forward(inputs)

        print(
            f"Input={inputs} "
            f"Target={targets} "
            f"Prediction={prediction[0]:.6f} "
            f"Class={int(prediction[0] >= 0.5)}"
        )

    network.save("neural_network_model.json")

    restored = NeuralNetwork.load("neural_network_model.json")

    print("\nRestored model verification:")

    for inputs, _ in dataset:
        print(inputs, restored.forward(inputs))


def demonstrate_validation() -> None:
    print("\n--- Validation and Edge Cases ---")

    try:
        weighted_sum(
            [1.0, 2.0],
            [0.5],
            0.1
        )
    except ValueError as error:
        print("Dimension validation:", error)

    try:
        weighted_sum(
            [1.0, float("nan")],
            [0.5, 0.2],
            0.1
        )
    except ValueError as error:
        print("Non-finite input validation:", error)

    print(
        "Sigmoid derivative verification:",
        verify_sigmoid_derivative()
    )


def main() -> None:
    demonstrate_single_neuron()
    demonstrate_activations()
    demonstrate_validation()
    demonstrate_xor()


if __name__ == "__main__":
    main()
