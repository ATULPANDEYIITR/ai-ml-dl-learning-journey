#!/usr/bin/env python3
"""
Activation Functions: Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Softmax.

Self-contained numerical implementation covering:
- Scalar and vector activation functions
- Numerically stable calculations
- Derivatives and gradient propagation
- Softmax Jacobian-vector products
- Activation configuration and validation
- Neural network layer execution
- Numerical gradient checking
- Unit tests and practical examples

Run: python activation_functions.py
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence


class ActivationType(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SOFTMAX = "softmax"


class ActivationError(ValueError):
    """Raised when activation inputs or configurations are invalid."""


def validate_number(x: float) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ActivationError("Input must be a real number.")

    value = float(x)

    if not math.isfinite(value):
        raise ActivationError("Input must be finite.")

    return value


def validate_vector(values: Sequence[float]) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise ActivationError("Input must be a numerical sequence.")

    try:
        result = [validate_number(x) for x in values]
    except TypeError as exc:
        raise ActivationError("Input must be iterable.") from exc

    if not result:
        raise ActivationError("Input vector cannot be empty.")

    return result


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    x = validate_number(x)

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid_derivative(x: float) -> float:
    y = sigmoid(x)
    return y * (1.0 - y)


def tanh(x: float) -> float:
    return math.tanh(validate_number(x))


def tanh_derivative(x: float) -> float:
    y = tanh(x)
    return 1.0 - y * y


def relu(x: float) -> float:
    return max(0.0, validate_number(x))


def relu_derivative(x: float) -> float:
    return 1.0 if validate_number(x) > 0.0 else 0.0


def leaky_relu(x: float, alpha: float = 0.01) -> float:
    x = validate_number(x)
    alpha = validate_number(alpha)

    if alpha < 0:
        raise ActivationError("Leaky ReLU alpha cannot be negative.")

    return x if x > 0 else alpha * x


def leaky_relu_derivative(
    x: float,
    alpha: float = 0.01
) -> float:
    x = validate_number(x)
    alpha = validate_number(alpha)

    if alpha < 0:
        raise ActivationError("Leaky ReLU alpha cannot be negative.")

    return 1.0 if x > 0 else alpha


def gelu(x: float) -> float:
    """
    GELU using the tanh approximation:

    GELU(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    """
    x = validate_number(x)

    coefficient = math.sqrt(2.0 / math.pi)
    inner = coefficient * (x + 0.044715 * x**3)

    return 0.5 * x * (1.0 + math.tanh(inner))


def gelu_derivative(x: float) -> float:
    x = validate_number(x)

    coefficient = math.sqrt(2.0 / math.pi)
    inner = coefficient * (x + 0.044715 * x**3)

    t = math.tanh(inner)
    sech_squared = 1.0 - t * t

    inner_derivative = coefficient * (
        1.0 + 3.0 * 0.044715 * x**2
    )

    return (
        0.5 * (1.0 + t)
        + 0.5 * x * sech_squared * inner_derivative
    )


def softmax(values: Sequence[float]) -> list[float]:
    """
    Stable softmax.

    Subtracting the maximum prevents exponential overflow.
    """
    values = validate_vector(values)

    maximum = max(values)
    exponentials = [math.exp(x - maximum) for x in values]

    denominator = math.fsum(exponentials)

    if denominator == 0 or not math.isfinite(denominator):
        raise ActivationError("Softmax normalization failed.")

    return [value / denominator for value in exponentials]


def softmax_jacobian(
    probabilities: Sequence[float]
) -> list[list[float]]:
    """
    Jacobian:
        J[i][j] = p[i] * (delta(i,j) - p[j])

    Time and space complexity: O(n^2).
    """
    p = validate_vector(probabilities)

    total = math.fsum(p)

    if any(value < 0 for value in p):
        raise ActivationError("Probabilities cannot be negative.")

    if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-9):
        raise ActivationError("Input must sum to one.")

    return [
        [
            p[i] * ((1.0 if i == j else 0.0) - p[j])
            for j in range(len(p))
        ]
        for i in range(len(p))
    ]


def softmax_jvp(
    probabilities: Sequence[float],
    upstream_gradient: Sequence[float]
) -> list[float]:
    """
    Efficient Jacobian-vector product.

    Avoids explicitly constructing an O(n^2) Jacobian.
    """
    p = validate_vector(probabilities)
    gradient = validate_vector(upstream_gradient)

    if len(p) != len(gradient):
        raise ActivationError("Gradient dimensions do not match.")

    total = math.fsum(p)

    if any(value < 0 for value in p):
        raise ActivationError("Probabilities cannot be negative.")

    if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-9):
        raise ActivationError("Input must sum to one.")

    dot_product = math.fsum(
        p[i] * gradient[i] for i in range(len(p))
    )

    return [
        p[i] * (gradient[i] - dot_product)
        for i in range(len(p))
    ]


@dataclass(frozen=True)
class ActivationConfig:
    activation: ActivationType
    alpha: float = 0.01

    def __post_init__(self):
        if not isinstance(self.activation, ActivationType):
            raise ActivationError("Invalid activation type.")

        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ActivationError("Alpha must be finite and nonnegative.")


class Activation:
    """Activation dispatcher."""

    @staticmethod
    def apply(
        x: float,
        config: ActivationConfig
    ) -> float:
        functions: dict[ActivationType, Callable] = {
            ActivationType.SIGMOID: sigmoid,
            ActivationType.TANH: tanh,
            ActivationType.RELU: relu,
            ActivationType.GELU: gelu,
        }

        if config.activation == ActivationType.SOFTMAX:
            raise ActivationError(
                "Softmax requires a vector, not a scalar."
            )

        if config.activation == ActivationType.LEAKY_RELU:
            return leaky_relu(x, config.alpha)

        return functions[config.activation](x)

    @staticmethod
    def derivative(
        x: float,
        config: ActivationConfig
    ) -> float:
        derivatives: dict[ActivationType, Callable] = {
            ActivationType.SIGMOID: sigmoid_derivative,
            ActivationType.TANH: tanh_derivative,
            ActivationType.RELU: relu_derivative,
            ActivationType.GELU: gelu_derivative,
        }

        if config.activation == ActivationType.SOFTMAX:
            raise ActivationError(
                "Use softmax_jvp for softmax gradients."
            )

        if config.activation == ActivationType.LEAKY_RELU:
            return leaky_relu_derivative(x, config.alpha)

        return derivatives[config.activation](x, config)


@dataclass
class DenseActivationLayer:
    """A single activation layer operating on a numerical vector."""

    config: ActivationConfig

    def forward(self, inputs: Sequence[float]) -> list[float]:
        values = validate_vector(inputs)

        if self.config.activation == ActivationType.SOFTMAX:
            return softmax(values)

        return [
            Activation.apply(value, self.config)
            for value in values
        ]

    def backward(
        self,
        inputs: Sequence[float],
        upstream_gradient: Sequence[float]
    ) -> list[float]:
        values = validate_vector(inputs)
        gradient = validate_vector(upstream_gradient)

        if len(values) != len(gradient):
            raise ActivationError("Gradient dimensions do not match.")

        if self.config.activation == ActivationType.SOFTMAX:
            return softmax_jvp(softmax(values), gradient)

        return [
            Activation.derivative(value, self.config) * grad
            for value, grad in zip(values, gradient)
        ]


def numerical_derivative(
    function: Callable[[float], float],
    x: float,
    epsilon: float = 1e-6
) -> float:
    """Central finite-difference approximation."""
    x = validate_number(x)
    epsilon = validate_number(epsilon)

    if epsilon <= 0:
        raise ActivationError("Epsilon must be positive.")

    return (
        function(x + epsilon) - function(x - epsilon)
    ) / (2.0 * epsilon)


class ActivationTests(unittest.TestCase):

    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertGreater(sigmoid(1000), 0.999)
        self.assertLess(sigmoid(-1000), 0.001)

    def test_tanh(self):
        self.assertAlmostEqual(tanh(0), 0)
        self.assertAlmostEqual(tanh(2), -tanh(-2))

    def test_relu(self):
        self.assertEqual(relu(-5), 0)
        self.assertEqual(relu(5), 5)
        self.assertEqual(relu_derivative(0), 0)

    def test_leaky_relu(self):
        self.assertAlmostEqual(leaky_relu(-5), -0.05)
        self.assertEqual(leaky_relu(5), 5)

    def test_gelu(self):
        self.assertAlmostEqual(gelu(0), 0)
        self.assertGreater(gelu(2), 1.9)

    def test_softmax(self):
        result = softmax([2, 1, 0])
        self.assertAlmostEqual(sum(result), 1.0)
        self.assertGreater(result[0], result[1])
        self.assertGreater(result[1], result[2])

    def test_softmax_stability(self):
        result = softmax([1000, 1001, 1002])
        self.assertAlmostEqual(sum(result), 1.0)

    def test_softmax_jvp(self):
        p = softmax([1, 2, 3])
        result = softmax_jvp(p, [1, 0, -1])
        self.assertAlmostEqual(sum(result), 0.0)

    def test_gradient_check(self):
        for function, derivative in [
            (sigmoid, sigmoid_derivative),
            (tanh, tanh_derivative),
            (relu, relu_derivative),
            (gelu, gelu_derivative),
        ]:
            x = 1.3
            approximate = numerical_derivative(function, x)
            self.assertAlmostEqual(
                derivative(x), approximate, places=5
            )

    def test_layer(self):
        layer = DenseActivationLayer(
            ActivationConfig(ActivationType.RELU)
        )
        self.assertEqual(
            layer.forward([-2, 0, 3]),
            [0, 0, 3]
        )

        self.assertEqual(
            layer.backward([-2, 0, 3], [1, 1, 1]),
            [0, 0, 1]
        )

    def test_invalid_input(self):
        with self.assertRaises(ActivationError):
            sigmoid(float("nan"))

        with self.assertRaises(ActivationError):
            softmax([])

        with self.assertRaises(ActivationError):
            leaky_relu(-2, -0.1)

        with self.assertRaises(ActivationError):
            softmax_jacobian([0.2, 0.2])


def demonstration():
    test_values = [-3, -1, 0, 1, 3]

    configurations = [
        ActivationConfig(ActivationType.SIGMOID),
        ActivationConfig(ActivationType.TANH),
        ActivationConfig(ActivationType.RELU),
        ActivationConfig(ActivationType.LEAKY_RELU, 0.01),
        ActivationConfig(ActivationType.GELU),
    ]

    print("ACTIVATION FUNCTION COMPARISON")
    print("-" * 80)
    print(f"{'Input':>10}", end="")

    for config in configurations:
        print(f"{config.activation.value:>15}", end="")

    print()

    for x in test_values:
        print(f"{x:>10.2f}", end="")

        for config in configurations:
            result = Activation.apply(x, config)
            print(f"{result:>15.6f}", end="")

        print()

    logits = [2.0, 1.0, 0.1]
    probabilities = softmax(logits)

    print("\nSOFTMAX CLASSIFICATION")
    for index, probability in enumerate(probabilities):
        print(f"Class {index}: {probability:.6f}")

    layer = DenseActivationLayer(
        ActivationConfig(ActivationType.RELU)
    )

    inputs = [-2, -1, 0, 1, 2]
    output = layer.forward(inputs)

    print("\nDENSE ACTIVATION LAYER")
    print("Inputs:", inputs)
    print("Outputs:", output)

    gradient = layer.backward(inputs, [1, 1, 1, 1, 1])
    print("Backward gradient:", gradient)

    print("\nSOFTMAX GRADIENT")
    upstream = [1.0, 0.0, -1.0]
    print("JVP:", softmax_jvp(probabilities, upstream))


if __name__ == "__main__":
    demonstration()
    print("\nRUNNING UNIT TESTS")
    unittest.main(argv=["activation_functions.py"], exit=False)
