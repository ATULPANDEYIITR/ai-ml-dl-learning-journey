#!/usr/bin/env python3
"""
Backpropagation: Chain Rule, Gradients, and Computational Graphs.

A self-contained scalar automatic differentiation engine with:
- Computational graph construction.
- Forward evaluation.
- Reverse-mode automatic differentiation.
- Gradient accumulation.
- Topological ordering.
- Numerical gradient checking.
- A trainable neural network.
- XOR classification using backpropagation.
- Validation and failure handling.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable


class DifferentiationError(Exception):
    pass


@dataclass(eq=False)
class Value:
    data: float
    parents: tuple["Value", ...] = ()
    operation: str = ""
    label: str = ""
    grad: float = 0.0
    backward_fn: Callable[[], None] = field(
        default=lambda: None, repr=False
    )

    def __post_init__(self):
        if not math.isfinite(self.data):
            raise ValueError("Node values must be finite.")

    @staticmethod
    def ensure(other):
        return other if isinstance(other, Value) else Value(float(other))

    def __add__(self, other):
        other = self.ensure(other)
        result = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def backward():
            self.grad += result.grad
            other.grad += result.grad

        result.backward_fn = backward
        return result

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self.ensure(other))

    def __rsub__(self, other):
        return self.ensure(other) - self

    def __mul__(self, other):
        other = self.ensure(other)
        result = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result.backward_fn = backward
        return result

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = self.ensure(other)
        if other.data == 0:
            raise ZeroDivisionError("Division by zero.")
        return self * other.power(-1)

    def power(self, exponent: float):
        if not math.isfinite(exponent):
            raise ValueError("Exponent must be finite.")

        if self.data < 0 and not exponent.is_integer():
            raise ValueError("Invalid fractional power.")

        if self.data == 0 and exponent < 1:
            raise ValueError("Undefined derivative at zero.")

        result = Value(
            self.data ** exponent,
            (self,),
            f"pow({exponent})"
        )

        def backward():
            derivative = exponent * (
                self.data ** (exponent - 1)
            )
            self.grad += derivative * result.grad

        result.backward_fn = backward
        return result

    def exp(self):
        result = Value(
            math.exp(self.data),
            (self,),
            "exp"
        )

        def backward():
            self.grad += result.data * result.grad

        result.backward_fn = backward
        return result

    def log(self):
        if self.data <= 0:
            raise ValueError("Logarithm requires a positive input.")

        result = Value(
            math.log(self.data),
            (self,),
            "log"
        )

        def backward():
            self.grad += result.grad / self.data

        result.backward_fn = backward
        return result

    def tanh(self):
        output = math.tanh(self.data)
        result = Value(output, (self,), "tanh")

        def backward():
            self.grad += (
                1.0 - output ** 2
            ) * result.grad

        result.backward_fn = backward
        return result

    def relu(self):
        output = max(0.0, self.data)
        result = Value(output, (self,), "relu")

        def backward():
            self.grad += (
                float(self.data > 0) * result.grad
            )

        result.backward_fn = backward
        return result

    def backward(self):
        topology = []
        visited = set()

        def build(node):
            if node in visited:
                return
            visited.add(node)

            for parent in node.parents:
                build(parent)

            topology.append(node)

        build(self)

        # Clear graph gradients before a fresh backward pass.
        for node in topology:
            node.grad = 0.0

        self.grad = 1.0

        for node in reversed(topology):
            node.backward_fn()

    def __repr__(self):
        return (
            f"Value(data={self.data:.6f}, "
            f"grad={self.grad:.6f}, "
            f"op={self.operation!r})"
        )


def numerical_gradient(
    function: Callable[[float], float],
    x: float,
    epsilon: float = 1e-6
) -> float:
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")

    return (
        function(x + epsilon) -
        function(x - epsilon)
    ) / (2.0 * epsilon)


def demonstrate_chain_rule():
    print("\n1. Chain Rule Demonstration")

    x = Value(2.0, label="x")
    y = x * x + 3 * x + 2

    y.backward()

    # dy/dx = 2x + 3 = 7
    print(f"x = {x.data}")
    print(f"f(x) = {y.data}")
    print(f"Analytical derivative = {x.grad}")

    expected = numerical_gradient(
        lambda z: z * z + 3 * z + 2,
        2.0
    )

    print(f"Numerical derivative = {expected:.6f}")
    assert abs(x.grad - expected) < 1e-5


def demonstrate_shared_graph():
    print("\n2. Shared Computational Graph")

    x = Value(3.0)
    square = x * x
    expression = square + square + x

    expression.backward()

    # d(2x² + x)/dx = 4x + 1 = 13.
    print(f"Expression = {expression.data}")
    print(f"Accumulated gradient = {x.grad}")

    assert abs(x.grad - 13.0) < 1e-9


@dataclass
class Neuron:
    weights: list[Value]
    bias: Value

    @classmethod
    def create(cls, input_size: int):
        return cls(
            [
                Value(random.uniform(-1.0, 1.0))
                for _ in range(input_size)
            ],
            Value(0.0)
        )

    def __call__(self, inputs: list[Value]) -> Value:
        if len(inputs) != len(self.weights):
            raise ValueError("Input dimension mismatch.")

        result = self.bias

        for weight, value in zip(self.weights, inputs):
            result = result + weight * value

        return result.tanh()

    def parameters(self):
        return self.weights + [self.bias]


class MLP:
    def __init__(self, input_size: int, hidden_size: int):
        if input_size < 1 or hidden_size < 1:
            raise ValueError("Network dimensions must be positive.")

        self.hidden = [
            Neuron.create(input_size)
            for _ in range(hidden_size)
        ]

        self.output = Neuron.create(hidden_size)

    def __call__(self, inputs: list[float]) -> Value:
        if len(inputs) != len(self.hidden[0].weights):
            raise ValueError("Network input dimension mismatch.")

        values = [
            Value(float(value))
            for value in inputs
        ]

        hidden_values = [
            neuron(values)
            for neuron in self.hidden
        ]

        return self.output(hidden_values)

    def parameters(self):
        result = []

        for neuron in self.hidden:
            result.extend(neuron.parameters())

        result.extend(self.output.parameters())
        return result

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0

    def update(self, learning_rate: float):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        for parameter in self.parameters():
            parameter.data -= learning_rate * parameter.grad

            if not math.isfinite(parameter.data):
                raise FloatingPointError(
                    "Non-finite parameter encountered."
                )


def mean_squared_error(
    predictions: list[Value],
    targets: list[float]
) -> Value:
    if len(predictions) != len(targets) or not targets:
        raise ValueError("Invalid loss inputs.")

    loss = Value(0.0)

    for prediction, target in zip(predictions, targets):
        loss = loss + (prediction - target) ** 2

    return loss / len(targets)


def train_xor(
    epochs: int = 3000,
    learning_rate: float = 0.08,
    seed: int = 42
):
    print("\n3. Training an XOR Neural Network")

    random.seed(seed)

    dataset = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0)
    ]

    model = MLP(2, 4)

    for epoch in range(epochs):
        model.zero_grad()

        predictions = [
            model(inputs)
            for inputs, _ in dataset
        ]

        targets = [target for _, target in dataset]

        loss = mean_squared_error(predictions, targets)

        loss.backward()
        model.update(learning_rate)

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss {loss.data:.6f}"
            )

        if not math.isfinite(loss.data):
            raise FloatingPointError("Training diverged.")

    print("\nXOR Predictions")

    for inputs, target in dataset:
        prediction = model(inputs).data
        print(
            f"{inputs} -> {prediction:.4f} "
            f"(target={target:.0f})"
        )

    return model


def main():
    demonstrate_chain_rule()
    demonstrate_shared_graph()

    print("\n4. Gradient Verification")

    x = Value(1.5)
    expression = (x * x).exp()

    expression.backward()

    expected = numerical_gradient(
        lambda z: math.exp(z * z),
        1.5
    )

    print(f"Automatic gradient: {x.grad:.6f}")
    print(f"Numerical gradient: {expected:.6f}")

    assert abs(x.grad - expected) < 1e-3

    train_xor()


if __name__ == "__main__":
    main()
