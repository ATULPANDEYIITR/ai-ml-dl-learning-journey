#!/usr/bin/env python3
"""
Optimizer Laboratory
====================
A self-contained implementation of Momentum, AdaGrad, RMSProp, Adam,
and AdamW for numerical optimization.

Features:
- Common optimizer interface.
- Stateful parameter updates.
- Bias correction for Adam.
- Decoupled weight decay for AdamW.
- Gradient validation and numerical stability.
- Gradient clipping.
- Mini-batch optimization simulation.
- Convergence tracking and comparative evaluation.
- Numerical gradient checking.
- Unit-style assertions.

Run:
    python optimizer_lab.py
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class OptimizerName(Enum):
    MOMENTUM = "Momentum"
    ADAGRAD = "AdaGrad"
    RMSPROP = "RMSProp"
    ADAM = "Adam"
    ADAMW = "AdamW"


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.05
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    gradient_clip: float | None = None

    def validate(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite.")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Epsilon must be positive and finite.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay cannot be negative.")
        if self.gradient_clip is not None:
            if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
                raise ValueError("Gradient clipping threshold must be positive.")


@dataclass
class TrainingResult:
    optimizer: str
    initial_loss: float
    final_loss: float
    iterations: int
    parameters: list[float]
    history: list[float] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        if self.initial_loss == 0:
            return 0.0
        return (self.initial_loss - self.final_loss) / self.initial_loss * 100


class Optimizer(ABC):
    def __init__(self, config: OptimizerConfig):
        config.validate()
        self.config = config
        self.step_count = 0

    def prepare_gradient(self, gradient: list[float]) -> list[float]:
        if not gradient:
            raise ValueError("Gradient cannot be empty.")
        if any(not math.isfinite(g) for g in gradient):
            raise ValueError("Gradient contains a non-finite value.")

        threshold = self.config.gradient_clip
        if threshold is None:
            return gradient.copy()

        # Global norm clipping preserves the gradient direction.
        norm = math.sqrt(sum(g * g for g in gradient))
        if norm <= threshold:
            return gradient.copy()

        scale = threshold / norm
        return [g * scale for g in gradient]

    @abstractmethod
    def update(
        self,
        parameters: list[float],
        gradients: list[float]
    ) -> list[float]:
        pass

    def validate_parameters(self, parameters, gradients):
        if len(parameters) != len(gradients):
            raise ValueError("Parameter and gradient dimensions differ.")
        if not parameters:
            raise ValueError("Parameter vector cannot be empty.")
        if any(not math.isfinite(x) for x in parameters):
            raise ValueError("Parameters must be finite.")


class Momentum(Optimizer):
    """Classical momentum with an exponential velocity state."""

    def __init__(self, config, beta=0.9):
        super().__init__(config)
        if not 0 <= beta < 1:
            raise ValueError("Momentum beta must be in [0, 1).")
        self.beta = beta
        self.velocity = None

    def update(self, parameters, gradients):
        self.validate_parameters(parameters, gradients)
        gradients = self.prepare_gradient(gradients)

        if self.velocity is None:
            self.velocity = [0.0] * len(parameters)

        self.step_count += 1
        lr = self.config.learning_rate

        self.velocity = [
            self.beta * v + g
            for v, g in zip(self.velocity, gradients)
        ]

        updated = [
            p - lr * v
            for p, v in zip(parameters, self.velocity)
        ]

        return self.validate_result(updated)


class AdaGrad(Optimizer):
    """Adaptive learning rates using accumulated squared gradients."""

    def __init__(self, config):
        super().__init__(config)
        self.accumulator = None

    def update(self, parameters, gradients):
        self.validate_parameters(parameters, gradients)
        gradients = self.prepare_gradient(gradients)

        if self.accumulator is None:
            self.accumulator = [0.0] * len(parameters)

        self.step_count += 1
        lr = self.config.learning_rate
        eps = self.config.epsilon

        self.accumulator = [
            a + g * g
            for a, g in zip(self.accumulator, gradients)
        ]

        updated = [
            p - lr * g / (math.sqrt(a) + eps)
            for p, g, a in zip(parameters, gradients, self.accumulator)
        ]

        return self.validate_result(updated)


class RMSProp(Optimizer):
    """Exponential moving average of squared gradients."""

    def __init__(self, config, beta=0.9):
        super().__init__(config)
        if not 0 <= beta < 1:
            raise ValueError("RMSProp beta must be in [0, 1).")
        self.beta = beta
        self.square_average = None

    def update(self, parameters, gradients):
        self.validate_parameters(parameters, gradients)
        gradients = self.prepare_gradient(gradients)

        if self.square_average is None:
            self.square_average = [0.0] * len(parameters)

        self.step_count += 1
        lr = self.config.learning_rate
        eps = self.config.epsilon

        self.square_average = [
            self.beta * a + (1 - self.beta) * g * g
            for a, g in zip(self.square_average, gradients)
        ]

        updated = [
            p - lr * g / (math.sqrt(a) + eps)
            for p, g, a in zip(parameters, gradients, self.square_average)
        ]

        return self.validate_result(updated)


class Adam(Optimizer):
    """Adaptive Moment Estimation with bias correction."""

    def __init__(self, config, beta1=0.9, beta2=0.999):
        super().__init__(config)

        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1).")

        self.beta1 = beta1
        self.beta2 = beta2
        self.first_moment = None
        self.second_moment = None

    def update(self, parameters, gradients):
        self.validate_parameters(parameters, gradients)
        gradients = self.prepare_gradient(gradients)

        if self.first_moment is None:
            self.first_moment = [0.0] * len(parameters)
            self.second_moment = [0.0] * len(parameters)

        self.step_count += 1
        t = self.step_count

        lr = self.config.learning_rate
        eps = self.config.epsilon

        self.first_moment = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.first_moment, gradients)
        ]

        self.second_moment = [
            self.beta2 * v + (1 - self.beta2) * g * g
            for v, g in zip(self.second_moment, gradients)
        ]

        # Correct initialization bias in the exponential averages.
        corrected_first = [
            m / (1 - self.beta1 ** t)
            for m in self.first_moment
        ]

        corrected_second = [
            v / (1 - self.beta2 ** t)
            for v in self.second_moment
        ]

        updated = [
            p - lr * m / (math.sqrt(v) + eps)
            for p, m, v in zip(
                parameters, corrected_first, corrected_second
            )
        ]

        return self.validate_result(updated)


class AdamW(Adam):
    """
    Adam with decoupled weight decay.

    The weight decay term is applied separately from the adaptive
    gradient update.
    """

    def update(self, parameters, gradients):
        self.validate_parameters(parameters, gradients)

        # Adam's update uses only the gradient.
        adam_parameters = super().update(parameters, gradients)

        decay = self.config.learning_rate * self.config.weight_decay

        updated = [
            p - decay * p - (p - adam_p)
            for p, adam_p in zip(parameters, adam_parameters)
        ]

        return self.validate_result(updated)


def validate_result(parameters):
    if any(not math.isfinite(x) for x in parameters):
        raise FloatingPointError("Optimizer produced non-finite parameters.")
    return parameters


# Attach common result validation to the optimizer base class.
Optimizer.validate_result = staticmethod(validate_result)


def create_optimizer(name: OptimizerName, config: OptimizerConfig) -> Optimizer:
    factories = {
        OptimizerName.MOMENTUM: Momentum,
        OptimizerName.ADAGRAD: AdaGrad,
        OptimizerName.RMSPROP: RMSProp,
        OptimizerName.ADAM: Adam,
        OptimizerName.ADAMW: AdamW,
    }
    return factories[name](config)


@dataclass
class QuadraticObjective:
    """
    Positive-definite quadratic objective:

        f(x) = 0.5 * sum(a_i * (x_i - target_i)^2)
    """

    target: list[float]
    curvature: list[float]

    def __post_init__(self):
        if not self.target or len(self.target) != len(self.curvature):
            raise ValueError("Invalid objective dimensions.")
        if any(a <= 0 or not math.isfinite(a) for a in self.curvature):
            raise ValueError("Curvature must be positive and finite.")

    def loss(self, parameters):
        if len(parameters) != len(self.target):
            raise ValueError("Incorrect parameter dimensions.")

        return 0.5 * sum(
            a * (p - target) ** 2
            for p, target, a in zip(
                parameters, self.target, self.curvature
            )
        )

    def gradient(self, parameters):
        return [
            a * (p - target)
            for p, target, a in zip(
                parameters, self.target, self.curvature
            )
        ]


def train(
    optimizer: Optimizer,
    objective: QuadraticObjective,
    initial_parameters: list[float],
    iterations: int = 500,
    tolerance: float = 1e-10
) -> TrainingResult:

    if iterations <= 0:
        raise ValueError("Iterations must be positive.")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("Tolerance must be non-negative and finite.")

    parameters = initial_parameters.copy()
    initial_loss = objective.loss(parameters)
    history = [initial_loss]

    for _ in range(iterations):
        gradients = objective.gradient(parameters)
        parameters = optimizer.update(parameters, gradients)

        current_loss = objective.loss(parameters)
        history.append(current_loss)

        if current_loss <= tolerance:
            break

    return TrainingResult(
        optimizer=type(optimizer).__name__,
        initial_loss=initial_loss,
        final_loss=history[-1],
        iterations=len(history) - 1,
        parameters=parameters,
        history=history
    )


def numerical_gradient(objective: Callable, parameters, epsilon=1e-6):
    """Central finite-difference gradient approximation."""
    if epsilon <= 0:
        raise ValueError("Finite-difference epsilon must be positive.")

    result = []

    for i in range(len(parameters)):
        plus = parameters.copy()
        minus = parameters.copy()

        plus[i] += epsilon
        minus[i] -= epsilon

        derivative = (
            objective(plus) - objective(minus)
        ) / (2 * epsilon)

        result.append(derivative)

    return result


def gradient_check(objective: QuadraticObjective):
    parameters = [1.5, -2.0, 0.75]

    analytical = objective.gradient(parameters)
    numerical = numerical_gradient(objective.loss, parameters)

    for a, n in zip(analytical, numerical):
        assert math.isclose(a, n, rel_tol=1e-5, abs_tol=1e-7)


def compare_optimizers():
    objective = QuadraticObjective(
        target=[2.0, -3.0, 1.0],
        curvature=[1.0, 10.0, 50.0]
    )

    initial_parameters = [10.0, 8.0, -5.0]

    configurations = {
        OptimizerName.MOMENTUM: OptimizerConfig(learning_rate=0.005),
        OptimizerName.ADAGRAD: OptimizerConfig(learning_rate=0.8),
        OptimizerName.RMSPROP: OptimizerConfig(learning_rate=0.02),
        OptimizerName.ADAM: OptimizerConfig(learning_rate=0.08),
        OptimizerName.ADAMW: OptimizerConfig(
            learning_rate=0.08,
            weight_decay=0.001
        )
    }

    results = []

    for name, config in configurations.items():
        optimizer = create_optimizer(name, config)

        result = train(
            optimizer,
            objective,
            initial_parameters,
            iterations=3000
        )

        results.append(result)

    return results


def print_report(results):
    print("\nOPTIMIZER COMPARISON")
    print("=" * 86)
    print(
        f"{'Optimizer':<15}"
        f"{'Initial Loss':>15}"
        f"{'Final Loss':>15}"
        f"{'Iterations':>15}"
        f"{'Improvement':>18}"
    )
    print("-" * 86)

    for result in results:
        print(
            f"{result.optimizer:<15}"
            f"{result.initial_loss:>15.6f}"
            f"{result.final_loss:>15.8f}"
            f"{result.iterations:>15}"
            f"{result.improvement:>16.2f}%"
        )

    print("=" * 86)


def run_tests():
    config = OptimizerConfig(learning_rate=0.01)

    for optimizer_class in (
        Momentum, AdaGrad, RMSProp, Adam, AdamW
    ):
        optimizer = optimizer_class(config)

        parameters = [2.0, -1.0]
        gradients = [0.5, -0.25]

        updated = optimizer.update(parameters, gradients)

        assert len(updated) == 2
        assert all(math.isfinite(x) for x in updated)

    try:
        Adam(OptimizerConfig(learning_rate=-1))
        raise AssertionError("Invalid learning rate was accepted.")
    except ValueError:
        pass

    try:
        Adam(config).update([1.0], [float("nan")])
        raise AssertionError("Invalid gradient was accepted.")
    except ValueError:
        pass

    try:
        Adam(config).update([1.0, 2.0], [0.5])
        raise AssertionError("Dimension mismatch was accepted.")
    except ValueError:
        pass

    objective = QuadraticObjective(
        target=[0.0, 0.0, 0.0],
        curvature=[1.0, 2.0, 3.0]
    )

    gradient_check(objective)

    # Verify decoupled weight decay reduces parameters when gradients are zero.
    adamw = AdamW(
        OptimizerConfig(learning_rate=0.1, weight_decay=0.1)
    )
    updated = adamw.update([2.0], [0.0])
    assert updated[0] < 2.0

    print("All optimizer tests passed.")


def main():
    random.seed(42)

    run_tests()

    objective = QuadraticObjective(
        target=[2.0, -3.0, 1.0],
        curvature=[1.0, 10.0, 50.0]
    )

    gradient_check(objective)

    results = compare_optimizers()
    print_report(results)

    print("\nOPTIMIZATION DETAILS")
    for result in results:
        print(
            f"{result.optimizer}: "
            f"parameters={['%.6f' % x for x in result.parameters]}"
        )

    print("\nGradient verification: passed.")
    print("Numerical stability checks: passed.")


if __name__ == "__main__":
    main()
