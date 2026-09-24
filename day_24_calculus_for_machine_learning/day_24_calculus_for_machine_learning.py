"""
Calculus for Machine Learning
==============================

A self-contained study program covering:
    - Functions
    - Limits
    - Continuity
    - Derivatives
    - Rules of differentiation
    - Higher-order derivatives
    - Partial derivatives
    - Gradients
    - Directional derivatives
    - Jacobians
    - Hessians
    - Chain rule
    - Numerical differentiation
    - Finite differences
    - Automatic differentiation concepts
    - Taylor approximation
    - Optimization connections
    - Gradient descent
    - Learning-rate effects
    - Multivariable optimization
    - Logistic regression gradient derivation
    - Mean squared error
    - Binary cross-entropy
    - Softmax and cross-entropy
    - Backpropagation concepts
    - Numerical stability
    - Edge cases and common mistakes

The program uses only the Python standard library.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


Number = float
ScalarFunction = Callable[[float], float]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def heading(title: str) -> None:
    """Print a readable section heading."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def subsection(title: str) -> None:
    """Print a smaller section heading."""
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def approximate_equal(a: float, b: float, tolerance: float = 1e-9) -> bool:
    """Compare floating-point values using an absolute tolerance."""
    return abs(a - b) <= tolerance


def safe_exp(value: float) -> float:
    """
    Numerically safer exponential.

    Extremely large positive exponents overflow in ordinary exp().
    Machine-learning implementations routinely guard against this.
    """
    if value > 700:
        return math.exp(700)
    if value < -700:
        return 0.0
    return math.exp(value)


# ---------------------------------------------------------------------------
# 1. Functions
# ---------------------------------------------------------------------------

def demonstrate_functions() -> None:
    heading("1. Functions")

    subsection("1.1 Basic function notation")

    # A function maps an input x to exactly one output f(x).
    def f(x: float) -> float:
        return x**2 + 2 * x + 1

    x = 3.0
    print(f"f(x) = x² + 2x + 1")
    print(f"f({x}) = {f(x)}")

    # Machine-learning models are functions too.
    def linear_model(x: float, weight: float, bias: float) -> float:
        return weight * x + bias

    print("\nLinear model:")
    print("y_hat = w*x + b")
    print(f"w=2, b=1, x=5 -> {linear_model(5, 2, 1)}")

    subsection("1.2 Composition of functions")

    def square(x: float) -> float:
        return x * x

    def add_three(x: float) -> float:
        return x + 3

    def compose(
        outer: ScalarFunction,
        inner: ScalarFunction,
    ) -> ScalarFunction:
        return lambda x: outer(inner(x))

    composed = compose(square, add_three)

    print("g(x) = x + 3")
    print("f(x) = x²")
    print("(f ∘ g)(x) = f(g(x))")
    print(f"(f ∘ g)(2) = {composed(2)}")

    subsection("1.3 Common function families")

    functions: dict[str, ScalarFunction] = {
        "constant": lambda x: 5.0,
        "linear": lambda x: 2 * x + 1,
        "quadratic": lambda x: x**2,
        "cubic": lambda x: x**3,
        "exponential": math.exp,
        "logarithm": lambda x: math.log(x),
        "sigmoid": lambda x: 1 / (1 + math.exp(-x)),
    }

    test_inputs = {
        "constant": 2.0,
        "linear": 2.0,
        "quadratic": 2.0,
        "cubic": 2.0,
        "exponential": 2.0,
        "logarithm": 2.0,
        "sigmoid": 2.0,
    }

    for name, function in functions.items():
        print(f"{name:12s}: f(2) = {function(test_inputs[name]):.6f}")

    subsection("1.4 Domain restrictions")

    # log(x) is defined for x > 0 over the real numbers.
    valid_value = 2.0
    invalid_value = -1.0

    print(f"log({valid_value}) = {math.log(valid_value):.6f}")

    try:
        print(math.log(invalid_value))
    except ValueError as error:
        print(f"log({invalid_value}) is undefined over the reals: {error}")


# ---------------------------------------------------------------------------
# 2. Limits
# ---------------------------------------------------------------------------

def numerical_limit(
    function: ScalarFunction,
    point: float,
    *,
    side: str = "both",
    epsilon: float = 1e-5,
) -> float | tuple[float, float]:
    """
    Estimate a limit using points increasingly close to the target.

    This is a numerical approximation, not a symbolic proof.
    """
    left = function(point - epsilon)
    right = function(point + epsilon)

    if side == "left":
        return left
    if side == "right":
        return right
    return left, right


def demonstrate_limits() -> None:
    heading("2. Limits")

    subsection("2.1 Intuition")

    # lim_{x -> 2} x² = 4.
    square_function = lambda x: x**2

    for distance in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        left = square_function(2 - distance)
        right = square_function(2 + distance)
        print(
            f"distance={distance:.0e}, "
            f"left={left:.10f}, right={right:.10f}"
        )

    subsection("2.2 A removable discontinuity")

    # (x² - 1)/(x - 1) simplifies to x + 1 for x != 1.
    # The original expression is undefined exactly at x=1.
    def removable(x: float) -> float:
        return (x**2 - 1) / (x - 1)

    for distance in [1e-1, 1e-2, 1e-3, 1e-5]:
        left = removable(1 - distance)
        right = removable(1 + distance)
        print(
            f"distance={distance:.0e}, "
            f"left={left:.8f}, right={right:.8f}"
        )

    subsection("2.3 One-sided limits")

    # The absolute-value-over-x function has different one-sided limits
    # at zero, so its two-sided limit does not exist.
    def sign_like(x: float) -> float:
        return abs(x) / x

    for distance in [1e-2, 1e-4, 1e-6]:
        left = sign_like(-distance)
        right = sign_like(distance)
        print(
            f"distance={distance:.0e}: "
            f"left={left}, right={right}"
        )

    print("Since the one-sided limits differ, the two-sided limit does not exist.")


# ---------------------------------------------------------------------------
# 3. Continuity
# ---------------------------------------------------------------------------

def demonstrate_continuity() -> None:
    heading("3. Continuity")

    subsection("3.1 Continuous functions")

    def polynomial(x: float) -> float:
        return x**3 - 4 * x + 2

    x0 = 1.5
    value = polynomial(x0)

    print(f"f({x0}) = {value}")
    print("A polynomial is continuous for every real input.")

    subsection("3.2 Numerical continuity check")

    # A practical numerical check compares nearby function values.
    for distance in [1e-1, 1e-2, 1e-3, 1e-4]:
        nearby = polynomial(x0 + distance)
        print(
            f"|f({x0}+{distance}) - f({x0})| = "
            f"{abs(nearby - value):.10f}"
        )

    subsection("3.3 Why continuity matters in ML")

    print(
        "Continuous model components make numerical optimization easier to reason "
        "about. Neural networks may contain deliberately non-smooth components "
        "such as ReLU, so continuity and differentiability must be considered "
        "separately."
    )


# ---------------------------------------------------------------------------
# 4. Derivatives
# ---------------------------------------------------------------------------

def numerical_derivative(
    function: ScalarFunction,
    x: float,
    step: float = 1e-5,
) -> float:
    """
    Central finite difference:

        f'(x) ≈ [f(x+h) - f(x-h)] / (2h)

    Central differences usually provide better accuracy than a simple
    forward difference for smooth functions.
    """
    if step <= 0:
        raise ValueError("step must be positive")

    return (function(x + step) - function(x - step)) / (2 * step)


def forward_difference(
    function: ScalarFunction,
    x: float,
    step: float = 1e-5,
) -> float:
    """First-order forward finite difference."""
    return (function(x + step) - function(x)) / step


def backward_difference(
    function: ScalarFunction,
    x: float,
    step: float = 1e-5,
) -> float:
    """First-order backward finite difference."""
    return (function(x) - function(x - step)) / step


def demonstrate_derivatives() -> None:
    heading("4. Derivatives")

    subsection("4.1 Geometric and computational meaning")

    def quadratic(x: float) -> float:
        return x**2

    def exact_quadratic_derivative(x: float) -> float:
        return 2 * x

    x = 3.0

    print(f"f(x) = x²")
    print(f"Exact f'({x}) = {exact_quadratic_derivative(x)}")
    print(f"Numerical f'({x}) = {numerical_derivative(quadratic, x):.8f}")

    subsection("4.2 Forward, backward, and central differences")

    for step in [1e-1, 1e-2, 1e-3, 1e-4]:
        forward = forward_difference(quadratic, x, step)
        backward = backward_difference(quadratic, x, step)
        central = numerical_derivative(quadratic, x, step)

        print(
            f"h={step:.0e} | "
            f"forward={forward:.8f} | "
            f"backward={backward:.8f} | "
            f"central={central:.8f}"
        )

    subsection("4.3 Basic differentiation rules")

    print("Power rule: d(x^n)/dx = n*x^(n-1)")
    print("Constant rule: d(c)/dx = 0")
    print("Sum rule: d(f+g)/dx = f' + g'")
    print("Product rule: d(fg)/dx = f'g + fg'")
    print("Quotient rule: d(f/g)/dx = (f'g - fg') / g²")
    print("Chain rule: d f(g(x))/dx = f'(g(x))*g'(x)")

    subsection("4.4 Important derivatives")

    test_x = 2.0

    derivative_examples = {
        "x²": 2 * test_x,
        "x³": 3 * test_x**2,
        "exp(x)": math.exp(test_x),
        "log(x)": 1 / test_x,
        "sin(x)": math.cos(test_x),
        "cos(x)": -math.sin(test_x),
    }

    for expression, derivative_value in derivative_examples.items():
        print(f"d({expression})/dx at x={test_x}: {derivative_value:.8f}")


# ---------------------------------------------------------------------------
# 5. Higher-order derivatives
# ---------------------------------------------------------------------------

def second_derivative(
    function: ScalarFunction,
    x: float,
    step: float = 1e-4,
) -> float:
    """Central finite-difference approximation of the second derivative."""
    if step <= 0:
        raise ValueError("step must be positive")

    return (
        function(x + step)
        - 2 * function(x)
        + function(x - step)
    ) / (step**2)


def demonstrate_higher_order_derivatives() -> None:
    heading("5. Higher-Order Derivatives")

    function = lambda x: x**4

    x = 2.0

    # f'(x) = 4x³
    # f''(x) = 12x²
    exact_second = 12 * x**2
    numerical_second = second_derivative(function, x)

    print(f"f(x) = x⁴")
    print(f"Exact f''({x}) = {exact_second}")
    print(f"Numerical f''({x}) ≈ {numerical_second:.6f}")

    print(
        "The second derivative describes curvature in one dimension. "
        "Curvature becomes important when analyzing optimization behavior."
    )


# ---------------------------------------------------------------------------
# 6. Partial derivatives
# ---------------------------------------------------------------------------

Vector = list[float]


def numerical_partial_derivative(
    function: Callable[[Vector], float],
    point: Vector,
    variable_index: int,
    step: float = 1e-5,
) -> float:
    """
    Central finite difference for one coordinate of a multivariable function.
    """
    if not 0 <= variable_index < len(point):
        raise IndexError("variable_index is outside the point")

    if step <= 0:
        raise ValueError("step must be positive")

    plus = point.copy()
    minus = point.copy()

    plus[variable_index] += step
    minus[variable_index] -= step

    return (function(plus) - function(minus)) / (2 * step)


def numerical_gradient(
    function: Callable[[Vector], float],
    point: Vector,
    step: float = 1e-5,
) -> Vector:
    """Compute all first-order partial derivatives numerically."""
    return [
        numerical_partial_derivative(function, point, i, step)
        for i in range(len(point))
    ]


def demonstrate_partial_derivatives() -> None:
    heading("6. Partial Derivatives")

    subsection("6.1 Definition")

    # f(x, y) = x² + 3xy + y²
    # ∂f/∂x = 2x + 3y
    # ∂f/∂y = 3x + 2y
    def function(point: Vector) -> float:
        x, y = point
        return x**2 + 3 * x * y + y**2

    point = [2.0, 4.0]

    analytical_dx = 2 * point[0] + 3 * point[1]
    analytical_dy = 3 * point[0] + 2 * point[1]

    numerical_gradient_value = numerical_gradient(function, point)

    print(f"f(x,y) = x² + 3xy + y²")
    print(f"Point = {point}")
    print(f"∂f/∂x = {analytical_dx}")
    print(f"∂f/∂y = {analytical_dy}")
    print(
        "Numerical gradient = "
        f"[{numerical_gradient_value[0]:.8f}, "
        f"{numerical_gradient_value[1]:.8f}]"
    )

    subsection("6.2 Why partial derivatives matter in ML")

    print(
        "A machine-learning model can have thousands, millions, or billions "
        "of parameters. A partial derivative measures how the output or loss "
        "changes with respect to one parameter while the other parameters are "
        "held fixed."
    )


# ---------------------------------------------------------------------------
# 7. Gradient and directional derivative
# ---------------------------------------------------------------------------

def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    return sum(x * y for x, y in zip(a, b))


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(dot_product(vector, vector))


def normalize(vector: Sequence[float]) -> Vector:
    magnitude = vector_norm(vector)

    if magnitude == 0:
        raise ValueError("zero vector cannot be normalized")

    return [value / magnitude for value in vector]


def directional_derivative(
    function: Callable[[Vector], float],
    point: Vector,
    direction: Vector,
    step: float = 1e-5,
) -> float:
    """
    Approximate the directional derivative along a normalized direction.
    """
    unit_direction = normalize(direction)

    plus = [
        coordinate + step * delta
        for coordinate, delta in zip(point, unit_direction)
    ]
    minus = [
        coordinate - step * delta
        for coordinate, delta in zip(point, unit_direction)
    ]

    return (function(plus) - function(minus)) / (2 * step)


def demonstrate_gradients() -> None:
    heading("7. Gradients and Directional Derivatives")

    def surface(point: Vector) -> float:
        x, y = point
        return x**2 + 2 * y**2

    point = [2.0, 3.0]
    gradient = numerical_gradient(surface, point)

    print(f"f(x,y) = x² + 2y²")
    print(f"Point = {point}")
    print(f"Gradient ≈ {gradient}")

    direction = [1.0, 1.0]

    directional = directional_derivative(
        surface,
        point,
        direction,
    )

    print(f"Directional derivative toward {direction}: {directional:.8f}")

    print(
        "The gradient points in the direction of greatest local increase. "
        "Its negative therefore gives the local steepest-descent direction."
    )


# ---------------------------------------------------------------------------
# 8. Jacobian and Hessian
# ---------------------------------------------------------------------------

def numerical_jacobian(
    functions: Sequence[Callable[[Vector], float]],
    point: Vector,
    step: float = 1e-5,
) -> list[list[float]]:
    """Numerically calculate a Jacobian matrix."""
    return [
        numerical_gradient(function, point, step)
        for function in functions
    ]


def numerical_hessian(
    function: Callable[[Vector], float],
    point: Vector,
    step: float = 1e-4,
) -> list[list[float]]:
    """
    Numerically calculate a Hessian matrix using nested finite differences.
    """
    dimension = len(point)
    hessian = [[0.0 for _ in range(dimension)] for _ in range(dimension)]

    for i in range(dimension):
        for j in range(dimension):
            pp = point.copy()
            pm = point.copy()
            mp = point.copy()
            mm = point.copy()

            pp[i] += step
            pp[j] += step

            pm[i] += step
            pm[j] -= step

            mp[i] -= step
            mp[j] += step

            mm[i] -= step
            mm[j] -= step

            hessian[i][j] = (
                function(pp)
                - function(pm)
                - function(mp)
                + function(mm)
            ) / (4 * step**2)

    return hessian


def demonstrate_jacobian_hessian() -> None:
    heading("8. Jacobians and Hessians")

    point = [2.0, 3.0]

    def output_one(p: Vector) -> float:
        x, y = p
        return x**2 + y

    def output_two(p: Vector) -> float:
        x, y = p
        return x * y

    jacobian = numerical_jacobian(
        [output_one, output_two],
        point,
    )

    print("Vector-valued function:")
    print("F(x,y) = [x²+y, xy]")
    print("Jacobian:")
    for row in jacobian:
        print("  ", [round(value, 6) for value in row])

    def scalar_loss(p: Vector) -> float:
        x, y = p
        return x**2 + 3 * y**2 + x * y

    hessian = numerical_hessian(scalar_loss, point)

    print("\nHessian of x² + 3y² + xy:")
    for row in hessian:
        print("  ", [round(value, 6) for value in row])

    print(
        "The Jacobian generalizes derivatives to vector-valued functions. "
        "The Hessian contains second-order partial derivatives of a scalar "
        "function."
    )


# ---------------------------------------------------------------------------
# 9. Chain rule
# ---------------------------------------------------------------------------

def demonstrate_chain_rule() -> None:
    heading("9. Chain Rule")

    subsection("9.1 Scalar chain rule")

    # y = sin(x²)
    # dy/dx = cos(x²) * 2x
    x = 1.2

    analytical = math.cos(x**2) * 2 * x

    function = lambda value: math.sin(value**2)
    numerical = numerical_derivative(function, x)

    print("y = sin(x²)")
    print(f"Analytical derivative at x={x}: {analytical:.8f}")
    print(f"Numerical derivative: {numerical:.8f}")

    subsection("9.2 Machine-learning interpretation")

    print(
        "Neural networks are compositions of functions. Backpropagation applies "
        "the chain rule repeatedly to calculate how a final loss changes with "
        "respect to parameters in earlier layers."
    )


# ---------------------------------------------------------------------------
# 10. Taylor approximation
# ---------------------------------------------------------------------------

def taylor_first_order(
    function: ScalarFunction,
    derivative: ScalarFunction,
    center: float,
    x: float,
) -> float:
    """First-order Taylor approximation."""
    return function(center) + derivative(center) * (x - center)


def taylor_second_order(
    function: ScalarFunction,
    derivative: ScalarFunction,
    second_derivative_function: ScalarFunction,
    center: float,
    x: float,
) -> float:
    """Second-order Taylor approximation."""
    delta = x - center

    return (
        function(center)
        + derivative(center) * delta
        + 0.5 * second_derivative_function(center) * delta**2
    )


def demonstrate_taylor() -> None:
    heading("10. Taylor Approximation")

    function = math.exp
    derivative = math.exp
    second_derivative_function = math.exp

    center = 0.0
    x = 0.5

    exact = function(x)
    first_order = taylor_first_order(
        function,
        derivative,
        center,
        x,
    )
    second_order = taylor_second_order(
        function,
        derivative,
        second_derivative_function,
        center,
        x,
    )

    print(f"exp({x}) exact = {exact:.8f}")
    print(f"First-order approximation = {first_order:.8f}")
    print(f"Second-order approximation = {second_order:.8f}")


# ---------------------------------------------------------------------------
# 11. Loss functions
# ---------------------------------------------------------------------------

def mean_squared_error(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> float:
    """Mean squared error."""
    if not predictions:
        raise ValueError("predictions cannot be empty")

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")

    return sum(
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets)
    ) / len(predictions)


def mse_gradient_predictions(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> Vector:
    """
    Derivative of MSE with respect to each prediction:

        dMSE/dy_hat_i = 2(y_hat_i - y_i)/n
    """
    if not predictions:
        raise ValueError("predictions cannot be empty")

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")

    n = len(predictions)

    return [
        2 * (prediction - target) / n
        for prediction, target in zip(predictions, targets)
    ]


def sigmoid(value: float) -> float:
    """Numerically safer sigmoid."""
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)

    z = math.exp(value)
    return z / (1 + z)


def binary_cross_entropy(
    predictions: Sequence[float],
    targets: Sequence[int],
) -> float:
    """
    Binary cross-entropy.

    Clipping avoids log(0), which is undefined.
    """
    if not predictions:
        raise ValueError("predictions cannot be empty")

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")

    epsilon = 1e-15
    total = 0.0

    for prediction, target in zip(predictions, targets):
        if target not in (0, 1):
            raise ValueError("binary targets must be 0 or 1")

        clipped = min(max(prediction, epsilon), 1 - epsilon)

        total += -(
            target * math.log(clipped)
            + (1 - target) * math.log(1 - clipped)
        )

    return total / len(predictions)


def demonstrate_loss_functions() -> None:
    heading("11. Loss Functions and Their Derivatives")

    predictions = [2.5, 3.5, 4.2]
    targets = [3.0, 3.0, 5.0]

    mse = mean_squared_error(predictions, targets)
    mse_gradients = mse_gradient_predictions(predictions, targets)

    print(f"Predictions: {predictions}")
    print(f"Targets:     {targets}")
    print(f"MSE = {mse:.8f}")
    print(f"dMSE/dPredictions = {mse_gradients}")

    probabilities = [0.9, 0.2, 0.8, 0.1]
    binary_targets = [1, 0, 1, 0]

    bce = binary_cross_entropy(probabilities, binary_targets)

    print(f"\nBinary cross-entropy = {bce:.8f}")

    print(
        "Loss functions convert model predictions into a scalar quantity that "
        "optimization algorithms can minimize."
    )


# ---------------------------------------------------------------------------
# 12. Logistic regression from first principles
# ---------------------------------------------------------------------------

@dataclass
class LogisticRegression:
    """Minimal binary logistic-regression model."""

    weights: Vector
    bias: float

    def predict_logit(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("feature dimension does not match weight dimension")

        return dot_product(self.weights, features) + self.bias

    def predict_probability(self, features: Sequence[float]) -> float:
        return sigmoid(self.predict_logit(features))

    def predict_class(self, features: Sequence[float], threshold: float = 0.5) -> int:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")

        return int(self.predict_probability(features) >= threshold)


def logistic_loss_and_gradients(
    model: LogisticRegression,
    features: Sequence[Sequence[float]],
    targets: Sequence[int],
) -> tuple[float, Vector, float]:
    """
    Binary cross-entropy gradient for logistic regression.

    For a sample:
        z = w·x + b
        p = sigmoid(z)

    For BCE + sigmoid:
        dL/dz = p - y

    Therefore:
        dL/dw = (p-y)x
        dL/db = p-y
    """
    if not features:
        raise ValueError("features cannot be empty")

    if len(features) != len(targets):
        raise ValueError("feature rows and targets must have equal length")

    dimension = len(model.weights)

    gradient_weights = [0.0] * dimension
    gradient_bias = 0.0
    total_loss = 0.0

    epsilon = 1e-15

    for row, target in zip(features, targets):
        if len(row) != dimension:
            raise ValueError("inconsistent feature dimensions")

        if target not in (0, 1):
            raise ValueError("targets must be 0 or 1")

        probability = model.predict_probability(row)
        clipped_probability = min(
            max(probability, epsilon),
            1 - epsilon,
        )

        total_loss += -(
            target * math.log(clipped_probability)
            + (1 - target) * math.log(1 - clipped_probability)
        )

        error = probability - target

        for index in range(dimension):
            gradient_weights[index] += error * row[index]

        gradient_bias += error

    sample_count = len(features)

    loss = total_loss / sample_count
    gradient_weights = [
        gradient / sample_count
        for gradient in gradient_weights
    ]
    gradient_bias /= sample_count

    return loss, gradient_weights, gradient_bias


def train_logistic_regression() -> LogisticRegression:
    heading("12. Logistic Regression and Gradient Descent")

    # A small linearly separable dataset.
    features = [
        [0.0, 0.0],
        [0.2, 0.1],
        [0.1, 0.3],
        [0.3, 0.2],
        [1.0, 1.0],
        [1.2, 0.9],
        [0.8, 1.1],
        [1.1, 1.3],
    ]

    targets = [0, 0, 0, 0, 1, 1, 1, 1]

    model = LogisticRegression(
        weights=[0.0, 0.0],
        bias=0.0,
    )

    learning_rate = 0.8

    print("Training logistic regression using gradient descent.")

    for epoch in range(1, 1001):
        loss, gradients, bias_gradient = logistic_loss_and_gradients(
            model,
            features,
            targets,
        )

        for index in range(len(model.weights)):
            model.weights[index] -= learning_rate * gradients[index]

        model.bias -= learning_rate * bias_gradient

        if epoch in {1, 2, 5, 10, 100, 500, 1000}:
            print(
                f"epoch={epoch:4d} "
                f"loss={loss:.8f} "
                f"weights={[round(w, 4) for w in model.weights]} "
                f"bias={model.bias:.4f}"
            )

    subsection("Predictions")

    for row, target in zip(features, targets):
        probability = model.predict_probability(row)
        prediction = model.predict_class(row)

        print(
            f"features={row}, target={target}, "
            f"probability={probability:.4f}, prediction={prediction}"
        )

    return model


# ---------------------------------------------------------------------------
# 13. Gradient descent on a mathematical function
# ---------------------------------------------------------------------------

def gradient_descent_one_dimension(
    function: ScalarFunction,
    derivative: ScalarFunction,
    initial_value: float,
    learning_rate: float,
    iterations: int,
) -> tuple[float, list[float]]:
    """Simple one-dimensional gradient descent."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if iterations < 0:
        raise ValueError("iterations cannot be negative")

    value = initial_value
    history = [value]

    for _ in range(iterations):
        value -= learning_rate * derivative(value)
        history.append(value)

    return value, history


def demonstrate_gradient_descent() -> None:
    heading("13. Gradient Descent")

    # f(x) = (x - 4)^2
    # f'(x) = 2(x - 4)
    function = lambda x: (x - 4) ** 2
    derivative = lambda x: 2 * (x - 4)

    final_value, history = gradient_descent_one_dimension(
        function,
        derivative,
        initial_value=0.0,
        learning_rate=0.1,
        iterations=30,
    )

    print(f"Initial x = {history[0]}")
    print(f"Final x ≈ {final_value:.8f}")
    print(f"Final f(x) ≈ {function(final_value):.8f}")

    print("\nSelected trajectory:")
    for index in [0, 1, 2, 5, 10, 20, 30]:
        value = history[index]
        print(
            f"iteration={index:2d}, "
            f"x={value:.8f}, "
            f"f(x)={function(value):.8f}"
        )

    subsection("13.1 Learning-rate behavior")

    for learning_rate in [0.05, 0.2, 0.5, 1.0, 1.1]:
        value, trajectory = gradient_descent_one_dimension(
            function,
            derivative,
            initial_value=0.0,
            learning_rate=learning_rate,
            iterations=20,
        )

        print(
            f"learning_rate={learning_rate:.2f}, "
            f"x_after_20={value:.6f}, "
            f"f(x)={function(value):.6f}"
        )

    print(
        "A learning rate that is too small can produce slow convergence. "
        "A learning rate that is too large can cause oscillation or divergence."
    )


# ---------------------------------------------------------------------------
# 14. ReLU, sigmoid, and activation derivatives
# ---------------------------------------------------------------------------

def relu(x: float) -> float:
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    # ReLU is not classically differentiable at x=0.
    # Many implementations choose derivative 0 at that point.
    if x > 0:
        return 1.0
    return 0.0


def sigmoid_derivative_from_output(output: float) -> float:
    """
    If s = sigmoid(x), then s' = s(1-s).
    """
    return output * (1 - output)


def demonstrate_activations() -> None:
    heading("14. Activation Functions and Differentiability")

    values = [-2.0, -0.5, 0.0, 0.5, 2.0]

    print("x        ReLU       ReLU'      sigmoid     sigmoid'")
    for value in values:
        sigmoid_value = sigmoid(value)

        print(
            f"{value:5.1f} "
            f"{relu(value):9.5f} "
            f"{relu_derivative(value):9.5f} "
            f"{sigmoid_value:10.5f} "
            f"{sigmoid_derivative_from_output(sigmoid_value):10.5f}"
        )

    print(
        "\nReLU is continuous at zero but not classically differentiable there. "
        "This illustrates why continuity and differentiability are distinct "
        "properties."
    )


# ---------------------------------------------------------------------------
# 15. Numerical differentiation pitfalls
# ---------------------------------------------------------------------------

def demonstrate_numerical_pitfalls() -> None:
    heading("15. Numerical Differentiation: Accuracy and Failure Modes")

    function = math.sin
    point = 1.0
    exact = math.cos(point)

    print(f"Exact derivative of sin(x) at x=1: {exact:.12f}")

    for step in [
        1e-1,
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
        1e-7,
        1e-8,
    ]:
        estimate = numerical_derivative(
            function,
            point,
            step,
        )

        error = abs(estimate - exact)

        print(
            f"h={step:.0e}, "
            f"estimate={estimate:.12f}, "
            f"absolute_error={error:.3e}"
        )

    print(
        "\nVery small finite-difference steps are not automatically better. "
        "Floating-point round-off can become significant because nearby "
        "function values are subtracted."
    )


# ---------------------------------------------------------------------------
# 16. Automatic differentiation concept
# ---------------------------------------------------------------------------

class DualNumber:
    """
    Tiny forward-mode automatic differentiation implementation.

    A dual number represents:

        value + derivative * ε

    where ε² = 0.

    This lets ordinary arithmetic propagate derivative information.
    """

    def __init__(self, value: float, derivative: float = 0.0):
        self.value = float(value)
        self.derivative = float(derivative)

    def __add__(self, other: "DualNumber | float") -> "DualNumber":
        other = to_dual(other)

        return DualNumber(
            self.value + other.value,
            self.derivative + other.derivative,
        )

    __radd__ = __add__

    def __sub__(self, other: "DualNumber | float") -> "DualNumber":
        other = to_dual(other)

        return DualNumber(
            self.value - other.value,
            self.derivative - other.derivative,
        )

    def __rsub__(self, other: "DualNumber | float") -> "DualNumber":
        other = to_dual(other)

        return DualNumber(
            other.value - self.value,
            other.derivative - self.derivative,
        )

    def __mul__(self, other: "DualNumber | float") -> "DualNumber":
        other = to_dual(other)

        return DualNumber(
            self.value * other.value,
            self.derivative * other.value
            + self.value * other.derivative,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: "DualNumber | float") -> "DualNumber":
        other = to_dual(other)

        if other.value == 0:
            raise ZeroDivisionError("division by zero")

        value = self.value / other.value

        derivative = (
            self.derivative * other.value
            - self.value * other.derivative
        ) / (other.value**2)

        return DualNumber(value, derivative)

    def __rtruediv__(self, other: "DualNumber | float") -> "DualNumber":
        other = to_dual(other)
        return other / self

    def __pow__(self, power: float) -> "DualNumber":
        if self.value == 0 and power < 1:
            raise ValueError("undefined real power at zero")

        value = self.value**power

        derivative = (
            power * self.value ** (power - 1) * self.derivative
            if power != 0
            else 0.0
        )

        return DualNumber(value, derivative)

    def __repr__(self) -> str:
        return (
            f"DualNumber(value={self.value}, "
            f"derivative={self.derivative})"
        )


def to_dual(value: DualNumber | float) -> DualNumber:
    if isinstance(value, DualNumber):
        return value

    return DualNumber(float(value), 0.0)


def dual_sin(value: DualNumber) -> DualNumber:
    return DualNumber(
        math.sin(value.value),
        math.cos(value.value) * value.derivative,
    )


def dual_exp(value: DualNumber) -> DualNumber:
    exponential = math.exp(value.value)

    return DualNumber(
        exponential,
        exponential * value.derivative,
    )


def dual_log(value: DualNumber) -> DualNumber:
    if value.value <= 0:
        raise ValueError("log requires a positive value")

    return DualNumber(
        math.log(value.value),
        value.derivative / value.value,
    )


def demonstrate_forward_mode_autodiff() -> None:
    heading("16. Forward-Mode Automatic Differentiation")

    # f(x) = sin(x²) + exp(x)
    # Seed x's derivative as 1.
    x = DualNumber(1.2, 1.0)

    result = dual_sin(x**2) + dual_exp(x)

    analytical = 2 * x.value * math.cos(x.value**2) + math.exp(x.value)

    print(f"Function: f(x) = sin(x²) + exp(x)")
    print(f"Value = {result.value:.10f}")
    print(f"AD derivative = {result.derivative:.10f}")
    print(f"Analytical derivative = {analytical:.10f}")

    print(
        "Forward-mode automatic differentiation propagates exact derivative "
        "information according to the arithmetic operations. It is different "
        "from finite differences because it does not estimate the derivative "
        "by subtracting two nearby function evaluations."
    )


# ---------------------------------------------------------------------------
# 17. Softmax and cross-entropy
# ---------------------------------------------------------------------------

def softmax(logits: Sequence[float]) -> Vector:
    """Stable softmax implementation using the max-shift trick."""
    if not logits:
        raise ValueError("logits cannot be empty")

    maximum = max(logits)

    exponentials = [
        math.exp(logit - maximum)
        for logit in logits
    ]

    denominator = sum(exponentials)

    return [
        value / denominator
        for value in exponentials
    ]


def multiclass_cross_entropy(
    probabilities: Sequence[float],
    target_index: int,
) -> float:
    if not probabilities:
        raise ValueError("probabilities cannot be empty")

    if not 0 <= target_index < len(probabilities):
        raise IndexError("target index is outside the probability vector")

    epsilon = 1e-15
    probability = min(
        max(probabilities[target_index], epsilon),
        1 - epsilon,
    )

    return -math.log(probability)


def demonstrate_softmax() -> None:
    heading("17. Softmax and Multiclass Cross-Entropy")

    logits = [2.0, 1.0, 0.1]
    probabilities = softmax(logits)

    print(f"Logits: {logits}")
    print(
        "Probabilities:",
        [round(probability, 8) for probability in probabilities],
    )
    print(f"Probability sum = {sum(probabilities):.8f}")

    target_index = 0
    loss = multiclass_cross_entropy(
        probabilities,
        target_index,
    )

    print(f"Target class = {target_index}")
    print(f"Cross-entropy loss = {loss:.8f}")

    print(
        "Subtracting the maximum logit before exponentiation preserves the "
        "softmax probabilities while reducing overflow risk."
    )


# ---------------------------------------------------------------------------
# 18. Simple neural-network forward pass
# ---------------------------------------------------------------------------

@dataclass
class TinyNeuralNetwork:
    """
    Two-input, two-hidden-unit, one-output neural network.

    Architecture:

        inputs -> linear hidden layer -> sigmoid -> linear output

    This is intentionally small enough to inspect mathematically.
    """

    hidden_weights: list[list[float]]
    hidden_biases: Vector
    output_weights: Vector
    output_bias: float

    def forward(self, inputs: Sequence[float]) -> tuple[Vector, float]:
        if len(inputs) != 2:
            raise ValueError("this network expects exactly two inputs")

        hidden_outputs: Vector = []

        for weights, bias in zip(
            self.hidden_weights,
            self.hidden_biases,
        ):
            z = dot_product(weights, inputs) + bias
            hidden_outputs.append(sigmoid(z))

        output_logit = (
            dot_product(self.output_weights, hidden_outputs)
            + self.output_bias
        )

        return hidden_outputs, sigmoid(output_logit)


def demonstrate_neural_network() -> None:
    heading("18. Tiny Neural Network Forward Pass")

    network = TinyNeuralNetwork(
        hidden_weights=[
            [0.5, -0.3],
            [0.8, 0.2],
        ],
        hidden_biases=[0.1, -0.1],
        output_weights=[0.7, -0.4],
        output_bias=0.05,
    )

    inputs = [1.0, 2.0]
    hidden, output = network.forward(inputs)

    print(f"Input: {inputs}")
    print(f"Hidden activations: {[round(v, 6) for v in hidden]}")
    print(f"Output probability: {output:.6f}")

    print(
        "The complete derivative of the output with respect to an early "
        "parameter would require the chain rule through every operation in "
        "the network."
    )


# ---------------------------------------------------------------------------
# 19. Optimization edge cases
# ---------------------------------------------------------------------------

def demonstrate_optimization_edge_cases() -> None:
    heading("19. Optimization Edge Cases")

    subsection("19.1 Flat gradient")

    # f(x)=constant has zero derivative everywhere.
    flat = lambda x: 7.0
    flat_derivative = lambda x: 0.0

    value, _ = gradient_descent_one_dimension(
        flat,
        flat_derivative,
        initial_value=100.0,
        learning_rate=0.1,
        iterations=10,
    )

    print(f"Flat function: starting at 100 -> {value}")

    subsection("19.2 Stationary point")

    # f(x)=x² has derivative 0 at x=0.
    quadratic = lambda x: x**2
    quadratic_derivative = lambda x: 2 * x

    value, _ = gradient_descent_one_dimension(
        quadratic,
        quadratic_derivative,
        initial_value=0.0,
        learning_rate=0.1,
        iterations=10,
    )

    print(f"Already at stationary point: x={value}")

    subsection("19.3 Invalid learning rate")

    try:
        gradient_descent_one_dimension(
            quadratic,
            quadratic_derivative,
            initial_value=1.0,
            learning_rate=-0.1,
            iterations=10,
        )
    except ValueError as error:
        print(f"Validation caught invalid rate: {error}")

    subsection("19.4 Zero direction")

    try:
        directional_derivative(
            lambda p: p[0] ** 2,
            [2.0],
            [0.0],
        )
    except ValueError as error:
        print(f"Validation caught zero direction: {error}")


# ---------------------------------------------------------------------------
# 20. Calculus and machine-learning concepts
# ---------------------------------------------------------------------------

def demonstrate_ml_connections() -> None:
    heading("20. Calculus Concepts Used in Machine Learning")

    concepts = [
        (
            "Function",
            "Maps inputs, such as features, to outputs or predictions.",
        ),
        (
            "Limit",
            "Provides the mathematical foundation for derivative definitions.",
        ),
        (
            "Derivative",
            "Measures local sensitivity of a scalar output to a scalar input.",
        ),
        (
            "Partial derivative",
            "Measures sensitivity to one variable while holding others fixed.",
        ),
        (
            "Gradient",
            "Collects first-order partial derivatives of a scalar function.",
        ),
        (
            "Jacobian",
            "Collects first-order derivatives of vector-valued outputs.",
        ),
        (
            "Hessian",
            "Collects second-order partial derivatives of a scalar function.",
        ),
        (
            "Chain rule",
            "Allows derivatives to propagate through compositions.",
        ),
        (
            "Gradient descent",
            "Uses negative gradients to iteratively reduce an objective.",
        ),
        (
            "Backpropagation",
            "Efficiently applies the chain rule through computational graphs.",
        ),
    ]

    for concept, explanation in concepts:
        print(f"{concept:20s}: {explanation}")


# ---------------------------------------------------------------------------
# 21. Integrated study example
# ---------------------------------------------------------------------------

def integrated_regression_example() -> None:
    heading("21. Integrated Example: Linear Regression")

    # y_hat = wx + b
    # MSE = mean((y_hat - y)^2)
    # dMSE/dw = mean(2(y_hat-y)x)
    # dMSE/db = mean(2(y_hat-y))
    features = [1.0, 2.0, 3.0, 4.0]
    targets = [3.0, 5.0, 7.0, 9.0]

    weight = 0.0
    bias = 0.0
    learning_rate = 0.05

    def predict(x: float) -> float:
        return weight * x + bias

    print("Target relationship is approximately y = 2x + 1.")

    for epoch in range(1, 301):
        predictions = [predict(x) for x in features]

        errors = [
            prediction - target
            for prediction, target in zip(predictions, targets)
        ]

        count = len(features)

        gradient_weight = sum(
            2 * error * x
            for error, x in zip(errors, features)
        ) / count

        gradient_bias = sum(
            2 * error
            for error in errors
        ) / count

        weight -= learning_rate * gradient_weight
        bias -= learning_rate * gradient_bias

        if epoch in {1, 2, 10, 100, 300}:
            loss = mean_squared_error(
                [weight * x + bias for x in features],
                targets,
            )

            print(
                f"epoch={epoch:3d}, "
                f"weight={weight:.6f}, "
                f"bias={bias:.6f}, "
                f"MSE={loss:.8f}"
            )

    print("\nFinal model:")
    print(f"y_hat = {weight:.6f}x + {bias:.6f}")


# ---------------------------------------------------------------------------
# 22. Testing
# ---------------------------------------------------------------------------

def run_tests() -> None:
    heading("22. Built-In Tests")

    # Function test.
    assert approximate_equal((lambda x: x**2)(3), 9)

    # Derivative test.
    derivative_estimate = numerical_derivative(
        lambda x: x**3,
        2.0,
        1e-6,
    )
    assert abs(derivative_estimate - 12.0) < 1e-5

    # Gradient test.
    gradient = numerical_gradient(
        lambda p: p[0] ** 2 + 3 * p[1] ** 2,
        [2.0, 4.0],
    )

    assert abs(gradient[0] - 4.0) < 1e-5
    assert abs(gradient[1] - 24.0) < 1e-5

    # Sigmoid test.
    assert approximate_equal(sigmoid(0), 0.5)

    # Softmax test.
    probabilities = softmax([1.0, 2.0, 3.0])
    assert approximate_equal(sum(probabilities), 1.0, 1e-12)

    # MSE test.
    assert approximate_equal(
        mean_squared_error([1.0, 2.0], [1.0, 4.0]),
        2.0,
    )

    # Logistic model dimension validation.
    model = LogisticRegression(weights=[1.0, 2.0], bias=0.0)

    try:
        model.predict_probability([1.0])
        raise AssertionError("Expected dimension validation to fail")
    except ValueError:
        pass

    print("All built-in tests passed.")


# ---------------------------------------------------------------------------
# 23. Study checklist
# ---------------------------------------------------------------------------

def study_checklist() -> None:
    heading("23. Calculus for ML Study Checklist")

    checklist = [
        "Understand function notation, domains, ranges, and composition.",
        "Understand limits and one-sided limits.",
        "Distinguish continuity from differentiability.",
        "Understand the derivative as a local rate of change.",
        "Apply power, product, quotient, and chain rules.",
        "Understand higher-order derivatives and curvature.",
        "Calculate partial derivatives.",
        "Interpret gradients geometrically.",
        "Understand directional derivatives.",
        "Recognize Jacobians and Hessians.",
        "Understand Taylor approximations.",
        "Understand finite-difference approximations.",
        "Understand why numerical differentiation has step-size trade-offs.",
        "Understand forward-mode automatic differentiation.",
        "Understand loss functions and their gradients.",
        "Derive logistic-regression gradients.",
        "Understand gradient descent.",
        "Understand learning-rate effects.",
        "Understand activation-function derivatives.",
        "Understand softmax and cross-entropy.",
        "Understand why backpropagation is repeated chain rule.",
        "Recognize numerical stability issues.",
    ]

    for index, item in enumerate(checklist, start=1):
        print(f"{index:02d}. {item}")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Execute the complete educational demonstration.

    The order moves from elementary calculus toward machine-learning
    optimization and automatic differentiation.
    """
    random.seed(42)

    demonstrate_functions()
    demonstrate_limits()
    demonstrate_continuity()
    demonstrate_derivatives()
    demonstrate_higher_order_derivatives()
    demonstrate_partial_derivatives()
    demonstrate_gradients()
    demonstrate_jacobian_hessian()
    demonstrate_chain_rule()
    demonstrate_taylor()
    demonstrate_loss_functions()
    train_logistic_regression()
    demonstrate_gradient_descent()
    demonstrate_activations()
    demonstrate_numerical_pitfalls()
    demonstrate_forward_mode_autodiff()
    demonstrate_softmax()
    demonstrate_neural_network()
    demonstrate_optimization_edge_cases()
    demonstrate_ml_connections()
    integrated_regression_example()
    run_tests()
    study_checklist()


if __name__ == "__main__":
    main()
