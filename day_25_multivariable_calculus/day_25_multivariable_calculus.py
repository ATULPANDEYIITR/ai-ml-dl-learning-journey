"""
Multivariable Calculus: Gradients, Jacobians, Hessians, and Directional Derivatives
A self-contained study and demonstration script.

The script develops the topic from first principles through:
- scalar and vector-valued functions
- partial derivatives
- differentiability and linear approximation
- gradients
- directional derivatives
- Jacobians
- Hessians
- second-order Taylor approximation
- optimization and critical points
- chain rule
- numerical differentiation
- finite-difference validation
- edge cases and common pitfalls
- a practical optimization case study

No external packages are required.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence


Number = float
Point = Sequence[Number]
Vector = list[Number]
Matrix = list[list[Number]]


# ---------------------------------------------------------------------------
# Basic numerical utilities
# ---------------------------------------------------------------------------

def add_vectors(a: Point, b: Point) -> Vector:
    """Add two vectors of equal dimension."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension.")
    return [x + y for x, y in zip(a, b)]


def subtract_vectors(a: Point, b: Point) -> Vector:
    """Subtract b from a."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension.")
    return [x - y for x, y in zip(a, b)]


def scale_vector(scalar: Number, vector: Point) -> Vector:
    """Multiply a vector by a scalar."""
    return [scalar * value for value in vector]


def dot(a: Point, b: Point) -> Number:
    """Compute the Euclidean dot product."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension.")
    return sum(x * y for x, y in zip(a, b))


def norm(vector: Point) -> Number:
    """Compute the Euclidean norm."""
    return math.sqrt(dot(vector, vector))


def normalize(vector: Point) -> Vector:
    """Return a unit vector."""
    length = norm(vector)
    if length == 0:
        raise ValueError("The zero vector cannot be normalized.")
    return [value / length for value in vector]


def matrix_vector_multiply(matrix: Matrix, vector: Point) -> Vector:
    """Compute A v."""
    if not matrix:
        return []

    columns = len(matrix[0])
    if columns != len(vector):
        raise ValueError("Matrix and vector dimensions are incompatible.")

    return [dot(row, vector) for row in matrix]


def quadratic_form(vector: Point, matrix: Matrix) -> Number:
    """Compute v^T A v."""
    return dot(vector, matrix_vector_multiply(matrix, vector))


# ---------------------------------------------------------------------------
# Multivariable functions
# ---------------------------------------------------------------------------

def surface(x: Number, y: Number) -> Number:
    """
    Scalar-valued function f: R^2 -> R.

    f(x, y) = x^2 + 3xy + 2y^2

    The output is one scalar, so its first derivative is represented by
    a gradient vector.
    """
    return x * x + 3 * x * y + 2 * y * y


def nonlinear_surface(x: Number, y: Number) -> Number:
    """A nonlinear scalar field used for directional and Hessian examples."""
    return math.sin(x) * math.exp(y) + x * x * y


def vector_function(x: Number, y: Number) -> Vector:
    """
    Vector-valued function F: R^2 -> R^3.

    F(x, y) =
        [x^2 + y,
         xy,
         sin(x) + cos(y)]
    """
    return [
        x * x + y,
        x * y,
        math.sin(x) + math.cos(y),
    ]


# ---------------------------------------------------------------------------
# Partial derivatives
# ---------------------------------------------------------------------------

def partial_x(f: Callable[[float, float], float],
              x: float,
              y: float,
              h: float = 1e-5) -> float:
    """
    Central finite-difference approximation to ∂f/∂x.

    Central differences generally provide better accuracy than a simple
    forward difference for smooth functions.
    """
    return (f(x + h, y) - f(x - h, y)) / (2 * h)


def partial_y(f: Callable[[float, float], float],
              x: float,
              y: float,
              h: float = 1e-5) -> float:
    """Central finite-difference approximation to ∂f/∂y."""
    return (f(x, y + h) - f(x, y - h)) / (2 * h)


def numerical_gradient(f: Callable[[float, float], float],
                       point: Point,
                       h: float = 1e-5) -> Vector:
    """Numerically estimate the gradient of a two-variable scalar function."""
    if len(point) != 2:
        raise ValueError("This demonstration expects a two-dimensional point.")

    x, y = point
    return [
        partial_x(f, x, y, h),
        partial_y(f, x, y, h),
    ]


# ---------------------------------------------------------------------------
# Analytical derivatives for the polynomial surface
# ---------------------------------------------------------------------------

def analytical_gradient(x: Number, y: Number) -> Vector:
    """
    For f(x,y) = x² + 3xy + 2y²,

    ∇f = [2x + 3y, 3x + 4y].
    """
    return [2 * x + 3 * y, 3 * x + 4 * y]


def analytical_hessian(x: Number, y: Number) -> Matrix:
    """
    For f(x,y) = x² + 3xy + 2y²,

             [2  3]
    H(f) =    [3  4].

    The Hessian contains all second-order partial derivatives.
    """
    return [
        [2.0, 3.0],
        [3.0, 4.0],
    ]


# ---------------------------------------------------------------------------
# Directional derivatives
# ---------------------------------------------------------------------------

def directional_derivative_from_gradient(
    gradient: Point,
    direction: Point
) -> Number:
    """
    D_u f = ∇f · u

    The direction must be converted to a unit vector first.
    """
    unit_direction = normalize(direction)
    return dot(gradient, unit_direction)


def numerical_directional_derivative(
    f: Callable[[float, float], float],
    point: Point,
    direction: Point,
    h: float = 1e-5
) -> Number:
    """
    Estimate D_u f using

        [f(p + h u) - f(p - h u)] / (2h).

    This is useful for validating an analytical gradient.
    """
    unit_direction = normalize(direction)

    forward = [
        point[i] + h * unit_direction[i]
        for i in range(len(point))
    ]
    backward = [
        point[i] - h * unit_direction[i]
        for i in range(len(point))
    ]

    if len(point) != 2:
        raise ValueError("This numerical example expects two dimensions.")

    return (
        f(forward[0], forward[1])
        - f(backward[0], backward[1])
    ) / (2 * h)


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------

def analytical_jacobian_vector_function(x: Number, y: Number) -> Matrix:
    """
    For

        F1 = x² + y
        F2 = xy
        F3 = sin(x) + cos(y)

    the Jacobian is

             [2x       1    ]
        J =  [y        x    ]
             [cos(x)  -sin(y)]

    Rows correspond to output components.
    Columns correspond to input variables.
    """
    return [
        [2 * x, 1.0],
        [y, x],
        [math.cos(x), -math.sin(y)],
    ]


def numerical_jacobian(
    function: Callable[[float, float], Vector],
    point: Point,
    h: float = 1e-5
) -> Matrix:
    """Numerically estimate a Jacobian using central differences."""
    if len(point) != 2:
        raise ValueError("This demonstration expects two input variables.")

    x, y = point
    base = function(x, y)
    output_dimension = len(base)

    plus_x = function(x + h, y)
    minus_x = function(x - h, y)
    plus_y = function(x, y + h)
    minus_y = function(x, y - h)

    jacobian = []

    for i in range(output_dimension):
        derivative_x = (plus_x[i] - minus_x[i]) / (2 * h)
        derivative_y = (plus_y[i] - minus_y[i]) / (2 * h)
        jacobian.append([derivative_x, derivative_y])

    return jacobian


# ---------------------------------------------------------------------------
# Hessian for a general two-variable scalar function
# ---------------------------------------------------------------------------

def numerical_hessian(
    f: Callable[[float, float], float],
    point: Point,
    h: float = 1e-4
) -> Matrix:
    """
    Estimate the Hessian of f(x,y).

    Diagonal terms:
        f_xx and f_yy

    Mixed term:
        f_xy

    For a sufficiently smooth function, f_xy = f_yx.
    """
    if len(point) != 2:
        raise ValueError("This demonstration expects two dimensions.")

    x, y = point

    f_xx = (
        f(x + h, y)
        - 2 * f(x, y)
        + f(x - h, y)
    ) / (h * h)

    f_yy = (
        f(x, y + h)
        - 2 * f(x, y)
        + f(x, y - h)
    ) / (h * h)

    f_xy = (
        f(x + h, y + h)
        - f(x + h, y - h)
        - f(x - h, y + h)
        + f(x - h, y - h)
    ) / (4 * h * h)

    return [
        [f_xx, f_xy],
        [f_xy, f_yy],
    ]


# ---------------------------------------------------------------------------
# First-order and second-order Taylor approximations
# ---------------------------------------------------------------------------

def first_order_taylor(
    f_value: Number,
    gradient: Point,
    displacement: Point
) -> Number:
    """
    First-order Taylor approximation:

        f(p + Δp) ≈ f(p) + ∇f(p)^T Δp
    """
    return f_value + dot(gradient, displacement)


def second_order_taylor(
    f_value: Number,
    gradient: Point,
    hessian: Matrix,
    displacement: Point
) -> Number:
    """
    Second-order Taylor approximation:

        f(p + Δp)
        ≈ f(p)
          + ∇f(p)^T Δp
          + 1/2 Δp^T H(p) Δp
    """
    linear_part = dot(gradient, displacement)
    quadratic_part = 0.5 * quadratic_form(displacement, hessian)
    return f_value + linear_part + quadratic_part


# ---------------------------------------------------------------------------
# Critical points and classification
# ---------------------------------------------------------------------------

def determinant_2x2(matrix: Matrix) -> Number:
    """Determinant of a 2x2 matrix."""
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("Expected a 2x2 matrix.")
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def classify_critical_point_2d(hessian: Matrix) -> str:
    """
    Use the two-dimensional second-derivative test.

    For H = [[fxx, fxy], [fyx, fyy]], define
        D = fxx*fyy - fxy².

    D > 0 and fxx > 0 -> local minimum
    D > 0 and fxx < 0 -> local maximum
    D < 0              -> saddle point
    D = 0              -> test is inconclusive
    """
    determinant = determinant_2x2(hessian)
    f_xx = hessian[0][0]

    if determinant > 0 and f_xx > 0:
        return "local minimum"
    if determinant > 0 and f_xx < 0:
        return "local maximum"
    if determinant < 0:
        return "saddle point"
    return "inconclusive"


# ---------------------------------------------------------------------------
# Gradient descent
# ---------------------------------------------------------------------------

def gradient_descent(
    f: Callable[[float, float], float],
    gradient_function: Callable[[float, float], Vector],
    initial_point: Point,
    learning_rate: float = 0.05,
    iterations: int = 100
) -> tuple[Vector, list[float]]:
    """
    Basic gradient descent.

    x_(k+1) = x_k - α ∇f(x_k)

    The method minimizes a differentiable scalar objective when the
    step size and objective structure permit convergence.
    """
    point = list(initial_point)
    history = [f(point[0], point[1])]

    for _ in range(iterations):
        gradient = gradient_function(point[0], point[1])
        point = [
            point[i] - learning_rate * gradient[i]
            for i in range(2)
        ]
        history.append(f(point[0], point[1]))

    return point, history


# ---------------------------------------------------------------------------
# Chain rule example
# ---------------------------------------------------------------------------

def composed_function(t: Number) -> Number:
    """
    Let

        x(t) = t²
        y(t) = sin(t)

    and

        f(x,y) = x² + 3xy + 2y².

    Then

        d/dt f(x(t), y(t))
        = ∇f(x(t),y(t)) · [x'(t), y'(t)].
    """
    x = t * t
    y = math.sin(t)
    return surface(x, y)


def chain_rule_derivative(t: Number) -> Number:
    """Analytical chain-rule derivative."""
    x = t * t
    y = math.sin(t)

    gradient = analytical_gradient(x, y)
    velocity = [2 * t, math.cos(t)]

    return dot(gradient, velocity)


def numerical_one_variable_derivative(
    f: Callable[[float], float],
    x: float,
    h: float = 1e-5
) -> float:
    """Central finite difference for a one-variable function."""
    return (f(x + h) - f(x - h)) / (2 * h)


# ---------------------------------------------------------------------------
# General conceptual demonstrations
# ---------------------------------------------------------------------------

def demonstrate_scalar_vs_vector() -> None:
    print("\n--- Scalar-valued versus vector-valued functions ---")

    p = (2.0, 1.0)

    scalar_value = surface(*p)
    vector_value = vector_function(*p)

    print(f"Point p = {p}")
    print(f"f(p) = {scalar_value}")
    print(f"F(p) = {vector_value}")
    print("A scalar field has one output; its derivative is represented by a gradient.")
    print("A vector-valued function has multiple outputs; its first derivative is a Jacobian.")


def demonstrate_gradient() -> None:
    print("\n--- Gradient ---")

    point = (2.0, 1.0)
    gradient = analytical_gradient(*point)
    numerical = numerical_gradient(surface, point)

    print(f"Point: {point}")
    print(f"Analytical gradient: {gradient}")
    print(f"Numerical gradient:  {[round(v, 8) for v in numerical]}")
    print("The gradient points in the direction of greatest instantaneous increase.")


def demonstrate_directional_derivatives() -> None:
    print("\n--- Directional derivatives ---")

    point = (2.0, 1.0)
    gradient = analytical_gradient(*point)

    directions = [
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (-1.0, -1.0),
    ]

    for direction in directions:
        derivative = directional_derivative_from_gradient(
            gradient,
            direction
        )
        numerical = numerical_directional_derivative(
            surface,
            point,
            direction
        )

        print(
            f"direction={direction}, "
            f"analytical={derivative:.8f}, "
            f"numerical={numerical:.8f}"
        )

    print(
        "Important: a directional derivative D_v f normally uses a unit "
        "direction u = v / ||v||. Without normalization, the result scales "
        "with the length of the supplied vector."
    )


def demonstrate_jacobian() -> None:
    print("\n--- Jacobian ---")

    point = (1.5, 0.75)

    analytical = analytical_jacobian_vector_function(*point)
    numerical = numerical_jacobian(vector_function, point)

    print("Analytical Jacobian:")
    for row in analytical:
        print("  ", [round(value, 8) for value in row])

    print("Numerical Jacobian:")
    for row in numerical:
        print("  ", [round(value, 8) for value in row])

    print(
        "The Jacobian maps a small input displacement Δx to an "
        "approximately corresponding output displacement JΔx."
    )


def demonstrate_hessian() -> None:
    print("\n--- Hessian ---")

    point = (2.0, 1.0)

    analytical = analytical_hessian(*point)
    numerical = numerical_hessian(surface, point)

    print("Analytical Hessian:")
    for row in analytical:
        print("  ", row)

    print("Numerical Hessian:")
    for row in numerical:
        print("  ", [round(value, 8) for value in row])

    classification = classify_critical_point_2d(analytical)
    print(f"Classification test for this Hessian: {classification}")

    print(
        "For this function the Hessian is constant because the function "
        "is quadratic."
    )


def demonstrate_taylor_approximation() -> None:
    print("\n--- Taylor approximation ---")

    base_point = (1.0, 1.0)
    displacement = (0.05, -0.03)
    target = add_vectors(base_point, displacement)

    base_value = nonlinear_surface(*base_point)
    gradient = numerical_gradient(nonlinear_surface, base_point)
    hessian = numerical_hessian(nonlinear_surface, base_point)

    exact = nonlinear_surface(*target)
    first_order = first_order_taylor(
        base_value,
        gradient,
        displacement
    )
    second_order = second_order_taylor(
        base_value,
        gradient,
        hessian,
        displacement
    )

    print(f"Base point: {base_point}")
    print(f"Target point: {target}")
    print(f"Exact value:             {exact:.10f}")
    print(f"First-order estimate:    {first_order:.10f}")
    print(f"Second-order estimate:   {second_order:.10f}")
    print(
        "The Hessian adds curvature information to the linear approximation."
    )


def demonstrate_chain_rule() -> None:
    print("\n--- Multivariable chain rule ---")

    t = 0.8

    analytical = chain_rule_derivative(t)
    numerical = numerical_one_variable_derivative(
        composed_function,
        t
    )

    print(f"t = {t}")
    print(f"Chain-rule derivative: {analytical:.10f}")
    print(f"Numerical derivative:  {numerical:.10f}")

    print(
        "The chain rule combines the gradient of the outer function with "
        "the derivative of the inner parameterized path."
    )


def demonstrate_optimization() -> None:
    print("\n--- Gradient descent optimization ---")

    # The polynomial surface has a stationary point at the origin because
    # its gradient vanishes there:
    #
    #   2x + 3y = 0
    #   3x + 4y = 0
    #
    # Its Hessian has determinant 8 - 9 = -1, so the origin is a saddle.
    #
    # Therefore gradient descent is not a suitable minimization demonstration
    # for this particular surface.

    def bowl(x: float, y: float) -> float:
        return 4 * x * x + 2 * y * y + x * y

    def bowl_gradient(x: float, y: float) -> Vector:
        return [
            8 * x + y,
            4 * y + x,
        ]

    start = (5.0, -4.0)

    final_point, history = gradient_descent(
        bowl,
        bowl_gradient,
        start,
        learning_rate=0.08,
        iterations=100
    )

    print(f"Starting point: {start}")
    print(
        f"Final point:    "
        f"({final_point[0]:.10f}, {final_point[1]:.10f})"
    )
    print(f"Initial objective: {history[0]:.10f}")
    print(f"Final objective:   {history[-1]:.10f}")

    print(
        "Gradient descent moves opposite the gradient because the gradient "
        "points toward locally increasing values."
    )


def demonstrate_edge_cases() -> None:
    print("\n--- Edge cases and important distinctions ---")

    try:
        normalize((0.0, 0.0))
    except ValueError as error:
        print(f"Zero direction error handled correctly: {error}")

    try:
        dot((1.0, 2.0), (1.0,))
    except ValueError as error:
        print(f"Dimension mismatch handled correctly: {error}")

    print(
        "Directional derivative distinction:\n"
        "  D_v f depends on how v is defined. If v is intended as a direction, "
        "normalizing it produces D_u f for a unit direction u.\n"
        "  If a non-unit vector is deliberately used as a velocity, "
        "∇f · v measures the rate associated with that velocity."
    )

    print(
        "Hessian distinction:\n"
        "  A Hessian is associated with a scalar-valued function. For a "
        "vector-valued function, each output component has its own Hessian."
    )

    print(
        "Jacobian distinction:\n"
        "  For F: R^n -> R^m, the Jacobian has m rows and n columns."
    )


# ---------------------------------------------------------------------------
# Self-checking tests
# ---------------------------------------------------------------------------

def approximately_equal(a: float, b: float, tolerance: float = 1e-6) -> bool:
    return abs(a - b) <= tolerance


def run_tests() -> None:
    print("\n--- Verification tests ---")

    gradient = analytical_gradient(2.0, 1.0)
    assert gradient == [7.0, 10.0]

    hessian = analytical_hessian(2.0, 1.0)
    assert hessian == [[2.0, 3.0], [3.0, 4.0]]

    assert approximately_equal(
        directional_derivative_from_gradient(
            [1.0, 0.0],
            (1.0, 0.0)
        ),
        1.0
    )

    jacobian = analytical_jacobian_vector_function(1.0, 2.0)
    assert approximately_equal(jacobian[0][0], 2.0)
    assert approximately_equal(jacobian[0][1], 1.0)
    assert approximately_equal(jacobian[1][0], 2.0)
    assert approximately_equal(jacobian[1][1], 1.0)

    assert classify_critical_point_2d(
        [[2.0, 0.0], [0.0, 2.0]]
    ) == "local minimum"

    assert classify_critical_point_2d(
        [[-2.0, 0.0], [0.0, -2.0]]
    ) == "local maximum"

    assert classify_critical_point_2d(
        [[2.0, 0.0], [0.0, -2.0]]
    ) == "saddle point"

    print("All verification tests passed.")


# ---------------------------------------------------------------------------
# Main educational program
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("MULTIVARIABLE CALCULUS")
    print("Gradients, Jacobians, Hessians, and Directional Derivatives")
    print("=" * 78)

    print(
        "\nCore mathematical map:\n"
        "  Scalar function:       f: R^n -> R\n"
        "  Gradient:              ∇f in R^n\n"
        "  Vector function:       F: R^n -> R^m\n"
        "  Jacobian:              J_F is an m x n matrix\n"
        "  Hessian of scalar f:   H_f is an n x n matrix\n"
        "  Directional derivative D_u f = ∇f · u for ||u|| = 1"
    )

    demonstrate_scalar_vs_vector()
    demonstrate_gradient()
    demonstrate_directional_derivatives()
    demonstrate_jacobian()
    demonstrate_hessian()
    demonstrate_taylor_approximation()
    demonstrate_chain_rule()
    demonstrate_optimization()
    demonstrate_edge_cases()
    run_tests()

    print("\n" + "=" * 78)
    print("KEY FORMULAS")
    print("=" * 78)

    print(
        "\nGradient:\n"
        "  ∇f(x) = [∂f/∂x₁, ..., ∂f/∂xₙ]^T"
    )

    print(
        "\nDirectional derivative:\n"
        "  D_u f(x) = ∇f(x) · u,   ||u|| = 1"
    )

    print(
        "\nJacobian:\n"
        "  J_ij = ∂F_i / ∂x_j"
    )

    print(
        "\nHessian:\n"
        "  H_ij = ∂²f / (∂x_i ∂x_j)"
    )

    print(
        "\nFirst-order approximation:\n"
        "  f(x + Δx) ≈ f(x) + ∇f(x)^T Δx"
    )

    print(
        "\nSecond-order approximation:\n"
        "  f(x + Δx) ≈ f(x) + ∇f(x)^T Δx "
        "+ 1/2 Δx^T H(x) Δx"
    )

    print(
        "\nChain rule for F: R^n -> R^m and x(t):\n"
        "  dF/dt = J_F(x(t)) x'(t)"
    )

    print(
        "\nThe central conceptual distinction is:\n"
        "  gradient = first derivative of a scalar field,\n"
        "  Jacobian = first derivative of a vector-valued mapping,\n"
        "  Hessian = second derivative of a scalar field,\n"
        "  directional derivative = gradient projected onto a direction."
    )


if __name__ == "__main__":
    main()
