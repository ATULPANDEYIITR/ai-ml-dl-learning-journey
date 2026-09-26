"""
OPTIMIZATION MATHEMATICS
========================

A self-contained study and executable demonstration of optimization mathematics,
covering:

1. Optimization terminology and mathematical formulation
2. Objective functions and decision variables
3. Local and global minima/maxima
4. Derivatives and stationary points
5. First- and second-order conditions
6. Convex and strictly convex functions
7. Concavity and maximization
8. Gradient descent
9. Newton's method
10. Step-size selection
11. Numerical convergence and stopping criteria
12. Multivariable optimization
13. Hessians and curvature
14. Quadratic optimization
15. Constrained optimization
16. Equality constraints
17. Inequality constraints
18. Penalty methods
19. Lagrange multipliers
20. KKT conditions
21. Projected gradient descent
22. Linear programming concepts
23. Integer and combinatorial optimization concepts
24. Regularized optimization
25. Logistic regression as an optimization problem
26. Least-squares optimization
27. Numerical stability
28. Common failure modes
29. Complexity and practical implementation considerations

The program uses only the Python standard library.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


Number = float
Vector = List[float]


# ---------------------------------------------------------------------------
# 1. FUNDAMENTAL MATHEMATICAL CONCEPTS
# ---------------------------------------------------------------------------

def square(x: float) -> float:
    """A simple objective function with a unique global minimum at x = 3."""
    return (x - 3.0) ** 2 + 2.0


def double_well(x: float) -> float:
    """
    A non-convex function.

    f(x) = x^4 - 4x^2 + 4

    The function has multiple stationary points and illustrates why local
    information does not always identify the global solution.
    """
    return x ** 4 - 4.0 * x ** 2 + 4.0


def derivative_central(
    function: Callable[[float], float],
    x: float,
    step: float = 1e-6,
) -> float:
    """Numerically estimate a first derivative using a central difference."""
    if step <= 0:
        raise ValueError("step must be positive")

    return (function(x + step) - function(x - step)) / (2.0 * step)


def second_derivative_central(
    function: Callable[[float], float],
    x: float,
    step: float = 1e-4,
) -> float:
    """Numerically estimate a second derivative using a central difference."""
    if step <= 0:
        raise ValueError("step must be positive")

    return (
        function(x + step)
        - 2.0 * function(x)
        + function(x - step)
    ) / (step ** 2)


# ---------------------------------------------------------------------------
# 2. VECTOR AND MATRIX UTILITIES
# ---------------------------------------------------------------------------

def vector_add(a: Sequence[float], b: Sequence[float]) -> Vector:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal dimensions")
    return [x + y for x, y in zip(a, b)]


def vector_subtract(a: Sequence[float], b: Sequence[float]) -> Vector:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal dimensions")
    return [x - y for x, y in zip(a, b)]


def scalar_multiply(scalar: float, vector: Sequence[float]) -> Vector:
    return [scalar * value for value in vector]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal dimensions")
    return sum(x * y for x, y in zip(a, b))


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(dot(vector, vector))


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return norm(vector_subtract(a, b))


def matrix_vector_multiply(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> Vector:
    if not matrix:
        return []

    columns = len(matrix[0])

    if any(len(row) != columns for row in matrix):
        raise ValueError("Matrix rows must have equal length")

    if columns != len(vector):
        raise ValueError("Matrix and vector dimensions are incompatible")

    return [dot(row, vector) for row in matrix]


def matrix_multiply(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
) -> List[Vector]:
    if not a or not b:
        return []

    a_columns = len(a[0])
    b_columns = len(b[0])
    b_rows = len(b)

    if any(len(row) != a_columns for row in a):
        raise ValueError("Invalid first matrix")

    if any(len(row) != b_columns for row in b):
        raise ValueError("Invalid second matrix")

    if a_columns != b_rows:
        raise ValueError("Matrix dimensions are incompatible")

    return [
        [
            sum(a[i][k] * b[k][j] for k in range(b_rows))
            for j in range(b_columns)
        ]
        for i in range(len(a))
    ]


def identity_matrix(size: int) -> List[Vector]:
    return [
        [1.0 if i == j else 0.0 for j in range(size)]
        for i in range(size)
    ]


def solve_linear_system(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
    tolerance: float = 1e-12,
) -> Vector:
    """
    Solve Ax=b using Gaussian elimination with partial pivoting.

    Partial pivoting reduces numerical error when a pivot is small.
    """
    n = len(matrix)

    if n == 0 or len(vector) != n:
        raise ValueError("Invalid linear system dimensions")

    augmented = [
        list(map(float, matrix[i])) + [float(vector[i])]
        for i in range(n)
    ]

    if any(len(row) != n + 1 for row in augmented):
        raise ValueError("Matrix must be square")

    for column in range(n):
        pivot_row = max(
            range(column, n),
            key=lambda row: abs(augmented[row][column]),
        )

        pivot = augmented[pivot_row][column]

        if abs(pivot) <= tolerance:
            raise ValueError("Linear system is singular or ill-conditioned")

        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )

        for row in range(column + 1, n):
            factor = augmented[row][column] / pivot

            for j in range(column, n + 1):
                augmented[row][j] -= factor * augmented[column][j]

    solution = [0.0] * n

    for row in range(n - 1, -1, -1):
        remaining = sum(
            augmented[row][j] * solution[j]
            for j in range(row + 1, n)
        )
        solution[row] = (
            augmented[row][n] - remaining
        ) / augmented[row][row]

    return solution


# ---------------------------------------------------------------------------
# 3. GRADIENT APPROXIMATION
# ---------------------------------------------------------------------------

def numerical_gradient(
    function: Callable[[Sequence[float]], float],
    point: Sequence[float],
    step: float = 1e-6,
) -> Vector:
    """
    Compute a numerical gradient using central differences.

    The gradient contains the partial derivative with respect to each
    decision variable.
    """
    if step <= 0:
        raise ValueError("step must be positive")

    gradient = []

    for index in range(len(point)):
        plus = list(point)
        minus = list(point)

        plus[index] += step
        minus[index] -= step

        partial = (
            function(plus) - function(minus)
        ) / (2.0 * step)

        gradient.append(partial)

    return gradient


def numerical_hessian(
    function: Callable[[Sequence[float]], float],
    point: Sequence[float],
    step: float = 1e-4,
) -> List[Vector]:
    """
    Approximate a Hessian matrix.

    The Hessian contains second-order partial derivatives and describes
    local curvature.
    """
    dimension = len(point)
    hessian = [[0.0] * dimension for _ in range(dimension)]

    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                plus = list(point)
                minus = list(point)

                plus[i] += step
                minus[i] -= step

                hessian[i][j] = (
                    function(plus)
                    - 2.0 * function(point)
                    + function(minus)
                ) / (step ** 2)
            else:
                pp = list(point)
                pm = list(point)
                mp = list(point)
                mm = list(point)

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
                ) / (4.0 * step ** 2)

    return hessian


# ---------------------------------------------------------------------------
# 4. ONE-DIMENSIONAL OPTIMIZATION
# ---------------------------------------------------------------------------

def classify_stationary_point(
    function: Callable[[float], float],
    point: float,
) -> str:
    """Classify a stationary point using the second derivative test."""
    second = second_derivative_central(function, point)

    tolerance = 1e-5

    if second > tolerance:
        return "local minimum"
    if second < -tolerance:
        return "local maximum"
    return "inconclusive or flat/saddle-like behavior"


def gradient_descent_1d(
    function: Callable[[float], float],
    start: float,
    learning_rate: float = 0.1,
    iterations: int = 100,
    tolerance: float = 1e-8,
) -> Tuple[float, List[Tuple[int, float, float, float]]]:
    """
    Gradient descent for a one-dimensional differentiable function.

    x_{k+1} = x_k - alpha * f'(x_k)
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if iterations <= 0:
        raise ValueError("iterations must be positive")

    x = float(start)
    history = []

    for iteration in range(iterations):
        gradient = derivative_central(function, x)
        value = function(x)

        history.append((iteration, x, value, gradient))

        if abs(gradient) <= tolerance:
            break

        new_x = x - learning_rate * gradient

        if abs(new_x - x) <= tolerance:
            x = new_x
            break

        x = new_x

    return x, history


def newton_1d(
    function: Callable[[float], float],
    start: float,
    iterations: int = 50,
    tolerance: float = 1e-8,
) -> Tuple[float, List[Tuple[int, float, float, float, float]]]:
    """
    Newton optimization uses:

        x_new = x - f'(x) / f''(x)

    Newton's method can converge very rapidly near a well-behaved optimum,
    but it can be unstable when the second derivative is small or has an
    unsuitable sign.
    """
    x = float(start)
    history = []

    for iteration in range(iterations):
        first = derivative_central(function, x)
        second = second_derivative_central(function, x)

        history.append(
            (iteration, x, function(x), first, second)
        )

        if abs(first) <= tolerance:
            break

        if abs(second) <= 1e-12:
            break

        x = x - first / second

    return x, history


# ---------------------------------------------------------------------------
# 5. CONVEXITY
# ---------------------------------------------------------------------------

def is_convex_1d(
    function: Callable[[float], float],
    start: float,
    end: float,
    samples: int = 100,
    tolerance: float = 1e-6,
) -> bool:
    """
    Numerical convexity check for a one-dimensional function.

    For a twice differentiable function, f''(x) >= 0 throughout the domain
    is a sufficient local characterization of convexity on an interval.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")

    for i in range(samples + 1):
        x = start + (end - start) * i / samples

        if second_derivative_central(function, x) < -tolerance:
            return False

    return True


def is_strictly_convex_1d(
    function: Callable[[float], float],
    start: float,
    end: float,
    samples: int = 100,
    tolerance: float = 1e-6,
) -> bool:
    """Numerical strict-convexity check based on positive second derivative."""
    for i in range(samples + 1):
        x = start + (end - start) * i / samples

        if second_derivative_central(function, x) <= tolerance:
            return False

    return True


# ---------------------------------------------------------------------------
# 6. MULTIVARIABLE QUADRATIC OPTIMIZATION
# ---------------------------------------------------------------------------

@dataclass
class QuadraticFunction:
    """
    Represents:

        f(x) = 1/2 x^T Q x + c^T x + constant

    If Q is symmetric positive definite, the function is strictly convex and
    has one unique global minimizer.
    """

    q: List[Vector]
    c: Vector
    constant: float = 0.0

    def __post_init__(self) -> None:
        n = len(self.q)

        if n == 0:
            raise ValueError("Q cannot be empty")

        if len(self.c) != n:
            raise ValueError("c dimension must match Q")

        if any(len(row) != n for row in self.q):
            raise ValueError("Q must be square")

    def __call__(self, x: Sequence[float]) -> float:
        if len(x) != len(self.c):
            raise ValueError("x has the wrong dimension")

        qx = matrix_vector_multiply(self.q, x)

        return (
            0.5 * dot(x, qx)
            + dot(self.c, x)
            + self.constant
        )

    def gradient(self, x: Sequence[float]) -> Vector:
        qx = matrix_vector_multiply(self.q, x)
        return vector_add(qx, self.c)

    def hessian(self) -> List[Vector]:
        return [row[:] for row in self.q]


def gradient_descent(
    function: Callable[[Sequence[float]], float],
    gradient: Callable[[Sequence[float]], Sequence[float]],
    start: Sequence[float],
    learning_rate: float = 0.05,
    iterations: int = 1000,
    tolerance: float = 1e-8,
) -> Tuple[Vector, List[dict]]:
    """
    General multivariable gradient descent.

    Stopping criteria use both gradient magnitude and movement in x.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    x = list(map(float, start))
    history = []

    for iteration in range(iterations):
        grad = list(gradient(x))
        gradient_norm = norm(grad)

        history.append(
            {
                "iteration": iteration,
                "x": x[:],
                "value": function(x),
                "gradient_norm": gradient_norm,
            }
        )

        if gradient_norm <= tolerance:
            break

        new_x = vector_subtract(
            x,
            scalar_multiply(learning_rate, grad),
        )

        if distance(new_x, x) <= tolerance:
            x = new_x
            break

        x = new_x

    return x, history


def quadratic_exact_minimizer(
    quadratic: QuadraticFunction,
) -> Vector:
    """
    For a differentiable quadratic:

        grad f(x) = Qx + c

    The stationary point satisfies:

        Qx = -c
    """
    negative_c = [-value for value in quadratic.c]

    return solve_linear_system(
        quadratic.q,
        negative_c,
    )


# ---------------------------------------------------------------------------
# 7. HESSIAN-BASED CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_multivariable_point(
    hessian: Sequence[Sequence[float]],
    tolerance: float = 1e-10,
) -> str:
    """
    A small-dimensional Hessian classifier.

    For this educational implementation, eigenvalues are estimated using
    a Jacobi rotation method for symmetric matrices.
    """
    eigenvalues = jacobi_eigenvalues(hessian)

    if all(value > tolerance for value in eigenvalues):
        return "strict local minimum"

    if all(value < -tolerance for value in eigenvalues):
        return "strict local maximum"

    if any(value > tolerance for value in eigenvalues) and any(
        value < -tolerance for value in eigenvalues
    ):
        return "saddle point"

    return "inconclusive or degenerate"


def jacobi_eigenvalues(
    matrix: Sequence[Sequence[float]],
    tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> Vector:
    """
    Compute eigenvalues of a real symmetric matrix using Jacobi rotations.

    This is included for educational purposes. Production numerical software
    normally uses optimized, thoroughly tested linear algebra libraries.
    """
    n = len(matrix)

    if n == 0:
        return []

    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square")

    a = [list(map(float, row)) for row in matrix]

    for _ in range(max_iterations):
        p = 0
        q = 1 if n > 1 else 0
        largest = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > largest:
                    largest = abs(a[i][j])
                    p, q = i, j

        if largest <= tolerance:
            break

        if abs(a[p][p] - a[q][q]) <= tolerance:
            angle = math.pi / 4.0
        else:
            angle = 0.5 * math.atan2(
                2.0 * a[p][q],
                a[p][p] - a[q][q],
            )

        cosine = math.cos(angle)
        sine = math.sin(angle)

        for i in range(n):
            if i != p and i != q:
                aip = a[i][p]
                aiq = a[i][q]

                a[i][p] = cosine * aip + sine * aiq
                a[p][i] = a[i][p]

                a[i][q] = -sine * aip + cosine * aiq
                a[q][i] = a[i][q]

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]

        a[p][p] = (
            cosine ** 2 * app
            + 2.0 * sine * cosine * apq
            + sine ** 2 * aqq
        )

        a[q][q] = (
            sine ** 2 * app
            - 2.0 * sine * cosine * apq
            + cosine ** 2 * aqq
        )

        a[p][q] = 0.0
        a[q][p] = 0.0

    return [a[i][i] for i in range(n)]


# ---------------------------------------------------------------------------
# 8. LINE SEARCH
# ---------------------------------------------------------------------------

def backtracking_line_search(
    function: Callable[[Sequence[float]], float],
    gradient: Callable[[Sequence[float]], Sequence[float]],
    x: Sequence[float],
    direction: Sequence[float],
    initial_step: float = 1.0,
    reduction: float = 0.5,
    armijo_constant: float = 1e-4,
    minimum_step: float = 1e-12,
) -> float:
    """
    Armijo backtracking line search.

    It reduces the step until the candidate provides sufficient decrease:

        f(x + alpha d)
        <=
        f(x) + c alpha grad(f(x))^T d
    """
    if not 0.0 < reduction < 1.0:
        raise ValueError("reduction must be between 0 and 1")

    if not 0.0 < armijo_constant < 1.0:
        raise ValueError("armijo_constant must be between 0 and 1")

    alpha = initial_step
    current_value = function(x)
    current_gradient = gradient(x)
    directional_derivative = dot(current_gradient, direction)

    while alpha >= minimum_step:
        candidate = vector_add(
            x,
            scalar_multiply(alpha, direction),
        )

        if function(candidate) <= (
            current_value
            + armijo_constant * alpha * directional_derivative
        ):
            return alpha

        alpha *= reduction

    return minimum_step


def gradient_descent_with_line_search(
    function: Callable[[Sequence[float]], float],
    gradient: Callable[[Sequence[float]], Sequence[float]],
    start: Sequence[float],
    iterations: int = 500,
    tolerance: float = 1e-8,
) -> Tuple[Vector, List[dict]]:
    """Gradient descent whose step size is selected adaptively."""
    x = list(map(float, start))
    history = []

    for iteration in range(iterations):
        grad = list(gradient(x))
        gradient_norm = norm(grad)

        history.append(
            {
                "iteration": iteration,
                "x": x[:],
                "value": function(x),
                "gradient_norm": gradient_norm,
            }
        )

        if gradient_norm <= tolerance:
            break

        direction = scalar_multiply(-1.0, grad)

        step = backtracking_line_search(
            function,
            gradient,
            x,
            direction,
        )

        new_x = vector_add(
            x,
            scalar_multiply(step, direction),
        )

        if distance(new_x, x) <= tolerance:
            x = new_x
            break

        x = new_x

    return x, history


# ---------------------------------------------------------------------------
# 9. PROJECTED GRADIENT DESCENT
# ---------------------------------------------------------------------------

def project_to_box(
    point: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> Vector:
    """Project each coordinate into [lower_i, upper_i]."""
    if not (
        len(point) == len(lower) == len(upper)
    ):
        raise ValueError("All vectors must have equal dimensions")

    projected = []

    for value, low, high in zip(point, lower, upper):
        if low > high:
            raise ValueError("lower bound cannot exceed upper bound")

        projected.append(
            max(low, min(value, high))
        )

    return projected


def projected_gradient_descent(
    function: Callable[[Sequence[float]], float],
    gradient: Callable[[Sequence[float]], Sequence[float]],
    start: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    learning_rate: float = 0.05,
    iterations: int = 1000,
    tolerance: float = 1e-8,
) -> Tuple[Vector, List[dict]]:
    """
    Solve box-constrained optimization:

        minimize f(x)
        subject to lower_i <= x_i <= upper_i

    Each gradient step is followed by projection onto the feasible set.
    """
    x = project_to_box(start, lower, upper)
    history = []

    for iteration in range(iterations):
        grad = list(gradient(x))

        history.append(
            {
                "iteration": iteration,
                "x": x[:],
                "value": function(x),
                "gradient_norm": norm(grad),
            }
        )

        candidate = vector_subtract(
            x,
            scalar_multiply(learning_rate, grad),
        )

        projected = project_to_box(
            candidate,
            lower,
            upper,
        )

        if distance(projected, x) <= tolerance:
            x = projected
            break

        x = projected

    return x, history


# ---------------------------------------------------------------------------
# 10. LAGRANGE MULTIPLIERS
# ---------------------------------------------------------------------------

def lagrange_equality_example() -> Tuple[Vector, float]:
    """
    Solve:

        minimize x^2 + y^2
        subject to x + y = 10

    L(x,y,lambda) = x^2 + y^2 + lambda(x+y-10)

    Stationarity:
        2x + lambda = 0
        2y + lambda = 0

    Therefore x = y = 5.
    """
    # The analytical solution is included explicitly because it demonstrates
    # the mathematical structure of the method without requiring a symbolic
    # algebra package.
    x = 5.0
    y = 5.0
    multiplier = -10.0

    return [x, y], multiplier


# ---------------------------------------------------------------------------
# 11. PENALTY METHOD
# ---------------------------------------------------------------------------

def equality_penalty_objective(
    point: Sequence[float],
    penalty_weight: float,
) -> float:
    """
    Approximate:

        minimize x^2 + y^2
        subject to x + y = 10

    by minimizing:

        x^2 + y^2 + rho(x+y-10)^2
    """
    x, y = point
    constraint_violation = x + y - 10.0

    return (
        x * x
        + y * y
        + penalty_weight * constraint_violation ** 2
    )


def equality_penalty_gradient(
    point: Sequence[float],
    penalty_weight: float,
) -> Vector:
    x, y = point
    violation = x + y - 10.0

    return [
        2.0 * x + 2.0 * penalty_weight * violation,
        2.0 * y + 2.0 * penalty_weight * violation,
    ]


# ---------------------------------------------------------------------------
# 12. KKT CONDITIONS
# ---------------------------------------------------------------------------

@dataclass
class KKTExample:
    """
    Analytical example:

        minimize x
        subject to x >= 3

    Write inequality as:

        g(x) = 3 - x <= 0

    Lagrangian:

        L = x + lambda(3-x)

    KKT stationarity:
        1 - lambda = 0

    Thus lambda = 1, x = 3.
    """

    optimum: float = 3.0
    multiplier: float = 1.0

    def verify(self) -> bool:
        stationarity = abs(1.0 - self.multiplier) <= 1e-12
        primal_feasibility = self.optimum >= 3.0
        dual_feasibility = self.multiplier >= 0.0
        complementary_slackness = abs(
            self.multiplier * (3.0 - self.optimum)
        ) <= 1e-12

        return (
            stationarity
            and primal_feasibility
            and dual_feasibility
            and complementary_slackness
        )


# ---------------------------------------------------------------------------
# 13. LEAST-SQUARES OPTIMIZATION
# ---------------------------------------------------------------------------

def least_squares_fit(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> Tuple[float, float]:
    """
    Fit:

        y = beta_0 + beta_1 x

    by minimizing the sum of squared residuals.

    The normal equations are:

        (X^T X) beta = X^T y
    """
    if len(x_values) != len(y_values):
        raise ValueError("x and y must have equal lengths")

    if len(x_values) < 2:
        raise ValueError("At least two observations are required")

    design_matrix = [
        [1.0, x]
        for x in x_values
    ]

    transpose = [
        [row[column] for row in design_matrix]
        for column in range(2)
    ]

    xtx = matrix_multiply(transpose, design_matrix)

    xty = [
        dot(transpose[row], y_values)
        for row in range(2)
    ]

    beta = solve_linear_system(xtx, xty)

    return beta[0], beta[1]


def mean_squared_error(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> float:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must match")

    if not predictions:
        raise ValueError("At least one observation is required")

    return sum(
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets)
    ) / len(predictions)


# ---------------------------------------------------------------------------
# 14. LOGISTIC REGRESSION AS OPTIMIZATION
# ---------------------------------------------------------------------------

def sigmoid(value: float) -> float:
    """
    Numerically stable sigmoid implementation.

    Direct exp(-value) can overflow for sufficiently negative values.
    """
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)

    z = math.exp(value)
    return z / (1.0 + z)


def logistic_loss_and_gradient(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    weights: Sequence[float],
    regularization: float = 0.0,
) -> Tuple[float, Vector]:
    """
    Binary logistic regression objective:

        L(w) = average[
            log(1 + exp(z_i)) - y_i z_i
        ] + lambda/2 ||w||^2

    where z_i = w^T x_i.

    The implementation uses a stable softplus expression.
    """
    if len(features) != len(labels):
        raise ValueError("features and labels must match")

    if not features:
        raise ValueError("At least one sample is required")

    dimension = len(weights)

    if any(len(row) != dimension for row in features):
        raise ValueError("Feature dimensions must match weights")

    loss = 0.0
    gradient = [0.0] * dimension

    for row, label in zip(features, labels):
        if label not in (0, 1):
            raise ValueError("Labels must be 0 or 1")

        score = dot(row, weights)

        # Stable softplus:
        softplus = max(score, 0.0) + math.log1p(
            math.exp(-abs(score))
        )

        loss += softplus - label * score

        probability = sigmoid(score)
        error = probability - label

        for j in range(dimension):
            gradient[j] += error * row[j]

    count = len(features)

    loss /= count
    gradient = [
        value / count
        for value in gradient
    ]

    if regularization > 0:
        loss += 0.5 * regularization * dot(weights, weights)

        gradient = [
            gradient[j] + regularization * weights[j]
            for j in range(dimension)
        ]

    return loss, gradient


def train_logistic_regression(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    learning_rate: float = 0.5,
    iterations: int = 1000,
    regularization: float = 0.01,
) -> Tuple[Vector, List[float]]:
    """Train logistic regression with gradient descent."""
    if not features:
        raise ValueError("No training data supplied")

    weights = [0.0] * len(features[0])
    losses = []

    for _ in range(iterations):
        loss, gradient = logistic_loss_and_gradient(
            features,
            labels,
            weights,
            regularization,
        )

        losses.append(loss)

        weights = vector_subtract(
            weights,
            scalar_multiply(learning_rate, gradient),
        )

    return weights, losses


def logistic_predict(
    features: Sequence[float],
    weights: Sequence[float],
) -> int:
    return int(sigmoid(dot(features, weights)) >= 0.5)


# ---------------------------------------------------------------------------
# 15. SIMPLE GRID SEARCH
# ---------------------------------------------------------------------------

def grid_search_1d(
    function: Callable[[float], float],
    start: float,
    end: float,
    points: int,
) -> Tuple[float, float]:
    """
    Brute-force search.

    It is simple and robust for small one-dimensional domains but becomes
    computationally expensive as dimensionality increases.
    """
    if points < 2:
        raise ValueError("points must be at least 2")

    best_x = start
    best_value = function(start)

    for i in range(points):
        x = start + (end - start) * i / (points - 1)
        value = function(x)

        if value < best_value:
            best_x = x
            best_value = value

    return best_x, best_value


# ---------------------------------------------------------------------------
# 16. SIMULATED ANNEALING
# ---------------------------------------------------------------------------

def simulated_annealing_1d(
    function: Callable[[float], float],
    start: float,
    lower: float,
    upper: float,
    initial_temperature: float = 10.0,
    cooling_rate: float = 0.995,
    iterations: int = 5000,
    step_scale: float = 1.0,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Simulated annealing is a stochastic global-search heuristic.

    Unlike gradient descent, it can occasionally accept an uphill move.
    This can help escape local minima in non-convex problems.
    """
    if initial_temperature <= 0:
        raise ValueError("Temperature must be positive")

    if not 0.0 < cooling_rate < 1.0:
        raise ValueError("cooling_rate must be between 0 and 1")

    rng = random.Random(seed)

    current = max(lower, min(start, upper))
    current_value = function(current)

    best = current
    best_value = current_value

    temperature = initial_temperature

    for _ in range(iterations):
        candidate = current + rng.gauss(0.0, step_scale)
        candidate = max(lower, min(candidate, upper))

        candidate_value = function(candidate)
        difference = candidate_value - current_value

        if (
            difference <= 0.0
            or rng.random() < math.exp(
                -difference / max(temperature, 1e-12)
            )
        ):
            current = candidate
            current_value = candidate_value

        if current_value < best_value:
            best = current
            best_value = current_value

        temperature *= cooling_rate

    return best, best_value


# ---------------------------------------------------------------------------
# 17. EXAMPLES AND STUDY OUTPUT
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_fundamentals() -> None:
    print_section("1. FUNDAMENTALS: OBJECTIVE FUNCTIONS AND MINIMA")

    points = [-2.0, 0.0, 2.0, 3.0, 5.0]

    for point in points:
        print(
            f"x={point:5.1f}, f(x)={square(point):8.3f}"
        )

    stationary = 3.0

    print(
        f"\nFor f(x)=(x-3)^2+2, x={stationary} is a "
        f"{classify_stationary_point(square, stationary)}."
    )

    print(
        "A global minimum is at least as good as every feasible point. "
        "A local minimum only needs to be best in some neighborhood."
    )


def run_nonconvex_example() -> None:
    print_section("2. LOCAL VERSUS GLOBAL OPTIMA IN A NON-CONVEX FUNCTION")

    candidate_points = [-math.sqrt(2.0), 0.0, math.sqrt(2.0)]

    for point in candidate_points:
        print(
            f"x={point: .6f}, f(x)={double_well(point): .6f}, "
            f"classification={classify_stationary_point(double_well, point)}"
        )

    print(
        "\nThe non-convex function has multiple local structures. "
        "Finding a stationary point does not automatically prove global optimality."
    )


def run_gradient_descent_examples() -> None:
    print_section("3. GRADIENT DESCENT")

    solution, history = gradient_descent_1d(
        square,
        start=-8.0,
        learning_rate=0.2,
        iterations=100,
    )

    print(f"Gradient descent solution: x={solution:.10f}")
    print(f"Iterations used: {len(history)}")

    for record in history[:5]:
        print(
            f"iteration={record[0]}, "
            f"x={record[1]:.6f}, "
            f"value={record[2]:.6f}, "
            f"gradient={record[3]:.6f}"
        )


def run_newton_example() -> None:
    print_section("4. NEWTON'S METHOD")

    solution, history = newton_1d(
        square,
        start=100.0,
        iterations=20,
    )

    print(f"Newton solution: x={solution:.10f}")
    print(f"Iterations used: {len(history)}")


def run_convexity_examples() -> None:
    print_section("5. CONVEXITY")

    convex_quadratic = lambda x: x * x + 2.0 * x + 5.0

    print(
        "x^2 + 2x + 5 convex:",
        is_convex_1d(convex_quadratic, -10, 10),
    )

    print(
        "x^2 + 2x + 5 strictly convex:",
        is_strictly_convex_1d(convex_quadratic, -10, 10),
    )

    print(
        "Non-convex double-well function convex:",
        is_convex_1d(double_well, -3, 3),
    )

    print(
        "\nFor a convex minimization problem, every local minimum is also "
        "a global minimum. Strict convexity provides uniqueness of the minimizer "
        "when a minimizer exists."
    )


def run_quadratic_example() -> None:
    print_section("6. MULTIVARIABLE QUADRATIC OPTIMIZATION")

    quadratic = QuadraticFunction(
        q=[
            [4.0, 1.0],
            [1.0, 2.0],
        ],
        c=[
            -8.0,
            -6.0,
        ],
        constant=0.0,
    )

    exact = quadratic_exact_minimizer(quadratic)

    gradient_solution, history = gradient_descent(
        quadratic,
        quadratic.gradient,
        start=[0.0, 0.0],
        learning_rate=0.15,
        iterations=1000,
    )

    print("Exact stationary point:", exact)
    print("Gradient descent point:", gradient_solution)
    print("Exact objective:", quadratic(exact))
    print("Gradient descent objective:", quadratic(gradient_solution))

    print(
        "Hessian classification:",
        classify_multivariable_point(quadratic.hessian()),
    )


def run_line_search_example() -> None:
    print_section("7. ADAPTIVE STEP SIZE WITH BACKTRACKING")

    function = lambda x: (
        20.0 * x[0] ** 2
        + x[1] ** 2
        + 3.0 * x[0]
    )

    gradient = lambda x: [
        40.0 * x[0] + 3.0,
        2.0 * x[1],
    ]

    solution, history = gradient_descent_with_line_search(
        function,
        gradient,
        start=[10.0, 10.0],
    )

    print("Solution:", solution)
    print("Objective:", function(solution))
    print("Iterations:", len(history))


def run_projection_example() -> None:
    print_section("8. BOX-CONSTRAINED OPTIMIZATION")

    function = lambda x: (
        (x[0] - 5.0) ** 2
        + (x[1] + 3.0) ** 2
    )

    gradient = lambda x: [
        2.0 * (x[0] - 5.0),
        2.0 * (x[1] + 3.0),
    ]

    solution, _ = projected_gradient_descent(
        function,
        gradient,
        start=[0.0, 0.0],
        lower=[0.0, -1.0],
        upper=[4.0, 2.0],
        learning_rate=0.1,
    )

    print("Constrained solution:", solution)
    print("Objective:", function(solution))

    print(
        "The unconstrained optimum is (5,-3), but the feasible box prevents "
        "both coordinates from reaching those values."
    )


def run_lagrange_example() -> None:
    print_section("9. LAGRANGE MULTIPLIERS")

    point, multiplier = lagrange_equality_example()

    print("Optimization problem:")
    print("  minimize x^2 + y^2")
    print("  subject to x + y = 10")
    print("Optimal point:", point)
    print("Lagrange multiplier:", multiplier)

    print(
        "The multiplier measures the first-order sensitivity of the optimal "
        "objective value to a change in the equality constraint's right-hand side."
    )


def run_penalty_example() -> None:
    print_section("10. PENALTY METHOD")

    for penalty in [1.0, 10.0, 100.0, 1000.0]:
        objective = (
            lambda point, p=penalty:
            equality_penalty_objective(point, p)
        )

        gradient = (
            lambda point, p=penalty:
            equality_penalty_gradient(point, p)
        )

        solution, _ = gradient_descent(
            objective,
            gradient,
            start=[0.0, 0.0],
            learning_rate=0.01,
            iterations=5000,
        )

        violation = solution[0] + solution[1] - 10.0

        print(
            f"penalty={penalty:8.1f}, "
            f"solution={solution}, "
            f"constraint violation={violation:.8f}"
        )

    print(
        "A larger penalty usually enforces feasibility more strongly, but "
        "very large penalties can create poorly conditioned numerical problems."
    )


def run_kkt_example() -> None:
    print_section("11. KKT CONDITIONS")

    example = KKTExample()

    print("KKT conditions satisfied:", example.verify())
    print("Optimal x:", example.optimum)
    print("Multiplier:", example.multiplier)

    print(
        "For inequality constraints, KKT conditions combine stationarity, "
        "primal feasibility, dual feasibility, and complementary slackness."
    )


def run_least_squares_example() -> None:
    print_section("12. LEAST-SQUARES REGRESSION")

    x_values = [1, 2, 3, 4, 5]
    y_values = [2.2, 4.1, 6.2, 7.9, 10.1]

    intercept, slope = least_squares_fit(
        x_values,
        y_values,
    )

    predictions = [
        intercept + slope * x
        for x in x_values
    ]

    error = mean_squared_error(
        predictions,
        y_values,
    )

    print(f"Intercept: {intercept:.6f}")
    print(f"Slope: {slope:.6f}")
    print(f"MSE: {error:.6f}")


def run_logistic_example() -> None:
    print_section("13. LOGISTIC REGRESSION AS OPTIMIZATION")

    features = [
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
        [1.0, 5.0],
    ]

    labels = [0, 0, 0, 1, 1, 1]

    weights, losses = train_logistic_regression(
        features,
        labels,
        learning_rate=0.5,
        iterations=1000,
        regularization=0.01,
    )

    print("Learned weights:", weights)
    print("Initial loss:", losses[0])
    print("Final loss:", losses[-1])

    predictions = [
        logistic_predict(row, weights)
        for row in features
    ]

    print("Predictions:", predictions)
    print("Actual:", labels)


def run_global_search_example() -> None:
    print_section("14. GLOBAL SEARCH HEURISTICS")

    grid_x, grid_value = grid_search_1d(
        double_well,
        -3.0,
        3.0,
        2001,
    )

    annealing_x, annealing_value = simulated_annealing_1d(
        double_well,
        start=0.0,
        lower=-3.0,
        upper=3.0,
        seed=42,
    )

    print(
        f"Grid search: x={grid_x:.6f}, value={grid_value:.6f}"
    )

    print(
        f"Simulated annealing: x={annealing_x:.6f}, "
        f"value={annealing_value:.6f}"
    )

    print(
        "Global optimization methods can be useful for non-convex landscapes "
        "where local gradient information may be insufficient."
    )


# ---------------------------------------------------------------------------
# 18. EDGE CASES AND FAILURE MODES
# ---------------------------------------------------------------------------

def run_edge_cases() -> None:
    print_section("15. EDGE CASES AND COMMON FAILURE MODES")

    print("1. Zero-dimensional vector:")
    try:
        print("norm([]) =", norm([]))
    except Exception as error:
        print("Error:", error)

    print("\n2. Mismatched vector dimensions:")
    try:
        print(vector_add([1, 2], [3]))
    except ValueError as error:
        print("Expected error:", error)

    print("\n3. Singular linear system:")
    try:
        solve_linear_system(
            [[1, 2], [2, 4]],
            [3, 6],
        )
    except ValueError as error:
        print("Expected error:", error)

    print("\n4. Invalid projected bounds:")
    try:
        project_to_box(
            [1],
            [5],
            [2],
        )
    except ValueError as error:
        print("Expected error:", error)

    print("\nImportant numerical issues:")
    print("- Poor learning rates can cause divergence or extremely slow convergence.")
    print("- Flat curvature can make gradient methods progress slowly.")
    print("- Ill-conditioned Hessians can make Newton-type methods unstable.")
    print("- Finite differences introduce truncation and floating-point errors.")
    print("- A stationary point is not necessarily a minimum.")
    print("- Feasibility must be checked separately from objective quality.")
    print("- Non-convex optimization can contain many local minima.")


# ---------------------------------------------------------------------------
# 19. TESTS
# ---------------------------------------------------------------------------

def assert_close(
    actual: float,
    expected: float,
    tolerance: float = 1e-5,
) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"Expected {expected}, received {actual}"
        )


def run_tests() -> None:
    print_section("16. BUILT-IN VALIDATION TESTS")

    solution, _ = gradient_descent_1d(
        square,
        start=-10,
        learning_rate=0.2,
        iterations=200,
    )

    assert_close(solution, 3.0, 1e-4)

    newton_solution, _ = newton_1d(
        square,
        start=-10,
    )

    assert_close(newton_solution, 3.0, 1e-5)

    quadratic = QuadraticFunction(
        q=[
            [4.0, 1.0],
            [1.0, 2.0],
        ],
        c=[
            -8.0,
            -6.0,
        ],
    )

    exact = quadratic_exact_minimizer(quadratic)

    assert_close(
        exact[0],
        10.0 / 7.0,
        1e-6,
    )

    assert_close(
        exact[1],
        17.0 / 7.0,
        1e-6,
    )

    kkt = KKTExample()
    assert kkt.verify()

    intercept, slope = least_squares_fit(
        [1, 2, 3],
        [2, 4, 6],
    )

    assert_close(intercept, 0.0, 1e-10)
    assert_close(slope, 2.0, 1e-10)

    print("All tests passed.")


# ---------------------------------------------------------------------------
# 20. MAIN PROGRAM
# ---------------------------------------------------------------------------

def main() -> None:
    print("OPTIMIZATION MATHEMATICS")
    print("Executable study program")
    print("Standard-library Python implementation")

    run_fundamentals()
    run_nonconvex_example()
    run_gradient_descent_examples()
    run_newton_example()
    run_convexity_examples()
    run_quadratic_example()
    run_line_search_example()
    run_projection_example()
    run_lagrange_example()
    run_penalty_example()
    run_kkt_example()
    run_least_squares_example()
    run_logistic_example()
    run_global_search_example()
    run_edge_cases()
    run_tests()

    print_section("21. KEY MATHEMATICAL RELATIONSHIPS")

    print(
        "Unconstrained first-order condition:       grad f(x*) = 0"
    )
    print(
        "Second-order minimum condition:            Hessian positive definite"
    )
    print(
        "Second-order maximum condition:            Hessian negative definite"
    )
    print(
        "Gradient descent update:                   x <- x - alpha grad f(x)"
    )
    print(
        "Newton update:                             x <- x - H^-1 grad f(x)"
    )
    print(
        "Equality-constrained Lagrangian:           L=f+lambda^T h"
    )
    print(
        "Inequality KKT requirement:                lambda >= 0 for g(x)<=0"
    )
    print(
        "Complementary slackness:                   lambda_i g_i(x)=0"
    )
    print(
        "Convex minimization:                       local minimum => global minimum"
    )

    print_section("22. STUDY COMPLETE")


if __name__ == "__main__":
    main()
