"""
MATHEMATICAL REVISION
Probability, Statistics, Linear Algebra, Calculus, and Optimization

A self-contained study and demonstration script progressing from foundational
ideas to advanced computational examples.

The script intentionally uses only the Python standard library.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


# =============================================================================
# 1. BASIC NUMERICAL FOUNDATIONS
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def subsection(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


section("MATHEMATICAL REVISION")

print(
    """
Topics:
1. Mathematical foundations
2. Probability
3. Statistics
4. Linear algebra
5. Calculus
6. Optimization
7. Numerical methods
8. Integrated examples
"""
)

subsection("1. Arithmetic, powers, logarithms, and numerical precision")

a = 12
b = 5

print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)
print("Floor division:", a // b)
print("Remainder:", a % b)
print("Power:", a**2)
print("Square root:", math.sqrt(a))
print("Natural logarithm:", math.log(a))
print("Logarithm base 10:", math.log10(a))
print("Exponential:", math.exp(2))

# Floating-point arithmetic uses finite binary representations, so some
# decimal fractions cannot be represented exactly.
floating_result = 0.1 + 0.2
print("0.1 + 0.2 =", floating_result)
print("Approximately equal to 0.3:", math.isclose(floating_result, 0.3))


# =============================================================================
# 2. FUNCTIONS AND ALGEBRA
# =============================================================================

subsection("2. Functions")

def linear_function(x: float, slope: float, intercept: float) -> float:
    """Evaluate f(x) = slope*x + intercept."""
    return slope * x + intercept


def quadratic(x: float, a: float, b: float, c: float) -> float:
    """Evaluate f(x) = ax² + bx + c."""
    return a * x * x + b * x + c


print("f(3) for f(x)=2x+1:", linear_function(3, 2, 1))
print("q(2) for q(x)=x²-3x+2:", quadratic(2, 1, -3, 2))


def solve_quadratic(a: float, b: float, c: float) -> tuple[complex, complex]:
    """Solve ax² + bx + c = 0 using the quadratic formula."""
    if a == 0:
        if b == 0:
            raise ValueError("This is not a valid linear equation.")
        root = -c / b
        return complex(root), complex(root)

    discriminant = b * b - 4 * a * c
    sqrt_discriminant = complex(discriminant) ** 0.5
    root1 = (-b + sqrt_discriminant) / (2 * a)
    root2 = (-b - sqrt_discriminant) / (2 * a)
    return root1, root2


print("Roots of x²-5x+6:", solve_quadratic(1, -5, 6))
print("Roots of x²+1:", solve_quadratic(1, 0, 1))


# =============================================================================
# 3. PROBABILITY
# =============================================================================

section("PROBABILITY")

subsection("3.1 Fundamental terminology")

print(
    """
Experiment:
    A repeatable process producing an outcome.

Sample space:
    The set of all possible outcomes.

Event:
    A subset of the sample space.

Probability:
    A numerical measure between 0 and 1 describing uncertainty.

Complement:
    P(A^c) = 1 - P(A)

Union:
    P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

Conditional probability:
    P(A|B) = P(A ∩ B) / P(B), when P(B) > 0

Independence:
    A and B are independent when P(A ∩ B) = P(A)P(B).
"""
)


subsection("3.2 Classical probability")

def probability_event(event_count: int, total_count: int) -> float:
    if total_count <= 0:
        raise ValueError("Total number of outcomes must be positive.")
    if not 0 <= event_count <= total_count:
        raise ValueError("Event count must be between zero and total count.")
    return event_count / total_count


print("Probability of rolling an even number:", probability_event(3, 6))


subsection("3.3 Combinations and permutations")

def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("Factorial is defined for non-negative integers.")
    return math.factorial(n)


def permutations(n: int, r: int) -> int:
    if not 0 <= r <= n:
        raise ValueError("Require 0 <= r <= n.")
    return math.factorial(n) // math.factorial(n - r)


def combinations(n: int, r: int) -> int:
    if not 0 <= r <= n:
        raise ValueError("Require 0 <= r <= n.")
    return math.comb(n, r)


print("5! =", factorial(5))
print("5P2 =", permutations(5, 2))
print("5C2 =", combinations(5, 2))


subsection("3.4 Conditional probability and Bayes' theorem")

def conditional_probability(
    intersection: float,
    condition_probability: float,
) -> float:
    if condition_probability <= 0:
        raise ValueError("Condition probability must be positive.")
    return intersection / condition_probability


p_disease = 0.01
p_positive_given_disease = 0.95
p_positive_given_no_disease = 0.05
p_no_disease = 1 - p_disease

p_positive = (
    p_positive_given_disease * p_disease
    + p_positive_given_no_disease * p_no_disease
)

p_disease_given_positive = (
    p_positive_given_disease * p_disease / p_positive
)

print("P(positive) =", p_positive)
print("P(disease | positive) =", p_disease_given_positive)


subsection("3.5 Discrete random variables")

def binomial_probability(n: int, k: int, p: float) -> float:
    if n < 0 or not 0 <= k <= n:
        raise ValueError("Require n >= 0 and 0 <= k <= n.")
    if not 0 <= p <= 1:
        raise ValueError("Probability must lie in [0, 1].")
    return combinations(n, k) * p**k * (1 - p) ** (n - k)


print(
    "Probability of exactly 3 successes in 10 trials with p=0.4:",
    binomial_probability(10, 3, 0.4),
)


def binomial_mean(n: int, p: float) -> float:
    return n * p


def binomial_variance(n: int, p: float) -> float:
    return n * p * (1 - p)


print("Binomial mean:", binomial_mean(10, 0.4))
print("Binomial variance:", binomial_variance(10, 0.4))


subsection("3.6 Expected value and variance")

def expected_value(
    values: Sequence[float],
    probabilities: Sequence[float],
) -> float:
    if len(values) != len(probabilities):
        raise ValueError("Values and probabilities must have equal length.")
    if any(p < 0 for p in probabilities):
        raise ValueError("Probabilities cannot be negative.")
    if not math.isclose(sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    return sum(x * p for x, p in zip(values, probabilities))


def discrete_variance(
    values: Sequence[float],
    probabilities: Sequence[float],
) -> float:
    mean = expected_value(values, probabilities)
    return sum(p * (x - mean) ** 2 for x, p in zip(values, probabilities))


die_values = [1, 2, 3, 4, 5, 6]
die_probabilities = [1 / 6] * 6

print("Expected fair-die value:", expected_value(die_values, die_probabilities))
print(
    "Fair-die variance:",
    discrete_variance(die_values, die_probabilities),
)


subsection("3.7 Law of total probability")

# If B1, B2, ..., Bn form a partition:
# P(A) = Σ P(A|Bi)P(Bi)

factory_defect_rates = [0.01, 0.03, 0.02]
factory_proportions = [0.50, 0.30, 0.20]

overall_defect_rate = sum(
    defect * proportion
    for defect, proportion in zip(factory_defect_rates, factory_proportions)
)

print("Overall defect probability:", overall_defect_rate)


subsection("3.8 Monte Carlo simulation")

def monte_carlo_pi(samples: int, seed: int = 42) -> float:
    if samples <= 0:
        raise ValueError("Number of samples must be positive.")

    generator = random.Random(seed)
    inside = 0

    for _ in range(samples):
        x = generator.uniform(-1, 1)
        y = generator.uniform(-1, 1)

        if x * x + y * y <= 1:
            inside += 1

    return 4 * inside / samples


print("Monte Carlo estimate of pi:", monte_carlo_pi(100_000))


# =============================================================================
# 4. STATISTICS
# =============================================================================

section("STATISTICS")

sample = [12, 15, 15, 18, 20, 22, 22, 24, 30]

subsection("4.1 Descriptive statistics")

print("Data:", sample)
print("Count:", len(sample))
print("Mean:", statistics.mean(sample))
print("Median:", statistics.median(sample))
print("Mode(s):", statistics.multimode(sample))
print("Minimum:", min(sample))
print("Maximum:", max(sample))
print("Range:", max(sample) - min(sample))
print("Population variance:", statistics.pvariance(sample))
print("Population standard deviation:", statistics.pstdev(sample))
print("Sample variance:", statistics.variance(sample))
print("Sample standard deviation:", statistics.stdev(sample))


def percentile(data: Sequence[float], percentage: float) -> float:
    """Linear-interpolation percentile implementation."""
    if not data:
        raise ValueError("Data cannot be empty.")
    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100.")

    values = sorted(data)
    position = (len(values) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


print("25th percentile:", percentile(sample, 25))
print("50th percentile:", percentile(sample, 50))
print("75th percentile:", percentile(sample, 75))


subsection("4.2 Z-scores")

def z_score(x: float, mean: float, standard_deviation: float) -> float:
    if standard_deviation <= 0:
        raise ValueError("Standard deviation must be positive.")
    return (x - mean) / standard_deviation


mean_sample = statistics.mean(sample)
std_sample = statistics.stdev(sample)

print(
    "Z-score of 30:",
    z_score(30, mean_sample, std_sample),
)


subsection("4.3 Covariance and correlation")

def sample_covariance(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Samples must have equal length and at least 2 values.")

    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)

    return sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    ) / (len(x_values) - 1)


def correlation(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> float:
    covariance = sample_covariance(x_values, y_values)
    x_std = statistics.stdev(x_values)
    y_std = statistics.stdev(y_values)

    if x_std == 0 or y_std == 0:
        raise ValueError("Correlation is undefined for zero variance.")

    return covariance / (x_std * y_std)


study_hours = [1, 2, 3, 4, 5, 6]
exam_scores = [45, 50, 58, 65, 73, 82]

print("Covariance:", sample_covariance(study_hours, exam_scores))
print("Correlation:", correlation(study_hours, exam_scores))


subsection("4.4 Simple linear regression")

@dataclass
class LinearRegressionModel:
    slope: float
    intercept: float

    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept


def fit_linear_regression(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> LinearRegressionModel:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Need equal-length samples with at least two values.")

    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    )

    denominator = sum((x - x_mean) ** 2 for x in x_values)

    if denominator == 0:
        raise ValueError("Regression slope is undefined.")

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return LinearRegressionModel(slope, intercept)


regression_model = fit_linear_regression(study_hours, exam_scores)

print("Regression slope:", regression_model.slope)
print("Regression intercept:", regression_model.intercept)
print("Predicted score for 7 hours:", regression_model.predict(7))


def coefficient_of_determination(
    model: LinearRegressionModel,
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> float:
    mean_y = statistics.mean(y_values)
    total_sum_squares = sum((y - mean_y) ** 2 for y in y_values)
    residual_sum_squares = sum(
        (y - model.predict(x)) ** 2
        for x, y in zip(x_values, y_values)
    )

    if total_sum_squares == 0:
        raise ValueError("R² is undefined for constant observations.")

    return 1 - residual_sum_squares / total_sum_squares


print(
    "R²:",
    coefficient_of_determination(
        regression_model,
        study_hours,
        exam_scores,
    ),
)


# =============================================================================
# 5. LINEAR ALGEBRA
# =============================================================================

section("LINEAR ALGEBRA")

subsection("5.1 Vectors")

Vector = list[float]
Matrix = list[list[float]]


def vector_add(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal dimensions.")
    return [x + y for x, y in zip(a, b)]


def vector_subtract(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal dimensions.")
    return [x - y for x, y in zip(a, b)]


def scalar_multiply(scalar: float, vector: Vector) -> Vector:
    return [scalar * x for x in vector]


def dot_product(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal dimensions.")
    return sum(x * y for x, y in zip(a, b))


def vector_norm(vector: Vector) -> float:
    return math.sqrt(dot_product(vector, vector))


def cosine_similarity(a: Vector, b: Vector) -> float:
    denominator = vector_norm(a) * vector_norm(b)
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    return dot_product(a, b) / denominator


v1 = [1, 2, 3]
v2 = [4, 5, 6]

print("v1 + v2:", vector_add(v1, v2))
print("v1 - v2:", vector_subtract(v1, v2))
print("2v1:", scalar_multiply(2, v1))
print("Dot product:", dot_product(v1, v2))
print("Norm of v1:", vector_norm(v1))
print("Cosine similarity:", cosine_similarity(v1, v2))


subsection("5.2 Matrices")

def validate_matrix(matrix: Matrix) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("Matrix cannot be empty.")

    columns = len(matrix[0])

    if any(len(row) != columns for row in matrix):
        raise ValueError("Matrix rows must have equal length.")

    return len(matrix), columns


def matrix_add(a: Matrix, b: Matrix) -> Matrix:
    rows_a, cols_a = validate_matrix(a)
    rows_b, cols_b = validate_matrix(b)

    if (rows_a, cols_a) != (rows_b, cols_b):
        raise ValueError("Matrices must have equal dimensions.")

    return [
        [x + y for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)
    ]


def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    rows_a, cols_a = validate_matrix(a)
    rows_b, cols_b = validate_matrix(b)

    if cols_a != rows_b:
        raise ValueError("Inner matrix dimensions must agree.")

    return [
        [
            sum(a[i][k] * b[k][j] for k in range(cols_a))
            for j in range(cols_b)
        ]
        for i in range(rows_a)
    ]


def transpose(matrix: Matrix) -> Matrix:
    validate_matrix(matrix)
    return [list(column) for column in zip(*matrix)]


A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

print("A + B:", matrix_add(A, B))
print("AB:", matrix_multiply(A, B))
print("Aᵀ:", transpose(A))


subsection("5.3 Determinant")

def determinant(matrix: Matrix) -> float:
    rows, cols = validate_matrix(matrix)

    if rows != cols:
        raise ValueError("Determinant requires a square matrix.")

    n = rows

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Laplace expansion is clear educationally but expensive for large n.
    total = 0.0

    for column in range(n):
        minor = [
            [
                matrix[i][j]
                for j in range(n)
                if j != column
            ]
            for i in range(1, n)
        ]

        sign = -1 if column % 2 else 1
        total += sign * matrix[0][column] * determinant(minor)

    return total


print("det(A) =", determinant(A))


subsection("5.4 Gaussian elimination and solving Ax=b")

def solve_linear_system(
    matrix: Matrix,
    constants: Vector,
    tolerance: float = 1e-12,
) -> Vector:
    rows, columns = validate_matrix(matrix)

    if rows != columns:
        raise ValueError("Coefficient matrix must be square.")

    if len(constants) != rows:
        raise ValueError("Constants vector has incorrect dimension.")

    augmented = [
        [float(value) for value in row] + [float(constants[i])]
        for i, row in enumerate(matrix)
    ]

    n = rows

    for pivot_column in range(n):
        pivot_row = max(
            range(pivot_column, n),
            key=lambda row: abs(augmented[row][pivot_column]),
        )

        if abs(augmented[pivot_row][pivot_column]) < tolerance:
            raise ValueError("Matrix is singular or nearly singular.")

        augmented[pivot_column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot_column],
        )

        pivot = augmented[pivot_column][pivot_column]

        for j in range(pivot_column, n + 1):
            augmented[pivot_column][j] /= pivot

        for row in range(n):
            if row == pivot_column:
                continue

            factor = augmented[row][pivot_column]

            for j in range(pivot_column, n + 1):
                augmented[row][j] -= factor * augmented[pivot_column][j]

    return [augmented[i][n] for i in range(n)]


system_matrix = [[2, 1], [1, -1]]
system_constants = [7, 1]

print(
    "Solution to 2x+y=7, x-y=1:",
    solve_linear_system(system_matrix, system_constants),
)


subsection("5.5 Eigenvalue intuition")

# For a triangular matrix, eigenvalues are its diagonal entries.
triangular_matrix = [[3, 1], [0, 5]]
print("Eigenvalues of triangular matrix:", [3, 5])


def power_iteration(
    matrix: Matrix,
    iterations: int = 100,
) -> tuple[float, Vector]:
    """Estimate the dominant eigenvalue and corresponding eigenvector."""
    rows, cols = validate_matrix(matrix)

    if rows != cols:
        raise ValueError("Power iteration requires a square matrix.")

    vector = [1.0] * rows

    for _ in range(iterations):
        transformed = [
            sum(matrix[i][j] * vector[j] for j in range(cols))
            for i in range(rows)
        ]

        norm = vector_norm(transformed)

        if norm == 0:
            raise ValueError("Iteration reached a zero vector.")

        vector = [value / norm for value in transformed]

    transformed = [
        sum(matrix[i][j] * vector[j] for j in range(cols))
        for i in range(rows)
    ]

    eigenvalue = dot_product(vector, transformed)

    return eigenvalue, vector


eigenvalue, eigenvector = power_iteration([[4, 1], [2, 3]])

print("Estimated dominant eigenvalue:", eigenvalue)
print("Estimated eigenvector:", eigenvector)


# =============================================================================
# 6. CALCULUS
# =============================================================================

section("CALCULUS")

subsection("6.1 Limits and continuity")

print(
    """
A limit describes the value a function approaches as its input approaches
a particular point.

A function is continuous at a point when:
    1. f(a) exists
    2. lim(x→a) f(x) exists
    3. lim(x→a) f(x) = f(a)
"""
)


def numerical_limit(
    function: Callable[[float], float],
    point: float,
    direction: float = 1.0,
    steps: int = 8,
) -> float:
    """Estimate a one-sided limit by successively reducing the step size."""
    if direction not in (-1, 1):
        raise ValueError("Direction must be -1 or +1.")

    result = None

    for exponent in range(1, steps + 1):
        h = 10 ** (-exponent)
        result = function(point + direction * h)

    if result is None:
        raise RuntimeError("Limit estimation failed.")

    return result


print(
    "lim(x→1) x²:",
    numerical_limit(lambda x: x * x, 1),
)


subsection("6.2 Numerical differentiation")

def forward_difference(
    function: Callable[[float], float],
    x: float,
    h: float = 1e-5,
) -> float:
    if h <= 0:
        raise ValueError("Step size must be positive.")
    return (function(x + h) - function(x)) / h


def central_difference(
    function: Callable[[float], float],
    x: float,
    h: float = 1e-5,
) -> float:
    if h <= 0:
        raise ValueError("Step size must be positive.")
    return (function(x + h) - function(x - h)) / (2 * h)


f = lambda x: x**3 - 2 * x + 1

print("Forward derivative at x=2:", forward_difference(f, 2))
print("Central derivative at x=2:", central_difference(f, 2))
print("Exact derivative at x=2:", 3 * 2**2 - 2)


subsection("6.3 Numerical second derivative")

def second_derivative(
    function: Callable[[float], float],
    x: float,
    h: float = 1e-4,
) -> float:
    if h <= 0:
        raise ValueError("Step size must be positive.")

    return (
        function(x + h)
        - 2 * function(x)
        + function(x - h)
    ) / h**2


print("Second derivative of x³-2x+1 at x=2:", second_derivative(f, 2))
print("Exact second derivative:", 6 * 2)


subsection("6.4 Numerical integration")

def trapezoidal_integral(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    intervals: int = 1000,
) -> float:
    if intervals <= 0:
        raise ValueError("Intervals must be positive.")

    if lower == upper:
        return 0.0

    step = (upper - lower) / intervals

    total = 0.5 * (function(lower) + function(upper))

    for index in range(1, intervals):
        total += function(lower + index * step)

    return total * step


print(
    "Integral of x² from 0 to 3:",
    trapezoidal_integral(lambda x: x * x, 0, 3),
)
print("Exact value:", 9)


def simpson_integral(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    intervals: int = 1000,
) -> float:
    if intervals <= 0 or intervals % 2 != 0:
        raise ValueError("Simpson's rule requires a positive even interval count.")

    h = (upper - lower) / intervals

    total = function(lower) + function(upper)

    for index in range(1, intervals):
        x = lower + index * h
        coefficient = 4 if index % 2 else 2
        total += coefficient * function(x)

    return total * h / 3


print(
    "Simpson integral of x² from 0 to 3:",
    simpson_integral(lambda x: x * x, 0, 3),
)


subsection("6.5 Taylor series")

def exponential_taylor(x: float, terms: int = 20) -> float:
    if terms <= 0:
        raise ValueError("Terms must be positive.")

    total = 1.0
    term = 1.0

    for n in range(1, terms):
        term *= x / n
        total += term

    return total


print("e^2 using Taylor series:", exponential_taylor(2))
print("e^2 using math.exp:", math.exp(2))


# =============================================================================
# 7. OPTIMIZATION
# =============================================================================

section("OPTIMIZATION")

subsection("7.1 Optimization terminology")

print(
    """
Objective function:
    The function being minimized or maximized.

Decision variable:
    A quantity controlled by the optimization problem.

Constraint:
    A condition restricting feasible solutions.

Feasible region:
    All points satisfying the constraints.

Local minimum:
    A point whose objective is no larger than nearby feasible points.

Global minimum:
    A point achieving the lowest objective value over the entire feasible
    domain.

Gradient:
    Vector of first partial derivatives.

Hessian:
    Matrix of second partial derivatives.

Convex function:
    A function whose line segments lie above its graph. Convex minimization
    has especially useful global-optimality properties.
"""
)


subsection("7.2 Gradient and numerical gradient")

def numerical_gradient(
    function: Callable[[Vector], float],
    point: Vector,
    h: float = 1e-6,
) -> Vector:
    if h <= 0:
        raise ValueError("Step size must be positive.")

    gradient = []

    for i in range(len(point)):
        plus = list(point)
        minus = list(point)

        plus[i] += h
        minus[i] -= h

        derivative = (function(plus) - function(minus)) / (2 * h)
        gradient.append(derivative)

    return gradient


def quadratic_objective(point: Vector) -> float:
    x, y = point
    return (x - 3) ** 2 + 2 * (y + 1) ** 2


point = [0.0, 0.0]
print("Gradient at [0,0]:", numerical_gradient(quadratic_objective, point))


subsection("7.3 Gradient descent")

def gradient_descent(
    function: Callable[[Vector], float],
    initial_point: Vector,
    learning_rate: float = 0.1,
    iterations: int = 100,
    tolerance: float = 1e-8,
) -> tuple[Vector, list[float]]:
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
    if iterations <= 0:
        raise ValueError("Iterations must be positive.")

    point = list(initial_point)
    history = [function(point)]

    for _ in range(iterations):
        gradient = numerical_gradient(function, point)

        new_point = [
            coordinate - learning_rate * derivative
            for coordinate, derivative in zip(point, gradient)
        ]

        new_value = function(new_point)
        history.append(new_value)

        movement = vector_norm(
            vector_subtract(new_point, point)
        )

        point = new_point

        if movement < tolerance:
            break

    return point, history


minimum, objective_history = gradient_descent(
    quadratic_objective,
    [0.0, 0.0],
    learning_rate=0.1,
    iterations=200,
)

print("Estimated minimum:", minimum)
print("Objective at minimum:", quadratic_objective(minimum))
print("Iterations used:", len(objective_history) - 1)


subsection("7.4 Newton's method for one variable")

def numerical_newton(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    initial_value: float,
    iterations: int = 20,
    tolerance: float = 1e-10,
) -> float:
    x = initial_value

    for _ in range(iterations):
        derivative_value = derivative(x)

        if abs(derivative_value) < 1e-15:
            raise ZeroDivisionError("Derivative is too close to zero.")

        new_x = x - function(x) / derivative_value

        if abs(new_x - x) < tolerance:
            return new_x

        x = new_x

    return x


root = numerical_newton(
    lambda x: x * x - 2,
    lambda x: 2 * x,
    1.0,
)

print("Newton estimate of sqrt(2):", root)
print("Squared value:", root * root)


subsection("7.5 Golden-section search")

def golden_section_minimize(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    tolerance: float = 1e-10,
) -> float:
    if lower >= upper:
        raise ValueError("Lower bound must be smaller than upper bound.")

    golden_ratio = (1 + math.sqrt(5)) / 2

    c = upper - (upper - lower) / golden_ratio
    d = lower + (upper - lower) / golden_ratio

    while abs(upper - lower) > tolerance:
        if function(c) < function(d):
            upper = d
        else:
            lower = c

        c = upper - (upper - lower) / golden_ratio
        d = lower + (upper - lower) / golden_ratio

    return (lower + upper) / 2


one_dimensional_minimum = golden_section_minimize(
    lambda x: (x - 4) ** 2 + 2,
    -10,
    10,
)

print("Golden-section minimum:", one_dimensional_minimum)


# =============================================================================
# 8. CONSTRAINED OPTIMIZATION
# =============================================================================

section("CONSTRAINED OPTIMIZATION")

subsection("8.1 A simple resource-allocation problem")

print(
    """
Suppose two products x and y generate profits of 40 and 30 units.
Production consumes two limited resources:

    2x + y <= 100
    x + 2y <= 80
    x >= 0
    y >= 0

The objective is:

    maximize 40x + 30y

For a small linear program, the optimum can be found by examining feasible
vertices. This example demonstrates the relationship between constraints,
feasibility, and objective values without requiring an external optimizer.
"""
)


@dataclass
class Allocation:
    x: float
    y: float

    @property
    def profit(self) -> float:
        return 40 * self.x + 30 * self.y

    @property
    def feasible(self) -> bool:
        return (
            self.x >= 0
            and self.y >= 0
            and 2 * self.x + self.y <= 100 + 1e-12
            and self.x + 2 * self.y <= 80 + 1e-12
        )


candidate_points = [
    Allocation(0, 0),
    Allocation(50, 0),
    Allocation(0, 40),
    Allocation(40, 20),
]

feasible_points = [point for point in candidate_points if point.feasible]

for candidate in feasible_points:
    print(
        f"x={candidate.x:5.1f}, "
        f"y={candidate.y:5.1f}, "
        f"profit={candidate.profit:7.1f}"
    )

best_allocation = max(feasible_points, key=lambda item: item.profit)
print("Highest-profit feasible candidate:", best_allocation)


# =============================================================================
# 9. NUMERICAL STABILITY AND ERROR
# =============================================================================

section("NUMERICAL ANALYSIS")

subsection("9.1 Absolute and relative error")

def absolute_error(approximation: float, exact: float) -> float:
    return abs(approximation - exact)


def relative_error(approximation: float, exact: float) -> float:
    if exact == 0:
        return math.inf if approximation != 0 else 0.0
    return abs(approximation - exact) / abs(exact)


approximation = 3.14159
exact = math.pi

print("Absolute error:", absolute_error(approximation, exact))
print("Relative error:", relative_error(approximation, exact))


subsection("9.2 Conditioning and cancellation")

# Computing sqrt(x+1)-sqrt(x) directly can lose precision when x is large.
# Algebraically:
#
# sqrt(x+1)-sqrt(x)
# = 1 / (sqrt(x+1)+sqrt(x))
#
# The second form avoids subtracting two nearly equal large numbers.

def direct_difference(x: float) -> float:
    return math.sqrt(x + 1) - math.sqrt(x)


def stable_difference(x: float) -> float:
    return 1 / (math.sqrt(x + 1) + math.sqrt(x))


large_x = 1e16

print("Direct expression:", direct_difference(large_x))
print("Stable expression:", stable_difference(large_x))


# =============================================================================
# 10. INTEGRATED PROBABILITY AND STATISTICS
# =============================================================================

section("INTEGRATED EXAMPLE: ESTIMATING A POPULATION MEAN")

subsection("10.1 Simulation")

def simulate_measurements(
    count: int,
    mean: float,
    standard_deviation: float,
    seed: int = 123,
) -> list[float]:
    if count <= 0:
        raise ValueError("Count must be positive.")
    if standard_deviation <= 0:
        raise ValueError("Standard deviation must be positive.")

    generator = random.Random(seed)

    return [
        generator.gauss(mean, standard_deviation)
        for _ in range(count)
    ]


measurements = simulate_measurements(100, 50, 8)

sample_mean = statistics.mean(measurements)
sample_std = statistics.stdev(measurements)
sample_size = len(measurements)

standard_error = sample_std / math.sqrt(sample_size)

print("Sample mean:", sample_mean)
print("Sample standard deviation:", sample_std)
print("Standard error:", standard_error)


subsection("10.2 Approximate 95% confidence interval")

# For a sufficiently large sample, the normal approximation uses z ≈ 1.96.
z_95 = 1.96

confidence_lower = sample_mean - z_95 * standard_error
confidence_upper = sample_mean + z_95 * standard_error

print("Approximate 95% CI:", (confidence_lower, confidence_upper))


# =============================================================================
# 11. NORMAL DISTRIBUTION
# =============================================================================

section("NORMAL DISTRIBUTION")

def normal_pdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    if std <= 0:
        raise ValueError("Standard deviation must be positive.")

    coefficient = 1 / (std * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2

    return coefficient * math.exp(exponent)


def normal_cdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    if std <= 0:
        raise ValueError("Standard deviation must be positive.")

    standardized = (x - mean) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(standardized))


print("Standard normal PDF at 0:", normal_pdf(0))
print("Standard normal CDF at 0:", normal_cdf(0))
print("P(-1 <= Z <= 1):", normal_cdf(1) - normal_cdf(-1))


# =============================================================================
# 12. CENTRAL LIMIT THEOREM DEMONSTRATION
# =============================================================================

section("CENTRAL LIMIT THEOREM")

def sample_means(
    population: Sequence[float],
    sample_size: int,
    repetitions: int,
    seed: int = 99,
) -> list[float]:
    if not population:
        raise ValueError("Population cannot be empty.")
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    if repetitions <= 0:
        raise ValueError("Repetitions must be positive.")

    generator = random.Random(seed)
    means = []

    for _ in range(repetitions):
        sample_values = [
            generator.choice(population)
            for _ in range(sample_size)
        ]
        means.append(statistics.mean(sample_values))

    return means


skewed_population = [1] * 70 + [5] * 20 + [20] * 10

means = sample_means(
    skewed_population,
    sample_size=30,
    repetitions=5000,
)

print("Population mean:", statistics.mean(skewed_population))
print("Mean of simulated sample means:", statistics.mean(means))
print("Standard deviation of sample means:", statistics.stdev(means))


# =============================================================================
# 13. MATRIX-BASED LEAST SQUARES CONCEPT
# =============================================================================

section("LINEAR ALGEBRA + STATISTICS")

subsection("13.1 Normal equations")

print(
    """
For a linear model

    y ≈ Xβ

the least-squares objective is

    ||Xβ - y||².

When XᵀX is invertible, the normal-equation solution is

    β = (XᵀX)⁻¹Xᵀy.

In numerical production software, QR or SVD methods are often preferred
because explicitly forming an inverse can be less numerically stable.
"""
)


def least_squares_two_parameter(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[float, float]:
    """
    Fit y = beta0 + beta1*x using the closed-form two-parameter equations.
    """
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Need equal-length data with at least two points.")

    n = len(x_values)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xx = sum(x * x for x in x_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))

    denominator = n * sum_xx - sum_x**2

    if math.isclose(denominator, 0.0):
        raise ValueError("Predictor has insufficient variation.")

    beta1 = (n * sum_xy - sum_x * sum_y) / denominator
    beta0 = (sum_y - beta1 * sum_x) / n

    return beta0, beta1


intercept, slope = least_squares_two_parameter(
    study_hours,
    exam_scores,
)

print("Least-squares intercept:", intercept)
print("Least-squares slope:", slope)


# =============================================================================
# 14. LOGISTIC FUNCTION AND DERIVATIVE
# =============================================================================

section("CALCULUS + PROBABILITY")

def sigmoid(x: float) -> float:
    # Numerically stable implementation avoids exp overflow for very negative
    # or positive inputs.
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)

    z = math.exp(x)
    return z / (1 + z)


def sigmoid_derivative(x: float) -> float:
    value = sigmoid(x)
    return value * (1 - value)


for value in [-10, -1, 0, 1, 10]:
    print(
        f"x={value:>3}, sigmoid={sigmoid(value):.6f}, "
        f"derivative={sigmoid_derivative(value):.6f}"
    )


# =============================================================================
# 15. GRADIENT DESCENT FOR LINEAR REGRESSION
# =============================================================================

section("OPTIMIZATION + STATISTICS")

def linear_regression_gradient_descent(
    x_values: Sequence[float],
    y_values: Sequence[float],
    learning_rate: float = 0.01,
    iterations: int = 5000,
) -> tuple[float, float, list[float]]:
    if len(x_values) != len(y_values) or not x_values:
        raise ValueError("X and y must have equal non-zero length.")

    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")

    intercept = 0.0
    slope = 0.0
    losses = []

    n = len(x_values)

    for _ in range(iterations):
        predictions = [
            intercept + slope * x
            for x in x_values
        ]

        errors = [
            prediction - y
            for prediction, y in zip(predictions, y_values)
        ]

        loss = sum(error**2 for error in errors) / n
        losses.append(loss)

        gradient_intercept = 2 * sum(errors) / n
        gradient_slope = (
            2 * sum(error * x for error, x in zip(errors, x_values))
            / n
        )

        intercept -= learning_rate * gradient_intercept
        slope -= learning_rate * gradient_slope

    return intercept, slope, losses


gd_intercept, gd_slope, losses = linear_regression_gradient_descent(
    study_hours,
    exam_scores,
    learning_rate=0.01,
    iterations=5000,
)

print("Gradient-descent intercept:", gd_intercept)
print("Gradient-descent slope:", gd_slope)
print("Initial MSE:", losses[0])
print("Final MSE:", losses[-1])


# =============================================================================
# 16. HESSIAN AND SECOND-ORDER OPTIMIZATION
# =============================================================================

section("SECOND-ORDER OPTIMIZATION")

def numerical_hessian(
    function: Callable[[Vector], float],
    point: Vector,
    h: float = 1e-4,
) -> Matrix:
    if h <= 0:
        raise ValueError("Step size must be positive.")

    dimension = len(point)
    hessian = [
        [0.0 for _ in range(dimension)]
        for _ in range(dimension)
    ]

    base = function(point)

    for i in range(dimension):
        plus = list(point)
        minus = list(point)
        plus[i] += h
        minus[i] -= h

        hessian[i][i] = (
            function(plus) - 2 * base + function(minus)
        ) / h**2

    for i in range(dimension):
        for j in range(i + 1, dimension):
            pp = list(point)
            pm = list(point)
            mp = list(point)
            mm = list(point)

            pp[i] += h
            pp[j] += h

            pm[i] += h
            pm[j] -= h

            mp[i] -= h
            mp[j] += h

            mm[i] -= h
            mm[j] -= h

            mixed = (
                function(pp)
                - function(pm)
                - function(mp)
                + function(mm)
            ) / (4 * h**2)

            hessian[i][j] = mixed
            hessian[j][i] = mixed

    return hessian


print(
    "Numerical Hessian at [3,-1]:",
    numerical_hessian(quadratic_objective, [3.0, -1.0]),
)


# =============================================================================
# 17. DISCRETE OPTIMIZATION
# =============================================================================

section("DISCRETE OPTIMIZATION")

subsection("17.1 Knapsack dynamic programming")

def zero_one_knapsack(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> tuple[int, list[int]]:
    if len(weights) != len(values):
        raise ValueError("Weights and values must have equal length.")

    if capacity < 0:
        raise ValueError("Capacity cannot be negative.")

    item_count = len(weights)

    if any(weight <= 0 for weight in weights):
        raise ValueError("Weights must be positive.")

    dp = [
        [0] * (capacity + 1)
        for _ in range(item_count + 1)
    ]

    for i in range(1, item_count + 1):
        weight = weights[i - 1]
        value = values[i - 1]

        for current_capacity in range(capacity + 1):
            dp[i][current_capacity] = dp[i - 1][current_capacity]

            if weight <= current_capacity:
                candidate = (
                    dp[i - 1][current_capacity - weight] + value
                )
                dp[i][current_capacity] = max(
                    dp[i][current_capacity],
                    candidate,
                )

    selected_items = []
    current_capacity = capacity

    for i in range(item_count, 0, -1):
        if dp[i][current_capacity] != dp[i - 1][current_capacity]:
            selected_items.append(i - 1)
            current_capacity -= weights[i - 1]

    selected_items.reverse()

    return dp[item_count][capacity], selected_items


weights = [2, 3, 4, 5]
values = [3, 4, 5, 8]

maximum_value, selected = zero_one_knapsack(weights, values, 8)

print("Maximum value:", maximum_value)
print("Selected item indices:", selected)


# =============================================================================
# 18. EDGE CASES
# =============================================================================

section("EDGE CASES AND COMMON FAILURES")

subsection("18.1 Empty and invalid inputs")

for invalid_input in [
    [],
    [1],
]:
    try:
        if len(invalid_input) < 2:
            raise ValueError("At least two observations are needed.")
    except ValueError as error:
        print("Handled:", error)


subsection("18.2 Division by zero")

try:
    result = 10 / 0
except ZeroDivisionError as error:
    print("Handled division error:", error)


subsection("18.3 Singular linear systems")

try:
    solve_linear_system(
        [[1, 2], [2, 4]],
        [3, 6],
    )
except ValueError as error:
    print("Handled singular matrix:", error)


# =============================================================================
# 19. COMPLEXITY REFERENCE
# =============================================================================

section("ALGORITHMIC COMPLEXITY")

print(
    """
Examples in this script:

Vector addition:
    O(n)

Dot product:
    O(n)

Naive matrix multiplication:
    O(n³) for square n×n matrices

Recursive Laplace determinant:
    Approximately O(n!) in the naive implementation

Gaussian elimination:
    O(n³)

Power iteration:
    O(k * n²) for k iterations on a dense n×n matrix

Numerical gradient:
    O(d) function evaluations per gradient for d variables

Gradient descent:
    O(k * d) gradient work for k iterations when objective evaluation
    is O(d)

Trapezoidal integration:
    O(n) function evaluations

0/1 knapsack dynamic programming:
    O(number_of_items * capacity)

Algorithmic complexity is separate from numerical stability. An algorithm
can be computationally efficient yet numerically unstable.
"""
)


# =============================================================================
# 20. INTEGRATED FINANCIAL-STYLE EXAMPLE
# =============================================================================

section("INTEGRATED APPLICATION: DISCOUNTED CASH FLOW")

def present_value(
    cash_flow: float,
    discount_rate: float,
    period: int,
) -> float:
    if discount_rate <= -1:
        raise ValueError("Discount rate must be greater than -100%.")
    if period < 0:
        raise ValueError("Period cannot be negative.")

    return cash_flow / (1 + discount_rate) ** period


def net_present_value(
    cash_flows: Sequence[float],
    discount_rate: float,
) -> float:
    return sum(
        present_value(cash_flow, discount_rate, period)
        for period, cash_flow in enumerate(cash_flows)
    )


cash_flows = [-1000, 250, 300, 400, 500]

print("NPV at 10%:", net_present_value(cash_flows, 0.10))


# =============================================================================
# 21. ROOT FINDING FOR INTERNAL RATE OF RETURN
# =============================================================================

section("ROOT FINDING + FINANCE")

def npv_at_rate(
    cash_flows: Sequence[float],
    rate: float,
) -> float:
    return net_present_value(cash_flows, rate)


def bisection_root(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    tolerance: float = 1e-10,
    iterations: int = 200,
) -> float:
    f_lower = function(lower)
    f_upper = function(upper)

    if f_lower == 0:
        return lower

    if f_upper == 0:
        return upper

    if f_lower * f_upper > 0:
        raise ValueError("Function must have opposite signs at the bounds.")

    for _ in range(iterations):
        midpoint = (lower + upper) / 2
        f_midpoint = function(midpoint)

        if abs(f_midpoint) < tolerance or abs(upper - lower) < tolerance:
            return midpoint

        if f_lower * f_midpoint < 0:
            upper = midpoint
            f_upper = f_midpoint
        else:
            lower = midpoint
            f_lower = f_midpoint

    return (lower + upper) / 2


irr = bisection_root(
    lambda rate: npv_at_rate(cash_flows, rate),
    0.0,
    1.0,
)

print("Approximate IRR:", irr)
print("NPV at IRR:", npv_at_rate(cash_flows, irr))


# =============================================================================
# 22. CHECKS AND REPRODUCIBILITY
# =============================================================================

section("VALIDATION CHECKS")

assert math.isclose(quadratic(2, 1, -3, 2), 0.0)
assert math.isclose(
    expected_value(die_values, die_probabilities),
    3.5,
)
assert math.isclose(determinant([[1, 2], [3, 4]]), -2.0)

solution = solve_linear_system([[2, 1], [1, -1]], [7, 1])
assert math.isclose(solution[0], 8 / 3)
assert math.isclose(solution[1], 5 / 3)

assert math.isclose(
    simpson_integral(lambda x: x * x, 0, 3),
    9.0,
    rel_tol=1e-8,
)

assert math.isclose(
    quadratic_objective(minimum),
    0.0,
    abs_tol=1e-4,
)

assert math.isclose(root, math.sqrt(2), rel_tol=1e-8)

print("All validation checks passed.")


# =============================================================================
# 23. FINAL STUDY REFERENCE
# =============================================================================

section("CORE FORMULAS")

print(
    """
PROBABILITY
-----------
P(A^c) = 1 - P(A)
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
P(A|B) = P(A ∩ B) / P(B)
P(A ∩ B) = P(A|B)P(B)
P(A) = Σ P(A|Bi)P(Bi)
P(B|A) = P(A|B)P(B) / P(A)

Binomial:
P(X=k) = C(n,k)p^k(1-p)^(n-k)
E[X] = np
Var(X) = np(1-p)

STATISTICS
----------
Mean = Σxi / n
Sample variance = Σ(xi-x̄)² / (n-1)
Population variance = Σ(xi-μ)² / N
z = (x-μ)/σ
Correlation = covariance / (sx sy)

LINEAR ALGEBRA
--------------
Ax = b
(A+B)ij = Aij + Bij
(AB)ij = Σ Aik Bkj
Aᵀ = transpose(A)
For a 2×2 matrix:
det([[a,b],[c,d]]) = ad-bc

CALCULUS
--------
Derivative:
f'(x) = lim(h→0)[f(x+h)-f(x)]/h

Product rule:
(fg)' = f'g + fg'

Chain rule:
(f(g(x)))' = f'(g(x))g'(x)

Fundamental theorem:
∫a^b f(x)dx = F(b)-F(a), where F'=f

OPTIMIZATION
------------
Gradient:
∇f = [∂f/∂x1, ..., ∂f/∂xn]

Gradient descent:
x_(k+1) = x_k - α∇f(x_k)

Newton's method:
x_(k+1) = x_k - f(x_k)/f'(x_k)

A local unconstrained optimum commonly satisfies:
∇f(x*) = 0

For a twice-differentiable function, the Hessian provides second-order
curvature information.
"""
)

print("\nMathematical revision demonstrations completed.")
