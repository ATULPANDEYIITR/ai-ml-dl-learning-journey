"""
NumPy Advanced Study Script
===========================

Topic:
    Vectorization, matrix operations, random numbers, and linear algebra operations

Purpose:
    A standalone, executable study file that progresses from NumPy fundamentals
    to advanced numerical computing techniques.

Requirements:
    Python 3.9+
    NumPy

Run:
    python numpy_advanced.py

The examples intentionally use deterministic random seeds where reproducibility
matters. The script prints results so that each concept can be observed directly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


# =============================================================================
# Utility functions
# =============================================================================

def section(title: str) -> None:
    """Print a readable section heading."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def subsection(title: str) -> None:
    """Print a smaller subsection heading."""
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def show(name: str, value: object) -> None:
    """Print a named value."""
    print(f"{name}:\n{value}")


def assert_close(actual: np.ndarray, expected: np.ndarray, message: str = "") -> None:
    """Validate floating-point arrays using NumPy's tolerance-aware comparison."""
    if not np.allclose(actual, expected):
        raise AssertionError(message or f"Arrays differ:\n{actual}\n{expected}")


# =============================================================================
# NumPy fundamentals
# =============================================================================

def fundamentals() -> None:
    section("1. NumPy fundamentals")

    subsection("Creating arrays")

    # A NumPy ndarray stores homogeneous numerical data efficiently.
    vector = np.array([10, 20, 30, 40])
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])

    show("vector", vector)
    show("matrix", matrix)
    show("vector.shape", vector.shape)
    show("matrix.shape", matrix.shape)
    show("matrix.ndim", matrix.ndim)
    show("matrix.size", matrix.size)
    show("matrix.dtype", matrix.dtype)

    subsection("Common constructors")

    show("zeros", np.zeros((2, 3)))
    show("ones", np.ones((2, 3)))
    show("full", np.full((2, 3), 7))
    show("identity matrix", np.eye(3))
    show("arange", np.arange(0, 10, 2))
    show("linspace", np.linspace(0, 1, 5))

    # reshape changes the view/shape when possible without changing the data.
    values = np.arange(12)
    reshaped = values.reshape(3, 4)
    show("reshaped", reshaped)

    subsection("Indexing and slicing")

    data = np.arange(1, 13).reshape(3, 4)
    show("data", data)
    show("first row", data[0])
    show("last column", data[:, -1])
    show("submatrix", data[0:2, 1:3])

    # Boolean indexing selects elements according to a condition.
    even_values = data[data % 2 == 0]
    show("even values", even_values)

    subsection("Data types")

    integers = np.array([1, 2, 3], dtype=np.int64)
    floating = np.array([1, 2, 3], dtype=np.float64)
    compact = np.array([1, 2, 3], dtype=np.int32)

    show("integers dtype", integers.dtype)
    show("floating dtype", floating.dtype)
    show("compact dtype", compact.dtype)

    # Explicit dtypes can reduce memory usage when the numerical range permits.
    show("integer nbytes", integers.nbytes)
    show("int32 nbytes", compact.nbytes)

    subsection("Copy versus view")

    original = np.array([1, 2, 3, 4])
    view = original[1:3]
    view[0] = 99

    show("original after modifying view", original)

    original = np.array([1, 2, 3, 4])
    copied = original[1:3].copy()
    copied[0] = 88

    show("original after modifying copy", original)
    show("copy", copied)


# =============================================================================
# Vectorization
# =============================================================================

def vectorization() -> None:
    section("2. Vectorization")

    subsection("Element-wise arithmetic")

    x = np.array([1, 2, 3, 4, 5], dtype=float)

    # NumPy applies operations to complete arrays without an explicit Python loop.
    show("x + 10", x + 10)
    show("x * 2", x * 2)
    show("x ** 2", x ** 2)
    show("sqrt(x)", np.sqrt(x))
    show("sin(x)", np.sin(x))

    subsection("Vectorized mathematical expressions")

    temperatures_celsius = np.array([0, 10, 20, 30, 40], dtype=float)

    # Conversion is expressed as an operation over the entire array.
    temperatures_fahrenheit = temperatures_celsius * 9 / 5 + 32
    show("Celsius", temperatures_celsius)
    show("Fahrenheit", temperatures_fahrenheit)

    subsection("Vectorized conditional logic")

    scores = np.array([45, 67, 82, 91, 38, 74])

    # np.where provides vectorized if/else behavior.
    status = np.where(scores >= 50, "Pass", "Fail")
    show("scores", scores)
    show("status", status)

    # np.select handles multiple conditions.
    grades = np.select(
        [
            scores >= 90,
            scores >= 75,
            scores >= 60,
            scores >= 50,
        ],
        [
            "A",
            "B",
            "C",
            "D",
        ],
        default="F",
    )
    show("grades", grades)

    subsection("Aggregation")

    values = np.array([
        [10, 20, 30],
        [40, 50, 60],
    ])

    show("values", values)
    show("total", np.sum(values))
    show("column sums", np.sum(values, axis=0))
    show("row sums", np.sum(values, axis=1))
    show("column means", np.mean(values, axis=0))
    show("row means", np.mean(values, axis=1))
    show("minimum", np.min(values))
    show("maximum", np.max(values))
    show("standard deviation", np.std(values))

    subsection("Performance comparison")

    # This benchmark demonstrates the principle of vectorization.
    # Exact timings vary by CPU, Python version, NumPy build, and system load.
    n = 1_000_000
    python_values = list(range(n))

    start = time.perf_counter()
    python_result = [value * value + 2 * value + 1 for value in python_values]
    python_time = time.perf_counter() - start

    numpy_values = np.arange(n)

    start = time.perf_counter()
    numpy_result = numpy_values * numpy_values + 2 * numpy_values + 1
    numpy_time = time.perf_counter() - start

    assert np.array_equal(
        np.array(python_result),
        numpy_result,
    )

    show("Python loop time (seconds)", python_time)
    show("NumPy vectorized time (seconds)", numpy_time)
    show("speedup estimate", python_time / numpy_time if numpy_time else math.inf)


# =============================================================================
# Broadcasting
# =============================================================================

def broadcasting() -> None:
    section("3. Broadcasting")

    subsection("Scalar broadcasting")

    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])

    show("matrix + 10", matrix + 10)

    subsection("Row-wise broadcasting")

    row = np.array([10, 20, 30])
    result = matrix + row

    show("matrix", matrix)
    show("row", row)
    show("matrix + row", result)

    subsection("Column-wise broadcasting")

    column = np.array([[100], [200]])
    result = matrix + column

    show("column", column)
    show("matrix + column", result)

    subsection("Broadcasting rules")

    compatible_a = np.zeros((3, 1))
    compatible_b = np.zeros((1, 4))
    compatible_result = compatible_a + compatible_b

    show("shape A", compatible_a.shape)
    show("shape B", compatible_b.shape)
    show("broadcast result shape", compatible_result.shape)

    # Dimensions are compatible when they are equal or one of them is 1.
    # Dimensions are aligned from the rightmost side.

    subsection("A broadcasting failure")

    incompatible_a = np.zeros((3, 2))
    incompatible_b = np.zeros((4,))

    try:
        incompatible_a + incompatible_b
    except ValueError as error:
        print("Expected broadcasting error:", error)


# =============================================================================
# Universal functions and numerical operations
# =============================================================================

def universal_functions() -> None:
    section("4. Universal functions and numerical operations")

    values = np.array([-4.5, -1.2, 0.0, 1.2, 4.5])

    show("absolute", np.abs(values))
    show("floor", np.floor(values))
    show("ceil", np.ceil(values))
    show("round", np.round(values))
    show("sign", np.sign(values))

    positive_values = np.array([1.0, 10.0, 100.0])
    show("natural log", np.log(positive_values))
    show("base-10 log", np.log10(positive_values))
    show("base-2 log", np.log2(positive_values))
    show("exponential", np.exp(positive_values))

    subsection("Stable numerical functions")

    # log1p(x) is more accurate than log(1 + x) for very small x.
    tiny = 1e-12
    direct = np.log(1 + tiny)
    stable = np.log1p(tiny)

    show("log(1 + tiny)", direct)
    show("log1p(tiny)", stable)

    # expm1(x) is more accurate than exp(x) - 1 for small x.
    direct_exponential = np.exp(tiny) - 1
    stable_exponential = np.expm1(tiny)

    show("exp(tiny) - 1", direct_exponential)
    show("expm1(tiny)", stable_exponential)

    subsection("Clipping")

    noisy = np.array([-10, 2, 7, 15])
    show("original", noisy)
    show("clipped to [0, 10]", np.clip(noisy, 0, 10))

    subsection("NaN and infinity")

    special = np.array([1.0, np.nan, np.inf, -np.inf])

    show("isnan", np.isnan(special))
    show("isfinite", np.isfinite(special))
    show("nan-aware mean", np.nanmean(special[np.isfinite(special)]))


# =============================================================================
# Matrix operations
# =============================================================================

def matrix_operations() -> None:
    section("5. Matrix operations")

    A = np.array([
        [1, 2],
        [3, 4],
    ])

    B = np.array([
        [5, 6],
        [7, 8],
    ])

    subsection("Element-wise multiplication")

    show("A * B", A * B)

    subsection("Matrix multiplication")

    # @ performs matrix multiplication.
    show("A @ B", A @ B)

    # np.matmul is equivalent for ordinary matrix multiplication.
    show("np.matmul(A, B)", np.matmul(A, B))

    subsection("Transpose")

    show("A.T", A.T)
    show("np.transpose(A)", np.transpose(A))

    subsection("Trace and diagonal")

    show("diagonal", np.diag(A))
    show("trace", np.trace(A))

    subsection("Matrix powers")

    show("A @ A", A @ A)
    show("np.linalg.matrix_power(A, 3)", np.linalg.matrix_power(A, 3))

    subsection("Dot product")

    vector_a = np.array([1, 2, 3])
    vector_b = np.array([4, 5, 6])

    show("dot product", np.dot(vector_a, vector_b))
    show("vector @ vector", vector_a @ vector_b)

    subsection("Outer product")

    show("outer product", np.outer(vector_a, vector_b))

    subsection("Matrix dimensions")

    matrix_2x3 = np.arange(6).reshape(2, 3)
    matrix_3x4 = np.arange(12).reshape(3, 4)

    product = matrix_2x3 @ matrix_3x4

    show("2x3 matrix", matrix_2x3)
    show("3x4 matrix", matrix_3x4)
    show("product shape", product.shape)
    show("product", product)

    subsection("Stacking and concatenation")

    first = np.array([[1, 2], [3, 4]])
    second = np.array([[5, 6], [7, 8]])

    show("vstack", np.vstack([first, second]))
    show("hstack", np.hstack([first, second]))
    show("concatenate axis=0", np.concatenate([first, second], axis=0))
    show("concatenate axis=1", np.concatenate([first, second], axis=1))

    subsection("einsum")

    # Einstein summation expresses many tensor operations compactly.
    # Matrix multiplication C[i,j] = sum_k A[i,k] * B[k,j].
    einsum_product = np.einsum("ik,kj->ij", A, B)

    assert np.array_equal(einsum_product, A @ B)
    show("einsum matrix product", einsum_product)

    # Dot product using Einstein notation.
    einsum_dot = np.einsum("i,i->", vector_a, vector_b)
    show("einsum dot product", einsum_dot)


# =============================================================================
# Advanced indexing and manipulation
# =============================================================================

def advanced_indexing() -> None:
    section("6. Advanced indexing and array manipulation")

    matrix = np.arange(1, 17).reshape(4, 4)
    show("matrix", matrix)

    subsection("Boolean masks")

    mask = matrix > 10
    show("mask", mask)
    show("values > 10", matrix[mask])

    subsection("Combining masks")

    selected = matrix[(matrix % 2 == 0) & (matrix > 5)]
    show("even values greater than 5", selected)

    # Use & and | for array conditions.
    # Python's 'and' and 'or' do not perform element-wise boolean operations.

    subsection("Fancy indexing")

    rows = np.array([0, 2, 3])
    columns = np.array([1, 3, 0])

    show("selected elements", matrix[rows, columns])

    subsection("Where indices occur")

    positions = np.where(matrix > 10)
    show("row indices", positions[0])
    show("column indices", positions[1])

    subsection("Unique values")

    repeated = np.array([1, 2, 2, 3, 3, 3, 4])
    unique, counts = np.unique(repeated, return_counts=True)

    show("unique values", unique)
    show("counts", counts)

    subsection("Sorting")

    unsorted = np.array([7, 2, 9, 1, 5])
    show("sorted", np.sort(unsorted))
    show("sort indices", np.argsort(unsorted))

    subsection("Argmin and argmax")

    values = np.array([15, 7, 23, 4, 18])

    show("argmin", np.argmin(values))
    show("minimum", values[np.argmin(values)])
    show("argmax", np.argmax(values))
    show("maximum", values[np.argmax(values)])

    subsection("Flattening")

    show("flatten", matrix.flatten())
    show("ravel", matrix.ravel())


# =============================================================================
# Random numbers
# =============================================================================

def random_numbers() -> None:
    section("7. Random numbers")

    subsection("Modern random number generator")

    # np.random.default_rng creates the modern Generator interface.
    rng = np.random.default_rng(42)

    show("random floats", rng.random(5))
    show("random integers", rng.integers(1, 11, size=10))
    show("uniform distribution", rng.uniform(0, 1, size=5))
    show("normal distribution", rng.normal(loc=0, scale=1, size=5))

    subsection("Reproducibility")

    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(12345)

    sample_a = rng_a.normal(size=10)
    sample_b = rng_b.normal(size=10)

    assert np.array_equal(sample_a, sample_b)
    show("reproducible sample", sample_a)

    subsection("Different distributions")

    rng = np.random.default_rng(7)

    show("binomial", rng.binomial(n=10, p=0.5, size=10))
    show("Poisson", rng.poisson(lam=4, size=10))
    show("exponential", rng.exponential(scale=2.0, size=10))
    show("uniform integers", rng.integers(0, 100, size=10))

    subsection("Random sampling")

    population = np.arange(1, 21)

    show(
        "sample without replacement",
        rng.choice(population, size=5, replace=False),
    )

    show(
        "sample with replacement",
        rng.choice(population, size=10, replace=True),
    )

    subsection("Random permutations")

    original = np.arange(10)
    shuffled = rng.permutation(original)

    show("original", original)
    show("permutation", shuffled)

    subsection("Monte Carlo estimation of pi")

    # Generate random points in the unit square.
    # A point lies inside the quarter circle when x^2 + y^2 <= 1.
    samples = 1_000_000
    points = rng.random((samples, 2))

    inside = np.sum(np.sum(points ** 2, axis=1) <= 1)
    estimated_pi = 4 * inside / samples

    show("samples", samples)
    show("estimated pi", estimated_pi)
    show("actual pi", np.pi)


# =============================================================================
# Probability and statistics with NumPy
# =============================================================================

def statistics() -> None:
    section("8. Statistical operations")

    data = np.array([12, 15, 14, 10, 18, 21, 16, 14, 13, 17], dtype=float)

    show("data", data)
    show("mean", np.mean(data))
    show("median", np.median(data))
    show("variance", np.var(data))
    show("standard deviation", np.std(data))
    show("percentile 25", np.percentile(data, 25))
    show("percentile 75", np.percentile(data, 75))

    subsection("Correlation")

    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 5, 8, 10], dtype=float)

    correlation_matrix = np.corrcoef(x, y)

    show("correlation matrix", correlation_matrix)
    show("correlation coefficient", correlation_matrix[0, 1])

    subsection("Covariance")

    covariance_matrix = np.cov(x, y)
    show("covariance matrix", covariance_matrix)


# =============================================================================
# Linear algebra basics
# =============================================================================

def linear_algebra_basics() -> None:
    section("9. Linear algebra basics")

    A = np.array([
        [4.0, 7.0],
        [2.0, 6.0],
    ])

    subsection("Determinant")

    determinant = np.linalg.det(A)
    show("A", A)
    show("det(A)", determinant)

    subsection("Inverse")

    inverse = np.linalg.inv(A)
    show("A inverse", inverse)

    identity = A @ inverse
    show("A @ A inverse", identity)

    assert_close(identity, np.eye(2), "A multiplied by its inverse should be I.")

    subsection("Solving a linear system")

    coefficients = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
    ])

    right_side = np.array([8.0, 13.0])

    # Solve A x = b directly instead of computing inv(A) @ b.
    solution = np.linalg.solve(coefficients, right_side)

    show("coefficient matrix", coefficients)
    show("right side", right_side)
    show("solution", solution)
    show("verification", coefficients @ solution)

    assert_close(coefficients @ solution, right_side)

    subsection("Least-squares solution")

    # The system has more equations than unknowns.
    X = np.array([
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
    ], dtype=float)

    target = np.array([2.1, 4.0, 5.9, 8.2])

    least_squares_solution, residuals, rank, singular_values = np.linalg.lstsq(
        X,
        target,
        rcond=None,
    )

    show("least-squares coefficients", least_squares_solution)
    show("residuals", residuals)
    show("rank", rank)
    show("singular values", singular_values)

    predictions = X @ least_squares_solution
    show("predictions", predictions)


# =============================================================================
# Eigenvalues and eigenvectors
# =============================================================================

def eigen_analysis() -> None:
    section("10. Eigenvalues and eigenvectors")

    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    eigenvalues, eigenvectors = np.linalg.eig(A)

    show("A", A)
    show("eigenvalues", eigenvalues)
    show("eigenvectors", eigenvectors)

    # Each column v of eigenvectors satisfies A @ v = lambda * v.
    for index in range(len(eigenvalues)):
        eigenvalue = eigenvalues[index]
        eigenvector = eigenvectors[:, index]

        left = A @ eigenvector
        right = eigenvalue * eigenvector

        assert_close(
            left,
            right,
            "Eigenvector equation was not satisfied.",
        )

    subsection("Symmetric matrix eigen-analysis")

    symmetric = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    # eigh is preferred for real symmetric or Hermitian matrices.
    symmetric_values, symmetric_vectors = np.linalg.eigh(symmetric)

    show("symmetric matrix", symmetric)
    show("eigenvalues", symmetric_values)
    show("eigenvectors", symmetric_vectors)


# =============================================================================
# Singular Value Decomposition
# =============================================================================

def singular_value_decomposition() -> None:
    section("11. Singular Value Decomposition")

    A = np.array([
        [3.0, 1.0],
        [1.0, 3.0],
        [2.0, 2.0],
    ])

    U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)

    show("A", A)
    show("U", U)
    show("singular values", singular_values)
    show("Vt", Vt)

    # SVD factorization:
    # A = U @ diag(s) @ Vt
    reconstructed = U @ np.diag(singular_values) @ Vt

    assert_close(reconstructed, A)
    show("reconstructed A", reconstructed)

    subsection("Low-rank approximation")

    # Retain only the largest singular value.
    rank_one = U[:, :1] @ np.diag(singular_values[:1]) @ Vt[:1, :]

    show("rank-one approximation", rank_one)
    show("Frobenius reconstruction error", np.linalg.norm(A - rank_one))


# =============================================================================
# Norms and distances
# =============================================================================

def norms_and_distances() -> None:
    section("12. Norms and distances")

    vector = np.array([3.0, 4.0])

    show("L1 norm", np.linalg.norm(vector, ord=1))
    show("L2 norm", np.linalg.norm(vector, ord=2))
    show("L-infinity norm", np.linalg.norm(vector, ord=np.inf))

    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    show("matrix Frobenius norm", np.linalg.norm(matrix, ord="fro"))
    show("matrix spectral norm", np.linalg.norm(matrix, ord=2))

    subsection("Euclidean distance")

    point_a = np.array([1.0, 2.0, 3.0])
    point_b = np.array([4.0, 6.0, 3.0])

    distance = np.linalg.norm(point_a - point_b)

    show("distance", distance)


# =============================================================================
# Matrix decompositions
# =============================================================================

def matrix_decompositions() -> None:
    section("13. Matrix decompositions")

    subsection("QR decomposition")

    A = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ])

    Q, R = np.linalg.qr(A)

    show("Q", Q)
    show("R", R)

    assert_close(Q @ R, A)
    show("Q.T @ Q", Q.T @ Q)

    subsection("Cholesky decomposition")

    positive_definite = np.array([
        [4.0, 2.0],
        [2.0, 3.0],
    ])

    L = np.linalg.cholesky(positive_definite)

    show("positive definite matrix", positive_definite)
    show("Cholesky factor L", L)

    assert_close(L @ L.T, positive_definite)

    subsection("Rank")

    rank = np.linalg.matrix_rank(A)
    show("rank", rank)


# =============================================================================
# Condition numbers and numerical stability
# =============================================================================

def numerical_stability() -> None:
    section("14. Numerical stability and conditioning")

    subsection("Condition number")

    well_conditioned = np.eye(2)
    ill_conditioned = np.array([
        [1.0, 1.0],
        [1.0, 1.000001],
    ])

    show(
        "condition number of identity",
        np.linalg.cond(well_conditioned),
    )

    show(
        "condition number of nearly dependent matrix",
        np.linalg.cond(ill_conditioned),
    )

    subsection("Why solve is preferred over explicit inverse")

    A = np.array([
        [3.0, 1.0],
        [1.0, 2.0],
    ])

    b = np.array([9.0, 8.0])

    solution_with_solve = np.linalg.solve(A, b)
    solution_with_inverse = np.linalg.inv(A) @ b

    assert_close(solution_with_solve, solution_with_inverse)

    show("solve result", solution_with_solve)
    show("inverse multiplication result", solution_with_inverse)

    print(
        "For ordinary linear systems, np.linalg.solve is generally preferred "
        "because it avoids explicitly constructing the inverse."
    )

    subsection("Floating-point comparison")

    a = np.array([0.1 + 0.2])
    b = np.array([0.3])

    show("0.1 + 0.2", a)
    show("0.3", b)
    show("exact equality", a == b)
    show("allclose", np.allclose(a, b))


# =============================================================================
# Linear regression implemented with NumPy
# =============================================================================

def linear_regression() -> None:
    section("15. Linear regression with NumPy")

    # Model:
    # y = beta_0 + beta_1*x
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([3.1, 5.0, 7.2, 8.9, 11.1], dtype=float)

    # Add a column of ones for the intercept.
    X = np.column_stack([np.ones_like(x), x])

    coefficients, residuals, rank, singular_values = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    intercept, slope = coefficients
    predictions = X @ coefficients
    errors = y - predictions

    show("intercept", intercept)
    show("slope", slope)
    show("predictions", predictions)
    show("errors", errors)
    show("sum of squared errors", np.sum(errors ** 2))

    subsection("R-squared")

    ss_res = np.sum((y - predictions) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - ss_res / ss_total
    show("R-squared", r_squared)


# =============================================================================
# Pairwise distances with vectorization
# =============================================================================

def pairwise_distances() -> None:
    section("16. Vectorized pairwise distances")

    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 2.0],
        [3.0, 4.0],
    ])

    # points[:, None, :] has shape (n, 1, dimensions)
    # points[None, :, :] has shape (1, n, dimensions)
    # Broadcasting creates all point-to-point differences.
    differences = points[:, None, :] - points[None, :, :]
    distance_matrix = np.linalg.norm(differences, axis=2)

    show("points", points)
    show("pairwise distance matrix", distance_matrix)

    # The diagonal should be zero.
    assert_close(np.diag(distance_matrix), np.zeros(len(points)))

    subsection("Squared Euclidean distances")

    squared_distances = np.sum(differences ** 2, axis=2)
    show("squared distances", squared_distances)


# =============================================================================
# Sliding windows and stride tricks
# =============================================================================

def sliding_windows() -> None:
    section("17. Sliding windows")

    values = np.arange(1, 11)

    # A simple moving average can be implemented with convolution.
    window_size = 3
    moving_average = np.convolve(
        values,
        np.ones(window_size) / window_size,
        mode="valid",
    )

    show("values", values)
    show("moving average", moving_average)

    subsection("Sliding windows with stride tricks")

    # sliding_window_view creates overlapping views.
    # It can be efficient, but the resulting arrays can become very large.
    windows = np.lib.stride_tricks.sliding_window_view(values, window_size)

    show("windows", windows)
    show("window means", windows.mean(axis=1))

    assert_close(windows.mean(axis=1), moving_average)


# =============================================================================
# Memory layout and performance
# =============================================================================

def memory_and_performance() -> None:
    section("18. Memory layout and performance")

    subsection("Contiguous arrays")

    array = np.arange(12).reshape(3, 4)

    show("C-contiguous", array.flags["C_CONTIGUOUS"])
    show("F-contiguous", array.flags["F_CONTIGUOUS"])

    transposed = array.T

    show("transposed C-contiguous", transposed.flags["C_CONTIGUOUS"])
    show("transposed F-contiguous", transposed.flags["F_CONTIGUOUS"])

    subsection("Memory sharing")

    base = np.arange(10)
    sliced = base[2:7]
    copied = sliced.copy()

    show("sliced shares memory", np.shares_memory(base, sliced))
    show("copied shares memory", np.shares_memory(base, copied))

    subsection("Avoiding unnecessary temporary arrays")

    large = np.arange(1_000_000, dtype=np.float64)

    # This creates a temporary array for large + 10 before multiplication.
    expression_result = (large + 10) * 2

    # out= allows NumPy to write directly into a preallocated destination.
    destination = np.empty_like(large)
    np.add(large, 10, out=destination)
    np.multiply(destination, 2, out=destination)

    assert_close(expression_result, destination)

    show("preallocated result first five", destination[:5])


# =============================================================================
# In-place operations
# =============================================================================

def in_place_operations() -> None:
    section("19. In-place operations")

    values = np.array([1.0, 2.0, 3.0])

    values *= 10
    show("after *= 10", values)

    values += 5
    show("after += 5", values)

    subsection("In-place operation caveat")

    integer_values = np.array([1, 2, 3], dtype=np.int32)

    try:
        # Adding floating-point values in-place cannot safely cast the result
        # back into int32 under NumPy's default casting rules.
        integer_values += 0.5
    except TypeError as error:
        print("Expected dtype/casting error:", error)


# =============================================================================
# Structured arrays
# =============================================================================

def structured_arrays() -> None:
    section("20. Structured arrays")

    employee_dtype = np.dtype([
        ("name", "U20"),
        ("age", "i4"),
        ("salary", "f8"),
    ])

    employees = np.array([
        ("Asha", 28, 65000.0),
        ("Rahul", 32, 82000.0),
        ("Meera", 26, 58000.0),
    ], dtype=employee_dtype)

    show("employees", employees)
    show("names", employees["name"])
    show("salaries", employees["salary"])

    high_salary = employees[employees["salary"] > 60000]
    show("employees with salary > 60000", high_salary)


# =============================================================================
# Masked arrays
# =============================================================================

def masked_arrays() -> None:
    section("21. Masked arrays")

    measurements = np.array([10.0, 12.0, -999.0, 15.0, 18.0])

    # -999 represents a missing/invalid measurement.
    masked = np.ma.masked_equal(measurements, -999.0)

    show("original", measurements)
    show("masked array", masked)
    show("masked mean", np.ma.mean(masked))


# =============================================================================
# Polynomial calculations
# =============================================================================

def polynomial_operations() -> None:
    section("22. Polynomial calculations")

    # Coefficients are ordered from highest degree to constant term.
    coefficients = np.array([2.0, -3.0, 1.0])

    polynomial = np.poly1d(coefficients)

    show("polynomial", polynomial)
    show("value at x=5", polynomial(5))
    show("roots", np.roots(coefficients))

    subsection("Polynomial fitting")

    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = 2 * x ** 2 - 3 * x + 1

    fitted = np.polyfit(x, y, deg=2)
    show("fitted coefficients", fitted)


# =============================================================================
# Fourier transform
# =============================================================================

def fast_fourier_transform() -> None:
    section("23. Fast Fourier Transform")

    sample_count = 1024
    sampling_rate = 1000.0

    time_axis = np.arange(sample_count) / sampling_rate

    # A signal containing two frequencies.
    signal = (
        np.sin(2 * np.pi * 50 * time_axis)
        + 0.5 * np.sin(2 * np.pi * 120 * time_axis)
    )

    frequencies = np.fft.rfftfreq(sample_count, d=1 / sampling_rate)
    spectrum = np.abs(np.fft.rfft(signal))

    # Find the strongest frequency components.
    strongest_indices = np.argsort(spectrum)[-5:][::-1]

    show(
        "strongest frequencies",
        frequencies[strongest_indices],
    )

    show(
        "corresponding amplitudes",
        spectrum[strongest_indices],
    )


# =============================================================================
# Random linear algebra simulation
# =============================================================================

def random_linear_algebra_simulation() -> None:
    section("24. Random matrix simulation")

    rng = np.random.default_rng(2026)

    matrix = rng.normal(size=(5, 5))

    eigenvalues = np.linalg.eigvals(matrix)
    singular_values = np.linalg.svd(matrix, compute_uv=False)

    show("random matrix", matrix)
    show("eigenvalues", eigenvalues)
    show("singular values", singular_values)
    show("matrix rank", np.linalg.matrix_rank(matrix))
    show("condition number", np.linalg.cond(matrix))


# =============================================================================
# Portfolio mathematics example
# =============================================================================

@dataclass
class PortfolioResult:
    """Container for portfolio calculations."""

    expected_return: float
    volatility: float
    sharpe_ratio: float


def portfolio_analysis(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    weights: np.ndarray,
    risk_free_rate: float = 0.0,
) -> PortfolioResult:
    """
    Calculate portfolio expected return, volatility, and Sharpe ratio.

    Portfolio return:
        w^T r

    Portfolio variance:
        w^T Sigma w

    Portfolio volatility:
        sqrt(w^T Sigma w)
    """
    expected_return = float(weights @ expected_returns)
    variance = float(weights @ covariance_matrix @ weights)
    volatility = math.sqrt(max(variance, 0.0))

    if volatility == 0:
        sharpe_ratio = math.nan
    else:
        sharpe_ratio = (expected_return - risk_free_rate) / volatility

    return PortfolioResult(
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
    )


def portfolio_example() -> None:
    section("25. Portfolio calculations with matrix algebra")

    expected_returns = np.array([0.08, 0.12, 0.06])

    covariance_matrix = np.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.09, 0.002],
        [0.005, 0.002, 0.025],
    ])

    weights = np.array([0.4, 0.35, 0.25])

    result = portfolio_analysis(
        expected_returns,
        covariance_matrix,
        weights,
        risk_free_rate=0.03,
    )

    show("portfolio weights", weights)
    show("expected return", result.expected_return)
    show("volatility", result.volatility)
    show("Sharpe ratio", result.sharpe_ratio)

    # A portfolio's weights normally sum to one.
    assert np.isclose(np.sum(weights), 1.0)


# =============================================================================
# Image-like array processing without external image packages
# =============================================================================

def array_image_processing() -> None:
    section("26. Image-like array operations")

    # A grayscale image can be represented as a 2D array where each value
    # represents pixel intensity.
    image = np.array([
        [0, 10, 30, 50],
        [20, 40, 60, 80],
        [30, 50, 70, 90],
        [40, 60, 80, 100],
    ], dtype=np.uint8)

    show("image", image)

    # Thresholding converts the grayscale image into a binary mask.
    threshold = 50
    binary = image >= threshold

    show("binary mask", binary)

    # Normalize values into a floating-point [0, 1] range.
    normalized = image.astype(np.float64) / 255.0
    show("normalized image", normalized)

    # Contrast transformation.
    enhanced = np.clip((normalized - 0.5) * 2 + 0.5, 0, 1)
    show("contrast-adjusted image", enhanced)


# =============================================================================
# Batched matrix multiplication
# =============================================================================

def batched_matrix_multiplication() -> None:
    section("27. Batched matrix multiplication")

    rng = np.random.default_rng(11)

    batch_a = rng.normal(size=(4, 2, 3))
    batch_b = rng.normal(size=(4, 3, 5))

    # Each pair of matrices is multiplied independently.
    batch_result = np.matmul(batch_a, batch_b)

    show("batch A shape", batch_a.shape)
    show("batch B shape", batch_b.shape)
    show("batch result shape", batch_result.shape)

    # np.matmul treats leading dimensions as batch dimensions.
    for batch_index in range(4):
        expected = batch_a[batch_index] @ batch_b[batch_index]
        assert_close(batch_result[batch_index], expected)


# =============================================================================
# Generalized tensor operations
# =============================================================================

def tensor_operations() -> None:
    section("28. Tensor operations")

    tensor = np.arange(24).reshape(2, 3, 4)

    show("tensor shape", tensor.shape)
    show("tensor", tensor)

    subsection("Axis operations")

    show("sum over axis 0", tensor.sum(axis=0))
    show("sum over axis 1", tensor.sum(axis=1))
    show("sum over axis 2", tensor.sum(axis=2))

    subsection("Move axes")

    moved = np.moveaxis(tensor, 0, -1)

    show("moved shape", moved.shape)
    show("moved tensor", moved)

    subsection("Einstein summation for tensor contraction")

    # Tensor contraction over the final dimension.
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    weighted = np.einsum("ijk,k->ij", tensor, weights)

    expected = np.sum(tensor * weights, axis=2)

    assert_close(weighted, expected)

    show("weighted tensor contraction", weighted)


# =============================================================================
# Numerical edge cases
# =============================================================================

def edge_cases() -> None:
    section("29. Numerical edge cases")

    subsection("Division by zero")

    numerator = np.array([1.0, 0.0, -1.0])
    denominator = np.array([0.0, 0.0, 0.0])

    with np.errstate(divide="ignore", invalid="ignore"):
        division = numerator / denominator

    show("division result", division)
    show("isfinite", np.isfinite(division))

    subsection("Overflow handling")

    huge = np.array([1000.0])

    with np.errstate(over="ignore"):
        exponential = np.exp(huge)

    show("exp(1000)", exponential)

    subsection("Invalid logarithm")

    values = np.array([-1.0, 0.0, 1.0])

    with np.errstate(divide="ignore", invalid="ignore"):
        logarithms = np.log(values)

    show("logarithms", logarithms)

    subsection("Empty arrays")

    empty = np.array([], dtype=float)

    show("empty shape", empty.shape)
    show("empty size", empty.size)

    try:
        np.mean(empty)
    except RuntimeWarning:
        print("Empty mean generated a warning.")

    subsection("Near-singular matrix")

    near_singular = np.array([
        [1.0, 1.0],
        [1.0, 1.0 + 1e-12],
    ])

    show("rank", np.linalg.matrix_rank(near_singular))
    show("condition number", np.linalg.cond(near_singular))


# =============================================================================
# Testing numerical implementations
# =============================================================================

def testing_examples() -> None:
    section("30. Testing NumPy computations")

    def normalize(values: np.ndarray) -> np.ndarray:
        """Min-max normalize an array to [0, 1]."""
        values = np.asarray(values, dtype=float)

        minimum = np.min(values)
        maximum = np.max(values)

        if np.isclose(minimum, maximum):
            return np.zeros_like(values)

        return (values - minimum) / (maximum - minimum)

    sample = np.array([10, 20, 30, 40], dtype=float)

    expected = np.array([0.0, 1 / 3, 2 / 3, 1.0])

    actual = normalize(sample)

    assert_close(actual, expected)
    assert np.isclose(np.min(actual), 0.0)
    assert np.isclose(np.max(actual), 1.0)

    show("normalized", actual)

    subsection("Shape validation")

    def matrix_multiply_checked(
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """Perform matrix multiplication after validating dimensions."""
        left = np.asarray(left)
        right = np.asarray(right)

        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("Both inputs must be two-dimensional matrices.")

        if left.shape[1] != right.shape[0]:
            raise ValueError(
                f"Incompatible shapes: {left.shape} and {right.shape}"
            )

        return left @ right

    valid_a = np.ones((2, 3))
    valid_b = np.ones((3, 4))

    result = matrix_multiply_checked(valid_a, valid_b)

    assert result.shape == (2, 4)
    show("validated multiplication shape", result.shape)

    try:
        matrix_multiply_checked(np.ones((2, 3)), np.ones((2, 4)))
    except ValueError as error:
        print("Expected validation error:", error)


# =============================================================================
# A compact advanced numerical workflow
# =============================================================================

def advanced_workflow() -> None:
    section("31. Complete advanced numerical workflow")

    rng = np.random.default_rng(99)

    subsection("Generate synthetic data")

    observations = 500
    features = 4

    X = rng.normal(size=(observations, features))

    true_weights = np.array([1.5, -2.0, 0.75, 3.0])
    noise = rng.normal(scale=0.5, size=observations)

    y = X @ true_weights + noise

    show("X shape", X.shape)
    show("y shape", y.shape)

    subsection("Fit least-squares model")

    estimated_weights, residuals, rank, singular_values = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    predictions = X @ estimated_weights

    mse = np.mean((y - predictions) ** 2)
    rmse = np.sqrt(mse)

    show("true weights", true_weights)
    show("estimated weights", estimated_weights)
    show("RMSE", rmse)
    show("rank", rank)
    show("singular values", singular_values)

    subsection("Feature standardization")

    means = X.mean(axis=0)
    standard_deviations = X.std(axis=0)

    standardized_X = (X - means) / standard_deviations

    show("standardized feature means", standardized_X.mean(axis=0))
    show("standardized feature std", standardized_X.std(axis=0))

    subsection("Covariance matrix")

    covariance = np.cov(standardized_X, rowvar=False)

    show("covariance shape", covariance.shape)
    show("covariance matrix", covariance)


# =============================================================================
# Common mistakes
# =============================================================================

def common_mistakes() -> None:
    section("32. Common mistakes and corrections")

    subsection("Mistake: using * for matrix multiplication")

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    elementwise = A * B
    matrix_product = A @ B

    show("A * B", elementwise)
    show("A @ B", matrix_product)

    subsection("Mistake: ambiguous shape assumptions")

    one_dimensional = np.array([1, 2, 3])

    show("1D shape", one_dimensional.shape)

    column_vector = one_dimensional[:, None]
    row_vector = one_dimensional[None, :]

    show("column shape", column_vector.shape)
    show("row shape", row_vector.shape)

    subsection("Mistake: using Python and/or with arrays")

    boolean_values = np.array([True, False, True])

    try:
        # This raises an error because an array does not have one scalar truth value.
        _ = boolean_values and boolean_values
    except ValueError as error:
        print("Expected boolean-array error:", error)

    show(
        "correct element-wise AND",
        boolean_values & boolean_values,
    )

    subsection("Mistake: assuming equality is reliable for floats")

    a = np.array([1 / 3])
    b = np.array([0.3333333333333333])

    show("array_equal", np.array_equal(a, b))
    show("allclose", np.allclose(a, b))


# =============================================================================
# Production-oriented best practices
# =============================================================================

def production_best_practices() -> None:
    section("33. Production-oriented NumPy practices")

    print(
        "Use explicit shapes and dtypes when numerical correctness depends on them."
    )
    print(
        "Prefer vectorized operations, broadcasting, and NumPy ufuncs over Python loops."
    )
    print(
        "Use np.random.default_rng rather than relying on the legacy global random state."
    )
    print(
        "Use np.linalg.solve for linear systems instead of explicitly computing inverses."
    )
    print(
        "Use np.linalg.eigh for real symmetric or Hermitian matrices."
    )
    print(
        "Use np.allclose when validating floating-point numerical results."
    )
    print(
        "Avoid unnecessary copies and large temporary arrays when memory is constrained."
    )
    print(
        "Validate array shapes at API boundaries in reusable numerical code."
    )
    print(
        "Check NaN, infinity, rank, conditioning, and dtype behavior for production data."
    )
    print(
        "Benchmark realistic workloads rather than assuming every vectorized expression is optimal."
    )
    print(
        "Be aware that NumPy may use optimized native linear algebra libraries underneath."
    )


# =============================================================================
# Main execution
# =============================================================================

def main() -> None:
    """Run the complete NumPy advanced study program."""
    np.set_printoptions(
        precision=5,
        suppress=True,
        linewidth=120,
    )

    fundamentals()
    vectorization()
    broadcasting()
    universal_functions()
    matrix_operations()
    advanced_indexing()
    random_numbers()
    statistics()
    linear_algebra_basics()
    eigen_analysis()
    singular_value_decomposition()
    norms_and_distances()
    matrix_decompositions()
    numerical_stability()
    linear_regression()
    pairwise_distances()
    sliding_windows()
    memory_and_performance()
    in_place_operations()
    structured_arrays()
    masked_arrays()
    polynomial_operations()
    fast_fourier_transform()
    random_linear_algebra_simulation()
    portfolio_example()
    array_image_processing()
    batched_matrix_multiplication()
    tensor_operations()
    edge_cases()
    testing_examples()
    advanced_workflow()
    common_mistakes()
    production_best_practices()

    section("NumPy study script completed")
    print("All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
