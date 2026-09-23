"""
Eigenvalues and Eigenvectors
============================

A self-contained study and implementation file covering:

1. Vectors and matrices
2. Matrix-vector multiplication
3. Linear transformations
4. Eigenvalue and eigenvector definitions
5. Characteristic equations
6. Determinants and trace
7. Exact eigenvalue calculations for small matrices
8. Eigenvector calculation
9. Algebraic and geometric multiplicity
10. Diagonalization
11. Eigen decomposition
12. Symmetric matrices
13. Spectral theorem
14. Orthogonal eigenvectors
15. Matrix powers through eigen decomposition
16. Positive semidefinite matrices
17. Covariance matrices and PCA
18. Principal components
19. Dimensionality reduction
20. Reconstruction error
21. Power iteration
22. Rayleigh quotient
23. Deflation
24. PageRank-style eigenvector computation
25. Markov transition matrices
26. Graph adjacency matrices
27. Graph Laplacians
28. Numerical precision and residuals
29. Degenerate and repeated eigenvalues
30. Complex eigenvalues
31. Common mistakes and implementation checks

The implementations use only the Python standard library.
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


Number = float | complex
Vector = list[Number]
Matrix = list[list[Number]]

EPSILON = 1e-10


# ---------------------------------------------------------------------------
# Basic linear algebra
# ---------------------------------------------------------------------------

def is_close(a: Number, b: Number, tolerance: float = EPSILON) -> bool:
    """Return True when two real or complex numbers are approximately equal."""
    return abs(a - b) <= tolerance


def clean_number(value: Number, tolerance: float = 1e-12) -> Number:
    """
    Remove insignificant numerical noise.

    Numerical algorithms frequently produce values such as
    2.0000000000000004 or 1e-16j. This helper makes printed output readable.
    """
    if isinstance(value, complex):
        real = 0.0 if abs(value.real) < tolerance else value.real
        imag = 0.0 if abs(value.imag) < tolerance else value.imag
        if abs(imag) < tolerance:
            return real
        return complex(real, imag)

    return 0.0 if abs(value) < tolerance else value


def format_number(value: Number, digits: int = 6) -> str:
    """Format real or complex values for educational output."""
    value = clean_number(value)

    if isinstance(value, complex):
        if abs(value.imag) < 1e-12:
            return f"{value.real:.{digits}f}"
        sign = "+" if value.imag >= 0 else "-"
        return (
            f"{value.real:.{digits}f}"
            f"{sign}{abs(value.imag):.{digits}f}i"
        )

    return f"{value:.{digits}f}"


def print_vector(vector: Sequence[Number], name: str = "vector") -> None:
    values = ", ".join(format_number(value) for value in vector)
    print(f"{name} = [{values}]")


def print_matrix(matrix: Matrix, name: str = "matrix") -> None:
    print(f"{name} =")
    for row in matrix:
        print("  [" + ", ".join(format_number(value) for value in row) + "]")


def zeros(rows: int, columns: int) -> Matrix:
    """Create a zero matrix."""
    return [[0.0 for _ in range(columns)] for _ in range(rows)]


def identity_matrix(size: int) -> Matrix:
    """Create an identity matrix."""
    matrix = zeros(size, size)
    for i in range(size):
        matrix[i][i] = 1.0
    return matrix


def copy_matrix(matrix: Matrix) -> Matrix:
    return [row[:] for row in matrix]


def shape(matrix: Matrix) -> tuple[int, int]:
    """Return (rows, columns), validating rectangular structure."""
    if not matrix:
        return 0, 0

    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("Matrix must be rectangular.")

    return len(matrix), columns


def vector_add(a: Sequence[Number], b: Sequence[Number]) -> Vector:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal length.")
    return [x + y for x, y in zip(a, b)]


def vector_subtract(a: Sequence[Number], b: Sequence[Number]) -> Vector:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal length.")
    return [x - y for x, y in zip(a, b)]


def scalar_multiply_vector(scalar: Number, vector: Sequence[Number]) -> Vector:
    return [scalar * value for value in vector]


def dot(a: Sequence[Number], b: Sequence[Number]) -> Number:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal length.")
    return sum(x * y for x, y in zip(a, b))


def conjugate_dot(a: Sequence[Number], b: Sequence[Number]) -> Number:
    """Hermitian inner product, appropriate for complex vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have equal length.")
    return sum(x.conjugate() * y for x, y in zip(a, b))


def norm(vector: Sequence[Number]) -> float:
    """Euclidean norm for real or complex vectors."""
    return math.sqrt(float(abs(conjugate_dot(vector, vector))))


def normalize(vector: Sequence[Number]) -> Vector:
    length = norm(vector)
    if length < EPSILON:
        raise ValueError("Cannot normalize a zero vector.")
    return [value / length for value in vector]


def matrix_add(a: Matrix, b: Matrix) -> Matrix:
    if shape(a) != shape(b):
        raise ValueError("Matrices must have equal dimensions.")
    return [
        [x + y for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)
    ]


def matrix_subtract(a: Matrix, b: Matrix) -> Matrix:
    if shape(a) != shape(b):
        raise ValueError("Matrices must have equal dimensions.")
    return [
        [x - y for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)
    ]


def scalar_multiply_matrix(scalar: Number, matrix: Matrix) -> Matrix:
    return [[scalar * value for value in row] for row in matrix]


def transpose(matrix: Matrix) -> Matrix:
    rows, columns = shape(matrix)
    return [[matrix[i][j] for i in range(rows)] for j in range(columns)]


def conjugate_transpose(matrix: Matrix) -> Matrix:
    return [[matrix[i][j].conjugate() for i in range(len(matrix))]
            for j in range(len(matrix[0]))]


def matrix_vector_multiply(matrix: Matrix, vector: Sequence[Number]) -> Vector:
    rows, columns = shape(matrix)
    if columns != len(vector):
        raise ValueError("Matrix columns must equal vector length.")

    return [sum(matrix[i][j] * vector[j] for j in range(columns))
            for i in range(rows)]


def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    rows_a, columns_a = shape(a)
    rows_b, columns_b = shape(b)

    if columns_a != rows_b:
        raise ValueError("Inner matrix dimensions must match.")

    result = zeros(rows_a, columns_b)

    for i in range(rows_a):
        for j in range(columns_b):
            result[i][j] = sum(
                a[i][k] * b[k][j] for k in range(columns_a)
            )

    return result


def matrix_power(matrix: Matrix, exponent: int) -> Matrix:
    """Exponentiation by squaring."""
    rows, columns = shape(matrix)

    if rows != columns:
        raise ValueError("Matrix powers require a square matrix.")

    if exponent < 0:
        raise ValueError("This implementation supports non-negative powers.")

    result = identity_matrix(rows)
    base = copy_matrix(matrix)

    while exponent:
        if exponent % 2 == 1:
            result = matrix_multiply(result, base)
        base = matrix_multiply(base, base)
        exponent //= 2

    return result


# ---------------------------------------------------------------------------
# Determinants and characteristic polynomials
# ---------------------------------------------------------------------------

def minor_matrix(matrix: Matrix, excluded_row: int, excluded_column: int) -> Matrix:
    return [
        [
            value
            for j, value in enumerate(row)
            if j != excluded_column
        ]
        for i, row in enumerate(matrix)
        if i != excluded_row
    ]


def determinant(matrix: Matrix) -> Number:
    """
    Recursive determinant implementation.

    This is intentionally educational rather than production-optimized.
    For large matrices, LU decomposition is substantially more efficient.
    """
    rows, columns = shape(matrix)

    if rows != columns:
        raise ValueError("Determinant requires a square matrix.")

    if rows == 0:
        return 1.0

    if rows == 1:
        return matrix[0][0]

    if rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    total = 0
    for column in range(columns):
        cofactor = (
            (-1) ** column
            * matrix[0][column]
            * determinant(minor_matrix(matrix, 0, column))
        )
        total += cofactor

    return total


def trace(matrix: Matrix) -> Number:
    rows, columns = shape(matrix)
    if rows != columns:
        raise ValueError("Trace requires a square matrix.")
    return sum(matrix[i][i] for i in range(rows))


def characteristic_polynomial_2x2(matrix: Matrix) -> tuple[float, float]:
    """
    For A = [[a,b],[c,d]],

        det(lambda I - A)
        = lambda^2 - trace(A) lambda + det(A).

    Returns coefficients (p, q) for:
        lambda^2 + p lambda + q
    """
    rows, columns = shape(matrix)
    if (rows, columns) != (2, 2):
        raise ValueError("This helper is specifically for 2x2 matrices.")

    return (
        -float(trace(matrix)),
        float(determinant(matrix)),
    )


def quadratic_roots(a: Number, b: Number, c: Number) -> tuple[complex, complex]:
    """Stable enough educational quadratic formula using complex arithmetic."""
    if abs(a) < EPSILON:
        if abs(b) < EPSILON:
            raise ValueError("Not a valid quadratic equation.")
        root = -c / b
        return complex(root), complex(root)

    discriminant = complex(b * b - 4 * a * c)
    square_root = cmath.sqrt(discriminant)

    root_1 = (-b + square_root) / (2 * a)
    root_2 = (-b - square_root) / (2 * a)

    return root_1, root_2


# ---------------------------------------------------------------------------
# Eigenvalues and eigenvectors for 2x2 matrices
# ---------------------------------------------------------------------------

def eigenvalues_2x2(matrix: Matrix) -> list[complex]:
    """Calculate both eigenvalues of a real 2x2 matrix."""
    p, q = characteristic_polynomial_2x2(matrix)
    root_1, root_2 = quadratic_roots(1.0, p, q)
    return [root_1, root_2]


def null_vector_2x2(matrix: Matrix, tolerance: float = 1e-10) -> Vector:
    """
    Find a non-zero vector in the null space of a 2x2 matrix.

    For educational purposes, the method chooses a numerically stable row
    and constructs a vector perpendicular to it.
    """
    row_1 = matrix[0]
    row_2 = matrix[1]

    candidates = [row_1, row_2]
    chosen = max(candidates, key=lambda row: abs(row[0]) + abs(row[1]))

    a, b = chosen

    if abs(a) < tolerance and abs(b) < tolerance:
        raise ValueError("Matrix has no useful non-zero row.")

    # [a, b] dot [-b, a] = -ab + ab = 0.
    candidate = [-b, a]

    if norm(candidate) < tolerance:
        candidate = [a, b]

    return normalize(candidate)


def eigenvectors_2x2(matrix: Matrix) -> list[tuple[complex, Vector]]:
    """
    Compute eigenvectors corresponding to the eigenvalues of a 2x2 matrix.

    For repeated eigenvalues, the same eigenvector may be returned twice.
    A repeated eigenvalue does not automatically imply two independent
    eigenvectors.
    """
    values = eigenvalues_2x2(matrix)
    results = []

    for eigenvalue in values:
        shifted = [
            [matrix[i][j] - (eigenvalue if i == j else 0)
             for j in range(2)]
            for i in range(2)
        ]

        vector = null_vector_2x2(shifted)
        results.append((eigenvalue, vector))

    return results


def verify_eigenpair(
    matrix: Matrix,
    eigenvalue: Number,
    eigenvector: Sequence[Number],
    tolerance: float = 1e-8,
) -> bool:
    """
    Verify Av = lambda v using a residual norm.

    The residual is:
        ||Av - lambda v||.
    """
    left = matrix_vector_multiply(matrix, eigenvector)
    right = scalar_multiply_vector(eigenvalue, eigenvector)
    residual = vector_subtract(left, right)
    return norm(residual) <= tolerance


def eigen_residual(
    matrix: Matrix,
    eigenvalue: Number,
    eigenvector: Sequence[Number],
) -> float:
    left = matrix_vector_multiply(matrix, eigenvector)
    right = scalar_multiply_vector(eigenvalue, eigenvector)
    return norm(vector_subtract(left, right))


# ---------------------------------------------------------------------------
# Linear transformations and geometric interpretation
# ---------------------------------------------------------------------------

def demonstrate_linear_transformation() -> None:
    print("\n" + "=" * 78)
    print("1. EIGENVECTORS AS SPECIAL DIRECTIONS")
    print("=" * 78)

    matrix = [
        [3.0, 0.0],
        [0.0, 2.0],
    ]

    vector = [1.0, 1.0]
    eigenvector = [1.0, 0.0]

    transformed_vector = matrix_vector_multiply(matrix, vector)
    transformed_eigenvector = matrix_vector_multiply(matrix, eigenvector)

    print_matrix(matrix, "A")
    print_vector(vector, "ordinary vector v")
    print_vector(transformed_vector, "Av")

    print_vector(eigenvector, "eigenvector u")
    print_vector(transformed_eigenvector, "Au")
    print("Au remains on the same line as u.")
    print("The scaling factor is the eigenvalue lambda = 3.")


# ---------------------------------------------------------------------------
# Diagonalization and eigen decomposition
# ---------------------------------------------------------------------------

def invert_matrix(matrix: Matrix) -> Matrix:
    """
    Gauss-Jordan matrix inversion.

    Suitable for educational small matrices.
    """
    n, columns = shape(matrix)

    if n != columns:
        raise ValueError("Only square matrices can be inverted.")

    augmented = [
        [complex(value) for value in matrix[i]]
        + [complex(identity_matrix(n)[i][j]) for j in range(n)]
        for i in range(n)
    ]

    for column in range(n):
        pivot_row = max(
            range(column, n),
            key=lambda row: abs(augmented[row][column]),
        )

        if abs(augmented[pivot_row][column]) < EPSILON:
            raise ValueError("Matrix is singular or numerically singular.")

        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )

        pivot = augmented[column][column]

        augmented[column] = [
            value / pivot for value in augmented[column]
        ]

        for row in range(n):
            if row == column:
                continue

            factor = augmented[row][column]

            augmented[row] = [
                augmented[row][j] - factor * augmented[column][j]
                for j in range(2 * n)
            ]

    return [
        [clean_number(augmented[i][j]) for j in range(n, 2 * n)]
        for i in range(n)
    ]


def diagonal_matrix(values: Sequence[Number]) -> Matrix:
    matrix = zeros(len(values), len(values))
    for i, value in enumerate(values):
        matrix[i][i] = value
    return matrix


def eigen_decomposition_2x2(matrix: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    """
    Return P, D, P_inverse where:

        A = P D P^-1

    for a diagonalizable 2x2 matrix.
    """
    eigenpairs = eigenvectors_2x2(matrix)

    values = [pair[0] for pair in eigenpairs]
    vectors = [pair[1] for pair in eigenpairs]

    p = [
        [vectors[column][row] for column in range(2)]
        for row in range(2)
    ]

    p_inverse = invert_matrix(p)
    d = diagonal_matrix(values)

    return p, d, p_inverse


def reconstruct_from_eigen_decomposition(
    p: Matrix,
    d: Matrix,
    p_inverse: Matrix,
) -> Matrix:
    return matrix_multiply(matrix_multiply(p, d), p_inverse)


def demonstrate_diagonalization() -> None:
    print("\n" + "=" * 78)
    print("2. DIAGONALIZATION")
    print("=" * 78)

    matrix = [
        [4.0, 1.0],
        [2.0, 3.0],
    ]

    print_matrix(matrix, "A")

    p, d, p_inverse = eigen_decomposition_2x2(matrix)

    print_matrix(p, "P: eigenvectors as columns")
    print_matrix(d, "D: eigenvalues on diagonal")
    print_matrix(p_inverse, "P^-1")

    reconstructed = reconstruct_from_eigen_decomposition(
        p,
        d,
        p_inverse,
    )

    print_matrix(reconstructed, "P D P^-1")

    print(
        "Diagonalization changes the coordinate system so that the "
        "transformation becomes scaling along eigenvector directions."
    )


# ---------------------------------------------------------------------------
# Matrix powers using eigen decomposition
# ---------------------------------------------------------------------------

def diagonal_power(diagonal: Matrix, exponent: int) -> Matrix:
    rows, columns = shape(diagonal)
    if rows != columns:
        raise ValueError("Matrix must be square.")

    result = zeros(rows, columns)

    for i in range(rows):
        result[i][i] = diagonal[i][i] ** exponent

    return result


def matrix_power_using_eigendecomposition(
    matrix: Matrix,
    exponent: int,
) -> Matrix:
    """
    Compute A^k through:

        A^k = P D^k P^-1.

    This is useful when many powers of the same diagonalizable matrix
    are required.
    """
    p, d, p_inverse = eigen_decomposition_2x2(matrix)
    d_power = diagonal_power(d, exponent)

    return matrix_multiply(
        matrix_multiply(p, d_power),
        p_inverse,
    )


# ---------------------------------------------------------------------------
# Symmetric matrices and the spectral theorem
# ---------------------------------------------------------------------------

def is_symmetric(matrix: Matrix, tolerance: float = 1e-10) -> bool:
    rows, columns = shape(matrix)

    if rows != columns:
        return False

    for i in range(rows):
        for j in range(columns):
            if abs(matrix[i][j] - matrix[j][i]) > tolerance:
                return False

    return True


def gram_matrix(data: Matrix) -> Matrix:
    """
    Compute X^T X.

    This is symmetric and positive semidefinite for real X.
    """
    return matrix_multiply(transpose(data), data)


def covariance_matrix(data: Matrix) -> Matrix:
    """
    Compute the sample covariance matrix.

    Rows are observations.
    Columns are features.
    """
    rows, columns = shape(data)

    if rows < 2:
        raise ValueError("At least two observations are required.")

    means = [
        sum(data[row][column] for row in range(rows)) / rows
        for column in range(columns)
    ]

    centered = [
        [data[i][j] - means[j] for j in range(columns)]
        for i in range(rows)
    ]

    covariance = gram_matrix(centered)

    return scalar_multiply_matrix(
        1.0 / (rows - 1),
        covariance,
    )


def rayleigh_quotient(matrix: Matrix, vector: Sequence[Number]) -> Number:
    """
    R(v) = (v^T A v) / (v^T v).

    For a symmetric matrix, the Rayleigh quotient is bounded by the
    smallest and largest eigenvalues and equals an eigenvalue when v
    is an eigenvector.
    """
    denominator = conjugate_dot(vector, vector)

    if abs(denominator) < EPSILON:
        raise ValueError("Rayleigh quotient is undefined for zero vector.")

    numerator = conjugate_dot(
        vector,
        matrix_vector_multiply(matrix, vector),
    )

    return numerator / denominator


# ---------------------------------------------------------------------------
# Power iteration
# ---------------------------------------------------------------------------

@dataclass
class PowerIterationResult:
    eigenvalue: float
    eigenvector: Vector
    iterations: int
    converged: bool


def power_iteration(
    matrix: Matrix,
    initial_vector: Sequence[Number] | None = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-10,
) -> PowerIterationResult:
    """
    Estimate the dominant eigenpair.

    The method repeatedly applies A to a vector and normalizes it.

    It generally converges toward the eigenvector associated with the
    eigenvalue having the largest magnitude when that eigenvalue is
    sufficiently separated from the others and the starting vector has
    a component in the corresponding direction.
    """
    rows, columns = shape(matrix)

    if rows != columns:
        raise ValueError("Power iteration requires a square matrix.")

    if initial_vector is None:
        vector = [1.0 for _ in range(rows)]
    else:
        if len(initial_vector) != rows:
            raise ValueError("Initial vector has the wrong dimension.")
        vector = list(initial_vector)

    vector = normalize(vector)
    previous_eigenvalue: Number | None = None

    for iteration in range(1, max_iterations + 1):
        transformed = matrix_vector_multiply(matrix, vector)
        transformed_norm = norm(transformed)

        if transformed_norm < EPSILON:
            raise ValueError(
                "Power iteration reached the zero vector; "
                "the starting vector may lie in a null direction."
            )

        vector = normalize(transformed)
        estimated_eigenvalue = rayleigh_quotient(matrix, vector)

        if (
            previous_eigenvalue is not None
            and abs(estimated_eigenvalue - previous_eigenvalue) < tolerance
        ):
            return PowerIterationResult(
                float(estimated_eigenvalue.real
                      if isinstance(estimated_eigenvalue, complex)
                      else estimated_eigenvalue),
                vector,
                iteration,
                True,
            )

        previous_eigenvalue = estimated_eigenvalue

    final_value = rayleigh_quotient(matrix, vector)

    return PowerIterationResult(
        float(final_value.real if isinstance(final_value, complex)
              else final_value),
        vector,
        max_iterations,
        False,
    )


# ---------------------------------------------------------------------------
# Deflation for multiple eigenpairs of symmetric matrices
# ---------------------------------------------------------------------------

def outer_product(a: Sequence[Number], b: Sequence[Number]) -> Matrix:
    return [[x * y for y in b] for x in a]


def deflate_symmetric_matrix(
    matrix: Matrix,
    eigenvalue: Number,
    eigenvector: Sequence[Number],
) -> Matrix:
    """
    For a normalized eigenvector v of a symmetric matrix:

        A_new = A - lambda v v^T.

    This removes the discovered eigen-component.
    """
    component = scalar_multiply_matrix(
        eigenvalue,
        outer_product(eigenvector, eigenvector),
    )

    return matrix_subtract(matrix, component)


def top_k_eigenpairs_symmetric(
    matrix: Matrix,
    k: int,
    tolerance: float = 1e-10,
) -> list[tuple[float, Vector]]:
    """
    Repeatedly apply power iteration with deflation.

    This is educational. Practical numerical libraries generally use
    specialized symmetric eigensolvers because they are more stable.
    """
    rows, columns = shape(matrix)

    if rows != columns:
        raise ValueError("Matrix must be square.")

    if k < 1 or k > rows:
        raise ValueError("k must be between 1 and matrix dimension.")

    working = copy_matrix(matrix)
    pairs = []

    for _ in range(k):
        result = power_iteration(
            working,
            max_iterations=5000,
            tolerance=tolerance,
        )

        eigenvalue = result.eigenvalue
        eigenvector = result.eigenvector

        pairs.append((eigenvalue, eigenvector))

        working = deflate_symmetric_matrix(
            working,
            eigenvalue,
            eigenvector,
        )

    return pairs


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

@dataclass
class PCAResult:
    means: Vector
    eigenvalues: list[float]
    eigenvectors: Matrix
    explained_variance_ratio: list[float]


def sort_eigenpairs(
    eigenpairs: Iterable[tuple[Number, Sequence[Number]]]
) -> list[tuple[Number, Vector]]:
    pairs = [
        (value, list(vector))
        for value, vector in eigenpairs
    ]
    pairs.sort(key=lambda pair: abs(pair[0]), reverse=True)
    return pairs


def pca_2d(data: Matrix) -> PCAResult:
    """
    PCA for a two-feature real dataset.

    Steps:
        1. Center the observations.
        2. Calculate covariance.
        3. Find covariance eigenvectors.
        4. Sort by descending eigenvalue.
        5. Convert eigenvalues into explained-variance ratios.

    The eigenvector with the largest eigenvalue represents the direction
    of maximum variance.
    """
    rows, columns = shape(data)

    if columns != 2:
        raise ValueError("This educational PCA function expects two features.")

    means = [
        sum(data[i][j] for i in range(rows)) / rows
        for j in range(columns)
    ]

    covariance = covariance_matrix(data)
    pairs = sort_eigenpairs(
        eigenvectors_2x2(covariance)
    )

    eigenvalues = [
        float(pair[0].real)
        for pair in pairs
    ]

    eigenvectors = [
        normalize(pair[1])
        for pair in pairs
    ]

    total_variance = sum(eigenvalues)

    if abs(total_variance) < EPSILON:
        ratios = [0.0 for _ in eigenvalues]
    else:
        ratios = [
            value / total_variance
            for value in eigenvalues
        ]

    return PCAResult(
        means=means,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_variance_ratio=ratios,
    )


def project_data_onto_component(
    data: Matrix,
    means: Sequence[Number],
    component: Sequence[Number],
) -> list[float]:
    """
    Project centered observations onto one normalized PCA component.
    """
    projected = []

    for observation in data:
        centered = [
            observation[j] - means[j]
            for j in range(len(means))
        ]

        score = conjugate_dot(component, centered)

        projected.append(float(score.real if isinstance(score, complex)
                               else score))

    return projected


def reconstruct_from_component(
    scores: Sequence[Number],
    means: Sequence[Number],
    component: Sequence[Number],
) -> Matrix:
    """
    Reconstruct observations from a one-dimensional PCA representation.
    """
    reconstructed = []

    for score in scores:
        point = [
            means[j] + score * component[j]
            for j in range(len(means))
        ]
        reconstructed.append(point)

    return reconstructed


def mean_squared_reconstruction_error(
    original: Matrix,
    reconstructed: Matrix,
) -> float:
    if shape(original) != shape(reconstructed):
        raise ValueError("Matrices must have the same shape.")

    rows, columns = shape(original)

    total = 0.0

    for i in range(rows):
        for j in range(columns):
            difference = original[i][j] - reconstructed[i][j]
            total += float(abs(difference) ** 2)

    return total / (rows * columns)


# ---------------------------------------------------------------------------
# Markov chains and stationary eigenvectors
# ---------------------------------------------------------------------------

def normalize_probability_vector(vector: Sequence[float]) -> list[float]:
    if any(value < 0 for value in vector):
        raise ValueError("Probabilities cannot be negative.")

    total = sum(vector)

    if total <= EPSILON:
        raise ValueError("Probability vector must have positive total.")

    return [value / total for value in vector]


def stationary_distribution_power_iteration(
    transition_matrix: Matrix,
    max_iterations: int = 10000,
    tolerance: float = 1e-12,
) -> Vector:
    """
    Estimate a stationary distribution.

    Here each row represents the current state and each row sums to one.
    We therefore repeatedly multiply a row probability vector by P.
    """
    rows, columns = shape(transition_matrix)

    if rows != columns:
        raise ValueError("Transition matrix must be square.")

    for row in transition_matrix:
        if any(value < -EPSILON for value in row):
            raise ValueError("Transition probabilities cannot be negative.")

        if abs(sum(row) - 1.0) > EPSILON:
            raise ValueError("Each transition row must sum to 1.")

    distribution = [1.0 / rows] * rows

    for _ in range(max_iterations):
        next_distribution = [
            sum(
                distribution[i] * transition_matrix[i][j]
                for i in range(rows)
            )
            for j in range(columns)
        ]

        difference = sum(
            abs(next_distribution[i] - distribution[i])
            for i in range(rows)
        )

        distribution = normalize_probability_vector(next_distribution)

        if difference < tolerance:
            return distribution

    return distribution


# ---------------------------------------------------------------------------
# Graph spectral concepts
# ---------------------------------------------------------------------------

def graph_laplacian(adjacency: Matrix) -> Matrix:
    """
    L = D - A.

    D is the diagonal degree matrix.
    """
    rows, columns = shape(adjacency)

    if rows != columns:
        raise ValueError("Adjacency matrix must be square.")

    degrees = [sum(row) for row in adjacency]

    laplacian = [
        [
            (degrees[i] if i == j else 0.0) - adjacency[i][j]
            for j in range(columns)
        ]
        for i in range(rows)
    ]

    return laplacian


# ---------------------------------------------------------------------------
# Complex eigenvalues
# ---------------------------------------------------------------------------

def demonstrate_complex_eigenvalues() -> None:
    """
    Rotation matrices can have complex eigenvalues.

    A 90-degree rotation has eigenvalues +i and -i over the complex numbers,
    but no real eigenvectors.
    """
    rotation = [
        [0.0, -1.0],
        [1.0, 0.0],
    ]

    values = eigenvalues_2x2(rotation)

    print("\n" + "=" * 78)
    print("3. COMPLEX EIGENVALUES")
    print("=" * 78)

    print_matrix(rotation, "90-degree rotation matrix")

    for index, value in enumerate(values, start=1):
        print(f"lambda_{index} = {format_number(value)}")

    print(
        "A real planar rotation may have no real eigenvector because "
        "every non-zero real direction changes direction."
    )


# ---------------------------------------------------------------------------
# Educational demonstrations
# ---------------------------------------------------------------------------

def demonstrate_basic_eigenpair() -> None:
    print("\n" + "=" * 78)
    print("4. BASIC EIGENVALUE AND EIGENVECTOR CALCULATION")
    print("=" * 78)

    matrix = [
        [4.0, 1.0],
        [2.0, 3.0],
    ]

    print_matrix(matrix, "A")

    print(f"trace(A) = {format_number(trace(matrix))}")
    print(f"det(A)   = {format_number(determinant(matrix))}")

    values = eigenvalues_2x2(matrix)

    for index, value in enumerate(values, start=1):
        print(f"lambda_{index} = {format_number(value)}")

    for value, vector in eigenvectors_2x2(matrix):
        print_vector(vector, f"eigenvector for {format_number(value)}")
        print(
            "Residual:",
            f"{eigen_residual(matrix, value, vector):.3e}",
        )
        print(
            "Verified:",
            verify_eigenpair(matrix, value, vector),
        )


def demonstrate_matrix_power() -> None:
    print("\n" + "=" * 78)
    print("5. MATRIX POWERS")
    print("=" * 78)

    matrix = [
        [2.0, 1.0],
        [1.0, 2.0],
    ]

    exponent = 5

    direct = matrix_power(matrix, exponent)
    eigen_based = matrix_power_using_eigendecomposition(
        matrix,
        exponent,
    )

    print_matrix(matrix, "A")
    print(f"Computing A^{exponent}")

    print_matrix(direct, "Direct multiplication")
    print_matrix(eigen_based, "Eigen decomposition")

    difference = matrix_subtract(direct, eigen_based)
    max_error = max(abs(value) for row in difference for value in row)

    print(f"Maximum numerical difference = {max_error:.3e}")


def demonstrate_pca() -> None:
    print("\n" + "=" * 78)
    print("6. PCA THROUGH COVARIANCE EIGENVECTORS")
    print("=" * 78)

    data = [
        [2.0, 1.1],
        [3.0, 1.9],
        [4.0, 3.0],
        [5.0, 3.9],
        [6.0, 5.1],
        [7.0, 5.8],
        [8.0, 7.2],
        [9.0, 8.0],
    ]

    print_matrix(data, "Original observations")

    covariance = covariance_matrix(data)
    print_matrix(covariance, "Covariance matrix")

    result = pca_2d(data)

    print_vector(result.means, "Feature means")

    for index, value in enumerate(result.eigenvalues, start=1):
        print(
            f"Principal component {index}: "
            f"eigenvalue={value:.6f}, "
            f"explained variance={result.explained_variance_ratio[index - 1]:.2%}"
        )
        print_vector(
            result.eigenvectors[index - 1],
            f"component {index}",
        )

    scores = project_data_onto_component(
        data,
        result.means,
        result.eigenvectors[0],
    )

    reconstructed = reconstruct_from_component(
        scores,
        result.means,
        result.eigenvectors[0],
    )

    error = mean_squared_reconstruction_error(
        data,
        reconstructed,
    )

    print_vector(scores, "One-dimensional PCA scores")
    print_matrix(reconstructed, "One-component reconstruction")
    print(f"Mean squared reconstruction error = {error:.6f}")

    print(
        "The first covariance eigenvector is the direction of maximum "
        "variance. Projection onto it compresses the two-feature data "
        "into one coordinate while retaining as much variance as possible."
    )


def demonstrate_power_iteration() -> None:
    print("\n" + "=" * 78)
    print("7. POWER ITERATION")
    print("=" * 78)

    matrix = [
        [5.0, 1.0],
        [1.0, 3.0],
    ]

    result = power_iteration(matrix)

    print_matrix(matrix, "A")
    print(f"Estimated dominant eigenvalue = {result.eigenvalue:.10f}")
    print_vector(result.eigenvector, "Estimated eigenvector")
    print(f"Iterations = {result.iterations}")
    print(f"Converged = {result.converged}")

    residual = eigen_residual(
        matrix,
        result.eigenvalue,
        result.eigenvector,
    )

    print(f"Eigenpair residual = {residual:.3e}")


def demonstrate_markov_chain() -> None:
    print("\n" + "=" * 78)
    print("8. STATIONARY DISTRIBUTION OF A MARKOV CHAIN")
    print("=" * 78)

    transition = [
        [0.90, 0.10, 0.00],
        [0.20, 0.60, 0.20],
        [0.00, 0.30, 0.70],
    ]

    print_matrix(transition, "Transition matrix P")

    stationary = stationary_distribution_power_iteration(
        transition
    )

    print_vector(
        stationary,
        "stationary distribution",
    )

    next_distribution = [
        sum(
            stationary[i] * transition[i][j]
            for i in range(3)
        )
        for j in range(3)
    ]

    print_vector(
        next_distribution,
        "stationary distribution after one transition",
    )

    print(
        "A stationary distribution is associated with eigenvalue 1. "
        "For a row-stochastic transition matrix, the distribution satisfies "
        "pi P = pi."
    )


def demonstrate_graph_laplacian() -> None:
    print("\n" + "=" * 78)
    print("9. GRAPH LAPLACIAN")
    print("=" * 78)

    # Undirected chain: 1 -- 2 -- 3 -- 4.
    adjacency = [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    laplacian = graph_laplacian(adjacency)

    print_matrix(adjacency, "Adjacency matrix A")
    print_matrix(laplacian, "Laplacian L = D - A")

    constant_vector = [1.0, 1.0, 1.0, 1.0]

    print_vector(
        matrix_vector_multiply(laplacian, constant_vector),
        "L * 1",
    )

    print(
        "For this connected graph, the constant vector is associated "
        "with Laplacian eigenvalue 0."
    )


# ---------------------------------------------------------------------------
# Edge cases and common mistakes
# ---------------------------------------------------------------------------

def demonstrate_edge_cases() -> None:
    print("\n" + "=" * 78)
    print("10. EDGE CASES AND NUMERICAL DETAILS")
    print("=" * 78)

    repeated = [
        [2.0, 1.0],
        [0.0, 2.0],
    ]

    print_matrix(repeated, "Repeated-eigenvalue matrix")

    values = eigenvalues_2x2(repeated)

    for value in values:
        print(f"Eigenvalue = {format_number(value)}")

    print(
        "A repeated eigenvalue does not guarantee diagonalizability. "
        "This matrix has only one independent eigenvector direction."
    )

    symmetric = [
        [4.0, 1.0],
        [1.0, 4.0],
    ]

    print_matrix(symmetric, "Symmetric matrix")

    pairs = eigenvectors_2x2(symmetric)

    for value, vector in pairs:
        print(
            f"lambda={format_number(value)}, "
            f"vector={vector}"
        )

    print(
        "Symmetric matrices have real eigenvalues and admit an orthonormal "
        "eigenbasis. This is the spectral theorem in finite-dimensional "
        "real linear algebra."
    )


# ---------------------------------------------------------------------------
# A compact educational checklist
# ---------------------------------------------------------------------------

def print_key_facts() -> None:
    print("\n" + "=" * 78)
    print("11. KEY FACTS")
    print("=" * 78)

    facts = [
        "An eigenpair satisfies A v = lambda v with v != 0.",
        "Eigenvalues are roots of det(A - lambda I) = 0.",
        "The trace equals the sum of eigenvalues, counting multiplicity.",
        "The determinant equals the product of eigenvalues, counting multiplicity.",
        "Eigenvectors are defined only up to non-zero scalar multiplication.",
        "A matrix is diagonalizable when it has enough linearly independent eigenvectors.",
        "For A = P D P^-1, powers become A^k = P D^k P^-1.",
        "Real symmetric matrices have real eigenvalues and orthonormal eigenvectors.",
        "PCA uses covariance-matrix eigenvectors as principal directions.",
        "The largest PCA eigenvalue corresponds to the greatest variance direction.",
        "Power iteration estimates a dominant eigenpair without solving the characteristic polynomial explicitly.",
        "Numerical eigenvalue algorithms require residual and stability checks.",
    ]

    for fact in facts:
        print(f"- {fact}")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("EIGENVALUES & EIGENVECTORS")
    print("Eigen decomposition, characteristic equations, and ML applications")
    print("=" * 78)

    demonstrate_linear_transformation()
    demonstrate_basic_eigenpair()
    demonstrate_diagonalization()
    demonstrate_matrix_power()
    demonstrate_pca()
    demonstrate_power_iteration()
    demonstrate_markov_chain()
    demonstrate_graph_laplacian()
    demonstrate_complex_eigenvalues()
    demonstrate_edge_cases()
    print_key_facts()

    print("\n" + "=" * 78)
    print("END OF STUDY PROGRAM")
    print("=" * 78)


if __name__ == "__main__":
    main()
