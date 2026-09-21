"""
Linear Algebra: Scalars, Vectors, Matrices, Tensors, and Matrix Operations

A self-contained study and demonstration program covering:
- Scalars
- Vectors
- Vector arithmetic
- Dot products
- Norms and distances
- Angles and projections
- Matrices
- Matrix indexing and construction
- Matrix addition, subtraction, and scalar multiplication
- Matrix multiplication
- Transpose
- Identity and diagonal matrices
- Determinants
- Inverse matrices
- Systems of linear equations
- Gaussian elimination
- Rank
- Linear independence
- Basis and span
- Matrix transformations
- Eigenvalues and eigenvectors
- Symmetric matrices
- Tensors
- Tensor shapes and indexing
- Tensor operations
- Broadcasting concepts
- Practical numerical issues
- Validation and error handling
- Performance considerations
- A small machine-learning-style matrix pipeline

The program uses only the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, degrees, isclose, sqrt
from pprint import pprint
from typing import Iterable, List, Sequence, Tuple, Union


Number = Union[int, float]
Vector = List[Number]
Matrix = List[List[Number]]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def almost_equal(a: Number, b: Number, tolerance: float = 1e-9) -> bool:
    """Compare floating-point values with a tolerance."""
    return isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def format_number(value: Number) -> str:
    """Produce readable numeric output."""
    if almost_equal(value, round(float(value))):
        return str(int(round(float(value))))
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------

def demonstrate_scalars() -> None:
    print_section("1. Scalars")

    # A scalar is a single numerical value rather than an ordered collection.
    temperature = 25.5
    price = 1499.0
    count = 7

    print("temperature =", temperature)
    print("price       =", price)
    print("count       =", count)

    # Scalar arithmetic can be applied directly.
    print("temperature + 2 =", temperature + 2)
    print("price * 1.18    =", price * 1.18)

    # A scalar can scale every component of a vector or matrix.
    vector = [1, 2, 3]
    scaled_vector = [4 * x for x in vector]
    print("4 *", vector, "=", scaled_vector)


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

def validate_vector(vector: Sequence[Number], name: str = "vector") -> None:
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers.")
    if len(vector) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not all(isinstance(value, (int, float)) for value in vector):
        raise TypeError(f"Every element of {name} must be numeric.")


def vector_add(a: Sequence[Number], b: Sequence[Number]) -> Vector:
    validate_vector(a, "a")
    validate_vector(b, "b")
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension.")
    return [x + y for x, y in zip(a, b)]


def vector_subtract(a: Sequence[Number], b: Sequence[Number]) -> Vector:
    validate_vector(a, "a")
    validate_vector(b, "b")
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension.")
    return [x - y for x, y in zip(a, b)]


def scalar_multiply_vector(scalar: Number, vector: Sequence[Number]) -> Vector:
    validate_vector(vector)
    if not isinstance(scalar, (int, float)):
        raise TypeError("scalar must be numeric.")
    return [scalar * value for value in vector]


def dot_product(a: Sequence[Number], b: Sequence[Number]) -> Number:
    validate_vector(a, "a")
    validate_vector(b, "b")
    if len(a) != len(b):
        raise ValueError("Dot product requires equal dimensions.")
    return sum(x * y for x, y in zip(a, b))


def vector_norm(vector: Sequence[Number], p: int = 2) -> float:
    validate_vector(vector)
    if p <= 0:
        raise ValueError("The norm order p must be positive.")
    if p == 2:
        return sqrt(sum(float(x) ** 2 for x in vector))
    if p == 1:
        return sum(abs(float(x)) for x in vector)
    return sum(abs(float(x)) ** p for x in vector) ** (1.0 / p)


def euclidean_distance(a: Sequence[Number], b: Sequence[Number]) -> float:
    return vector_norm(vector_subtract(a, b))


def vector_angle_degrees(a: Sequence[Number], b: Sequence[Number]) -> float:
    denominator = vector_norm(a) * vector_norm(b)
    if almost_equal(denominator, 0):
        raise ValueError("The angle is undefined for a zero vector.")
    cosine = dot_product(a, b) / denominator
    # Floating-point arithmetic can produce values such as 1.00000000001.
    cosine = max(-1.0, min(1.0, float(cosine)))
    return degrees(acos(cosine))


def projection_of_a_onto_b(
    a: Sequence[Number], b: Sequence[Number]
) -> Vector:
    denominator = dot_product(b, b)
    if almost_equal(denominator, 0):
        raise ValueError("Cannot project onto the zero vector.")
    coefficient = dot_product(a, b) / denominator
    return scalar_multiply_vector(coefficient, b)


def demonstrate_vectors() -> None:
    print_section("2. Vectors")

    vector_a = [2, -1, 3]
    vector_b = [4, 5, 1]

    print("a =", vector_a)
    print("b =", vector_b)
    print("a + b =", vector_add(vector_a, vector_b))
    print("a - b =", vector_subtract(vector_a, vector_b))
    print("3a =", scalar_multiply_vector(3, vector_a))
    print("a · b =", dot_product(vector_a, vector_b))
    print("||a||₂ =", round(vector_norm(vector_a), 6))
    print("||a||₁ =", round(vector_norm(vector_a, 1), 6))
    print("distance(a, b) =", round(euclidean_distance(vector_a, vector_b), 6))
    print("angle(a, b) =", round(vector_angle_degrees(vector_a, vector_b), 4), "degrees")
    print("projection of a onto b =", projection_of_a_onto_b(vector_a, vector_b))

    # Orthogonality occurs when the dot product is zero.
    perpendicular_a = [1, 0]
    perpendicular_b = [0, 1]
    print("dot([1, 0], [0, 1]) =", dot_product(perpendicular_a, perpendicular_b))


# ---------------------------------------------------------------------------
# Matrix validation and basic operations
# ---------------------------------------------------------------------------

def validate_matrix(matrix: Sequence[Sequence[Number]], name: str = "matrix") -> None:
    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of rows.")
    if len(matrix) == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if any(
        not isinstance(row, Sequence) or isinstance(row, (str, bytes))
        for row in matrix
    ):
        raise TypeError(f"Every row of {name} must be a sequence.")

    column_count = len(matrix[0])
    if column_count == 0:
        raise ValueError(f"{name} must contain at least one column.")

    if any(len(row) != column_count for row in matrix):
        raise ValueError(f"{name} must be rectangular.")

    if not all(
        isinstance(value, (int, float))
        for row in matrix
        for value in row
    ):
        raise TypeError(f"Every matrix element in {name} must be numeric.")


def matrix_shape(matrix: Sequence[Sequence[Number]]) -> Tuple[int, int]:
    validate_matrix(matrix)
    return len(matrix), len(matrix[0])


def copy_matrix(matrix: Sequence[Sequence[Number]]) -> Matrix:
    validate_matrix(matrix)
    return [list(row) for row in matrix]


def matrix_add(a: Sequence[Sequence[Number]],
               b: Sequence[Sequence[Number]]) -> Matrix:
    validate_matrix(a, "a")
    validate_matrix(b, "b")
    if matrix_shape(a) != matrix_shape(b):
        raise ValueError("Matrices must have the same dimensions.")
    return [
        [x + y for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)
    ]


def matrix_subtract(a: Sequence[Sequence[Number]],
                    b: Sequence[Sequence[Number]]) -> Matrix:
    validate_matrix(a, "a")
    validate_matrix(b, "b")
    if matrix_shape(a) != matrix_shape(b):
        raise ValueError("Matrices must have the same dimensions.")
    return [
        [x - y for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)
    ]


def scalar_multiply_matrix(
    scalar: Number,
    matrix: Sequence[Sequence[Number]]
) -> Matrix:
    validate_matrix(matrix)
    if not isinstance(scalar, (int, float)):
        raise TypeError("scalar must be numeric.")
    return [[scalar * value for value in row] for row in matrix]


def matrix_multiply(
    a: Sequence[Sequence[Number]],
    b: Sequence[Sequence[Number]]
) -> Matrix:
    """
    Standard matrix multiplication.

    If A is m x n and B is n x p, A @ B is m x p.
    Each output element is a dot product between a row of A
    and a column of B.
    """
    validate_matrix(a, "a")
    validate_matrix(b, "b")

    a_rows, a_columns = matrix_shape(a)
    b_rows, b_columns = matrix_shape(b)

    if a_columns != b_rows:
        raise ValueError(
            f"Cannot multiply shapes {a_rows}x{a_columns} and "
            f"{b_rows}x{b_columns}."
        )

    result = []
    for i in range(a_rows):
        output_row = []
        for j in range(b_columns):
            value = sum(a[i][k] * b[k][j] for k in range(a_columns))
            output_row.append(value)
        result.append(output_row)

    return result


def transpose(matrix: Sequence[Sequence[Number]]) -> Matrix:
    validate_matrix(matrix)
    return [list(column) for column in zip(*matrix)]


def identity_matrix(size: int) -> Matrix:
    if size <= 0:
        raise ValueError("Identity matrix size must be positive.")
    return [
        [1 if row == column else 0 for column in range(size)]
        for row in range(size)
    ]


def diagonal_matrix(values: Sequence[Number]) -> Matrix:
    validate_vector(values, "values")
    size = len(values)
    return [
        [values[i] if i == j else 0 for j in range(size)]
        for i in range(size)
    ]


def print_matrix(matrix: Sequence[Sequence[Number]]) -> None:
    validate_matrix(matrix)
    for row in matrix:
        print("[ " + "  ".join(format_number(x) for x in row) + " ]")


def demonstrate_matrices() -> None:
    print_section("3. Matrices")

    a = [
        [1, 2, 3],
        [4, 5, 6],
    ]

    b = [
        [6, 5, 4],
        [3, 2, 1],
    ]

    print("Matrix A:")
    print_matrix(a)

    print("Matrix B:")
    print_matrix(b)

    print("A + B:")
    print_matrix(matrix_add(a, b))

    print("A - B:")
    print_matrix(matrix_subtract(a, b))

    print("3A:")
    print_matrix(scalar_multiply_matrix(3, a))

    print("Aᵀ:")
    print_matrix(transpose(a))

    left = [
        [1, 2, 3],
        [4, 5, 6],
    ]

    right = [
        [7, 8],
        [9, 10],
        [11, 12],
    ]

    print("A @ B:")
    print_matrix(matrix_multiply(left, right))

    print("Identity matrix I₃:")
    print_matrix(identity_matrix(3))

    print("Diagonal matrix:")
    print_matrix(diagonal_matrix([10, 20, 30]))


# ---------------------------------------------------------------------------
# Determinants
# ---------------------------------------------------------------------------

def determinant(matrix: Sequence[Sequence[Number]]) -> float:
    """
    Compute a determinant using Gaussian elimination.

    The algorithm is O(n^3), unlike a naive recursive expansion,
    which becomes extremely expensive for larger matrices.
    """
    validate_matrix(matrix)
    rows, columns = matrix_shape(matrix)

    if rows != columns:
        raise ValueError("Determinant requires a square matrix.")

    work = [[float(value) for value in row] for row in matrix]
    det = 1.0
    n = rows

    for column in range(n):
        pivot_row = max(
            range(column, n),
            key=lambda row: abs(work[row][column])
        )

        pivot = work[pivot_row][column]

        if almost_equal(pivot, 0):
            return 0.0

        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            det *= -1

        pivot = work[column][column]
        det *= pivot

        for row in range(column + 1, n):
            factor = work[row][column] / pivot
            for j in range(column + 1, n):
                work[row][j] -= factor * work[column][j]

    return det


def demonstrate_determinants() -> None:
    print_section("4. Determinants")

    matrix = [
        [4, 7],
        [2, 6],
    ]

    print("Matrix:")
    print_matrix(matrix)
    print("det(A) =", determinant(matrix))

    singular = [
        [1, 2],
        [2, 4],
    ]

    print("Singular matrix determinant =", determinant(singular))
    print(
        "A zero determinant indicates that a square matrix is singular "
        "and therefore has no ordinary inverse."
    )


# ---------------------------------------------------------------------------
# Gaussian elimination, rank, and linear systems
# ---------------------------------------------------------------------------

def rref(matrix: Sequence[Sequence[Number]],
         tolerance: float = 1e-10) -> Matrix:
    """
    Convert a matrix to reduced row-echelon form.

    The function uses partial pivoting, which improves numerical stability
    compared with blindly selecting the first available pivot.
    """
    validate_matrix(matrix)
    work = [[float(value) for value in row] for row in matrix]

    row_count = len(work)
    column_count = len(work[0])
    pivot_row = 0

    for column in range(column_count):
        if pivot_row >= row_count:
            break

        best_row = max(
            range(pivot_row, row_count),
            key=lambda row: abs(work[row][column])
        )

        if abs(work[best_row][column]) <= tolerance:
            continue

        work[pivot_row], work[best_row] = (
            work[best_row],
            work[pivot_row],
        )

        pivot = work[pivot_row][column]
        work[pivot_row] = [
            value / pivot for value in work[pivot_row]
        ]

        for row in range(row_count):
            if row == pivot_row:
                continue

            factor = work[row][column]

            if abs(factor) <= tolerance:
                continue

            work[row] = [
                work[row][j] - factor * work[pivot_row][j]
                for j in range(column_count)
            ]

        pivot_row += 1

    for row in range(row_count):
        for column in range(column_count):
            if abs(work[row][column]) <= tolerance:
                work[row][column] = 0.0

    return work


def matrix_rank(matrix: Sequence[Sequence[Number]]) -> int:
    reduced = rref(matrix)
    return sum(
        any(abs(value) > 1e-10 for value in row)
        for row in reduced
    )


def solve_linear_system(
    coefficient_matrix: Sequence[Sequence[Number]],
    constants: Sequence[Number],
) -> Vector:
    """
    Solve a square linear system A x = b using RREF.

    This routine intentionally detects inconsistent and underdetermined
    systems rather than silently returning an incorrect answer.
    """
    validate_matrix(coefficient_matrix, "coefficient_matrix")
    validate_vector(constants, "constants")

    rows, columns = matrix_shape(coefficient_matrix)

    if rows != len(constants):
        raise ValueError("A and b must have compatible dimensions.")

    augmented = [
        list(map(float, coefficient_matrix[row])) + [float(constants[row])]
        for row in range(rows)
    ]

    reduced = rref(augmented)

    # Detect 0 = non-zero rows.
    for row in reduced:
        if all(abs(value) <= 1e-10 for value in row[:-1]) and abs(row[-1]) > 1e-10:
            raise ValueError("The linear system is inconsistent.")

    rank_a = matrix_rank(coefficient_matrix)

    if rank_a < columns:
        raise ValueError(
            "The system does not have a unique solution."
        )

    solution = [0.0] * columns

    for row in reduced:
        pivot_index = None
        for column in range(columns):
            if abs(row[column]) > 1e-10:
                pivot_index = column
                break
        if pivot_index is not None:
            solution[pivot_index] = row[-1]

    return solution


def demonstrate_linear_systems() -> None:
    print_section("5. Linear Systems and Gaussian Elimination")

    coefficients = [
        [2, 1],
        [1, -1],
    ]
    constants = [7, 1]

    solution = solve_linear_system(coefficients, constants)

    print("System:")
    print("2x + y = 7")
    print(" x - y = 1")
    print("solution =", solution)

    print("RREF of augmented matrix:")
    augmented = [
        coefficients[0] + [constants[0]],
        coefficients[1] + [constants[1]],
    ]
    print_matrix(rref(augmented))

    dependent_system = [
        [1, 2],
        [2, 4],
    ]

    print("rank of dependent matrix =", matrix_rank(dependent_system))


# ---------------------------------------------------------------------------
# Matrix inverse
# ---------------------------------------------------------------------------

def inverse_matrix(matrix: Sequence[Sequence[Number]]) -> Matrix:
    """
    Compute A^-1 using augmented Gaussian elimination.

    [A | I] -> [I | A^-1]
    """
    validate_matrix(matrix)
    rows, columns = matrix_shape(matrix)

    if rows != columns:
        raise ValueError("Only square matrices have ordinary inverses.")

    n = rows
    augmented = [
        [float(value) for value in matrix[row]] + identity_matrix(n)[row]
        for row in range(n)
    ]

    reduced = rref(augmented)

    left = [row[:n] for row in reduced]
    right = [row[n:] for row in reduced]

    if any(
        not almost_equal(left[i][i], 1.0)
        for i in range(n)
    ):
        raise ValueError("Matrix is singular and cannot be inverted.")

    return right


def demonstrate_inverse() -> None:
    print_section("6. Matrix Inverse")

    matrix = [
        [4, 7],
        [2, 6],
    ]

    inverse = inverse_matrix(matrix)

    print("A:")
    print_matrix(matrix)

    print("A⁻¹:")
    print_matrix(inverse)

    print("A @ A⁻¹:")
    print_matrix(matrix_multiply(matrix, inverse))


# ---------------------------------------------------------------------------
# Linear independence, span, and basis
# ---------------------------------------------------------------------------

def vectors_as_columns(vectors: Sequence[Sequence[Number]]) -> Matrix:
    validate_vector(vectors[0], "first vector")
    dimension = len(vectors[0])

    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("All vectors must have the same dimension.")

    return [
        [vectors[column][row] for column in range(len(vectors))]
        for row in range(dimension)
    ]


def are_linearly_independent(vectors: Sequence[Sequence[Number]]) -> bool:
    if len(vectors) == 0:
        return True
    matrix = vectors_as_columns(vectors)
    return matrix_rank(matrix) == len(vectors)


def demonstrate_span_and_independence() -> None:
    print_section("7. Span, Basis, and Linear Independence")

    independent_vectors = [
        [1, 0],
        [0, 1],
    ]

    dependent_vectors = [
        [1, 2],
        [2, 4],
    ]

    print(
        "Independent vectors:",
        independent_vectors,
        "->",
        are_linearly_independent(independent_vectors),
    )

    print(
        "Dependent vectors:",
        dependent_vectors,
        "->",
        are_linearly_independent(dependent_vectors),
    )

    print(
        "The standard basis vectors [1, 0] and [0, 1] span R² because "
        "every two-dimensional vector can be expressed as a linear "
        "combination of them."
    )


# ---------------------------------------------------------------------------
# Matrix transformations
# ---------------------------------------------------------------------------

def apply_matrix_to_vector(
    matrix: Sequence[Sequence[Number]],
    vector: Sequence[Number],
) -> Vector:
    validate_matrix(matrix)
    validate_vector(vector)

    _, columns = matrix_shape(matrix)

    if columns != len(vector):
        raise ValueError(
            "Matrix column count must equal vector dimension."
        )

    return [
        sum(matrix[row][column] * vector[column] for column in range(columns))
        for row in range(len(matrix))
    ]


def demonstrate_transformations() -> None:
    print_section("8. Matrices as Transformations")

    # This matrix reflects a vector across the y-axis.
    reflection_y = [
        [-1, 0],
        [0, 1],
    ]

    point = [3, 2]

    print("point =", point)
    print("reflection across y-axis =", apply_matrix_to_vector(reflection_y, point))

    # This matrix scales x by 2 and y by 3.
    scaling = [
        [2, 0],
        [0, 3],
    ]

    print("scaling transformation =", apply_matrix_to_vector(scaling, point))

    # Transformation composition is matrix multiplication.
    first = [
        [2, 0],
        [0, 1],
    ]

    second = [
        [1, 1],
        [0, 1],
    ]

    combined = matrix_multiply(second, first)

    print("combined transformation:")
    print_matrix(combined)
    print("combined result =", apply_matrix_to_vector(combined, point))


# ---------------------------------------------------------------------------
# Eigenvalues and eigenvectors for a 2x2 matrix
# ---------------------------------------------------------------------------

def eigenvalues_2x2(matrix: Sequence[Sequence[Number]]) -> Tuple[float, float]:
    """
    Compute the two eigenvalues of a real 2x2 matrix.

    For A = [[a,b],[c,d]]:
        characteristic polynomial:
        λ² - trace(A)λ + det(A) = 0

    Complex eigenvalues are intentionally rejected because this small
    educational implementation works with real numbers only.
    """
    validate_matrix(matrix)

    if matrix_shape(matrix) != (2, 2):
        raise ValueError("This function supports only 2x2 matrices.")

    a, b = matrix[0]
    c, d = matrix[1]

    trace = a + d
    det = a * d - b * c
    discriminant = trace * trace - 4 * det

    if discriminant < -1e-10:
        raise ValueError("The matrix has complex eigenvalues.")

    discriminant = max(0.0, float(discriminant))

    root = sqrt(discriminant)

    return (
        (float(trace) + root) / 2,
        (float(trace) - root) / 2,
    )


def demonstrate_eigenvalues() -> None:
    print_section("9. Eigenvalues and Eigenvectors")

    matrix = [
        [4, 1],
        [2, 3],
    ]

    values = eigenvalues_2x2(matrix)

    print("Matrix:")
    print_matrix(matrix)
    print("eigenvalues =", values)

    print(
        "An eigenvector associated with an eigenvalue λ satisfies "
        "A v = λ v. The matrix changes the vector's magnitude and/or "
        "direction according to λ, but the vector remains on the same line."
    )


# ---------------------------------------------------------------------------
# Tensor implementation
# ---------------------------------------------------------------------------

@dataclass
class Tensor:
    """
    A small educational dense tensor implementation.

    A tensor generalizes scalars, vectors, and matrices:
        rank 0 -> scalar
        rank 1 -> vector
        rank 2 -> matrix
        rank 3+ -> higher-order tensor

    Data is represented recursively as Python lists.
    """

    data: object

    def __post_init__(self) -> None:
        self._shape = self._infer_shape(self.data)

    @classmethod
    def _infer_shape(cls, value: object) -> Tuple[int, ...]:
        if not isinstance(value, list):
            if not isinstance(value, (int, float)):
                raise TypeError("Tensor values must be numeric.")
            return ()

        if len(value) == 0:
            raise ValueError("Tensor dimensions cannot contain empty lists.")

        child_shapes = [cls._infer_shape(child) for child in value]

        if not all(shape == child_shapes[0] for shape in child_shapes):
            raise ValueError("Tensor data must be rectangular.")

        return (len(value),) + child_shapes[0]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def rank(self) -> int:
        return len(self._shape)

    def flatten(self) -> List[Number]:
        output: List[Number] = []

        def visit(value: object) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item)
            else:
                output.append(value)

        visit(self.data)
        return output

    def elementwise_add(self, other: "Tensor") -> "Tensor":
        if self.shape != other.shape:
            raise ValueError("Tensor shapes must match for this operation.")

        def add_values(a: object, b: object) -> object:
            if isinstance(a, list) and isinstance(b, list):
                return [
                    add_values(x, y)
                    for x, y in zip(a, b)
                ]
            return a + b

        return Tensor(add_values(self.data, other.data))

    def scalar_multiply(self, scalar: Number) -> "Tensor":
        if not isinstance(scalar, (int, float)):
            raise TypeError("Tensor scalar must be numeric.")

        def scale(value: object) -> object:
            if isinstance(value, list):
                return [scale(item) for item in value]
            return scalar * value

        return Tensor(scale(self.data))

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.data!r})"


def demonstrate_tensors() -> None:
    print_section("10. Tensors")

    scalar_tensor = Tensor(5)
    vector_tensor = Tensor([1, 2, 3])
    matrix_tensor = Tensor([
        [1, 2],
        [3, 4],
    ])
    three_dimensional_tensor = Tensor([
        [
            [1, 2],
            [3, 4],
        ],
        [
            [5, 6],
            [7, 8],
        ],
    ])

    for name, tensor in [
        ("scalar", scalar_tensor),
        ("vector", vector_tensor),
        ("matrix", matrix_tensor),
        ("3D tensor", three_dimensional_tensor),
    ]:
        print(
            f"{name}: rank={tensor.rank}, shape={tensor.shape}, "
            f"data={tensor.data}"
        )

    other = Tensor([
        [10, 20],
        [30, 40],
    ])

    print("elementwise matrix-tensor addition:")
    print(matrix_tensor.elementwise_add(other))

    print("tensor scalar multiplication:")
    print(matrix_tensor.scalar_multiply(0.5))


# ---------------------------------------------------------------------------
# Tensor shape reasoning and a simple neural-network-style computation
# ---------------------------------------------------------------------------

def relu(values: Sequence[Number]) -> Vector:
    """Rectified Linear Unit: max(0, x)."""
    validate_vector(values)
    return [max(0, value) for value in values]


def matrix_vector_pipeline(
    features: Sequence[Number],
    weights: Sequence[Sequence[Number]],
    bias: Sequence[Number],
) -> Vector:
    """
    A simplified dense neural-network layer.

    If W has shape m x n and x has shape n, then:
        z = W x + b

    The output has m components.
    """
    weighted = apply_matrix_to_vector(weights, features)

    if len(weighted) != len(bias):
        raise ValueError("Bias dimension must match layer output.")

    return relu(vector_add(weighted, bias))


def demonstrate_machine_learning_connection() -> None:
    print_section("11. Linear Algebra in Machine Learning")

    features = [0.5, 1.0, -0.5]

    weights = [
        [1.0, 0.2, -0.4],
        [0.5, -0.3, 0.8],
    ]

    bias = [0.1, -0.2]

    output = matrix_vector_pipeline(features, weights, bias)

    print("features shape:", (len(features),))
    print("weights shape:", matrix_shape(weights))
    print("bias shape:", (len(bias),))
    print("ReLU output:", output)

    print(
        "This demonstrates the central linear-algebra pattern used by a "
        "dense neural-network layer: matrix-vector multiplication followed "
        "by vector addition and a nonlinear activation."
    )


# ---------------------------------------------------------------------------
# Edge cases and numerical considerations
# ---------------------------------------------------------------------------

def demonstrate_edge_cases() -> None:
    print_section("12. Edge Cases and Validation")

    examples = [
        ("zero vector norm", lambda: vector_norm([0, 0, 0])),
        ("invalid matrix multiplication", lambda: matrix_multiply([[1, 2]], [[1, 2]])),
        ("inverse of singular matrix", lambda: inverse_matrix([[1, 2], [2, 4]])),
        ("mismatched vector addition", lambda: vector_add([1, 2], [1, 2, 3])),
        ("projection onto zero vector", lambda: projection_of_a_onto_b([1, 2], [0, 0])),
    ]

    for description, operation in examples:
        try:
            print(description, "->", operation())
        except (ValueError, TypeError) as error:
            print(description, "-> correctly rejected:", error)

    print(
        "\nImportant numerical rule: floating-point calculations should "
        "usually be compared using a tolerance rather than exact equality."
    )

    print(
        "For large-scale numerical linear algebra, specialized numerical "
        "libraries are normally preferred because they provide optimized "
        "memory layouts, vectorized operations, stable decompositions, and "
        "hardware acceleration."
    )


# ---------------------------------------------------------------------------
# Complexity notes
# ---------------------------------------------------------------------------

def demonstrate_complexity_notes() -> None:
    print_section("13. Performance Considerations")

    complexity = {
        "Vector addition": "O(n)",
        "Dot product": "O(n)",
        "Vector norm": "O(n)",
        "Matrix addition": "O(mn)",
        "Matrix-vector multiplication": "O(mn)",
        "Dense matrix multiplication": "O(n^3) for n x n matrices",
        "Gaussian elimination": "O(n^3)",
        "Matrix inversion by elimination": "O(n^3)",
        "Tensor elementwise operation": "O(number of tensor elements)",
    }

    for operation, cost in complexity.items():
        print(f"{operation:45} {cost}")


# ---------------------------------------------------------------------------
# Integrated demonstration
# ---------------------------------------------------------------------------

def integrated_example() -> None:
    print_section("14. Integrated Linear Algebra Example")

    # A 2D point can be represented by a vector.
    point = [3, 4]

    # A matrix represents a linear transformation.
    transformation = [
        [0, -1],
        [1, 0],
    ]

    transformed_point = apply_matrix_to_vector(transformation, point)

    print("original point:", point)
    print("90-degree rotation matrix:")
    print_matrix(transformation)
    print("transformed point:", transformed_point)

    # Applying the transpose reverses this particular orthogonal rotation.
    reverse = transpose(transformation)
    recovered = apply_matrix_to_vector(reverse, transformed_point)

    print("transpose:")
    print_matrix(reverse)
    print("recovered point:", recovered)

    print(
        "The example connects vectors, matrices, matrix-vector "
        "multiplication, transpose, and geometric transformations."
    )


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:
    print("LINEAR ALGEBRA STUDY PROGRAM")
    print("Scalars, vectors, matrices, tensors, and matrix operations")

    demonstrate_scalars()
    demonstrate_vectors()
    demonstrate_matrices()
    demonstrate_determinants()
    demonstrate_linear_systems()
    demonstrate_inverse()
    demonstrate_span_and_independence()
    demonstrate_transformations()
    demonstrate_eigenvalues()
    demonstrate_tensors()
    demonstrate_machine_learning_connection()
    demonstrate_edge_cases()
    demonstrate_complexity_notes()
    integrated_example()

    print_section("15. End of Demonstration")
    print(
        "The implementations above build linear algebra from individual "
        "numbers through vectors and matrices to higher-order tensors, "
        "while connecting algebraic operations to geometry, systems of "
        "equations, transformations, and machine-learning computations."
    )


if __name__ == "__main__":
    main()
