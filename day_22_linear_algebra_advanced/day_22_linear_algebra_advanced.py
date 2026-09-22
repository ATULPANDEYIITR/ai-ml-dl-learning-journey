"""
Linear Algebra Advanced
Topic:
    Dot products, norms, orthogonality, projections, and linear transformations

A self-contained study and practice program progressing from fundamentals
to advanced computational examples.

No third-party packages are required.
"""

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


EPSILON = 1e-10


# ---------------------------------------------------------------------------
# Basic vector utilities
# ---------------------------------------------------------------------------

Vector = List[float]
Matrix = List[List[float]]


def clean_zero(value: float, tolerance: float = EPSILON) -> float:
    """Replace numerically tiny values with exact zero."""
    return 0.0 if abs(value) < tolerance else value


def format_vector(vector: Sequence[float], digits: int = 4) -> str:
    values = [clean_zero(x) for x in vector]
    return "[" + ", ".join(f"{x:.{digits}f}" for x in values) + "]"


def format_matrix(matrix: Matrix, digits: int = 4) -> str:
    return "\n".join(
        "  [" + ", ".join(f"{clean_zero(x):.{digits}f}" for x in row) + "]"
        for row in matrix
    )


def validate_vector(vector: Sequence[float], name: str = "vector") -> None:
    if not vector:
        raise ValueError(f"{name} must not be empty.")
    if not all(isinstance(x, (int, float)) for x in vector):
        raise TypeError(f"{name} must contain only numeric values.")


def same_dimension(a: Sequence[float], b: Sequence[float]) -> None:
    validate_vector(a, "a")
    validate_vector(b, "b")
    if len(a) != len(b):
        raise ValueError(
            f"Vectors must have the same dimension: {len(a)} != {len(b)}."
        )


def add_vectors(a: Sequence[float], b: Sequence[float]) -> Vector:
    same_dimension(a, b)
    return [x + y for x, y in zip(a, b)]


def subtract_vectors(a: Sequence[float], b: Sequence[float]) -> Vector:
    same_dimension(a, b)
    return [x - y for x, y in zip(a, b)]


def scalar_multiply(scalar: float, vector: Sequence[float]) -> Vector:
    validate_vector(vector)
    return [scalar * x for x in vector]


# ---------------------------------------------------------------------------
# Dot products
# ---------------------------------------------------------------------------

def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Compute a dot b.

    Algebraically:
        a · b = a1*b1 + a2*b2 + ... + an*bn

    The dot product is defined for vectors with equal dimensions.
    """
    same_dimension(a, b)
    return sum(x * y for x, y in zip(a, b))


def dot_product_explicit(a: Sequence[float], b: Sequence[float]) -> float:
    """Same operation written explicitly to expose the accumulation process."""
    same_dimension(a, b)
    total = 0.0
    for index in range(len(a)):
        total += a[index] * b[index]
    return total


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Return the angle in radians.

    cos(theta) = (a · b) / (||a|| ||b||)

    Zero vectors have no defined direction, so their angle is undefined.
    """
    denominator = norm(a) * norm(b)
    if denominator < EPSILON:
        raise ValueError("Angle with a zero vector is undefined.")

    cosine = dot_product(a, b) / denominator
    # Floating-point arithmetic can produce 1.00000000001 or -1.00000000001.
    cosine = max(-1.0, min(1.0, cosine))
    return math.acos(cosine)


# ---------------------------------------------------------------------------
# Norms and distances
# ---------------------------------------------------------------------------

def norm(vector: Sequence[float], p: float = 2) -> float:
    """
    Compute the Lp norm.

    p=1  -> Manhattan norm
    p=2  -> Euclidean norm
    p=inf -> maximum absolute coordinate

    For p >= 1 this is a genuine norm.
    """
    validate_vector(vector)

    if p == math.inf:
        return max(abs(x) for x in vector)

    if p <= 0:
        raise ValueError("p must be positive.")

    return sum(abs(x) ** p for x in vector) ** (1.0 / p)


def squared_norm(vector: Sequence[float]) -> float:
    """Return ||v||^2 without calculating a square root."""
    return dot_product(vector, vector)


def normalize(vector: Sequence[float]) -> Vector:
    """
    Convert a nonzero vector into a unit vector.

    A unit vector has norm 1 and preserves direction.
    """
    magnitude = norm(vector)
    if magnitude < EPSILON:
        raise ValueError("The zero vector cannot be normalized.")
    return [x / magnitude for x in vector]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two points/vectors."""
    return norm(subtract_vectors(a, b))


# ---------------------------------------------------------------------------
# Orthogonality and geometric relationships
# ---------------------------------------------------------------------------

def is_orthogonal(a: Sequence[float], b: Sequence[float],
                  tolerance: float = EPSILON) -> bool:
    """Two vectors are orthogonal when their dot product is zero."""
    return abs(dot_product(a, b)) <= tolerance


def is_parallel(a: Sequence[float], b: Sequence[float],
                tolerance: float = EPSILON) -> bool:
    """
    Check whether two vectors are parallel.

    Cross products are only directly available in R^3, so this uses
    pairwise proportionality and works in arbitrary finite dimensions.
    """
    same_dimension(a, b)

    if norm(a) < tolerance or norm(b) < tolerance:
        return True

    ratio = None

    for x, y in zip(a, b):
        if abs(y) <= tolerance:
            if abs(x) > tolerance:
                return False
            continue

        current_ratio = x / y

        if ratio is None:
            ratio = current_ratio
        elif not math.isclose(current_ratio, ratio, abs_tol=tolerance,
                              rel_tol=tolerance):
            return False

    return True


def classify_relationship(a: Sequence[float], b: Sequence[float]) -> str:
    """Classify a pair of vectors geometrically."""
    if norm(a) < EPSILON or norm(b) < EPSILON:
        return "at least one vector is zero"

    if is_orthogonal(a, b):
        return "orthogonal"

    if is_parallel(a, b):
        return "parallel"

    angle = math.degrees(angle_between(a, b))

    if angle < 90:
        return "acute angle"
    if angle > 90:
        return "obtuse angle"
    return "nonzero right angle"


# ---------------------------------------------------------------------------
# Projection and rejection
# ---------------------------------------------------------------------------

def projection_onto_vector(vector: Sequence[float],
                           direction: Sequence[float]) -> Vector:
    """
    Project vector v onto direction u.

    proj_u(v) = ((v · u) / (u · u)) u

    The direction vector must be nonzero.
    """
    same_dimension(vector, direction)

    denominator = dot_product(direction, direction)
    if denominator < EPSILON:
        raise ValueError("Cannot project onto the zero vector.")

    coefficient = dot_product(vector, direction) / denominator
    return scalar_multiply(coefficient, direction)


def projection_onto_unit_vector(vector: Sequence[float],
                                unit_direction: Sequence[float]) -> Vector:
    """
    If u is already a unit vector:

        proj_u(v) = (v · u)u
    """
    if not math.isclose(norm(unit_direction), 1.0, abs_tol=1e-9):
        raise ValueError("The supplied direction is not a unit vector.")

    return scalar_multiply(dot_product(vector, unit_direction), unit_direction)


def rejection_from_vector(vector: Sequence[float],
                          direction: Sequence[float]) -> Vector:
    """
    The rejection is the component perpendicular to direction:

        v = projection + rejection
    """
    return subtract_vectors(vector, projection_onto_vector(vector, direction))


def verify_projection_decomposition(vector: Sequence[float],
                                    direction: Sequence[float]) -> bool:
    projection = projection_onto_vector(vector, direction)
    rejection = rejection_from_vector(vector, direction)

    reconstructed = add_vectors(projection, rejection)

    return (
        all(math.isclose(a, b, abs_tol=1e-9)
            for a, b in zip(vector, reconstructed))
        and is_orthogonal(rejection, direction, 1e-9)
    )


# ---------------------------------------------------------------------------
# Gram-Schmidt orthogonalization
# ---------------------------------------------------------------------------

def gram_schmidt(vectors: Sequence[Sequence[float]]) -> List[Vector]:
    """
    Produce an orthonormal basis using classical Gram-Schmidt.

    For each vector:
        subtract projections onto previously accepted orthonormal vectors
        normalize the remaining component.

    Linearly dependent vectors produce a zero residual and are skipped.
    """
    if not vectors:
        return []

    basis: List[Vector] = []

    for original in vectors:
        validate_vector(original, "input vector")
        working = list(map(float, original))

        for basis_vector in basis:
            working = subtract_vectors(
                working,
                projection_onto_vector(working, basis_vector)
            )

        magnitude = norm(working)

        if magnitude > EPSILON:
            basis.append(normalize(working))

    return basis


def is_orthonormal_set(vectors: Sequence[Sequence[float]],
                       tolerance: float = 1e-8) -> bool:
    for vector in vectors:
        if not math.isclose(norm(vector), 1.0, abs_tol=tolerance):
            return False

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if not is_orthogonal(vectors[i], vectors[j], tolerance):
                return False

    return True


# ---------------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------------

def validate_matrix(matrix: Matrix, name: str = "matrix") -> None:
    if not matrix:
        raise ValueError(f"{name} must not be empty.")

    row_lengths = {len(row) for row in matrix}

    if 0 in row_lengths:
        raise ValueError(f"{name} must not contain empty rows.")

    if len(row_lengths) != 1:
        raise ValueError(f"{name} must be rectangular.")


def matrix_shape(matrix: Matrix) -> Tuple[int, int]:
    validate_matrix(matrix)
    return len(matrix), len(matrix[0])


def identity_matrix(size: int) -> Matrix:
    if size <= 0:
        raise ValueError("Matrix size must be positive.")

    return [
        [1.0 if i == j else 0.0 for j in range(size)]
        for i in range(size)
    ]


def matrix_vector_multiply(matrix: Matrix,
                           vector: Sequence[float]) -> Vector:
    """
    Apply a linear transformation represented by matrix A to vector x.

        y = Ax

    If A is m x n, x must have n components and y has m components.
    """
    validate_matrix(matrix)
    validate_vector(vector)

    rows, columns = matrix_shape(matrix)

    if columns != len(vector):
        raise ValueError(
            f"Matrix has {columns} columns but vector has {len(vector)} values."
        )

    return [
        sum(matrix[row][column] * vector[column] for column in range(columns))
        for row in range(rows)
    ]


def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    """Multiply compatible matrices using the definition of matrix product."""
    validate_matrix(a, "a")
    validate_matrix(b, "b")

    rows_a, columns_a = matrix_shape(a)
    rows_b, columns_b = matrix_shape(b)

    if columns_a != rows_b:
        raise ValueError(
            "Matrix multiplication requires columns of A to equal rows of B."
        )

    return [
        [
            sum(a[i][k] * b[k][j] for k in range(columns_a))
            for j in range(columns_b)
        ]
        for i in range(rows_a)
    ]


def transpose(matrix: Matrix) -> Matrix:
    validate_matrix(matrix)
    rows, columns = matrix_shape(matrix)

    return [
        [matrix[row][column] for row in range(rows)]
        for column in range(columns)
    ]


def matrix_add(a: Matrix, b: Matrix) -> Matrix:
    if matrix_shape(a) != matrix_shape(b):
        raise ValueError("Matrices must have the same shape.")

    return [
        [x + y for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)
    ]


def matrix_scalar_multiply(scalar: float, matrix: Matrix) -> Matrix:
    validate_matrix(matrix)

    return [
        [scalar * value for value in row]
        for row in matrix
    ]


# ---------------------------------------------------------------------------
# Linear transformations
# ---------------------------------------------------------------------------

def rotation_2d(theta_degrees: float) -> Matrix:
    """
    Counterclockwise rotation matrix in R^2.

        [cos(theta) -sin(theta)]
        [sin(theta)  cos(theta)]
    """
    theta = math.radians(theta_degrees)
    c = math.cos(theta)
    s = math.sin(theta)

    return [
        [c, -s],
        [s, c],
    ]


def scaling_2d(scale_x: float, scale_y: float) -> Matrix:
    return [
        [scale_x, 0.0],
        [0.0, scale_y],
    ]


def reflection_x_axis() -> Matrix:
    return [
        [1.0, 0.0],
        [0.0, -1.0],
    ]


def shear_x_2d(amount: float) -> Matrix:
    """
    Horizontal shear:

        x' = x + k*y
        y' = y
    """
    return [
        [1.0, amount],
        [0.0, 1.0],
    ]


def apply_transformation(matrix: Matrix,
                         vector: Sequence[float]) -> Vector:
    return matrix_vector_multiply(matrix, vector)


def preserves_dot_products(matrix: Matrix,
                           vectors: Sequence[Sequence[float]],
                           tolerance: float = 1e-8) -> bool:
    """
    Test whether A preserves dot products for the supplied vectors.

    An orthogonal matrix satisfies:
        (Ax) · (Ay) = x · y

    for all vectors x and y.
    """
    for x in vectors:
        for y in vectors:
            transformed_x = matrix_vector_multiply(matrix, x)
            transformed_y = matrix_vector_multiply(matrix, y)

            if not math.isclose(
                dot_product(transformed_x, transformed_y),
                dot_product(x, y),
                abs_tol=tolerance
            ):
                return False

    return True


def preserves_norm(matrix: Matrix,
                   vectors: Sequence[Sequence[float]],
                   tolerance: float = 1e-8) -> bool:
    return all(
        math.isclose(
            norm(matrix_vector_multiply(matrix, vector)),
            norm(vector),
            abs_tol=tolerance
        )
        for vector in vectors
    )


# ---------------------------------------------------------------------------
# Least-squares projection onto a subspace
# ---------------------------------------------------------------------------

def coordinates_in_orthonormal_basis(
    vector: Sequence[float],
    orthonormal_basis: Sequence[Sequence[float]]
) -> Vector:
    """
    Coordinates of v in an orthonormal basis are simply dot products:

        c_i = v · q_i
    """
    return [
        dot_product(vector, basis_vector)
        for basis_vector in orthonormal_basis
    ]


def reconstruct_from_orthonormal_basis(
    coordinates: Sequence[float],
    orthonormal_basis: Sequence[Sequence[float]]
) -> Vector:
    if len(coordinates) != len(orthonormal_basis):
        raise ValueError("Coordinate and basis counts must match.")

    if not orthonormal_basis:
        raise ValueError("Basis cannot be empty.")

    result = [0.0] * len(orthonormal_basis[0])

    for coefficient, basis_vector in zip(coordinates, orthonormal_basis):
        result = add_vectors(
            result,
            scalar_multiply(coefficient, basis_vector)
        )

    return result


def project_onto_subspace(
    vector: Sequence[float],
    spanning_vectors: Sequence[Sequence[float]]
) -> Vector:
    """
    Project v onto span(spanning_vectors).

    Gram-Schmidt creates an orthonormal basis Q. The projection is:

        proj_W(v) = sum_i (v · q_i) q_i
    """
    basis = gram_schmidt(spanning_vectors)

    if not basis:
        return [0.0] * len(vector)

    return reconstruct_from_orthonormal_basis(
        coordinates_in_orthonormal_basis(vector, basis),
        basis
    )


# ---------------------------------------------------------------------------
# Least squares using QR-like orthogonalization
# ---------------------------------------------------------------------------

def least_squares_solution(
    design_matrix: Matrix,
    observations: Sequence[float]
) -> Vector:
    """
    Solve an overdetermined system approximately.

    Given A x ~= b, construct orthonormal columns Q from A.

    A = QR

    Then:
        QRx = b
        Rx = Q^T b

    This implementation assumes independent columns.
    """
    validate_matrix(design_matrix, "design_matrix")
    validate_vector(observations, "observations")

    rows, columns = matrix_shape(design_matrix)

    if rows != len(observations):
        raise ValueError("Observation count must equal matrix row count.")

    columns_as_vectors = [
        [design_matrix[row][column] for row in range(rows)]
        for column in range(columns)
    ]

    q_columns = gram_schmidt(columns_as_vectors)

    if len(q_columns) != columns:
        raise ValueError(
            "Design matrix columns are linearly dependent; "
            "a unique least-squares solution is unavailable."
        )

    # R = Q^T A
    r = [
        [
            dot_product(q_columns[i], columns_as_vectors[j])
            for j in range(columns)
        ]
        for i in range(columns)
    ]

    # Q^T b
    qtb = [
        dot_product(q, observations)
        for q in q_columns
    ]

    # Back substitution for upper triangular R.
    x = [0.0] * columns

    for i in range(columns - 1, -1, -1):
        known = sum(r[i][j] * x[j] for j in range(i + 1, columns))

        if abs(r[i][i]) < EPSILON:
            raise ValueError("Singular triangular system.")

        x[i] = (qtb[i] - known) / r[i][i]

    return x


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_fundamentals() -> None:
    print("\n" + "=" * 80)
    print("1. VECTOR FUNDAMENTALS")
    print("=" * 80)

    a = [3, 4]
    b = [1, 2]

    print("a =", format_vector(a))
    print("b =", format_vector(b))
    print("a + b =", format_vector(add_vectors(a, b)))
    print("a - b =", format_vector(subtract_vectors(a, b)))
    print("3a =", format_vector(scalar_multiply(3, a)))

    print("\na · b =", dot_product(a, b))
    print("||a|| =", norm(a))
    print("||a||² =", squared_norm(a))
    print("distance(a, b) =", distance(a, b))
    print("unit(a) =", format_vector(normalize(a)))


def demo_dot_products_and_geometry() -> None:
    print("\n" + "=" * 80)
    print("2. DOT PRODUCTS AND GEOMETRY")
    print("=" * 80)

    examples = [
        ([1, 0], [0, 1]),
        ([1, 1], [2, 2]),
        ([1, 2], [-2, 1]),
        ([2, 1], [3, 4]),
    ]

    for a, b in examples:
        print(
            f"{format_vector(a)} and {format_vector(b)} -> "
            f"dot={dot_product(a, b):.4f}, "
            f"relationship={classify_relationship(a, b)}"
        )

    angle = math.degrees(angle_between([1, 0], [1, 1]))
    print(f"\nAngle between [1,0] and [1,1]: {angle:.2f} degrees")


def demo_norms() -> None:
    print("\n" + "=" * 80)
    print("3. DIFFERENT NORMS")
    print("=" * 80)

    vector = [-3, 4, -2]

    for p in [1, 2, 3, math.inf]:
        label = "infinity" if p == math.inf else str(p)
        print(f"L{label} norm: {norm(vector, p):.6f}")

    print(
        "\nThe L1 norm sums absolute coordinates, "
        "the L2 norm is Euclidean length, and "
        "the infinity norm selects the largest absolute coordinate."
    )


def demo_projection() -> None:
    print("\n" + "=" * 80)
    print("4. PROJECTIONS")
    print("=" * 80)

    v = [4, 3]
    u = [2, 0]

    projection = projection_onto_vector(v, u)
    rejection = rejection_from_vector(v, u)

    print("v =", format_vector(v))
    print("u =", format_vector(u))
    print("projection =", format_vector(projection))
    print("rejection =", format_vector(rejection))
    print("projection + rejection =", format_vector(
        add_vectors(projection, rejection)
    ))
    print("rejection orthogonal to u:", is_orthogonal(rejection, u))
    print("decomposition verified:", verify_projection_decomposition(v, u))

    # Projection onto a non-axis direction.
    v2 = [5, 1]
    u2 = [1, 2]
    p2 = projection_onto_vector(v2, u2)

    print("\nNon-axis projection:")
    print("v =", format_vector(v2))
    print("u =", format_vector(u2))
    print("proj_u(v) =", format_vector(p2))


def demo_gram_schmidt() -> None:
    print("\n" + "=" * 80)
    print("5. GRAM-SCHMIDT ORTHOGONALIZATION")
    print("=" * 80)

    vectors = [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]

    orthonormal_basis = gram_schmidt(vectors)

    print("Original vectors:")
    for vector in vectors:
        print(" ", format_vector(vector))

    print("\nOrthonormal basis:")
    for vector in orthonormal_basis:
        print(" ", format_vector(vector), "norm =", norm(vector))

    print("\nIs orthonormal:", is_orthonormal_set(orthonormal_basis))

    print("\nPairwise dot products:")
    for i in range(len(orthonormal_basis)):
        for j in range(i + 1, len(orthonormal_basis)):
            print(
                f"q{i+1} · q{j+1} = "
                f"{dot_product(orthonormal_basis[i], orthonormal_basis[j]):.8f}"
            )


def demo_subspace_projection() -> None:
    print("\n" + "=" * 80)
    print("6. PROJECTION ONTO A SUBSPACE")
    print("=" * 80)

    vector = [3, 4, 5]
    spanning_vectors = [
        [1, 0, 1],
        [0, 1, 1],
    ]

    projection = project_onto_subspace(vector, spanning_vectors)
    residual = subtract_vectors(vector, projection)

    print("Vector:", format_vector(vector))
    print("Subspace spanning vectors:")

    for item in spanning_vectors:
        print(" ", format_vector(item))

    print("Projection:", format_vector(projection))
    print("Residual:", format_vector(residual))

    for basis_vector in spanning_vectors:
        print(
            "Residual orthogonal to span vector:",
            is_orthogonal(residual, basis_vector, 1e-8)
        )


def demo_linear_transformations() -> None:
    print("\n" + "=" * 80)
    print("7. LINEAR TRANSFORMATIONS")
    print("=" * 80)

    point = [2, 1]

    transformations = {
        "rotation 90°": rotation_2d(90),
        "scaling": scaling_2d(2, 3),
        "reflection across x-axis": reflection_x_axis(),
        "horizontal shear": shear_x_2d(1.5),
    }

    print("Original point:", format_vector(point))

    for name, matrix in transformations.items():
        transformed = apply_transformation(matrix, point)

        print(f"\n{name}")
        print(format_matrix(matrix))
        print("result:", format_vector(transformed))

    rotation = rotation_2d(45)
    test_vectors = [
        [1, 2],
        [3, -1],
        [-2, 4],
    ]

    print(
        "\nRotation preserves dot products:",
        preserves_dot_products(rotation, test_vectors)
    )
    print(
        "Rotation preserves norms:",
        preserves_norm(rotation, test_vectors)
    )

    scaling = scaling_2d(2, 2)

    print(
        "Uniform scaling by 2 preserves dot products:",
        preserves_dot_products(scaling, test_vectors)
    )


def demo_composition() -> None:
    print("\n" + "=" * 80)
    print("8. COMPOSITION OF LINEAR TRANSFORMATIONS")
    print("=" * 80)

    rotation = rotation_2d(90)
    scaling = scaling_2d(2, 3)

    point = [1, 2]

    first_scale_then_rotate = matrix_multiply(rotation, scaling)
    first_rotate_then_scale = matrix_multiply(scaling, rotation)

    result_a = matrix_vector_multiply(first_scale_then_rotate, point)
    result_b = matrix_vector_multiply(first_rotate_then_scale, point)

    print("point:", format_vector(point))

    print("\nRotation × Scaling:")
    print(format_matrix(first_scale_then_rotate))
    print("result:", format_vector(result_a))

    print("\nScaling × Rotation:")
    print(format_matrix(first_rotate_then_scale))
    print("result:", format_vector(result_b))

    print(
        "\nThe results differ because matrix multiplication is generally "
        "not commutative."
    )


def demo_least_squares() -> None:
    print("\n" + "=" * 80)
    print("9. LEAST-SQUARES REGRESSION THROUGH ORTHOGONAL PROJECTION")
    print("=" * 80)

    # Model:
    # y = beta0 + beta1*x
    #
    # Each row of A is [1, x].
    # We solve A beta ~= y.
    x_values = [0, 1, 2, 3, 4]
    y_values = [1.1, 2.9, 5.2, 6.8, 9.1]

    design = [[1.0, x] for x in x_values]

    coefficients = least_squares_solution(design, y_values)

    intercept, slope = coefficients

    print("Observed data:")
    for x, y in zip(x_values, y_values):
        print(f"  x={x}, y={y}")

    print("\nLeast-squares model:")
    print(f"  y ≈ {intercept:.6f} + {slope:.6f}x")

    predictions = [
        intercept + slope * x
        for x in x_values
    ]

    residuals = [
        observed - predicted
        for observed, predicted in zip(y_values, predictions)
    ]

    print("\nPredictions:")
    for x, observed, predicted, residual in zip(
        x_values, y_values, predictions, residuals
    ):
        print(
            f"  x={x}: observed={observed:.3f}, "
            f"predicted={predicted:.3f}, residual={residual:.3f}"
        )

    print(
        "\nResidual norm:",
        f"{norm(residuals):.6f}"
    )


def demo_edge_cases() -> None:
    print("\n" + "=" * 80)
    print("10. EDGE CASES AND VALIDATION")
    print("=" * 80)

    cases = [
        ("zero vector normalization", lambda: normalize([0, 0])),
        ("projection onto zero vector",
         lambda: projection_onto_vector([1, 2], [0, 0])),
        ("dimension mismatch",
         lambda: dot_product([1, 2], [1, 2, 3])),
        ("invalid norm p",
         lambda: norm([1, 2], 0)),
    ]

    for name, operation in cases:
        try:
            operation()
        except (ValueError, TypeError) as error:
            print(f"{name}: correctly rejected -> {error}")


# ---------------------------------------------------------------------------
# A small practical simulation
# ---------------------------------------------------------------------------

@dataclass
class SensorMeasurement:
    timestamp: int
    vector: Vector


def detect_directional_alignment(
    measurements: Sequence[SensorMeasurement],
    reference_direction: Sequence[float]
) -> List[Tuple[int, float, str]]:
    """
    Compare sensor vectors with a reference direction.

    Cosine similarity:
        cos(theta) = (v · r)/(||v|| ||r||)

    This is useful when magnitude and direction have different meanings.
    """
    results = []

    reference_norm = norm(reference_direction)

    if reference_norm < EPSILON:
        raise ValueError("Reference direction cannot be zero.")

    for measurement in measurements:
        vector_norm = norm(measurement.vector)

        if vector_norm < EPSILON:
            similarity = 0.0
            category = "zero measurement"
        else:
            similarity = (
                dot_product(measurement.vector, reference_direction)
                / (vector_norm * reference_norm)
            )

            if similarity >= 0.8:
                category = "strong alignment"
            elif similarity >= 0.3:
                category = "moderate alignment"
            elif similarity > -0.3:
                category = "approximately perpendicular"
            elif similarity > -0.8:
                category = "moderate opposition"
            else:
                category = "strong opposition"

        results.append(
            (measurement.timestamp, similarity, category)
        )

    return results


def demo_sensor_application() -> None:
    print("\n" + "=" * 80)
    print("11. PRACTICAL APPLICATION: DIRECTIONAL SENSOR ANALYSIS")
    print("=" * 80)

    random.seed(7)

    reference = [1.0, 0.0, 0.0]

    measurements = [
        SensorMeasurement(1, [1.0, 0.1, 0.0]),
        SensorMeasurement(2, [0.8, 0.5, 0.1]),
        SensorMeasurement(3, [0.0, 1.0, 0.0]),
        SensorMeasurement(4, [-0.9, 0.2, 0.0]),
        SensorMeasurement(5, [0.0, 0.0, 0.0]),
    ]

    results = detect_directional_alignment(measurements, reference)

    print("Reference direction:", format_vector(reference))

    for timestamp, similarity, category in results:
        print(
            f"measurement {timestamp}: "
            f"cosine similarity={similarity:.4f}, {category}"
        )


# ---------------------------------------------------------------------------
# Verification suite
# ---------------------------------------------------------------------------

def run_assertion_tests() -> None:
    print("\n" + "=" * 80)
    print("12. SELF-TESTS")
    print("=" * 80)

    assert dot_product([1, 2], [3, 4]) == 11
    assert math.isclose(norm([3, 4]), 5.0)
    assert is_orthogonal([1, 0], [0, 1])
    assert is_parallel([1, 2], [2, 4])

    projection = projection_onto_vector([3, 4], [1, 0])
    assert all(
        math.isclose(a, b, abs_tol=1e-9)
        for a, b in zip(projection, [3, 0])
    )

    basis = gram_schmidt([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    assert is_orthonormal_set(basis)

    rotation = rotation_2d(90)
    transformed = matrix_vector_multiply(rotation, [1, 0])

    assert math.isclose(transformed[0], 0.0, abs_tol=1e-9)
    assert math.isclose(transformed[1], 1.0, abs_tol=1e-9)

    print("All tests passed.")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("LINEAR ALGEBRA ADVANCED STUDY PROGRAM")
    print("Dot Products, Norms, Orthogonality, Projections, Linear Transformations")
    print("=" * 80)

    demo_fundamentals()
    demo_dot_products_and_geometry()
    demo_norms()
    demo_projection()
    demo_gram_schmidt()
    demo_subspace_projection()
    demo_linear_transformations()
    demo_composition()
    demo_least_squares()
    demo_edge_cases()
    demo_sensor_application()
    run_assertion_tests()

    print("\n" + "=" * 80)
    print("PROGRAM COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
