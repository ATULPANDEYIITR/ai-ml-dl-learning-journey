"""
Kernel Methods: kernel trick, polynomial kernels, RBF kernels, and similarity spaces.

A self-contained implementation using only the Python standard library.
The script:
- represents feature vectors and similarity spaces,
- implements linear, polynomial, and RBF kernels,
- builds Gram matrices,
- validates positive-semidefinite behavior numerically,
- implements kernel nearest-neighbor classification,
- implements kernel ridge regression using the Gram matrix,
- demonstrates why an implicit feature space can be useful.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Iterable, Sequence


Vector = tuple[float, ...]
Kernel = Callable[[Vector, Vector], float]


def validate_vector(x: Sequence[float], name: str = "vector") -> Vector:
    """Convert a numeric sequence to an immutable vector and validate it."""
    if not x:
        raise ValueError(f"{name} must not be empty")

    result = tuple(float(value) for value in x)

    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains a non-finite value")

    return result


def validate_same_dimension(x: Vector, y: Vector) -> None:
    if len(x) != len(y):
        raise ValueError(
            f"dimension mismatch: {len(x)} != {len(y)}"
        )


def dot(x: Vector, y: Vector) -> float:
    validate_same_dimension(x, y)
    return sum(a * b for a, b in zip(x, y))


def squared_distance(x: Vector, y: Vector) -> float:
    validate_same_dimension(x, y)
    return sum((a - b) ** 2 for a, b in zip(x, y))


def linear_kernel(x: Vector, y: Vector) -> float:
    """Standard inner product."""
    return dot(x, y)


def polynomial_kernel(
    x: Vector,
    y: Vector,
    degree: int = 3,
    gamma: float | None = None,
    coefficient: float = 1.0,
) -> float:
    """
    Polynomial kernel:

        K(x, y) = (gamma * <x,y> + coefficient)^degree

    gamma defaults to 1 / number_of_features, a common normalization.
    """
    validate_same_dimension(x, y)

    if degree < 1:
        raise ValueError("degree must be at least 1")

    if gamma is None:
        gamma = 1.0 / len(x)

    if gamma <= 0:
        raise ValueError("gamma must be positive")

    if coefficient < 0:
        raise ValueError("coefficient must be non-negative")

    return (gamma * dot(x, y) + coefficient) ** degree


def rbf_kernel(
    x: Vector,
    y: Vector,
    gamma: float | None = None,
) -> float:
    """
    Radial basis function kernel:

        K(x,y) = exp(-gamma * ||x-y||^2)

    Unlike an explicit polynomial feature expansion, the RBF kernel
    represents similarity through distance without constructing the
    usually very large feature vector.
    """
    validate_same_dimension(x, y)

    if gamma is None:
        gamma = 1.0 / len(x)

    if gamma <= 0:
        raise ValueError("gamma must be positive")

    return math.exp(-gamma * squared_distance(x, y))


def gram_matrix(
    samples: Sequence[Vector],
    kernel: Kernel,
) -> list[list[float]]:
    """Construct the symmetric Gram matrix K_ij = kernel(x_i, x_j)."""
    if not samples:
        raise ValueError("at least one sample is required")

    normalized = [
        validate_vector(sample, f"sample[{index}]")
        for index, sample in enumerate(samples)
    ]

    dimension = len(normalized[0])

    if any(len(sample) != dimension for sample in normalized):
        raise ValueError("all samples must have the same dimension")

    matrix = [[0.0 for _ in normalized] for _ in normalized]

    for i in range(len(normalized)):
        for j in range(i, len(normalized)):
            value = kernel(normalized[i], normalized[j])
            matrix[i][j] = value
            matrix[j][i] = value

    return matrix


def cholesky_psd_check(
    matrix: Sequence[Sequence[float]],
    tolerance: float = 1e-9,
) -> bool:
    """
    Numerically test whether a symmetric matrix is positive semidefinite.

    A Gram matrix of a valid Mercer kernel should be PSD. Small negative
    values caused by floating-point arithmetic are tolerated.
    """
    n = len(matrix)

    if n == 0:
        return True

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")

    lower = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j]

            for k in range(j):
                value -= lower[i][k] * lower[j][k]

            if i == j:
                if value < -tolerance:
                    return False
                lower[i][j] = math.sqrt(max(0.0, value))
            else:
                pivot = lower[j][j]
                if abs(pivot) <= tolerance:
                    if abs(value) > tolerance:
                        return False
                    lower[i][j] = 0.0
                else:
                    lower[i][j] = value / pivot

    return True


def solve_linear_system(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> list[float]:
    """
    Gaussian elimination with partial pivoting.

    Used here for the small educational kernel-ridge example.
    """
    n = len(matrix)

    if n == 0 or len(vector) != n:
        raise ValueError("incompatible linear system")

    augmented = [
        [float(value) for value in matrix[i]] + [float(vector[i])]
        for i in range(n)
    ]

    if any(len(row) != n + 1 for row in augmented):
        raise ValueError("matrix must be square")

    for column in range(n):
        pivot_row = max(
            range(column, n),
            key=lambda row: abs(augmented[row][column]),
        )

        pivot = augmented[pivot_row][column]

        if abs(pivot) < 1e-12:
            raise ValueError("singular or numerically unstable system")

        if pivot_row != column:
            augmented[column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[column],
            )

        for row in range(column + 1, n):
            factor = augmented[row][column] / augmented[column][column]

            for k in range(column, n + 1):
                augmented[row][k] -= factor * augmented[column][k]

    solution = [0.0] * n

    for row in range(n - 1, -1, -1):
        remainder = augmented[row][n]

        for column in range(row + 1, n):
            remainder -= augmented[row][column] * solution[column]

        solution[row] = remainder / augmented[row][row]

    return solution


@dataclass(frozen=True)
class KernelKNN:
    """Nearest-neighbor classifier using kernel similarity."""

    samples: tuple[Vector, ...]
    labels: tuple[int, ...]
    kernel: Kernel

    def __post_init__(self) -> None:
        if len(self.samples) != len(self.labels):
            raise ValueError("samples and labels must have equal length")

        if not self.samples:
            raise ValueError("classifier requires training samples")

        dimension = len(self.samples[0])

        if any(len(sample) != dimension for sample in self.samples):
            raise ValueError("training samples have inconsistent dimensions")

    def predict(self, query: Vector) -> int:
        query = validate_vector(query, "query")

        scores: dict[int, float] = {}

        for sample, label in zip(self.samples, self.labels):
            similarity = self.kernel(sample, query)
            scores[label] = scores.get(label, 0.0) + similarity

        return max(scores, key=lambda label: scores[label])


@dataclass
class KernelRidge:
    """
    Kernel ridge regression.

    Training solves:

        (K + lambda I) alpha = y

    Prediction is:

        f(x) = sum_i alpha_i K(x_i, x)
    """

    kernel: Kernel
    regularization: float = 0.1

    def __post_init__(self) -> None:
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")

        self._samples: list[Vector] = []
        self._alpha: list[float] = []

    def fit(
        self,
        samples: Sequence[Vector],
        targets: Sequence[float],
    ) -> None:
        if len(samples) != len(targets):
            raise ValueError("samples and targets must have equal length")

        if not samples:
            raise ValueError("at least one training sample is required")

        self._samples = [
            validate_vector(sample, f"sample[{i}]")
            for i, sample in enumerate(samples)
        ]

        kernel_matrix = gram_matrix(self._samples, self.kernel)
        n = len(kernel_matrix)

        regularized = [
            [
                kernel_matrix[i][j]
                + (self.regularization if i == j else 0.0)
                for j in range(n)
            ]
            for i in range(n)
        ]

        self._alpha = solve_linear_system(
            regularized,
            [float(target) for target in targets],
        )

    def predict(self, query: Vector) -> float:
        if not self._alpha:
            raise RuntimeError("model has not been fitted")

        query = validate_vector(query, "query")

        return sum(
            coefficient * self.kernel(sample, query)
            for coefficient, sample in zip(self._alpha, self._samples)
        )


def make_xor_dataset() -> tuple[list[Vector], list[int]]:
    """
    XOR is linearly non-separable in the original two-dimensional space.

    A nonlinear kernel can measure similarity in an implicit feature space
    where nonlinear relationships become easier to model.
    """
    samples = [
        (-1.0, -1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-0.8, -1.1),
        (-1.1, 0.8),
        (0.9, -1.2),
        (1.1, 1.0),
    ]

    labels = [0, 1, 1, 0, 0, 1, 1, 0]

    return samples, labels


def demo_similarity_space() -> None:
    samples = [
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
    ]

    print("Similarity values")

    for name, kernel in (
        ("linear", linear_kernel),
        (
            "polynomial",
            lambda x, y: polynomial_kernel(
                x,
                y,
                degree=3,
                gamma=0.5,
                coefficient=1.0,
            ),
        ),
        (
            "rbf",
            lambda x, y: rbf_kernel(
                x,
                y,
                gamma=0.5,
            ),
        ),
    ):
        print(f"\n{name}")
        matrix = gram_matrix(samples, kernel)

        for row in matrix:
            print(" ".join(f"{value:8.4f}" for value in row))

        print(f"PSD check: {cholesky_psd_check(matrix)}")


def demo_kernel_classification() -> None:
    samples, labels = make_xor_dataset()

    classifier = KernelKNN(
        samples=tuple(samples),
        labels=tuple(labels),
        kernel=lambda x, y: rbf_kernel(x, y, gamma=2.0),
    )

    queries = [
        (-0.9, -0.9),
        (-0.9, 0.9),
        (0.9, -0.9),
        (0.9, 0.9),
    ]

    print("\nRBF kernel similarity classification")

    for query in queries:
        prediction = classifier.predict(query)
        print(f"{query} -> class {prediction}")


def demo_kernel_ridge() -> None:
    samples = [
        (-2.0,),
        (-1.0,),
        (0.0,),
        (1.0,),
        (2.0,),
    ]

    targets = [4.0, 1.0, 0.0, 1.0, 4.0]

    model = KernelRidge(
        kernel=lambda x, y: rbf_kernel(x, y, gamma=0.7),
        regularization=0.05,
    )

    model.fit(samples, targets)

    print("\nKernel ridge regression")

    for value in (-1.5, -0.5, 0.5, 1.5):
        prediction = model.predict((value,))
        print(f"x={value:4.1f} -> prediction={prediction:8.4f}")


def demo_edge_cases() -> None:
    print("\nValidation examples")

    try:
        polynomial_kernel((1.0, 2.0), (1.0,), degree=2)
    except ValueError as exc:
        print(f"dimension validation: {exc}")

    try:
        rbf_kernel((1.0,), (2.0,), gamma=0.0)
    except ValueError as exc:
        print(f"gamma validation: {exc}")

    try:
        validate_vector((1.0, math.inf))
    except ValueError as exc:
        print(f"finite-value validation: {exc}")


def main() -> None:
    random.seed(42)

    demo_similarity_space()
    demo_kernel_classification()
    demo_kernel_ridge()
    demo_edge_cases()

    print("\nKernel method workflow completed successfully.")


if __name__ == "__main__":
    main()
