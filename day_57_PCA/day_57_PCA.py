from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import math
import statistics


class PCAError(ValueError):
    """Raised when PCA input or numerical assumptions are invalid."""


@dataclass(frozen=True)
class PCAResult:
    means: list[float]
    covariance: list[list[float]]
    eigenvalues: list[float]
    eigenvectors: list[list[float]]
    explained_variance_ratio: list[float]
    transformed: list[list[float]]
    components: int

    @property
    def cumulative_explained_variance(self) -> list[float]:
        total = 0.0
        result = []
        for ratio in self.explained_variance_ratio:
            total += ratio
            result.append(total)
        return result


def validate_matrix(data: list[list[float]]) -> None:
    if not data:
        raise PCAError("Dataset must contain at least one observation.")
    width = len(data[0])
    if width == 0:
        raise PCAError("Dataset must contain at least one feature.")

    for row_number, row in enumerate(data, start=1):
        if len(row) != width:
            raise PCAError(f"Row {row_number} has an inconsistent number of features.")
        for value in row:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise PCAError("All observations must be finite numeric values.")

    if len(data) < 2:
        raise PCAError("At least two observations are required for sample covariance.")


def mean_center(data: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    validate_matrix(data)
    dimensions = len(data[0])
    means = [
        statistics.fmean(row[column] for row in data)
        for column in range(dimensions)
    ]
    centered = [
        [row[column] - means[column] for column in range(dimensions)]
        for row in data
    ]
    return means, centered


def covariance_matrix(centered: list[list[float]]) -> list[list[float]]:
    observations = len(centered)
    dimensions = len(centered[0])
    denominator = observations - 1

    covariance = [[0.0] * dimensions for _ in range(dimensions)]

    for i in range(dimensions):
        for j in range(i, dimensions):
            value = sum(row[i] * row[j] for row in centered) / denominator
            covariance[i][j] = value
            covariance[j][i] = value

    return covariance


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(dot(vector, vector))
    if length < 1e-14:
        raise PCAError("Encountered a numerically zero eigenvector.")
    return [value / length for value in vector]


def mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def power_iteration(
    matrix: list[list[float]],
    max_iterations: int = 10_000,
    tolerance: float = 1e-12,
) -> tuple[float, list[float]]:
    n = len(matrix)
    vector = normalize([1.0 + i / n for i in range(n)])

    eigenvalue = 0.0

    for _ in range(max_iterations):
        candidate = normalize(mat_vec(matrix, vector))
        new_eigenvalue = dot(candidate, mat_vec(matrix, candidate))

        difference = math.sqrt(
            sum((candidate[i] - vector[i]) ** 2 for i in range(n))
        )

        vector = candidate
        if abs(new_eigenvalue - eigenvalue) < tolerance and difference < tolerance:
            eigenvalue = new_eigenvalue
            break

        eigenvalue = new_eigenvalue
    else:
        raise PCAError("Power iteration did not converge.")

    # Eigenvector sign is mathematically arbitrary. This convention
    # makes repeated executions easier to inspect.
    pivot = max(range(n), key=lambda i: abs(vector[i]))
    if vector[pivot] < 0:
        vector = [-value for value in vector]

    return eigenvalue, vector


def deflate(
    matrix: list[list[float]],
    eigenvalue: float,
    eigenvector: list[float],
) -> list[list[float]]:
    n = len(matrix)
    return [
        [
            matrix[i][j] - eigenvalue * eigenvector[i] * eigenvector[j]
            for j in range(n)
        ]
        for i in range(n)
    ]


def eigen_decomposition(
    symmetric_matrix: list[list[float]],
) -> tuple[list[float], list[list[float]]]:
    matrix = [row[:] for row in symmetric_matrix]
    n = len(matrix)
    eigenvalues: list[float] = []
    eigenvectors: list[list[float]] = []

    for _ in range(n):
        value, vector = power_iteration(matrix)

        # Small negative values can arise from floating-point error.
        if abs(value) < 1e-12:
            value = 0.0

        eigenvalues.append(value)
        eigenvectors.append(vector)
        matrix = deflate(matrix, value, vector)

    order = sorted(range(n), key=lambda i: eigenvalues[i], reverse=True)

    sorted_values = [eigenvalues[i] for i in order]
    sorted_vectors = [eigenvectors[i] for i in order]

    return sorted_values, sorted_vectors


def transform(
    centered: list[list[float]],
    eigenvectors: list[list[float]],
    components: int,
) -> list[list[float]]:
    return [
        [
            dot(row, eigenvectors[component])
            for component in range(components)
        ]
        for row in centered
    ]


def fit_pca(data: list[list[float]], components: int | None = None) -> PCAResult:
    means, centered = mean_center(data)
    covariance = covariance_matrix(centered)
    eigenvalues, eigenvectors = eigen_decomposition(covariance)

    dimensions = len(covariance)
    if components is None:
        components = dimensions

    if not 1 <= components <= dimensions:
        raise PCAError("components must be between 1 and the number of features.")

    total_variance = sum(max(value, 0.0) for value in eigenvalues)
    if total_variance <= 1e-14:
        raise PCAError(
            "Total variance is zero. PCA cannot identify meaningful directions."
        )

    ratios = [
        max(value, 0.0) / total_variance
        for value in eigenvalues
    ]

    projected = transform(centered, eigenvectors, components)

    return PCAResult(
        means=means,
        covariance=covariance,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_variance_ratio=ratios,
        transformed=projected,
        components=components,
    )


def standardize(data: list[list[float]]) -> list[list[float]]:
    validate_matrix(data)
    dimensions = len(data[0])
    means = [
        statistics.fmean(row[j] for row in data)
        for j in range(dimensions)
    ]
    standard_deviations = []

    for j in range(dimensions):
        variance = sum(
            (row[j] - means[j]) ** 2 for row in data
        ) / (len(data) - 1)
        standard_deviations.append(math.sqrt(variance))

    if any(sd < 1e-14 for sd in standard_deviations):
        raise PCAError("Standardization is undefined for a constant feature.")

    return [
        [
            (row[j] - means[j]) / standard_deviations[j]
            for j in range(dimensions)
        ]
        for row in data
    ]


def choose_components(result: PCAResult, threshold: float = 0.95) -> int:
    if not 0 < threshold <= 1:
        raise PCAError("Explained-variance threshold must be in (0, 1].")

    for index, cumulative in enumerate(
        result.cumulative_explained_variance,
        start=1,
    ):
        if cumulative >= threshold:
            return index

    return len(result.eigenvalues)


def reconstruct(result: PCAResult, reduced: list[list[float]]) -> list[list[float]]:
    if len(reduced) == 0:
        return []

    if any(len(row) != result.components for row in reduced):
        raise PCAError("Reduced data has an invalid component count.")

    reconstructed = []
    for row in reduced:
        original = []
        for feature in range(len(result.means)):
            value = result.means[feature]
            for component in range(result.components):
                value += row[component] * result.eigenvectors[component][feature]
            original.append(value)
        reconstructed.append(original)

    return reconstructed


def reconstruction_error(
    original: list[list[float]],
    reconstructed: list[list[float]],
) -> float:
    if len(original) != len(reconstructed):
        raise PCAError("Datasets must contain the same number of observations.")

    squared_error = 0.0
    count = 0

    for left, right in zip(original, reconstructed):
        if len(left) != len(right):
            raise PCAError("Datasets must have the same dimensions.")
        for x, y in zip(left, right):
            squared_error += (x - y) ** 2
            count += 1

    return math.sqrt(squared_error / count)


def load_csv(path: str | Path) -> list[list[float]]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            try:
                rows.append([float(value.strip()) for value in row])
            except ValueError as exc:
                raise PCAError(
                    f"Non-numeric CSV value on row {row_number}."
                ) from exc

    validate_matrix(rows)
    return rows


def print_matrix(title: str, matrix: list[list[float]]) -> None:
    print(f"\n{title}")
    for row in matrix:
        print("  " + " ".join(f"{value:10.5f}" for value in row))


def print_result(result: PCAResult) -> None:
    print_matrix("Covariance matrix:", result.covariance)

    print("\nEigenvalues:")
    for index, value in enumerate(result.eigenvalues, start=1):
        print(
            f"  PC{index}: {value:.6f} | "
            f"explained={result.explained_variance_ratio[index - 1]:.2%} | "
            f"cumulative={result.cumulative_explained_variance[index - 1]:.2%}"
        )

    print("\nEigenvectors:")
    for index, vector in enumerate(result.eigenvectors, start=1):
        print(
            f"  PC{index}: "
            + " ".join(f"{value: .6f}" for value in vector)
        )

    print_matrix("Projected observations:", result.transformed)


def main() -> None:
    # Measurements deliberately have different physical scales. PCA on
    # covariance therefore preserves the scale-sensitive interpretation.
    data = [
        [170.0, 65.0, 5.2, 12.0],
        [172.0, 67.0, 5.4, 12.5],
        [168.0, 61.0, 5.0, 11.0],
        [175.0, 72.0, 5.8, 14.0],
        [178.0, 76.0, 6.1, 15.0],
        [180.0, 79.0, 6.3, 16.0],
        [169.0, 63.0, 5.1, 11.5],
        [174.0, 70.0, 5.7, 13.5],
    ]

    print("PCA technical case study")
    print("========================")
    print(f"Observations: {len(data)}")
    print(f"Features: {len(data[0])}")

    result = fit_pca(data, components=len(data[0]))
    print_result(result)

    selected = choose_components(result, threshold=0.95)
    reduced_result = fit_pca(data, components=selected)

    reconstructed = reconstruct(
        reduced_result,
        reduced_result.transformed,
    )
    error = reconstruction_error(data, reconstructed)

    print(f"\nComponents required for 95% variance: {selected}")
    print(f"RMSE reconstruction error: {error:.6f}")

    standardized_data = standardize(data)
    standardized_result = fit_pca(
        standardized_data,
        components=len(data[0]),
    )

    print("\nStandardized-feature comparison:")
    for index, ratio in enumerate(
        standardized_result.explained_variance_ratio,
        start=1,
    ):
        print(f"  PC{index}: {ratio:.2%}")

    # Assertions act as executable correctness checks.
    assert len(result.eigenvalues) == len(data[0])
    assert len(result.eigenvectors) == len(data[0])
    assert abs(sum(result.explained_variance_ratio) - 1.0) < 1e-9
    assert all(
        result.cumulative_explained_variance[i]
        <= result.cumulative_explained_variance[i + 1] + 1e-12
        for i in range(len(result.eigenvalues) - 1)
    )

    try:
        fit_pca([[1.0, 1.0], [1.0, 1.0]])
    except PCAError as exc:
        print(f"\nExpected validation failure: {exc}")

    print("\nProduction notes:")
    print("- Centering is required before covariance-based PCA.")
    print("- Sample covariance uses n-1 as the denominator.")
    print("- Standardization should be considered when feature scales differ.")
    print("- Large datasets should use optimized numerical linear algebra.")
    print("- Sensitive source data should be validated and handled under the "
          "application's data-access policy.")


if __name__ == "__main__":
    main()
