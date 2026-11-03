#!/usr/bin/env python3
"""
Bias-Variance Tradeoff Experiment

Runs repeated polynomial-regression experiments to measure:
- Underfitting and overfitting
- Model complexity
- Bias squared
- Prediction variance
- Irreducible noise
- Expected test error
- Generalization gap

Uses only the Python standard library.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


NOISE_STDDEV = 0.20
RIDGE_LAMBDA = 1e-6


class ModelInputError(ValueError):
    """Raised when model configuration or training data is invalid."""


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Evaluation:
    degree: int
    average_training_mse: float
    bias_squared: float
    prediction_variance: float
    irreducible_noise: float
    signal_prediction_mse: float
    expected_test_mse: float
    generalization_gap: float
    risk: str


def true_signal(x: float) -> float:
    """Underlying nonlinear relationship without observation noise."""
    return math.sin(2.0 * math.pi * x) + 0.25 * x


def validate_points(points: Sequence[Point], minimum_size: int = 2) -> None:
    if len(points) < minimum_size:
        raise ModelInputError(
            f"At least {minimum_size} observations are required."
        )

    for index, point in enumerate(points):
        if not math.isfinite(point.x) or not math.isfinite(point.y):
            raise ModelInputError(
                f"Observation {index} contains a non-finite value."
            )

        if not -1.0 <= point.x <= 1.0:
            raise ModelInputError(
                f"Observation {index} has x outside [-1, 1]."
            )


def solve_linear_system(
    matrix: list[list[float]],
    vector: list[float],
) -> list[float]:
    """
    Solve Ax=b using Gaussian elimination with partial pivoting.

    Partial pivoting improves numerical stability. Ridge regularization
    is applied during model fitting to reduce coefficient instability.
    """
    size = len(vector)

    if size == 0 or len(matrix) != size:
        raise ModelInputError("The coefficient matrix must be square.")

    augmented = [
        list(matrix[row]) + [vector[row]]
        for row in range(size)
    ]

    for column in range(size):
        pivot_row = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )

        pivot_value = augmented[pivot_row][column]

        if abs(pivot_value) < 1e-14:
            raise ArithmeticError(
                "Singular or numerically unstable regression system."
            )

        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )

        pivot_value = augmented[column][column]

        for row in range(column + 1, size):
            factor = augmented[row][column] / pivot_value

            for item in range(column, size + 1):
                augmented[row][item] -= (
                    factor * augmented[column][item]
                )

    coefficients = [0.0] * size

    for row in range(size - 1, -1, -1):
        remaining = augmented[row][size]

        for column in range(row + 1, size):
            remaining -= (
                augmented[row][column] * coefficients[column]
            )

        coefficients[row] = remaining / augmented[row][row]

    if not all(math.isfinite(value) for value in coefficients):
        raise ArithmeticError("Regression produced non-finite coefficients.")

    return coefficients


class PolynomialRegressor:
    """
    Polynomial regression with L2 regularization.

    Degree controls model complexity. The intercept is not regularized.
    """

    def __init__(self, degree: int, ridge_lambda: float = RIDGE_LAMBDA):
        if not isinstance(degree, int) or degree < 0 or degree > 12:
            raise ModelInputError("Degree must be an integer between 0 and 12.")

        if not math.isfinite(ridge_lambda) or ridge_lambda < 0:
            raise ModelInputError(
                "Ridge regularization must be finite and non-negative."
            )

        self.degree = degree
        self.ridge_lambda = ridge_lambda
        self.coefficients: list[float] | None = None

    @staticmethod
    def _features(x: float, degree: int) -> list[float]:
        features = [1.0]

        for _ in range(degree):
            features.append(features[-1] * x)

        return features

    def fit(self, points: Sequence[Point]) -> PolynomialRegressor:
        validate_points(points)

        parameter_count = self.degree + 1
        normal_matrix = [
            [0.0] * parameter_count
            for _ in range(parameter_count)
        ]
        right_hand_side = [0.0] * parameter_count

        for point in points:
            features = self._features(point.x, self.degree)

            for row in range(parameter_count):
                right_hand_side[row] += features[row] * point.y

                for column in range(parameter_count):
                    normal_matrix[row][column] += (
                        features[row] * features[column]
                    )

        # Penalize polynomial coefficients, but not the intercept.
        for index in range(1, parameter_count):
            normal_matrix[index][index] += self.ridge_lambda

        self.coefficients = solve_linear_system(
            normal_matrix,
            right_hand_side,
        )

        return self

    def predict_one(self, x: float) -> float:
        if self.coefficients is None:
            raise RuntimeError("The model must be fitted before prediction.")

        if not math.isfinite(x) or not -1.0 <= x <= 1.0:
            raise ModelInputError("Prediction input must be finite and in [-1, 1].")

        features = self._features(x, self.degree)
        prediction = sum(
            coefficient * feature
            for coefficient, feature in zip(self.coefficients, features)
        )

        if not math.isfinite(prediction):
            raise ArithmeticError("Model prediction is not finite.")

        return prediction

    def predict(self, values: Iterable[float]) -> list[float]:
        return [self.predict_one(value) for value in values]


def generate_dataset(
    sample_count: int,
    noise_stddev: float,
    rng: random.Random,
) -> list[Point]:
    if sample_count < 2:
        raise ModelInputError("Sample count must be at least two.")

    if not math.isfinite(noise_stddev) or noise_stddev < 0:
        raise ModelInputError("Noise standard deviation must be non-negative.")

    return [
        Point(
            x=(x := rng.uniform(-1.0, 1.0)),
            y=true_signal(x) + rng.gauss(0.0, noise_stddev),
        )
        for _ in range(sample_count)
    ]


def mean_squared_error(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    if not actual or len(actual) != len(predicted):
        raise ModelInputError(
            "Actual and predicted values must have equal non-zero lengths."
        )

    return sum(
        (observed - estimate) ** 2
        for observed, estimate in zip(actual, predicted)
    ) / len(actual)


def classify_model(
    expected_test_mse: float,
    training_mse: float,
    best_test_mse: float,
    best_training_mse: float,
) -> str:
    """
    Heuristic classification for this controlled experiment.

    The minimum test error is the reference. A model with substantially
    worse test error and high training error is underfitting. A model
    with substantially worse test error but low training error is
    overfitting.
    """
    test_threshold = max(best_test_mse, 1e-12) * 1.10
    training_threshold = max(best_training_mse, 1e-12) * 1.20

    if expected_test_mse > test_threshold:
        if training_mse > training_threshold:
            return "UNDERFITTING"

        return "OVERFITTING"

    return "GENERALIZES_WELL"


def run_experiment(
    degrees: Sequence[int] = (1, 2, 5, 9),
    sample_count: int = 36,
    repetitions: int = 80,
    grid_size: int = 81,
    noise_stddev: float = NOISE_STDDEV,
    seed: int = 42,
) -> list[Evaluation]:
    if repetitions < 2:
        raise ModelInputError("At least two repetitions are required.")

    if sample_count < 10:
        raise ModelInputError("Use at least ten training observations.")

    if grid_size < 3:
        raise ModelInputError("The evaluation grid must contain at least three points.")

    if not degrees or len(set(degrees)) != len(degrees):
        raise ModelInputError("Model degrees must be non-empty and unique.")

    for degree in degrees:
        if not isinstance(degree, int) or not 0 <= degree <= 12:
            raise ModelInputError("Every degree must be between 0 and 12.")

    if not math.isfinite(noise_stddev) or noise_stddev < 0:
        raise ModelInputError("Noise standard deviation is invalid.")

    rng = random.Random(seed)
    grid = [
        -1.0 + 2.0 * index / (grid_size - 1)
        for index in range(grid_size)
    ]
    signal_values = [true_signal(x) for x in grid]

    predictions: dict[int, list[list[float]]] = {
        degree: [] for degree in degrees
    }
    training_errors: dict[int, list[float]] = {
        degree: [] for degree in degrees
    }

    # Each repetition draws a new independent training sample.
    # This makes between-fit prediction variation measurable.
    for _ in range(repetitions):
        training_data = generate_dataset(
            sample_count,
            noise_stddev,
            rng,
        )

        for degree in degrees:
            model = PolynomialRegressor(
                degree=degree,
                ridge_lambda=RIDGE_LAMBDA,
            ).fit(training_data)

            fitted_values = model.predict(
                [point.x for point in training_data]
            )
            observed_values = [point.y for point in training_data]

            training_errors[degree].append(
                mean_squared_error(observed_values, fitted_values)
            )
            predictions[degree].append(model.predict(grid))

    preliminary: list[dict[str, float | int]] = []

    for degree in degrees:
        model_predictions = predictions[degree]

        mean_predictions = [
            sum(model_predictions[run][index] for run in range(repetitions))
            / repetitions
            for index in range(grid_size)
        ]

        squared_bias = sum(
            (mean_predictions[index] - signal_values[index]) ** 2
            for index in range(grid_size)
        ) / grid_size

        prediction_variance = sum(
            sum(
                (
                    model_predictions[run][index]
                    - mean_predictions[index]
                ) ** 2
                for run in range(repetitions)
            ) / repetitions
            for index in range(grid_size)
        ) / grid_size

        average_training_mse = (
            sum(training_errors[degree]) / repetitions
        )

        # Expected squared error against a noisy future observation:
        # bias^2 + variance + irreducible observation noise.
        signal_prediction_mse = squared_bias + prediction_variance
        expected_test_mse = signal_prediction_mse + noise_stddev ** 2

        preliminary.append(
            {
                "degree": degree,
                "average_training_mse": average_training_mse,
                "bias_squared": squared_bias,
                "prediction_variance": prediction_variance,
                "irreducible_noise": noise_stddev ** 2,
                "signal_prediction_mse": signal_prediction_mse,
                "expected_test_mse": expected_test_mse,
            }
        )

    best_test_mse = min(
        float(item["expected_test_mse"])
        for item in preliminary
    )
    best_training_mse = min(
        float(item["average_training_mse"])
        for item in preliminary
    )

    results: list[Evaluation] = []

    for item in preliminary:
        test_mse = float(item["expected_test_mse"])
        training_mse = float(item["average_training_mse"])

        results.append(
            Evaluation(
                degree=int(item["degree"]),
                average_training_mse=training_mse,
                bias_squared=float(item["bias_squared"]),
                prediction_variance=float(item["prediction_variance"]),
                irreducible_noise=float(item["irreducible_noise"]),
                signal_prediction_mse=float(item["signal_prediction_mse"]),
                expected_test_mse=test_mse,
                generalization_gap=test_mse - training_mse,
                risk=classify_model(
                    test_mse,
                    training_mse,
                    best_test_mse,
                    best_training_mse,
                ),
            )
        )

    return sorted(results, key=lambda result: result.degree)


def validate_edge_cases() -> None:
    try:
        PolynomialRegressor(degree=-1)
    except ModelInputError:
        pass
    else:
        raise AssertionError("Negative model degree should be rejected.")

    try:
        PolynomialRegressor(degree=3).predict_one(0.25)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Prediction before fitting should be rejected.")

    try:
        generate_dataset(5, -0.1, random.Random(1))
    except ModelInputError:
        pass
    else:
        raise AssertionError("Negative noise should be rejected.")

    model = PolynomialRegressor(degree=0)
    model.fit([Point(-1.0, 2.0), Point(1.0, 2.0)])
    assert math.isclose(model.predict_one(0.0), 2.0, rel_tol=1e-6)


def main() -> None:
    validate_edge_cases()

    results = run_experiment()

    print("BIAS-VARIANCE TRADEOFF EXPERIMENT")
    print("Repeated polynomial regression on independently sampled datasets")
    print()

    print(
        f"{'Degree':>6} "
        f"{'Train MSE':>12} "
        f"{'Bias^2':>12} "
        f"{'Variance':>12} "
        f"{'Noise':>10} "
        f"{'Expected Test':>15} "
        f"{'Gap':>12} "
        f"{'Assessment':>18}"
    )

    for result in results:
        print(
            f"{result.degree:>6} "
            f"{result.average_training_mse:>12.5f} "
            f"{result.bias_squared:>12.5f} "
            f"{result.prediction_variance:>12.5f} "
            f"{result.irreducible_noise:>10.5f} "
            f"{result.expected_test_mse:>15.5f} "
            f"{result.generalization_gap:>12.5f} "
            f"{result.risk:>18}"
        )

    best_model = min(results, key=lambda result: result.expected_test_mse)

    print()
    print(f"Best degree by expected test MSE: {best_model.degree}")
    print(f"Expected test MSE: {best_model.expected_test_mse:.6f}")
    print()
    print("Machine-readable results:")
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ModelInputError, ArithmeticError, RuntimeError) as error:
        print(f"Experiment failed: {error}")
        raise SystemExit(1)
