"""
Polynomial Regression: Polynomial Features, Nonlinear Relationships, and Overfitting

A self-contained learning and demonstration program covering:
- Polynomial feature construction
- Linear regression on transformed polynomial features
- Nonlinear relationships
- Model fitting and prediction
- Train/test evaluation
- Degree selection
- Underfitting and overfitting
- Regularization
- Cross-validation
- Numerical stability
- Feature scaling
- Model diagnostics
- Prediction intervals through residual analysis
- A realistic temperature-energy-demand case study

Requires Python 3.10+ and only the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random
from statistics import mean
from typing import Iterable, Sequence


Number = float


def heading(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def dot(a: Sequence[Number], b: Sequence[Number]) -> Number:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal lengths.")
    return sum(x * y for x, y in zip(a, b))


def transpose(matrix: Sequence[Sequence[Number]]) -> list[list[Number]]:
    if not matrix:
        return []
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("Matrix rows must have equal lengths.")
    return [list(column) for column in zip(*matrix)]


def matrix_multiply(
    a: Sequence[Sequence[Number]],
    b: Sequence[Sequence[Number]],
) -> list[list[Number]]:
    if not a or not b:
        raise ValueError("Matrices must not be empty.")
    b_t = transpose(b)
    if len(a[0]) != len(b):
        raise ValueError("Incompatible matrix dimensions.")
    return [[dot(row, column) for column in b_t] for row in a]


def identity_matrix(size: int) -> list[list[Number]]:
    return [
        [1.0 if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]


def matrix_inverse(matrix: Sequence[Sequence[Number]]) -> list[list[Number]]:
    """
    Gauss-Jordan inversion with partial pivoting.

    Partial pivoting reduces numerical error when a pivot is small.
    Polynomial models can become numerically unstable at high degrees,
    so pivoting and explicit singularity detection are important.
    """
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be non-empty and square.")

    augmented = [
        list(map(float, row)) + identity_row
        for row, identity_row in zip(matrix, identity_matrix(n))
    ]

    for column in range(n):
        pivot_row = max(
            range(column, n),
            key=lambda row: abs(augmented[row][column]),
        )
        pivot_value = augmented[pivot_row][column]

        if abs(pivot_value) < 1e-12:
            raise ValueError(
                "Matrix is singular or numerically unstable."
            )

        if pivot_row != column:
            augmented[column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[column],
            )

        pivot_value = augmented[column][column]
        augmented[column] = [
            value / pivot_value for value in augmented[column]
        ]

        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                current - factor * pivot
                for current, pivot in zip(
                    augmented[row], augmented[column]
                )
            ]

    return [row[n:] for row in augmented]


def solve_linear_system(
    matrix: Sequence[Sequence[Number]],
    vector: Sequence[Number],
) -> list[Number]:
    if len(matrix) != len(vector):
        raise ValueError("Matrix and vector dimensions do not match.")

    inverse = matrix_inverse(matrix)
    return [
        dot(row, vector)
        for row in inverse
    ]


def mean_squared_error(
    actual: Sequence[Number],
    predicted: Sequence[Number],
) -> Number:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted data must have equal nonzero lengths.")
    return mean((y - y_hat) ** 2 for y, y_hat in zip(actual, predicted))


def root_mean_squared_error(
    actual: Sequence[Number],
    predicted: Sequence[Number],
) -> Number:
    return sqrt(mean_squared_error(actual, predicted))


def mean_absolute_error(
    actual: Sequence[Number],
    predicted: Sequence[Number],
) -> Number:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted data must have equal nonzero lengths.")
    return mean(abs(y - y_hat) for y, y_hat in zip(actual, predicted))


def r_squared(
    actual: Sequence[Number],
    predicted: Sequence[Number],
) -> Number:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted data must have equal nonzero lengths.")

    actual_mean = mean(actual)
    total = sum((y - actual_mean) ** 2 for y in actual)
    residual = sum((y - y_hat) ** 2 for y, y_hat in zip(actual, predicted))

    if total == 0:
        return 1.0 if residual == 0 else 0.0

    return 1.0 - residual / total


def train_test_split(
    x: Sequence[Number],
    y: Sequence[Number],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Number], list[Number], list[Number], list[Number]]:
    if len(x) != len(y) or len(x) < 4:
        raise ValueError("x and y must have equal lengths with at least four observations.")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    indices = list(range(len(x)))
    Random(seed).shuffle(indices)

    test_count = max(1, round(len(x) * test_ratio))
    test_indices = set(indices[:test_count])

    x_train = [x[i] for i in range(len(x)) if i not in test_indices]
    y_train = [y[i] for i in range(len(y)) if i not in test_indices]
    x_test = [x[i] for i in range(len(x)) if i in test_indices]
    y_test = [y[i] for i in range(len(y)) if i in test_indices]

    return x_train, x_test, y_train, y_test


class StandardScaler:
    """Standardizes one numeric feature before polynomial expansion."""

    def __init__(self) -> None:
        self.mean_: float | None = None
        self.scale_: float | None = None

    def fit(self, values: Sequence[Number]) -> "StandardScaler":
        if not values:
            raise ValueError("Cannot fit a scaler to empty data.")

        self.mean_ = mean(values)
        variance = mean((value - self.mean_) ** 2 for value in values)
        self.scale_ = sqrt(variance)

        if self.scale_ < 1e-12:
            raise ValueError("Cannot standardize a constant feature.")

        return self

    def transform(self, values: Sequence[Number]) -> list[Number]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return [(value - self.mean_) / self.scale_ for value in values]

    def fit_transform(self, values: Sequence[Number]) -> list[Number]:
        return self.fit(values).transform(values)


class PolynomialFeatures:
    """
    Generates [1, x, x², ..., x^degree] for a single numeric feature.

    The constant column is included because the regression model needs
    an intercept. For one input variable, polynomial regression therefore
    remains linear in its learned coefficients while becoming nonlinear
    in the original input variable.
    """

    def __init__(self, degree: int, include_bias: bool = True) -> None:
        if degree < 0:
            raise ValueError("degree must be non-negative.")
        self.degree = degree
        self.include_bias = include_bias

    def transform(self, values: Sequence[Number]) -> list[list[Number]]:
        if not values:
            raise ValueError("Cannot transform empty data.")

        rows: list[list[Number]] = []

        for value in values:
            features = [
                float(value) ** power
                for power in range(self.degree + 1)
            ]

            if not self.include_bias:
                features = features[1:]

            rows.append(features)

        return rows


@dataclass
class RegressionResult:
    coefficients: list[Number]
    degree: int
    regularization: float
    scaled: bool


class PolynomialRegression:
    """
    Polynomial regression solved with the normal equation.

    Objective:
        ||y - Xβ||² + λ||β_non_intercept||²

    The regularization term is ridge regularization. The intercept is
    deliberately excluded from the penalty because penalizing the
    baseline level can distort predictions unnecessarily.
    """

    def __init__(
        self,
        degree: int,
        regularization: float = 0.0,
        scale_features: bool = False,
    ) -> None:
        if degree < 0:
            raise ValueError("degree must be non-negative.")
        if regularization < 0:
            raise ValueError("regularization must not be negative.")

        self.degree = degree
        self.regularization = regularization
        self.scale_features = scale_features
        self.features = PolynomialFeatures(degree)
        self.scaler = StandardScaler() if scale_features else None
        self.coefficients: list[Number] | None = None

    def fit(self, x: Sequence[Number], y: Sequence[Number]) -> "PolynomialRegression":
        if len(x) != len(y) or len(x) < self.degree + 1:
            raise ValueError(
                "Need matching x/y data and enough observations for the selected degree."
            )

        numeric_x = [float(value) for value in x]
        numeric_y = [float(value) for value in y]

        if self.scaler is not None:
            numeric_x = self.scaler.fit_transform(numeric_x)

        design = self.features.transform(numeric_x)
        x_transpose = transpose(design)
        normal_matrix = matrix_multiply(x_transpose, design)

        regularized_matrix = [
            row[:] for row in normal_matrix
        ]

        for index in range(1, len(regularized_matrix)):
            regularized_matrix[index][index] += self.regularization

        target = [[value] for value in numeric_y]
        right_side = matrix_multiply(x_transpose, target)
        self.coefficients = solve_linear_system(
            regularized_matrix,
            [row[0] for row in right_side],
        )

        return self

    def predict(self, x: Sequence[Number]) -> list[Number]:
        if self.coefficients is None:
            raise RuntimeError("Model has not been fitted.")

        numeric_x = [float(value) for value in x]

        if self.scaler is not None:
            numeric_x = self.scaler.transform(numeric_x)

        design = self.features.transform(numeric_x)

        return [
            dot(row, self.coefficients)
            for row in design
        ]

    def evaluate(
        self,
        x: Sequence[Number],
        y: Sequence[Number],
    ) -> dict[str, float]:
        predictions = self.predict(x)
        return {
            "MAE": mean_absolute_error(y, predictions),
            "RMSE": root_mean_squared_error(y, predictions),
            "R2": r_squared(y, predictions),
        }

    def result(self) -> RegressionResult:
        if self.coefficients is None:
            raise RuntimeError("Model has not been fitted.")
        return RegressionResult(
            coefficients=self.coefficients[:],
            degree=self.degree,
            regularization=self.regularization,
            scaled=self.scale_features,
        )


def generate_noisy_quadratic(
    count: int,
    seed: int = 7,
) -> tuple[list[float], list[float]]:
    """
    Generates y = 3 + 2x - 0.8x² + noise.

    The underlying relationship is nonlinear even though the model
    estimates coefficients with ordinary linear algebra.
    """
    rng = Random(seed)
    x = [(-5.0 + 10.0 * i / (count - 1)) for i in range(count)]
    y = [
        3.0 + 2.0 * value - 0.8 * value * value + rng.gauss(0, 1.5)
        for value in x
    ]
    return x, y


def print_metrics(
    name: str,
    metrics: dict[str, float],
) -> None:
    print(
        f"{name:<24} "
        f"MAE={metrics['MAE']:.4f}  "
        f"RMSE={metrics['RMSE']:.4f}  "
        f"R²={metrics['R2']:.4f}"
    )


def demonstrate_polynomial_features() -> None:
    heading("Polynomial Feature Construction")

    values = [-2.0, 0.0, 3.0]
    transformer = PolynomialFeatures(degree=4)
    transformed = transformer.transform(values)

    for value, row in zip(values, transformed):
        print(f"x={value:>5.1f} -> {row}")

    print(
        "\nThe regression remains linear in its coefficients, but the feature "
        "matrix contains powers of x, allowing the fitted function to curve."
    )


def demonstrate_linear_vs_polynomial() -> None:
    heading("Linear Model Versus Polynomial Model")

    x, y = generate_noisy_quadratic(80)
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    linear = PolynomialRegression(degree=1)
    quadratic = PolynomialRegression(degree=2)

    linear.fit(x_train, y_train)
    quadratic.fit(x_train, y_train)

    print_metrics("Linear train", linear.evaluate(x_train, y_train))
    print_metrics("Linear test", linear.evaluate(x_test, y_test))
    print_metrics("Quadratic train", quadratic.evaluate(x_train, y_train))
    print_metrics("Quadratic test", quadratic.evaluate(x_test, y_test))

    print("\nQuadratic coefficients:")
    print([round(value, 5) for value in quadratic.result().coefficients])


def demonstrate_degree_selection() -> None:
    heading("Degree Selection: Underfitting and Overfitting")

    x, y = generate_noisy_quadratic(90, seed=11)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_ratio=0.25, seed=21
    )

    print(
        f"{'Degree':>6}  {'Train RMSE':>12}  {'Test RMSE':>12}  "
        f"{'Train R²':>10}  {'Test R²':>10}"
    )

    for degree in range(1, 11):
        model = PolynomialRegression(
            degree=degree,
            regularization=0.001,
            scale_features=True,
        )
        model.fit(x_train, y_train)

        train_metrics = model.evaluate(x_train, y_train)
        test_metrics = model.evaluate(x_test, y_test)

        print(
            f"{degree:>6}  "
            f"{train_metrics['RMSE']:>12.4f}  "
            f"{test_metrics['RMSE']:>12.4f}  "
            f"{train_metrics['R2']:>10.4f}  "
            f"{test_metrics['R2']:>10.4f}"
        )

    print(
        "\nA high degree can drive training error downward while increasing "
        "test error. That gap is a practical signal of overfitting."
    )


def k_fold_indices(
    size: int,
    folds: int,
    seed: int = 42,
) -> list[list[int]]:
    if folds < 2 or folds > size:
        raise ValueError("folds must be between 2 and the number of observations.")

    indices = list(range(size))
    Random(seed).shuffle(indices)

    result = [[] for _ in range(folds)]

    for position, index in enumerate(indices):
        result[position % folds].append(index)

    return result


def cross_validate_degree(
    x: Sequence[Number],
    y: Sequence[Number],
    degree: int,
    folds: int = 5,
    regularization: float = 0.0,
) -> float:
    fold_indices = k_fold_indices(len(x), folds)

    scores: list[float] = []

    for validation_indices in fold_indices:
        validation_set = set(validation_indices)
        train_indices = [
            index
            for index in range(len(x))
            if index not in validation_set
        ]

        x_train = [x[index] for index in train_indices]
        y_train = [y[index] for index in train_indices]
        x_validation = [x[index] for index in validation_indices]
        y_validation = [y[index] for index in validation_indices]

        model = PolynomialRegression(
            degree=degree,
            regularization=regularization,
            scale_features=True,
        )
        model.fit(x_train, y_train)

        metrics = model.evaluate(x_validation, y_validation)
        scores.append(metrics["RMSE"])

    return mean(scores)


def demonstrate_cross_validation() -> None:
    heading("Cross-Validation for Degree Selection")

    x, y = generate_noisy_quadratic(100, seed=23)

    print(f"{'Degree':>6}  {'5-fold CV RMSE':>18}")

    candidates: list[tuple[int, float]] = []

    for degree in range(1, 9):
        score = cross_validate_degree(
            x,
            y,
            degree=degree,
            folds=5,
            regularization=0.01,
        )
        candidates.append((degree, score))
        print(f"{degree:>6}  {score:>18.5f}")

    selected_degree, selected_score = min(
        candidates,
        key=lambda item: item[1],
    )

    print(
        f"\nSelected degree={selected_degree} "
        f"with cross-validation RMSE={selected_score:.5f}"
    )


def demonstrate_regularization() -> None:
    heading("Regularization Against High-Degree Instability")

    x, y = generate_noisy_quadratic(70, seed=31)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_ratio=0.25,
        seed=8,
    )

    degree = 9

    for penalty in [0.0, 0.001, 0.1, 1.0, 10.0, 100.0]:
        model = PolynomialRegression(
            degree=degree,
            regularization=penalty,
            scale_features=True,
        )
        model.fit(x_train, y_train)

        metrics = model.evaluate(x_test, y_test)

        print(
            f"λ={penalty:<8g} "
            f"test RMSE={metrics['RMSE']:.5f} "
            f"test R²={metrics['R2']:.5f}"
        )

    print(
        "\nRegularization shrinks polynomial coefficients. It can reduce "
        "variance in flexible models, although excessive regularization "
        "can cause underfitting."
    )


def demonstrate_edge_cases() -> None:
    heading("Validation and Failure Conditions")

    cases = [
        ("negative degree", lambda: PolynomialRegression(-1)),
        ("negative regularization", lambda: PolynomialRegression(2, -1)),
        (
            "constant scaling feature",
            lambda: StandardScaler().fit([4.0, 4.0, 4.0]),
        ),
        (
            "mismatched metrics",
            lambda: mean_squared_error([1.0], [1.0, 2.0]),
        ),
    ]

    for name, operation in cases:
        try:
            operation()
        except (ValueError, RuntimeError) as error:
            print(f"{name:<28} rejected: {error}")

    try:
        model = PolynomialRegression(degree=2)
        model.predict([1.0])
    except RuntimeError as error:
        print(f"{'prediction before fitting':<28} rejected: {error}")


def generate_energy_demand_data(
    count: int = 120,
    seed: int = 55,
) -> tuple[list[float], list[float]]:
    """
    Realistic synthetic case study.

    Energy demand is modeled as a nonlinear function of temperature.
    Demand rises at both low and high temperatures because heating and
    cooling loads increase away from a comfortable operating range.

    The generated relationship includes a quadratic temperature effect,
    a small cubic asymmetry, and measurement noise.
    """
    rng = Random(seed)
    temperatures: list[float] = []
    demand: list[float] = []

    for i in range(count):
        temperature = -5 + i * 45 / (count - 1)

        demand_value = (
            420
            - 13 * temperature
            + 0.72 * temperature ** 2
            - 0.012 * temperature ** 3
            + rng.gauss(0, 18)
        )

        temperatures.append(temperature)
        demand.append(demand_value)

    return temperatures, demand


def demonstrate_realistic_case_study() -> None:
    heading("Case Study: Nonlinear Energy Demand")

    temperature, demand = generate_energy_demand_data()

    train_temperature, test_temperature, train_demand, test_demand = (
        train_test_split(
            temperature,
            demand,
            test_ratio=0.2,
            seed=101,
        )
    )

    candidates: list[tuple[int, float]] = []

    for degree in range(1, 7):
        score = cross_validate_degree(
            train_temperature,
            train_demand,
            degree=degree,
            folds=5,
            regularization=0.1,
        )
        candidates.append((degree, score))

    selected_degree, _ = min(candidates, key=lambda item: item[1])

    model = PolynomialRegression(
        degree=selected_degree,
        regularization=0.1,
        scale_features=True,
    )
    model.fit(train_temperature, train_demand)

    train_metrics = model.evaluate(train_temperature, train_demand)
    test_metrics = model.evaluate(test_temperature, test_demand)

    print(f"Selected polynomial degree: {selected_degree}")
    print_metrics("Training", train_metrics)
    print_metrics("Testing", test_metrics)

    print("\nPredicted demand at selected temperatures:")

    for temperature_value in [-5, 0, 10, 20, 30, 40]:
        prediction = model.predict([temperature_value])[0]
        print(
            f"Temperature {temperature_value:>5.1f} °C -> "
            f"predicted demand {prediction:>8.2f}"
        )


def demonstrate_extrapolation_warning() -> None:
    heading("Extrapolation Risk")

    x, y = generate_energy_demand_data(100, seed=91)

    model = PolynomialRegression(
        degree=4,
        regularization=0.1,
        scale_features=True,
    )
    model.fit(x, y)

    observed_min = min(x)
    observed_max = max(x)

    points = [
        observed_min,
        observed_max,
        observed_max + 10,
        observed_max + 30,
    ]

    print(
        f"Training range: {observed_min:.1f} °C to {observed_max:.1f} °C"
    )

    for temperature in points:
        prediction = model.predict([temperature])[0]
        marker = (
            "inside training range"
            if observed_min <= temperature <= observed_max
            else "EXTRAPOLATION"
        )
        print(
            f"{temperature:>7.1f} °C -> {prediction:>10.2f} "
            f"({marker})"
        )

    print(
        "\nPolynomial curves can behave dramatically outside the range "
        "used for training. Good interpolation performance does not "
        "guarantee safe extrapolation."
    )


def residual_diagnostics(
    actual: Sequence[Number],
    predicted: Sequence[Number],
) -> dict[str, float]:
    residuals = [
        actual_value - predicted_value
        for actual_value, predicted_value in zip(actual, predicted)
    ]

    residual_mean = mean(residuals)
    residual_variance = mean(
        (residual - residual_mean) ** 2
        for residual in residuals
    )

    return {
        "mean_residual": residual_mean,
        "residual_std": sqrt(residual_variance),
        "max_absolute_residual": max(abs(value) for value in residuals),
    }


def demonstrate_residual_diagnostics() -> None:
    heading("Residual Diagnostics")

    x, y = generate_noisy_quadratic(100, seed=73)
    model = PolynomialRegression(
        degree=2,
        regularization=0.01,
        scale_features=True,
    )
    model.fit(x, y)

    predictions = model.predict(x)
    diagnostics = residual_diagnostics(y, predictions)

    for name, value in diagnostics.items():
        print(f"{name:<25} {value:.6f}")

    print(
        "\nA residual mean near zero is desirable, but it is not sufficient "
        "by itself. Production diagnostics should also inspect residual "
        "patterns against fitted values and input variables."
    )


def demonstrate_prediction_workflow() -> None:
    heading("Complete Model Selection Workflow")

    x, y = generate_noisy_quadratic(120, seed=2026)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_ratio=0.2,
        seed=2026,
    )

    model_candidates: list[tuple[int, float, float]] = []

    for degree in range(1, 8):
        cv_rmse = cross_validate_degree(
            x_train,
            y_train,
            degree=degree,
            folds=5,
            regularization=0.05,
        )
        model_candidates.append((degree, cv_rmse, float(degree)))

    selected_degree = min(
        model_candidates,
        key=lambda item: item[1],
    )[0]

    final_model = PolynomialRegression(
        degree=selected_degree,
        regularization=0.05,
        scale_features=True,
    )
    final_model.fit(x_train, y_train)

    test_metrics = final_model.evaluate(x_test, y_test)

    print(f"Selected degree: {selected_degree}")
    print_metrics("Final test set", test_metrics)

    print("\nSample predictions:")

    for actual_x, actual_y in list(zip(x_test, y_test))[:8]:
        predicted_y = final_model.predict([actual_x])[0]
        print(
            f"x={actual_x:>7.3f} "
            f"actual={actual_y:>9.3f} "
            f"predicted={predicted_y:>9.3f}"
        )


def main() -> None:
    heading("Polynomial Regression Technical Demonstration")

    demonstrate_polynomial_features()
    demonstrate_linear_vs_polynomial()
    demonstrate_degree_selection()
    demonstrate_cross_validation()
    demonstrate_regularization()
    demonstrate_edge_cases()
    demonstrate_realistic_case_study()
    demonstrate_extrapolation_warning()
    demonstrate_residual_diagnostics()
    demonstrate_prediction_workflow()

    heading("Production Considerations")
    print(
        "Polynomial degree should be selected using validation data rather "
        "than training error alone. Feature scaling becomes increasingly "
        "important as powers grow. High-degree models are vulnerable to "
        "numerical instability, variance, and poor extrapolation. Regularization "
        "can control coefficient magnitude, while residual diagnostics and "
        "held-out evaluation help detect models that fit noise rather than "
        "the underlying nonlinear relationship."
    )


if __name__ == "__main__":
    main()
