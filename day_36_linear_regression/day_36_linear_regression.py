"""
Linear Regression: Simple Regression, Multiple Regression, Coefficients, and Intercept

A self-contained implementation using only the Python standard library.

The script demonstrates:
- Simple linear regression with one predictor
- Multiple linear regression with several predictors
- Coefficient and intercept estimation using ordinary least squares
- Predictions and residuals
- R-squared and adjusted R-squared
- Residual diagnostics
- Multicollinearity detection through correlation and VIF
- Feature scaling and coefficient interpretation
- Matrix operations implemented with the standard library
- Gradient-descent estimation as a complementary approach
- Train/test evaluation
- Confidence intervals for coefficients
- Validation and common failure conditions

No third-party packages are required.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence


Number = float


# ---------------------------------------------------------------------------
# Basic numerical utilities
# ---------------------------------------------------------------------------

def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Mean requires at least one value.")
    return sum(values) / len(values)


def variance(values: Sequence[float], sample: bool = True) -> float:
    if not values:
        raise ValueError("Variance requires at least one value.")
    if sample and len(values) < 2:
        raise ValueError("Sample variance requires at least two values.")

    center = mean(values)
    denominator = len(values) - 1 if sample else len(values)
    return sum((x - center) ** 2 for x in values) / denominator


def standard_deviation(values: Sequence[float], sample: bool = True) -> float:
    return math.sqrt(variance(values, sample=sample))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Dot product requires equal-length vectors.")
    return sum(x * y for x, y in zip(a, b))


def transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    if not matrix:
        return []
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("Matrix rows must have equal length.")
    return [list(column) for column in zip(*matrix)]


def matrix_multiply(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
) -> list[list[float]]:
    if not a or not b:
        raise ValueError("Matrices must not be empty.")

    a_width = len(a[0])
    b_height = len(b)

    if any(len(row) != a_width for row in a):
        raise ValueError("Matrix A is not rectangular.")
    if any(len(row) != len(b[0]) for row in b):
        raise ValueError("Matrix B is not rectangular.")
    if a_width != b_height:
        raise ValueError("Matrix dimensions are incompatible.")

    b_t = transpose(b)
    return [[dot(row, column) for column in b_t] for row in a]


def identity_matrix(size: int) -> list[list[float]]:
    return [
        [1.0 if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]


def matrix_inverse(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """
    Gauss-Jordan inversion with partial pivoting.

    Partial pivoting improves numerical stability by selecting the largest
    available pivot in the current column.
    """
    n = len(matrix)

    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Only non-empty square matrices can be inverted.")

    augmented = [
        list(map(float, matrix[row])) + identity_matrix(n)[row]
        for row in range(n)
    ]

    for column in range(n):
        pivot_row = max(
            range(column, n),
            key=lambda row: abs(augmented[row][column]),
        )
        pivot_value = augmented[pivot_row][column]

        if abs(pivot_value) < 1e-12:
            raise ValueError(
                "Matrix is singular or numerically singular. "
                "The regression design matrix may contain redundant predictors."
            )

        if pivot_row != column:
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
            if abs(factor) < 1e-15:
                continue

            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row],
                    augmented[column],
                )
            ]

    return [row[n:] for row in augmented]


def matrix_vector_multiply(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> list[float]:
    return [dot(row, vector) for row in matrix]


# ---------------------------------------------------------------------------
# Simple linear regression
# ---------------------------------------------------------------------------

@dataclass
class SimpleRegressionResult:
    intercept: float
    slope: float
    predictions: list[float]
    residuals: list[float]
    r_squared: float


def fit_simple_regression(
    x: Sequence[float],
    y: Sequence[float],
) -> SimpleRegressionResult:
    """
    Fits:

        y = intercept + slope * x

    The slope is obtained from the least-squares closed-form estimator:

        slope = sum((x_i - x_mean)(y_i - y_mean))
                --------------------------------
                sum((x_i - x_mean)^2)

    The intercept follows from:

        intercept = y_mean - slope * x_mean
    """
    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of observations.")
    if len(x) < 2:
        raise ValueError("At least two observations are required.")

    x_mean = mean(x)
    y_mean = mean(y)

    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator < 1e-15:
        raise ValueError(
            "Simple regression cannot estimate a slope when all x values are equal."
        )

    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    )

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    predictions = [intercept + slope * value for value in x]
    residuals = [
        actual - predicted
        for actual, predicted in zip(y, predictions)
    ]

    sse = sum(residual ** 2 for residual in residuals)
    total_sum_of_squares = sum(
        (actual - y_mean) ** 2 for actual in y
    )

    if total_sum_of_squares < 1e-15:
        r_squared = 1.0 if sse < 1e-15 else 0.0
    else:
        r_squared = 1.0 - sse / total_sum_of_squares

    return SimpleRegressionResult(
        intercept=intercept,
        slope=slope,
        predictions=predictions,
        residuals=residuals,
        r_squared=r_squared,
    )


# ---------------------------------------------------------------------------
# Multiple linear regression
# ---------------------------------------------------------------------------

@dataclass
class RegressionMetrics:
    r_squared: float
    adjusted_r_squared: float
    mse: float
    rmse: float
    mae: float
    sse: float


@dataclass
class LinearRegressionModel:
    """
    Coefficients are stored as:

        [intercept, coefficient_1, coefficient_2, ...]

    Therefore a feature vector:

        [x1, x2, x3]

    is evaluated as:

        intercept + b1*x1 + b2*x2 + b3*x3
    """

    intercept: float
    coefficients: list[float]
    feature_names: list[str]
    training_predictions: list[float]
    training_residuals: list[float]
    metrics: RegressionMetrics
    residual_variance: float
    coefficient_standard_errors: list[float]

    def predict_row(self, features: Sequence[float]) -> float:
        if len(features) != len(self.coefficients):
            raise ValueError(
                f"Expected {len(self.coefficients)} features, "
                f"received {len(features)}."
            )
        return self.intercept + dot(self.coefficients, features)

    def predict(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [self.predict_row(row) for row in rows]

    def equation(self) -> str:
        equation = f"y = {self.intercept:.6f}"

        for name, coefficient in zip(self.feature_names, self.coefficients):
            sign = "+" if coefficient >= 0 else "-"
            equation += (
                f" {sign} {abs(coefficient):.6f}*{name}"
            )

        return equation

    def summary(self) -> None:
        print("\nRegression equation:")
        print(self.equation())

        print("\nCoefficients:")
        print(f"{'Feature':<24} {'Coefficient':>14} {'Std. Error':>14}")
        print("-" * 56)
        print(
            f"{'Intercept':<24} "
            f"{self.intercept:>14.6f} "
            f"{'-':>14}"
        )

        for name, coefficient, error in zip(
            self.feature_names,
            self.coefficients,
            self.coefficient_standard_errors,
        ):
            print(
                f"{name:<24} "
                f"{coefficient:>14.6f} "
                f"{error:>14.6f}"
            )

        print("\nFit metrics:")
        print(f"R-squared:          {self.metrics.r_squared:.6f}")
        print(f"Adjusted R-squared: {self.metrics.adjusted_r_squared:.6f}")
        print(f"MAE:                {self.metrics.mae:.6f}")
        print(f"MSE:                {self.metrics.mse:.6f}")
        print(f"RMSE:               {self.metrics.rmse:.6f}")


def validate_regression_data(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
) -> None:
    if not X:
        raise ValueError("X must contain at least one observation.")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of observations.")

    feature_count = len(X[0])

    if feature_count == 0:
        raise ValueError("At least one predictor is required.")

    if any(len(row) != feature_count for row in X):
        raise ValueError("Every X row must contain the same number of features.")

    if len(X) <= feature_count:
        raise ValueError(
            "There must be more observations than predictor columns "
            "when fitting a model with an intercept."
        )

    for row in X:
        for value in row:
            if not math.isfinite(value):
                raise ValueError("Predictor values must be finite numbers.")

    for value in y:
        if not math.isfinite(value):
            raise ValueError("Target values must be finite numbers.")


def design_matrix(X: Sequence[Sequence[float]]) -> list[list[float]]:
    """
    Adds a leading column of ones.

    The first coefficient therefore represents the intercept.
    """
    return [[1.0, *map(float, row)] for row in X]


def calculate_regression_metrics(
    y: Sequence[float],
    predictions: Sequence[float],
    predictor_count: int,
) -> RegressionMetrics:
    if len(y) != len(predictions):
        raise ValueError("y and predictions must have equal lengths.")

    residuals = [
        actual - predicted
        for actual, predicted in zip(y, predictions)
    ]

    sse = sum(error ** 2 for error in residuals)
    mae = sum(abs(error) for error in residuals) / len(residuals)
    mse = sse / len(residuals)
    rmse = math.sqrt(mse)

    y_mean = mean(y)
    total_sum_of_squares = sum(
        (actual - y_mean) ** 2 for actual in y
    )

    if total_sum_of_squares < 1e-15:
        r_squared = 1.0 if sse < 1e-15 else 0.0
    else:
        r_squared = 1.0 - sse / total_sum_of_squares

    n = len(y)
    p = predictor_count

    if n > p + 1 and r_squared != 1.0:
        adjusted = 1.0 - (
            (1.0 - r_squared) * (n - 1) / (n - p - 1)
        )
    elif n > p + 1:
        adjusted = 1.0
    else:
        adjusted = float("nan")

    return RegressionMetrics(
        r_squared=r_squared,
        adjusted_r_squared=adjusted,
        mse=mse,
        rmse=rmse,
        mae=mae,
        sse=sse,
    )


def fit_multiple_regression(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    feature_names: Sequence[str] | None = None,
) -> LinearRegressionModel:
    """
    Ordinary Least Squares using the normal equation:

        beta = (X^T X)^(-1) X^T y

    This is intentionally implemented explicitly for educational purposes.
    Production numerical software generally uses QR or SVD decomposition,
    which is more stable for poorly conditioned matrices.
    """
    validate_regression_data(X, y)

    p = len(X[0])

    if feature_names is None:
        feature_names = [
            f"x{i + 1}"
            for i in range(p)
        ]

    if len(feature_names) != p:
        raise ValueError("feature_names must match the number of predictors.")

    A = design_matrix(X)
    A_transpose = transpose(A)

    normal_matrix = matrix_multiply(A_transpose, A)
    inverse_normal_matrix = matrix_inverse(normal_matrix)

    y_column = [[float(value)] for value in y]
    A_transpose_y = matrix_multiply(A_transpose, y_column)

    beta_column = matrix_multiply(
        inverse_normal_matrix,
        A_transpose_y,
    )

    beta = [row[0] for row in beta_column]
    intercept = beta[0]
    coefficients = beta[1:]

    predictions = [
        intercept + dot(coefficients, row)
        for row in X
    ]

    residuals = [
        actual - predicted
        for actual, predicted in zip(y, predictions)
    ]

    metrics = calculate_regression_metrics(
        y,
        predictions,
        predictor_count=p,
    )

    degrees_of_freedom = len(y) - (p + 1)

    if degrees_of_freedom <= 0:
        raise ValueError(
            "Insufficient residual degrees of freedom for variance estimation."
        )

    residual_variance = metrics.sse / degrees_of_freedom

    covariance_matrix = [
        [
            residual_variance * inverse_normal_matrix[row][column]
            for column in range(p + 1)
        ]
        for row in range(p + 1)
    ]

    standard_errors = [
        math.sqrt(max(covariance_matrix[index][index], 0.0))
        for index in range(1, p + 1)
    ]

    return LinearRegressionModel(
        intercept=intercept,
        coefficients=coefficients,
        feature_names=list(feature_names),
        training_predictions=predictions,
        training_residuals=residuals,
        metrics=metrics,
        residual_variance=residual_variance,
        coefficient_standard_errors=standard_errors,
    )


# ---------------------------------------------------------------------------
# Correlation and multicollinearity diagnostics
# ---------------------------------------------------------------------------

def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Correlation requires equal-length sequences of size >= 2.")

    x_mean = mean(x)
    y_mean = mean(y)

    numerator = sum(
        (a - x_mean) * (b - y_mean)
        for a, b in zip(x, y)
    )

    x_scale = math.sqrt(
        sum((a - x_mean) ** 2 for a in x)
    )
    y_scale = math.sqrt(
        sum((b - y_mean) ** 2 for b in y)
    )

    if x_scale < 1e-15 or y_scale < 1e-15:
        raise ValueError("Correlation is undefined for a constant variable.")

    return numerator / (x_scale * y_scale)


def correlation_matrix(
    X: Sequence[Sequence[float]],
) -> list[list[float]]:
    if not X:
        return []

    columns = transpose(X)
    return [
        [
            correlation(column_a, column_b)
            for column_b in columns
        ]
        for column_a in columns
    ]


def calculate_vif(X: Sequence[Sequence[float]]) -> list[float]:
    """
    Variance Inflation Factor:

        VIF_j = 1 / (1 - R_j^2)

    R_j^2 comes from regressing predictor j against all other predictors.

    A high VIF indicates that a predictor can be explained well by the
    remaining predictors, making individual coefficient estimates unstable.
    """
    if not X:
        return []

    p = len(X[0])

    if p == 1:
        return [1.0]

    values: list[float] = []

    for target_index in range(p):
        target = [
            row[target_index]
            for row in X
        ]

        other_indices = [
            index
            for index in range(p)
            if index != target_index
        ]

        predictors = [
            [row[index] for index in other_indices]
            for row in X
        ]

        try:
            model = fit_multiple_regression(
                predictors,
                target,
            )
            r_squared = model.metrics.r_squared

            if r_squared >= 1.0 - 1e-12:
                values.append(float("inf"))
            else:
                values.append(1.0 / (1.0 - r_squared))
        except ValueError:
            values.append(float("inf"))

    return values


# ---------------------------------------------------------------------------
# Gradient descent
# ---------------------------------------------------------------------------

def fit_gradient_descent(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    learning_rate: float = 0.01,
    epochs: int = 5000,
) -> tuple[float, list[float], list[float]]:
    """
    Estimates regression parameters by minimizing mean squared error.

    Unlike the normal equation implementation, this method is iterative.
    Feature scaling is important because predictors with very different
    magnitudes can produce inefficient or unstable optimization.
    """
    validate_regression_data(X, y)

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    n = len(X)
    p = len(X[0])

    intercept = 0.0
    coefficients = [0.0] * p
    losses: list[float] = []

    for _ in range(epochs):
        predictions = [
            intercept + dot(coefficients, row)
            for row in X
        ]

        errors = [
            prediction - actual
            for prediction, actual in zip(predictions, y)
        ]

        loss = sum(error ** 2 for error in errors) / n
        losses.append(loss)

        intercept_gradient = 2.0 * sum(errors) / n

        coefficient_gradients = [
            2.0 * sum(
                error * row[feature_index]
                for error, row in zip(errors, X)
            ) / n
            for feature_index in range(p)
        ]

        intercept -= learning_rate * intercept_gradient

        coefficients = [
            coefficient - learning_rate * gradient
            for coefficient, gradient in zip(
                coefficients,
                coefficient_gradients,
            )
        ]

    return intercept, coefficients, losses


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------

@dataclass
class StandardScaler:
    means: list[float]
    standard_deviations: list[float]

    @classmethod
    def fit(cls, X: Sequence[Sequence[float]]) -> "StandardScaler":
        if not X:
            raise ValueError("Cannot fit a scaler to empty data.")

        columns = transpose(X)
        means = [mean(column) for column in columns]
        standard_deviations = [
            standard_deviation(column, sample=False)
            for column in columns
        ]

        if any(scale < 1e-15 for scale in standard_deviations):
            raise ValueError(
                "A constant feature cannot be standardized."
            )

        return cls(means, standard_deviations)

    def transform(
        self,
        X: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        if not X:
            return []

        if len(X[0]) != len(self.means):
            raise ValueError("Feature count does not match the scaler.")

        return [
            [
                (value - center) / scale
                for value, center, scale in zip(
                    row,
                    self.means,
                    self.standard_deviations,
                )
            ]
            for row in X
        ]

    def inverse_transform(
        self,
        X: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        return [
            [
                value * scale + center
                for value, center, scale in zip(
                    row,
                    self.means,
                    self.standard_deviations,
                )
            ]
            for row in X
        ]


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def train_test_split(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[float],
    list[float],
]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    if len(X) != len(y):
        raise ValueError("X and y must have equal lengths.")

    indices = list(range(len(X)))
    random.Random(seed).shuffle(indices)

    test_size = max(1, round(len(X) * test_ratio))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    if not train_indices:
        raise ValueError("Training set cannot be empty.")

    X_train = [list(X[index]) for index in train_indices]
    X_test = [list(X[index]) for index in test_indices]
    y_train = [y[index] for index in train_indices]
    y_test = [y[index] for index in test_indices]

    return X_train, X_test, y_train, y_test


def evaluate_predictions(
    y: Sequence[float],
    predictions: Sequence[float],
) -> RegressionMetrics:
    if len(y) != len(predictions):
        raise ValueError("y and predictions must have equal lengths.")

    return calculate_regression_metrics(
        y,
        predictions,
        predictor_count=1,
    )


# ---------------------------------------------------------------------------
# Realistic examples
# ---------------------------------------------------------------------------

def demonstrate_simple_regression() -> None:
    print("=" * 78)
    print("SIMPLE LINEAR REGRESSION")
    print("=" * 78)

    study_hours = [
        1, 2, 3, 4, 5, 6, 7, 8,
    ]

    exam_scores = [
        45, 49, 54, 61, 65, 70, 76, 81,
    ]

    model = fit_simple_regression(
        study_hours,
        exam_scores,
    )

    print(f"Intercept: {model.intercept:.4f}")
    print(f"Slope:     {model.slope:.4f}")
    print(f"R²:        {model.r_squared:.4f}")

    new_hours = 6.5
    predicted_score = (
        model.intercept + model.slope * new_hours
    )

    print(
        f"Predicted score for {new_hours} study hours: "
        f"{predicted_score:.2f}"
    )

    print("\nInterpretation:")
    print(
        f"Each additional study hour is associated with an estimated "
        f"{model.slope:.2f}-point increase in exam score in this dataset."
    )
    print(
        f"The intercept ({model.intercept:.2f}) is the model's predicted "
        f"score when study hours equal zero."
    )


def demonstrate_multiple_regression() -> LinearRegressionModel:
    print("\n" + "=" * 78)
    print("MULTIPLE LINEAR REGRESSION")
    print("=" * 78)

    # Predict monthly electricity consumption from:
    # temperature, household size, and floor area.
    X = [
        [18, 2, 70],
        [20, 3, 80],
        [22, 3, 85],
        [24, 4, 90],
        [25, 4, 110],
        [27, 5, 120],
        [29, 5, 130],
        [31, 6, 140],
        [33, 6, 150],
        [35, 7, 165],
        [17, 2, 65],
        [23, 3, 88],
        [28, 5, 125],
        [32, 6, 145],
        [26, 4, 105],
    ]

    y = [
        290, 330, 345, 390, 425,
        470, 505, 555, 590, 650,
        275, 360, 520, 575, 445,
    ]

    names = [
        "temperature_c",
        "household_size",
        "floor_area_m2",
    ]

    model = fit_multiple_regression(
        X,
        y,
        feature_names=names,
    )

    model.summary()

    new_house = [30, 5, 135]
    prediction = model.predict_row(new_house)

    print(
        f"\nPredicted monthly consumption for "
        f"temperature={new_house[0]}°C, "
        f"household_size={new_house[1]}, "
        f"floor_area={new_house[2]}m²: "
        f"{prediction:.2f}"
    )

    print(
        "\nCoefficient interpretation requires holding the other "
        "predictors constant."
    )

    for name, coefficient in zip(
        model.feature_names,
        model.coefficients,
    ):
        direction = "increase" if coefficient >= 0 else "decrease"
        print(
            f"  {name}: a one-unit increase is associated with an "
            f"estimated {abs(coefficient):.4f}-unit {direction} in y, "
            f"holding other predictors constant."
        )

    return model


def demonstrate_multicollinearity(X: Sequence[Sequence[float]]) -> None:
    print("\n" + "=" * 78)
    print("MULTICOLLINEARITY DIAGNOSTICS")
    print("=" * 78)

    matrix = correlation_matrix(X)

    print("Predictor correlation matrix:")
    for row in matrix:
        print("  " + " ".join(f"{value:8.4f}" for value in row))

    vif_values = calculate_vif(X)

    print("\nVIF values:")
    for index, vif in enumerate(vif_values, start=1):
        if math.isinf(vif):
            display = "infinite"
        else:
            display = f"{vif:.4f}"
        print(f"  x{index}: {display}")

    print(
        "\nHigh VIF values indicate that individual coefficients may be "
        "unstable even when the model's overall predictive fit is strong."
    )


def demonstrate_gradient_descent(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
) -> None:
    print("\n" + "=" * 78)
    print("GRADIENT DESCENT")
    print("=" * 78)

    scaler = StandardScaler.fit(X)
    X_scaled = scaler.transform(X)

    intercept, coefficients, losses = fit_gradient_descent(
        X_scaled,
        y,
        learning_rate=0.03,
        epochs=8000,
    )

    print(f"Estimated intercept: {intercept:.4f}")

    for index, coefficient in enumerate(coefficients, start=1):
        print(
            f"Standardized coefficient x{index}: "
            f"{coefficient:.4f}"
        )

    print(f"Initial MSE: {losses[0]:.4f}")
    print(f"Final MSE:   {losses[-1]:.4f}")

    print(
        "\nStandardization makes the gradient descent optimization easier "
        "when predictors have very different numerical scales."
    )


def demonstrate_train_test_evaluation() -> None:
    print("\n" + "=" * 78)
    print("TRAIN / TEST EVALUATION")
    print("=" * 78)

    X = [
        [10, 2],
        [12, 3],
        [15, 4],
        [18, 5],
        [20, 6],
        [22, 7],
        [25, 7],
        [28, 8],
        [30, 9],
        [33, 10],
        [35, 11],
        [38, 12],
        [40, 12],
        [43, 13],
        [45, 14],
        [48, 15],
        [50, 16],
        [53, 17],
        [55, 18],
        [58, 19],
    ]

    y = [
        30, 37, 43, 51, 56,
        62, 68, 75, 80, 88,
        92, 100, 105, 113, 118,
        126, 132, 139, 145, 153,
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_ratio=0.25,
        seed=7,
    )

    model = fit_multiple_regression(
        X_train,
        y_train,
        feature_names=["experience", "team_size"],
    )

    predictions = model.predict(X_test)
    test_metrics = evaluate_predictions(
        y_test,
        predictions,
    )

    print(f"Training RMSE: {model.metrics.rmse:.4f}")
    print(f"Test RMSE:     {test_metrics.rmse:.4f}")
    print(f"Test MAE:      {test_metrics.mae:.4f}")

    print("\nTest observations:")
    for features, actual, predicted in zip(
        X_test,
        y_test,
        predictions,
    ):
        print(
            f"  X={features} "
            f"actual={actual:.2f} "
            f"predicted={predicted:.2f} "
            f"residual={actual - predicted:.2f}"
        )


def demonstrate_edge_cases() -> None:
    print("\n" + "=" * 78)
    print("VALIDATION AND FAILURE CONDITIONS")
    print("=" * 78)

    cases = [
        (
            "Mismatched simple-regression lengths",
            lambda: fit_simple_regression([1, 2], [3]),
        ),
        (
            "Constant simple-regression predictor",
            lambda: fit_simple_regression([5, 5, 5], [10, 11, 12]),
        ),
        (
            "Singular multiple-regression predictors",
            lambda: fit_multiple_regression(
                [
                    [1, 2],
                    [2, 4],
                    [3, 6],
                    [4, 8],
                ],
                [5, 7, 9, 11],
            ),
        ),
        (
            "Non-finite predictor",
            lambda: fit_multiple_regression(
                [[1], [2], [float("nan")]],
                [2, 4, 6],
            ),
        ),
    ]

    for description, operation in cases:
        try:
            operation()
        except (ValueError, ZeroDivisionError) as error:
            print(f"{description}: correctly rejected")
            print(f"  Reason: {error}")


def demonstrate_residual_diagnostics(
    model: LinearRegressionModel,
) -> None:
    print("\n" + "=" * 78)
    print("RESIDUAL DIAGNOSTICS")
    print("=" * 78)

    residuals = model.training_residuals

    print(f"Residual mean: {mean(residuals):.12f}")
    print(
        f"Residual standard deviation: "
        f"{standard_deviation(residuals):.6f}"
    )

    print("\nResiduals:")
    for index, residual in enumerate(residuals, start=1):
        print(f"  observation {index:02d}: {residual: .6f}")

    print(
        "\nFor an OLS model with an intercept, residuals should sum very "
        "close to zero. Residual patterns can still reveal nonlinear "
        "relationships, unequal variance, outliers, or omitted variables."
    )


def demonstrate_coefficient_confidence_intervals(
    model: LinearRegressionModel,
) -> None:
    print("\n" + "=" * 78)
    print("APPROXIMATE 95% COEFFICIENT INTERVALS")
    print("=" * 78)

    # A normal critical value is used here instead of a t-distribution so
    # that the implementation remains entirely within the standard library.
    z_critical = 1.96

    for name, coefficient, standard_error in zip(
        model.feature_names,
        model.coefficients,
        model.coefficient_standard_errors,
    ):
        lower = coefficient - z_critical * standard_error
        upper = coefficient + z_critical * standard_error

        print(
            f"{name:<24} "
            f"estimate={coefficient: .6f} "
            f"95% interval=[{lower:.6f}, {upper:.6f}]"
        )

    print(
        "\nThese intervals use a normal approximation. For small samples, "
        "a t critical value is generally more appropriate."
    )


def demonstrate_model_comparison() -> None:
    print("\n" + "=" * 78)
    print("SIMPLE VS MULTIPLE REGRESSION")
    print("=" * 78)

    size = [50, 60, 70, 80, 90, 100, 110, 120]
    bedrooms = [1, 1, 2, 2, 3, 3, 4, 4]
    prices = [120, 135, 160, 175, 210, 225, 270, 290]

    simple = fit_simple_regression(
        size,
        prices,
    )

    multiple = fit_multiple_regression(
        [
            [area, bedroom_count]
            for area, bedroom_count in zip(size, bedrooms)
        ],
        prices,
        feature_names=["area_m2", "bedrooms"],
    )

    print(
        f"Simple regression R²:   {simple.r_squared:.4f}"
    )
    print(
        f"Multiple regression R²: {multiple.metrics.r_squared:.4f}"
    )

    print("\nSimple model:")
    print(
        f"price = {simple.intercept:.4f} "
        f"+ {simple.slope:.4f}*area"
    )

    print("\nMultiple model:")
    print(multiple.equation())

    print(
        "\nThe multiple model estimates the partial effect of area while "
        "accounting for the bedroom predictor. This is different from "
        "simply adding independent one-variable regressions together."
    )


def main() -> None:
    demonstrate_simple_regression()

    multiple_model = demonstrate_multiple_regression()

    X_for_diagnostics = [
        [18, 2, 70],
        [20, 3, 80],
        [22, 3, 85],
        [24, 4, 90],
        [25, 4, 110],
        [27, 5, 120],
        [29, 5, 130],
        [31, 6, 140],
        [33, 6, 150],
        [35, 7, 165],
        [17, 2, 65],
        [23, 3, 88],
        [28, 5, 125],
        [32, 6, 145],
        [26, 4, 105],
    ]

    demonstrate_multicollinearity(X_for_diagnostics)
    demonstrate_gradient_descent(
        X_for_diagnostics,
        [
            290, 330, 345, 390, 425,
            470, 505, 555, 590, 650,
            275, 360, 520, 575, 445,
        ],
    )
    demonstrate_train_test_evaluation()
    demonstrate_residual_diagnostics(multiple_model)
    demonstrate_coefficient_confidence_intervals(multiple_model)
    demonstrate_model_comparison()
    demonstrate_edge_cases()

    print("\n" + "=" * 78)
    print("LINEAR REGRESSION DEMONSTRATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
