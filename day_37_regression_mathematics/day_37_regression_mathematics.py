"""
Regression Mathematics: Least Squares, Residuals, MSE, R², Adjusted R²

A self-contained progression from simple linear regression to multiple
linear regression, diagnostics, numerical validation, prediction intervals,
and model-comparison metrics.

Only the Python standard library is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Core mathematical data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegressionResult:
    coefficients: tuple[float, ...]
    predictions: tuple[float, ...]
    residuals: tuple[float, ...]
    sse: float
    mse: float
    r_squared: float
    adjusted_r_squared: float
    n: int
    p: int

    @property
    def rmse(self) -> float:
        return sqrt(self.mse)

    @property
    def intercept(self) -> float:
        return self.coefficients[0]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_xy(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
) -> None:
    if not y:
        raise ValueError("The response vector cannot be empty.")

    if len(x) != len(y):
        raise ValueError("X and y must contain the same number of observations.")

    if any(len(row) == 0 for row in x):
        raise ValueError("Every observation must contain at least one predictor.")

    width = len(x[0])
    if any(len(row) != width for row in x):
        raise ValueError("All predictor rows must have the same number of columns.")

    if len(y) <= width + 1:
        raise ValueError(
            "At least p + 2 observations are required for a useful adjusted R²."
        )

    for row in x:
        for value in row:
            if not isinstance(value, (int, float)):
                raise TypeError("Predictor values must be numeric.")

    for value in y:
        if not isinstance(value, (int, float)):
            raise TypeError("Response values must be numeric.")


# ---------------------------------------------------------------------------
# Basic descriptive quantities
# ---------------------------------------------------------------------------

def total_sum_of_squares(y: Sequence[float]) -> float:
    """SST = Σ(y_i - ȳ)², the total variation in the response."""
    y_bar = mean(y)
    return sum((value - y_bar) ** 2 for value in y)


def residual_sum_of_squares(
    y: Sequence[float],
    predictions: Sequence[float],
) -> float:
    """SSE = Σ(y_i - ŷ_i)², the unexplained squared error."""
    if len(y) != len(predictions):
        raise ValueError("y and predictions must have equal length.")

    return sum((actual - predicted) ** 2 for actual, predicted in zip(y, predictions))


def mean_squared_error(
    y: Sequence[float],
    predictions: Sequence[float],
) -> float:
    """Prediction-error MSE = SSE / n."""
    if not y:
        raise ValueError("Cannot calculate MSE for an empty dataset.")

    return residual_sum_of_squares(y, predictions) / len(y)


def coefficient_of_determination(
    y: Sequence[float],
    predictions: Sequence[float],
) -> float:
    """
    R² = 1 - SSE/SST.

    R² measures the fraction of observed response variation explained by
    the fitted model relative to an intercept-only baseline.
    """
    sst = total_sum_of_squares(y)
    if sst == 0:
        raise ValueError("R² is undefined when the response has zero variance.")

    sse = residual_sum_of_squares(y, predictions)
    return 1.0 - (sse / sst)


def adjusted_r_squared(
    r_squared: float,
    n: int,
    p: int,
) -> float:
    """
    Adjusted R² = 1 - (1 - R²) * (n - 1)/(n - p - 1).

    n is the number of observations.
    p is the number of predictors, excluding the intercept.
    """
    if n <= p + 1:
        raise ValueError("Adjusted R² requires n > p + 1.")

    return 1.0 - (1.0 - r_squared) * ((n - 1) / (n - p - 1))


# ---------------------------------------------------------------------------
# Small matrix implementation
# ---------------------------------------------------------------------------

Matrix = list[list[float]]


def transpose(matrix: Matrix) -> Matrix:
    if not matrix:
        return []
    return [list(column) for column in zip(*matrix)]


def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b:
        raise ValueError("Matrices cannot be empty.")

    if len(a[0]) != len(b):
        raise ValueError("Matrix dimensions are incompatible.")

    result = [
        [0.0 for _ in range(len(b[0]))]
        for _ in range(len(a))
    ]

    for i in range(len(a)):
        for k in range(len(b)):
            for j in range(len(b[0])):
                result[i][j] += a[i][k] * b[k][j]

    return result


def identity_matrix(size: int) -> Matrix:
    return [
        [1.0 if i == j else 0.0 for j in range(size)]
        for i in range(size)
    ]


def matrix_inverse(matrix: Matrix) -> Matrix:
    """
    Gauss-Jordan inversion with partial pivoting.

    Partial pivoting reduces numerical instability by selecting the largest
    available pivot in the current column.
    """
    n = len(matrix)

    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Only non-empty square matrices can be inverted.")

    augmented = [
        list(map(float, row)) + identity_row
        for row, identity_row in zip(matrix, identity_matrix(n))
    ]

    for column in range(n):
        pivot_row = max(
            range(column, n),
            key=lambda row: abs(augmented[row][column]),
        )

        pivot = augmented[pivot_row][column]

        if abs(pivot) < 1e-12:
            raise ValueError(
                "Design matrix is singular or nearly singular. "
                "Predictors may be perfectly or nearly collinear."
            )

        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )

        pivot = augmented[column][column]

        for j in range(2 * n):
            augmented[column][j] /= pivot

        for row in range(n):
            if row == column:
                continue

            factor = augmented[row][column]

            if factor != 0.0:
                for j in range(2 * n):
                    augmented[row][j] -= factor * augmented[column][j]

    return [row[n:] for row in augmented]


def vector_from_matrix(matrix: Matrix) -> list[float]:
    if any(len(row) != 1 for row in matrix):
        raise ValueError("Expected a column vector.")
    return [row[0] for row in matrix]


# ---------------------------------------------------------------------------
# Ordinary Least Squares
# ---------------------------------------------------------------------------

def add_intercept(x: Sequence[Sequence[float]]) -> Matrix:
    return [[1.0, *map(float, row)] for row in x]


def fit_ols(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
) -> RegressionResult:
    """
    Ordinary Least Squares using the normal equation:

        β̂ = (XᵀX)^(-1)Xᵀy

    The intercept is included automatically.
    """
    validate_xy(x, y)

    design = add_intercept(x)
    xt = transpose(design)
    xtx = matrix_multiply(xt, design)
    xtx_inverse = matrix_inverse(xtx)

    y_column = [[float(value)] for value in y]
    xty = matrix_multiply(xt, y_column)
    beta_column = matrix_multiply(xtx_inverse, xty)
    coefficients = vector_from_matrix(beta_column)

    predictions = []
    for row in design:
        predictions.append(sum(a * b for a, b in zip(row, coefficients)))

    residuals = [
        actual - predicted
        for actual, predicted in zip(y, predictions)
    ]

    sse = residual_sum_of_squares(y, predictions)

    # MSE for model-error estimation uses residual degrees of freedom:
    # n - p - 1, unlike predictive MSE which divides SSE by n.
    n = len(y)
    p = len(x[0])
    error_mse = sse / (n - p - 1)

    r2 = coefficient_of_determination(y, predictions)
    adjusted_r2 = adjusted_r_squared(r2, n, p)

    return RegressionResult(
        coefficients=tuple(coefficients),
        predictions=tuple(predictions),
        residuals=tuple(residuals),
        sse=sse,
        mse=error_mse,
        r_squared=r2,
        adjusted_r_squared=adjusted_r2,
        n=n,
        p=p,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    coefficients: Sequence[float],
    observations: Sequence[Sequence[float]],
) -> list[float]:
    """Evaluate ŷ = β₀ + β₁x₁ + ... + βₚxₚ."""
    expected_predictors = len(coefficients) - 1

    predictions = []

    for row in observations:
        if len(row) != expected_predictors:
            raise ValueError(
                f"Expected {expected_predictors} predictors, got {len(row)}."
            )

        predictions.append(
            coefficients[0]
            + sum(
                coefficient * value
                for coefficient, value in zip(coefficients[1:], row)
            )
        )

    return predictions


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------

def residual_summary(result: RegressionResult) -> dict[str, float]:
    residuals = result.residuals
    return {
        "mean": sum(residuals) / len(residuals),
        "minimum": min(residuals),
        "maximum": max(residuals),
        "absolute_mean": sum(abs(r) for r in residuals) / len(residuals),
        "rmse": result.rmse,
    }


def leverage_values(x: Sequence[Sequence[float]]) -> list[float]:
    """
    Hat-matrix diagonal:

        H = X(XᵀX)^(-1)Xᵀ

    h_ii measures how unusual observation i's predictor values are.
    """
    design = add_intercept(x)
    xt = transpose(design)
    inverse = matrix_inverse(matrix_multiply(xt, design))
    hat_matrix = matrix_multiply(
        matrix_multiply(design, inverse),
        transpose(design),
    )
    return [hat_matrix[i][i] for i in range(len(design))]


def standardized_residuals(
    result: RegressionResult,
) -> list[float]:
    """
    Internally standardized residuals use the estimated residual standard
    deviation and leverage. They help identify observations with unusually
    large residual errors.
    """
    mse = result.sse / (result.n - result.p - 1)
    sigma = sqrt(mse)
    leverage = leverage_values_from_result(result)

    values = []

    for residual, h in zip(result.residuals, leverage):
        denominator = sigma * sqrt(max(1e-15, 1.0 - h))
        values.append(residual / denominator)

    return values


def leverage_values_from_result(
    result: RegressionResult,
) -> list[float]:
    """
    This helper reconstructs a synthetic design only when the result stores
    no original X. For real diagnostics, use diagnose_regression(), which
    retains the actual predictor matrix.
    """
    raise ValueError(
        "Original predictors are required for leverage diagnostics. "
        "Use diagnose_regression(x, y)."
    )


def diagnose_regression(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
) -> dict[str, object]:
    result = fit_ols(x, y)
    leverage = leverage_values(x)

    sigma = sqrt(result.sse / (result.n - result.p - 1))

    standardized = []
    for residual, h in zip(result.residuals, leverage):
        standardized.append(
            residual / (sigma * sqrt(max(1e-15, 1.0 - h)))
        )

    high_leverage_threshold = 2.0 * (result.p + 1) / result.n

    return {
        "result": result,
        "leverage": leverage,
        "standardized_residuals": standardized,
        "high_leverage_threshold": high_leverage_threshold,
        "high_leverage_indices": [
            index
            for index, value in enumerate(leverage)
            if value > high_leverage_threshold
        ],
        "large_residual_indices": [
            index
            for index, value in enumerate(standardized)
            if abs(value) > 2.0
        ],
    }


# ---------------------------------------------------------------------------
# Simple linear regression through covariance structure
# ---------------------------------------------------------------------------

def simple_linear_regression(
    x: Sequence[float],
    y: Sequence[float],
) -> RegressionResult:
    """
    Closed-form simple linear regression:

        b1 = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
        b0 = ȳ - b1x̄
    """
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("Simple regression requires equal vectors with >= 3 rows.")

    x_bar = mean(x)
    y_bar = mean(y)

    denominator = sum((value - x_bar) ** 2 for value in x)

    if denominator == 0:
        raise ValueError("A constant predictor cannot estimate a slope.")

    slope = sum(
        (xi - x_bar) * (yi - y_bar)
        for xi, yi in zip(x, y)
    ) / denominator

    intercept = y_bar - slope * x_bar

    predictions = [
        intercept + slope * xi
        for xi in x
    ]

    residuals = [
        yi - prediction
        for yi, prediction in zip(y, predictions)
    ]

    sse = residual_sum_of_squares(y, predictions)
    predictive_mse = sse / len(y)
    r2 = coefficient_of_determination(y, predictions)

    return RegressionResult(
        coefficients=(intercept, slope),
        predictions=tuple(predictions),
        residuals=tuple(residuals),
        sse=sse,
        mse=predictive_mse,
        r_squared=r2,
        adjusted_r_squared=adjusted_r_squared(r2, len(y), 1),
        n=len(y),
        p=1,
    )


# ---------------------------------------------------------------------------
# Formatting and examples
# ---------------------------------------------------------------------------

def print_result(title: str, result: RegressionResult) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)

    for index, coefficient in enumerate(result.coefficients):
        label = "intercept" if index == 0 else f"beta_{index}"
        print(f"{label:>12}: {coefficient:12.6f}")

    print(f"{'SSE':>12}: {result.sse:12.6f}")
    print(f"{'MSE':>12}: {result.mse:12.6f}")
    print(f"{'RMSE':>12}: {result.rmse:12.6f}")
    print(f"{'R²':>12}: {result.r_squared:12.6f}")
    print(f"{'Adjusted R²':>12}: {result.adjusted_r_squared:12.6f}")

    print("\nObserved / Predicted / Residual")
    for actual, prediction, residual in zip(
        observed_for_result(result),
        result.predictions,
        result.residuals,
    ):
        print(f"{actual:10.3f} {prediction:12.3f} {residual:12.3f}")


def observed_for_result(result: RegressionResult) -> list[float]:
    """
    RegressionResult deliberately stores predictions and residuals rather
    than duplicating y. Therefore y is reconstructed as ŷ + e for display.
    """
    return [
        prediction + residual
        for prediction, residual in zip(
            result.predictions,
            result.residuals,
        )
    ]


def compare_models(
    y: Sequence[float],
    models: dict[str, RegressionResult],
) -> None:
    print(f"\n{'=' * 72}")
    print("MODEL COMPARISON")
    print("=" * 72)
    print(
        f"{'Model':25} {'SSE':>12} {'MSE':>12} "
        f"{'R²':>10} {'Adj R²':>10}"
    )

    for name, result in models.items():
        print(
            f"{name:25} "
            f"{result.sse:12.4f} "
            f"{result.mse:12.4f} "
            f"{result.r_squared:10.4f} "
            f"{result.adjusted_r_squared:10.4f}"
        )

    print(
        "\nAdjusted R² is useful when models contain different numbers of "
        "predictors because it penalizes unnecessary complexity."
    )


# ---------------------------------------------------------------------------
# Practical case study: operational delivery-time prediction
# ---------------------------------------------------------------------------

def main() -> None:
    print("REGRESSION MATHEMATICS")
    print("Least Squares | Residuals | MSE | R² | Adjusted R²")

    # A small operational dataset:
    # x1 = processing hours
    # x2 = number of items in the order
    # y  = observed delivery time in hours
    #
    # The data are intentionally non-perfect so residual analysis has
    # meaningful behavior.
    x = [
        [2.0, 10.0],
        [3.0, 12.0],
        [4.0, 14.0],
        [5.0, 18.0],
        [6.0, 17.0],
        [7.0, 21.0],
        [8.0, 23.0],
        [9.0, 24.0],
        [10.0, 28.0],
        [11.0, 29.0],
        [12.0, 32.0],
        [13.0, 35.0],
    ]

    y = [
        8.4,
        10.2,
        11.7,
        14.1,
        14.8,
        17.0,
        18.9,
        19.7,
        23.0,
        24.2,
        26.0,
        29.1,
    ]

    result = fit_ols(x, y)

    print_result(
        "Multiple Linear Regression: Processing Time + Order Size",
        result,
    )

    diagnostics = diagnose_regression(x, y)

    print("\nResidual diagnostics")
    for key, value in residual_summary(result).items():
        print(f"{key:>18}: {value:.6f}")

    print(
        f"\nHigh-leverage threshold: "
        f"{diagnostics['high_leverage_threshold']:.4f}"
    )
    print(
        f"High-leverage observations: "
        f"{diagnostics['high_leverage_indices']}"
    )
    print(
        f"Large standardized residuals: "
        f"{diagnostics['large_residual_indices']}"
    )

    # The fitted coefficients can be used on unseen observations.
    future_orders = [
        [6.5, 20.0],
        [9.5, 25.0],
        [14.0, 38.0],
    ]

    future_predictions = predict(result.coefficients, future_orders)

    print("\nPredictions for unseen operational cases")
    for features, prediction in zip(future_orders, future_predictions):
        print(
            f"processing_hours={features[0]:4.1f}, "
            f"items={features[1]:4.1f} -> "
            f"predicted_delivery_hours={prediction:7.3f}"
        )

    # Demonstrate the closed-form simple regression independently.
    simple_x = [1, 2, 3, 4, 5, 6]
    simple_y = [2.2, 4.1, 5.9, 8.2, 10.1, 11.8]

    simple_result = simple_linear_regression(simple_x, simple_y)

    print_result(
        "Simple Linear Regression: Processing Hours -> Delivery Time",
        simple_result,
    )

    # Demonstrate why adjusted R² matters. The second model introduces an
    # additional predictor that is only weakly useful.
    one_predictor_x = [[row[0]] for row in x]
    one_predictor_result = fit_ols(one_predictor_x, y)

    compare_models(
        y,
        {
            "Processing hours": one_predictor_result,
            "Processing + order size": result,
        },
    )

    # Edge case: a constant response makes SST zero, so R² has no meaningful
    # denominator.
    print("\nEdge-case validation")

    try:
        coefficient_of_determination(
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
        )
    except ValueError as exc:
        print(f"Constant-response R² rejected: {exc}")

    # Edge case: perfect predictor collinearity makes XᵀX singular.
    try:
        fit_ols(
            [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]],
            [2.0, 4.1, 6.2, 8.0],
        )
    except ValueError as exc:
        print(f"Collinearity detected: {exc}")

    # Mathematical identity:
    #
    # SST = SSR + SSE
    #
    # where SSR = Σ(ŷ_i - ȳ)². With an intercept, this decomposition should
    # hold up to floating-point rounding.
    sst = total_sum_of_squares(y)
    sse = result.sse
    y_bar = mean(y)
    ssr = sum((prediction - y_bar) ** 2 for prediction in result.predictions)

    print("\nVariance decomposition")
    print(f"SST = {sst:.10f}")
    print(f"SSR = {ssr:.10f}")
    print(f"SSE = {sse:.10f}")
    print(f"SSR + SSE = {ssr + sse:.10f}")
    print(f"Decomposition error = {abs(sst - ssr - sse):.12f}")


if __name__ == "__main__":
    main()
