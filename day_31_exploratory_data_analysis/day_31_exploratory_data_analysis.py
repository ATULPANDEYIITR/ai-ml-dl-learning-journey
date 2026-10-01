#!/usr/bin/env python3
"""
Exploratory Data Analysis: Univariate, Bivariate, and Multivariate Analysis

A self-contained technical implementation of EDA using a synthetic sales dataset.
The program demonstrates:
- Dataset inspection and data-quality validation
- Univariate numerical and categorical analysis
- Bivariate relationships, covariance, correlation, and simple regression
- Multivariate correlation structure, grouped analysis, and multiple regression
- Outlier detection using the IQR rule and z-scores
- Missing-value analysis
- Distribution summaries
- CSV export
- A compact EDA report

Only the Python standard library is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, log
from pathlib import Path
from statistics import mean, median, stdev, variance
from typing import Any, Iterable, Sequence
import csv
import random


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SaleRecord:
    transaction_id: int
    region: str
    channel: str
    product: str
    marketing_spend: float
    discount_pct: float
    units_sold: int
    customer_rating: float
    delivery_days: float
    revenue: float
    returned: bool


@dataclass
class NumericSummary:
    count: int
    missing: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    standard_deviation: float | None
    variance: float | None
    q1: float | None
    q3: float | None
    iqr: float | None


@dataclass
class RegressionResult:
    intercept: float
    slope: float
    r_squared: float
    correlation: float
    residual_standard_error: float


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def generate_sales_dataset(
    size: int = 160,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Create a realistic business dataset for demonstrating EDA.

    The generated variables intentionally have relationships:
    - Marketing spend tends to increase units sold and revenue.
    - Discounts can increase units sold but reduce effective revenue per unit.
    - Delivery time is negatively associated with customer rating.
    - Customer rating has a weak relationship with returns.
    """
    rng = random.Random(seed)

    regions = ["North", "South", "East", "West"]
    channels = ["Online", "Retail", "Partner"]
    products = ["Alpha", "Beta", "Gamma", "Delta"]

    product_base_price = {
        "Alpha": 72.0,
        "Beta": 105.0,
        "Gamma": 145.0,
        "Delta": 185.0,
    }

    rows: list[dict[str, Any]] = []

    for transaction_id in range(1, size + 1):
        region = rng.choice(regions)
        channel = rng.choice(channels)
        product = rng.choice(products)

        marketing_spend = max(50.0, rng.gauss(950.0, 300.0))
        discount_pct = min(30.0, max(0.0, rng.gauss(10.0, 5.0)))

        channel_effect = {
            "Online": 30,
            "Retail": 10,
            "Partner": -5,
        }[channel]

        units_mean = (
            20
            + marketing_spend * 0.018
            + discount_pct * 1.4
            + channel_effect
        )

        units_sold = max(1, int(round(rng.gauss(units_mean, 8))))

        delivery_days = max(1.0, rng.gauss(
            {
                "Online": 3.5,
                "Retail": 2.5,
                "Partner": 4.5,
            }[channel],
            1.0,
        ))

        customer_rating = min(
            5.0,
            max(
                1.0,
                4.9 - delivery_days * 0.22 + rng.gauss(0, 0.45),
            ),
        )

        price = product_base_price[product]
        effective_price = price * (1 - discount_pct / 100)
        revenue = max(
            0.0,
            units_sold * effective_price + rng.gauss(0, price * 2),
        )

        return_probability = (
            0.025
            + max(0, 3.4 - customer_rating) * 0.08
            + max(0, delivery_days - 4) * 0.015
        )
        returned = rng.random() < min(0.65, return_probability)

        rows.append({
            "transaction_id": transaction_id,
            "region": region,
            "channel": channel,
            "product": product,
            "marketing_spend": round(marketing_spend, 2),
            "discount_pct": round(discount_pct, 2),
            "units_sold": units_sold,
            "customer_rating": round(customer_rating, 2),
            "delivery_days": round(delivery_days, 2),
            "revenue": round(revenue, 2),
            "returned": returned,
        })

    # Deliberately introduce a small number of missing observations so that
    # missing-value profiling can be demonstrated as part of EDA.
    if size >= 20:
        rows[17]["customer_rating"] = None
        rows[53]["marketing_spend"] = None
        rows[111]["delivery_days"] = None

    # Add a legitimate high-value transaction to make outlier analysis useful.
    if size >= 100:
        rows[99]["marketing_spend"] = 4800.0
        rows[99]["units_sold"] = 170
        rows[99]["revenue"] = 22000.0

    return rows


# ---------------------------------------------------------------------------
# General data-access helpers
# ---------------------------------------------------------------------------

def numeric_values(
    rows: Sequence[dict[str, Any]],
    column: str,
) -> list[float]:
    values = []
    for row in rows:
        value = row.get(column)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def categorical_values(
    rows: Sequence[dict[str, Any]],
    column: str,
) -> list[str]:
    return [
        str(row[column])
        for row in rows
        if row.get(column) is not None
    ]


def quantile(values: Sequence[float], probability: float) -> float:
    """
    Linear-interpolation quantile.

    This avoids relying on a third-party statistical library and gives stable
    results for both even and odd sample sizes.
    """
    if not values:
        raise ValueError("Cannot calculate a quantile for an empty dataset.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def safe_mean(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def safe_median(values: Sequence[float]) -> float | None:
    return median(values) if values else None


# ---------------------------------------------------------------------------
# Data-quality analysis
# ---------------------------------------------------------------------------

def profile_dataset(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("EDA requires at least one observation.")

    columns = list(rows[0].keys())

    missing = {}
    unique_counts = {}

    for column in columns:
        missing[column] = sum(
            1 for row in rows if row.get(column) is None
        )
        unique_counts[column] = len({
            row.get(column)
            for row in rows
            if row.get(column) is not None
        })

    return {
        "rows": len(rows),
        "columns": len(columns),
        "column_names": columns,
        "missing": missing,
        "unique_counts": unique_counts,
    }


def print_dataset_profile(rows: Sequence[dict[str, Any]]) -> None:
    profile = profile_dataset(rows)

    print("\nDATASET PROFILE")
    print("-" * 72)
    print(f"Observations : {profile['rows']}")
    print(f"Variables    : {profile['columns']}")

    print("\nColumn quality:")
    print(f"{'Column':<22}{'Missing':>10}{'Unique':>10}")
    for column in profile["column_names"]:
        print(
            f"{column:<22}"
            f"{profile['missing'][column]:>10}"
            f"{profile['unique_counts'][column]:>10}"
        )


# ---------------------------------------------------------------------------
# Univariate analysis
# ---------------------------------------------------------------------------

def summarize_numeric(
    rows: Sequence[dict[str, Any]],
    column: str,
) -> NumericSummary:
    values = numeric_values(rows, column)
    missing = sum(1 for row in rows if row.get(column) is None)

    if not values:
        return NumericSummary(
            count=0,
            missing=missing,
            minimum=None,
            maximum=None,
            mean=None,
            median=None,
            standard_deviation=None,
            variance=None,
            q1=None,
            q3=None,
            iqr=None,
        )

    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)

    return NumericSummary(
        count=len(values),
        missing=missing,
        minimum=min(values),
        maximum=max(values),
        mean=safe_mean(values),
        median=safe_median(values),
        standard_deviation=stdev(values) if len(values) >= 2 else None,
        variance=variance(values) if len(values) >= 2 else None,
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
    )


def frequency_table(
    rows: Sequence[dict[str, Any]],
    column: str,
) -> dict[Any, int]:
    frequencies: dict[Any, int] = {}

    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        frequencies[value] = frequencies.get(value, 0) + 1

    return dict(
        sorted(
            frequencies.items(),
            key=lambda item: (-item[1], str(item[0])),
        )
    )


def print_numeric_summary(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> None:
    print("\nUNIVARIATE NUMERICAL ANALYSIS")
    print("-" * 100)
    header = (
        f"{'Variable':<20}"
        f"{'N':>7}"
        f"{'Missing':>9}"
        f"{'Mean':>12}"
        f"{'Median':>12}"
        f"{'Std Dev':>12}"
        f"{'Q1':>12}"
        f"{'Q3':>12}"
    )
    print(header)

    for column in columns:
        summary = summarize_numeric(rows, column)

        def fmt(value: float | None) -> str:
            return "NA" if value is None else f"{value:.2f}"

        print(
            f"{column:<20}"
            f"{summary.count:>7}"
            f"{summary.missing:>9}"
            f"{fmt(summary.mean):>12}"
            f"{fmt(summary.median):>12}"
            f"{fmt(summary.standard_deviation):>12}"
            f"{fmt(summary.q1):>12}"
            f"{fmt(summary.q3):>12}"
        )


def print_categorical_analysis(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> None:
    print("\nUNIVARIATE CATEGORICAL ANALYSIS")
    print("-" * 72)

    for column in columns:
        frequencies = frequency_table(rows, column)
        total = sum(frequencies.values())

        print(f"\n{column}")
        for value, count in frequencies.items():
            percentage = 100 * count / total if total else 0
            print(
                f"  {str(value):<15}"
                f"count={count:<4}"
                f"share={percentage:6.2f}%"
            )


def histogram(
    values: Sequence[float],
    bins: int = 8,
) -> list[tuple[str, int]]:
    """
    Produce a textual histogram suitable for a terminal.

    Histogram binning is useful for seeing concentration, skewness, and
    unusually sparse regions without requiring a plotting library.
    """
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        return [(f"{minimum:.2f}", len(values))]

    width = (maximum - minimum) / bins
    counts = [0] * bins

    for value in values:
        index = int((value - minimum) / width)
        index = min(index, bins - 1)
        counts[index] += 1

    result = []
    for index, count in enumerate(counts):
        lower = minimum + index * width
        upper = lower + width
        result.append((f"{lower:.1f}-{upper:.1f}", count))

    return result


def print_histogram(
    rows: Sequence[dict[str, Any]],
    column: str,
    bins: int = 8,
) -> None:
    values = numeric_values(rows, column)

    print(f"\nDISTRIBUTION: {column}")
    print("-" * 72)

    for label, count in histogram(values, bins):
        print(f"{label:>18} | {'#' * min(count, 60)} ({count})")


# ---------------------------------------------------------------------------
# Outlier analysis
# ---------------------------------------------------------------------------

def iqr_outliers(
    values: Sequence[float],
) -> tuple[float, float, list[float]]:
    if len(values) < 4:
        return 0.0, 0.0, []

    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = [
        value
        for value in values
        if value < lower or value > upper
    ]

    return lower, upper, outliers


def z_score_outliers(
    values: Sequence[float],
    threshold: float = 3.0,
) -> list[float]:
    if len(values) < 2:
        return []

    standard_deviation = stdev(values)

    if standard_deviation == 0:
        return []

    average = mean(values)

    return [
        value
        for value in values
        if abs((value - average) / standard_deviation) > threshold
    ]


def print_outlier_analysis(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> None:
    print("\nOUTLIER ANALYSIS")
    print("-" * 72)

    for column in columns:
        values = numeric_values(rows, column)
        lower, upper, iqr_values = iqr_outliers(values)
        z_values = z_score_outliers(values)

        print(f"\n{column}")
        print(f"  IQR lower fence : {lower:.2f}")
        print(f"  IQR upper fence : {upper:.2f}")
        print(f"  IQR outliers    : {len(iqr_values)}")
        print(f"  Z-score outliers: {len(z_values)}")


# ---------------------------------------------------------------------------
# Bivariate analysis
# ---------------------------------------------------------------------------

def paired_numeric_values(
    rows: Sequence[dict[str, Any]],
    x_column: str,
    y_column: str,
) -> tuple[list[float], list[float]]:
    x_values = []
    y_values = []

    for row in rows:
        x = row.get(x_column)
        y = row.get(y_column)

        if (
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and isinstance(y, (int, float))
            and not isinstance(y, bool)
        ):
            x_values.append(float(x))
            y_values.append(float(y))

    return x_values, y_values


def covariance(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> float:
    if len(x_values) != len(y_values):
        raise ValueError("Paired observations must have equal lengths.")

    if len(x_values) < 2:
        raise ValueError("At least two observations are required.")

    x_mean = mean(x_values)
    y_mean = mean(y_values)

    return sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    ) / (len(x_values) - 1)


def pearson_correlation(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> float:
    if len(x_values) != len(y_values):
        raise ValueError("Paired observations must have equal lengths.")

    if len(x_values) < 2:
        raise ValueError("At least two observations are required.")

    x_mean = mean(x_values)
    y_mean = mean(y_values)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    )

    x_deviation = sum(
        (x - x_mean) ** 2 for x in x_values
    )
    y_deviation = sum(
        (y - y_mean) ** 2 for y in y_values
    )

    denominator = sqrt(x_deviation * y_deviation)

    if denominator == 0:
        raise ValueError(
            "Pearson correlation is undefined for a constant variable."
        )

    return numerator / denominator


def simple_linear_regression(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> RegressionResult:
    correlation = pearson_correlation(x_values, y_values)

    x_mean = mean(x_values)
    y_mean = mean(y_values)

    denominator = sum(
        (x - x_mean) ** 2 for x in x_values
    )

    if denominator == 0:
        raise ValueError("Regression requires variation in x.")

    slope = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    ) / denominator

    intercept = y_mean - slope * x_mean

    predictions = [
        intercept + slope * x
        for x in x_values
    ]

    residuals = [
        y - prediction
        for y, prediction in zip(y_values, predictions)
    ]

    residual_standard_error = (
        sqrt(sum(residual ** 2 for residual in residuals) / (len(x_values) - 2))
        if len(x_values) > 2
        else 0.0
    )

    return RegressionResult(
        intercept=intercept,
        slope=slope,
        r_squared=correlation ** 2,
        correlation=correlation,
        residual_standard_error=residual_standard_error,
    )


def print_bivariate_analysis(
    rows: Sequence[dict[str, Any]],
    x_column: str,
    y_column: str,
) -> None:
    x_values, y_values = paired_numeric_values(
        rows,
        x_column,
        y_column,
    )

    result = simple_linear_regression(x_values, y_values)

    print("\nBIVARIATE ANALYSIS")
    print("-" * 72)
    print(f"X variable            : {x_column}")
    print(f"Y variable            : {y_column}")
    print(f"Paired observations   : {len(x_values)}")
    print(f"Covariance            : {covariance(x_values, y_values):.4f}")
    print(f"Pearson correlation   : {result.correlation:.4f}")
    print(f"R-squared             : {result.r_squared:.4f}")
    print(f"Regression intercept  : {result.intercept:.4f}")
    print(f"Regression slope      : {result.slope:.4f}")
    print(
        "Residual standard err.: "
        f"{result.residual_standard_error:.4f}"
    )


def grouped_mean(
    rows: Sequence[dict[str, Any]],
    group_column: str,
    value_column: str,
) -> dict[Any, dict[str, float]]:
    grouped: dict[Any, list[float]] = {}

    for row in rows:
        group = row.get(group_column)
        value = row.get(value_column)

        if group is None or value is None:
            continue

        if isinstance(value, bool):
            continue

        if isinstance(value, (int, float)):
            grouped.setdefault(group, []).append(float(value))

    return {
        group: {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
        }
        for group, values in grouped.items()
    }


def print_grouped_bivariate_analysis(
    rows: Sequence[dict[str, Any]],
    group_column: str,
    value_column: str,
) -> None:
    print("\nGROUPED BIVARIATE ANALYSIS")
    print("-" * 72)

    results = grouped_mean(
        rows,
        group_column,
        value_column,
    )

    print(
        f"{group_column:<18}"
        f"{'Count':>10}"
        f"{'Mean':>14}"
        f"{'Median':>14}"
    )

    for group, result in sorted(results.items(), key=lambda item: str(item[0])):
        print(
            f"{str(group):<18}"
            f"{int(result['count']):>10}"
            f"{result['mean']:>14.2f}"
            f"{result['median']:>14.2f}"
        )


# ---------------------------------------------------------------------------
# Multivariate analysis
# ---------------------------------------------------------------------------

def correlation_matrix(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {
        column: {} for column in columns
    }

    for x_column in columns:
        for y_column in columns:
            x_values, y_values = paired_numeric_values(
                rows,
                x_column,
                y_column,
            )

            if len(x_values) < 2:
                matrix[x_column][y_column] = float("nan")
                continue

            try:
                matrix[x_column][y_column] = pearson_correlation(
                    x_values,
                    y_values,
                )
            except ValueError:
                matrix[x_column][y_column] = float("nan")

    return matrix


def print_correlation_matrix(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> None:
    matrix = correlation_matrix(rows, columns)

    print("\nMULTIVARIATE CORRELATION MATRIX")
    print("-" * 100)

    print(f"{'Variable':<20}", end="")
    for column in columns:
        print(f"{column[:10]:>11}", end="")
    print()

    for row_column in columns:
        print(f"{row_column:<20}", end="")
        for column in columns:
            value = matrix[row_column][column]
            if value != value:
                text_value = "NA"
            else:
                text_value = f"{value:.2f}"
            print(f"{text_value:>11}", end="")
        print()


def matrix_determinant_3x3(matrix: list[list[float]]) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    return (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )


def multiple_linear_regression(
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
    target_column: str,
) -> dict[str, Any]:
    """
    Ordinary least squares for three predictors using normal equations.

    This deliberately uses a small explicit matrix implementation so that the
    mechanics of multivariate modeling remain visible without NumPy.
    """
    if len(feature_columns) != 3:
        raise ValueError(
            "This educational implementation expects exactly three predictors."
        )

    observations: list[tuple[list[float], float]] = []

    for row in rows:
        features = []
        valid = True

        for column in feature_columns:
            value = row.get(column)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                valid = False
                break
            features.append(float(value))

        target = row.get(target_column)

        if (
            valid
            and isinstance(target, (int, float))
            and not isinstance(target, bool)
        ):
            observations.append((features, float(target)))

    if len(observations) < 8:
        raise ValueError("Too few complete observations for regression.")

    # Design matrix includes an intercept column.
    design = [
        [1.0, features[0], features[1], features[2]]
        for features, _ in observations
    ]
    target = [y for _, y in observations]

    # Construct X'X and X'y.
    xtx = [
        [0.0 for _ in range(4)]
        for _ in range(4)
    ]
    xty = [0.0 for _ in range(4)]

    for row, y in zip(design, target):
        for i in range(4):
            xty[i] += row[i] * y
            for j in range(4):
                xtx[i][j] += row[i] * row[j]

    # Gaussian elimination with partial pivoting.
    augmented = [
        xtx[i] + [xty[i]]
        for i in range(4)
    ]

    for pivot in range(4):
        pivot_row = max(
            range(pivot, 4),
            key=lambda row_index: abs(augmented[row_index][pivot]),
        )

        if abs(augmented[pivot_row][pivot]) < 1e-12:
            raise ValueError(
                "Regression matrix is singular or nearly singular."
            )

        augmented[pivot], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot],
        )

        pivot_value = augmented[pivot][pivot]

        for column in range(pivot, 5):
            augmented[pivot][column] /= pivot_value

        for row_index in range(4):
            if row_index == pivot:
                continue

            factor = augmented[row_index][pivot]

            for column in range(pivot, 5):
                augmented[row_index][column] -= (
                    factor * augmented[pivot][column]
                )

    coefficients = [
        augmented[index][4]
        for index in range(4)
    ]

    predictions = []
    residuals = []

    for row, actual in zip(design, target):
        predicted = sum(
            coefficient * value
            for coefficient, value in zip(coefficients, row)
        )
        predictions.append(predicted)
        residuals.append(actual - predicted)

    target_mean = mean(target)
    total_sum_squares = sum(
        (actual - target_mean) ** 2
        for actual in target
    )
    residual_sum_squares = sum(
        residual ** 2
        for residual in residuals
    )

    r_squared = (
        1 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else 0.0
    )

    return {
        "observations": len(observations),
        "intercept": coefficients[0],
        "coefficients": dict(
            zip(feature_columns, coefficients[1:])
        ),
        "r_squared": r_squared,
        "residual_sum_squares": residual_sum_squares,
    }


def print_multivariate_regression(
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
    target_column: str,
) -> None:
    result = multiple_linear_regression(
        rows,
        feature_columns,
        target_column,
    )

    print("\nMULTIVARIATE REGRESSION")
    print("-" * 72)
    print(f"Target              : {target_column}")
    print(f"Complete observations: {result['observations']}")
    print(f"Intercept            : {result['intercept']:.4f}")

    for feature, coefficient in result["coefficients"].items():
        print(f"{feature:<21}: {coefficient:.4f}")

    print(f"R-squared            : {result['r_squared']:.4f}")
    print(
        "Residual sum squares : "
        f"{result['residual_sum_squares']:.4f}"
    )


def crosstab(
    rows: Sequence[dict[str, Any]],
    row_column: str,
    column_column: str,
) -> dict[Any, dict[Any, int]]:
    table: dict[Any, dict[Any, int]] = {}

    for row in rows:
        row_value = row.get(row_column)
        column_value = row.get(column_column)

        if row_value is None or column_value is None:
            continue

        table.setdefault(row_value, {})
        table[row_value][column_value] = (
            table[row_value].get(column_value, 0) + 1
        )

    return table


def print_crosstab(
    rows: Sequence[dict[str, Any]],
    row_column: str,
    column_column: str,
) -> None:
    table = crosstab(rows, row_column, column_column)
    column_values = sorted({
        value
        for row in table.values()
        for value in row
    }, key=str)

    print("\nMULTIVARIATE CATEGORICAL CROSS-TABULATION")
    print("-" * 72)

    print(f"{row_column:<18}", end="")
    for value in column_values:
        print(f"{str(value)[:10]:>12}", end="")
    print()

    for row_value, counts in sorted(table.items(), key=lambda item: str(item[0])):
        print(f"{str(row_value):<18}", end="")
        for value in column_values:
            print(f"{counts.get(value, 0):>12}", end="")
        print()


# ---------------------------------------------------------------------------
# Relationship interpretation helpers
# ---------------------------------------------------------------------------

def correlation_strength(correlation: float) -> str:
    magnitude = abs(correlation)

    if magnitude < 0.20:
        return "very weak"
    if magnitude < 0.40:
        return "weak"
    if magnitude < 0.60:
        return "moderate"
    if magnitude < 0.80:
        return "strong"
    return "very strong"


def print_relationship_interpretation(
    correlation: float,
    x_column: str,
    y_column: str,
) -> None:
    direction = "positive" if correlation > 0 else "negative"

    if correlation == 0:
        direction = "no linear"

    print("\nRELATIONSHIP INTERPRETATION")
    print("-" * 72)
    print(
        f"The Pearson coefficient between {x_column} and {y_column} "
        f"is {correlation:.3f}."
    )
    print(
        f"The observed linear association is "
        f"{correlation_strength(correlation)} and {direction}."
    )
    print(
        "Correlation describes linear association; it does not establish "
        "that changes in one variable cause changes in the other."
    )


# ---------------------------------------------------------------------------
# CSV handling
# ---------------------------------------------------------------------------

def save_csv(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
) -> None:
    path = Path(path)

    if not rows:
        raise ValueError("Cannot export an empty dataset.")

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: str | Path) -> list[dict[str, Any]]:
    """
    Load a CSV while preserving common numeric fields.

    This function is intentionally conservative. Unknown columns remain
    strings, while recognized numeric columns are converted to floats.
    """
    numeric_columns = {
        "marketing_spend",
        "discount_pct",
        "units_sold",
        "customer_rating",
        "delivery_days",
        "revenue",
    }

    rows = []

    with Path(path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for raw_row in reader:
            row: dict[str, Any] = {}

            for key, value in raw_row.items():
                if value == "":
                    row[key] = None
                elif key in numeric_columns:
                    row[key] = float(value)
                elif key == "transaction_id":
                    row[key] = int(value)
                elif key == "returned":
                    row[key] = value.lower() == "true"
                else:
                    row[key] = value

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# EDA report orchestration
# ---------------------------------------------------------------------------

def run_eda(rows: Sequence[dict[str, Any]]) -> None:
    numeric_columns = [
        "marketing_spend",
        "discount_pct",
        "units_sold",
        "customer_rating",
        "delivery_days",
        "revenue",
    ]

    categorical_columns = [
        "region",
        "channel",
        "product",
        "returned",
    ]

    print("=" * 72)
    print("EXPLORATORY DATA ANALYSIS")
    print("Univariate | Bivariate | Multivariate")
    print("=" * 72)

    print_dataset_profile(rows)

    print_numeric_summary(
        rows,
        numeric_columns,
    )

    print_categorical_analysis(
        rows,
        categorical_columns,
    )

    print_histogram(
        rows,
        "revenue",
        bins=10,
    )

    print_outlier_analysis(
        rows,
        numeric_columns,
    )

    print_bivariate_analysis(
        rows,
        "marketing_spend",
        "revenue",
    )

    print_relationship_interpretation(
        pearson_correlation(
            *paired_numeric_values(
                rows,
                "marketing_spend",
                "revenue",
            )
        ),
        "marketing_spend",
        "revenue",
    )

    print_grouped_bivariate_analysis(
        rows,
        "channel",
        "revenue",
    )

    print_grouped_bivariate_analysis(
        rows,
        "region",
        "customer_rating",
    )

    print_correlation_matrix(
        rows,
        numeric_columns,
    )

    print_crosstab(
        rows,
        "region",
        "channel",
    )

    print_multivariate_regression(
        rows,
        [
            "marketing_spend",
            "discount_pct",
            "delivery_days",
        ],
        "revenue",
    )


# ---------------------------------------------------------------------------
# Edge-case demonstrations
# ---------------------------------------------------------------------------

def demonstrate_edge_cases() -> None:
    print("\nEDGE CASES AND FAILURE CONDITIONS")
    print("-" * 72)

    try:
        pearson_correlation(
            [5.0, 5.0, 5.0],
            [1.0, 2.0, 3.0],
        )
    except ValueError as error:
        print(f"Constant-variable correlation: handled -> {error}")

    try:
        covariance(
            [1.0],
            [2.0],
        )
    except ValueError as error:
        print(f"Insufficient observations: handled -> {error}")

    try:
        simple_linear_regression(
            [4.0, 4.0, 4.0],
            [1.0, 2.0, 3.0],
        )
    except ValueError as error:
        print(f"Zero-variance predictor: handled -> {error}")

    empty_values = []
    print(
        "Empty numeric selection: "
        f"summary={summarize_numeric([{"x": None}], "x")}"
    )

    print(
        "Important EDA rule: an outlier is an observation requiring "
        "investigation, not an automatic instruction to delete data."
    )


# ---------------------------------------------------------------------------
# Production-oriented validation
# ---------------------------------------------------------------------------

def validate_business_constraints(
    rows: Sequence[dict[str, Any]],
) -> list[str]:
    errors = []

    for index, row in enumerate(rows, start=1):
        rating = row.get("customer_rating")
        discount = row.get("discount_pct")
        delivery = row.get("delivery_days")
        revenue = row.get("revenue")

        if rating is not None and not 1 <= rating <= 5:
            errors.append(
                f"row {index}: customer_rating outside [1, 5]"
            )

        if discount is not None and not 0 <= discount <= 100:
            errors.append(
                f"row {index}: discount_pct outside [0, 100]"
            )

        if delivery is not None and delivery < 0:
            errors.append(
                f"row {index}: negative delivery_days"
            )

        if revenue is not None and revenue < 0:
            errors.append(
                f"row {index}: negative revenue"
            )

    return errors


def print_business_validation(
    rows: Sequence[dict[str, Any]],
) -> None:
    errors = validate_business_constraints(rows)

    print("\nBUSINESS-RULE VALIDATION")
    print("-" * 72)

    if not errors:
        print("No business-rule violations detected.")
        return

    for error in errors:
        print(error)


# ---------------------------------------------------------------------------
# Main executable workflow
# ---------------------------------------------------------------------------

def main() -> None:
    rows = generate_sales_dataset(
        size=160,
        seed=42,
    )

    output_path = Path("eda_sales_dataset.csv")
    save_csv(rows, output_path)

    run_eda(rows)
    print_business_validation(rows)
    demonstrate_edge_cases()

    print("\nGENERATED ARTIFACT")
    print("-" * 72)
    print(f"CSV dataset written to: {output_path.resolve()}")

    print("\nEDA DECISION FRAMEWORK")
    print("-" * 72)
    print(
        "Univariate analysis describes individual variables and their "
        "distributions."
    )
    print(
        "Bivariate analysis investigates relationships between two "
        "variables or grouped comparisons."
    )
    print(
        "Multivariate analysis examines several variables jointly to "
        "identify conditional relationships and interacting patterns."
    )
    print(
        "EDA findings should be checked against data quality, measurement "
        "definitions, missingness, outliers, and domain context before "
        "being used for modeling or operational decisions."
    )


if __name__ == "__main__":
    main()
