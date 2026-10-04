"""
Feature Scaling: Standardization, Normalization, Robust Scaling, and Transformations

A self-contained executable study of feature scaling for machine-learning data.
The implementation uses only the Python standard library and demonstrates:

- Why feature scaling matters
- Z-score standardization
- Min-max normalization
- Robust scaling using median and IQR
- Max-absolute scaling
- Log and signed-log transformations
- Square-root and Box-Cox-style positive transformations
- Quantile/rank transformation
- Training-set-only parameter fitting
- Transformation and inverse transformation
- Constant-feature handling
- Missing-value validation
- Outlier sensitivity
- Data leakage prevention
- Distance calculations before and after scaling
- A small gradient-descent example showing scale effects
- A composable preprocessing pipeline
- Comparison of scaling strategies on realistic measurements

The script is intentionally self-contained so that it can be executed directly.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


Number = float
Vector = list[Number]
Matrix = list[Vector]


# ---------------------------------------------------------------------------
# Validation and basic numerical utilities
# ---------------------------------------------------------------------------

def validate_matrix(data: Matrix, name: str = "data") -> None:
    """Validate that a matrix is rectangular and contains finite numbers."""
    if not data:
        raise ValueError(f"{name} must contain at least one row.")

    width = len(data[0])
    if width == 0:
        raise ValueError(f"{name} must contain at least one feature.")

    for row_index, row in enumerate(data):
        if len(row) != width:
            raise ValueError(
                f"{name} is not rectangular: row {row_index} has "
                f"{len(row)} values instead of {width}."
            )

        for column_index, value in enumerate(row):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"{name}[{row_index}][{column_index}] must be numeric."
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{name}[{row_index}][{column_index}] must be finite."
                )


def column(matrix: Matrix, index: int) -> Vector:
    """Extract one feature column."""
    return [row[index] for row in matrix]


def transpose(matrix: Matrix) -> Matrix:
    """Transpose a rectangular matrix."""
    validate_matrix(matrix)
    return [column(matrix, i) for i in range(len(matrix[0]))]


def quantile(values: Sequence[Number], probability: float) -> float:
    """
    Linear-interpolated quantile.

    This definition is deterministic and avoids dependence on external
    statistics packages. It is appropriate for demonstrating percentile-based
    scaling, although production systems should use a documented convention
    consistently across training and inference.
    """
    if not values:
        raise ValueError("Cannot calculate a quantile of an empty sequence.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1.")

    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def mean(values: Sequence[Number]) -> float:
    if not values:
        raise ValueError("Mean requires at least one value.")
    return statistics.fmean(values)


def population_std(values: Sequence[Number]) -> float:
    if not values:
        raise ValueError("Standard deviation requires at least one value.")
    return math.sqrt(statistics.fmean([(x - mean(values)) ** 2 for x in values]))


def print_matrix(title: str, matrix: Matrix, digits: int = 4) -> None:
    """Print a compact numerical matrix."""
    print(f"\n{title}")
    for row in matrix:
        print("  " + "  ".join(f"{value:.{digits}f}" for value in row))


# ---------------------------------------------------------------------------
# Scaler abstraction
# ---------------------------------------------------------------------------

class FeatureScaler:
    """
    Base class for fit/transform-style feature scalers.

    A scaler must be fitted on training data before it can transform new data.
    Keeping fitted parameters separate from transformation logic makes it
    possible to apply exactly the same learned mapping to validation and test
    observations.
    """

    def __init__(self) -> None:
        self.fitted = False
        self.n_features: int | None = None

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before transform()."
            )

    def _validate_input(self, data: Matrix) -> None:
        validate_matrix(data)
        if self.n_features is not None and len(data[0]) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, received {len(data[0])}."
            )


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

class StandardScaler(FeatureScaler):
    """
    Z-score standardization.

    For each feature:
        z = (x - mean) / standard_deviation

    Standardization centers the feature near zero and scales its spread to one.
    It is sensitive to extreme outliers because both mean and standard
    deviation are affected by extreme observations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.means: Vector = []
        self.scales: Vector = []

    def fit(self, data: Matrix) -> StandardScaler:
        self._validate_training_data(data)
        self.n_features = len(data[0])
        self.means = []
        self.scales = []

        for values in transpose(data):
            feature_mean = mean(values)
            feature_std = population_std(values)

            # A constant feature contains no scale information. Mapping it to
            # zero is safer than dividing by zero.
            self.means.append(feature_mean)
            self.scales.append(feature_std if feature_std > 0.0 else 1.0)

        self.fitted = True
        return self

    def transform(self, data: Matrix) -> Matrix:
        self._require_fitted()
        self._validate_input(data)
        return [
            [
                (value - self.means[j]) / self.scales[j]
                for j, value in enumerate(row)
            ]
            for row in data
        ]

    def inverse_transform(self, data: Matrix) -> Matrix:
        self._require_fitted()
        self._validate_input(data)
        return [
            [
                value * self.scales[j] + self.means[j]
                for j, value in enumerate(row)
            ]
            for row in data
        ]

    @staticmethod
    def _validate_training_data(data: Matrix) -> None:
        validate_matrix(data)


# ---------------------------------------------------------------------------
# Min-max normalization
# ---------------------------------------------------------------------------

class MinMaxScaler(FeatureScaler):
    """
    Min-max normalization.

    Default mapping:
        x_scaled = (x - min) / (max - min)

    The result is normally in [0, 1] for training observations. New values
    outside the training range can legitimately produce values below 0 or
    above 1. Clipping them is a policy choice rather than an inherent property
    of min-max scaling.
    """

    def __init__(self, lower: float = 0.0, upper: float = 1.0) -> None:
        super().__init__()
        if lower >= upper:
            raise ValueError("lower must be smaller than upper.")
        self.lower = lower
        self.upper = upper
        self.minimums: Vector = []
        self.maximums: Vector = []

    def fit(self, data: Matrix) -> MinMaxScaler:
        validate_matrix(data)
        self.n_features = len(data[0])
        self.minimums = []
        self.maximums = []

        for values in transpose(data):
            self.minimums.append(min(values))
            self.maximums.append(max(values))

        self.fitted = True
        return self

    def transform(self, data: Matrix) -> Matrix:
        self._require_fitted()
        self._validate_input(data)

        output: Matrix = []
        for row in data:
            transformed_row = []
            for j, value in enumerate(row):
                minimum = self.minimums[j]
                maximum = self.maximums[j]
                if maximum == minimum:
                    transformed_row.append(self.lower)
                    continue

                ratio = (value - minimum) / (maximum - minimum)
                transformed_row.append(
                    self.lower + ratio * (self.upper - self.lower)
                )
            output.append(transformed_row)
        return output

    def inverse_transform(self, data: Matrix) -> Matrix:
        self._require_fitted()
        self._validate_input(data)

        output: Matrix = []
        for row in data:
            restored = []
            for j, value in enumerate(row):
                minimum = self.minimums[j]
                maximum = self.maximums[j]
                if maximum == minimum:
                    restored.append(minimum)
                    continue

                ratio = (value - self.lower) / (self.upper - self.lower)
                restored.append(minimum + ratio * (maximum - minimum))
            output.append(restored)
        return output


# ---------------------------------------------------------------------------
# Robust scaling
# ---------------------------------------------------------------------------

class RobustScaler(FeatureScaler):
    """
    Median/IQR-based scaling.

    For each feature:
        scaled = (x - median) / IQR

    The median and interquartile range are less affected by isolated extreme
    observations than the mean and standard deviation, making this useful for
    heavily skewed operational or financial measurements.
    """

    def __init__(self, q_low: float = 0.25, q_high: float = 0.75) -> None:
        super().__init__()
        if not 0.0 <= q_low < q_high <= 1.0:
            raise ValueError("Quantiles must satisfy 0 <= q_low < q_high <= 1.")
        self.q_low = q_low
        self.q_high = q_high
        self.medians: Vector = []
        self.iqrs: Vector = []

    def fit(self, data: Matrix) -> RobustScaler:
        validate_matrix(data)
        self.n_features = len(data[0])
        self.medians = []
        self.iqrs = []

        for values in transpose(data):
            q1 = quantile(values, self.q_low)
            q3 = quantile(values, self.q_high)
            iqr = q3 - q1

            self.medians.append(statistics.median(values))
            self.iqrs.append(iqr if iqr > 0.0 else 1.0)

        self.fitted = True
        return self

    def transform(self, data: Matrix) -> Matrix:
        self._require_fitted()
        self._validate_input(data)
        return [
            [
                (value - self.medians[j]) / self.iqrs[j]
                for j, value in enumerate(row)
            ]
            for row in data
        ]

    def inverse_transform(self, data: Matrix) -> Matrix:
        self._require_fitted()
        self._validate_input(data)
        return [
            [
                value * self.iqrs[j] + self.medians[j]
                for j, value in enumerate(row)
            ]
            for row in data
        ]


# ---------------------------------------------------------------------------
# Max-absolute scaling
# ---------------------------------------------------------------------------

class MaxAbsScaler(FeatureScaler):
    """
    Scale each feature by its largest absolute training value.

    x_scaled = x / max(abs(x))

    This preserves zero and is useful when sparse-like signed measurements
    should remain centered around zero without subtracting a mean.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scales: Vector = []

    def fit(self, data: Matrix) -> MaxAbsScaler:
        validate_matrix(data)
        self.n_features = len(data[0])
        self.scales = [
            max(abs(value) for value in values)
            for values in transpose(data)
        ]
        self.scales = [scale if scale > 0.0 else 1.0 for scale in self.scales]
        self.fitted = True
        return self

    def transform(self, data: Matrix) -> Matrix:
        self._require_fitted()
        self._validate_input(data)
        return [
            [value / self.scales[j] for j, value in enumerate(row)]
            for row in data
        ]

    def inverse_transform(self, data: Matrix) -> Matrix:
        self._require_fitted()
        self._validate_input(data)
        return [
            [value * self.scales[j] for j, value in enumerate(row)]
            for row in data
        ]


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def log_transform(values: Iterable[Number]) -> Vector:
    """
    Natural logarithm for strictly positive values.

    Log transformation compresses large positive values and is often useful
    for right-skewed measurements such as revenue, transaction counts, or
    population-like quantities.
    """
    result = []
    for value in values:
        if value <= 0:
            raise ValueError(
                "Natural log requires strictly positive values. "
                "Use signed_log_transform() when zeros or negatives are valid."
            )
        result.append(math.log(value))
    return result


def inverse_log_transform(values: Iterable[Number]) -> Vector:
    """Undo log_transform."""
    return [math.exp(value) for value in values]


def signed_log_transform(values: Iterable[Number]) -> Vector:
    """
    Apply sign(x) * log(1 + abs(x)).

    Unlike ordinary logarithms, this transformation supports negative and zero
    values while still compressing large magnitudes.
    """
    return [
        math.copysign(math.log1p(abs(value)), value)
        if value != 0
        else 0.0
        for value in values
    ]


def inverse_signed_log_transform(values: Iterable[Number]) -> Vector:
    """Undo signed_log_transform."""
    return [
        math.copysign(math.expm1(abs(value)), value)
        if value != 0
        else 0.0
        for value in values
    ]


def sqrt_transform(values: Iterable[Number]) -> Vector:
    """
    Square-root transformation for non-negative data.

    It compresses high values less aggressively than logarithms and is useful
    for count-like measurements where zero is valid.
    """
    result = []
    for value in values:
        if value < 0:
            raise ValueError("Square-root transformation requires x >= 0.")
        result.append(math.sqrt(value))
    return result


def yeo_johnson_like_transform(
    values: Iterable[Number],
    lam: float,
) -> Vector:
    """
    A Yeo-Johnson-style power transformation with a supplied lambda.

    This implementation demonstrates the transformation family without
    depending on a statistical package that estimates lambda automatically.
    It handles zero and negative observations.

    For x >= 0:
        lambda != 0: ((x + 1)^lambda - 1) / lambda
        lambda == 0: log(x + 1)

    For x < 0:
        lambda != 2: -(((-x + 1)^(2-lambda) - 1) / (2-lambda))
        lambda == 2: -log(-x + 1)
    """
    transformed = []

    for x in values:
        if x >= 0:
            if abs(lam) < 1e-12:
                transformed.append(math.log1p(x))
            else:
                transformed.append(((x + 1.0) ** lam - 1.0) / lam)
        else:
            exponent = 2.0 - lam
            if abs(exponent) < 1e-12:
                transformed.append(-math.log1p(-x))
            else:
                transformed.append(
                    -(((-x + 1.0) ** exponent - 1.0) / exponent)
                )

    return transformed


# ---------------------------------------------------------------------------
# Rank/quantile transformation
# ---------------------------------------------------------------------------

def rank_to_uniform(values: Sequence[Number]) -> Vector:
    """
    Convert values to approximately uniform ranks in (0, 1).

    Ties receive the average rank. The implementation is deterministic and
    demonstrates the central idea of quantile-based transformation without
    requiring an external machine-learning package.
    """
    if not values:
        return []

    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)

    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1

        average_rank = (position + end - 1) / 2.0
        denominator = max(len(values) - 1, 1)

        for index in range(position, end):
            original_index = indexed[index][0]
            ranks[original_index] = average_rank / denominator

        position = end

    return ranks


# ---------------------------------------------------------------------------
# Distance demonstration
# ---------------------------------------------------------------------------

def euclidean_distance(a: Sequence[Number], b: Sequence[Number]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal length.")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compare_distance_effects(data: Matrix) -> None:
    """
    Show how a large-unit feature can dominate Euclidean distance.

    The example uses annual income and account activity. Without scaling, the
    numerical magnitude of income can overwhelm the activity feature even if
    activity is important to the model.
    """
    first = data[0]
    second = data[1]

    print("\nDistance sensitivity to feature units")
    print(f"Raw distance: {euclidean_distance(first, second):.4f}")

    scalers = {
        "Standardization": StandardScaler(),
        "Min-max normalization": MinMaxScaler(),
        "Robust scaling": RobustScaler(),
    }

    for name, scaler in scalers.items():
        transformed = scaler.fit(data).transform(data)
        distance = euclidean_distance(transformed[0], transformed[1])
        print(f"{name}: {distance:.4f}")


# ---------------------------------------------------------------------------
# Training-only preprocessing demonstration
# ---------------------------------------------------------------------------

def demonstrate_train_test_discipline() -> None:
    """
    Demonstrate why scaling parameters must be learned only from training data.

    If test observations are included while calculating means, extrema, or
    quantiles, information from the evaluation set can influence preprocessing.
    This is a form of data leakage.
    """
    train = [
        [20_000.0, 10.0],
        [30_000.0, 12.0],
        [40_000.0, 14.0],
        [50_000.0, 15.0],
    ]

    test = [
        [55_000.0, 16.0],
        [200_000.0, 17.0],
    ]

    train_fitted = StandardScaler().fit(train)
    correct = train_fitted.transform(test)

    leaked = StandardScaler().fit(train + test).transform(test)

    print("\nTraining-only fitting and leakage")
    print("Test transformed with training parameters:")
    for row in correct:
        print(" ", [round(x, 4) for x in row])

    print("Test transformed after incorrectly fitting on test data:")
    for row in leaked:
        print(" ", [round(x, 4) for x in row])

    print(
        "The two outputs differ because the second scaler has allowed test-set "
        "statistics to influence the transformation."
    )


# ---------------------------------------------------------------------------
# Gradient-descent scale demonstration
# ---------------------------------------------------------------------------

@dataclass
class LinearModel:
    weight_income: float = 0.0
    weight_activity: float = 0.0
    bias: float = 0.0

    def predict(self, row: Sequence[Number]) -> float:
        return (
            self.weight_income * row[0]
            + self.weight_activity * row[1]
            + self.bias
        )


def train_linear_regression(
    features: Matrix,
    targets: Sequence[Number],
    learning_rate: float = 0.01,
    epochs: int = 500,
) -> tuple[LinearModel, list[float]]:
    """
    Train a tiny linear model with batch gradient descent.

    The purpose is not to compete with a production optimizer. It makes the
    numerical role of feature scale visible: large feature magnitudes can
    produce gradients that require substantially smaller learning rates.
    """
    if len(features) != len(targets):
        raise ValueError("Feature rows and targets must have equal length.")
    validate_matrix(features)

    if len(features[0]) != 2:
        raise ValueError("This demonstration expects exactly two features.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    model = LinearModel()
    losses = []

    n = len(features)

    for _ in range(epochs):
        errors = [model.predict(x) - y for x, y in zip(features, targets)]

        loss = sum(error * error for error in errors) / n
        losses.append(loss)

        gradient_w0 = (
            2.0 / n
            * sum(error * row[0] for error, row in zip(errors, features))
        )
        gradient_w1 = (
            2.0 / n
            * sum(error * row[1] for error, row in zip(errors, features))
        )
        gradient_bias = 2.0 / n * sum(errors)

        model.weight_income -= learning_rate * gradient_w0
        model.weight_activity -= learning_rate * gradient_w1
        model.bias -= learning_rate * gradient_bias

        if not all(
            math.isfinite(value)
            for value in (
                model.weight_income,
                model.weight_activity,
                model.bias,
                loss,
            )
        ):
            raise FloatingPointError(
                "Gradient descent became numerically unstable. "
                "Reduce the learning rate or scale the features."
            )

    return model, losses


def demonstrate_gradient_descent() -> None:
    """
    Compare optimization behavior on raw versus standardized features.

    The target is intentionally tied to both features. Standardization puts
    the optimizer on a more balanced numerical landscape.
    """
    raw_features = [
        [20_000.0, 10.0],
        [30_000.0, 12.0],
        [40_000.0, 14.0],
        [50_000.0, 16.0],
        [60_000.0, 18.0],
        [70_000.0, 20.0],
    ]

    targets = [
        40.0,
        49.0,
        58.0,
        67.0,
        76.0,
        85.0,
    ]

    scaled_features = StandardScaler().fit(raw_features).transform(raw_features)

    print("\nGradient-descent sensitivity")

    # The raw representation needs a tiny learning rate because the income
    # feature is measured in tens of thousands.
    raw_model, raw_losses = train_linear_regression(
        raw_features,
        targets,
        learning_rate=1e-10,
        epochs=500,
    )

    # The standardized representation can use a much larger learning rate
    # because the features have comparable numerical magnitudes.
    scaled_model, scaled_losses = train_linear_regression(
        scaled_features,
        targets,
        learning_rate=0.05,
        epochs=500,
    )

    print(f"Raw final loss: {raw_losses[-1]:.6f}")
    print(f"Scaled final loss: {scaled_losses[-1]:.6f}")
    print(
        "Raw model weights:",
        round(raw_model.weight_income, 8),
        round(raw_model.weight_activity, 8),
        round(raw_model.bias, 4),
    )
    print(
        "Scaled model weights:",
        round(scaled_model.weight_income, 4),
        round(scaled_model.weight_activity, 4),
        round(scaled_model.bias, 4),
    )


# ---------------------------------------------------------------------------
# Composable preprocessing pipeline
# ---------------------------------------------------------------------------

@dataclass
class FeaturePipeline:
    """
    A small preprocessing pipeline.

    The pipeline first applies feature-wise transformations and then a scaler.
    The transform functions operate column-wise so that a skewed feature can
    be transformed before its scale is standardized.
    """

    column_transforms: dict[int, Callable[[Iterable[Number]], Vector]]
    scaler: FeatureScaler

    def fit(self, data: Matrix) -> FeaturePipeline:
        validate_matrix(data)
        transformed = self._apply_column_transforms(data)
        self.scaler.fit(transformed)
        return self

    def transform(self, data: Matrix) -> Matrix:
        transformed = self._apply_column_transforms(data)
        return self.scaler.transform(transformed)

    def _apply_column_transforms(self, data: Matrix) -> Matrix:
        validate_matrix(data)
        output = [row[:] for row in data]

        for index, transform in self.column_transforms.items():
            if not 0 <= index < len(data[0]):
                raise IndexError(
                    f"Transformation refers to nonexistent feature {index}."
                )

            transformed_column = transform(column(data, index))
            for row_index, value in enumerate(transformed_column):
                output[row_index][index] = value

        return output


# ---------------------------------------------------------------------------
# Realistic data set and demonstrations
# ---------------------------------------------------------------------------

def build_customer_measurements() -> Matrix:
    """
    Customer measurements deliberately use different units and distributions.

    Feature columns:
        monthly income in currency units
        monthly transactions
        average transaction value
        account age in months
    """
    return [
        [28_000.0, 8.0, 1_200.0, 14.0],
        [35_000.0, 12.0, 1_450.0, 20.0],
        [42_000.0, 15.0, 1_700.0, 28.0],
        [48_000.0, 20.0, 2_100.0, 35.0],
        [55_000.0, 24.0, 2_300.0, 42.0],
        [63_000.0, 31.0, 2_600.0, 51.0],
        [75_000.0, 38.0, 3_000.0, 66.0],
        [95_000.0, 47.0, 3_500.0, 82.0],
        [140_000.0, 75.0, 5_100.0, 120.0],
        [850_000.0, 110.0, 8_900.0, 144.0],
    ]


def demonstrate_scalers(data: Matrix) -> None:
    """Fit and compare the principal scaling methods."""
    print_matrix("Raw customer measurements", data, digits=2)

    scalers = [
        ("Standardization", StandardScaler()),
        ("Min-max normalization", MinMaxScaler()),
        ("Robust scaling", RobustScaler()),
        ("Max-absolute scaling", MaxAbsScaler()),
    ]

    for name, scaler in scalers:
        transformed = scaler.fit(data).transform(data)
        print_matrix(name, transformed, digits=3)


def demonstrate_transformations() -> None:
    """Demonstrate transformations for skewed and signed measurements."""
    revenue = [100.0, 250.0, 500.0, 1_000.0, 5_000.0, 25_000.0, 250_000.0]
    signed_changes = [-500.0, -100.0, -10.0, 0.0, 15.0, 200.0, 2_000.0]
    counts = [0.0, 1.0, 4.0, 9.0, 25.0, 100.0, 400.0]

    print("\nTransformations for skewed and signed features")
    print("Revenue:", revenue)
    print(
        "Log:",
        [round(x, 4) for x in log_transform(revenue)],
    )
    print(
        "Inverse log:",
        [round(x, 2) for x in inverse_log_transform(log_transform(revenue))],
    )

    print("Signed changes:", signed_changes)
    print(
        "Signed log:",
        [round(x, 4) for x in signed_log_transform(signed_changes)],
    )

    print("Counts:", counts)
    print(
        "Square root:",
        [round(x, 4) for x in sqrt_transform(counts)],
    )

    mixed = [-100.0, -10.0, -1.0, 0.0, 2.0, 20.0, 100.0]
    print("Mixed signed feature:", mixed)
    print(
        "Yeo-Johnson-style lambda=0.5:",
        [round(x, 4) for x in yeo_johnson_like_transform(mixed, 0.5)],
    )

    print(
        "Rank-to-uniform:",
        [round(x, 4) for x in rank_to_uniform(revenue)],
    )


def demonstrate_outlier_effect() -> None:
    """
    Show why robust scaling exists.

    The final observation is an extreme income outlier. Standardization shifts
    because its mean and standard deviation respond to the outlier, while
    median/IQR scaling is intentionally more resistant.
    """
    ordinary = [
        [30_000.0],
        [32_000.0],
        [35_000.0],
        [37_000.0],
        [40_000.0],
        [42_000.0],
    ]

    with_outlier = ordinary + [[2_000_000.0]]

    standard_without = StandardScaler().fit(ordinary).transform(ordinary)
    standard_with = StandardScaler().fit(with_outlier).transform(ordinary)

    robust_without = RobustScaler().fit(ordinary).transform(ordinary)
    robust_with = RobustScaler().fit(with_outlier).transform(ordinary)

    print("\nOutlier sensitivity")
    print("First ordinary value under standard scaling:")
    print("  without outlier:", round(standard_without[0][0], 4))
    print("  with outlier:   ", round(standard_with[0][0], 4))

    print("First ordinary value under robust scaling:")
    print("  without outlier:", round(robust_without[0][0], 4))
    print("  with outlier:   ", round(robust_with[0][0], 4))


def demonstrate_constant_feature() -> None:
    """
    Demonstrate safe handling of zero-variance features.

    A constant feature contains no discriminating information. A scaler must
    avoid division by zero; mapping it to a fixed value is a practical policy.
    """
    data = [
        [10.0, 100.0],
        [20.0, 100.0],
        [30.0, 100.0],
    ]

    for name, scaler in [
        ("Standardization", StandardScaler()),
        ("Min-max", MinMaxScaler()),
        ("Robust", RobustScaler()),
        ("Max-absolute", MaxAbsScaler()),
    ]:
        transformed = scaler.fit(data).transform(data)
        print(f"\n{name} with constant feature:")
        print_matrix("Transformed", transformed, digits=3)


def demonstrate_pipeline() -> None:
    """
    Build a pipeline for a skewed financial feature.

    The transaction value is first transformed using log1p, then all features
    are standardized. The key design point is that transformation parameters
    and scaling parameters are fitted on the training matrix only.
    """
    train = [
        [30_000.0, 8.0, 100.0],
        [35_000.0, 10.0, 250.0],
        [40_000.0, 15.0, 500.0],
        [50_000.0, 20.0, 1_000.0],
        [65_000.0, 25.0, 2_500.0],
        [80_000.0, 32.0, 5_000.0],
    ]

    inference = [
        [55_000.0, 18.0, 750.0],
        [100_000.0, 40.0, 15_000.0],
    ]

    pipeline = FeaturePipeline(
        column_transforms={
            2: lambda values: [math.log1p(value) for value in values]
        },
        scaler=StandardScaler(),
    )

    pipeline.fit(train)
    transformed = pipeline.transform(inference)

    print_matrix(
        "Pipeline output: log1p transaction value followed by standardization",
        transformed,
        digits=4,
    )


def demonstrate_inverse_transform() -> None:
    """Verify that fitted scaler parameters can restore original values."""
    data = [
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0],
    ]

    scaler = StandardScaler().fit(data)
    scaled = scaler.transform(data)
    restored = scaler.inverse_transform(scaled)

    print_matrix("Original", data)
    print_matrix("Scaled", scaled)
    print_matrix("Restored", restored)

    maximum_error = max(
        abs(original - recovered)
        for original_row, restored_row in zip(data, restored)
        for original, recovered in zip(original_row, restored_row)
    )

    print(f"Maximum reconstruction error: {maximum_error:.12f}")


# ---------------------------------------------------------------------------
# Validation and failure demonstrations
# ---------------------------------------------------------------------------

def demonstrate_failure_conditions() -> None:
    """Exercise important validation rules."""
    print("\nValidation and failure conditions")

    try:
        StandardScaler().transform([[1.0, 2.0]])
    except RuntimeError as error:
        print("Unfitted scaler:", error)

    try:
        log_transform([10.0, 0.0, 20.0])
    except ValueError as error:
        print("Invalid logarithm:", error)

    try:
        sqrt_transform([4.0, -1.0])
    except ValueError as error:
        print("Invalid square root:", error)

    try:
        StandardScaler().fit([[1.0, 2.0], [3.0]])
    except ValueError as error:
        print("Non-rectangular matrix:", error)

    scaler = StandardScaler().fit([[1.0, 2.0], [2.0, 4.0]])
    try:
        scaler.transform([[1.0, 2.0, 3.0]])
    except ValueError as error:
        print("Wrong feature count:", error)

    try:
        MinMaxScaler(lower=1.0, upper=1.0)
    except ValueError as error:
        print("Invalid min-max range:", error)


# ---------------------------------------------------------------------------
# Concept-specific diagnostics
# ---------------------------------------------------------------------------

def print_scaler_parameters(scaler: FeatureScaler) -> None:
    """Print learned parameters without exposing implementation internals."""
    if isinstance(scaler, StandardScaler):
        print("Means:", [round(x, 3) for x in scaler.means])
        print("Standard deviations:", [round(x, 3) for x in scaler.scales])

    elif isinstance(scaler, MinMaxScaler):
        print("Minimums:", [round(x, 3) for x in scaler.minimums])
        print("Maximums:", [round(x, 3) for x in scaler.maximums])

    elif isinstance(scaler, RobustScaler):
        print("Medians:", [round(x, 3) for x in scaler.medians])
        print("IQRs:", [round(x, 3) for x in scaler.iqrs])

    elif isinstance(scaler, MaxAbsScaler):
        print("Maximum absolute values:", [round(x, 3) for x in scaler.scales])


def demonstrate_parameter_interpretation(data: Matrix) -> None:
    """
    Display the statistics that define each scaler.

    The learned parameters are part of the preprocessing model and must be
    persisted alongside the trained machine-learning model in production.
    """
    print("\nLearned scaling parameters")

    for name, scaler in [
        ("Standardization", StandardScaler()),
        ("Min-max normalization", MinMaxScaler()),
        ("Robust scaling", RobustScaler()),
        ("Max-absolute scaling", MaxAbsScaler()),
    ]:
        scaler.fit(data)
        print(f"\n{name}")
        print_scaler_parameters(scaler)


# ---------------------------------------------------------------------------
# Main executable study
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("FEATURE SCALING: STANDARDIZATION, NORMALIZATION, ROBUST SCALING")
    print("AND TRANSFORMATIONS")
    print("=" * 78)

    customer_data = build_customer_measurements()

    print(
        "\nThe central preprocessing rule demonstrated here is: "
        "fit preprocessing parameters on training data, then reuse those "
        "parameters unchanged for validation, test, and production inference."
    )

    demonstrate_scalers(customer_data)
    demonstrate_parameter_interpretation(customer_data)
    demonstrate_transformations()
    demonstrate_outlier_effect()
    demonstrate_constant_feature()
    compare_distance_effects(customer_data)
    demonstrate_train_test_discipline()
    demonstrate_pipeline()
    demonstrate_inverse_transform()
    demonstrate_gradient_descent()
    demonstrate_failure_conditions()

    print("\nPractical interpretation")
    print(
        "- Standardization is useful when distance, variance, or optimization "
        "behavior benefits from centered features with comparable spread."
    )
    print(
        "- Min-max normalization is useful when a bounded training-domain "
        "representation is desirable, but unseen values can leave the range."
    )
    print(
        "- Robust scaling is useful when median and IQR better represent the "
        "feature because isolated extreme observations are present."
    )
    print(
        "- Transformations such as logarithms change distribution shape, while "
        "scalers primarily change location and scale."
    )
    print(
        "- Transformations and scaling can be combined, but their order should "
        "be deliberate and identical during training and inference."
    )

    print("\nExecution completed successfully.")


if __name__ == "__main__":
    main()
