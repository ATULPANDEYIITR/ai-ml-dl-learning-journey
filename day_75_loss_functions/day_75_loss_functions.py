#!/usr/bin/env python3
"""
Loss Functions: MSE, MAE, Binary Cross-Entropy and Categorical Cross-Entropy.

A self-contained implementation covering:
- Numerical loss calculations
- Probability and shape validation
- Logit-based stable calculations
- Analytical gradients
- Batch and per-sample reductions
- Multiclass and binary classification
- Numerical edge cases
- Unit tests and practical examples

Run:
    python loss_functions.py
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


EPSILON = 1e-12


class LossError(ValueError):
    """Raised when loss inputs are invalid."""


class Reduction(Enum):
    NONE = "none"
    SUM = "sum"
    MEAN = "mean"


@dataclass(frozen=True)
class LossResult:
    name: str
    values: tuple[float, ...]
    reduction: Reduction
    result: float | tuple[float, ...]

    def __str__(self) -> str:
        return f"{self.name}: {self.result}"


def validate_equal_lengths(
    actual: Sequence,
    predicted: Sequence
) -> None:
    if len(actual) == 0:
        raise LossError("Input collections cannot be empty.")

    if len(actual) != len(predicted):
        raise LossError("Actual and predicted lengths must match.")


def reduce_values(
    name: str,
    values: Sequence[float],
    reduction: Reduction
) -> LossResult:
    values = tuple(float(v) for v in values)

    if not values:
        raise LossError("Cannot reduce an empty collection.")

    if not all(math.isfinite(v) for v in values):
        raise LossError("Loss values must be finite.")

    if reduction == Reduction.NONE:
        result = values
    elif reduction == Reduction.SUM:
        result = math.fsum(values)
    elif reduction == Reduction.MEAN:
        result = math.fsum(values) / len(values)
    else:
        raise LossError("Unsupported reduction.")

    return LossResult(name, values, reduction, result)


def validate_numeric_inputs(
    actual: Sequence[float],
    predicted: Sequence[float]
) -> None:
    validate_equal_lengths(actual, predicted)

    for value in list(actual) + list(predicted):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise LossError("All inputs must be numeric.")

        if not math.isfinite(value):
            raise LossError("Inputs must be finite.")


def mean_squared_error(
    actual: Sequence[float],
    predicted: Sequence[float],
    reduction: Reduction = Reduction.MEAN
) -> LossResult:
    """MSE penalizes squared prediction errors."""

    validate_numeric_inputs(actual, predicted)

    values = [
        (float(y) - float(p)) ** 2
        for y, p in zip(actual, predicted)
    ]

    return reduce_values("MSE", values, reduction)


def mean_absolute_error(
    actual: Sequence[float],
    predicted: Sequence[float],
    reduction: Reduction = Reduction.MEAN
) -> LossResult:
    """MAE measures absolute prediction errors."""

    validate_numeric_inputs(actual, predicted)

    values = [
        abs(float(y) - float(p))
        for y, p in zip(actual, predicted)
    ]

    return reduce_values("MAE", values, reduction)


def _validate_binary_labels(labels: Sequence[int]) -> None:
    if not labels:
        raise LossError("Labels cannot be empty.")

    if any(label not in (0, 1) or isinstance(label, bool)
           for label in labels):
        raise LossError("Binary labels must be integers 0 or 1.")


def _validate_probabilities(probabilities: Sequence[float]) -> None:
    for probability in probabilities:
        if not math.isfinite(probability):
            raise LossError("Probabilities must be finite.")

        if not 0.0 <= probability <= 1.0:
            raise LossError("Probabilities must be between 0 and 1.")


def binary_cross_entropy(
    actual: Sequence[int],
    predicted: Sequence[float],
    reduction: Reduction = Reduction.MEAN,
    epsilon: float = EPSILON
) -> LossResult:
    """
    Binary cross-entropy.

    L = -[y log(p) + (1-y) log(1-p)]

    Probabilities are clipped to avoid log(0).
    """

    validate_equal_lengths(actual, predicted)
    _validate_binary_labels(actual)
    _validate_probabilities(predicted)

    if not 0 < epsilon < 0.5:
        raise LossError("Epsilon must be between 0 and 0.5.")

    values = []

    for label, probability in zip(actual, predicted):
        p = min(max(float(probability), epsilon), 1.0 - epsilon)

        loss = -(
            label * math.log(p)
            + (1 - label) * math.log1p(-p)
        )

        values.append(loss)

    return reduce_values("Binary Cross-Entropy", values, reduction)


def sigmoid(value: float) -> float:
    """Numerically stable sigmoid."""

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))

    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def softplus(value: float) -> float:
    """Stable log(1 + exp(value))."""

    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def binary_cross_entropy_with_logits(
    actual: Sequence[int],
    logits: Sequence[float],
    reduction: Reduction = Reduction.MEAN
) -> LossResult:
    """
    Stable binary cross-entropy directly from logits.

    L = max(z, 0) - z*y + log(1 + exp(-abs(z)))
    """

    validate_equal_lengths(actual, logits)
    _validate_binary_labels(actual)

    if not all(math.isfinite(z) for z in logits):
        raise LossError("Logits must be finite.")

    values = [
        max(z, 0.0) - z * y + softplus(-abs(z))
        for y, z in zip(actual, logits)
    ]

    return reduce_values(
        "Binary Cross-Entropy with Logits",
        values,
        reduction
    )


def validate_multiclass_inputs(
    actual: Sequence[int],
    probabilities: Sequence[Sequence[float]]
) -> int:
    if len(actual) == 0 or len(actual) != len(probabilities):
        raise LossError("Invalid batch dimensions.")

    classes = len(probabilities[0])

    if classes < 2:
        raise LossError("At least two classes are required.")

    for label, row in zip(actual, probabilities):
        if not isinstance(label, int) or isinstance(label, bool):
            raise LossError("Class labels must be integers.")

        if not 0 <= label < classes:
            raise LossError("Class label is outside the valid range.")

        if len(row) != classes:
            raise LossError("All probability rows must have equal length.")

        if not all(math.isfinite(p) and 0 <= p <= 1 for p in row):
            raise LossError("Invalid class probability.")

        if not math.isclose(math.fsum(row), 1.0, abs_tol=1e-8):
            raise LossError("Class probabilities must sum to one.")

    return classes


def categorical_cross_entropy(
    actual: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    reduction: Reduction = Reduction.MEAN,
    epsilon: float = EPSILON
) -> LossResult:
    """
    Categorical cross-entropy using integer class labels.

    L = -log(p_correct_class)
    """

    classes = validate_multiclass_inputs(actual, probabilities)

    if not 0 < epsilon < 0.5:
        raise LossError("Invalid epsilon.")

    values = []

    for label, row in zip(actual, probabilities):
        probability = max(float(row[label]), epsilon)
        values.append(-math.log(probability))

    return reduce_values("Categorical Cross-Entropy", values, reduction)


def softmax(logits: Sequence[float]) -> list[float]:
    """Stable softmax using maximum subtraction."""

    if len(logits) < 2:
        raise LossError("At least two logits are required.")

    if not all(math.isfinite(z) for z in logits):
        raise LossError("Logits must be finite.")

    maximum = max(logits)
    exponentials = [math.exp(z - maximum) for z in logits]
    denominator = math.fsum(exponentials)

    return [value / denominator for value in exponentials]


def categorical_cross_entropy_with_logits(
    actual: Sequence[int],
    logits: Sequence[Sequence[float]],
    reduction: Reduction = Reduction.MEAN
) -> LossResult:
    """
    Numerically stable categorical cross-entropy.

    L = log(sum(exp(z_j))) - z_correct

    Log-sum-exp is calculated using maximum subtraction.
    """

    if not actual or len(actual) != len(logits):
        raise LossError("Invalid batch dimensions.")

    classes = len(logits[0])

    if classes < 2:
        raise LossError("At least two classes are required.")

    values = []

    for label, row in zip(actual, logits):
        if not isinstance(label, int) or isinstance(label, bool):
            raise LossError("Labels must be integers.")

        if not 0 <= label < classes:
            raise LossError("Invalid class label.")

        if len(row) != classes:
            raise LossError("Inconsistent logit dimensions.")

        if not all(math.isfinite(z) for z in row):
            raise LossError("Logits must be finite.")

        maximum = max(row)

        log_sum_exp = maximum + math.log(
            math.fsum(math.exp(z - maximum) for z in row)
        )

        values.append(log_sum_exp - row[label])

    return reduce_values(
        "Categorical Cross-Entropy with Logits",
        values,
        reduction
    )


def mse_gradient(
    actual: Sequence[float],
    predicted: Sequence[float]
) -> list[float]:
    """Gradient of mean MSE with respect to predictions."""

    validate_numeric_inputs(actual, predicted)
    n = len(actual)

    return [
        2.0 * (float(p) - float(y)) / n
        for y, p in zip(actual, predicted)
    ]


def mae_subgradient(
    actual: Sequence[float],
    predicted: Sequence[float]
) -> list[float]:
    """
    One valid subgradient of mean MAE.

    At zero error, the selected subgradient is zero.
    """

    validate_numeric_inputs(actual, predicted)
    n = len(actual)

    return [
        (1.0 if p > y else -1.0 if p < y else 0.0) / n
        for y, p in zip(actual, predicted)
    ]


def binary_logits_gradient(
    actual: Sequence[int],
    logits: Sequence[float]
) -> list[float]:
    """Gradient of mean binary cross-entropy with respect to logits."""

    validate_equal_lengths(actual, logits)
    _validate_binary_labels(actual)

    if not all(math.isfinite(z) for z in logits):
        raise LossError("Logits must be finite.")

    n = len(actual)

    return [
        (sigmoid(z) - y) / n
        for y, z in zip(actual, logits)
    ]


def categorical_logits_gradient(
    actual: Sequence[int],
    logits: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Gradient of mean categorical cross-entropy with respect to logits."""

    if not actual or len(actual) != len(logits):
        raise LossError("Invalid batch dimensions.")

    classes = len(logits[0])
    gradients = []

    for label, row in zip(actual, logits):
        if not isinstance(label, int) or isinstance(label, bool):
            raise LossError("Labels must be integers.")

        if not 0 <= label < classes:
            raise LossError("Invalid class label.")

        probabilities = softmax(row)

        gradients.append([
            (p - (1.0 if index == label else 0.0)) / len(actual)
            for index, p in enumerate(probabilities)
        ])

    return gradients


class LossFunctionTests(unittest.TestCase):

    def test_mse(self):
        result = mean_squared_error([3, -0.5, 2], [2.5, 0, 2])
        self.assertAlmostEqual(result.result, 0.25)

    def test_mae(self):
        result = mean_absolute_error([3, -0.5, 2], [2.5, 0, 2])
        self.assertAlmostEqual(result.result, 1 / 3)

    def test_binary_cross_entropy(self):
        result = binary_cross_entropy([1, 0], [0.9, 0.1])
        self.assertAlmostEqual(result.result, -math.log(0.9))

    def test_binary_logits_extreme_values(self):
        result = binary_cross_entropy_with_logits(
            [1, 0],
            [1000, -1000]
        )
        self.assertTrue(math.isfinite(result.result))
        self.assertLess(result.result, 1e-10)

    def test_categorical_cross_entropy(self):
        result = categorical_cross_entropy(
            [0, 2],
            [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]
        )
        self.assertAlmostEqual(result.result, -math.log(0.7))

    def test_categorical_logits(self):
        result = categorical_cross_entropy_with_logits(
            [0],
            [[3.0, 1.0, 0.2]]
        )
        self.assertGreater(result.result, 0)

    def test_invalid_probability(self):
        with self.assertRaises(LossError):
            binary_cross_entropy([1], [1.5])

    def test_invalid_label(self):
        with self.assertRaises(LossError):
            binary_cross_entropy([2], [0.5])

    def test_empty_input(self):
        with self.assertRaises(LossError):
            mean_squared_error([], [])

    def test_mismatched_input(self):
        with self.assertRaises(LossError):
            mean_absolute_error([1, 2], [1])

    def test_gradient(self):
        gradient = mse_gradient([1, 2], [2, 4])
        self.assertEqual(gradient, [1.0, 2.0])


def demonstrate() -> None:
    print("\nLOSS FUNCTION BENCHMARK")
    print("=" * 55)

    actual = [3.0, -0.5, 2.0, 7.0]
    predicted = [2.5, 0.0, 2.0, 8.0]

    print(mean_squared_error(actual, predicted))
    print(mean_absolute_error(actual, predicted))

    binary_actual = [1, 0, 1, 0]
    binary_predictions = [0.95, 0.10, 0.80, 0.20]

    print(binary_cross_entropy(
        binary_actual,
        binary_predictions
    ))

    print(binary_cross_entropy_with_logits(
        binary_actual,
        [3.0, -2.2, 1.4, -1.4]
    ))

    multiclass_actual = [0, 2, 1]

    multiclass_predictions = [
        [0.70, 0.20, 0.10],
        [0.10, 0.20, 0.70],
        [0.15, 0.75, 0.10]
    ]

    print(categorical_cross_entropy(
        multiclass_actual,
        multiclass_predictions
    ))

    multiclass_logits = [
        [3.0, 1.0, 0.2],
        [0.1, 0.3, 2.5],
        [0.2, 2.8, 0.1]
    ]

    print(categorical_cross_entropy_with_logits(
        multiclass_actual,
        multiclass_logits
    ))

    print("\nMSE gradient:", mse_gradient(actual, predicted))
    print("MAE subgradient:", mae_subgradient(actual, predicted))

    print(
        "Binary logit gradient:",
        binary_logits_gradient(binary_actual, [3, -2.2, 1.4, -1.4])
    )

    print(
        "Categorical logit gradient:",
        categorical_logits_gradient(multiclass_actual, multiclass_logits)
    )


if __name__ == "__main__":
    demonstrate()

    print("\nRUNNING VALIDATION TESTS")
    unittest.main(argv=["loss_functions.py"], exit=False)
