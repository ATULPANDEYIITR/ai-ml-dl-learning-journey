#!/usr/bin/env python3
"""
Classification Metrics: accuracy, precision, recall, F1, specificity,
and confusion matrices.

Self-contained educational implementation using only the Python standard
library. The examples use binary and multiclass classification and show
how metrics behave under class imbalance, threshold changes, and
different averaging strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BinaryConfusionMatrix:
    """Counts for a binary classifier."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def __post_init__(self) -> None:
        values = (
            self.true_positive,
            self.false_positive,
            self.true_negative,
            self.false_negative,
        )
        if any(value < 0 for value in values):
            raise ValueError("Confusion-matrix counts cannot be negative.")

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )


def validate_binary_labels(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths.")
    if not y_true:
        raise ValueError("At least one observation is required.")

    valid = {0, 1}
    if any(value not in valid for value in y_true):
        raise ValueError("Binary y_true values must be 0 or 1.")
    if any(value not in valid for value in y_pred):
        raise ValueError("Binary y_pred values must be 0 or 1.")


def confusion_matrix_binary(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> BinaryConfusionMatrix:
    """Construct TP, FP, TN, and FN from binary labels."""
    validate_binary_labels(y_true, y_pred)

    tp = fp = tn = fn = 0

    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        else:
            fn += 1

    return BinaryConfusionMatrix(tp, fp, tn, fn)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return zero for an undefined metric instead of raising ZeroDivisionError."""
    return numerator / denominator if denominator else 0.0


def accuracy(cm: BinaryConfusionMatrix) -> float:
    """Fraction of all predictions that are correct."""
    return safe_divide(cm.true_positive + cm.true_negative, cm.total)


def precision(cm: BinaryConfusionMatrix) -> float:
    """Among predicted positives, fraction that are actually positive."""
    return safe_divide(
        cm.true_positive,
        cm.true_positive + cm.false_positive,
    )


def recall(cm: BinaryConfusionMatrix) -> float:
    """Sensitivity/true-positive rate: fraction of positives detected."""
    return safe_divide(
        cm.true_positive,
        cm.true_positive + cm.false_negative,
    )


def specificity(cm: BinaryConfusionMatrix) -> float:
    """True-negative rate: fraction of negatives correctly rejected."""
    return safe_divide(
        cm.true_negative,
        cm.true_negative + cm.false_positive,
    )


def f1_score(cm: BinaryConfusionMatrix) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(cm)
    r = recall(cm)
    return safe_divide(2 * p * r, p + r)


def metric_report(cm: BinaryConfusionMatrix) -> dict[str, float]:
    return {
        "accuracy": accuracy(cm),
        "precision": precision(cm),
        "recall": recall(cm),
        "specificity": specificity(cm),
        "f1": f1_score(cm),
    }


def print_binary_report(
    title: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> None:
    cm = confusion_matrix_binary(y_true, y_pred)

    print(f"\n=== {title} ===")
    print("Confusion matrix:")
    print("                 Predicted 0   Predicted 1")
    print(f"Actual 0         {cm.true_negative:12d}   {cm.false_positive:12d}")
    print(f"Actual 1         {cm.false_negative:12d}   {cm.true_positive:12d}")

    for name, value in metric_report(cm).items():
        print(f"{name:12s}: {value:.4f}")


def explain_binary_metrics(cm: BinaryConfusionMatrix) -> None:
    print("\nMetric interpretation:")
    print(
        f"TP={cm.true_positive}, FP={cm.false_positive}, "
        f"TN={cm.true_negative}, FN={cm.false_negative}"
    )
    print(
        "Precision answers: when the model predicts positive, "
        "how often is it correct?"
    )
    print(
        "Recall answers: among actual positives, how many did the model find?"
    )
    print(
        "Specificity answers: among actual negatives, how many did the model reject?"
    )
    print(
        "F1 balances precision and recall and is useful when both false "
        "positives and false negatives matter."
    )


def classification_metrics_from_labels(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    cm = confusion_matrix_binary(y_true, y_pred)
    return metric_report(cm)


def binary_metrics_from_scores(
    y_true: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Convert positive-class scores into predictions using a threshold.

    A higher threshold generally makes positive predictions harder to obtain,
    which can reduce false positives while increasing false negatives.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have equal lengths.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    y_pred = [1 if score >= threshold else 0 for score in scores]
    return classification_metrics_from_labels(y_true, y_pred)


def print_threshold_comparison(
    y_true: Sequence[int],
    scores: Sequence[float],
    thresholds: Iterable[float],
) -> None:
    print("\n=== Threshold comparison ===")
    print("Threshold   Accuracy   Precision   Recall   Specificity   F1")

    for threshold in thresholds:
        metrics = binary_metrics_from_scores(y_true, scores, threshold)
        print(
            f"{threshold:9.2f}   "
            f"{metrics['accuracy']:8.3f}   "
            f"{metrics['precision']:9.3f}   "
            f"{metrics['recall']:6.3f}   "
            f"{metrics['specificity']:11.3f}   "
            f"{metrics['f1']:5.3f}"
        )


def multiclass_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> list[list[int]]:
    """
    Build a multiclass confusion matrix.

    Rows represent actual classes and columns represent predicted classes.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths.")
    if not y_true:
        raise ValueError("At least one observation is required.")
    if len(set(labels)) != len(labels):
        raise ValueError("Class labels must be unique.")

    index = {label: position for position, label in enumerate(labels)}

    for value in list(y_true) + list(y_pred):
        if value not in index:
            raise ValueError(f"Unknown class label: {value!r}")

    matrix = [[0 for _ in labels] for _ in labels]

    for actual, predicted in zip(y_true, y_pred):
        matrix[index[actual]][index[predicted]] += 1

    return matrix


def print_multiclass_matrix(
    matrix: list[list[int]], labels: Sequence[str]
) -> None:
    width = max(
        8,
        max(len(label) for label in labels) + 2,
        max(len(str(value)) for row in matrix for value in row) + 2,
    )

    print("\nMulticlass confusion matrix")
    print(" " * width + "".join(label.rjust(width) for label in labels))

    for label, row in zip(labels, matrix):
        print(label.rjust(width) + "".join(str(value).rjust(width) for value in row))


def multiclass_one_vs_rest_counts(
    matrix: list[list[int]],
    class_index: int,
) -> BinaryConfusionMatrix:
    """Convert one multiclass class into a one-vs-rest binary problem."""
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("Confusion matrix must be square.")
    if not 0 <= class_index < size:
        raise IndexError("Invalid class index.")

    tp = matrix[class_index][class_index]
    fn = sum(matrix[class_index][j] for j in range(size) if j != class_index)
    fp = sum(matrix[i][class_index] for i in range(size) if i != class_index)
    tn = sum(
        matrix[i][j]
        for i in range(size)
        for j in range(size)
        if i != class_index and j != class_index
    )

    return BinaryConfusionMatrix(tp, fp, tn, fn)


def macro_f1(
    matrix: list[list[int]],
) -> float:
    scores = [
        f1_score(multiclass_one_vs_rest_counts(matrix, i))
        for i in range(len(matrix))
    ]
    return sum(scores) / len(scores) if scores else 0.0


def weighted_f1(
    matrix: list[list[int]],
) -> float:
    supports = [sum(row) for row in matrix]
    total = sum(supports)

    if total == 0:
        return 0.0

    return sum(
        f1_score(multiclass_one_vs_rest_counts(matrix, i)) * supports[i]
        for i in range(len(matrix))
    ) / total


def multiclass_accuracy(matrix: list[list[int]]) -> float:
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    return safe_divide(correct, total)


def demonstrate_imbalanced_data() -> None:
    """
    Accuracy can look excellent when the negative class dominates.

    A classifier that almost always predicts the majority class may therefore
    have high accuracy while having poor recall for the minority class.
    """
    y_true = [0] * 95 + [1] * 5
    y_pred = [0] * 99 + [1]

    cm = confusion_matrix_binary(y_true, y_pred)

    print("\n=== Class imbalance example ===")
    for name, value in metric_report(cm).items():
        print(f"{name:12s}: {value:.4f}")

    print(
        "\nThe high accuracy is misleading because only one of five "
        "positive cases was detected."
    )


def validate_metric_relationships() -> None:
    """
    Small executable checks help catch implementation errors.

    The checks use exact values where the underlying fractions are simple.
    """
    cm = BinaryConfusionMatrix(
        true_positive=40,
        false_positive=10,
        true_negative=45,
        false_negative=5,
    )

    assert isclose(accuracy(cm), 0.85)
    assert isclose(precision(cm), 40 / 50)
    assert isclose(recall(cm), 40 / 45)
    assert isclose(specificity(cm), 45 / 55)
    assert isclose(f1_score(cm), 2 * (40 / 50) * (40 / 45) / ((40 / 50) + (40 / 45)))

    try:
        confusion_matrix_binary([1, 0], [1])
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched lengths should fail validation.")

    try:
        binary_metrics_from_scores([1], [1.2], threshold=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Scores are allowed to be represented as values, "
                             "but invalid score range should not be silently accepted.")


def validate_probability_scores(
    scores: Sequence[float],
) -> None:
    if not scores:
        raise ValueError("At least one score is required.")
    if any(score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("Probability scores must be between 0 and 1.")


def demonstrate_probability_validation() -> None:
    """Probability validation is useful when the score is explicitly a probability."""
    probabilities = [0.08, 0.17, 0.51, 0.72, 0.93]
    validate_probability_scores(probabilities)

    print("\n=== Probability validation ===")
    print("Valid probabilities:", probabilities)

    try:
        validate_probability_scores([0.2, 1.4])
    except ValueError as exc:
        print("Rejected invalid probabilities:", exc)


def demonstrate_multiclass_metrics() -> None:
    actual = [
        "spam", "spam", "spam", "ham", "ham",
        "ham", "promotion", "promotion", "promotion", "ham",
    ]
    predicted = [
        "spam", "spam", "ham", "ham", "promotion",
        "ham", "promotion", "ham", "promotion", "spam",
    ]
    labels = ["spam", "ham", "promotion"]

    matrix = multiclass_confusion_matrix(actual, predicted, labels)
    print_multiclass_matrix(matrix, labels)

    print(f"\nMulticlass accuracy: {multiclass_accuracy(matrix):.4f}")
    print(f"Macro F1:             {macro_f1(matrix):.4f}")
    print(f"Weighted F1:          {weighted_f1(matrix):.4f}")

    print("\nOne-vs-rest class metrics:")
    for index, label in enumerate(labels):
        cm = multiclass_one_vs_rest_counts(matrix, index)
        print(
            f"{label:10s} "
            f"precision={precision(cm):.3f} "
            f"recall={recall(cm):.3f} "
            f"specificity={specificity(cm):.3f} "
            f"f1={f1_score(cm):.3f}"
        )


def demonstrate_review_triage() -> None:
    """
    A realistic classification use case: predicting whether a code change
    requires security review.

    Positive means "security review required". False negatives can be costly
    because a risky change may bypass specialist review.
    """
    actual = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    scores = [0.92, 0.12, 0.66, 0.31, 0.27, 0.58, 0.44, 0.81, 0.73, 0.18, 0.39, 0.08]

    print("\n=== Security-review triage ===")
    print_threshold_comparison(actual, scores, [0.30, 0.50, 0.70])

    selected_threshold = 0.50
    predictions = [int(score >= selected_threshold) for score in scores]
    cm = confusion_matrix_binary(actual, predictions)
    explain_binary_metrics(cm)

    print(
        "\nOperational interpretation: lowering the threshold can improve "
        "recall for risky changes, but it can increase the review workload "
        "through additional false positives."
    )


def main() -> None:
    print("CLASSIFICATION METRICS LAB")
    print("==========================")

    basic_actual = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    basic_predicted = [1, 1, 0, 0, 0, 1, 1, 0, 1, 0]

    print_binary_report(
        "Binary classification fundamentals",
        basic_actual,
        basic_predicted,
    )

    cm = confusion_matrix_binary(basic_actual, basic_predicted)
    explain_binary_metrics(cm)

    score_actual = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    score_predictions = [0.91, 0.15, 0.62, 0.48, 0.55, 0.29, 0.67, 0.77, 0.33, 0.42]

    print_threshold_comparison(
        score_actual,
        score_predictions,
        [0.20, 0.40, 0.50, 0.60, 0.80],
    )

    demonstrate_imbalanced_data()
    demonstrate_probability_validation()
    demonstrate_multiclass_metrics()
    demonstrate_review_triage()
    validate_metric_relationships()

    print("\nAll built-in validation checks passed.")


if __name__ == "__main__":
    main()
