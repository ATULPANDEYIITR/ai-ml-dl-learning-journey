#!/usr/bin/env python3
"""
ROC & Precision-Recall: ROC curve, AUC, PR curve, and threshold optimization.

This executable module builds the concepts from binary classification scores
through ROC/PR metrics, threshold selection, validation, calibration-aware
reasoning, bootstrap confidence intervals, and a cost-sensitive threshold
optimizer.

Only the Python standard library is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random
from statistics import mean, median
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Observation:
    actual: int
    score: float


@dataclass(frozen=True)
class ConfusionMatrix:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def predicted_positive(self) -> int:
        return self.tp + self.fp

    @property
    def actual_positive(self) -> int:
        return self.tp + self.fn

    @property
    def actual_negative(self) -> int:
        return self.tn + self.fp

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def specificity(self) -> float:
        denominator = self.tn + self.fp
        return self.tn / denominator if denominator else 0.0

    @property
    def false_positive_rate(self) -> float:
        return 1.0 - self.specificity

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2.0 * p * r / (p + r) if p + r else 0.0


def validate_dataset(observations: Sequence[Observation]) -> None:
    if not observations:
        raise ValueError("The dataset cannot be empty.")

    labels = {row.actual for row in observations}
    if labels - {0, 1}:
        raise ValueError("Labels must be binary 0/1.")
    if labels != {0, 1}:
        raise ValueError("ROC and PR curves require both positive and negative classes.")

    for row in observations:
        if not 0.0 <= row.score <= 1.0:
            raise ValueError("Scores must be probabilities/scores in [0, 1].")


def make_observations(
    actual: Sequence[int], scores: Sequence[float]
) -> list[Observation]:
    if len(actual) != len(scores):
        raise ValueError("Labels and scores must have equal length.")
    observations = [Observation(int(y), float(s)) for y, s in zip(actual, scores)]
    validate_dataset(observations)
    return observations


def confusion_matrix(
    observations: Sequence[Observation], threshold: float
) -> ConfusionMatrix:
    validate_dataset(observations)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be in [0, 1].")

    tp = fp = tn = fn = 0
    for row in observations:
        predicted = row.score >= threshold
        if row.actual == 1 and predicted:
            tp += 1
        elif row.actual == 0 and predicted:
            fp += 1
        elif row.actual == 0 and not predicted:
            tn += 1
        else:
            fn += 1

    return ConfusionMatrix(threshold, tp, fp, tn, fn)


def candidate_thresholds(observations: Sequence[Observation]) -> list[float]:
    """
    Include 0, 1, and every distinct score. These thresholds capture every
    possible prediction partition for a score-based classifier.
    """
    validate_dataset(observations)
    scores = sorted({row.score for row in observations}, reverse=True)
    return [1.0] + scores + [0.0]


def roc_curve(
    observations: Sequence[Observation],
) -> list[tuple[float, float, float]]:
    """
    Returns (threshold, FPR, TPR) points.

    Thresholds are evaluated from high to low. Lowering a threshold can only
    add observations to the predicted-positive set.
    """
    points = []
    for threshold in candidate_thresholds(observations):
        cm = confusion_matrix(observations, threshold)
        points.append((cm.false_positive_rate, cm.recall, threshold))

    # ROC integration requires points ordered by increasing FPR.
    return sorted(points, key=lambda point: (point[0], point[1]))


def trapezoidal_auc(points: Sequence[tuple[float, float]]) -> float:
    """Integrate y with respect to x using the trapezoidal rule."""
    if len(points) < 2:
        return 0.0

    ordered = sorted(points)
    area = 0.0
    for (x1, y1), (x2, y2) in zip(ordered, ordered[1:]):
        area += (x2 - x1) * (y1 + y2) / 2.0
    return area


def roc_auc(observations: Sequence[Observation]) -> float:
    points = [(fpr, tpr) for fpr, tpr, _ in roc_curve(observations)]
    return trapezoidal_auc(points)


def precision_recall_curve(
    observations: Sequence[Observation],
) -> list[tuple[float, float, float]]:
    """
    Returns (recall, precision, threshold).

    The final threshold of 0 predicts every sample positive. It is useful for
    displaying the complete operating range even though it is not paired with
    a conventional thresholded prediction in many libraries.
    """
    points = []
    for threshold in candidate_thresholds(observations):
        cm = confusion_matrix(observations, threshold)
        points.append((cm.recall, cm.precision, threshold))
    return sorted(points, key=lambda point: (point[0], point[2]))


def average_precision(observations: Sequence[Observation]) -> float:
    """
    Step-wise average precision.

    Each newly reached recall level contributes precision * recall increase.
    This differs from simply applying trapezoidal integration to a PR curve.
    """
    validate_dataset(observations)

    ranked = sorted(observations, key=lambda row: row.score, reverse=True)
    positives = sum(row.actual for row in ranked)
    if positives == 0:
        return 0.0

    tp = 0
    fp = 0
    previous_recall = 0.0
    score = 0.0

    for row in ranked:
        if row.actual == 1:
            tp += 1
        else:
            fp += 1

        recall = tp / positives
        precision = tp / (tp + fp)
        score += precision * (recall - previous_recall)
        previous_recall = recall

    return score


def pr_auc_trapezoid(observations: Sequence[Observation]) -> float:
    points = [(recall, precision) for recall, precision, _ in precision_recall_curve(observations)]
    return trapezoidal_auc(points)


def threshold_report(
    observations: Sequence[Observation], thresholds: Iterable[float]
) -> list[dict[str, float | int]]:
    report = []
    for threshold in thresholds:
        cm = confusion_matrix(observations, threshold)
        report.append(
            {
                "threshold": threshold,
                "tp": cm.tp,
                "fp": cm.fp,
                "tn": cm.tn,
                "fn": cm.fn,
                "precision": cm.precision,
                "recall": cm.recall,
                "specificity": cm.specificity,
                "fpr": cm.false_positive_rate,
                "f1": cm.f1,
                "accuracy": cm.accuracy,
            }
        )
    return report


def optimize_threshold(
    observations: Sequence[Observation],
    objective: Callable[[ConfusionMatrix], float],
) -> tuple[float, float, ConfusionMatrix]:
    """
    Search all score-induced operating points.

    Exhaustive threshold search is exact for this finite validation set because
    predictions change only when the threshold crosses an observed score.
    """
    best_threshold = None
    best_value = float("-inf")
    best_cm = None

    for threshold in candidate_thresholds(observations):
        cm = confusion_matrix(observations, threshold)
        value = objective(cm)

        if value > best_value:
            best_value = value
            best_threshold = threshold
            best_cm = cm

    assert best_threshold is not None
    assert best_cm is not None
    return best_threshold, best_value, best_cm


def f_beta(cm: ConfusionMatrix, beta: float = 1.0) -> float:
    if beta <= 0:
        raise ValueError("beta must be positive.")
    beta_squared = beta * beta
    p = cm.precision
    r = cm.recall
    denominator = beta_squared * p + r
    return (1 + beta_squared) * p * r / denominator if denominator else 0.0


@dataclass(frozen=True)
class CostModel:
    false_positive_cost: float
    false_negative_cost: float
    true_positive_value: float = 0.0
    true_negative_value: float = 0.0

    def validate(self) -> None:
        values = (
            self.false_positive_cost,
            self.false_negative_cost,
            self.true_positive_value,
            self.true_negative_value,
        )
        if any(value < 0 for value in values):
            raise ValueError("Costs and values cannot be negative.")

    def utility(self, cm: ConfusionMatrix) -> float:
        self.validate()
        return (
            cm.tp * self.true_positive_value
            + cm.tn * self.true_negative_value
            - cm.fp * self.false_positive_cost
            - cm.fn * self.false_negative_cost
        )


def bootstrap_auc(
    observations: Sequence[Observation],
    metric: Callable[[Sequence[Observation]], float],
    iterations: int = 500,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap percentile interval.

    A bootstrap sample preserves the number of rows but samples rows with
    replacement. Samples containing only one class cannot support ROC AUC, so
    those replicates are skipped.
    """
    validate_dataset(observations)
    if iterations < 50:
        raise ValueError("Use at least 50 bootstrap iterations.")

    rng = Random(seed)
    estimates = []

    for _ in range(iterations):
        sample = [observations[rng.randrange(len(observations))] for _ in observations]
        if {row.actual for row in sample} != {0, 1}:
            continue
        estimates.append(metric(sample))

    if not estimates:
        raise RuntimeError("No valid bootstrap replicates were produced.")

    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return mean(estimates), lower, upper


def generate_imbalanced_dataset() -> list[Observation]:
    """
    Fraud-like example: positives are rare, making PR analysis especially
    informative because the precision baseline is the positive prevalence.
    """
    negatives = [
        0.02, 0.04, 0.05, 0.07, 0.08, 0.10, 0.11, 0.13,
        0.15, 0.16, 0.18, 0.19, 0.20, 0.22, 0.24, 0.26,
        0.28, 0.31, 0.33, 0.35, 0.38, 0.40, 0.43, 0.46,
        0.49, 0.52, 0.55, 0.58, 0.61, 0.64,
    ]
    positives = [
        0.31, 0.42, 0.47, 0.54, 0.59, 0.63, 0.68, 0.72,
        0.75, 0.79,
    ]
    return make_observations([0] * len(negatives) + [1] * len(positives),
                             negatives + positives)


def demonstrate_basic_metrics(observations: Sequence[Observation]) -> None:
    print("\n=== Threshold-dependent classification metrics ===")
    for threshold in (0.30, 0.50, 0.70):
        cm = confusion_matrix(observations, threshold)
        print(
            f"threshold={threshold:.2f} "
            f"TP={cm.tp} FP={cm.fp} TN={cm.tn} FN={cm.fn} "
            f"precision={cm.precision:.3f} recall={cm.recall:.3f} "
            f"F1={cm.f1:.3f} FPR={cm.false_positive_rate:.3f}"
        )


def demonstrate_roc(observations: Sequence[Observation]) -> None:
    print("\n=== ROC curve ===")
    for fpr, tpr, threshold in roc_curve(observations):
        print(f"threshold={threshold:.2f} FPR={fpr:.3f} TPR={tpr:.3f}")
    print(f"ROC AUC={roc_auc(observations):.4f}")


def demonstrate_pr(observations: Sequence[Observation]) -> None:
    prevalence = sum(row.actual for row in observations) / len(observations)

    print("\n=== Precision-Recall curve ===")
    print(f"Positive prevalence / random precision baseline={prevalence:.3f}")

    for recall, precision, threshold in precision_recall_curve(observations):
        print(
            f"threshold={threshold:.2f} "
            f"recall={recall:.3f} precision={precision:.3f}"
        )

    print(f"Average Precision={average_precision(observations):.4f}")
    print(f"Trapezoidal PR AUC={pr_auc_trapezoid(observations):.4f}")


def demonstrate_threshold_optimization(observations: Sequence[Observation]) -> None:
    print("\n=== Threshold optimization ===")

    threshold, value, cm = optimize_threshold(
        observations,
        lambda matrix: matrix.f1,
    )
    print(
        f"Maximum F1: threshold={threshold:.2f}, "
        f"F1={value:.4f}, precision={cm.precision:.4f}, recall={cm.recall:.4f}"
    )

    threshold, value, cm = optimize_threshold(
        observations,
        lambda matrix: f_beta(matrix, beta=2.0),
    )
    print(
        f"Maximum F2: threshold={threshold:.2f}, "
        f"F2={value:.4f}, precision={cm.precision:.4f}, recall={cm.recall:.4f}"
    )

    threshold, value, cm = optimize_threshold(
        observations,
        lambda matrix: matrix.recall if matrix.false_positive_rate <= 0.10 else -1.0,
    )
    print(
        f"Maximum recall subject to FPR <= 0.10: "
        f"threshold={threshold:.2f}, recall={cm.recall:.4f}, "
        f"FPR={cm.false_positive_rate:.4f}"
    )

    cost_model = CostModel(
        false_positive_cost=1.0,
        false_negative_cost=5.0,
        true_positive_value=0.0,
        true_negative_value=0.0,
    )
    threshold, utility, cm = optimize_threshold(
        observations,
        cost_model.utility,
    )
    print(
        f"Cost-sensitive optimum: threshold={threshold:.2f}, "
        f"utility={utility:.2f}, FP={cm.fp}, FN={cm.fn}"
    )


def demonstrate_rank_equivalence() -> None:
    """
    ROC AUC is fundamentally rank-based. If score ordering is unchanged,
    strictly monotonic score transformations preserve the ideal ranking.
    """
    labels = [0, 0, 1, 1, 0, 1]
    scores = [0.10, 0.25, 0.60, 0.90, 0.40, 0.80]

    original = make_observations(labels, scores)
    transformed = make_observations(
        labels,
        [sqrt(score) for score in scores],
    )

    print("\n=== Ranking perspective ===")
    print(f"Original ROC AUC:   {roc_auc(original):.4f}")
    print(f"sqrt(score) ROC AUC: {roc_auc(transformed):.4f}")
    print(
        "AUC measures ranking quality, not whether a score of 0.8 is "
        "a well-calibrated 80% probability."
    )


def demonstrate_edge_cases() -> None:
    print("\n=== Validation and edge cases ===")

    try:
        make_observations([0, 1], [0.2, 1.4])
    except ValueError as exc:
        print(f"Invalid score rejected: {exc}")

    try:
        make_observations([1, 1], [0.2, 0.9])
    except ValueError as exc:
        print(f"Single-class dataset rejected: {exc}")

    single = make_observations([0, 1], [0.2, 0.8])
    cm = confusion_matrix(single, 0.5)
    print(
        f"Two-row dataset remains valid: precision={cm.precision:.3f}, "
        f"recall={cm.recall:.3f}"
    )


def demonstrate_bootstrap(observations: Sequence[Observation]) -> None:
    print("\n=== Bootstrap uncertainty for ROC AUC ===")
    estimate, lower, upper = bootstrap_auc(
        observations,
        roc_auc,
        iterations=300,
        seed=7,
    )
    print(
        f"bootstrap mean AUC={estimate:.4f}; "
        f"approximate 95% percentile interval=[{lower:.4f}, {upper:.4f}]"
    )


def print_threshold_table(observations: Sequence[Observation]) -> None:
    print("\n=== Operating-point table ===")
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    print("threshold  precision  recall  specificity  FPR   F1")
    print("-" * 58)

    for row in threshold_report(observations, thresholds):
        print(
            f"{row['threshold']:9.2f} "
            f"{row['precision']:10.3f} "
            f"{row['recall']:7.3f} "
            f"{row['specificity']:11.3f} "
            f"{row['fpr']:5.3f} "
            f"{row['f1']:5.3f}"
        )


def main() -> None:
    observations = generate_imbalanced_dataset()

    print("ROC & Precision-Recall Analysis")
    print("===============================")
    print(f"Observations: {len(observations)}")
    print(f"Positive cases: {sum(row.actual for row in observations)}")
    print(f"Negative cases: {sum(row.actual == 0 for row in observations)}")

    demonstrate_basic_metrics(observations)
    print_threshold_table(observations)
    demonstrate_roc(observations)
    demonstrate_pr(observations)
    demonstrate_threshold_optimization(observations)
    demonstrate_rank_equivalence()
    demonstrate_bootstrap(observations)
    demonstrate_edge_cases()

    print("\n=== Interpretation ===")
    print(
        "ROC AUC summarizes discrimination across thresholds. "
        "The PR view emphasizes performance among predicted positives and "
        "is especially informative when the positive class is rare."
    )
    print(
        "The selected threshold is an operating-policy decision. "
        "It should reflect error costs, capacity, risk tolerance, and "
        "validation data rather than being chosen solely from AUC."
    )


if __name__ == "__main__":
    main()
