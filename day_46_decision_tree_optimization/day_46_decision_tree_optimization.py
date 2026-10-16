#!/usr/bin/env python3
"""
Decision Tree Optimization: depth, pruning, overfitting, and feature importance.

A self-contained educational implementation using only the Python standard
library. It builds classification trees, evaluates generalization, implements
pre-pruning and cost-complexity post-pruning, computes impurity-based feature
importance, and performs stratified evaluation on a deterministic synthetic
dataset.

Run:
    python decision_tree_optimization.py
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence


@dataclass(frozen=True)
class Sample:
    features: tuple[float, ...]
    label: str


@dataclass
class Node:
    prediction: str
    class_counts: Counter[str]
    impurity: float
    sample_count: int
    depth: int
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


def validate_dataset(
    samples: Sequence[Sample],
    feature_count: Optional[int] = None,
) -> int:
    """Reject malformed datasets before training or evaluation."""
    if not samples:
        raise ValueError("The dataset must contain at least one sample.")

    inferred = len(samples[0].features)
    if inferred == 0:
        raise ValueError("Every sample must have at least one feature.")

    if feature_count is not None and inferred != feature_count:
        raise ValueError("Unexpected feature count.")

    for sample in samples:
        if len(sample.features) != inferred:
            raise ValueError("Samples have inconsistent feature counts.")
        if not sample.label:
            raise ValueError("Labels must not be empty.")
        if any(not math.isfinite(value) for value in sample.features):
            raise ValueError("Feature values must be finite.")

    return inferred


def gini_impurity(labels: Iterable[str]) -> float:
    """Gini impurity measures the probability of misclassification."""
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def entropy(labels: Iterable[str]) -> float:
    """Entropy measures class uncertainty in bits."""
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count
    )


def majority_label(labels: Iterable[str]) -> str:
    """Resolve ties deterministically to keep experiments reproducible."""
    counts = Counter(labels)
    if not counts:
        raise ValueError("Cannot select a majority label from empty data.")
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def split_samples(
    samples: Sequence[Sample],
    feature_index: int,
    threshold: float,
) -> tuple[list[Sample], list[Sample]]:
    left, right = [], []
    for sample in samples:
        if sample.features[feature_index] <= threshold:
            left.append(sample)
        else:
            right.append(sample)
    return left, right


def information_gain(
    samples: Sequence[Sample],
    feature_index: int,
    threshold: float,
    impurity_function=gini_impurity,
) -> float:
    """Compute parent impurity minus weighted child impurity."""
    parent_impurity = impurity_function(sample.label for sample in samples)
    left, right = split_samples(samples, feature_index, threshold)
    if not left or not right:
        return 0.0

    weighted_children = (
        len(left) / len(samples)
    ) * impurity_function(sample.label for sample in left) + (
        len(right) / len(samples)
    ) * impurity_function(sample.label for sample in right)

    return max(0.0, parent_impurity - weighted_children)


class DecisionTreeClassifier:
    """
    Numeric, binary classification tree.

    max_depth and min_samples_split provide pre-pruning.
    min_samples_leaf prevents splits producing tiny leaves.
    min_impurity_decrease rejects negligible improvements.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        criterion: str = "gini",
    ) -> None:
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease cannot be negative.")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.root: Optional[Node] = None
        self.n_features_: int = 0
        self.feature_importances_: list[float] = []

    def _impurity(self, labels: Iterable[str]) -> float:
        return gini_impurity(labels) if self.criterion == "gini" else entropy(labels)

    def fit(self, samples: Sequence[Sample]) -> "DecisionTreeClassifier":
        self.n_features_ = validate_dataset(samples)
        self.root = self._build(list(samples), depth=0)
        self._refresh_feature_importances()
        return self

    def _build(self, samples: list[Sample], depth: int) -> Node:
        labels = [sample.label for sample in samples]
        counts = Counter(labels)
        node = Node(
            prediction=majority_label(labels),
            class_counts=counts,
            impurity=self._impurity(labels),
            sample_count=len(samples),
            depth=depth,
        )

        # A pure node cannot improve by splitting.
        if node.impurity == 0.0:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(samples) < self.min_samples_split:
            return node
        if len(samples) < 2 * self.min_samples_leaf:
            return node

        best_gain = -1.0
        best_split: Optional[tuple[int, float, list[Sample], list[Sample]]] = None

        # Midpoints between sorted distinct values represent every useful
        # threshold for a numeric feature without scanning arbitrary values.
        for feature_index in range(self.n_features_):
            values = sorted({sample.features[feature_index] for sample in samples})
            for lower, upper in zip(values, values[1:]):
                threshold = lower + (upper - lower) / 2.0
                left, right = split_samples(samples, feature_index, threshold)

                if (
                    len(left) < self.min_samples_leaf
                    or len(right) < self.min_samples_leaf
                ):
                    continue

                gain = node.impurity - (
                    len(left) / len(samples)
                ) * self._impurity(sample.label for sample in left) - (
                    len(right) / len(samples)
                ) * self._impurity(sample.label for sample in right)

                # Strict comparison preserves the first split in deterministic
                # feature and threshold order when gains tie.
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold, left, right)

        if best_split is None or best_gain < self.min_impurity_decrease:
            return node
        if best_gain <= 0.0:
            return node

        feature_index, threshold, left_samples, right_samples = best_split
        node.feature_index = feature_index
        node.threshold = threshold
        node.gain = best_gain
        node.left = self._build(left_samples, depth + 1)
        node.right = self._build(right_samples, depth + 1)
        return node

    def _require_root(self) -> Node:
        if self.root is None:
            raise RuntimeError("Call fit() before prediction or inspection.")
        return self.root

    def predict_one(self, features: Sequence[float]) -> str:
        if len(features) != self.n_features_:
            raise ValueError("Prediction has an unexpected feature count.")
        if any(not math.isfinite(value) for value in features):
            raise ValueError("Prediction features must be finite.")

        node = self._require_root()
        while not node.is_leaf:
            assert node.feature_index is not None
            assert node.threshold is not None
            assert node.left is not None and node.right is not None
            node = (
                node.left
                if features[node.feature_index] <= node.threshold
                else node.right
            )
        return node.prediction

    def predict(self, rows: Sequence[Sequence[float]]) -> list[str]:
        return [self.predict_one(row) for row in rows]

    def _walk(self, node: Optional[Node] = None) -> Iterable[Node]:
        current = self._require_root() if node is None else node
        yield current
        if current.left is not None:
            yield from self._walk(current.left)
        if current.right is not None:
            yield from self._walk(current.right)

    def node_count(self) -> int:
        return sum(1 for _ in self._walk())

    def leaf_count(self) -> int:
        return sum(1 for node in self._walk() if node.is_leaf)

    def depth(self) -> int:
        return max(node.depth for node in self._walk())

    def _refresh_feature_importances(self) -> None:
        """Aggregate sample-weighted impurity reduction over internal nodes."""
        totals = [0.0] * self.n_features_
        root = self._require_root()

        for node in self._walk(root):
            if node.is_leaf:
                continue
            assert node.feature_index is not None
            assert node.left is not None and node.right is not None

            reduction = node.sample_count * node.gain
            totals[node.feature_index] += reduction

        denominator = sum(totals)
        self.feature_importances_ = (
            [value / denominator for value in totals]
            if denominator > 0
            else totals
        )

    def cost_complexity_prune(self, alpha: float) -> "DecisionTreeClassifier":
        """
        Apply weakest-link cost-complexity pruning.

        A subtree is replaced by its leaf when its impurity increase is no
        greater than alpha times the number of leaves removed. Impurity is
        weighted by the root training sample count, matching the normalized
        empirical-risk formulation for this implementation.
        """
        if alpha < 0 or not math.isfinite(alpha):
            raise ValueError("alpha must be finite and non-negative.")

        root = self._require_root()

        def subtree_risk(node: Node) -> float:
            if node.is_leaf:
                return node.sample_count * node.impurity
            assert node.left is not None and node.right is not None
            return subtree_risk(node.left) + subtree_risk(node.right)

        def leaves(node: Node) -> int:
            if node.is_leaf:
                return 1
            assert node.left is not None and node.right is not None
            return leaves(node.left) + leaves(node.right)

        def prune(node: Node) -> Node:
            if node.is_leaf:
                return node
            assert node.left is not None and node.right is not None

            node.left = prune(node.left)
            node.right = prune(node.right)

            subtree_leaf_count = leaves(node)
            if subtree_leaf_count <= 1:
                return node

            leaf_risk = node.sample_count * node.impurity
            branch_risk = subtree_risk(node)
            effective_alpha = (
                leaf_risk - branch_risk
            ) / (subtree_leaf_count - 1)

            if effective_alpha <= alpha:
                # The node's majority prediction and counts already describe
                # the replacement leaf, so discarding its children is enough.
                node.feature_index = None
                node.threshold = None
                node.left = None
                node.right = None
                node.gain = 0.0

            return node

        self.root = prune(root)
        self._refresh_depths()
        self._refresh_feature_importances()
        return self

    def _refresh_depths(self) -> None:
        def visit(node: Node, depth: int) -> None:
            node.depth = depth
            if node.left is not None:
                visit(node.left, depth + 1)
            if node.right is not None:
                visit(node.right, depth + 1)

        visit(self._require_root(), 0)

    def describe(self) -> str:
        """Return a readable representation of the trained decision rules."""
        lines: list[str] = []

        def visit(node: Node, indent: str) -> None:
            if node.is_leaf:
                lines.append(
                    f"{indent}Leaf: predict={node.prediction!r}, "
                    f"samples={node.sample_count}, impurity={node.impurity:.3f}"
                )
                return

            assert node.feature_index is not None
            assert node.threshold is not None
            assert node.left is not None and node.right is not None

            lines.append(
                f"{indent}if feature[{node.feature_index}] <= "
                f"{node.threshold:.3f} (gain={node.gain:.3f}):"
            )
            visit(node.left, indent + "  ")
            lines.append(f"{indent}else:")
            visit(node.right, indent + "  ")

        visit(self._require_root(), "")
        return "\n".join(lines)


def accuracy_score(
    actual: Sequence[str],
    predicted: Sequence[str],
) -> float:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Inputs must be non-empty and have equal lengths.")
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def stratified_train_test_split(
    samples: Sequence[Sample],
    test_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Preserve approximate class proportions in both partitions."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one.")

    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.label].append(sample)

    generator = random.Random(seed)
    training: list[Sample] = []
    testing: list[Sample] = []

    for label in sorted(groups):
        group = groups[label][:]
        generator.shuffle(group)

        # Keep at least one training example for each represented class.
        test_count = min(
            max(1, round(len(group) * test_fraction)),
            max(0, len(group) - 1),
        )
        testing.extend(group[:test_count])
        training.extend(group[test_count:])

    generator.shuffle(training)
    generator.shuffle(testing)

    if not training or not testing:
        raise ValueError("The split must produce non-empty partitions.")

    return training, testing


def make_maintenance_dataset(
    size: int = 240,
    seed: int = 19,
) -> tuple[list[Sample], list[str]]:
    """
    Create a synthetic equipment-maintenance dataset.

    Features:
        vibration_mm_s, operating_hours_thousands, temperature_c,
        previous_fault_count.

    Failure probability depends mainly on vibration, temperature, and history.
    Random noise represents unobserved operating conditions.
    """
    generator = random.Random(seed)
    samples: list[Sample] = []

    for _ in range(size):
        vibration = generator.uniform(0.2, 9.0)
        operating_hours = generator.uniform(0.1, 20.0)
        temperature = generator.uniform(30.0, 115.0)
        previous_faults = float(generator.randint(0, 5))

        risk = (
            0.95 * vibration
            + 0.025 * temperature
            + 0.8 * previous_faults
            + 0.035 * operating_hours
            + generator.gauss(0.0, 1.4)
        )
        label = "failure" if risk > 10.0 else "normal"

        samples.append(
            Sample(
                (
                    round(vibration, 3),
                    round(operating_hours, 3),
                    round(temperature, 3),
                    previous_faults,
                ),
                label,
            )
        )

    names = [
        "vibration_mm_s",
        "operating_hours_thousands",
        "temperature_c",
        "previous_fault_count",
    ]
    return samples, names


def print_model_report(
    title: str,
    model: DecisionTreeClassifier,
    training: Sequence[Sample],
    testing: Sequence[Sample],
    feature_names: Sequence[str],
) -> None:
    train_accuracy = accuracy_score(
        [sample.label for sample in training],
        model.predict([sample.features for sample in training]),
    )
    test_accuracy = accuracy_score(
        [sample.label for sample in testing],
        model.predict([sample.features for sample in testing]),
    )

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy:     {test_accuracy:.3f}")
    print(f"Tree depth:        {model.depth()}")
    print(f"Internal nodes:    {model.node_count() - model.leaf_count()}")
    print(f"Leaves:            {model.leaf_count()}")

    print("Impurity-based feature importance:")
    ranking = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda item: (-item[1], item[0]),
    )
    for feature, importance in ranking:
        print(f"  {feature:<30} {importance:.4f}")


def demonstrate_edge_cases() -> None:
    print("\nValidation and edge cases")
    print("-------------------------")

    try:
        DecisionTreeClassifier(max_depth=-1)
    except ValueError as error:
        print(f"Rejected invalid depth: {error}")

    try:
        validate_dataset(
            [
                Sample((1.0, 2.0), "normal"),
                Sample((1.0,), "failure"),
            ]
        )
    except ValueError as error:
        print(f"Rejected inconsistent features: {error}")

    constant_features = [
        Sample((4.0, 4.0), "normal"),
        Sample((4.0, 4.0), "failure"),
        Sample((4.0, 4.0), "normal"),
    ]
    constant_tree = DecisionTreeClassifier().fit(constant_features)
    print(
        "Constant-feature dataset:",
        f"depth={constant_tree.depth()}, "
        f"leaves={constant_tree.leaf_count()}, "
        f"importances={constant_tree.feature_importances_}",
    )

    try:
        constant_tree.predict_one((float("nan"), 4.0))
    except ValueError as error:
        print(f"Rejected non-finite prediction input: {error}")


def main() -> None:
    print("Decision Tree Optimization for Equipment Maintenance")
    print("=" * 54)

    dataset, feature_names = make_maintenance_dataset()
    training, testing = stratified_train_test_split(dataset)

    print(
        f"Samples: {len(dataset)} | Training: {len(training)} | "
        f"Testing: {len(testing)}"
    )
    print("Training labels:", dict(Counter(s.label for s in training)))
    print("Testing labels: ", dict(Counter(s.label for s in testing)))

    # An unrestricted tree illustrates why memorizing training data is risky.
    unrestricted = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        criterion="gini",
    ).fit(training)

    print_model_report(
        "Unrestricted tree",
        unrestricted,
        training,
        testing,
        feature_names,
    )

    # Pre-pruning constrains complexity while the tree is being constructed.
    controlled = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=8,
        min_samples_leaf=4,
        min_impurity_decrease=0.002,
        criterion="gini",
    ).fit(training)

    print_model_report(
        "Pre-pruned tree",
        controlled,
        training,
        testing,
        feature_names,
    )

    # Cost-complexity pruning reduces an already fitted tree. The alpha value
    # is illustrative; a real deployment should select it on validation data.
    post_pruned = DecisionTreeClassifier(
        min_samples_leaf=2,
        min_samples_split=4,
    ).fit(training)

    post_pruned.cost_complexity_prune(alpha=0.01)

    print_model_report(
        "Cost-complexity post-pruned tree",
        post_pruned,
        training,
        testing,
        feature_names,
    )

    print("\nReadable rules from the controlled tree")
    print("----------------------------------------")
    print(controlled.describe())

    print("\nExample prediction")
    print("------------------")
    example = (7.2, 12.0, 98.0, 3.0)
    print(dict(zip(feature_names, example)))
    print("Predicted state:", controlled.predict_one(example))

    print("\nOptimization interpretation")
    print("---------------------------")
    print(
        "A smaller tree is not automatically a better model. Compare "
        "validation performance, tree complexity, and operational costs. "
        "Impurity-based importance describes how the fitted tree uses "
        "features, not whether those features cause equipment failure."
    )

    demonstrate_edge_cases()


if __name__ == "__main__":
    main()
