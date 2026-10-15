"""
Decision Trees: Entropy, Gini Impurity, Information Gain, and Splitting

A self-contained implementation of a classification decision tree using
only the Python standard library.

The example predicts whether a software deployment should be approved
automatically or sent for manual review, based on validation results,
test coverage, and security findings.

The implementation includes:
- Class entropy and Gini impurity.
- Weighted impurity after a split.
- Information gain and Gini reduction.
- Numerical and categorical split candidates.
- Recursive tree construction.
- Maximum depth, minimum split size, and minimum gain controls.
- Prediction, class probabilities, feature importance, and evaluation.
- Cross-validation-style stratified evaluation.
- Model serialization, input validation, and deterministic examples.

Run with: python decision_trees.py
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


Label = str
FeatureValue = Any
Row = dict[str, FeatureValue]


def validate_labels(labels: Sequence[Label]) -> None:
    """Require a nonempty collection of hashable classification labels."""
    if not labels:
        raise ValueError("At least one label is required.")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Labels must be nonempty strings.")


def class_counts(labels: Sequence[Label]) -> dict[Label, int]:
    """Count labels without assuming a particular label ordering."""
    return dict(Counter(labels))


def entropy(labels: Sequence[Label]) -> float:
    """
    Calculate Shannon entropy in bits.

    Entropy is zero for a pure node. A balanced binary node has entropy one.
    The logarithm base determines the unit; base two gives bits.
    """
    if not labels:
        return 0.0

    counts = Counter(labels)
    total = len(labels)

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


def gini_impurity(labels: Sequence[Label]) -> float:
    """
    Calculate Gini impurity: 1 - sum(p_k ** 2).

    Gini impurity measures the probability of misclassification when a label
    is selected according to the node's empirical class distribution and
    then predicted using an independent draw from that distribution.
    """
    if not labels:
        return 0.0

    total = len(labels)
    return 1.0 - sum(
        (count / total) ** 2
        for count in Counter(labels).values()
    )


def weighted_child_impurity(
    child_label_groups: Sequence[Sequence[Label]],
    impurity_function,
) -> float:
    """Calculate child impurity weighted by the number of observations."""
    total = sum(len(group) for group in child_label_groups)

    if total == 0:
        raise ValueError("A split must contain at least one observation.")

    return sum(
        len(group) / total * impurity_function(group)
        for group in child_label_groups
    )


def information_gain(
    parent_labels: Sequence[Label],
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
) -> float:
    """Calculate parent entropy minus weighted child entropy."""
    if not left_labels or not right_labels:
        return 0.0

    child_entropy = weighted_child_impurity(
        [left_labels, right_labels],
        entropy,
    )
    return entropy(parent_labels) - child_entropy


def gini_gain(
    parent_labels: Sequence[Label],
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
) -> float:
    """Calculate the reduction in weighted Gini impurity."""
    if not left_labels or not right_labels:
        return 0.0

    child_gini = weighted_child_impurity(
        [left_labels, right_labels],
        gini_impurity,
    )
    return gini_impurity(parent_labels) - child_gini


@dataclass
class Split:
    """A binary split on a numerical or categorical feature."""

    feature: str
    kind: str
    threshold: Optional[float] = None
    category: Optional[FeatureValue] = None
    score: float = 0.0
    left_indices: list[int] = field(default_factory=list)
    right_indices: list[int] = field(default_factory=list)


@dataclass
class TreeNode:
    """A leaf predicts a class; an internal node routes observations."""

    prediction: Label
    counts: dict[Label, int]
    samples: int
    impurity: float
    depth: int
    feature: Optional[str] = None
    kind: Optional[str] = None
    threshold: Optional[float] = None
    category: Optional[FeatureValue] = None
    gain: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTreeClassifier:
    """
    Small CART-style binary classification tree.

    Numerical features use threshold splits of the form x <= threshold.
    Categorical features use equality splits of the form x == category.

    Categorical equality splits are suitable for nominal features. They do
    not impose an artificial ordering on values such as deployment regions.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 4,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        min_gain: float = 1e-12,
        max_threshold_candidates: Optional[int] = None,
    ) -> None:
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if min_gain < 0:
            raise ValueError("min_gain cannot be negative.")
        if (
            max_threshold_candidates is not None
            and max_threshold_candidates < 1
        ):
            raise ValueError("max_threshold_candidates must be positive.")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.max_threshold_candidates = max_threshold_candidates
        self.root: Optional[TreeNode] = None
        self.feature_names: list[str] = []
        self.feature_importances_: dict[str, float] = {}

    def _impurity(self, labels: Sequence[Label]) -> float:
        return gini_impurity(labels) if self.criterion == "gini" else entropy(labels)

    @staticmethod
    def _majority_label(labels: Sequence[Label]) -> Label:
        counts = Counter(labels)
        highest = max(counts.values())
        # Sorting gives deterministic tie-breaking across repeated runs.
        return sorted(
            label for label, count in counts.items() if count == highest
        )[0]

    @staticmethod
    def _validate_rows(
        rows: Sequence[Row],
        feature_names: Sequence[str],
        labels: Sequence[Label],
    ) -> None:
        if not rows:
            raise ValueError("Training data cannot be empty.")
        if len(rows) != len(labels):
            raise ValueError("The number of rows and labels must match.")
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("Feature names must be nonempty and unique.")

        validate_labels(labels)

        expected = set(feature_names)
        for index, row in enumerate(rows):
            missing = expected - row.keys()
            if missing:
                raise ValueError(
                    f"Row {index} is missing features: {sorted(missing)}"
                )
            for feature in feature_names:
                value = row[feature]
                if value is None:
                    raise ValueError(
                        f"Missing value for {feature!r} in row {index}."
                    )
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value for {feature!r} in row {index}."
                    )
                if not isinstance(value, (str, int, float, bool)):
                    raise TypeError(
                        f"Unsupported value type for {feature!r} in row {index}."
                    )

    def _candidate_thresholds(
        self,
        values: Sequence[float],
        labels: Sequence[Label],
    ) -> list[float]:
        """Generate midpoints only where adjacent sorted values differ."""
        pairs = sorted(zip(values, labels), key=lambda pair: pair[0])
        candidates: list[float] = []

        for (left_value, left_label), (right_value, right_label) in zip(
            pairs, pairs[1:]
        ):
            if left_value == right_value or left_label == right_label:
                continue

            midpoint = left_value + (right_value - left_value) / 2.0
            if math.isfinite(midpoint):
                candidates.append(midpoint)

        candidates = sorted(set(candidates))

        limit = self.max_threshold_candidates
        if limit is not None and len(candidates) > limit:
            # Deterministic subsampling bounds training work on high-cardinality
            # features while preserving the lowest and highest candidate.
            if limit == 1:
                return [candidates[len(candidates) // 2]]
            positions = [
                round(i * (len(candidates) - 1) / (limit - 1))
                for i in range(limit)
            ]
            candidates = [candidates[position] for position in positions]

        return candidates

    def _make_split(
        self,
        rows: Sequence[Row],
        labels: Sequence[Label],
        feature: str,
        kind: str,
        threshold: Optional[float] = None,
        category: Optional[FeatureValue] = None,
    ) -> Optional[Split]:
        """Partition the current observations and reject undersized children."""
        left_indices: list[int] = []
        right_indices: list[int] = []

        for index, row in enumerate(rows):
            value = row[feature]

            if kind == "numeric":
                goes_left = float(value) <= float(threshold)
            else:
                goes_left = value == category

            (left_indices if goes_left else right_indices).append(index)

        if (
            len(left_indices) < self.min_samples_leaf
            or len(right_indices) < self.min_samples_leaf
        ):
            return None

        parent_impurity = self._impurity(labels)
        left_labels = [labels[i] for i in left_indices]
        right_labels = [labels[i] for i in right_indices]

        weighted = weighted_child_impurity(
            [left_labels, right_labels],
            self._impurity,
        )
        score = parent_impurity - weighted

        return Split(
            feature=feature,
            kind=kind,
            threshold=threshold,
            category=category,
            score=score,
            left_indices=left_indices,
            right_indices=right_indices,
        )

    def _best_split(
        self,
        rows: Sequence[Row],
        labels: Sequence[Label],
    ) -> Optional[Split]:
        best: Optional[Split] = None

        for feature in self.feature_names:
            values = [row[feature] for row in rows]
            numeric = all(
                isinstance(value, (int, float, bool))
                and not isinstance(value, str)
                for value in values
            )

            candidates: Iterable[tuple[str, Optional[float], Any]]

            if numeric:
                thresholds = self._candidate_thresholds(
                    [float(value) for value in values],
                    labels,
                )
                candidates = (
                    ("numeric", threshold, None)
                    for threshold in thresholds
                )
            else:
                # Equality splits avoid pretending that nominal categories
                # have a meaningful numeric distance.
                categories = sorted(set(values), key=lambda value: str(value))
                candidates = (
                    ("categorical", None, category)
                    for category in categories[:-1]
                )

            for kind, threshold, category in candidates:
                split = self._make_split(
                    rows,
                    labels,
                    feature,
                    kind,
                    threshold,
                    category,
                )
                if split is None:
                    continue

                if best is None or split.score > best.score + 1e-15:
                    best = split

        return best

    def fit(
        self,
        rows: Sequence[Row],
        labels: Sequence[Label],
        feature_names: Optional[Sequence[str]] = None,
    ) -> "DecisionTreeClassifier":
        """Learn a tree by recursively selecting the highest-gain split."""
        if feature_names is None:
            if not rows:
                raise ValueError("Training data cannot be empty.")
            feature_names = list(rows[0].keys())

        self._validate_rows(rows, feature_names, labels)
        self.feature_names = list(feature_names)
        self.feature_importances_ = {name: 0.0 for name in self.feature_names}

        def build(indices: list[int], depth: int) -> TreeNode:
            subset_rows = [rows[i] for i in indices]
            subset_labels = [labels[i] for i in indices]
            counts = class_counts(subset_labels)
            prediction = self._majority_label(subset_labels)
            node_impurity = self._impurity(subset_labels)

            node = TreeNode(
                prediction=prediction,
                counts=counts,
                samples=len(indices),
                impurity=node_impurity,
                depth=depth,
            )

            if (
                depth >= self.max_depth
                or len(indices) < self.min_samples_split
                or node_impurity <= 0.0
            ):
                return node

            split = self._best_split(subset_rows, subset_labels)
            if split is None or split.score <= self.min_gain:
                return node

            left_global = [indices[i] for i in split.left_indices]
            right_global = [indices[i] for i in split.right_indices]

            node.feature = split.feature
            node.kind = split.kind
            node.threshold = split.threshold
            node.category = split.category
            node.gain = split.score

            # Importance accumulates the reduction in impurity weighted by
            # the fraction of the full training set that reaches the node.
            self.feature_importances_[split.feature] += (
                len(indices) / len(rows)
            ) * split.score

            node.left = build(left_global, depth + 1)
            node.right = build(right_global, depth + 1)
            return node

        self.root = build(list(range(len(rows))), 0)

        total_importance = sum(self.feature_importances_.values())
        if total_importance > 0:
            self.feature_importances_ = {
                feature: importance / total_importance
                for feature, importance in self.feature_importances_.items()
            }

        return self

    def _require_fitted(self) -> TreeNode:
        if self.root is None:
            raise RuntimeError("Call fit() before prediction.")
        return self.root

    def _leaf_for(self, row: Row) -> TreeNode:
        node = self._require_fitted()

        for feature in self.feature_names:
            if feature not in row or row[feature] is None:
                raise ValueError(f"Missing prediction feature: {feature!r}.")
            value = row[feature]
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Non-finite prediction value: {feature!r}.")

        while not node.is_leaf:
            value = row[node.feature]  # type: ignore[index]

            if node.kind == "numeric":
                if not isinstance(value, (int, float, bool)):
                    raise TypeError(
                        f"Feature {node.feature!r} must be numeric."
                    )
                goes_left = float(value) <= float(node.threshold)
            else:
                goes_left = value == node.category

            next_node = node.left if goes_left else node.right
            if next_node is None:
                raise RuntimeError("The fitted tree contains an invalid branch.")
            node = next_node

        return node

    def predict(self, rows: Sequence[Row]) -> list[Label]:
        return [self._leaf_for(row).prediction for row in rows]

    def predict_proba(self, rows: Sequence[Row]) -> list[dict[Label, float]]:
        """
        Return empirical class frequencies at each leaf.

        These are leaf frequencies, not calibrated probability estimates.
        """
        result: list[dict[Label, float]] = []

        for row in rows:
            leaf = self._leaf_for(row)
            result.append({
                label: count / leaf.samples
                for label, count in leaf.counts.items()
            })

        return result

    def render(self) -> str:
        """Produce a readable tree with node impurity and sample counts."""
        root = self._require_fitted()
        lines: list[str] = []

        def visit(node: TreeNode, prefix: str, branch: str) -> None:
            distribution = ", ".join(
                f"{label}={count}"
                for label, count in sorted(node.counts.items())
            )

            if node.is_leaf:
                lines.append(
                    f"{prefix}{branch}predict={node.prediction!r}, "
                    f"samples={node.samples}, impurity={node.impurity:.4f}, "
                    f"classes=[{distribution}]"
                )
                return

            if node.kind == "numeric":
                condition = f"{node.feature} <= {node.threshold:.4g}"
            else:
                condition = f"{node.feature} == {node.category!r}"

            lines.append(
                f"{prefix}{branch}if {condition}: "
                f"gain={node.gain:.4f}, samples={node.samples}, "
                f"impurity={node.impurity:.4f}"
            )
            assert node.left is not None and node.right is not None
            visit(node.left, prefix + "  ", "then ")
            visit(node.right, prefix + "  ", "else ")

        visit(root, "", "")
        return "\n".join(lines)


def accuracy_score(
    actual: Sequence[Label],
    predicted: Sequence[Label],
) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Accuracy requires equal, nonempty label sequences.")
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def confusion_matrix(
    actual: Sequence[Label],
    predicted: Sequence[Label],
    labels: Sequence[Label],
) -> dict[Label, dict[Label, int]]:
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths differ.")

    matrix = {
        actual_label: {predicted_label: 0 for predicted_label in labels}
        for actual_label in labels
    }

    for actual_label, predicted_label in zip(actual, predicted):
        if actual_label not in matrix or predicted_label not in matrix:
            raise ValueError("An observed label is absent from the label list.")
        matrix[actual_label][predicted_label] += 1

    return matrix


def stratified_train_test_split(
    labels: Sequence[Label],
    test_fraction: float = 0.25,
    seed: int = 17,
) -> tuple[list[int], list[int]]:
    """Split each class independently to retain approximate class balance."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one.")

    by_class: dict[Label, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for indices in by_class.values():
        rng.shuffle(indices)

        if len(indices) < 2:
            train_indices.extend(indices)
            continue

        test_count = max(1, round(len(indices) * test_fraction))
        test_count = min(test_count, len(indices) - 1)
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def make_deployment_dataset() -> tuple[list[Row], list[Label]]:
    """
    Create an illustrative, deterministic dataset.

    The values model software quality signals, not a validated production
    release policy. Labels encode an example governance decision.
    """
    rows: list[Row] = []
    labels: list[Label] = []

    for coverage in (42, 55, 63, 71, 78, 84, 91, 96):
        for failures in (0, 1, 2):
            for security in ("clear", "review"):
                for validation in ("passed", "failed"):
                    risk = (
                        coverage < 70
                        or failures >= 2
                        or security == "review"
                        or validation == "failed"
                    )
                    rows.append({
                        "test_coverage": coverage,
                        "failed_tests": failures,
                        "security_scan": security,
                        "validation": validation,
                    })
                    labels.append("manual_review" if risk else "auto_approve")

    return rows, labels


def demonstrate_impurity() -> None:
    print("\nImpurity calculations")

    examples = {
        "pure node": ["approved"] * 8,
        "balanced node": ["approved"] * 4 + ["review"] * 4,
        "uneven node": ["approved"] * 6 + ["review"] * 2,
    }

    for name, labels in examples.items():
        print(
            f"{name:15s} entropy={entropy(labels):.4f}, "
            f"gini={gini_impurity(labels):.4f}"
        )

    parent = ["approved"] * 6 + ["review"] * 6
    left = ["approved"] * 5 + ["review"]
    right = ["approved"] + ["review"] * 5

    print(f"Example information gain: {information_gain(parent, left, right):.4f}")
    print(f"Example Gini reduction:   {gini_gain(parent, left, right):.4f}")


def demonstrate_edge_cases() -> None:
    print("\nEdge cases")

    assert entropy([]) == 0.0
    assert gini_impurity([]) == 0.0
    assert entropy(["approved"] * 5) == 0.0
    assert gini_impurity(["approved"] * 5) == 0.0

    try:
        DecisionTreeClassifier().fit([], [])
    except ValueError as error:
        print(f"Empty training data rejected: {error}")

    try:
        DecisionTreeClassifier().fit(
            [{"coverage": 80}],
            ["approved", "review"],
            ["coverage"],
        )
    except ValueError as error:
        print(f"Length mismatch rejected: {error}")

    try:
        DecisionTreeClassifier().fit(
            [{"coverage": float("nan")}],
            ["approved"],
            ["coverage"],
        )
    except ValueError as error:
        print(f"Non-finite feature rejected: {error}")


def main() -> None:
    demonstrate_impurity()
    demonstrate_edge_cases()

    rows, labels = make_deployment_dataset()
    train_indices, test_indices = stratified_train_test_split(labels)

    train_rows = [rows[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_rows = [rows[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    print("\nTraining an entropy-based tree")
    entropy_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        min_samples_split=8,
        min_samples_leaf=3,
        min_gain=1e-8,
    ).fit(train_rows, train_labels)

    print(entropy_tree.render())

    predictions = entropy_tree.predict(test_rows)
    print(f"\nHeld-out accuracy: {accuracy_score(test_labels, predictions):.3f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(json.dumps(
        confusion_matrix(
            test_labels,
            predictions,
            ["auto_approve", "manual_review"],
        ),
        indent=2,
    ))

    print("\nNormalized feature importance")
    for feature, importance in sorted(
        entropy_tree.feature_importances_.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"{feature:18s} {importance:.3f}")

    print("\nPrediction for a proposed deployment")
    candidate = {
        "test_coverage": 88,
        "failed_tests": 0,
        "security_scan": "clear",
        "validation": "passed",
    }
    print(f"Decision: {entropy_tree.predict([candidate])[0]}")
    print(f"Leaf frequencies: {entropy_tree.predict_proba([candidate])[0]}")

    print("\nComparing impurity criteria")
    for criterion in ("gini", "entropy"):
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=4,
            min_samples_split=8,
            min_samples_leaf=3,
        ).fit(train_rows, train_labels)
        score = accuracy_score(test_labels, model.predict(test_rows))
        print(f"{criterion:8s} held-out accuracy={score:.3f}")

    print("\nDepth regularization experiment")
    for depth in (1, 2, 4, 8):
        model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            min_samples_split=8,
            min_samples_leaf=3,
        ).fit(train_rows, train_labels)
        score = accuracy_score(test_labels, model.predict(test_rows))
        print(f"max_depth={depth}: held-out accuracy={score:.3f}")

    # Model persistence stores the learned tree structure rather than
    # executable code. The JSON can be inspected and validated before use.
    model_path = Path("deployment_decision_tree.json")
    root = entropy_tree._require_fitted()

    def node_to_dict(node: TreeNode) -> dict[str, Any]:
        return {
            "prediction": node.prediction,
            "counts": node.counts,
            "samples": node.samples,
            "impurity": node.impurity,
            "depth": node.depth,
            "feature": node.feature,
            "kind": node.kind,
            "threshold": node.threshold,
            "category": node.category,
            "gain": node.gain,
            "left": node_to_dict(node.left) if node.left else None,
            "right": node_to_dict(node.right) if node.right else None,
        }

    payload = {
        "criterion": entropy_tree.criterion,
        "feature_names": entropy_tree.feature_names,
        "root": node_to_dict(root),
        "warning": (
            "Illustrative model; validate policy and generalization before use."
        ),
    }
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nModel structure written to {model_path.resolve()}")

    # Round-trip parsing catches malformed output without executing it.
    parsed = json.loads(model_path.read_text(encoding="utf-8"))
    assert parsed["root"]["samples"] == len(train_rows)
    print("Serialized model structure verified.")


if __name__ == "__main__":
    main()
