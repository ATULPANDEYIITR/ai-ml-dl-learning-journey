"""
Random Forest: Bagging, Bootstrap Sampling, and Random Feature Selection

A self-contained implementation using only the Python standard library.
It demonstrates decision trees, bootstrap sampling, feature subsampling,
out-of-bag evaluation, feature importance, and a practical classification case.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class Sample:
    """One observation with numerical features and a categorical target."""
    features: tuple[float, ...]
    label: str


@dataclass
class TreeNode:
    """A decision-tree node; leaf nodes store a predicted class."""
    prediction: str
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    probabilities: Optional[dict[str, float]] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


def validate_dataset(
    samples: Sequence[Sample],
    expected_features: Optional[int] = None,
) -> int:
    if not samples:
        raise ValueError("The dataset cannot be empty.")

    feature_count = len(samples[0].features)
    if feature_count == 0:
        raise ValueError("Each observation needs at least one feature.")

    if expected_features is not None and feature_count != expected_features:
        raise ValueError("Unexpected number of features.")

    for sample in samples:
        if len(sample.features) != feature_count:
            raise ValueError("Inconsistent feature dimensions.")
        if not sample.label:
            raise ValueError("Labels must be non-empty.")
        if not all(math.isfinite(value) for value in sample.features):
            raise ValueError("Features must contain finite numerical values.")

    return feature_count


def majority_label(labels: Sequence[str]) -> str:
    """Resolve ties deterministically instead of depending on input order."""
    counts = Counter(labels)
    highest_count = max(counts.values())
    return min(label for label, count in counts.items() if count == highest_count)


def gini_impurity(labels: Sequence[str]) -> float:
    """Measure class mixing: zero means a pure node."""
    if not labels:
        return 0.0

    counts = Counter(labels)
    total = len(labels)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def entropy(labels: Sequence[str]) -> float:
    """Alternative impurity measure based on information theory."""
    if not labels:
        return 0.0

    counts = Counter(labels)
    total = len(labels)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


def bootstrap_sample(
    samples: Sequence[Sample],
    rng: random.Random,
) -> tuple[list[Sample], set[int]]:
    """
    Draw n observations with replacement.

    The returned index set identifies observations never selected, which
    can be used for out-of-bag evaluation.
    """
    selected_indices = [
        rng.randrange(len(samples)) for _ in range(len(samples))
    ]
    selected = [samples[index] for index in selected_indices]
    out_of_bag = set(range(len(samples))) - set(selected_indices)
    return selected, out_of_bag


class DecisionTree:
    """
    CART-style classification tree.

    At every node, candidate features are sampled randomly. Thresholds are
    derived from adjacent sorted values. Gini impurity chooses the split.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        max_features: Optional[int] = None,
        random_state: int = 0,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive.")
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be positive.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = random.Random(random_state)
        self.root: Optional[TreeNode] = None
        self.n_features = 0
        self.split_importance: Counter[int] = Counter()

    def fit(self, samples: Sequence[Sample]) -> "DecisionTree":
        self.n_features = validate_dataset(samples)
        self.split_importance.clear()
        self.root = self._grow(list(samples), depth=0)
        return self

    def _grow(self, samples: list[Sample], depth: int) -> TreeNode:
        labels = [sample.label for sample in samples]
        prediction = majority_label(labels)

        node = TreeNode(
            prediction=prediction,
            probabilities={
                label: count / len(labels)
                for label, count in Counter(labels).items()
            },
        )

        if (
            depth >= self.max_depth
            or len(samples) < self.min_samples_split
            or len(set(labels)) == 1
        ):
            return node

        requested = self.max_features or self.n_features
        feature_count = min(requested, self.n_features)
        candidate_features = self.rng.sample(
            range(self.n_features), feature_count
        )

        best_gain = 0.0
        best_split: Optional[tuple[int, float, list[Sample], list[Sample]]] = None
        parent_impurity = gini_impurity(labels)

        for feature_index in candidate_features:
            values = sorted({sample.features[feature_index] for sample in samples})

            # A split between adjacent distinct values avoids redundant
            # thresholds at repeated observations.
            thresholds = [
                (values[i] + values[i + 1]) / 2.0
                for i in range(len(values) - 1)
            ]

            for threshold in thresholds:
                left = [
                    sample for sample in samples
                    if sample.features[feature_index] <= threshold
                ]
                right = [
                    sample for sample in samples
                    if sample.features[feature_index] > threshold
                ]

                if (
                    len(left) < self.min_samples_leaf
                    or len(right) < self.min_samples_leaf
                ):
                    continue

                weighted_impurity = (
                    len(left) / len(samples)
                    * gini_impurity([sample.label for sample in left])
                    + len(right) / len(samples)
                    * gini_impurity([sample.label for sample in right])
                )
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold, left, right)

        if best_split is None:
            return node

        feature_index, threshold, left_samples, right_samples = best_split

        # Record weighted impurity reduction, not just split frequency.
        self.split_importance[feature_index] += (
            best_gain * len(samples)
        )

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._grow(left_samples, depth + 1)
        node.right = self._grow(right_samples, depth + 1)
        return node

    def predict(self, features: Sequence[float]) -> str:
        if self.root is None:
            raise RuntimeError("Fit the decision tree before prediction.")
        if len(features) != self.n_features:
            raise ValueError("Unexpected number of input features.")
        if not all(math.isfinite(value) for value in features):
            raise ValueError("Prediction features must be finite.")

        node = self.root
        while not node.is_leaf:
            assert node.feature_index is not None
            assert node.threshold is not None
            if features[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right

        return node.prediction


class RandomForest:
    """
    Bootstrap aggregation of randomized decision trees.

    Each tree receives a bootstrap dataset and uses a different random
    sequence for feature selection and candidate split exploration.
    """

    def __init__(
        self,
        n_estimators: int = 31,
        max_depth: int = 5,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees: list[DecisionTree] = []
        self.oob_indices: list[set[int]] = []
        self.n_features = 0
        self.labels: list[str] = []
        self.feature_importances_: dict[int, float] = {}
        self.oob_predictions_: dict[int, str] = {}
        self.oob_score_: Optional[float] = None

    def fit(self, samples: Sequence[Sample]) -> "RandomForest":
        self.n_features = validate_dataset(samples)
        self.labels = sorted({sample.label for sample in samples})

        if self.max_features is not None and self.max_features > self.n_features:
            raise ValueError("max_features cannot exceed the feature count.")

        master_rng = random.Random(self.random_state)
        self.trees.clear()
        self.oob_indices.clear()
        aggregate_importance: defaultdict[int, float] = defaultdict(float)
        oob_votes: defaultdict[int, list[str]] = defaultdict(list)

        for _ in range(self.n_estimators):
            tree_seed = master_rng.randrange(0, 2**32)
            rng = random.Random(tree_seed)
            bootstrapped, out_of_bag = bootstrap_sample(samples, rng)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            tree.fit(bootstrapped)

            self.trees.append(tree)
            self.oob_indices.append(out_of_bag)

            for feature_index, importance in tree.split_importance.items():
                aggregate_importance[feature_index] += importance

            for index in out_of_bag:
                oob_votes[index].append(tree.predict(samples[index].features))

        total_importance = sum(aggregate_importance.values())
        self.feature_importances_ = {
            feature_index: (
                aggregate_importance[feature_index] / total_importance
                if total_importance > 0 else 0.0
            )
            for feature_index in range(self.n_features)
        }

        self.oob_predictions_ = {
            index: majority_label(votes)
            for index, votes in oob_votes.items()
            if votes
        }

        if self.oob_predictions_:
            correct = sum(
                self.oob_predictions_[index] == samples[index].label
                for index in self.oob_predictions_
            )
            self.oob_score_ = correct / len(self.oob_predictions_)
        else:
            self.oob_score_ = None

        return self

    def predict(self, features: Sequence[float]) -> str:
        if not self.trees:
            raise RuntimeError("Fit the random forest before prediction.")
        if len(features) != self.n_features:
            raise ValueError("Unexpected number of input features.")

        votes = [tree.predict(features) for tree in self.trees]
        return majority_label(votes)

    def predict_proba(self, features: Sequence[float]) -> dict[str, float]:
        if not self.trees:
            raise RuntimeError("Fit the random forest before prediction.")
        if len(features) != self.n_features:
            raise ValueError("Unexpected number of input features.")

        votes = Counter(tree.predict(features) for tree in self.trees)
        return {
            label: votes[label] / len(self.trees)
            for label in self.labels
        }


def accuracy(samples: Sequence[Sample], predictions: Sequence[str]) -> float:
    if not samples or len(samples) != len(predictions):
        raise ValueError("Samples and predictions must have equal non-zero lengths.")
    return sum(
        sample.label == prediction
        for sample, prediction in zip(samples, predictions)
    ) / len(samples)


def make_equipment_dataset(
    seed: int = 17,
    observations: int = 240,
) -> list[Sample]:
    """
    Simulate equipment inspection records.

    Features: vibration, temperature, operating hours, pressure deviation.
    Failure risk depends on nonlinear combinations of these measurements.
    """
    rng = random.Random(seed)
    samples: list[Sample] = []

    for _ in range(observations):
        vibration = rng.uniform(0.0, 10.0)
        temperature = rng.uniform(35.0, 110.0)
        operating_hours = rng.uniform(0.0, 12000.0)
        pressure_deviation = rng.uniform(0.0, 8.0)

        risk = (
            0.28 * vibration
            + 0.045 * max(0.0, temperature - 70.0)
            + 0.00025 * operating_hours
            + 0.65 * pressure_deviation
            + 0.12 * vibration * pressure_deviation
        )

        # The small noise term avoids making the boundary perfectly rigid.
        risk += rng.gauss(0.0, 0.8)
        label = "maintenance_required" if risk > 8.0 else "normal"

        samples.append(
            Sample(
                features=(
                    vibration,
                    temperature,
                    operating_hours,
                    pressure_deviation,
                ),
                label=label,
            )
        )

    rng.shuffle(samples)
    return samples


def main() -> None:
    print("Random Forest: Equipment Failure Classification")
    print("=" * 54)

    dataset = make_equipment_dataset()
    split = int(len(dataset) * 0.8)
    training = dataset[:split]
    testing = dataset[split:]

    print(f"Training observations: {len(training)}")
    print(f"Testing observations:  {len(testing)}")
    print(f"Training class counts: {dict(Counter(s.label for s in training))}")
    print(f"Training Gini impurity: {gini_impurity([s.label for s in training]):.4f}")
    print(f"Training entropy:       {entropy([s.label for s in training]):.4f}")

    # sqrt(p) is a common classification default. The explicit integer
    # makes the feature-sampling choice reproducible and easy to inspect.
    forest = RandomForest(
        n_estimators=51,
        max_depth=6,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features=2,
        random_state=2026,
    )
    forest.fit(training)

    predictions = [
        forest.predict(sample.features)
        for sample in testing
    ]

    print(f"\nTest accuracy: {accuracy(testing, predictions):.3f}")
    if forest.oob_score_ is None:
        print("Out-of-bag accuracy: unavailable")
    else:
        print(f"Out-of-bag accuracy: {forest.oob_score_:.3f}")
        print(f"Observations with OOB predictions: {len(forest.oob_predictions_)}")

    print("\nFeature importance")
    feature_names = [
        "vibration",
        "temperature",
        "operating_hours",
        "pressure_deviation",
    ]
    for index, importance in sorted(
        forest.feature_importances_.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{feature_names[index]:22s} {importance:.4f}")

    examples = [
        (2.0, 48.0, 1200.0, 1.0),
        (8.2, 96.0, 9800.0, 6.5),
        (5.5, 78.0, 5500.0, 3.0),
    ]

    print("\nInspection predictions")
    for features in examples:
        probabilities = forest.predict_proba(features)
        prediction = forest.predict(features)
        print(f"Features: {features}")
        print(f"Prediction: {prediction}")
        print(f"Tree-vote proportions: {probabilities}")

    print("\nBootstrap sampling demonstration")
    _, oob = bootstrap_sample(training, random.Random(7))
    print(f"Training rows: {len(training)}")
    print(f"Unique out-of-bag rows: {len(oob)}")

    print("\nInterpretation")
    print(
        "Each tree trains on a bootstrap sample. Random feature selection "
        "reduces similarity between trees. Majority voting aggregates their "
        "predictions, while out-of-bag observations provide an internal "
        "generalization estimate."
    )


if __name__ == "__main__":
    main()
