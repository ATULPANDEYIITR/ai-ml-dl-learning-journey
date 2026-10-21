#!/usr/bin/env python3
"""
LightGBM-style histogram boosting and CatBoost-style categorical boosting.

This self-contained implementation focuses on the mechanisms behind:
- histogram-based split search
- leaf-wise tree growth
- categorical target statistics
- ordered target encoding
- additive gradient boosting

It uses only the Python standard library and is intended as an executable
technical learning implementation rather than a replacement for the optimized
LightGBM or CatBoost libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log
from random import Random
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Row:
    age: float
    income: float
    tenure: float
    region: str
    plan: str
    churn: int


DATA = [
    Row(23, 31000, 4, "north", "basic", 1),
    Row(25, 42000, 8, "south", "basic", 1),
    Row(29, 58000, 18, "west", "pro", 0),
    Row(31, 61000, 24, "north", "pro", 0),
    Row(35, 72000, 30, "east", "pro", 0),
    Row(41, 83000, 44, "west", "enterprise", 0),
    Row(48, 91000, 52, "north", "enterprise", 0),
    Row(52, 39000, 9, "south", "basic", 1),
    Row(27, 36000, 6, "east", "basic", 1),
    Row(33, 69000, 27, "south", "pro", 0),
    Row(38, 51000, 15, "east", "basic", 1),
    Row(45, 87000, 47, "west", "enterprise", 0),
    Row(22, 28000, 3, "north", "basic", 1),
    Row(57, 99000, 72, "north", "enterprise", 0),
    Row(30, 47000, 11, "west", "pro", 0),
    Row(36, 45000, 10, "south", "basic", 1),
    Row(43, 78000, 36, "east", "pro", 0),
    Row(50, 54000, 21, "south", "pro", 1),
    Row(28, 62000, 22, "north", "pro", 0),
    Row(39, 34000, 7, "east", "basic", 1),
]


def sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + exp(-value))


def logloss(y: Sequence[int], p: Sequence[float]) -> float:
    eps = 1e-12
    return -sum(
        target * log(max(eps, probability))
        + (1 - target) * log(max(eps, 1 - probability))
        for target, probability in zip(y, p)
    ) / len(y)


def encode_categories(rows: Sequence[Row]) -> Tuple[List[List[float]], List[int]]:
    """
    A direct numeric representation used by the histogram model.

    Production LightGBM can natively work with categorical features. This
    compact model uses deterministic category codes so the histogram engine
    can remain self-contained.
    """
    region_map = {value: i for i, value in enumerate(sorted({r.region for r in rows}))}
    plan_map = {value: i for i, value in enumerate(sorted({r.plan for r in rows}))}

    X = [
        [
            row.age,
            row.income,
            row.tenure,
            float(region_map[row.region]),
            float(plan_map[row.plan]),
        ]
        for row in rows
    ]
    y = [row.churn for row in rows]
    return X, y


@dataclass
class HistNode:
    indices: List[int]
    depth: int
    value: float = 0.0
    split_feature: Optional[int] = None
    split_threshold: Optional[float] = None
    left: Optional["HistNode"] = None
    right: Optional["HistNode"] = None


class HistogramTree:
    """
    Small histogram decision tree.

    The important optimization is that continuous values are compressed into
    a fixed number of bins. Split search then operates over bins instead of
    testing every distinct floating-point value.
    """

    def __init__(self, max_depth: int = 3, max_bins: int = 8, min_leaf: int = 2):
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_leaf = min_leaf
        self.root: Optional[HistNode] = None
        self.bin_edges: Dict[int, List[float]] = {}

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _build_edges(self, X: Sequence[Sequence[float]]) -> None:
        feature_count = len(X[0])
        for feature in range(feature_count):
            values = sorted({row[feature] for row in X})
            if len(values) <= self.max_bins:
                self.bin_edges[feature] = values
                continue

            edges = []
            for i in range(1, self.max_bins):
                position = int(i * len(values) / self.max_bins)
                position = min(len(values) - 1, max(0, position))
                edges.append(values[position])
            self.bin_edges[feature] = sorted(set(edges))

    def _bin(self, feature: int, value: float) -> int:
        edges = self.bin_edges[feature]
        for index, edge in enumerate(edges):
            if value <= edge:
                return index
        return len(edges)

    def _split_gain(
        self,
        gradients: Sequence[float],
        hessians: Sequence[float],
        left: Sequence[int],
        right: Sequence[int],
        l2: float = 1.0,
    ) -> float:
        if len(left) < self.min_leaf or len(right) < self.min_leaf:
            return float("-inf")

        def score(indices: Sequence[int]) -> float:
            gradient_sum = sum(gradients[i] for i in indices)
            hessian_sum = sum(hessians[i] for i in indices)
            return gradient_sum * gradient_sum / (hessian_sum + l2)

        return 0.5 * (score(left) + score(right))

    def fit(
        self,
        X: Sequence[Sequence[float]],
        gradients: Sequence[float],
        hessians: Sequence[float],
    ) -> None:
        if not X:
            raise ValueError("Training data cannot be empty.")
        if len(X) != len(gradients) or len(X) != len(hessians):
            raise ValueError("X, gradients and hessians must have equal length.")

        self._build_edges(X)
        indices = list(range(len(X)))

        def grow(current: List[int], depth: int) -> HistNode:
            gradient_sum = sum(gradients[i] for i in current)
            hessian_sum = sum(hessians[i] for i in current)
            leaf_value = -gradient_sum / (hessian_sum + 1.0)

            node = HistNode(current, depth, leaf_value)

            if depth >= self.max_depth or len(current) < 2 * self.min_leaf:
                return node

            best_gain = 0.0
            best_feature = None
            best_threshold = None
            best_left: List[int] = []
            best_right: List[int] = []

            for feature in range(len(X[0])):
                bins: Dict[int, List[int]] = {}
                for index in current:
                    bucket = self._bin(feature, X[index][feature])
                    bins.setdefault(bucket, []).append(index)

                ordered_bins = sorted(bins)
                if len(ordered_bins) < 2:
                    continue

                left_indices: List[int] = []
                for bucket in ordered_bins[:-1]:
                    left_indices.extend(bins[bucket])
                    right_indices = [
                        index for bucket2 in ordered_bins if bucket2 > bucket
                        for index in bins[bucket2]
                    ]

                    gain = self._split_gain(
                        gradients, hessians, left_indices, right_indices
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = self.bin_edges[feature][bucket]
                        best_left = left_indices[:]
                        best_right = right_indices[:]

            if best_feature is None:
                return node

            node.split_feature = best_feature
            node.split_threshold = best_threshold
            node.left = grow(best_left, depth + 1)
            node.right = grow(best_right, depth + 1)
            return node

        self.root = grow(indices, 0)

    def predict_one(self, row: Sequence[float]) -> float:
        if self.root is None:
            raise RuntimeError("Tree has not been fitted.")

        node = self.root
        while node.split_feature is not None:
            if row[node.split_feature] <= node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node.value


class LightGBMStyleClassifier:
    """
    Logistic gradient boosting with histogram-based trees and leaf-wise
    candidate selection.

    The implementation uses second-order logistic gradients and Hessians.
    """

    def __init__(
        self,
        estimators: int = 30,
        learning_rate: float = 0.08,
        max_depth: int = 3,
        max_bins: int = 8,
    ):
        if estimators <= 0 or learning_rate <= 0:
            raise ValueError("estimators and learning_rate must be positive.")
        self.estimators = estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.base_score = 0.0
        self.trees: List[HistogramTree] = []

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        if not X or len(X) != len(y):
            raise ValueError("X and y must contain the same non-zero number of rows.")
        if any(target not in (0, 1) for target in y):
            raise ValueError("This classifier expects binary targets.")

        positive_rate = sum(y) / len(y)
        positive_rate = min(1 - 1e-6, max(1e-6, positive_rate))
        self.base_score = log(positive_rate / (1 - positive_rate))
        raw = [self.base_score] * len(y)
        self.trees.clear()

        for _ in range(self.estimators):
            probabilities = [sigmoid(value) for value in raw]
            gradients = [p - target for p, target in zip(probabilities, y)]
            hessians = [max(1e-6, p * (1 - p)) for p in probabilities]

            tree = HistogramTree(
                max_depth=self.max_depth,
                max_bins=self.max_bins,
                min_leaf=2,
            )
            tree.fit(X, gradients, hessians)
            self.trees.append(tree)

            for i, row in enumerate(X):
                raw[i] += self.learning_rate * tree.predict_one(row)

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        predictions = []
        for row in X:
            raw = self.base_score
            raw += self.learning_rate * sum(tree.predict_one(row) for tree in self.trees)
            predictions.append(sigmoid(raw))
        return predictions

    def predict(self, X: Sequence[Sequence[float]], threshold: float = 0.5) -> List[int]:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1.")
        return [int(p >= threshold) for p in self.predict_proba(X)]


class OrderedTargetEncoder:
    """
    CatBoost-inspired ordered target statistics.

    For each row, the category statistic is calculated only from earlier
    observations in a deterministic permutation. This avoids using the
    current row's target directly and reduces target leakage.
    """

    def __init__(self, columns: Sequence[int], prior_weight: float = 5.0, seed: int = 7):
        self.columns = list(columns)
        self.prior_weight = prior_weight
        self.seed = seed
        self.global_mean = 0.0
        self.maps: Dict[int, Dict[Any, Tuple[float, int]]] = {}

    def fit_transform(
        self, X: Sequence[Sequence[Any]], y: Sequence[int]
    ) -> List[List[float]]:
        if not X:
            raise ValueError("X cannot be empty.")
        self.global_mean = sum(y) / len(y)
        result = [list(map(float, row[:3])) for row in X]

        rng = Random(self.seed)
        order = list(range(len(X)))
        rng.shuffle(order)

        self.maps = {column: {} for column in self.columns}

        for position, row_index in enumerate(order):
            for column in self.columns:
                category = X[row_index][column]
                total, count = self.maps[column].get(category, (0.0, 0))
                statistic = (
                    total + self.prior_weight * self.global_mean
                ) / (count + self.prior_weight)
                result[row_index].append(statistic)

            target = y[row_index]
            for column in self.columns:
                category = X[row_index][column]
                total, count = self.maps[column].get(category, (0.0, 0))
                self.maps[column][category] = (total + target, count + 1)

        return result

    def transform(self, X: Sequence[Sequence[Any]]) -> List[List[float]]:
        result = []
        for row in X:
            encoded = list(map(float, row[:3]))
            for column in self.columns:
                category = row[column]
                total, count = self.maps[column].get(category, (0.0, 0))
                encoded.append(
                    (total + self.prior_weight * self.global_mean)
                    / (count + self.prior_weight)
                )
            result.append(encoded)
        return result


def train_test_split(
    X: Sequence[Any], y: Sequence[int], test_fraction: float = 0.25
) -> Tuple[List[Any], List[Any], List[int], List[int]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    split = max(1, int(len(X) * (1 - test_fraction)))
    return (
        list(X[:split]),
        list(X[split:]),
        list(y[:split]),
        list(y[split:]),
    )


def run_demo() -> None:
    X, y = encode_categories(DATA)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LightGBMStyleClassifier(
        estimators=35,
        learning_rate=0.06,
        max_depth=3,
        max_bins=6,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    print("LightGBM-style histogram boosting")
    print("--------------------------------")
    print(f"training rows : {len(X_train)}")
    print(f"testing rows  : {len(X_test)}")
    print(f"test log loss : {logloss(y_test, probabilities):.4f}")
    print(f"predictions   : {predictions}")
    print(f"probabilities : {[round(p, 3) for p in probabilities]}")

    raw_categorical = [
        [r.age, r.income, r.tenure, r.region, r.plan] for r in DATA
    ]
    categorical_encoder = OrderedTargetEncoder(columns=[3, 4], seed=17)
    encoded_train = categorical_encoder.fit_transform(
        raw_categorical[:len(X_train)], y_train
    )
    encoded_test = categorical_encoder.transform(
        raw_categorical[len(X_train):]
    )

    cat_model = LightGBMStyleClassifier(
        estimators=25,
        learning_rate=0.07,
        max_depth=3,
        max_bins=6,
    )
    cat_model.fit(encoded_train, y_train)
    cat_probabilities = cat_model.predict_proba(encoded_test)

    print()
    print("CatBoost-style ordered categorical preprocessing")
    print("-----------------------------------------------")
    print(f"encoded feature count : {len(encoded_train[0])}")
    print(f"test log loss         : {logloss(y_test, cat_probabilities):.4f}")
    print(f"probabilities         : {[round(p, 3) for p in cat_probabilities]}")

    print()
    print("Production considerations")
    print("--------------------------")
    print("* Histogram bins reduce repeated split-search work.")
    print("* Smaller bin counts reduce memory but can lose split precision.")
    print("* Leaf-wise growth can overfit when trees are unconstrained.")
    print("* Ordered statistics reduce categorical target leakage.")
    print("* Production models should use validated train/validation splits,")
    print("  early stopping, feature monitoring, reproducible seeds, and")
    print("  library implementations optimized for large datasets.")


if __name__ == "__main__":
    run_demo()
