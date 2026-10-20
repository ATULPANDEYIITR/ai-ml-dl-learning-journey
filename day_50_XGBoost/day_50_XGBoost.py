from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log
from pathlib import Path
from typing import Iterable, Sequence
import json
import random


def sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def logloss(y: Sequence[int], probabilities: Sequence[float]) -> float:
    eps = 1e-15
    total = 0.0
    for actual, probability in zip(y, probabilities):
        p = min(max(probability, eps), 1.0 - eps)
        total -= actual * log(p) + (1 - actual) * log(1 - p)
    return total / len(y)


@dataclass
class Row:
    features: tuple[float, ...]
    label: int


@dataclass
class TreeNode:
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    weight: float = 0.0

    def predict(self, features: Sequence[float]) -> float:
        if self.feature is None:
            return self.weight
        if features[self.feature] <= self.threshold:
            return self.left.predict(features)
        return self.right.predict(features)


@dataclass
class XGBoostLike:
    max_depth: int = 2
    learning_rate: float = 0.15
    n_estimators: int = 40
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    random_state: int = 7
    base_score: float = 0.5
    trees: list[TreeNode] = field(default_factory=list)
    feature_count: int = 0

    def _leaf_weight(self, gradients: list[float], hessians: list[float]) -> float:
        g = sum(gradients)
        h = sum(hessians)

        if h < self.min_child_weight:
            return 0.0

        # L1 regularization applies a soft-thresholding operation to G.
        if self.reg_alpha:
            if g > self.reg_alpha:
                g -= self.reg_alpha
            elif g < -self.reg_alpha:
                g += self.reg_alpha
            else:
                g = 0.0

        # XGBoost's second-order leaf objective:
        # w* = -G / (H + lambda)
        return -g / (h + self.reg_lambda)

    def _split_gain(
        self,
        left_g: float,
        left_h: float,
        right_g: float,
        right_h: float,
        parent_g: float,
        parent_h: float,
    ) -> float:
        if left_h < self.min_child_weight or right_h < self.min_child_weight:
            return float("-inf")

        def score(g: float, h: float) -> float:
            if self.reg_alpha:
                if g > self.reg_alpha:
                    g -= self.reg_alpha
                elif g < -self.reg_alpha:
                    g += self.reg_alpha
                else:
                    g = 0.0
            return (g * g) / (h + self.reg_lambda)

        gain = 0.5 * (
            score(left_g, left_h)
            + score(right_g, right_h)
            - score(parent_g, parent_h)
        )
        return gain - self.gamma

    def _build_tree(
        self,
        rows: list[Row],
        gradients: list[float],
        hessians: list[float],
        depth: int,
        rng: random.Random,
    ) -> TreeNode:
        node = TreeNode()
        node.weight = self._leaf_weight(gradients, hessians)

        if depth >= self.max_depth or len(rows) <= 1:
            return node

        feature_count = len(rows[0].features)
        available_features = list(range(feature_count))
        rng.shuffle(available_features)
        keep = max(1, round(feature_count * self.colsample_bytree))
        available_features = available_features[:keep]

        best_gain = self.gamma
        best_feature = None
        best_threshold = None
        best_left: list[int] = []
        best_right: list[int] = []

        for feature in available_features:
            values = sorted({row.features[feature] for row in rows})
            if len(values) < 2:
                continue

            thresholds = [
                (values[i] + values[i + 1]) / 2
                for i in range(len(values) - 1)
            ]

            for threshold in thresholds:
                left = [
                    i for i, row in enumerate(rows)
                    if row.features[feature] <= threshold
                ]
                right = [
                    i for i, row in enumerate(rows)
                    if row.features[feature] > threshold
                ]

                if not left or not right:
                    continue

                lg = sum(gradients[i] for i in left)
                lh = sum(hessians[i] for i in left)
                rg = sum(gradients[i] for i in right)
                rh = sum(hessians[i] for i in right)
                pg = sum(gradients)
                ph = sum(hessians)

                gain = self._split_gain(lg, lh, rg, rh, pg, ph)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left = left
                    best_right = right

        if best_feature is None:
            return node

        left_rows = [rows[i] for i in best_left]
        right_rows = [rows[i] for i in best_right]
        left_g = [gradients[i] for i in best_left]
        left_h = [hessians[i] for i in best_left]
        right_g = [gradients[i] for i in best_right]
        right_h = [hessians[i] for i in best_right]

        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(
            left_rows, left_g, left_h, depth + 1, rng
        )
        node.right = self._build_tree(
            right_rows, right_g, right_h, depth + 1, rng
        )
        node.weight = 0.0
        return node

    def fit(self, rows: Sequence[Row]) -> "XGBoostLike":
        if not rows:
            raise ValueError("Training data cannot be empty.")

        if any(row.label not in (0, 1) for row in rows):
            raise ValueError("This example supports binary labels 0 and 1.")

        self.feature_count = len(rows[0].features)
        if self.feature_count == 0:
            raise ValueError("At least one feature is required.")

        if any(len(row.features) != self.feature_count for row in rows):
            raise ValueError("All rows must have the same feature count.")

        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")
        if self.max_depth < 1:
            raise ValueError("max_depth must be positive.")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must be in (0, 1].")

        rng = random.Random(self.random_state)
        working_rows = list(rows)
        margin = [log(self.base_score / (1 - self.base_score))] * len(rows)

        self.trees.clear()

        for _ in range(self.n_estimators):
            probabilities = [sigmoid(value) for value in margin]

            gradients = [
                probability - row.label
                for probability, row in zip(probabilities, rows)
            ]
            hessians = [
                max(probability * (1 - probability), 1e-6)
                for probability in probabilities
            ]

            sample_size = max(1, round(len(rows) * self.subsample))
            selected = rng.sample(range(len(rows)), sample_size)

            sampled_rows = [rows[i] for i in selected]
            sampled_gradients = [gradients[i] for i in selected]
            sampled_hessians = [hessians[i] for i in selected]

            tree = self._build_tree(
                sampled_rows,
                sampled_gradients,
                sampled_hessians,
                0,
                rng,
            )
            self.trees.append(tree)

            for i, row in enumerate(rows):
                margin[i] += self.learning_rate * tree.predict(row.features)

        return self

    def predict_proba(self, features: Sequence[float]) -> float:
        if len(features) != self.feature_count:
            raise ValueError("Feature count does not match the trained model.")

        margin = log(self.base_score / (1 - self.base_score))
        for tree in self.trees:
            margin += self.learning_rate * tree.predict(features)
        return sigmoid(margin)

    def predict(self, features: Sequence[float], threshold: float = 0.5) -> int:
        if not 0 < threshold < 1:
            raise ValueError("Classification threshold must be between 0 and 1.")
        return int(self.predict_proba(features) >= threshold)


def make_dataset() -> list[Row]:
    # Features represent normalized project signals:
    # changed_files, test_failures, historical_risk, review_count.
    raw = [
        (1, 0, 0.10, 3, 0),
        (2, 0, 0.20, 4, 0),
        (3, 0, 0.15, 5, 0),
        (4, 1, 0.40, 3, 1),
        (5, 1, 0.55, 2, 1),
        (6, 2, 0.65, 2, 1),
        (7, 2, 0.80, 1, 1),
        (8, 3, 0.90, 1, 1),
        (2, 0, 0.05, 6, 0),
        (3, 1, 0.35, 4, 0),
        (5, 0, 0.30, 5, 1),
        (6, 1, 0.50, 3, 1),
        (9, 3, 0.95, 0, 1),
        (1, 0, 0.05, 7, 0),
        (4, 1, 0.45, 3, 1),
        (7, 1, 0.70, 2, 1),
        (3, 0, 0.25, 6, 0),
        (8, 2, 0.85, 1, 1),
        (5, 2, 0.60, 2, 1),
        (2, 0, 0.12, 5, 0),
    ]
    return [
        Row((float(a), float(b), float(c), float(d)), label)
        for a, b, c, d, label in raw
    ]


def save_model_metadata(model: XGBoostLike, path: str) -> None:
    metadata = {
        "model_type": "second_order_gradient_boosting_tree",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "learning_rate": model.learning_rate,
        "min_child_weight": model.min_child_weight,
        "reg_lambda": model.reg_lambda,
        "reg_alpha": model.reg_alpha,
        "gamma": model.gamma,
        "subsample": model.subsample,
        "colsample_bytree": model.colsample_bytree,
    }
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    data = make_dataset()
    random.Random(7).shuffle(data)

    split = int(len(data) * 0.75)
    train, test = data[:split], data[split:]

    model = XGBoostLike(
        max_depth=2,
        learning_rate=0.12,
        n_estimators=60,
        min_child_weight=1.0,
        reg_lambda=2.0,
        reg_alpha=0.05,
        gamma=0.02,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=7,
    ).fit(train)

    probabilities = [model.predict_proba(row.features) for row in test]
    predictions = [int(p >= 0.5) for p in probabilities]
    labels = [row.label for row in test]

    accuracy = sum(a == b for a, b in zip(labels, predictions)) / len(labels)

    print("XGBoost-style binary classifier")
    print(f"Trees: {len(model.trees)}")
    print(f"Test log loss: {logloss(labels, probabilities):.4f}")
    print(f"Test accuracy: {accuracy:.2%}")

    for row, probability, prediction in zip(test, probabilities, predictions):
        print(
            f"features={row.features} "
            f"probability={probability:.3f} prediction={prediction}"
        )

    save_model_metadata(model, "xgboost_model_metadata.json")
