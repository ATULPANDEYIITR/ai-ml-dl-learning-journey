#!/usr/bin/env python3
"""
ML Model Selection:
Cross-validation, K-fold CV, stratified CV, and repeated CV.

Self-contained standard-library implementation using a deterministic synthetic
binary-classification dataset. The script implements:
- Logistic regression with L2 regularization
- Decision tree classifier
- K-fold cross-validation
- Stratified K-fold cross-validation
- Repeated stratified K-fold cross-validation
- Accuracy, precision, recall, F1, and log loss
- Model selection using validation-only data
- Final holdout evaluation
- Validation of folds and class distributions
- Reproducible randomization
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random
import statistics
from typing import Callable, Iterable, Sequence


class CVStrategy(Enum):
    K_FOLD = "k-fold"
    STRATIFIED_K_FOLD = "stratified-k-fold"
    REPEATED_STRATIFIED_K_FOLD = "repeated-stratified-k-fold"


@dataclass(frozen=True)
class Dataset:
    X: list[list[float]]
    y: list[int]

    def validate(self) -> None:
        if not self.X:
            raise ValueError("Dataset cannot be empty.")
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have equal lengths.")
        width = len(self.X[0])
        if width == 0:
            raise ValueError("Features cannot be empty.")
        if any(len(row) != width for row in self.X):
            raise ValueError("All feature rows must have equal width.")
        if any(label not in (0, 1) for label in self.y):
            raise ValueError("This example requires binary labels 0 and 1.")


@dataclass(frozen=True)
class Fold:
    train_indices: list[int]
    validation_indices: list[int]


@dataclass(frozen=True)
class ModelResult:
    name: str
    mean_f1: float
    std_f1: float
    mean_accuracy: float
    mean_log_loss: float
    scores: list[float]


class LogisticRegression:
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.08,
        epochs: int = 500,
        l2: float = 0.01,
    ) -> None:
        if learning_rate <= 0 or epochs <= 0 or l2 < 0:
            raise ValueError("Invalid logistic-regression hyperparameters.")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights: list[float] = []
        self.bias = 0.0

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-60.0, min(60.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        if not X or len(X) != len(y):
            raise ValueError("Training data is invalid.")

        features = len(X[0])
        if any(len(row) != features for row in X):
            raise ValueError("Feature dimensions are inconsistent.")

        self.weights = [0.0] * features
        self.bias = 0.0

        n = len(X)

        for _ in range(self.epochs):
            gradient_w = [0.0] * features
            gradient_b = 0.0

            for row, target in zip(X, y):
                linear = self.bias + sum(w * value for w, value in zip(self.weights, row))
                probability = self._sigmoid(linear)
                error = probability - target

                for j, value in enumerate(row):
                    gradient_w[j] += error * value

                gradient_b += error

            for j in range(features):
                gradient_w[j] = gradient_w[j] / n + self.l2 * self.weights[j]
                self.weights[j] -= self.learning_rate * gradient_w[j]

            self.bias -= self.learning_rate * gradient_b / n

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        if not self.weights:
            raise RuntimeError("Model must be fitted before prediction.")

        return [
            self._sigmoid(
                self.bias + sum(w * value for w, value in zip(self.weights, row))
            )
            for row in X
        ]

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        return [int(p >= 0.5) for p in self.predict_proba(X)]


class DecisionTreeClassifier:
    """Small binary decision tree using Gini impurity."""

    def __init__(self, max_depth: int = 3, min_samples_split: int = 4) -> None:
        if max_depth < 1 or min_samples_split < 2:
            raise ValueError("Invalid tree hyperparameters.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: tuple | None = None

    @staticmethod
    def _gini(labels: Sequence[int]) -> float:
        if not labels:
            return 0.0
        positive = sum(labels) / len(labels)
        return 2.0 * positive * (1.0 - positive)

    @staticmethod
    def _majority(labels: Sequence[int]) -> int:
        return 1 if sum(labels) * 2 >= len(labels) else 0

    def _best_split(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        indices: list[int],
    ) -> tuple[int, float, float] | None:
        parent_impurity = self._gini([y[i] for i in indices])
        best: tuple[int, float, float] | None = None

        if parent_impurity == 0.0:
            return None

        features = len(X[0])

        for feature in range(features):
            values = sorted({X[i][feature] for i in indices})
            thresholds = [
                (a + b) / 2.0 for a, b in zip(values, values[1:])
            ]

            for threshold in thresholds:
                left = [i for i in indices if X[i][feature] <= threshold]
                right = [i for i in indices if X[i][feature] > threshold]

                if not left or not right:
                    continue

                impurity = (
                    len(left) * self._gini([y[i] for i in left])
                    + len(right) * self._gini([y[i] for i in right])
                ) / len(indices)

                gain = parent_impurity - impurity

                if best is None or gain > best[2]:
                    best = (feature, threshold, gain)

        return best

    def _build(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        indices: list[int],
        depth: int,
    ) -> tuple:
        labels = [y[i] for i in indices]
        prediction = self._majority(labels)

        if (
            depth >= self.max_depth
            or len(indices) < self.min_samples_split
            or len(set(labels)) == 1
        ):
            return ("leaf", prediction)

        split = self._best_split(X, y, indices)

        if split is None or split[2] <= 0.0:
            return ("leaf", prediction)

        feature, threshold, _ = split
        left = [i for i in indices if X[i][feature] <= threshold]
        right = [i for i in indices if X[i][feature] > threshold]

        return (
            "node",
            feature,
            threshold,
            self._build(X, y, left, depth + 1),
            self._build(X, y, right, depth + 1),
        )

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        if not X or len(X) != len(y):
            raise ValueError("Training data is invalid.")
        self.root = self._build(X, y, list(range(len(X))), 0)

    def _predict_one(self, row: Sequence[float], node: tuple) -> int:
        if node[0] == "leaf":
            return node[1]

        _, feature, threshold, left, right = node
        if row[feature] <= threshold:
            return self._predict_one(row, left)
        return self._predict_one(row, right)

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        if self.root is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return [self._predict_one(row, self.root) for row in X]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        """
        Produces a conservative probability estimate based on the tree class.
        The estimate is intentionally simple because model selection here
        emphasizes validation mechanics rather than probability calibration.
        """
        predictions = self.predict(X)
        return [0.85 if value else 0.15 for value in predictions]


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        raise ValueError("Cannot calculate accuracy on an empty set.")
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def precision(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tp = sum(a == b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    return tp / (tp + fp) if tp + fp else 0.0


def recall(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tp = sum(a == b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    return tp / (tp + fn) if tp + fn else 0.0


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2.0 * p * r / (p + r) if p + r else 0.0


def log_loss(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    if len(y_true) != len(probabilities) or not y_true:
        raise ValueError("Invalid inputs for log loss.")

    total = 0.0
    for target, probability in zip(y_true, probabilities):
        p = min(max(probability, 1e-15), 1.0 - 1e-15)
        total -= target * math.log(p) + (1 - target) * math.log(1 - p)

    return total / len(y_true)


def kfold_indices(
    n_samples: int,
    k: int,
    seed: int,
    stratified_labels: Sequence[int] | None = None,
) -> list[Fold]:
    if n_samples < 2:
        raise ValueError("At least two samples are required.")
    if not 2 <= k <= n_samples:
        raise ValueError("k must be between 2 and the number of samples.")

    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]

    if stratified_labels is None:
        indices = list(range(n_samples))
        rng.shuffle(indices)

        for position, index in enumerate(indices):
            folds[position % k].append(index)
    else:
        if len(stratified_labels) != n_samples:
            raise ValueError("Label count must equal sample count.")

        by_class: dict[int, list[int]] = {}
        for index, label in enumerate(stratified_labels):
            by_class.setdefault(label, []).append(index)

        for class_indices in by_class.values():
            rng.shuffle(class_indices)
            for position, index in enumerate(class_indices):
                folds[position % k].append(index)

        for fold in folds:
            rng.shuffle(fold)

    result: list[Fold] = []

    for validation in folds:
        validation_set = set(validation)
        train = [
            index
            for fold in folds
            if fold is not validation
            for index in fold
            if index not in validation_set
        ]
        result.append(Fold(train, validation))

    if any(not fold.validation_indices for fold in result):
        raise ValueError("A fold contains no validation samples.")

    return result


def split_holdout(
    dataset: Dataset,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    indices = list(range(len(dataset.y)))
    rng.shuffle(indices)

    test_size = max(1, round(len(indices) * test_ratio))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    if len(train_indices) < 2:
        raise ValueError("Training split is too small.")

    train = Dataset(
        [dataset.X[i] for i in train_indices],
        [dataset.y[i] for i in train_indices],
    )
    test = Dataset(
        [dataset.X[i] for i in test_indices],
        [dataset.y[i] for i in test_indices],
    )
    return train, test


def evaluate_model(
    model_factory: Callable[[], object],
    dataset: Dataset,
    folds: Sequence[Fold],
) -> tuple[float, float, float, float, list[float]]:
    f1_scores: list[float] = []
    accuracy_scores: list[float] = []
    losses: list[float] = []

    for fold in folds:
        X_train = [dataset.X[i] for i in fold.train_indices]
        y_train = [dataset.y[i] for i in fold.train_indices]
        X_val = [dataset.X[i] for i in fold.validation_indices]
        y_val = [dataset.y[i] for i in fold.validation_indices]

        model = model_factory()
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)

        f1_scores.append(f1_score(y_val, predictions))
        accuracy_scores.append(accuracy(y_val, predictions))
        losses.append(log_loss(y_val, probabilities))

    return (
        statistics.mean(f1_scores),
        statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
        statistics.mean(accuracy_scores),
        statistics.mean(losses),
        f1_scores,
    )


def generate_dataset(n: int = 180, seed: int = 7) -> Dataset:
    if n < 40:
        raise ValueError("Synthetic dataset is intentionally kept above 40 rows.")

    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[int] = []

    for _ in range(n):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)
        x3 = rng.gauss(0, 1)
        x4 = rng.gauss(0, 1)

        score = (
            1.6 * x1
            - 1.2 * x2
            + 0.7 * x3
            + 0.25 * x4
            + rng.gauss(0, 0.8)
        )

        label = int(score > 0.35)
        X.append([x1, x2, x3, x4])
        y.append(label)

    dataset = Dataset(X, y)
    dataset.validate()
    return dataset


def class_distribution(labels: Sequence[int]) -> dict[int, int]:
    result: dict[int, int] = {}
    for label in labels:
        result[label] = result.get(label, 0) + 1
    return result


def main() -> None:
    seed = 42
    dataset = generate_dataset()

    train, test = split_holdout(dataset, test_ratio=0.20, seed=seed)

    print("ML MODEL SELECTION")
    print("=" * 72)
    print(f"Samples: {len(dataset.y)}")
    print(f"Training samples: {len(train.y)}")
    print(f"Holdout samples: {len(test.y)}")
    print(f"Training class distribution: {class_distribution(train.y)}")
    print(f"Holdout class distribution: {class_distribution(test.y)}")
    print()

    k = 5
    folds = kfold_indices(
        len(train.y),
        k=k,
        seed=seed,
        stratified_labels=train.y,
    )

    print("STRATIFIED K-FOLD VALIDATION")
    print("-" * 72)

    models: list[tuple[str, Callable[[], object]]] = [
        (
            "Logistic Regression",
            lambda: LogisticRegression(
                learning_rate=0.08,
                epochs=600,
                l2=0.02,
            ),
        ),
        (
            "Decision Tree",
            lambda: DecisionTreeClassifier(
                max_depth=4,
                min_samples_split=6,
            ),
        ),
    ]

    results: list[ModelResult] = []

    for name, factory in models:
        mean_f1, std_f1, mean_acc, mean_loss, scores = evaluate_model(
            factory,
            train,
            folds,
        )

        result = ModelResult(
            name=name,
            mean_f1=mean_f1,
            std_f1=std_f1,
            mean_accuracy=mean_acc,
            mean_log_loss=mean_loss,
            scores=scores,
        )
        results.append(result)

        print(f"{name}")
        print(f"  Fold F1: {[round(v, 4) for v in scores]}")
        print(f"  Mean F1: {mean_f1:.4f}")
        print(f"  F1 std:  {std_f1:.4f}")
        print(f"  Accuracy: {mean_acc:.4f}")
        print(f"  Log loss: {mean_loss:.4f}")
        print()

    # Repeated CV estimates the stability of the selected model.
    repetitions = 3
    repeated_scores: list[float] = []

    best = max(results, key=lambda item: (item.mean_f1, -item.std_f1))

    for repetition in range(repetitions):
        repeated_folds = kfold_indices(
            len(train.y),
            k=k,
            seed=seed + repetition,
            stratified_labels=train.y,
        )
        mean_f1, _, _, _, _ = evaluate_model(
            dict(models)[best.name],
            train,
            repeated_folds,
        )
        repeated_scores.append(mean_f1)

    print("REPEATED STRATIFIED K-FOLD")
    print("-" * 72)
    print(f"Selected candidate: {best.name}")
    print(f"Repetition F1: {[round(v, 4) for v in repeated_scores]}")
    print(f"Repeated mean F1: {statistics.mean(repeated_scores):.4f}")
    print(
        f"Repeated F1 std: "
        f"{statistics.stdev(repeated_scores) if len(repeated_scores) > 1 else 0.0:.4f}"
    )
    print()

    # The holdout set is not used during model selection.
    final_model = dict(models)[best.name]()
    final_model.fit(train.X, train.y)

    holdout_predictions = final_model.predict(test.X)
    holdout_probabilities = final_model.predict_proba(test.X)

    print("FINAL HOLDOUT EVALUATION")
    print("-" * 72)
    print(f"Selected model: {best.name}")
    print(f"Accuracy:  {accuracy(test.y, holdout_predictions):.4f}")
    print(f"Precision: {precision(test.y, holdout_predictions):.4f}")
    print(f"Recall:    {recall(test.y, holdout_predictions):.4f}")
    print(f"F1:        {f1_score(test.y, holdout_predictions):.4f}")
    print(f"Log loss:  {log_loss(test.y, holdout_probabilities):.4f}")
    print()

    # Edge-case validation: stratification must preserve class presence.
    for fold_number, fold in enumerate(folds, start=1):
        validation_labels = [train.y[i] for i in fold.validation_indices]
        if set(validation_labels) != {0, 1}:
            raise AssertionError(
                f"Fold {fold_number} lost a class during stratification."
            )

    print("VALIDATION CHECKS")
    print("-" * 72)
    print("All folds contain both classes.")
    print("No holdout record was used during cross-validation.")
    print("Model selection was based only on training-fold validation results.")
    print("Final evaluation was performed once on the untouched holdout set.")


if __name__ == "__main__":
    main()
