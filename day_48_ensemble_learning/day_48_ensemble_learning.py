"""
Ensemble Learning: Voting, Bagging, Boosting, Stacking, and Blending.

Self-contained demonstrations using the Python standard library.
Includes train/test separation, out-of-fold predictions, model evaluation,
bootstrap sampling, weighted voting, and gradient-based boosting.

Run:
    python ensemble_learning.py
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class Example:
    features: tuple[float, ...]
    label: int


def sigmoid(value: float) -> float:
    """Numerically stable logistic transformation."""
    if value >= 0:
        z = math.exp(-min(value, 700))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700))
    return z / (1.0 + z)


def make_dataset(seed: int = 17) -> list[Example]:
    """Create a deterministic binary classification dataset."""
    rng = random.Random(seed)
    examples = []

    for _ in range(240):
        x1 = rng.uniform(-3.0, 3.0)
        x2 = rng.uniform(-3.0, 3.0)

        # A nonlinear boundary gives different models complementary strengths.
        score = x1 * x1 + 0.7 * x2 - 1.5
        score += rng.gauss(0, 0.9)
        label = int(score > 0)

        examples.append(Example((x1, x2), label))

    rng.shuffle(examples)
    return examples


def train_test_split(
    examples: Sequence[Example],
    test_fraction: float = 0.25,
) -> tuple[list[Example], list[Example]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    split = int(len(examples) * (1 - test_fraction))
    if split == 0 or split == len(examples):
        raise ValueError("Both partitions must contain examples.")

    return list(examples[:split]), list(examples[split:])


def accuracy(actual: Sequence[int], predicted: Sequence[int]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Labels must be non-empty and equally sized.")
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def log_loss(actual: Sequence[int], probabilities: Sequence[float]) -> float:
    """Binary cross-entropy; clipping avoids log(0)."""
    if len(actual) != len(probabilities) or not actual:
        raise ValueError("Labels and probabilities must have equal nonzero length.")

    total = 0.0
    for label, probability in zip(actual, probabilities):
        if label not in (0, 1):
            raise ValueError("Binary labels must be 0 or 1.")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probabilities must lie between 0 and 1.")

        p = min(max(probability, 1e-15), 1 - 1e-15)
        total -= label * math.log(p) + (1 - label) * math.log(1 - p)

    return total / len(actual)


class LogisticRegression:
    """L2-regularized logistic regression trained with batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.15,
        epochs: int = 500,
        regularization: float = 0.01,
    ):
        if learning_rate <= 0 or epochs < 1 or regularization < 0:
            raise ValueError("Invalid logistic regression configuration.")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.weights: list[float] = []
        self.bias = 0.0

    def fit(self, examples: Sequence[Example]) -> "LogisticRegression":
        if not examples:
            raise ValueError("Training data cannot be empty.")

        dimension = len(examples[0].features)
        if dimension == 0 or any(
            len(row.features) != dimension or row.label not in (0, 1)
            for row in examples
        ):
            raise ValueError("Inconsistent features or invalid labels.")

        self.weights = [0.0] * dimension
        self.bias = 0.0
        n = len(examples)

        for _ in range(self.epochs):
            weight_gradients = [0.0] * dimension
            bias_gradient = 0.0

            for example in examples:
                score = self.bias + sum(
                    w * x for w, x in zip(self.weights, example.features)
                )
                error = sigmoid(score) - example.label

                for j, value in enumerate(example.features):
                    weight_gradients[j] += error * value

                bias_gradient += error

            for j in range(dimension):
                gradient = weight_gradients[j] / n
                gradient += self.regularization * self.weights[j]
                self.weights[j] -= self.learning_rate * gradient

            self.bias -= self.learning_rate * bias_gradient / n

        return self

    def predict_proba_one(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("Feature dimension differs from training data.")

        score = self.bias + sum(
            w * x for w, x in zip(self.weights, features)
        )
        return sigmoid(score)

    def predict_proba(self, examples: Sequence[Example]) -> list[float]:
        return [self.predict_proba_one(row.features) for row in examples]

    def predict(self, examples: Sequence[Example]) -> list[int]:
        return [int(p >= 0.5) for p in self.predict_proba(examples)]


class DecisionStump:
    """
    One-level decision tree.

    A threshold is selected to minimize weighted classification error.
    This weak learner is useful for demonstrating bagging and boosting.
    """

    def __init__(self):
        self.feature = 0
        self.threshold = 0.0
        self.left_label = 0
        self.right_label = 1

    def fit(
        self,
        examples: Sequence[Example],
        weights: Sequence[float] | None = None,
    ) -> "DecisionStump":
        if not examples:
            raise ValueError("Training data cannot be empty.")

        n = len(examples)
        if weights is None:
            weights = [1.0 / n] * n

        if len(weights) != n or any(w < 0 for w in weights):
            raise ValueError("Weights must be nonnegative and match the data.")

        if sum(weights) <= 0:
            raise ValueError("At least one sample weight must be positive.")

        dimension = len(examples[0].features)
        if dimension == 0 or any(
            len(row.features) != dimension or row.label not in (0, 1)
            for row in examples
        ):
            raise ValueError("Inconsistent features or invalid labels.")

        best_error = float("inf")

        for feature in range(dimension):
            values = sorted({row.features[feature] for row in examples})
            thresholds = [
                (values[i] + values[i + 1]) / 2
                for i in range(len(values) - 1)
            ]

            if not thresholds:
                thresholds = [values[0]]

            for threshold in thresholds:
                for left_label, right_label in ((0, 1), (1, 0)):
                    error = 0.0

                    for row, weight in zip(examples, weights):
                        predicted = (
                            left_label
                            if row.features[feature] <= threshold
                            else right_label
                        )
                        if predicted != row.label:
                            error += weight

                    if error < best_error:
                        best_error = error
                        self.feature = feature
                        self.threshold = threshold
                        self.left_label = left_label
                        self.right_label = right_label

        return self

    def predict_one(self, features: Sequence[float]) -> int:
        return (
            self.left_label
            if features[self.feature] <= self.threshold
            else self.right_label
        )

    def predict(self, examples: Sequence[Example]) -> list[int]:
        return [self.predict_one(row.features) for row in examples]


def bootstrap_sample(
    examples: Sequence[Example],
    rng: random.Random,
) -> list[Example]:
    """Sample n observations with replacement."""
    if not examples:
        raise ValueError("Cannot bootstrap an empty dataset.")
    return [rng.choice(examples) for _ in range(len(examples))]


class BaggingClassifier:
    """Bootstrap aggregation with independently trained decision stumps."""

    def __init__(self, n_estimators: int = 31, seed: int = 23):
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        self.n_estimators = n_estimators
        self.seed = seed
        self.models: list[DecisionStump] = []

    def fit(self, examples: Sequence[Example]) -> "BaggingClassifier":
        if not examples:
            raise ValueError("Training data cannot be empty.")

        rng = random.Random(self.seed)
        self.models = []

        for _ in range(self.n_estimators):
            sample = bootstrap_sample(examples, rng)
            model = DecisionStump().fit(sample)
            self.models.append(model)

        return self

    def predict(self, examples: Sequence[Example]) -> list[int]:
        if not self.models:
            raise RuntimeError("Fit the bagging classifier before prediction.")

        predictions = [
            model.predict(examples) for model in self.models
        ]

        return [
            int(sum(model_predictions[i] for model_predictions in predictions)
                > len(predictions) / 2)
            if sum(model_predictions[i] for model_predictions in predictions)
            != len(predictions) / 2
            else predictions[0][i]
            for i in range(len(examples))
        ]


class AdaBoostClassifier:
    """
    Binary AdaBoost with decision stumps.

    Each round increases the relative importance of incorrectly classified
    training examples. Perfect weak learners terminate the fitting process.
    """

    def __init__(self, n_estimators: int = 25):
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        self.n_estimators = n_estimators
        self.models: list[DecisionStump] = []
        self.alphas: list[float] = []

    def fit(self, examples: Sequence[Example]) -> "AdaBoostClassifier":
        if not examples:
            raise ValueError("Training data cannot be empty.")
        if any(row.label not in (0, 1) for row in examples):
            raise ValueError("AdaBoost requires binary labels.")

        n = len(examples)
        sample_weights = [1.0 / n] * n
        self.models = []
        self.alphas = []

        for _ in range(self.n_estimators):
            model = DecisionStump().fit(examples, sample_weights)
            predictions = model.predict(examples)

            error = sum(
                weight
                for weight, row, predicted in zip(
                    sample_weights, examples, predictions
                )
                if predicted != row.label
            )

            if error >= 0.5 - 1e-12:
                break

            if error <= 1e-12:
                self.models.append(model)
                self.alphas.append(10.0)
                break

            alpha = 0.5 * math.log((1.0 - error) / error)
            self.models.append(model)
            self.alphas.append(alpha)

            updated = []
            for weight, row, predicted in zip(
                sample_weights, examples, predictions
            ):
                sign = 1 if row.label == predicted else -1
                updated.append(weight * math.exp(-alpha * sign))

            normalizer = sum(updated)
            if normalizer <= 0 or not math.isfinite(normalizer):
                raise ArithmeticError("Invalid AdaBoost weight normalization.")

            sample_weights = [weight / normalizer for weight in updated]

        if not self.models:
            raise RuntimeError("No useful weak learner was found.")

        return self

    def decision_function_one(self, features: Sequence[float]) -> float:
        return sum(
            alpha * (1 if model.predict_one(features) == 1 else -1)
            for model, alpha in zip(self.models, self.alphas)
        )

    def predict_proba_one(self, features: Sequence[float]) -> float:
        # This monotonic transformation produces scores, not calibrated
        # probabilities. Calibration requires a separate validation procedure.
        return sigmoid(2.0 * self.decision_function_one(features))

    def predict(self, examples: Sequence[Example]) -> list[int]:
        return [
            int(self.decision_function_one(row.features) >= 0)
            for row in examples
        ]


def voting_predictions(
    model_predictions: Sequence[Sequence[int]],
    weights: Sequence[float] | None = None,
) -> list[int]:
    """Hard voting, with optional nonnegative model weights."""
    if not model_predictions:
        raise ValueError("At least one model is required.")

    size = len(model_predictions[0])
    if any(len(predictions) != size for predictions in model_predictions):
        raise ValueError("Every model must predict the same examples.")

    if any(label not in (0, 1) for predictions in model_predictions
           for label in predictions):
        raise ValueError("Voting predictions must be binary.")

    if weights is None:
        weights = [1.0] * len(model_predictions)

    if len(weights) != len(model_predictions):
        raise ValueError("One weight is required per model.")

    if any(weight < 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("Voting weights must be finite and nonnegative.")

    if sum(weights) == 0:
        raise ValueError("At least one voting weight must be positive.")

    result = []
    for i in range(size):
        positive_weight = sum(
            weight
            for weight, predictions in zip(weights, model_predictions)
            if predictions[i] == 1
        )
        negative_weight = sum(weights) - positive_weight
        result.append(int(positive_weight > negative_weight))

    return result


def stratified_folds(
    examples: Sequence[Example],
    n_folds: int,
    seed: int = 41,
) -> list[list[int]]:
    """Create folds while distributing each class across folds."""
    if n_folds < 2 or n_folds > len(examples):
        raise ValueError("n_folds must be between 2 and dataset size.")

    by_class: dict[int, list[int]] = {}
    for index, row in enumerate(examples):
        by_class.setdefault(row.label, []).append(index)

    rng = random.Random(seed)
    folds = [[] for _ in range(n_folds)]

    for indices in by_class.values():
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            folds[position % n_folds].append(index)

    for fold in folds:
        rng.shuffle(fold)

    return folds


def out_of_fold_predictions(
    examples: Sequence[Example],
    n_folds: int = 5,
) -> tuple[list[int], list[int]]:
    """
    Return out-of-fold predictions and their corresponding row indices.

    Each row is predicted by a model that did not train on that row.
    These predictions can safely form training features for a stacking
    meta-model, provided preprocessing is also fitted inside each fold.
    """
    folds = stratified_folds(examples, n_folds)
    predictions = [-1] * len(examples)

    for validation_indices in folds:
        validation_set = set(validation_indices)
        training = [
            row for index, row in enumerate(examples)
            if index not in validation_set
        ]

        validation = [examples[index] for index in validation_indices]

        logistic = LogisticRegression().fit(training)
        stump = DecisionStump().fit(training)

        logistic_predictions = logistic.predict(validation)
        stump_predictions = stump.predict(validation)

        for index, logistic_label, stump_label in zip(
            validation_indices, logistic_predictions, stump_predictions
        ):
            predictions[index] = int(
                logistic_label + stump_label >= 1
            )

    if any(prediction not in (0, 1) for prediction in predictions):
        raise RuntimeError("Some out-of-fold predictions are missing.")

    return predictions, list(range(len(examples)))


def demonstrate() -> None:
    dataset = make_dataset()
    training, testing = train_test_split(dataset)

    actual = [row.label for row in testing]

    logistic = LogisticRegression().fit(training)
    bagging = BaggingClassifier(n_estimators=41).fit(training)
    boosting = AdaBoostClassifier(n_estimators=30).fit(training)
    stump = DecisionStump().fit(training)

    model_predictions = [
        logistic.predict(testing),
        stump.predict(testing),
        bagging.predict(testing),
        boosting.predict(testing),
    ]

    print("Ensemble Learning Demonstration")
    print("=" * 38)
    print(f"Training examples: {len(training)}")
    print(f"Testing examples:  {len(testing)}")
    print()

    names = ["Logistic regression", "Decision stump", "Bagging", "AdaBoost"]

    for name, predictions in zip(names, model_predictions):
        print(f"{name:22s} accuracy={accuracy(actual, predictions):.3f}")

    print("\nHard voting")
    hard_vote = voting_predictions(model_predictions)
    print(f"Accuracy: {accuracy(actual, hard_vote):.3f}")

    print("\nWeighted voting")
    weighted_vote = voting_predictions(
        model_predictions,
        weights=[2.0, 0.5, 1.5, 1.5],
    )
    print(f"Accuracy: {accuracy(actual, weighted_vote):.3f}")

    print("\nProbability averaging")
    logistic_probabilities = logistic.predict_proba(testing)
    boosting_probabilities = [
        boosting.predict_proba_one(row.features) for row in testing
    ]
    averaged_probabilities = [
        (a + b) / 2
        for a, b in zip(logistic_probabilities, boosting_probabilities)
    ]
    print(f"Logistic log loss: {log_loss(actual, logistic_probabilities):.4f}")
    print(f"Blended log loss:  {log_loss(actual, averaged_probabilities):.4f}")

    print("\nOut-of-fold stacking demonstration")
    oof_predictions, row_indices = out_of_fold_predictions(training)
    oof_actual = [training[index].label for index in row_indices]
    print(f"OOF rows: {len(oof_predictions)}")
    print(f"OOF accuracy: {accuracy(oof_actual, oof_predictions):.3f}")

    print("\nInterpretation")
    print("Voting combines predictions from separately fitted models.")
    print("Bagging reduces instability by fitting models on bootstrap samples.")
    print("Boosting focuses later weak learners on previous errors.")
    print("Stacking learns a meta-model from out-of-fold base predictions.")
    print("Blending learns or selects a combination using a held-out validation set.")
    print("Evaluate every ensemble on data excluded from model selection.")


if __name__ == "__main__":
    demonstrate()
