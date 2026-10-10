"""
Logistic Regression: sigmoid, logits, decision boundaries, and binary classification.

Self-contained implementation using only the Python standard library.
The implementation covers:
- Binary classification data
- Logits and sigmoid probabilities
- Binary cross-entropy loss
- Gradient descent
- Decision boundaries
- Classification metrics
- Probability thresholds
- Numerical stability
- L2 regularization
- Feature standardization
- Prediction and model validation
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


Vector = list[float]
Matrix = list[Vector]


def sigmoid(z: float) -> float:
    """Compute sigmoid safely for both large positive and negative logits."""
    if z >= 0:
        exp_neg = math.exp(-z)
        return 1.0 / (1.0 + exp_neg)

    exp_pos = math.exp(z)
    return exp_pos / (1.0 + exp_pos)


def log_sigmoid(z: float) -> float:
    """Numerically stable log(sigmoid(z))."""
    if z >= 0:
        return -math.log1p(math.exp(-z))
    return z - math.log1p(math.exp(z))


def stable_log_one_minus_sigmoid(z: float) -> float:
    """Numerically stable log(1 - sigmoid(z))."""
    if z >= 0:
        return -z - math.log1p(math.exp(-z))
    return -math.log1p(math.exp(z))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal lengths.")
    return sum(x * y for x, y in zip(a, b))


def add_intercept(X: Matrix) -> Matrix:
    return [[1.0, *row] for row in X]


def validate_binary_labels(y: Sequence[int]) -> None:
    if not y:
        raise ValueError("The label sequence cannot be empty.")
    if any(label not in (0, 1) for label in y):
        raise ValueError("Binary logistic regression requires labels 0 and 1.")


def validate_matrix(X: Matrix) -> None:
    if not X:
        raise ValueError("Feature matrix cannot be empty.")
    width = len(X[0])
    if width == 0:
        raise ValueError("Feature rows cannot be empty.")
    if any(len(row) != width for row in X):
        raise ValueError("All feature rows must have equal length.")


@dataclass
class Standardizer:
    means: Vector | None = None
    scales: Vector | None = None

    def fit(self, X: Matrix) -> "Standardizer":
        validate_matrix(X)
        width = len(X[0])
        self.means = [
            sum(row[j] for row in X) / len(X)
            for j in range(width)
        ]

        scales: Vector = []
        for j in range(width):
            variance = sum(
                (row[j] - self.means[j]) ** 2 for row in X
            ) / len(X)
            scale = math.sqrt(variance)
            # A constant feature must not create a division-by-zero error.
            scales.append(scale if scale > 1e-12 else 1.0)

        self.scales = scales
        return self

    def transform(self, X: Matrix) -> Matrix:
        if self.means is None or self.scales is None:
            raise RuntimeError("Standardizer must be fitted before transform().")
        validate_matrix(X)
        if len(X[0]) != len(self.means):
            raise ValueError("Feature count differs from fitted data.")

        return [
            [
                (value - self.means[j]) / self.scales[j]
                for j, value in enumerate(row)
            ]
            for row in X
        ]

    def fit_transform(self, X: Matrix) -> Matrix:
        return self.fit(X).transform(X)


@dataclass
class TrainingHistory:
    losses: list[float]
    accuracies: list[float]


class LogisticRegression:
    """
    Binary logistic regression trained with batch gradient descent.

    The model computes:
        z = w.x + b
        p = sigmoid(z)

    The decision rule is:
        class 1 when p >= threshold
        class 0 otherwise

    L2 regularization is applied to feature weights, not the intercept.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 3000,
        l2_strength: float = 0.0,
        threshold: float = 0.5,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if l2_strength < 0:
            raise ValueError("l2_strength cannot be negative.")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1.")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_strength = l2_strength
        self.threshold = threshold
        self.weights: Vector = []
        self.bias = 0.0
        self.history = TrainingHistory([], [])

    def _logit(self, row: Sequence[float]) -> float:
        return self.bias + dot(self.weights, row)

    def logits(self, X: Matrix) -> Vector:
        self._check_fitted()
        return [self._logit(row) for row in X]

    def predict_proba(self, X: Matrix) -> Vector:
        return [sigmoid(z) for z in self.logits(X)]

    def _loss(self, X: Matrix, y: Sequence[int]) -> float:
        probabilities = self.predict_proba(X)
        n = len(y)

        loss = 0.0
        for label, probability in zip(y, probabilities):
            # Clamping is useful when probabilities are later used directly
            # in ordinary logarithms. The gradient itself remains p-y.
            p = min(max(probability, 1e-15), 1.0 - 1e-15)
            loss -= label * math.log(p) + (1 - label) * math.log(1 - p)

        loss /= n

        if self.l2_strength:
            loss += (
                self.l2_strength
                / (2.0 * n)
                * sum(weight * weight for weight in self.weights)
            )

        return loss

    def _stable_loss(self, X: Matrix, y: Sequence[int]) -> float:
        """Compute cross-entropy directly from logits to avoid probability underflow."""
        n = len(y)
        loss = 0.0

        for row, label in zip(X, y):
            z = self._logit(row)

            # BCE from logits:
            # max(z, 0) - y*z + log(1 + exp(-abs(z)))
            loss += max(z, 0.0) - label * z + math.log1p(math.exp(-abs(z)))

        loss /= n

        if self.l2_strength:
            loss += (
                self.l2_strength
                / (2.0 * n)
                * sum(weight * weight for weight in self.weights)
            )

        return loss

    def fit(self, X: Matrix, y: Sequence[int]) -> TrainingHistory:
        validate_matrix(X)
        validate_binary_labels(y)

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        n_samples = len(X)
        n_features = len(X[0])

        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.history = TrainingHistory([], [])

        for epoch in range(self.epochs):
            gradients = [0.0] * n_features
            bias_gradient = 0.0

            for row, label in zip(X, y):
                z = self._logit(row)
                probability = sigmoid(z)
                error = probability - label

                for j, value in enumerate(row):
                    gradients[j] += error * value

                bias_gradient += error

            regularization_scale = self.l2_strength / n_samples

            for j in range(n_features):
                gradients[j] = (
                    gradients[j] / n_samples
                    + regularization_scale * self.weights[j]
                )

            bias_gradient /= n_samples

            for j in range(n_features):
                self.weights[j] -= self.learning_rate * gradients[j]

            self.bias -= self.learning_rate * bias_gradient

            if epoch == 0 or (epoch + 1) % 100 == 0 or epoch == self.epochs - 1:
                loss = self._stable_loss(X, y)
                accuracy = self.score(X, y)
                self.history.losses.append(loss)
                self.history.accuracies.append(accuracy)

        return self.history

    def predict(self, X: Matrix) -> list[int]:
        return [
            int(probability >= self.threshold)
            for probability in self.predict_proba(X)
        ]

    def score(self, X: Matrix, y: Sequence[int]) -> float:
        validate_binary_labels(y)
        predictions = self.predict(X)
        if len(predictions) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        return sum(p == actual for p, actual in zip(predictions, y)) / len(y)

    def decision_boundary_2d(self) -> str:
        """
        Return the boundary equation for a two-feature model.

        The boundary occurs at p = 0.5, which means:
            sigmoid(z) = 0.5
            z = 0

        Therefore:
            w0*x + w1*y + b = 0
        """
        self._check_fitted()
        if len(self.weights) != 2:
            raise ValueError("A two-feature model is required.")

        w1, w2 = self.weights

        if abs(w2) < 1e-12:
            if abs(w1) < 1e-12:
                return "No meaningful two-dimensional boundary."
            x_value = -self.bias / w1
            return f"Vertical boundary: x = {x_value:.4f}"

        slope = -w1 / w2
        intercept = -self.bias / w2
        return f"Decision boundary: y = {slope:.4f}x + {intercept:.4f}"

    def _check_fitted(self) -> None:
        if not self.weights:
            raise RuntimeError("Model has not been fitted.")


def make_transaction_dataset(
    seed: int = 42,
    samples: int = 240,
) -> tuple[Matrix, list[int]]:
    """
    Generate a fraud-screening dataset.

    Features:
    - transaction amount in hundreds
    - account age in months

    Fraud probability increases with large transaction amounts
    and decreases with account age.
    """
    rng = random.Random(seed)
    X: Matrix = []
    y: list[int] = []

    for _ in range(samples):
        amount = rng.uniform(5.0, 250.0)
        account_age = rng.uniform(1.0, 120.0)

        # A nonlinear-looking but intentionally learnable boundary is created
        # through a noisy latent score.
        latent_score = (
            0.035 * amount
            - 0.028 * account_age
            - 1.2
            + rng.gauss(0.0, 1.0)
        )

        label = int(latent_score > 0.0)
        X.append([amount, account_age])
        y.append(label)

    return X, y


def train_test_split(
    X: Matrix,
    y: Sequence[int],
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[Matrix, list[int], Matrix, list[int]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")

    if len(X) != len(y):
        raise ValueError("X and y must have equal lengths.")

    indices = list(range(len(X)))
    random.Random(seed).shuffle(indices)

    test_size = max(1, int(len(X) * test_ratio))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, y_train, X_test, y_test


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths.")

    validate_binary_labels(y_true)
    validate_binary_labels(y_pred)

    return {
        "true_positive": sum(
            actual == 1 and predicted == 1
            for actual, predicted in zip(y_true, y_pred)
        ),
        "true_negative": sum(
            actual == 0 and predicted == 0
            for actual, predicted in zip(y_true, y_pred)
        ),
        "false_positive": sum(
            actual == 0 and predicted == 1
            for actual, predicted in zip(y_true, y_pred)
        ),
        "false_negative": sum(
            actual == 1 and predicted == 0
            for actual, predicted in zip(y_true, y_pred)
        ),
    }


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred)

    tp = matrix["true_positive"]
    tn = matrix["true_negative"]
    fp = matrix["false_positive"]
    fn = matrix["false_negative"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(y_true)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def demonstrate_sigmoid_and_logits() -> None:
    print("\n=== Logits and Sigmoid ===")

    values = [-8.0, -2.0, 0.0, 2.0, 8.0]

    for z in values:
        print(f"logit={z:6.2f} -> probability={sigmoid(z):.6f}")

    print(
        "\nA logit of zero maps to probability 0.5. "
        "That is why a 0.5 classification threshold corresponds "
        "to a decision boundary at z = 0."
    )


def demonstrate_manual_probability() -> None:
    print("\n=== Manual Logistic Regression Calculation ===")

    weights = [0.8, -0.5]
    bias = -1.0
    features = [2.0, 1.0]

    z = dot(weights, features) + bias
    probability = sigmoid(z)

    print(f"features = {features}")
    print(f"logit = {z:.4f}")
    print(f"P(class=1) = {probability:.4f}")
    print(f"class at threshold 0.5 = {int(probability >= 0.5)}")


def demonstrate_model() -> None:
    print("\n=== Fraud Screening Model ===")

    X, y = make_transaction_dataset()
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    standardizer = Standardizer()
    X_train_scaled = standardizer.fit_transform(X_train)
    X_test_scaled = standardizer.transform(X_test)

    model = LogisticRegression(
        learning_rate=0.08,
        epochs=2500,
        l2_strength=0.15,
        threshold=0.5,
    )

    history = model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    metrics = classification_report(y_test, predictions)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Learned weights: {[round(w, 4) for w in model.weights]}")
    print(f"Learned bias: {model.bias:.4f}")
    print(f"Decision boundary: {model.decision_boundary_2d()}")

    print("\nTest metrics:")
    for name, value in metrics.items():
        print(f"{name:12s}: {value:.4f}")

    print("\nRecent training checkpoints:")
    for loss, accuracy in zip(
        history.losses[-5:],
        history.accuracies[-5:],
    ):
        print(f"loss={loss:.6f}, accuracy={accuracy:.4f}")

    print("\nExample transactions:")
    examples = [
        [225.0, 3.0],
        [45.0, 72.0],
        [150.0, 18.0],
    ]

    scaled_examples = standardizer.transform(examples)

    for original, scaled in zip(examples, scaled_examples):
        probability = model.predict_proba([scaled])[0]
        predicted_class = int(probability >= model.threshold)
        print(
            f"transaction={original}, "
            f"fraud_probability={probability:.4f}, "
            f"class={predicted_class}"
        )

    print("\nThreshold comparison:")
    probabilities = model.predict_proba(X_test_scaled)

    for threshold in (0.30, 0.50, 0.70, 0.90):
        threshold_predictions = [
            int(probability >= threshold)
            for probability in probabilities
        ]
        threshold_metrics = classification_report(
            y_test,
            threshold_predictions,
        )
        print(
            f"threshold={threshold:.2f} "
            f"precision={threshold_metrics['precision']:.3f} "
            f"recall={threshold_metrics['recall']:.3f} "
            f"f1={threshold_metrics['f1']:.3f}"
        )


def demonstrate_edge_cases() -> None:
    print("\n=== Edge Cases and Numerical Stability ===")

    for z in (-1000.0, -100.0, 0.0, 100.0, 1000.0):
        print(f"sigmoid({z:7.1f}) = {sigmoid(z):.12f}")

    try:
        LogisticRegression(threshold=1.0)
    except ValueError as error:
        print(f"Invalid threshold rejected: {error}")

    try:
        validate_binary_labels([0, 1, 2])
    except ValueError as error:
        print(f"Invalid label rejected: {error}")

    try:
        LogisticRegression().fit([[1.0], [2.0]], [0])
    except ValueError as error:
        print(f"Feature/label mismatch rejected: {error}")


def demonstrate_regularization() -> None:
    print("\n=== L2 Regularization Comparison ===")

    X, y = make_transaction_dataset(seed=7, samples=180)
    X_train, y_train, X_test, y_test = train_test_split(X, y, seed=7)

    standardizer = Standardizer()
    X_train = standardizer.fit_transform(X_train)
    X_test = standardizer.transform(X_test)

    for strength in (0.0, 0.1, 1.0):
        model = LogisticRegression(
            learning_rate=0.08,
            epochs=2000,
            l2_strength=strength,
        )
        model.fit(X_train, y_train)
        metrics = classification_report(y_test, model.predict(X_test))
        weight_magnitude = math.sqrt(
            sum(weight * weight for weight in model.weights)
        )

        print(
            f"l2={strength:.1f} "
            f"weight_norm={weight_magnitude:.4f} "
            f"test_f1={metrics['f1']:.4f}"
        )


def main() -> None:
    demonstrate_sigmoid_and_logits()
    demonstrate_manual_probability()
    demonstrate_model()
    demonstrate_edge_cases()
    demonstrate_regularization()


if __name__ == "__main__":
    main()
