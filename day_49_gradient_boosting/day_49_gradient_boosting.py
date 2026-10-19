"""
Gradient Boosting: weak learners, residual fitting, learning rate, and trees.

Self-contained educational implementation using only the Python standard library.
The implementation builds regression trees as weak learners and combines them
sequentially. Each new tree fits the current residuals rather than the original
target directly.

The script also demonstrates:
- squared-error residual fitting
- shallow decision trees as weak learners
- learning-rate shrinkage
- stochastic row subsampling
- feature subsampling
- early stopping
- validation loss
- prediction
- feature importance
- edge cases and validation
- comparison of learning rates and tree depths
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from random import Random
from typing import List, Optional, Sequence, Tuple


Number = float
Dataset = Sequence[Sequence[Number]]


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean() requires at least one value")
    return sum(values) / len(values)


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")
    if not y_true:
        raise ValueError("MSE requires at least one observation")
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def validate_dataset(X: Dataset, y: Sequence[float]) -> None:
    if not X:
        raise ValueError("X cannot be empty")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")

    width = len(X[0])
    if width == 0:
        raise ValueError("Each row must contain at least one feature")

    for row in X:
        if len(row) != width:
            raise ValueError("All feature rows must have equal width")
        if any(not isinstance(value, (int, float)) for value in row):
            raise TypeError("Features must be numeric")

    if any(not isinstance(value, (int, float)) for value in y):
        raise TypeError("Targets must be numeric")


@dataclass
class TreeNode:
    prediction: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class RegressionTree:
    """
    A small CART-style regression tree.

    For squared-error boosting, a leaf predicts the mean residual assigned to
    that leaf. The split is selected by minimizing the total within-node
    squared error.
    """

    def __init__(
        self,
        max_depth: int = 2,
        min_samples_leaf: int = 2,
        max_features: Optional[int] = None,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive")

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root: Optional[TreeNode] = None
        self.feature_importance_: List[float] = []

    @staticmethod
    def _squared_error(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        center = mean(values)
        return sum((value - center) ** 2 for value in values)

    def _best_split(
        self,
        X: Dataset,
        residuals: Sequence[float],
        indices: Sequence[int],
        features: Sequence[int],
    ) -> Optional[Tuple[int, float, float]]:
        best: Optional[Tuple[int, float, float]] = None
        parent_error = self._squared_error([residuals[i] for i in indices])

        if len(indices) < 2 * self.min_samples_leaf:
            return None

        for feature in features:
            ordered = sorted(indices, key=lambda i: X[i][feature])

            for position in range(
                self.min_samples_leaf,
                len(ordered) - self.min_samples_leaf + 1,
            ):
                left_indices = ordered[:position]
                right_indices = ordered[position:]

                left_value = X[left_indices[-1]][feature]
                right_value = X[right_indices[0]][feature]

                if left_value == right_value:
                    continue

                threshold = (left_value + right_value) / 2.0
                left_error = self._squared_error(
                    [residuals[i] for i in left_indices]
                )
                right_error = self._squared_error(
                    [residuals[i] for i in right_indices]
                )
                improvement = parent_error - left_error - right_error

                if improvement > 1e-12 and (
                    best is None or improvement > best[2]
                ):
                    best = (feature, threshold, improvement)

        return best

    def fit(
        self,
        X: Dataset,
        residuals: Sequence[float],
        rng: Random,
        feature_indices: Optional[Sequence[int]] = None,
    ) -> "RegressionTree":
        validate_dataset(X, residuals)

        n_features = len(X[0])
        self.feature_importance_ = [0.0] * n_features

        if feature_indices is None:
            features = list(range(n_features))
        else:
            features = list(feature_indices)

        if not features:
            raise ValueError("At least one feature is required")

        if self.max_features is not None:
            count = min(self.max_features, len(features))
            features = rng.sample(features, count)

        indices = list(range(len(X)))

        def build(current_indices: List[int], depth: int) -> TreeNode:
            prediction = mean([residuals[i] for i in current_indices])

            if (
                depth >= self.max_depth
                or len(current_indices) < 2 * self.min_samples_leaf
            ):
                return TreeNode(prediction=prediction)

            split = self._best_split(
                X,
                residuals,
                current_indices,
                features,
            )

            if split is None:
                return TreeNode(prediction=prediction)

            feature, threshold, improvement = split

            left_indices = [
                i for i in current_indices if X[i][feature] <= threshold
            ]
            right_indices = [
                i for i in current_indices if X[i][feature] > threshold
            ]

            if (
                len(left_indices) < self.min_samples_leaf
                or len(right_indices) < self.min_samples_leaf
            ):
                return TreeNode(prediction=prediction)

            self.feature_importance_[feature] += improvement

            return TreeNode(
                prediction=prediction,
                feature_index=feature,
                threshold=threshold,
                left=build(left_indices, depth + 1),
                right=build(right_indices, depth + 1),
            )

        self.root = build(indices, 0)
        return self

    def predict_one(self, row: Sequence[float]) -> float:
        if self.root is None:
            raise RuntimeError("Tree has not been fitted")

        node = self.root

        while not node.is_leaf:
            assert node.feature_index is not None
            assert node.threshold is not None
            assert node.left is not None
            assert node.right is not None

            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.prediction

    def predict(self, X: Dataset) -> List[float]:
        if self.root is None:
            raise RuntimeError("Tree has not been fitted")
        return [self.predict_one(row) for row in X]


class GradientBoostingRegressor:
    """
    Gradient boosting for regression using squared error.

    Initial model:
        F_0(x) = mean(y)

    At iteration t:
        residual_i = y_i - F_{t-1}(x_i)

        h_t(x) approximately fits residual_i

        F_t(x) = F_{t-1}(x) + learning_rate * h_t(x)

    The learning rate controls how strongly each weak learner changes the
    ensemble. Smaller values generally require more trees but can reduce the
    risk of allowing one tree to dominate the model.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 2,
        min_samples_leaf: int = 2,
        subsample: float = 1.0,
        max_features: Optional[int] = None,
        random_state: int = 42,
        early_stopping_rounds: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative")
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if early_stopping_rounds is not None and early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds

        self.initial_prediction_: float = 0.0
        self.trees: List[RegressionTree] = []
        self.train_loss_: List[float] = []
        self.validation_loss_: List[float] = []
        self.feature_importances_: List[float] = []
        self.best_iteration_: Optional[int] = None

    def fit(
        self,
        X: Dataset,
        y: Sequence[float],
        validation_data: Optional[Tuple[Dataset, Sequence[float]]] = None,
    ) -> "GradientBoostingRegressor":
        validate_dataset(X, y)

        if validation_data is not None:
            X_val, y_val = validation_data
            validate_dataset(X_val, y_val)
            if len(X_val[0]) != len(X[0]):
                raise ValueError("Training and validation feature counts differ")

        rng = Random(self.random_state)
        n_samples = len(X)
        n_features = len(X[0])

        self.initial_prediction_ = mean(y)
        train_prediction = [self.initial_prediction_] * n_samples

        validation_prediction: Optional[List[float]] = None
        if validation_data is not None:
            validation_prediction = [
                self.initial_prediction_
            ] * len(validation_data[1])

        self.trees = []
        self.train_loss_ = []
        self.validation_loss_ = []
        self.feature_importances_ = [0.0] * n_features

        best_validation_loss = inf
        rounds_without_improvement = 0
        best_tree_count = 0

        for iteration in range(self.n_estimators):
            residuals = [
                target - prediction
                for target, prediction in zip(y, train_prediction)
            ]

            sample_count = max(
                2 * self.min_samples_leaf,
                int(round(self.subsample * n_samples)),
            )
            sample_count = min(sample_count, n_samples)

            if sample_count < 2 * self.min_samples_leaf:
                raise ValueError(
                    "subsample leaves too few observations for min_samples_leaf"
                )

            sampled_indices = rng.sample(range(n_samples), sample_count)

            tree_X = [X[i] for i in sampled_indices]
            tree_residuals = [residuals[i] for i in sampled_indices]

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            )
            tree.fit(
                tree_X,
                tree_residuals,
                rng,
                feature_indices=list(range(n_features)),
            )

            update = tree.predict(X)

            for i in range(n_samples):
                train_prediction[i] += self.learning_rate * update[i]

            self.trees.append(tree)

            train_loss = mse(y, train_prediction)
            self.train_loss_.append(train_loss)

            for feature in range(n_features):
                self.feature_importances_[feature] += (
                    tree.feature_importance_[feature]
                )

            if validation_data is not None:
                assert validation_prediction is not None
                X_val, y_val = validation_data
                val_update = tree.predict(X_val)

                for i in range(len(validation_prediction)):
                    validation_prediction[i] += (
                        self.learning_rate * val_update[i]
                    )

                validation_loss = mse(y_val, validation_prediction)
                self.validation_loss_.append(validation_loss)

                if validation_loss < best_validation_loss - 1e-12:
                    best_validation_loss = validation_loss
                    best_tree_count = len(self.trees)
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if (
                    self.early_stopping_rounds is not None
                    and rounds_without_improvement
                    >= self.early_stopping_rounds
                ):
                    break

        if validation_data is not None and best_tree_count:
            self.best_iteration_ = best_tree_count
            self.trees = self.trees[:best_tree_count]
        else:
            self.best_iteration_ = len(self.trees)

        total_importance = sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ = [
                value / total_importance
                for value in self.feature_importances_
            ]

        return self

    def predict(self, X: Dataset) -> List[float]:
        if not self.trees:
            raise RuntimeError("GradientBoostingRegressor has not been fitted")

        prediction = [self.initial_prediction_] * len(X)

        for tree in self.trees:
            update = tree.predict(X)
            for i in range(len(prediction)):
                prediction[i] += self.learning_rate * update[i]

        return prediction

    def staged_predict(self, X: Dataset):
        if not self.trees:
            raise RuntimeError("Model has not been fitted")

        prediction = [self.initial_prediction_] * len(X)

        for tree in self.trees:
            update = tree.predict(X)
            for i in range(len(prediction)):
                prediction[i] += self.learning_rate * update[i]
            yield prediction.copy()


def make_dataset() -> Tuple[List[List[float]], List[float]]:
    """
    Synthetic house-value data.

    Size, location quality, age, and distance influence price. The nonlinear
    interaction between size and location gives shallow trees useful residual
    patterns to learn sequentially.
    """
    rows = [
        ([900, 2, 30, 18], 145),
        ([1100, 3, 20, 14], 182),
        ([1250, 3, 15, 12], 215),
        ([1500, 4, 10, 9], 285),
        ([1700, 4, 8, 7], 320),
        ([1900, 5, 5, 5], 390),
        ([2100, 5, 4, 4], 430),
        ([2300, 5, 3, 3], 475),
        ([2500, 4, 2, 4], 465),
        ([2700, 5, 1, 2], 520),
        ([1000, 2, 25, 17], 158),
        ([1200, 3, 18, 13], 205),
        ([1400, 3, 12, 11], 238),
        ([1600, 4, 9, 8], 300),
        ([1800, 4, 6, 7], 335),
        ([2000, 5, 4, 5], 410),
        ([2200, 5, 3, 4], 450),
        ([2400, 5, 2, 3], 492),
        ([2600, 4, 1, 4], 480),
        ([2800, 5, 1, 2], 535),
        ([950, 2, 35, 20], 130),
        ([1150, 2, 22, 16], 170),
        ([1350, 3, 17, 13], 215),
        ([1550, 4, 11, 9], 280),
        ([1750, 4, 7, 7], 315),
        ([1950, 5, 5, 5], 395),
        ([2150, 5, 3, 4], 440),
        ([2350, 5, 2, 3], 485),
        ([2550, 4, 2, 4], 470),
        ([2750, 5, 1, 2], 528),
    ]
    X = [features for features, _ in rows]
    y = [target for _, target in rows]
    return X, y


def train_test_split(
    X: Dataset,
    y: Sequence[float],
    test_ratio: float = 0.2,
    seed: int = 7,
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between zero and one")

    rng = Random(seed)
    indices = list(range(len(X)))
    rng.shuffle(indices)

    test_size = max(1, int(round(len(X) * test_ratio)))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = [list(X[i]) for i in train_indices]
    X_test = [list(X[i]) for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


def demonstrate_residual_fitting(
    X: Dataset,
    y: Sequence[float],
) -> None:
    baseline = mean(y)
    residuals = [target - baseline for target in y]

    print("\nResidual fitting mechanics")
    print(f"Initial constant prediction: {baseline:.3f}")
    print("Sample target / prediction / residual:")

    for target, residual in list(zip(y, residuals))[:5]:
        print(
            f"  target={target:7.2f} "
            f"prediction={baseline:7.2f} "
            f"residual={residual:7.2f}"
        )

    tree = RegressionTree(max_depth=1, min_samples_leaf=2)
    tree.fit(X, residuals, Random(42))
    update = tree.predict(X)

    print("After fitting one depth-1 weak learner:")
    for target, residual, correction in list(
        zip(y, residuals, update)
    )[:5]:
        print(
            f"  target={target:7.2f} "
            f"residual={residual:7.2f} "
            f"tree_correction={correction:7.2f}"
        )


def compare_learning_rates(
    X_train: Dataset,
    X_test: Dataset,
    y_train: Sequence[float],
    y_test: Sequence[float],
) -> None:
    print("\nLearning-rate comparison")

    configurations = [
        (0.20, 40),
        (0.05, 120),
        (0.01, 300),
    ]

    for learning_rate, trees in configurations:
        model = GradientBoostingRegressor(
            n_estimators=trees,
            learning_rate=learning_rate,
            max_depth=2,
            min_samples_leaf=2,
            random_state=11,
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        print(
            f"learning_rate={learning_rate:.2f}, "
            f"trees={len(model.trees):3d}, "
            f"test_mse={mse(y_test, prediction):8.3f}"
        )


def compare_tree_depths(
    X_train: Dataset,
    X_test: Dataset,
    y_train: Sequence[float],
    y_test: Sequence[float],
) -> None:
    print("\nWeak-learner depth comparison")

    for depth in [1, 2, 3, 4]:
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=depth,
            min_samples_leaf=2,
            random_state=17,
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        print(
            f"max_depth={depth}, "
            f"trees={len(model.trees):3d}, "
            f"test_mse={mse(y_test, prediction):8.3f}"
        )


def demonstrate_staged_learning(
    model: GradientBoostingRegressor,
    X_test: Dataset,
    y_test: Sequence[float],
) -> None:
    print("\nLoss as trees are added")

    staged = model.staged_predict(X_test)
    checkpoints = {1, 5, 10, 25, 50, len(model.trees)}

    for iteration, prediction in enumerate(staged, start=1):
        if iteration in checkpoints:
            print(
                f"trees={iteration:3d}, "
                f"test_mse={mse(y_test, prediction):8.3f}"
            )


def main() -> None:
    X, y = make_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_ratio=0.2,
        seed=21,
    )

    demonstrate_residual_fitting(X_train, y_train)

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.04,
        max_depth=2,
        min_samples_leaf=2,
        subsample=0.85,
        max_features=3,
        random_state=42,
        early_stopping_rounds=15,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
    )

    predictions = model.predict(X_test)

    print("\nGradient boosting model")
    print(f"Initial prediction: {model.initial_prediction_:.3f}")
    print(f"Trees retained: {len(model.trees)}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Tree depth: {model.max_depth}")
    print(f"Training MSE: {mse(y_train, model.predict(X_train)):.3f}")
    print(f"Test MSE: {mse(y_test, predictions):.3f}")

    print("\nPredictions")
    for features, actual, predicted in zip(X_test, y_test, predictions):
        print(
            f"features={features} "
            f"actual={actual:7.2f} "
            f"predicted={predicted:7.2f}"
        )

    feature_names = [
        "floor_area",
        "location_quality",
        "house_age",
        "distance_to_center",
    ]

    print("\nFeature importance")
    for name, importance in zip(
        feature_names,
        model.feature_importances_,
    ):
        print(f"{name:22s} {importance:.4f}")

    demonstrate_staged_learning(model, X_test, y_test)
    compare_learning_rates(X_train, X_test, y_train, y_test)
    compare_tree_depths(X_train, X_test, y_train, y_test)

    print("\nStochastic boosting")
    stochastic = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=2,
        subsample=0.70,
        random_state=99,
    )
    stochastic.fit(X_train, y_train)
    print(
        f"subsample={stochastic.subsample:.2f}, "
        f"test_mse={mse(y_test, stochastic.predict(X_test)):.3f}"
    )

    print("\nProduction considerations")
    print("Shallow trees are deliberately used as weak learners.")
    print("Learning rate shrinks every tree's contribution.")
    print("More trees compensate for smaller learning rates.")
    print("Deep trees can fit complex residual structure too aggressively.")
    print("Validation loss is useful for detecting excessive boosting.")
    print("Missing-value policy, categorical encoding, leakage prevention,")
    print("monitoring, calibration, and reproducible training must be handled")
    print("explicitly when moving this educational implementation to production.")


if __name__ == "__main__":
    main()
