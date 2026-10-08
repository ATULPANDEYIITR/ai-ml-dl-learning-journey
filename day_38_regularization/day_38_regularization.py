"""
Regularization: Ridge, Lasso, Elastic Net, and Bias-Variance Analysis

A self-contained implementation using only the Python standard library.
The script builds a small linear-regression framework and demonstrates:

- Why ordinary least squares can overfit
- L2 regularization (Ridge)
- L1 regularization (Lasso)
- Elastic Net
- Feature standardization
- Gradient descent
- Coordinate descent for Lasso and Elastic Net
- Coefficient shrinkage and sparsity
- Bias-variance behavior
- Train/test evaluation
- Cross-validation for selecting regularization strength
- Numerical and modeling edge cases
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean
from typing import Callable, Sequence


Vector = list[float]
Matrix = list[Vector]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal length.")
    return sum(x * y for x, y in zip(a, b))


def mean_squared_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted values must have equal non-zero length.")
    return mean((y - p) ** 2 for y, p in zip(actual, predicted))


def root_mean_squared_error(
    actual: Sequence[float], predicted: Sequence[float]
) -> float:
    return math.sqrt(mean_squared_error(actual, predicted))


def r2_score(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted values must have equal non-zero length.")

    y_bar = mean(actual)
    total = sum((y - y_bar) ** 2 for y in actual)

    if total == 0:
        return 1.0 if all(abs(y - p) < 1e-12 for y, p in zip(actual, predicted)) else 0.0

    residual = sum((y - p) ** 2 for y, p in zip(actual, predicted))
    return 1.0 - residual / total


def train_test_split(
    x: Matrix,
    y: Sequence[float],
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[Matrix, Vector, Matrix, Vector]:
    if not x or len(x) != len(y):
        raise ValueError("X and y must contain the same non-zero number of rows.")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    indices = list(range(len(x)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    test_size = max(1, round(len(x) * test_ratio))
    test_indices = set(indices[:test_size])

    x_train = [list(x[i]) for i in indices if i not in test_indices]
    y_train = [float(y[i]) for i in indices if i not in test_indices]
    x_test = [list(x[i]) for i in indices if i in test_indices]
    y_test = [float(y[i]) for i in indices if i in test_indices]

    return x_train, y_train, x_test, y_test


class StandardScaler:
    """Standardizes each feature to mean zero and standard deviation one."""

    def __init__(self) -> None:
        self.means: Vector = []
        self.scales: Vector = []

    def fit(self, x: Matrix) -> "StandardScaler":
        if not x:
            raise ValueError("Cannot fit a scaler to empty data.")

        width = len(x[0])
        if width == 0:
            raise ValueError("Input must contain at least one feature.")

        if any(len(row) != width for row in x):
            raise ValueError("All rows must have equal feature counts.")

        self.means = [mean(row[j] for row in x) for j in range(width)]
        self.scales = []

        for j in range(width):
            variance = mean((row[j] - self.means[j]) ** 2 for row in x)
            scale = math.sqrt(variance)
            self.scales.append(scale if scale > 1e-12 else 1.0)

        return self

    def transform(self, x: Matrix) -> Matrix:
        if not self.means:
            raise RuntimeError("Scaler has not been fitted.")

        if any(len(row) != len(self.means) for row in x):
            raise ValueError("Input feature count does not match fitted scaler.")

        return [
            [
                (value - self.means[j]) / self.scales[j]
                for j, value in enumerate(row)
            ]
            for row in x
        ]

    def fit_transform(self, x: Matrix) -> Matrix:
        return self.fit(x).transform(x)

    def inverse_coefficients(self, coefficients: Vector) -> Vector:
        if len(coefficients) != len(self.scales):
            raise ValueError("Coefficient count does not match scaler.")
        return [coef / scale for coef, scale in zip(coefficients, self.scales)]


@dataclass
class RegressionResult:
    intercept: float
    coefficients: Vector
    iterations: int
    converged: bool


class OrdinaryLeastSquares:
    """
    Gradient-descent linear regression without regularization.

    The absence of a penalty makes the model useful as a baseline.
    In high-dimensional or highly correlated data, it can produce unstable
    coefficients and low training error without good generalization.
    """

    def __init__(
        self,
        learning_rate: float = 0.03,
        max_iter: int = 5000,
        tolerance: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.result: RegressionResult | None = None

    def fit(self, x: Matrix, y: Sequence[float]) -> "OrdinaryLeastSquares":
        validate_training_data(x, y)

        n = len(x)
        p = len(x[0])
        coefficients = [0.0] * p
        intercept = mean(y)
        previous_loss = float("inf")
        converged = False

        for iteration in range(1, self.max_iter + 1):
            predictions = [
                intercept + dot(row, coefficients)
                for row in x
            ]

            errors = [prediction - target for prediction, target in zip(predictions, y)]

            gradient_intercept = 2.0 * mean(errors)
            gradient_coefficients = [
                2.0 * mean(error * row[j] for error, row in zip(errors, x))
                for j in range(p)
            ]

            intercept -= self.learning_rate * gradient_intercept

            for j in range(p):
                coefficients[j] -= self.learning_rate * gradient_coefficients[j]

            loss = mean(error * error for error in errors)

            if abs(previous_loss - loss) < self.tolerance:
                converged = True
                break

            previous_loss = loss

        self.result = RegressionResult(
            intercept=intercept,
            coefficients=coefficients,
            iterations=iteration,
            converged=converged,
        )
        return self

    def predict(self, x: Matrix) -> Vector:
        if self.result is None:
            raise RuntimeError("Model has not been fitted.")
        return [
            self.result.intercept + dot(row, self.result.coefficients)
            for row in x
        ]


def validate_training_data(x: Matrix, y: Sequence[float]) -> None:
    if not x or not y:
        raise ValueError("Training data cannot be empty.")
    if len(x) != len(y):
        raise ValueError("X and y must have equal row counts.")

    width = len(x[0])
    if width == 0:
        raise ValueError("At least one feature is required.")

    if any(len(row) != width for row in x):
        raise ValueError("Every feature row must have equal length.")

    for row in x:
        if any(not math.isfinite(value) for value in row):
            raise ValueError("Features must contain finite numeric values.")

    if any(not math.isfinite(value) for value in y):
        raise ValueError("Targets must contain finite numeric values.")


def soft_threshold(value: float, threshold: float) -> float:
    """The proximal operation responsible for exact zero coefficients in L1 models."""
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


class RidgeRegression:
    """
    Ridge regression using gradient descent.

    Objective:
        MSE + alpha * sum(beta_j^2)

    The intercept is deliberately excluded from the penalty. Penalizing the
    intercept would make the model depend on the arbitrary origin of y.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        learning_rate: float = 0.01,
        max_iter: int = 10000,
        tolerance: float = 1e-9,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.result: RegressionResult | None = None

    def fit(self, x: Matrix, y: Sequence[float]) -> "RidgeRegression":
        validate_training_data(x, y)

        p = len(x[0])
        coefficients = [0.0] * p
        intercept = mean(y)
        previous_loss = float("inf")
        converged = False

        for iteration in range(1, self.max_iter + 1):
            predictions = [
                intercept + dot(row, coefficients)
                for row in x
            ]
            errors = [prediction - target for prediction, target in zip(predictions, y)]

            gradient_intercept = 2.0 * mean(errors)

            gradient_coefficients = [
                2.0 * mean(error * row[j] for error, row in zip(errors, x))
                + 2.0 * self.alpha * coefficients[j]
                for j in range(p)
            ]

            intercept -= self.learning_rate * gradient_intercept

            for j in range(p):
                coefficients[j] -= self.learning_rate * gradient_coefficients[j]

            mse = mean(error * error for error in errors)
            penalty = self.alpha * sum(coef * coef for coef in coefficients)
            loss = mse + penalty

            if abs(previous_loss - loss) < self.tolerance:
                converged = True
                break

            previous_loss = loss

        self.result = RegressionResult(
            intercept=intercept,
            coefficients=coefficients,
            iterations=iteration,
            converged=converged,
        )
        return self

    def predict(self, x: Matrix) -> Vector:
        if self.result is None:
            raise RuntimeError("Model has not been fitted.")
        return [
            self.result.intercept + dot(row, self.result.coefficients)
            for row in x
        ]


class LassoRegression:
    """
    Lasso regression using coordinate descent.

    Objective:
        MSE + alpha * sum(abs(beta_j))

    The L1 penalty has a kink at zero. Coordinate descent handles this with
    soft-thresholding and can therefore produce exactly zero coefficients.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        max_iter: int = 3000,
        tolerance: float = 1e-7,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.result: RegressionResult | None = None

    def fit(self, x: Matrix, y: Sequence[float]) -> "LassoRegression":
        validate_training_data(x, y)

        n = len(x)
        p = len(x[0])
        coefficients = [0.0] * p
        intercept = mean(y)

        residuals = [
            target - intercept
            for target in y
        ]

        converged = False

        for iteration in range(1, self.max_iter + 1):
            old_coefficients = coefficients.copy()

            intercept += mean(residuals)

            residuals = [
                residual - mean(residuals)
                for residual in residuals
            ]

            for j in range(p):
                column = [row[j] for row in x]

                residuals = [
                    residuals[i] + column[i] * coefficients[j]
                    for i in range(n)
                ]

                rho = sum(column[i] * residuals[i] for i in range(n))
                denominator = sum(value * value for value in column)

                if denominator == 0:
                    coefficients[j] = 0.0
                else:
                    coefficients[j] = soft_threshold(
                        rho,
                        self.alpha * n / 2.0,
                    ) / denominator

                residuals = [
                    residuals[i] - column[i] * coefficients[j]
                    for i in range(n)
                ]

            maximum_change = max(
                abs(new - old)
                for new, old in zip(coefficients, old_coefficients)
            )

            if maximum_change < self.tolerance:
                converged = True
                break

        self.result = RegressionResult(
            intercept=intercept,
            coefficients=coefficients,
            iterations=iteration,
            converged=converged,
        )
        return self

    def predict(self, x: Matrix) -> Vector:
        if self.result is None:
            raise RuntimeError("Model has not been fitted.")
        return [
            self.result.intercept + dot(row, self.result.coefficients)
            for row in x
        ]


class ElasticNetRegression:
    """
    Elastic Net using coordinate descent.

    Objective:
        MSE
        + alpha * l1_ratio * sum(abs(beta_j))
        + alpha * (1 - l1_ratio) * sum(beta_j^2)

    l1_ratio = 1 corresponds to Lasso.
    l1_ratio = 0 corresponds to Ridge-like L2 regularization.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        max_iter: int = 5000,
        tolerance: float = 1e-7,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1.")

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.result: RegressionResult | None = None

    def fit(self, x: Matrix, y: Sequence[float]) -> "ElasticNetRegression":
        validate_training_data(x, y)

        n = len(x)
        p = len(x[0])
        coefficients = [0.0] * p
        intercept = mean(y)

        residuals = [
            target - intercept
            for target in y
        ]

        converged = False

        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)

        for iteration in range(1, self.max_iter + 1):
            old_coefficients = coefficients.copy()

            intercept += mean(residuals)

            residual_mean = mean(residuals)
            residuals = [residual - residual_mean for residual in residuals]

            for j in range(p):
                column = [row[j] for row in x]

                residuals = [
                    residuals[i] + column[i] * coefficients[j]
                    for i in range(n)
                ]

                rho = sum(column[i] * residuals[i] for i in range(n))
                denominator = sum(value * value for value in column) + n * l2

                if denominator == 0:
                    coefficients[j] = 0.0
                else:
                    coefficients[j] = soft_threshold(
                        rho,
                        n * l1 / 2.0,
                    ) / denominator

                residuals = [
                    residuals[i] - column[i] * coefficients[j]
                    for i in range(n)
                ]

            maximum_change = max(
                abs(new - old)
                for new, old in zip(coefficients, old_coefficients)
            )

            if maximum_change < self.tolerance:
                converged = True
                break

        self.result = RegressionResult(
            intercept=intercept,
            coefficients=coefficients,
            iterations=iteration,
            converged=converged,
        )
        return self

    def predict(self, x: Matrix) -> Vector:
        if self.result is None:
            raise RuntimeError("Model has not been fitted.")
        return [
            self.result.intercept + dot(row, self.result.coefficients)
            for row in x
        ]


def generate_regression_dataset(
    samples: int = 160,
    seed: int = 7,
) -> tuple[Matrix, Vector, list[str]]:
    """
    Creates a regression problem with correlated predictors.

    The true process depends strongly on three features. Several additional
    features are noise. Correlated features make coefficient allocation
    ambiguous, which is particularly useful for comparing Ridge and Lasso.
    """
    rng = random.Random(seed)

    names = [
        "temperature",
        "humidity",
        "pressure",
        "advertising",
        "discount",
        "holiday",
        "competitor_price",
        "store_traffic",
        "noise_feature_a",
        "noise_feature_b",
        "noise_feature_c",
        "noise_feature_d",
    ]

    x: Matrix = []
    y: Vector = []

    for _ in range(samples):
        temperature = rng.gauss(22, 4)
        humidity = rng.gauss(60, 12)

        # Pressure is correlated with temperature, creating multicollinearity.
        pressure = 1000 + temperature * 1.8 + rng.gauss(0, 3)

        advertising = rng.uniform(0, 100)
        discount = rng.uniform(0, 30)
        holiday = 1.0 if rng.random() < 0.18 else 0.0
        competitor_price = rng.gauss(50, 7)
        store_traffic = rng.gauss(500, 100)

        noise_features = [rng.gauss(0, 1) for _ in range(4)]

        row = [
            temperature,
            humidity,
            pressure,
            advertising,
            discount,
            holiday,
            competitor_price,
            store_traffic,
            *noise_features,
        ]

        target = (
            80
            + 2.8 * temperature
            - 0.9 * humidity
            + 0.6 * advertising
            + 1.2 * discount
            + 18 * holiday
            - 0.5 * competitor_price
            + 0.04 * store_traffic
            + rng.gauss(0, 8)
        )

        x.append(row)
        y.append(target)

    return x, y, names


def evaluate_model(
    name: str,
    model,
    x_train: Matrix,
    y_train: Vector,
    x_test: Matrix,
    y_test: Vector,
) -> None:
    model.fit(x_train, y_train)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    print(f"\n{name}")
    print(f"  train RMSE : {root_mean_squared_error(y_train, train_predictions):.4f}")
    print(f"  test RMSE  : {root_mean_squared_error(y_test, test_predictions):.4f}")
    print(f"  train R²   : {r2_score(y_train, train_predictions):.4f}")
    print(f"  test R²    : {r2_score(y_test, test_predictions):.4f}")

    result = model.result
    if result is not None:
        zero_count = sum(abs(value) < 1e-8 for value in result.coefficients)
        print(f"  zero coefficients: {zero_count}/{len(result.coefficients)}")
        print(f"  iterations: {result.iterations}")
        print(f"  converged : {result.converged}")


def compare_coefficients(
    names: Sequence[str],
    models: Sequence[tuple[str, object]],
) -> None:
    print("\nCoefficient comparison")

    header = f"{'Feature':<22}"
    for model_name, _ in models:
        header += f"{model_name:>16}"
    print(header)
    print("-" * len(header))

    for index, feature_name in enumerate(names):
        line = f"{feature_name:<22}"

        for _, model in models:
            result = model.result
            coefficient = result.coefficients[index] if result else float("nan")
            line += f"{coefficient:>16.5f}"

        print(line)


def k_fold_indices(
    n_samples: int,
    k: int,
    seed: int = 42,
) -> list[list[int]]:
    if k < 2:
        raise ValueError("k must be at least 2.")
    if k > n_samples:
        raise ValueError("k cannot exceed the number of samples.")

    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)

    folds = [[] for _ in range(k)]

    for position, index in enumerate(indices):
        folds[position % k].append(index)

    return folds


def cross_validate(
    model_factory: Callable[[float], object],
    x: Matrix,
    y: Vector,
    alphas: Sequence[float],
    k: int = 5,
) -> tuple[float, list[tuple[float, float]]]:
    """
    Selects alpha using validation folds.

    Each fold is fitted independently. This avoids evaluating alpha on the
    same observations used to fit the corresponding model.
    """
    folds = k_fold_indices(len(x), k)
    scores: list[tuple[float, float]] = []

    for alpha in alphas:
        fold_errors = []

        for validation_fold in folds:
            validation_set = set(validation_fold)
            training_indices = [
                index
                for index in range(len(x))
                if index not in validation_set
            ]

            x_train = [x[i] for i in training_indices]
            y_train = [y[i] for i in training_indices]
            x_valid = [x[i] for i in validation_fold]
            y_valid = [y[i] for i in validation_fold]

            model = model_factory(alpha)
            model.fit(x_train, y_train)

            predictions = model.predict(x_valid)
            fold_errors.append(
                mean_squared_error(y_valid, predictions)
            )

        scores.append((alpha, mean(fold_errors)))

    best_alpha = min(scores, key=lambda item: item[1])[0]
    return best_alpha, scores


def demonstrate_bias_variance(
    x: Matrix,
    y: Vector,
    seed: int = 19,
) -> None:
    """
    Empirically illustrates the bias-variance trade-off.

    A fixed validation point is predicted by many models trained on different
    random samples. The variance of those predictions estimates instability.
    """
    rng = random.Random(seed)

    candidate_x = x[:1]

    models = {
        "OLS": lambda: OrdinaryLeastSquares(
            learning_rate=0.01,
            max_iter=4000,
        ),
        "Ridge": lambda: RidgeRegression(
            alpha=1.0,
            learning_rate=0.005,
            max_iter=6000,
        ),
        "Lasso": lambda: LassoRegression(
            alpha=0.08,
            max_iter=2500,
        ),
        "ElasticNet": lambda: ElasticNetRegression(
            alpha=0.08,
            l1_ratio=0.5,
            max_iter=3000,
        ),
    }

    print("\nEmpirical prediction variance")

    for name, factory in models.items():
        predictions = []

        for _ in range(30):
            selected = [
                rng.randrange(len(x))
                for _ in range(max(30, len(x) // 2))
            ]

            sample_x = [x[i] for i in selected]
            sample_y = [y[i] for i in selected]

            model = factory()
            model.fit(sample_x, sample_y)
            predictions.append(model.predict(candidate_x)[0])

        prediction_mean = mean(predictions)
        variance = mean(
            (prediction - prediction_mean) ** 2
            for prediction in predictions
        )

        print(
            f"  {name:<12} "
            f"mean prediction={prediction_mean:>10.4f} "
            f"variance={variance:>10.4f}"
        )


def demonstrate_regularization_path(
    x: Matrix,
    y: Vector,
    names: Sequence[str],
) -> None:
    """
    Shows how increasing Lasso alpha changes coefficient sparsity.
    """
    print("\nLasso regularization path")

    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)

    for alpha in [0.005, 0.02, 0.05, 0.1, 0.2, 0.5]:
        model = LassoRegression(
            alpha=alpha,
            max_iter=4000,
        )
        model.fit(scaled_x, y)

        coefficients = model.result.coefficients if model.result else []
        active = [
            name
            for name, coefficient in zip(names, coefficients)
            if abs(coefficient) > 1e-6
        ]

        print(
            f"  alpha={alpha:<6} "
            f"active_features={len(active):<2} "
            f"{', '.join(active)}"
        )


def demonstrate_edge_cases() -> None:
    print("\nValidation and edge cases")

    try:
        RidgeRegression(alpha=-1).fit([[1.0]], [2.0])
    except ValueError as exc:
        print(f"  Negative alpha rejected: {exc}")

    try:
        StandardScaler().fit([[1.0], [2.0]).transform([[1.0]])
    except Exception as exc:
        print(f"  Scaler validation example: {exc}")

    try:
        LassoRegression(alpha=0.1).fit([[1.0], [2.0]], [1.0])
    except ValueError as exc:
        print(f"  Mismatched data rejected: {exc}")

    constant_feature = [[5.0], [5.0], [5.0], [5.0]]
    scaler = StandardScaler()
    transformed = scaler.fit_transform(constant_feature)

    print(f"  Constant feature after scaling: {transformed}")


def main() -> None:
    print("=" * 78)
    print("REGULARIZATION: RIDGE, LASSO, ELASTIC NET")
    print("=" * 78)

    x, y, feature_names = generate_regression_dataset()

    x_train, y_train, x_test, y_test = train_test_split(
        x,
        y,
        test_ratio=0.25,
        seed=42,
    )

    # Standardization is important for regularization because penalties act
    # directly on coefficient magnitudes. Without scaling, a feature measured
    # in large units can be penalized differently from an equivalent feature
    # measured in small units.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print("\nDataset")
    print(f"  observations: {len(x)}")
    print(f"  features: {len(feature_names)}")
    print(f"  training rows: {len(x_train)}")
    print(f"  testing rows: {len(x_test)}")

    ols = OrdinaryLeastSquares(
        learning_rate=0.003,
        max_iter=10000,
    )
    ridge = RidgeRegression(
        alpha=0.8,
        learning_rate=0.003,
        max_iter=12000,
    )
    lasso = LassoRegression(
        alpha=0.08,
        max_iter=5000,
    )
    elastic_net = ElasticNetRegression(
        alpha=0.08,
        l1_ratio=0.5,
        max_iter=5000,
    )

    evaluate_model(
        "Ordinary Least Squares",
        ols,
        x_train_scaled,
        y_train,
        x_test_scaled,
        y_test,
    )

    evaluate_model(
        "Ridge Regression",
        ridge,
        x_train_scaled,
        y_train,
        x_test_scaled,
        y_test,
    )

    evaluate_model(
        "Lasso Regression",
        lasso,
        x_train_scaled,
        y_train,
        x_test_scaled,
        y_test,
    )

    evaluate_model(
        "Elastic Net",
        elastic_net,
        x_train_scaled,
        y_train,
        x_test_scaled,
        y_test,
    )

    compare_coefficients(
        feature_names,
        [
            ("OLS", ols),
            ("Ridge", ridge),
            ("Lasso", lasso),
            ("ElasticNet", elastic_net),
        ],
    )

    print("\nCross-validation for Ridge")

    alpha_grid = [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    best_ridge_alpha, ridge_scores = cross_validate(
        lambda alpha: RidgeRegression(
            alpha=alpha,
            learning_rate=0.003,
            max_iter=10000,
        ),
        x_train_scaled,
        y_train,
        alpha_grid,
        k=5,
    )

    for alpha, mse in ridge_scores:
        print(f"  alpha={alpha:<6} validation MSE={mse:.5f}")

    print(f"  selected Ridge alpha: {best_ridge_alpha}")

    print("\nCross-validation for Elastic Net")

    best_en_alpha, elastic_scores = cross_validate(
        lambda alpha: ElasticNetRegression(
            alpha=alpha,
            l1_ratio=0.5,
            max_iter=5000,
        ),
        x_train_scaled,
        y_train,
        alpha_grid,
        k=5,
    )

    for alpha, mse in elastic_scores:
        print(f"  alpha={alpha:<6} validation MSE={mse:.5f}")

    print(f"  selected Elastic Net alpha: {best_en_alpha}")

    demonstrate_regularization_path(
        x_train,
        y_train,
        feature_names,
    )

    demonstrate_bias_variance(
        x_train_scaled,
        y_train,
    )

    demonstrate_edge_cases()

    print("\nInterpretation")
    print(
        "  Ridge usually keeps correlated predictors together while shrinking "
        "their coefficients toward zero."
    )
    print(
        "  Lasso can remove predictors entirely by producing exact zero "
        "coefficients through its L1 penalty."
    )
    print(
        "  Elastic Net combines L1 sparsity with L2 stabilization, which can "
        "be useful when predictors are numerous and correlated."
    )
    print(
        "  Regularization generally increases bias relative to an unpenalized "
        "model while reducing estimator variance."
    )
    print(
        "  The useful penalty is not necessarily the largest penalty: model "
        "selection must balance underfitting against overfitting."
    )


if __name__ == "__main__":
    main()
