#!/usr/bin/env python3
"""
Model Interpretability
Feature Importance, Permutation Importance, and Partial Dependence.

A self-contained regression interpretability implementation using
Python standard-library modules only.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Sequence


class InterpretabilityError(Exception):
    """Base exception for interpretability operations."""


class ValidationError(InterpretabilityError):
    """Raised when model inputs are invalid."""


class Metric(Enum):
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"


@dataclass(frozen=True)
class Feature:
    name: str
    description: str
    unit: str

    def __post_init__(self):
        if not self.name.strip():
            raise ValidationError("Feature name cannot be empty.")


@dataclass
class EvaluationResult:
    mse: float
    mae: float
    rmse: float
    sample_count: int


@dataclass
class ImportanceResult:
    feature: str
    score: float
    normalized_score: float
    method: str


@dataclass
class PartialDependenceResult:
    feature: str
    values: list[float]
    predictions: list[float]


def validate_dataset(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    expected_features: int | None = None,
) -> None:
    if not X or not y:
        raise ValidationError("Dataset cannot be empty.")

    if len(X) != len(y):
        raise ValidationError("Feature and target counts differ.")

    feature_count = len(X[0])

    if feature_count == 0:
        raise ValidationError("At least one feature is required.")

    if expected_features is not None and feature_count != expected_features:
        raise ValidationError("Unexpected feature count.")

    for row in X:
        if len(row) != feature_count:
            raise ValidationError("Inconsistent feature dimensions.")

        if any(not math.isfinite(float(v)) for v in row):
            raise ValidationError("Features must contain finite numbers.")

    if any(not math.isfinite(float(v)) for v in y):
        raise ValidationError("Targets must contain finite numbers.")


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def solve_linear_system(matrix, vector):
    """Gaussian elimination with partial pivoting."""

    n = len(vector)
    augmented = [
        list(map(float, matrix[i])) + [float(vector[i])]
        for i in range(n)
    ]

    for col in range(n):
        pivot = max(
            range(col, n),
            key=lambda row: abs(augmented[row][col])
        )

        if abs(augmented[pivot][col]) < 1e-12:
            raise InterpretabilityError(
                "Singular matrix. Features may be perfectly correlated."
            )

        augmented[col], augmented[pivot] = (
            augmented[pivot],
            augmented[col],
        )

        divisor = augmented[col][col]

        for j in range(col, n + 1):
            augmented[col][j] /= divisor

        for row in range(n):
            if row == col:
                continue

            factor = augmented[row][col]

            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]

    return [augmented[i][n] for i in range(n)]


class LinearRegressionModel:
    """Ordinary least-squares regression with standardized coefficients."""

    def __init__(self, feature_names: list[str]):
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValidationError("Feature names must be unique and nonempty.")

        self.feature_names = feature_names
        self.means: list[float] = []
        self.scales: list[float] = []
        self.coefficients: list[float] = []
        self.intercept = 0.0
        self.fitted = False

    def fit(self, X, y):
        validate_dataset(X, y, len(self.feature_names))

        columns = transpose(X)
        self.means = [statistics.mean(c) for c in columns]
        self.scales = [
            statistics.pstdev(c) or 1.0 for c in columns
        ]

        standardized = [
            [
                (row[j] - self.means[j]) / self.scales[j]
                for j in range(len(self.feature_names))
            ]
            for row in X
        ]

        design = [[1.0] + row for row in standardized]

        # Compute the normal equations: (X'X)b = X'y.
        p = len(design[0])
        xtx = [
            [
                sum(row[i] * row[j] for row in design)
                for j in range(p)
            ]
            for i in range(p)
        ]

        xty = [
            sum(design[i][j] * y[i] for i in range(len(y)))
            for j in range(p)
        ]

        solution = solve_linear_system(xtx, xty)

        self.intercept = solution[0]
        self.coefficients = solution[1:]
        self.fitted = True
        return self

    def predict_one(self, row):
        if not self.fitted:
            raise InterpretabilityError("Model has not been fitted.")

        if len(row) != len(self.feature_names):
            raise ValidationError("Incorrect prediction feature count.")

        if any(not math.isfinite(float(v)) for v in row):
            raise ValidationError("Prediction features must be finite.")

        return self.intercept + sum(
            self.coefficients[j]
            * ((row[j] - self.means[j]) / self.scales[j])
            for j in range(len(row))
        )

    def predict(self, X):
        if not X:
            raise ValidationError("Prediction dataset cannot be empty.")

        return [self.predict_one(row) for row in X]


def evaluate_regression(actual, predicted) -> EvaluationResult:
    if len(actual) != len(predicted) or not actual:
        raise ValidationError("Evaluation arrays must have equal nonzero length.")

    errors = [
        float(a) - float(p)
        for a, p in zip(actual, predicted)
    ]

    if any(not math.isfinite(e) for e in errors):
        raise ValidationError("Evaluation values must be finite.")

    mse = sum(e * e for e in errors) / len(errors)
    mae = sum(abs(e) for e in errors) / len(errors)

    return EvaluationResult(
        mse=mse,
        mae=mae,
        rmse=math.sqrt(mse),
        sample_count=len(errors),
    )


def coefficient_importance(model: LinearRegressionModel):
    """Rank absolute standardized regression coefficients."""

    if not model.fitted:
        raise InterpretabilityError("Model has not been fitted.")

    raw = [
        abs(coefficient)
        for coefficient in model.coefficients
    ]

    total = sum(raw)

    return [
        ImportanceResult(
            feature=model.feature_names[i],
            score=raw[i],
            normalized_score=raw[i] / total if total else 0.0,
            method="standardized_coefficient",
        )
        for i in sorted(
            range(len(raw)),
            key=lambda j: raw[j],
            reverse=True,
        )
    ]


def permutation_importance(
    model,
    X,
    y,
    repeats=10,
    seed=42,
):
    """
    Measure performance degradation after independently shuffling
    each feature column.

    Higher positive values indicate greater predictive dependence.
    """

    validate_dataset(X, y, len(model.feature_names))

    if repeats < 1:
        raise ValidationError("Repeats must be at least one.")

    baseline = evaluate_regression(y, model.predict(X)).mse
    rng = random.Random(seed)
    results = []

    for feature_index, feature_name in enumerate(model.feature_names):
        drops = []

        for _ in range(repeats):
            shuffled = [list(row) for row in X]
            values = [row[feature_index] for row in shuffled]
            rng.shuffle(values)

            for i, value in enumerate(values):
                shuffled[i][feature_index] = value

            modified_score = evaluate_regression(
                y,
                model.predict(shuffled),
            ).mse

            drops.append(modified_score - baseline)

        results.append(
            ImportanceResult(
                feature=feature_name,
                score=statistics.mean(drops),
                normalized_score=0.0,
                method="permutation_mse_increase",
            )
        )

    total = sum(max(0.0, item.score) for item in results)

    for item in results:
        item.normalized_score = (
            max(0.0, item.score) / total if total else 0.0
        )

    return sorted(results, key=lambda item: item.score, reverse=True)


def partial_dependence(
    model,
    X,
    feature_index,
    grid_points=10,
):
    """
    Calculate one-dimensional partial dependence.

    Each grid value replaces one feature for every observation.
    Predictions are averaged across the dataset.
    """

    validate_dataset(X, [0.0] * len(X), len(model.feature_names))

    if not 0 <= feature_index < len(model.feature_names):
        raise ValidationError("Feature index is out of range.")

    if grid_points < 2:
        raise ValidationError("At least two grid points are required.")

    column = [row[feature_index] for row in X]
    minimum = min(column)
    maximum = max(column)

    if minimum == maximum:
        grid = [minimum]
    else:
        grid = [
            minimum + (maximum - minimum) * i / (grid_points - 1)
            for i in range(grid_points)
        ]

    averages = []

    for value in grid:
        modified = [list(row) for row in X]

        for row in modified:
            row[feature_index] = value

        predictions = model.predict(modified)
        averages.append(statistics.mean(predictions))

    return PartialDependenceResult(
        feature=model.feature_names[feature_index],
        values=grid,
        predictions=averages,
    )


def generate_dataset(samples=240, seed=42):
    """Generate a deterministic business-risk prediction dataset."""

    rng = random.Random(seed)
    X = []
    y = []

    for _ in range(samples):
        revenue = rng.uniform(100, 1000)
        debt_ratio = rng.uniform(0.05, 0.95)
        customer_retention = rng.uniform(0.40, 0.99)

        target = (
            0.035 * revenue
            - 120.0 * debt_ratio
            + 180.0 * customer_retention
            + rng.gauss(0, 8)
        )

        X.append([
            revenue,
            debt_ratio,
            customer_retention,
        ])
        y.append(target)

    return X, y


def train_test_split(X, y, test_ratio=0.25, seed=42):
    if not 0 < test_ratio < 1:
        raise ValidationError("Test ratio must be between zero and one.")

    indices = list(range(len(X)))
    random.Random(seed).shuffle(indices)

    split = int(len(indices) * (1 - test_ratio))

    if split == 0 or split == len(indices):
        raise ValidationError("Dataset is too small for this split.")

    train_indices = indices[:split]
    test_indices = indices[split:]

    return (
        [X[i] for i in train_indices],
        [X[i] for i in test_indices],
        [y[i] for i in train_indices],
        [y[i] for i in test_indices],
    )


def save_report(report, filename="interpretability_report.json"):
    """Write JSON without accepting executable serialized objects."""

    path = Path(filename)

    if path.suffix.lower() != ".json":
        raise ValidationError("Report filename must end in .json.")

    path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main():
    features = [
        Feature("revenue", "Annual business revenue", "thousand USD"),
        Feature("debt_ratio", "Debt relative to assets", "ratio"),
        Feature(
            "customer_retention",
            "Customer retention rate",
            "ratio",
        ),
    ]

    names = [feature.name for feature in features]

    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LinearRegressionModel(names)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    evaluation = evaluate_regression(y_test, predictions)

    coefficients = coefficient_importance(model)

    permutations = permutation_importance(
        model,
        X_test,
        y_test,
        repeats=15,
    )

    dependence = [
        partial_dependence(model, X_test, i, grid_points=8)
        for i in range(len(features))
    ]

    report = {
        "model": "ordinary_least_squares",
        "feature_metadata": [asdict(f) for f in features],
        "evaluation": asdict(evaluation),
        "coefficients": [asdict(x) for x in coefficients],
        "permutation_importance": [asdict(x) for x in permutations],
        "partial_dependence": [asdict(x) for x in dependence],
        "standardized_coefficients": model.coefficients,
        "intercept": model.intercept,
    }

    save_report(report)

    print("MODEL INTERPRETABILITY REPORT")
    print("=" * 55)

    print("\nModel Evaluation")
    print(f"MSE:  {evaluation.mse:.4f}")
    print(f"MAE:  {evaluation.mae:.4f}")
    print(f"RMSE: {evaluation.rmse:.4f}")

    print("\nStandardized Coefficient Importance")
    for item in coefficients:
        print(
            f"{item.feature:24s} "
            f"{item.score:10.4f} "
            f"{item.normalized_score:8.2%}"
        )

    print("\nPermutation Importance")
    for item in permutations:
        print(
            f"{item.feature:24s} "
            f"{item.score:12.4f} "
            f"{item.normalized_score:8.2%}"
        )

    print("\nPartial Dependence")
    for result in dependence:
        print(f"\nFeature: {result.feature}")

        for value, prediction in zip(
            result.values,
            result.predictions,
        ):
            print(f"  {value:10.4f} -> {prediction:10.4f}")

    print("\nReport saved to interpretability_report.json")

    # Basic behavioral assertions.
    assert evaluation.sample_count == len(X_test)
    assert len(coefficients) == len(features)
    assert len(permutations) == len(features)
    assert all(len(p.values) == len(p.predictions) for p in dependence)


if __name__ == "__main__":
    main()
