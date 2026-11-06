#!/usr/bin/env python3
"""
Explainable AI: SHAP, LIME, local explanations, and global explanations.

Implements:
- A deterministic binary classification model.
- Model prediction and probability calculation.
- Exact SHAP values for a small feature set using coalition enumeration.
- Local LIME-style explanations using weighted linear regression.
- Global feature importance through mean absolute SHAP values.
- Explanation consistency checks and JSON reporting.

Uses only the Python standard library.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ExplanationError(Exception):
    """Base exception for explanation failures."""


class ValidationError(ExplanationError):
    """Raised when explanation input is invalid."""


class ExplanationType(str, Enum):
    SHAP = "SHAP"
    LIME = "LIME"
    GLOBAL = "GLOBAL"


@dataclass(frozen=True)
class Feature:
    name: str
    minimum: float
    maximum: float
    default: float

    def validate(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValidationError(f"{self.name} must be finite")
        if not self.minimum <= value <= self.maximum:
            raise ValidationError(
                f"{self.name} must be between "
                f"{self.minimum} and {self.maximum}"
            )


@dataclass
class Explanation:
    explanation_type: str
    prediction: float
    baseline: float
    contributions: dict[str, float]
    reconstructed_prediction: float
    metadata: dict[str, Any]


class RiskModel:
    """
    Transparent logistic classification model.

    The model is deliberately explicit so that explanation
    algorithms can be verified against known behavior.
    """

    FEATURES = [
        Feature("income", 10000, 200000, 60000),
        Feature("debt_ratio", 0, 1, 0.35),
        Feature("credit_history", 0, 30, 8),
        Feature("missed_payments", 0, 20, 1),
        Feature("employment_years", 0, 40, 5),
    ]

    COEFFICIENTS = {
        "income": 0.000015,
        "debt_ratio": -2.2,
        "credit_history": 0.09,
        "missed_payments": -0.45,
        "employment_years": 0.055,
    }

    INTERCEPT = -0.5

    def __init__(self) -> None:
        self.feature_names = [f.name for f in self.FEATURES]
        self.feature_map = {f.name: f for f in self.FEATURES}

    def validate(self, sample: dict[str, float]) -> None:
        if not isinstance(sample, dict):
            raise ValidationError("Input must be a dictionary")

        if set(sample) != set(self.feature_names):
            missing = set(self.feature_names) - set(sample)
            extra = set(sample) - set(self.feature_names)
            raise ValidationError(
                f"Invalid features. Missing={missing}, extra={extra}"
            )

        for name, value in sample.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(f"{name} must be numeric")
            self.feature_map[name].validate(float(value))

    @staticmethod
    def sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def logit(self, sample: dict[str, float]) -> float:
        self.validate(sample)

        result = self.INTERCEPT
        for name, value in sample.items():
            result += self.COEFFICIENTS[name] * value

        return result

    def predict_probability(self, sample: dict[str, float]) -> float:
        return self.sigmoid(self.logit(sample))

    def predict(self, sample: dict[str, float]) -> int:
        return int(self.predict_probability(sample) >= 0.5)


class ExplainableAI:
    def __init__(
        self,
        model: RiskModel,
        background: list[dict[str, float]],
    ) -> None:
        if not background:
            raise ValidationError("Background dataset cannot be empty")

        for sample in background:
            model.validate(sample)

        self.model = model
        self.background = background
        self.feature_names = model.feature_names
        self.feature_count = len(self.feature_names)

    def expected_value(self) -> float:
        return statistics.mean(
            self.model.predict_probability(sample)
            for sample in self.background
        )

    def _hybrid_prediction(
        self,
        sample: dict[str, float],
        baseline: dict[str, float],
        selected: set[str],
    ) -> float:
        """
        Replace unselected features with baseline values.

        Averaging these hybrid predictions across background records
        approximates the interventional SHAP value function.
        """

        predictions = []

        for background_sample in self.background:
            hybrid = {
                name: (
                    sample[name]
                    if name in selected
                    else background_sample[name]
                )
                for name in self.feature_names
            }

            predictions.append(
                self.model.predict_probability(hybrid)
            )

        return statistics.mean(predictions)

    def shap_values(
        self,
        sample: dict[str, float],
    ) -> Explanation:
        """
        Exact Shapley values by enumerating all feature coalitions.

        Complexity: O(2^M * M * B), where M is feature count
        and B is background dataset size.
        """

        self.model.validate(sample)

        names = self.feature_names
        count = len(names)

        from itertools import combinations

        baseline = self.expected_value()
        contributions = {name: 0.0 for name in names}

        coalition_values: dict[frozenset[str], float] = {}

        for size in range(count + 1):
            for combination in combinations(names, size):
                coalition = frozenset(combination)
                coalition_values[coalition] = (
                    self._hybrid_prediction(
                        sample,
                        {},
                        set(coalition),
                    )
                )

        for name in names:
            others = [feature for feature in names if feature != name]

            for size in range(count):
                weight = (
                    math.factorial(size)
                    * math.factorial(count - size - 1)
                    / math.factorial(count)
                )

                for combination in combinations(others, size):
                    coalition = frozenset(combination)

                    difference = (
                        coalition_values[coalition | {name}]
                        - coalition_values[coalition]
                    )

                    contributions[name] += weight * difference

        prediction = self.model.predict_probability(sample)
        reconstructed = baseline + sum(contributions.values())

        return Explanation(
            explanation_type=ExplanationType.SHAP.value,
            prediction=prediction,
            baseline=baseline,
            contributions=contributions,
            reconstructed_prediction=reconstructed,
            metadata={
                "method": "Exact interventional Shapley enumeration",
                "feature_count": count,
                "background_size": len(self.background),
                "efficiency_error": abs(prediction - reconstructed),
            },
        )

    @staticmethod
    def _solve_linear_system(
        matrix: list[list[float]],
        vector: list[float],
    ) -> list[float]:
        """Gaussian elimination with partial pivoting."""

        n = len(vector)
        augmented = [
            matrix[i][:] + [vector[i]]
            for i in range(n)
        ]

        for col in range(n):
            pivot = max(
                range(col, n),
                key=lambda row: abs(augmented[row][col]),
            )

            if abs(augmented[pivot][col]) < 1e-12:
                raise ExplanationError(
                    "Singular matrix in weighted regression"
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
                    augmented[row][j] -= (
                        factor * augmented[col][j]
                    )

        return [augmented[i][n] for i in range(n)]

    def lime_explanation(
        self,
        sample: dict[str, float],
        num_samples: int = 1500,
        kernel_width: float = 0.75,
        seed: int = 42,
    ) -> Explanation:
        """
        LIME-style local surrogate explanation.

        Generates perturbed observations, measures model predictions,
        weights observations by proximity, and fits a weighted linear
        model using normal equations with regularization.
        """

        self.model.validate(sample)

        if num_samples < 20 or num_samples > 100000:
            raise ValidationError(
                "num_samples must be between 20 and 100000"
            )

        if kernel_width <= 0 or not math.isfinite(kernel_width):
            raise ValidationError("kernel_width must be positive")

        rng = random.Random(seed)
        names = self.feature_names
        dimensions = len(names)

        perturbations: list[tuple[list[float], float, float]] = []

        ranges = {
            name: self.model.feature_map[name].maximum
            - self.model.feature_map[name].minimum
            for name in names
        }

        for _ in range(num_samples):
            perturbed = {}
            standardized = []

            for name in names:
                feature = self.model.feature_map[name]
                scale = ranges[name]

                value = rng.gauss(sample[name], scale * 0.15)
                value = max(
                    feature.minimum,
                    min(feature.maximum, value),
                )

                perturbed[name] = value

                standardized.append(
                    (value - sample[name]) / scale
                )

            distance = math.sqrt(
                sum(value * value for value in standardized)
            )

            weight = math.exp(
                -(distance ** 2) / (kernel_width ** 2)
            )

            prediction = self.model.predict_probability(perturbed)

            perturbations.append(
                (standardized, prediction, weight)
            )

        size = dimensions + 1
        matrix = [[0.0] * size for _ in range(size)]
        vector = [0.0] * size

        # The first coordinate represents the intercept.
        for features, target, weight in perturbations:
            row = [1.0] + features

            for i in range(size):
                vector[i] += weight * row[i] * target

                for j in range(size):
                    matrix[i][j] += weight * row[i] * row[j]

        # Ridge regularization stabilizes the local surrogate.
        regularization = 1e-6

        for i in range(1, size):
            matrix[i][i] += regularization

        coefficients = self._solve_linear_system(matrix, vector)

        contributions = {
            names[i]: coefficients[i + 1]
            * (0.0 - 0.0)
            for i in range(dimensions)
        }

        # Convert standardized surrogate coefficients into
        # contributions relative to the original observation.
        local_intercept = coefficients[0]

        contributions = {
            names[i]: coefficients[i + 1]
            * (
                sample[names[i]]
                - self.model.feature_map[names[i]].default
            )
            / ranges[names[i]]
            for i in range(dimensions)
        }

        prediction = self.model.predict_probability(sample)

        # A LIME surrogate is an approximation, not an additive
        # decomposition guaranteed to equal the black-box prediction.
        reconstructed = local_intercept + sum(
            coefficients[i + 1] * 0.0
            for i in range(dimensions)
        )

        return Explanation(
            explanation_type=ExplanationType.LIME.value,
            prediction=prediction,
            baseline=local_intercept,
            contributions=contributions,
            reconstructed_prediction=reconstructed,
            metadata={
                "method": "Locally weighted linear surrogate",
                "perturbation_count": num_samples,
                "kernel_width": kernel_width,
                "regularization": regularization,
                "seed": seed,
                "surrogate_coefficients": {
                    names[i]: coefficients[i + 1]
                    for i in range(dimensions)
                },
            },
        )

    def global_importance(
        self,
        dataset: list[dict[str, float]],
    ) -> dict[str, float]:
        """Calculate global importance using mean absolute SHAP."""

        if not dataset:
            raise ValidationError("Dataset cannot be empty")

        totals = {name: 0.0 for name in self.feature_names}

        for sample in dataset:
            explanation = self.shap_values(sample)

            for name, contribution in explanation.contributions.items():
                totals[name] += abs(contribution)

        importance = {
            name: value / len(dataset)
            for name, value in totals.items()
        }

        return dict(
            sorted(
                importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )


def create_background() -> list[dict[str, float]]:
    return [
        {
            "income": 45000,
            "debt_ratio": 0.55,
            "credit_history": 3,
            "missed_payments": 4,
            "employment_years": 2,
        },
        {
            "income": 70000,
            "debt_ratio": 0.25,
            "credit_history": 12,
            "missed_payments": 0,
            "employment_years": 10,
        },
        {
            "income": 55000,
            "debt_ratio": 0.40,
            "credit_history": 7,
            "missed_payments": 2,
            "employment_years": 5,
        },
        {
            "income": 95000,
            "debt_ratio": 0.20,
            "credit_history": 18,
            "missed_payments": 0,
            "employment_years": 15,
        },
    ]


def main() -> None:
    model = RiskModel()
    explainer = ExplainableAI(model, create_background())

    applicant = {
        "income": 85000,
        "debt_ratio": 0.30,
        "credit_history": 14,
        "missed_payments": 1,
        "employment_years": 9,
    }

    print("EXPLAINABLE CREDIT RISK MODEL")
    print("=" * 55)

    probability = model.predict_probability(applicant)
    prediction = model.predict(applicant)

    print(f"Risk probability: {probability:.6f}")
    print(f"Predicted class: {prediction}")

    shap = explainer.shap_values(applicant)
    lime = explainer.lime_explanation(applicant)

    print("\nLOCAL SHAP EXPLANATION")
    print("-" * 55)
    print(f"Baseline: {shap.baseline:.6f}")

    for name, value in sorted(
        shap.contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    ):
        print(f"{name:22s}: {value:+.6f}")

    print(
        f"Reconstructed probability: "
        f"{shap.reconstructed_prediction:.6f}"
    )

    print("\nLOCAL LIME EXPLANATION")
    print("-" * 55)

    for name, value in sorted(
        lime.contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    ):
        print(f"{name:22s}: {value:+.6f}")

    print("\nGLOBAL FEATURE IMPORTANCE")
    print("-" * 55)

    dataset = [
        applicant,
        *create_background(),
        {
            "income": 120000,
            "debt_ratio": 0.15,
            "credit_history": 20,
            "missed_payments": 0,
            "employment_years": 18,
        },
    ]

    importance = explainer.global_importance(dataset)

    for name, value in importance.items():
        print(f"{name:22s}: {value:.6f}")

    assert abs(
        shap.prediction - shap.reconstructed_prediction
    ) < 1e-9, "SHAP additivity check failed"

    assert all(
        math.isfinite(value)
        for value in importance.values()
    )

    report = {
        "applicant": applicant,
        "prediction": {
            "probability": probability,
            "class": prediction,
        },
        "shap": asdict(shap),
        "lime": asdict(lime),
        "global_importance": importance,
    }

    output_path = Path("explainability_report.json")
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to {output_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ExplanationError, OSError, ValueError) as error:
        print(f"Execution failed: {error}")
        raise SystemExit(1)
