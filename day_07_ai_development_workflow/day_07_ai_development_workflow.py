"""
AI DEVELOPMENT WORKFLOW
=======================

A self-contained study script covering the complete lifecycle:

1. Problem definition
2. Data collection
3. Data understanding and preprocessing
4. Feature engineering
5. Model training
6. Evaluation
7. Error analysis
8. Model persistence
9. Deployment concepts
10. Monitoring and production considerations

The examples use only Python's standard library so that the workflow can be
studied and executed without installing external packages.

The script implements a small machine-learning system from first principles:
a logistic regression classifier trained with gradient descent.

Dataset:
    Synthetic customer churn prediction.

Target:
    Predict whether a customer is likely to churn.

Features:
    - age
    - monthly_spend
    - support_tickets
    - months_as_customer
    - uses_premium_plan

The implementation demonstrates the same development workflow used for many
machine-learning projects, even though real projects often use libraries such
as NumPy, pandas, scikit-learn, PyTorch, TensorFlow, databases, cloud storage,
experiment trackers, containers, and model-serving systems.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# 1. PROBLEM DEFINITION
# =============================================================================

"""
An AI or machine-learning project should begin with a precise problem
definition rather than immediately selecting a model.

A useful problem definition answers:

Business problem:
    What undesirable or desirable outcome are we trying to influence?

Prediction problem:
    What should the system predict?

Unit of prediction:
    What does one prediction represent?

Target variable:
    What is the correct answer the model should learn?

Features:
    What information is available before the prediction is required?

Decision:
    What action will be taken using the prediction?

Success metric:
    How will technical and business performance be measured?

Constraints:
    What restrictions apply to latency, cost, privacy, fairness, explainability,
    reliability, security, and available data?

For this example:
    Business problem:
        Reduce customer churn.

    Prediction problem:
        Estimate whether a customer will churn.

    Unit of prediction:
        One customer.

    Target:
        churned = 1 means churn.
        churned = 0 means retained.

    Important constraint:
        Features must represent information available before churn occurs.
        Using information discovered after churn would create data leakage.
"""


@dataclass
class ProblemDefinition:
    """Structured documentation for an ML problem."""

    business_problem: str
    prediction_task: str
    unit_of_prediction: str
    target_variable: str
    positive_class: str
    primary_metric: str
    secondary_metrics: List[str]
    constraints: List[str]


def demonstrate_problem_definition() -> ProblemDefinition:
    problem = ProblemDefinition(
        business_problem="Identify customers at elevated risk of churn.",
        prediction_task="Binary classification",
        unit_of_prediction="One customer",
        target_variable="churned",
        positive_class="Customer churns",
        primary_metric="Recall, while monitoring precision",
        secondary_metrics=["Precision", "Accuracy", "F1 score", "ROC-AUC"],
        constraints=[
            "No feature may use information unavailable at prediction time.",
            "Predictions should be reproducible.",
            "False positives may trigger unnecessary retention offers.",
            "False negatives may miss customers who are about to churn.",
        ],
    )

    print("\n" + "=" * 80)
    print("1. PROBLEM DEFINITION")
    print("=" * 80)

    for field_name, value in asdict(problem).items():
        print(f"{field_name.replace('_', ' ').title()}: {value}")

    return problem


# =============================================================================
# 2. DATA COLLECTION
# =============================================================================

"""
Data collection means obtaining observations relevant to the defined problem.

Common sources include:
    - relational databases
    - transactional systems
    - APIs
    - sensors
    - application logs
    - documents
    - images
    - audio
    - manually labeled datasets
    - public datasets

Data quality is often more important than model complexity.

Important collection risks:
    - missing labels
    - inconsistent schemas
    - duplicate records
    - sampling bias
    - stale data
    - measurement errors
    - privacy violations
    - unauthorized data access
    - target leakage
    - label errors

This script generates synthetic data so that the entire workflow remains
self-contained.
"""


@dataclass
class CustomerRecord:
    customer_id: int
    age: Optional[float]
    monthly_spend: Optional[float]
    support_tickets: Optional[float]
    months_as_customer: Optional[float]
    uses_premium_plan: Optional[float]
    churned: Optional[int]


FEATURE_NAMES = [
    "age",
    "monthly_spend",
    "support_tickets",
    "months_as_customer",
    "uses_premium_plan",
]

TARGET_NAME = "churned"


def generate_synthetic_customer_data(
    number_of_records: int = 800,
    random_seed: int = 42,
) -> List[CustomerRecord]:
    """
    Generate synthetic binary classification data.

    The target is created from an underlying probability model.

    This is useful for demonstration because we know that a relationship
    between features and the target exists.

    Real datasets generally contain:
        - unknown relationships
        - noise
        - missing values
        - inconsistent values
        - correlated features
        - changing distributions
    """

    random.seed(random_seed)
    records: List[CustomerRecord] = []

    for customer_id in range(1, number_of_records + 1):
        age = random.uniform(18, 80)
        monthly_spend = random.uniform(10, 500)
        support_tickets = random.randint(0, 12)
        months_as_customer = random.uniform(1, 120)
        uses_premium_plan = random.choice([0, 1])

        # A hidden data-generating relationship.
        # More support tickets and lower tenure increase churn probability.
        churn_score = (
            -0.8
            - 0.018 * age
            - 0.006 * monthly_spend
            + 0.42 * support_tickets
            - 0.018 * months_as_customer
            - 0.75 * uses_premium_plan
            + random.gauss(0, 0.7)
        )

        churn_probability = sigmoid(churn_score)
        churned = 1 if random.random() < churn_probability else 0

        record = CustomerRecord(
            customer_id=customer_id,
            age=age,
            monthly_spend=monthly_spend,
            support_tickets=float(support_tickets),
            months_as_customer=months_as_customer,
            uses_premium_plan=float(uses_premium_plan),
            churned=churned,
        )

        records.append(record)

    # Introduce realistic data problems deliberately.
    if len(records) >= 10:
        records[5].age = None
        records[20].monthly_spend = None
        records[35].support_tickets = None
        records[50].churned = None

        # Duplicate one record with a different ID.
        duplicate_source = records[70]
        records.append(
            CustomerRecord(
                customer_id=number_of_records + 1,
                age=duplicate_source.age,
                monthly_spend=duplicate_source.monthly_spend,
                support_tickets=duplicate_source.support_tickets,
                months_as_customer=duplicate_source.months_as_customer,
                uses_premium_plan=duplicate_source.uses_premium_plan,
                churned=duplicate_source.churned,
            )
        )

        # Invalid value.
        records[90].age = 250

    return records


def demonstrate_data_collection() -> List[CustomerRecord]:
    print("\n" + "=" * 80)
    print("2. DATA COLLECTION")
    print("=" * 80)

    records = generate_synthetic_customer_data()

    print(f"Collected records: {len(records)}")
    print("Synthetic source: generated customer observations")
    print("Target variable:", TARGET_NAME)
    print("Features:", FEATURE_NAMES)

    return records


# =============================================================================
# 3. DATA UNDERSTANDING
# =============================================================================

"""
Before preprocessing, inspect the data.

Important questions:
    - How many rows exist?
    - What is the target distribution?
    - Are values missing?
    - Are there duplicates?
    - Are numerical ranges plausible?
    - Are classes imbalanced?
    - Are features correlated?
    - Does the training data represent deployment conditions?

Data understanding is not a one-time step. New issues are often discovered
during training, evaluation, deployment, and monitoring.
"""


def get_feature_value(record: CustomerRecord, feature_name: str) -> Any:
    return getattr(record, feature_name)


def missing_value_count(
    records: Sequence[CustomerRecord],
    field_name: str,
) -> int:
    return sum(
        1
        for record in records
        if getattr(record, field_name) is None
    )


def demonstrate_data_understanding(records: Sequence[CustomerRecord]) -> None:
    print("\n" + "=" * 80)
    print("3. DATA UNDERSTANDING")
    print("=" * 80)

    print(f"Number of records: {len(records)}")

    for feature_name in FEATURE_NAMES + [TARGET_NAME]:
        count = missing_value_count(records, feature_name)
        print(f"Missing values in {feature_name}: {count}")

    valid_targets = [
        record.churned
        for record in records
        if record.churned is not None
    ]

    positive_count = sum(valid_targets)
    negative_count = len(valid_targets) - positive_count

    print(f"Positive class count: {positive_count}")
    print(f"Negative class count: {negative_count}")

    if valid_targets:
        positive_rate = positive_count / len(valid_targets)
        print(f"Positive class rate: {positive_rate:.3f}")

    # Basic range inspection.
    for feature_name in FEATURE_NAMES:
        numeric_values = [
            get_feature_value(record, feature_name)
            for record in records
            if get_feature_value(record, feature_name) is not None
        ]

        if numeric_values:
            print(
                f"{feature_name}: "
                f"min={min(numeric_values):.2f}, "
                f"max={max(numeric_values):.2f}, "
                f"mean={statistics.mean(numeric_values):.2f}"
            )


# =============================================================================
# 4. DATA PREPROCESSING
# =============================================================================

"""
Preprocessing transforms raw data into a representation suitable for training.

Common operations:
    - schema validation
    - duplicate handling
    - missing-value treatment
    - invalid-value treatment
    - categorical encoding
    - numerical scaling
    - text normalization
    - image resizing
    - audio normalization
    - feature construction

A critical production principle:

    Fit preprocessing parameters using training data only.

For example, the mean and standard deviation used for standardization must be
calculated from training data rather than the full dataset.

Calculating preprocessing statistics using validation or test data leaks
information from those datasets into training.
"""


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]


def validate_customer_record(record: CustomerRecord) -> ValidationResult:
    """Validate basic domain constraints."""

    errors: List[str] = []

    if record.customer_id <= 0:
        errors.append("customer_id must be positive")

    if record.age is not None and not (0 <= record.age <= 120):
        errors.append("age is outside the allowed range")

    if (
        record.monthly_spend is not None
        and record.monthly_spend < 0
    ):
        errors.append("monthly_spend cannot be negative")

    if (
        record.support_tickets is not None
        and record.support_tickets < 0
    ):
        errors.append("support_tickets cannot be negative")

    if (
        record.months_as_customer is not None
        and record.months_as_customer < 0
    ):
        errors.append("months_as_customer cannot be negative")

    if (
        record.uses_premium_plan is not None
        and record.uses_premium_plan not in (0, 1)
    ):
        errors.append("uses_premium_plan must be 0 or 1")

    if record.churned is not None and record.churned not in (0, 1):
        errors.append("churned must be 0 or 1")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
    )


def record_signature(record: CustomerRecord) -> Tuple[Any, ...]:
    """
    Create a duplicate-detection signature.

    customer_id is intentionally excluded because two records may represent
    the same observation while having different identifiers.
    """

    return (
        record.age,
        record.monthly_spend,
        record.support_tickets,
        record.months_as_customer,
        record.uses_premium_plan,
        record.churned,
    )


def remove_duplicates(
    records: Sequence[CustomerRecord],
) -> List[CustomerRecord]:
    seen = set()
    unique_records: List[CustomerRecord] = []

    for record in records:
        signature = record_signature(record)

        if signature not in seen:
            seen.add(signature)
            unique_records.append(record)

    return unique_records


def filter_invalid_records(
    records: Sequence[CustomerRecord],
) -> Tuple[List[CustomerRecord], List[CustomerRecord]]:
    valid_records: List[CustomerRecord] = []
    invalid_records: List[CustomerRecord] = []

    for record in records:
        validation = validate_customer_record(record)

        if validation.valid:
            valid_records.append(record)
        else:
            invalid_records.append(record)

    return valid_records, invalid_records


def calculate_feature_medians(
    records: Sequence[CustomerRecord],
) -> Dict[str, float]:
    """
    Median imputation is relatively robust to extreme numerical values.

    Mean imputation is another common choice.
    More advanced methods can use predictive imputation.
    """

    medians: Dict[str, float] = {}

    for feature_name in FEATURE_NAMES:
        values = [
            float(get_feature_value(record, feature_name))
            for record in records
            if get_feature_value(record, feature_name) is not None
        ]

        if not values:
            raise ValueError(
                f"Cannot calculate median for {feature_name}: "
                "all values are missing."
            )

        medians[feature_name] = statistics.median(values)

    return medians


def impute_missing_values(
    records: Sequence[CustomerRecord],
    medians: Dict[str, float],
) -> List[CustomerRecord]:
    """
    Return new records instead of modifying original data.

    Immutable-style transformations reduce accidental side effects.
    """

    processed: List[CustomerRecord] = []

    for record in records:
        processed.append(
            CustomerRecord(
                customer_id=record.customer_id,
                age=(
                    record.age
                    if record.age is not None
                    else medians["age"]
                ),
                monthly_spend=(
                    record.monthly_spend
                    if record.monthly_spend is not None
                    else medians["monthly_spend"]
                ),
                support_tickets=(
                    record.support_tickets
                    if record.support_tickets is not None
                    else medians["support_tickets"]
                ),
                months_as_customer=(
                    record.months_as_customer
                    if record.months_as_customer is not None
                    else medians["months_as_customer"]
                ),
                uses_premium_plan=(
                    record.uses_premium_plan
                    if record.uses_premium_plan is not None
                    else medians["uses_premium_plan"]
                ),
                churned=record.churned,
            )
        )

    return processed


# =============================================================================
# 5. DATA SPLITTING
# =============================================================================

"""
The most common split is:

    Training set
        Used to fit model parameters.

    Validation set
        Used for model selection and hyperparameter decisions.

    Test set
        Used for final evaluation.

The test set should not repeatedly influence model selection.

For time-dependent data, random splitting can be inappropriate because future
observations may leak into the training set. Chronological splitting is often
safer.

For this educational example, a shuffled split is used.
"""


def train_validation_test_split(
    records: Sequence[CustomerRecord],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[
    List[CustomerRecord],
    List[CustomerRecord],
    List[CustomerRecord],
]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1"
        )

    shuffled = list(records)
    random.Random(random_seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)

    training = shuffled[:train_end]
    validation = shuffled[train_end:validation_end]
    test = shuffled[validation_end:]

    return training, validation, test


# =============================================================================
# 6. FEATURE SCALING
# =============================================================================

"""
Gradient-based models usually benefit from numerical scaling.

Standardization:

    z = (x - mean) / standard_deviation

Benefits:
    - features have comparable numerical scales
    - gradient descent often converges more reliably
    - regularization treats feature scales more consistently

Potential issue:
    A zero standard deviation means the feature is constant.

A constant feature contains no variation and generally provides no predictive
information for many models.
"""


@dataclass
class StandardScaler:
    means: Dict[str, float]
    standard_deviations: Dict[str, float]

    @classmethod
    def fit(
        cls,
        records: Sequence[CustomerRecord],
        feature_names: Sequence[str],
    ) -> "StandardScaler":
        means: Dict[str, float] = {}
        standard_deviations: Dict[str, float] = {}

        for feature_name in feature_names:
            values = [
                float(get_feature_value(record, feature_name))
                for record in records
            ]

            mean_value = statistics.mean(values)

            # Population standard deviation is sufficient here because
            # preprocessing treats the training dataset as the fitted reference.
            standard_deviation = statistics.pstdev(values)

            if standard_deviation == 0:
                standard_deviation = 1.0

            means[feature_name] = mean_value
            standard_deviations[feature_name] = standard_deviation

        return cls(
            means=means,
            standard_deviations=standard_deviations,
        )

    def transform_record(
        self,
        record: CustomerRecord,
        feature_names: Sequence[str],
    ) -> List[float]:
        transformed: List[float] = []

        for feature_name in feature_names:
            value = float(get_feature_value(record, feature_name))
            scaled = (
                (value - self.means[feature_name])
                / self.standard_deviations[feature_name]
            )
            transformed.append(scaled)

        return transformed

    def transform(
        self,
        records: Sequence[CustomerRecord],
        feature_names: Sequence[str],
    ) -> List[List[float]]:
        return [
            self.transform_record(record, feature_names)
            for record in records
        ]


# =============================================================================
# 7. MATHEMATICAL BUILDING BLOCKS
# =============================================================================

"""
Binary logistic regression models:

    P(y = 1 | x) = sigmoid(z)

where:

    z = w1*x1 + w2*x2 + ... + wn*xn + bias

The sigmoid function converts any real-valued score into a value between 0 and 1.

Binary cross-entropy loss:

    L = -[y*log(p) + (1-y)*log(1-p)]

Lower loss indicates that predicted probabilities are closer to observed labels.

Gradient descent repeatedly updates parameters:

    parameter = parameter - learning_rate * gradient
"""


def sigmoid(value: float) -> float:
    """
    Numerically stable sigmoid.

    Directly calculating exp(-value) can overflow for extreme negative values.
    """

    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)

    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def clamp_probability(
    probability: float,
    epsilon: float = 1e-15,
) -> float:
    """Avoid log(0) during loss calculations."""

    return min(max(probability, epsilon), 1.0 - epsilon)


def binary_cross_entropy(
    actual: Sequence[int],
    probabilities: Sequence[float],
) -> float:
    if len(actual) != len(probabilities):
        raise ValueError("actual and probabilities must have equal lengths")

    if not actual:
        raise ValueError("Cannot calculate loss for an empty dataset")

    total_loss = 0.0

    for target, probability in zip(actual, probabilities):
        probability = clamp_probability(probability)

        total_loss += -(
            target * math.log(probability)
            + (1 - target) * math.log(1 - probability)
        )

    return total_loss / len(actual)


# =============================================================================
# 8. LOGISTIC REGRESSION FROM FIRST PRINCIPLES
# =============================================================================


@dataclass
class LogisticRegressionModel:
    """
    Binary logistic regression.

    Parameters:
        weights:
            One coefficient per feature.

        bias:
            Intercept term.

        learning_rate:
            Step size used by gradient descent.

        epochs:
            Number of optimization passes through the training data.
    """

    learning_rate: float = 0.05
    epochs: int = 1000
    weights: Optional[List[float]] = None
    bias: float = 0.0
    training_history: Optional[List[float]] = None

    def fit(
        self,
        features: Sequence[Sequence[float]],
        targets: Sequence[int],
        verbose: bool = False,
    ) -> "LogisticRegressionModel":
        if len(features) != len(targets):
            raise ValueError(
                "features and targets must have equal numbers of rows"
            )

        if not features:
            raise ValueError("Cannot train on an empty dataset")

        number_of_samples = len(features)
        number_of_features = len(features[0])

        if number_of_features == 0:
            raise ValueError("At least one feature is required")

        for row in features:
            if len(row) != number_of_features:
                raise ValueError(
                    "All feature rows must have the same number of columns"
                )

        for target in targets:
            if target not in (0, 1):
                raise ValueError("Binary targets must be 0 or 1")

        self.weights = [0.0] * number_of_features
        self.bias = 0.0
        self.training_history = []

        for epoch in range(self.epochs):
            probabilities = [
                self.predict_probability(row)
                for row in features
            ]

            loss = binary_cross_entropy(targets, probabilities)
            self.training_history.append(loss)

            weight_gradients = [0.0] * number_of_features
            bias_gradient = 0.0

            for row, target, probability in zip(
                features,
                targets,
                probabilities,
            ):
                error = probability - target

                for index in range(number_of_features):
                    weight_gradients[index] += error * row[index]

                bias_gradient += error

            # Average gradients.
            for index in range(number_of_features):
                weight_gradients[index] /= number_of_samples

            bias_gradient /= number_of_samples

            # Gradient descent parameter updates.
            for index in range(number_of_features):
                self.weights[index] -= (
                    self.learning_rate
                    * weight_gradients[index]
                )

            self.bias -= self.learning_rate * bias_gradient

            if verbose and (
                epoch == 0
                or (epoch + 1) % 200 == 0
                or epoch == self.epochs - 1
            ):
                print(
                    f"Epoch {epoch + 1:4d} | "
                    f"Loss: {loss:.6f}"
                )

        return self

    def predict_probability(
        self,
        feature_row: Sequence[float],
    ) -> float:
        if self.weights is None:
            raise RuntimeError("Model must be trained before prediction")

        if len(feature_row) != len(self.weights):
            raise ValueError(
                "Feature count does not match trained model"
            )

        linear_score = self.bias

        for weight, feature_value in zip(
            self.weights,
            feature_row,
        ):
            linear_score += weight * feature_value

        return sigmoid(linear_score)

    def predict(
        self,
        feature_row: Sequence[float],
        threshold: float = 0.5,
    ) -> int:
        if not 0 < threshold < 1:
            raise ValueError(
                "Classification threshold must be between 0 and 1"
            )

        probability = self.predict_probability(feature_row)

        return 1 if probability >= threshold else 0

    def predict_many(
        self,
        features: Sequence[Sequence[float]],
        threshold: float = 0.5,
    ) -> List[int]:
        return [
            self.predict(row, threshold)
            for row in features
        ]

    def predict_probabilities(
        self,
        features: Sequence[Sequence[float]],
    ) -> List[float]:
        return [
            self.predict_probability(row)
            for row in features
        ]


# =============================================================================
# 9. EVALUATION METRICS
# =============================================================================

"""
A model should not be evaluated using only accuracy.

For binary classification:

True Positive:
    Predicted positive and actually positive.

True Negative:
    Predicted negative and actually negative.

False Positive:
    Predicted positive but actually negative.

False Negative:
    Predicted negative but actually positive.

Accuracy:
    Correct predictions / all predictions

Precision:
    True positives / predicted positives

Recall:
    True positives / actual positives

F1:
    Harmonic mean of precision and recall

The appropriate metric depends on the business cost of different errors.
"""


@dataclass
class ConfusionMatrix:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        )


def build_confusion_matrix(
    actual: Sequence[int],
    predicted: Sequence[int],
) -> ConfusionMatrix:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have equal lengths")

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for target, prediction in zip(actual, predicted):
        if target == 1 and prediction == 1:
            true_positive += 1
        elif target == 0 and prediction == 0:
            true_negative += 1
        elif target == 0 and prediction == 1:
            false_positive += 1
        elif target == 1 and prediction == 0:
            false_negative += 1
        else:
            raise ValueError(
                "Both actual and predicted values must be binary"
            )

    return ConfusionMatrix(
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Metrics can have zero denominators.

    Example:
        If a model predicts no positive cases,
        precision has zero predicted positives.

    Returning 0 is a practical convention for this educational implementation.
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator


def classification_metrics(
    actual: Sequence[int],
    predicted: Sequence[int],
) -> Dict[str, float]:
    matrix = build_confusion_matrix(actual, predicted)

    accuracy = safe_divide(
        matrix.true_positive + matrix.true_negative,
        matrix.total,
    )

    precision = safe_divide(
        matrix.true_positive,
        matrix.true_positive + matrix.false_positive,
    )

    recall = safe_divide(
        matrix.true_positive,
        matrix.true_positive + matrix.false_negative,
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def approximate_roc_auc(
    actual: Sequence[int],
    probabilities: Sequence[float],
) -> float:
    """
    Approximate ROC-AUC using rank statistics.

    ROC-AUC can be interpreted as the probability that a randomly selected
    positive example receives a higher score than a randomly selected negative
    example.

    This implementation handles ties by assigning average ranks.
    """

    if len(actual) != len(probabilities):
        raise ValueError("Inputs must have equal lengths")

    positive_count = sum(actual)
    negative_count = len(actual) - positive_count

    if positive_count == 0 or negative_count == 0:
        return 0.0

    indexed = sorted(
        enumerate(probabilities),
        key=lambda item: item[1],
    )

    ranks = [0.0] * len(probabilities)
    index = 0

    while index < len(indexed):
        tie_end = index

        while (
            tie_end + 1 < len(indexed)
            and indexed[tie_end + 1][1] == indexed[index][1]
        ):
            tie_end += 1

        average_rank = (
            (index + 1 + tie_end + 1) / 2
        )

        for position in range(index, tie_end + 1):
            original_index = indexed[position][0]
            ranks[original_index] = average_rank

        index = tie_end + 1

    positive_rank_sum = sum(
        rank
        for rank, target in zip(ranks, actual)
        if target == 1
    )

    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)

    return auc


# =============================================================================
# 10. THRESHOLD SELECTION
# =============================================================================

"""
Probability prediction and classification decisions are separate concepts.

A model may predict:

    P(churn) = 0.42

Whether this becomes "will churn" depends on the decision threshold.

The common threshold 0.5 is not automatically optimal.

Examples:
    Lower threshold:
        More positive predictions.
        Often higher recall.
        Often lower precision.

    Higher threshold:
        Fewer positive predictions.
        Often higher precision.
        Often lower recall.

Threshold selection should reflect business costs and be validated on data that
was not used to fit model parameters.
"""


def find_best_threshold_by_f1(
    actual: Sequence[int],
    probabilities: Sequence[float],
) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {
        "accuracy": -1.0,
        "precision": -1.0,
        "recall": -1.0,
        "f1": -1.0,
    }

    for threshold_step in range(1, 100):
        threshold = threshold_step / 100

        predictions = [
            1 if probability >= threshold else 0
            for probability in probabilities
        ]

        metrics = classification_metrics(
            actual,
            predictions,
        )

        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


# =============================================================================
# 11. ERROR ANALYSIS
# =============================================================================

"""
Aggregate metrics hide individual failure patterns.

Error analysis investigates:
    - false positives
    - false negatives
    - difficult subgroups
    - missing feature patterns
    - label quality
    - unusual input ranges
    - possible leakage

False negatives and false positives may have different business consequences.
"""


def print_error_analysis(
    records: Sequence[CustomerRecord],
    probabilities: Sequence[float],
    predictions: Sequence[int],
    maximum_examples: int = 5,
) -> None:
    print("\nERROR ANALYSIS")

    false_negatives = []
    false_positives = []

    for record, probability, prediction in zip(
        records,
        probabilities,
        predictions,
    ):
        if record.churned == 1 and prediction == 0:
            false_negatives.append(
                (record, probability)
            )
        elif record.churned == 0 and prediction == 1:
            false_positives.append(
                (record, probability)
            )

    print(f"False negatives: {len(false_negatives)}")
    print(f"False positives: {len(false_positives)}")

    for label, examples in [
        ("False negative examples", false_negatives),
        ("False positive examples", false_positives),
    ]:
        print(f"\n{label}:")

        for record, probability in examples[:maximum_examples]:
            print(
                f"  customer_id={record.customer_id}, "
                f"actual={record.churned}, "
                f"probability={probability:.3f}, "
                f"age={record.age:.1f}, "
                f"spend={record.monthly_spend:.1f}, "
                f"tickets={record.support_tickets:.0f}"
            )


# =============================================================================
# 12. BASELINE COMPARISON
# =============================================================================

"""
A sophisticated model should be compared against a baseline.

For imbalanced classification, a majority-class baseline can reveal whether a
model is producing useful improvement over a trivial strategy.

A baseline does not need to be complex. Its purpose is to establish a minimum
reference level.
"""


def majority_class_baseline(
    training_targets: Sequence[int],
    number_of_predictions: int,
) -> List[int]:
    if not training_targets:
        raise ValueError("Training targets cannot be empty")

    majority_class = (
        1
        if sum(training_targets) >= len(training_targets) / 2
        else 0
    )

    return [majority_class] * number_of_predictions


# =============================================================================
# 13. PREPROCESSING PIPELINE
# =============================================================================

"""
Production systems should apply training and inference transformations
consistently.

A common failure occurs when:
    Training uses one preprocessing implementation,
    Deployment uses a different implementation.

This is called training-serving skew.

The following class stores preprocessing parameters and applies them
consistently.
"""


@dataclass
class ChurnPreprocessor:
    feature_names: List[str]
    medians: Dict[str, float]
    scaler: StandardScaler

    @classmethod
    def fit(
        cls,
        training_records: Sequence[CustomerRecord],
    ) -> "ChurnPreprocessor":
        medians = calculate_feature_medians(training_records)

        imputed_records = impute_missing_values(
            training_records,
            medians,
        )

        scaler = StandardScaler.fit(
            imputed_records,
            FEATURE_NAMES,
        )

        return cls(
            feature_names=list(FEATURE_NAMES),
            medians=medians,
            scaler=scaler,
        )

    def transform(
        self,
        records: Sequence[CustomerRecord],
    ) -> List[List[float]]:
        imputed = impute_missing_values(
            records,
            self.medians,
        )

        return self.scaler.transform(
            imputed,
            self.feature_names,
        )


# =============================================================================
# 14. MODEL SERIALIZATION
# =============================================================================

"""
A trained model must normally be persisted for deployment.

The deployed artifact should include:
    - model parameters
    - feature order
    - preprocessing parameters
    - version metadata

Feature order is critical.

If a model expects:

    [age, spend, tickets]

but receives:

    [tickets, age, spend]

predictions can be silently incorrect.
"""


@dataclass
class ModelArtifact:
    model_version: str
    feature_names: List[str]
    weights: List[float]
    bias: float
    medians: Dict[str, float]
    means: Dict[str, float]
    standard_deviations: Dict[str, float]
    classification_threshold: float
    created_at_unix_time: float


def save_model_artifact(
    file_path: Path,
    model: LogisticRegressionModel,
    preprocessor: ChurnPreprocessor,
    threshold: float,
    model_version: str = "1.0.0",
) -> None:
    if model.weights is None:
        raise RuntimeError("Cannot save an untrained model")

    artifact = ModelArtifact(
        model_version=model_version,
        feature_names=preprocessor.feature_names,
        weights=model.weights,
        bias=model.bias,
        medians=preprocessor.medians,
        means=preprocessor.scaler.means,
        standard_deviations=(
            preprocessor.scaler.standard_deviations
        ),
        classification_threshold=threshold,
        created_at_unix_time=time.time(),
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(artifact),
            file,
            indent=2,
        )


def load_model_artifact(
    file_path: Path,
) -> Tuple[
    LogisticRegressionModel,
    ChurnPreprocessor,
    float,
]:
    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    model = LogisticRegressionModel()
    model.weights = list(data["weights"])
    model.bias = float(data["bias"])

    scaler = StandardScaler(
        means={
            key: float(value)
            for key, value in data["means"].items()
        },
        standard_deviations={
            key: float(value)
            for key, value in data[
                "standard_deviations"
            ].items()
        },
    )

    preprocessor = ChurnPreprocessor(
        feature_names=list(data["feature_names"]),
        medians={
            key: float(value)
            for key, value in data["medians"].items()
        },
        scaler=scaler,
    )

    threshold = float(
        data["classification_threshold"]
    )

    return model, preprocessor, threshold


# =============================================================================
# 15. INFERENCE INPUT VALIDATION
# =============================================================================

"""
Production inference must not assume that incoming data is valid.

Common production inputs may contain:
    - missing fields
    - strings instead of numbers
    - unexpected categories
    - extremely large values
    - negative values
    - malicious payloads

Input validation is both a reliability and security concern.

This example uses strict conversion and domain validation.
"""


def create_record_from_input(
    input_data: Dict[str, Any],
) -> CustomerRecord:
    required_features = FEATURE_NAMES

    missing_fields = [
        field_name
        for field_name in required_features
        if field_name not in input_data
    ]

    if missing_fields:
        raise ValueError(
            "Missing required fields: "
            + ", ".join(missing_fields)
        )

    try:
        record = CustomerRecord(
            customer_id=int(
                input_data.get("customer_id", 1)
            ),
            age=float(input_data["age"]),
            monthly_spend=float(
                input_data["monthly_spend"]
            ),
            support_tickets=float(
                input_data["support_tickets"]
            ),
            months_as_customer=float(
                input_data["months_as_customer"]
            ),
            uses_premium_plan=float(
                input_data["uses_premium_plan"]
            ),
            churned=None,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Input contains values that cannot be converted "
            "to required numeric types."
        ) from error

    validation = validate_customer_record(record)

    # churned is None for inference, which is acceptable.
    if not validation.valid:
        raise ValueError(
            "Invalid input: "
            + "; ".join(validation.errors)
        )

    return record


def predict_customer_churn(
    input_data: Dict[str, Any],
    model: LogisticRegressionModel,
    preprocessor: ChurnPreprocessor,
    threshold: float,
) -> Dict[str, Any]:
    record = create_record_from_input(input_data)

    transformed_features = preprocessor.transform(
        [record]
    )

    probability = model.predict_probability(
        transformed_features[0]
    )

    prediction = (
        1 if probability >= threshold else 0
    )

    return {
        "customer_id": record.customer_id,
        "churn_probability": probability,
        "predicted_churn": prediction,
        "classification_threshold": threshold,
    }


# =============================================================================
# 16. SIMPLE DEPLOYMENT INTERFACE
# =============================================================================

"""
Deployment means making a trained model available for inference.

Common deployment patterns:

Batch inference:
    Predictions are generated periodically for many records.

Online inference:
    Applications request predictions in real time.

Streaming inference:
    Predictions are generated continuously from event streams.

Embedded inference:
    A model runs directly inside a device or application.

The function below demonstrates batch inference without requiring a web
framework.

A production HTTP API could wrap the same validated inference function.
"""


def run_batch_inference(
    input_rows: Sequence[Dict[str, Any]],
    model: LogisticRegressionModel,
    preprocessor: ChurnPreprocessor,
    threshold: float,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for row in input_rows:
        try:
            result = predict_customer_churn(
                row,
                model,
                preprocessor,
                threshold,
            )
            results.append(result)
        except ValueError as error:
            results.append(
                {
                    "input": row,
                    "error": str(error),
                }
            )

    return results


# =============================================================================
# 17. MONITORING CONCEPTS
# =============================================================================

"""
Deployment is not the end of the workflow.

Models can degrade because the real world changes.

Important monitoring categories:

Data drift:
    Input distributions change.

Concept drift:
    The relationship between inputs and targets changes.

Prediction drift:
    Model output distributions change.

Performance degradation:
    Accuracy, recall, or other metrics decline after labels become available.

Operational failures:
    Timeouts, crashes, dependency failures, unavailable storage, or invalid
    incoming data.

Monitoring must distinguish between:
    What is available immediately?
        Input and prediction statistics.

    What is available later?
        True performance, after ground-truth labels arrive.
"""


def calculate_mean_feature_values(
    records: Sequence[CustomerRecord],
    feature_names: Sequence[str],
) -> Dict[str, float]:
    means: Dict[str, float] = {}

    for feature_name in feature_names:
        values = [
            float(get_feature_value(record, feature_name))
            for record in records
            if get_feature_value(record, feature_name)
            is not None
        ]

        means[feature_name] = (
            statistics.mean(values)
            if values
            else float("nan")
        )

    return means


def compare_feature_means(
    reference_records: Sequence[CustomerRecord],
    current_records: Sequence[CustomerRecord],
) -> Dict[str, float]:
    """
    A simple drift indicator.

    Mean differences alone are not sufficient for production drift detection.
    Distributional tests and monitoring systems are commonly used.

    This educational metric demonstrates the underlying idea.
    """

    reference_means = calculate_mean_feature_values(
        reference_records,
        FEATURE_NAMES,
    )

    current_means = calculate_mean_feature_values(
        current_records,
        FEATURE_NAMES,
    )

    return {
        feature_name: (
            current_means[feature_name]
            - reference_means[feature_name]
        )
        for feature_name in FEATURE_NAMES
    }


# =============================================================================
# 18. TESTING
# =============================================================================

"""
Machine-learning systems require more than model evaluation.

Useful test categories include:

Unit tests:
    Verify individual functions.

Integration tests:
    Verify components work together.

Data validation tests:
    Verify expected schemas and ranges.

Regression tests:
    Detect unexpected changes in behavior.

Inference tests:
    Ensure deployed models can process valid inputs.

This script contains lightweight executable assertions.
"""


def run_self_tests() -> None:
    print("\n" + "=" * 80)
    print("SELF TESTS")
    print("=" * 80)

    # Sigmoid properties.
    assert abs(sigmoid(0) - 0.5) < 1e-12
    assert sigmoid(100) > 0.999
    assert sigmoid(-100) < 0.001

    # Cross-entropy should be low for accurate confident predictions.
    good_loss = binary_cross_entropy(
        [1, 0],
        [0.99, 0.01],
    )

    bad_loss = binary_cross_entropy(
        [1, 0],
        [0.01, 0.99],
    )

    assert good_loss < bad_loss

    # Metric correctness.
    metrics = classification_metrics(
        [1, 0, 1, 0],
        [1, 0, 0, 0],
    )

    assert abs(metrics["accuracy"] - 0.75) < 1e-12
    assert abs(metrics["precision"] - 1.0) < 1e-12
    assert abs(metrics["recall"] - 0.5) < 1e-12

    # Validation should reject impossible age.
    invalid_record = CustomerRecord(
        customer_id=1,
        age=500,
        monthly_spend=100,
        support_tickets=1,
        months_as_customer=10,
        uses_premium_plan=1,
        churned=0,
    )

    validation = validate_customer_record(
        invalid_record
    )

    assert not validation.valid

    print("All self-tests passed.")


# =============================================================================
# 19. CSV EXPORT EXAMPLE
# =============================================================================

"""
Batch systems frequently exchange data through files.

CSV is convenient but has limitations:
    - weak type representation
    - no nested structure
    - schema ambiguity
    - quoting issues

JSON is more flexible for nested data.

Columnar formats and databases are often preferred for large production
datasets.
"""


def export_predictions_to_csv(
    file_path: Path,
    predictions: Sequence[Dict[str, Any]],
) -> None:
    field_names = [
        "customer_id",
        "churn_probability",
        "predicted_churn",
        "classification_threshold",
        "error",
    ]

    with file_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=field_names,
            extrasaction="ignore",
        )

        writer.writeheader()

        for prediction in predictions:
            writer.writerow(prediction)


# =============================================================================
# 20. END-TO-END WORKFLOW
# =============================================================================


def main() -> None:
    # -------------------------------------------------------------------------
    # Step 1: Define the problem.
    # -------------------------------------------------------------------------
    demonstrate_problem_definition()

    # -------------------------------------------------------------------------
    # Step 2: Collect data.
    # -------------------------------------------------------------------------
    raw_records = demonstrate_data_collection()

    # -------------------------------------------------------------------------
    # Step 3: Understand data quality and distributions.
    # -------------------------------------------------------------------------
    demonstrate_data_understanding(raw_records)

    # -------------------------------------------------------------------------
    # Step 4: Validate and clean records.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. DATA CLEANING")
    print("=" * 80)

    valid_records, invalid_records = (
        filter_invalid_records(raw_records)
    )

    print(
        f"Records before validation: {len(raw_records)}"
    )
    print(
        f"Invalid records removed: {len(invalid_records)}"
    )
    print(
        f"Records after validation: {len(valid_records)}"
    )

    unique_records = remove_duplicates(
        valid_records
    )

    print(
        f"Duplicate records removed: "
        f"{len(valid_records) - len(unique_records)}"
    )
    print(
        f"Records after deduplication: "
        f"{len(unique_records)}"
    )

    # Records without targets cannot be used for supervised training.
    labeled_records = [
        record
        for record in unique_records
        if record.churned is not None
    ]

    print(
        f"Labeled records available for training: "
        f"{len(labeled_records)}"
    )

    # -------------------------------------------------------------------------
    # Step 5: Split before fitting preprocessing parameters.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 80)

    (
        training_records,
        validation_records,
        test_records,
    ) = train_validation_test_split(
        labeled_records
    )

    print(
        f"Training records: {len(training_records)}"
    )
    print(
        f"Validation records: {len(validation_records)}"
    )
    print(f"Test records: {len(test_records)}")

    # -------------------------------------------------------------------------
    # Step 6: Fit preprocessing using training data only.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. PREPROCESSING")
    print("=" * 80)

    preprocessor = ChurnPreprocessor.fit(
        training_records
    )

    print("Median values used for imputation:")

    for feature_name, median_value in (
        preprocessor.medians.items()
    ):
        print(
            f"  {feature_name}: {median_value:.4f}"
        )

    training_features = preprocessor.transform(
        training_records
    )

    validation_features = preprocessor.transform(
        validation_records
    )

    test_features = preprocessor.transform(
        test_records
    )

    training_targets = [
        int(record.churned)
        for record in training_records
    ]

    validation_targets = [
        int(record.churned)
        for record in validation_records
    ]

    test_targets = [
        int(record.churned)
        for record in test_records
    ]

    # -------------------------------------------------------------------------
    # Step 7: Establish a baseline.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("7. BASELINE MODEL")
    print("=" * 80)

    baseline_predictions = majority_class_baseline(
        training_targets,
        len(validation_targets),
    )

    baseline_metrics = classification_metrics(
        validation_targets,
        baseline_predictions,
    )

    for metric_name, metric_value in (
        baseline_metrics.items()
    ):
        print(
            f"Baseline {metric_name}: "
            f"{metric_value:.4f}"
        )

    # -------------------------------------------------------------------------
    # Step 8: Train model.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("8. MODEL TRAINING")
    print("=" * 80)

    model = LogisticRegressionModel(
        learning_rate=0.05,
        epochs=1500,
    )

    model.fit(
        training_features,
        training_targets,
        verbose=True,
    )

    print("\nLearned model parameters:")

    for feature_name, weight in zip(
        FEATURE_NAMES,
        model.weights or [],
    ):
        print(
            f"  Weight for {feature_name}: "
            f"{weight:.4f}"
        )

    print(f"  Bias: {model.bias:.4f}")

    # -------------------------------------------------------------------------
    # Step 9: Evaluate on validation data.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("9. VALIDATION EVALUATION")
    print("=" * 80)

    validation_probabilities = (
        model.predict_probabilities(
            validation_features
        )
    )

    default_validation_predictions = [
        1 if probability >= 0.5 else 0
        for probability in validation_probabilities
    ]

    default_metrics = classification_metrics(
        validation_targets,
        default_validation_predictions,
    )

    validation_auc = approximate_roc_auc(
        validation_targets,
        validation_probabilities,
    )

    for metric_name, metric_value in (
        default_metrics.items()
    ):
        print(
            f"Threshold 0.50 {metric_name}: "
            f"{metric_value:.4f}"
        )

    print(f"Validation ROC-AUC: {validation_auc:.4f}")

    # -------------------------------------------------------------------------
    # Step 10: Select a classification threshold.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("10. THRESHOLD SELECTION")
    print("=" * 80)

    (
        selected_threshold,
        selected_validation_metrics,
    ) = find_best_threshold_by_f1(
        validation_targets,
        validation_probabilities,
    )

    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    for metric_name, metric_value in (
        selected_validation_metrics.items()
    ):
        print(
            f"Validation {metric_name}: "
            f"{metric_value:.4f}"
        )

    # -------------------------------------------------------------------------
    # Step 11: Final test evaluation.
    #
    # The threshold was selected using validation data.
    # The test set is evaluated only after model decisions are finalized.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("11. FINAL TEST EVALUATION")
    print("=" * 80)

    test_probabilities = model.predict_probabilities(
        test_features
    )

    test_predictions = [
        1
        if probability >= selected_threshold
        else 0
        for probability in test_probabilities
    ]

    test_metrics = classification_metrics(
        test_targets,
        test_predictions,
    )

    test_auc = approximate_roc_auc(
        test_targets,
        test_probabilities,
    )

    matrix = build_confusion_matrix(
        test_targets,
        test_predictions,
    )

    print("Confusion matrix:")
    print(
        f"  True Positive:  "
        f"{matrix.true_positive}"
    )
    print(
        f"  True Negative:  "
        f"{matrix.true_negative}"
    )
    print(
        f"  False Positive: "
        f"{matrix.false_positive}"
    )
    print(
        f"  False Negative: "
        f"{matrix.false_negative}"
    )

    for metric_name, metric_value in (
        test_metrics.items()
    ):
        print(
            f"Test {metric_name}: "
            f"{metric_value:.4f}"
        )

    print(f"Test ROC-AUC: {test_auc:.4f}")

    # -------------------------------------------------------------------------
    # Step 12: Error analysis.
    # -------------------------------------------------------------------------
    print_error_analysis(
        test_records,
        test_probabilities,
        test_predictions,
    )

    # -------------------------------------------------------------------------
    # Step 13: Save model artifact.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("12. MODEL ARTIFACT")
    print("=" * 80)

    artifact_path = Path(
        "churn_model_artifact.json"
    )

    save_model_artifact(
        artifact_path,
        model,
        preprocessor,
        selected_threshold,
    )

    print(
        f"Model artifact saved to: "
        f"{artifact_path.resolve()}"
    )

    # -------------------------------------------------------------------------
    # Step 14: Load artifact and perform inference.
    # -------------------------------------------------------------------------
    loaded_model, loaded_preprocessor, loaded_threshold = (
        load_model_artifact(artifact_path)
    )

    example_input = {
        "customer_id": 9999,
        "age": 35,
        "monthly_spend": 85,
        "support_tickets": 8,
        "months_as_customer": 4,
        "uses_premium_plan": 0,
    }

    inference_result = predict_customer_churn(
        example_input,
        loaded_model,
        loaded_preprocessor,
        loaded_threshold,
    )

    print("\nSingle prediction:")
    print(
        json.dumps(
            inference_result,
            indent=2,
        )
    )

    # -------------------------------------------------------------------------
    # Step 15: Batch inference with error handling.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("13. BATCH INFERENCE")
    print("=" * 80)

    batch_inputs = [
        example_input,
        {
            "customer_id": 10000,
            "age": 55,
            "monthly_spend": 400,
            "support_tickets": 0,
            "months_as_customer": 72,
            "uses_premium_plan": 1,
        },
        {
            # Invalid age demonstrates production validation.
            "customer_id": 10001,
            "age": -20,
            "monthly_spend": 50,
            "support_tickets": 3,
            "months_as_customer": 2,
            "uses_premium_plan": 0,
        },
    ]

    batch_results = run_batch_inference(
        batch_inputs,
        loaded_model,
        loaded_preprocessor,
        loaded_threshold,
    )

    for result in batch_results:
        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    prediction_csv_path = Path(
        "batch_predictions.csv"
    )

    export_predictions_to_csv(
        prediction_csv_path,
        batch_results,
    )

    print(
        f"Batch results exported to: "
        f"{prediction_csv_path.resolve()}"
    )

    # -------------------------------------------------------------------------
    # Step 16: Demonstrate simple data-drift monitoring.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("14. MONITORING AND DRIFT")
    print("=" * 80)

    simulated_current_records = (
        generate_synthetic_customer_data(
            number_of_records=200,
            random_seed=999,
        )
    )

    current_valid_records, _ = (
        filter_invalid_records(
            simulated_current_records
        )
    )

    current_records = [
        record
        for record in current_valid_records
        if record.churned is not None
    ]

    drift_indicators = compare_feature_means(
        training_records,
        current_records,
    )

    print(
        "Difference between current and training "
        "feature means:"
    )

    for feature_name, difference in (
        drift_indicators.items()
    ):
        print(
            f"  {feature_name}: "
            f"{difference:.4f}"
        )

    print(
        "\nInterpretation: a large difference may indicate "
        "distribution change, but mean comparison alone is not "
        "sufficient for production drift detection."
    )

    # -------------------------------------------------------------------------
    # Step 17: Run tests.
    # -------------------------------------------------------------------------
    run_self_tests()

    print("\n" + "=" * 80)
    print("AI DEVELOPMENT WORKFLOW COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
