"""
Data Understanding: Features, Labels, Observations, Variables, and Target Variables

A self-contained study program covering:
- observations and datasets
- variables and data types
- features and feature matrices
- labels and target variables
- predictors versus targets
- numerical, categorical, ordinal, binary, temporal, textual, and identifier variables
- supervised versus unsupervised learning
- feature/target separation
- data validation and leakage prevention
- missing values, duplicates, inconsistent categories, and invalid values
- train/validation/test splits
- classification, regression, and multi-output targets
- feature engineering and transformations
- correlation versus causation
- data dictionaries
- exploratory inspection
- model-ready dataset design
- production considerations
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Any, Iterable
import math
import random


# ---------------------------------------------------------------------------
# 1. FOUNDATIONAL TERMINOLOGY
# ---------------------------------------------------------------------------

print("=" * 80)
print("DATA UNDERSTANDING: FOUNDATIONS")
print("=" * 80)


@dataclass
class Observation:
    """
    An observation is one recorded instance in a dataset.

    In a tabular customer dataset, one observation might represent one
    customer. In a transaction dataset, it might represent one transaction.
    The meaning of a row depends on the unit of analysis.
    """

    customer_id: str
    age: int
    annual_income: float
    city: str
    membership: str
    visits_per_month: int
    purchased: int


observations = [
    Observation("C001", 28, 52000, "Lucknow", "Basic", 3, 0),
    Observation("C002", 35, 76000, "Delhi", "Premium", 8, 1),
    Observation("C003", 42, 91000, "Mumbai", "Premium", 10, 1),
    Observation("C004", 23, 41000, "Lucknow", "Basic", 2, 0),
]

for observation in observations:
    print(observation)


# ---------------------------------------------------------------------------
# 2. VARIABLES
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VARIABLES")
print("=" * 80)

"""
A variable is a characteristic that can take different values across
observations.

Examples:
    age              -> numerical variable
    annual_income    -> numerical variable
    city             -> categorical variable
    membership       -> categorical variable
    purchased        -> binary target variable

A variable is not necessarily a machine-learning feature. A target variable
is also a variable, but it is normally separated from the input features.
"""

variable_examples = {
    "age": 35,
    "annual_income": 76000.0,
    "city": "Delhi",
    "membership": "Premium",
    "purchased": 1,
}

for name, value in variable_examples.items():
    print(f"{name:20} value={value!r:15} Python type={type(value).__name__}")


# ---------------------------------------------------------------------------
# 3. FEATURES, LABELS, AND TARGETS
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FEATURES, LABELS, AND TARGET VARIABLES")
print("=" * 80)

"""
A feature is an input variable used to make a prediction.

A target variable is the outcome we want to predict.

A label commonly refers to the known target value attached to an observation,
especially in supervised classification. In many contexts "label" and
"target" are used interchangeably, although "target" is broader.

For the customer purchase example:

Features:
    age
    annual_income
    city
    membership
    visits_per_month

Target:
    purchased

customer_id is an identifier and should generally not be treated as a
predictive feature simply because it is present in the table.
"""

feature_names = [
    "age",
    "annual_income",
    "city",
    "membership",
    "visits_per_month",
]

target_name = "purchased"

print("Features:", feature_names)
print("Target:", target_name)
print("Identifier:", "customer_id")


# ---------------------------------------------------------------------------
# 4. REPRESENTING DATA AS RECORDS
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("TABULAR DATA")
print("=" * 80)

dataset = [
    {
        "customer_id": "C001",
        "age": 28,
        "annual_income": 52000.0,
        "city": "Lucknow",
        "membership": "Basic",
        "visits_per_month": 3,
        "purchased": 0,
    },
    {
        "customer_id": "C002",
        "age": 35,
        "annual_income": 76000.0,
        "city": "Delhi",
        "membership": "Premium",
        "visits_per_month": 8,
        "purchased": 1,
    },
    {
        "customer_id": "C003",
        "age": 42,
        "annual_income": 91000.0,
        "city": "Mumbai",
        "membership": "Premium",
        "visits_per_month": 10,
        "purchased": 1,
    },
    {
        "customer_id": "C004",
        "age": 23,
        "annual_income": 41000.0,
        "city": "Lucknow",
        "membership": "Basic",
        "visits_per_month": 2,
        "purchased": 0,
    },
]

print("Number of observations:", len(dataset))
print("Variables per observation:", len(dataset[0]))

for row in dataset:
    print(row)


# ---------------------------------------------------------------------------
# 5. UNIT OF ANALYSIS
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("UNIT OF ANALYSIS")
print("=" * 80)

"""
The unit of analysis tells us what one observation represents.

Examples:
    one customer
    one transaction
    one product
    one hospital visit
    one loan application
    one web session
    one sensor reading
    one day

This distinction is critical.

If a dataset contains one row per transaction, a customer can appear in many
rows. Calling each row a "customer" would be incorrect.

The same variable can also have different meanings depending on the unit.
For example, "amount" could mean:
    transaction amount
    monthly spending
    annual spending
"""

transactions = [
    {"transaction_id": "T001", "customer_id": "C001", "amount": 1200},
    {"transaction_id": "T002", "customer_id": "C001", "amount": 800},
    {"transaction_id": "T003", "customer_id": "C002", "amount": 2200},
]

print("Each row above represents one transaction.")
print("Customer C001 appears twice because one customer can make multiple transactions.")


# ---------------------------------------------------------------------------
# 6. VARIABLE CLASSIFICATIONS
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VARIABLE CLASSIFICATIONS")
print("=" * 80)

"""
Common classifications:

1. Numerical / quantitative
   - continuous: temperature, weight, income
   - discrete: number of purchases, number of children

2. Categorical / qualitative
   - nominal: city, color, department
   - ordinal: low, medium, high

3. Binary
   - exactly two meaningful states
   - yes/no, fraud/not fraud

4. Temporal
   - date, time, timestamp, duration

5. Text
   - reviews, comments, support tickets

6. Identifier
   - customer ID, transaction ID, account number

7. Target
   - the outcome being predicted

The categories can overlap conceptually. A binary target is simultaneously a
variable and a categorical variable.
"""

variable_schema = {
    "customer_id": "identifier",
    "age": "discrete numerical",
    "annual_income": "continuous numerical",
    "city": "nominal categorical",
    "membership": "nominal categorical",
    "visits_per_month": "discrete numerical",
    "purchased": "binary target",
}

for variable, classification in variable_schema.items():
    print(f"{variable:20} -> {classification}")


# ---------------------------------------------------------------------------
# 7. CONTINUOUS VS DISCRETE
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("CONTINUOUS VS DISCRETE NUMERICAL VARIABLES")
print("=" * 80)

continuous_values = [52.1, 52.15, 52.157, 52.1578]
discrete_values = [0, 1, 2, 3, 4]

print("Continuous:", continuous_values)
print("Discrete:", discrete_values)

"""
A continuous quantity can theoretically take values across an interval.
A discrete quantity consists of countable values.

Real datasets can store continuous measurements with finite precision, but
that does not make the underlying concept discrete.
"""


# ---------------------------------------------------------------------------
# 8. NOMINAL VS ORDINAL CATEGORICAL VARIABLES
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("NOMINAL VS ORDINAL")
print("=" * 80)

nominal = ["Delhi", "Mumbai", "Lucknow"]
ordinal = ["Low", "Medium", "High"]

print("Nominal categories:", nominal)
print("Ordinal categories:", ordinal)

"""
Nominal categories have no intrinsic ordering.

Ordinal categories have meaningful order, but the distance between categories
is not necessarily equal.

For example:
    Low < Medium < High

does not imply that the numerical distance from Low to Medium is identical to
the distance from Medium to High.
"""


# ---------------------------------------------------------------------------
# 9. IDENTIFIERS ARE NOT AUTOMATICALLY FEATURES
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("IDENTIFIERS")
print("=" * 80)

customer_ids = ["C001", "C002", "C003", "C004"]

print("Identifiers:", customer_ids)

"""
Identifiers distinguish observations.

Examples:
    customer_id
    employee_id
    transaction_id
    invoice_number

A unique identifier usually has no useful predictive meaning by itself.

A dangerous mistake is to encode IDs as numerical values and allow a model
to interpret them as quantities.

For example:
    customer_id = 1000
    customer_id = 2000

does NOT mean customer 2000 is twice customer 1000.

There are specialized cases where an identifier contains meaningful structure,
but that structure must be deliberately extracted and validated.
"""


# ---------------------------------------------------------------------------
# 10. SEPARATING FEATURES AND TARGET
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FEATURE-TARGET SEPARATION")
print("=" * 80)


def split_features_target(
    rows: list[dict[str, Any]],
    feature_columns: list[str],
    target_column: str,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Separate predictors from the target while validating column names."""

    if not rows:
        raise ValueError("Dataset cannot be empty.")

    available_columns = set(rows[0])

    required_columns = set(feature_columns) | {target_column}
    missing_columns = required_columns - available_columns

    if missing_columns:
        raise KeyError(f"Missing columns: {sorted(missing_columns)}")

    if target_column in feature_columns:
        raise ValueError("Target must not also be listed as a feature.")

    features = [
        {column: row[column] for column in feature_columns}
        for row in rows
    ]

    targets = [row[target_column] for row in rows]

    return features, targets


X, y = split_features_target(
    dataset,
    feature_names,
    target_name,
)

print("X:")
for row in X:
    print(row)

print("y:", y)


# ---------------------------------------------------------------------------
# 11. TARGET TYPES
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("TARGET TYPES")
print("=" * 80)


def classify_target(values: Iterable[Any]) -> str:
    values = list(values)

    if not values:
        return "empty"

    unique_values = set(values)

    if all(isinstance(value, bool) for value in values):
        return "binary categorical"

    if all(isinstance(value, (int, float)) and not isinstance(value, bool)
           for value in values):
        if len(unique_values) == 2:
            return "binary numerical encoding"
        return "continuous/count numerical target"

    return "categorical target"


binary_target = [0, 1, 0, 1]
regression_target = [120.5, 250.0, 180.75, 90.0]
multiclass_target = ["low", "medium", "high", "medium"]

print("Binary:", classify_target(binary_target))
print("Regression:", classify_target(regression_target))
print("Multiclass:", classify_target(multiclass_target))


# ---------------------------------------------------------------------------
# 12. CLASSIFICATION VS REGRESSION
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("CLASSIFICATION VS REGRESSION")
print("=" * 80)

"""
Classification predicts categories.

Examples:
    spam / not spam
    fraud / legitimate
    low / medium / high
    cat / dog / horse

Regression predicts a numerical quantity.

Examples:
    house price
    temperature
    revenue
    demand

The data type alone does not always determine the modeling task. A numerical
value can represent a category, and a category can sometimes be encoded
numerically.
"""

classification_example = {
    "features": {"age": 35, "income": 76000},
    "target": "purchased",
    "target_values": [0, 1],
}

regression_example = {
    "features": {"area_sqft": 1500, "bedrooms": 3},
    "target": "price",
    "target_values": [6500000.0],
}

print("Classification example:", classification_example)
print("Regression example:", regression_example)


# ---------------------------------------------------------------------------
# 13. MULTICLASS AND MULTI-LABEL TARGETS
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("MULTICLASS VS MULTI-LABEL")
print("=" * 80)

multiclass_rows = [
    {"ticket": "A", "category": "billing"},
    {"ticket": "B", "category": "technical"},
    {"ticket": "C", "category": "account"},
]

multi_label_rows = [
    {"document": "D1", "labels": ["finance", "risk"]},
    {"document": "D2", "labels": ["technology"]},
    {"document": "D3", "labels": ["finance", "technology"]},
]

print("Multiclass target:", multiclass_rows)
print("Multi-label target:", multi_label_rows)

"""
Multiclass:
    one observation receives one class from several possible classes.

Multi-label:
    one observation can receive multiple labels simultaneously.
"""


# ---------------------------------------------------------------------------
# 14. MULTI-OUTPUT TARGETS
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("MULTI-OUTPUT TARGETS")
print("=" * 80)

multi_output_targets = [
    {"temperature": 25.4, "humidity": 61.0},
    {"temperature": 26.1, "humidity": 59.5},
]

for target in multi_output_targets:
    print(target)

"""
Some systems predict multiple outputs for one observation.

Example:
    predict temperature AND humidity
    predict price AND demand
    predict multiple disease indicators

The distinction between multi-output and multi-label depends on the nature of
the outputs.
"""


# ---------------------------------------------------------------------------
# 15. MISSING VALUES
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

messy_dataset = [
    {"age": 28, "income": 52000.0, "city": "Lucknow"},
    {"age": None, "income": 76000.0, "city": "Delhi"},
    {"age": 42, "income": None, "city": "Mumbai"},
]

for row in messy_dataset:
    missing = [key for key, value in row.items() if value is None]
    print(row, "missing:", missing)

"""
Missingness must be understood rather than blindly replaced.

Possible interpretations:
    value was not collected
    value was unknown
    value was not applicable
    system failed
    user deliberately withheld it

Different causes may require different treatment.
"""


# ---------------------------------------------------------------------------
# 16. DUPLICATE OBSERVATIONS
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DUPLICATES")
print("=" * 80)

duplicate_rows = [
    {"id": 1, "age": 25, "income": 40000},
    {"id": 2, "age": 30, "income": 60000},
    {"id": 2, "age": 30, "income": 60000},
]


def find_duplicate_rows(rows: list[dict[str, Any]]) -> list[int]:
    seen: set[tuple[Any, ...]] = set()
    duplicates: list[int] = []

    for index, row in enumerate(rows):
        signature = tuple(sorted(row.items()))

        if signature in seen:
            duplicates.append(index)
        else:
            seen.add(signature)

    return duplicates


print("Duplicate row indices:", find_duplicate_rows(duplicate_rows))


# ---------------------------------------------------------------------------
# 17. INVALID VALUES
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)


def validate_customer_row(row: dict[str, Any]) -> list[str]:
    errors = []

    age = row.get("age")
    income = row.get("annual_income")
    membership = row.get("membership")
    purchased = row.get("purchased")

    if age is not None and not 0 <= age <= 120:
        errors.append("age must be between 0 and 120")

    if income is not None and income < 0:
        errors.append("annual_income cannot be negative")

    if membership not in {"Basic", "Premium"}:
        errors.append("membership must be Basic or Premium")

    if purchased not in {0, 1}:
        errors.append("purchased must be 0 or 1")

    return errors


valid_row = dataset[0]
invalid_row = {
    "age": -4,
    "annual_income": -100,
    "membership": "Unknown",
    "purchased": 7,
}

print("Valid row errors:", validate_customer_row(valid_row))
print("Invalid row errors:", validate_customer_row(invalid_row))


# ---------------------------------------------------------------------------
# 18. DATA DICTIONARY
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DATA DICTIONARY")
print("=" * 80)

data_dictionary = {
    "customer_id": {
        "role": "identifier",
        "type": "string",
        "description": "Unique customer identifier",
    },
    "age": {
        "role": "feature",
        "type": "integer",
        "description": "Customer age in years",
    },
    "annual_income": {
        "role": "feature",
        "type": "float",
        "description": "Annual income in currency units",
    },
    "city": {
        "role": "feature",
        "type": "nominal categorical",
        "description": "Customer's city",
    },
    "membership": {
        "role": "feature",
        "type": "nominal categorical",
        "description": "Membership tier",
    },
    "visits_per_month": {
        "role": "feature",
        "type": "integer",
        "description": "Average monthly visits",
    },
    "purchased": {
        "role": "target",
        "type": "binary",
        "description": "Whether the customer purchased",
    },
}

for name, definition in data_dictionary.items():
    print(f"\n{name}")
    for key, value in definition.items():
        print(f"  {key}: {value}")


# ---------------------------------------------------------------------------
# 19. BASIC DATA PROFILING
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("BASIC DATA PROFILING")
print("=" * 80)


def profile_column(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    values = [row.get(column) for row in rows]
    non_missing = [value for value in values if value is not None]

    profile = {
        "column": column,
        "row_count": len(values),
        "missing_count": len(values) - len(non_missing),
        "unique_count": len(set(non_missing)),
    }

    if non_missing and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in non_missing
    ):
        profile["minimum"] = min(non_missing)
        profile["maximum"] = max(non_missing)
        profile["mean"] = mean(non_missing)

    return profile


for column in dataset[0]:
    print(profile_column(dataset, column))


# ---------------------------------------------------------------------------
# 20. CATEGORICAL FREQUENCIES
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("CATEGORICAL FREQUENCIES")
print("=" * 80)


def value_counts(rows: list[dict[str, Any]], column: str) -> dict[Any, int]:
    counts: dict[Any, int] = {}

    for row in rows:
        value = row.get(column)
        counts[value] = counts.get(value, 0) + 1

    return counts


print("City frequencies:", value_counts(dataset, "city"))
print("Membership frequencies:", value_counts(dataset, "membership"))
print("Target frequencies:", value_counts(dataset, "purchased"))


# ---------------------------------------------------------------------------
# 21. CLASS BALANCE
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("CLASS BALANCE")
print("=" * 80)


def class_distribution(values: list[Any]) -> dict[Any, float]:
    counts = value_counts(
        [{"value": value} for value in values],
        "value",
    )

    total = len(values)

    return {
        category: count / total
        for category, count in counts.items()
    }


distribution = class_distribution([0, 0, 0, 0, 1, 1])
print("Class proportions:", distribution)

"""
A strongly imbalanced target can affect evaluation and model training.

For example:
    99% legitimate
    1% fraud

A model that predicts "legitimate" every time achieves 99% accuracy but
detects no fraud. Accuracy therefore needs to be interpreted in context.
"""


# ---------------------------------------------------------------------------
# 22. DATA LEAKAGE
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DATA LEAKAGE")
print("=" * 80)

"""
Data leakage occurs when information unavailable at prediction time enters
the feature set.

Example:

Suppose the goal is to predict whether a loan applicant will default.

Valid:
    income
    employment history
    credit history available at application time

Potential leakage:
    collection agency action after default
    final account closure status
    recovery amount after default

The model can appear extremely accurate while failing in production.

The key question is:
    "Would this information genuinely be available at the moment the
     prediction must be made?"
"""

loan_row = {
    "income": 80000,
    "credit_score": 730,
    "employment_years": 5,
    "collection_action_taken": False,
    "defaulted": False,
}

print("Example loan record:", loan_row)
print("Prediction-time feature review is required before using every field.")


# ---------------------------------------------------------------------------
# 23. TEMPORAL LEAKAGE
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("TEMPORAL DATA")
print("=" * 80)


events = [
    {"timestamp": "2026-01-01T09:00:00", "sales": 10},
    {"timestamp": "2026-01-02T09:00:00", "sales": 13},
    {"timestamp": "2026-01-03T09:00:00", "sales": 15},
]

parsed_events = [
    {
        **event,
        "timestamp": datetime.fromisoformat(event["timestamp"]),
    }
    for event in events
]

for event in parsed_events:
    print(event)

"""
For time-dependent prediction, future information must not leak into the
past.

A random train/test split can be inappropriate for certain time-series
problems because observations from the future may influence training while
earlier observations are evaluated.
"""


# ---------------------------------------------------------------------------
# 24. TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DATASET SPLITTING")
print("=" * 80)


def split_indices(
    number_of_rows: int,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    if number_of_rows < 3:
        raise ValueError("At least three observations are required.")

    if not math.isclose(train_ratio + validation_ratio, 0.85):
        raise ValueError("This demonstration expects a 70/15/15 split.")

    indices = list(range(number_of_rows))
    random.Random(seed).shuffle(indices)

    train_end = round(number_of_rows * train_ratio)
    validation_end = train_end + round(number_of_rows * validation_ratio)

    train = indices[:train_end]
    validation = indices[train_end:validation_end]
    test = indices[validation_end:]

    return train, validation, test


train_indices, validation_indices, test_indices = split_indices(20)

print("Train:", train_indices)
print("Validation:", validation_indices)
print("Test:", test_indices)


# ---------------------------------------------------------------------------
# 25. STRATIFIED CONCEPT
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STRATIFICATION")
print("=" * 80)


def simple_stratified_split(
    labels: list[Any],
    test_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Educational implementation of a basic stratified split.

    Each class is shuffled separately and approximately the same class
    proportions are placed in train and test.
    """

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    groups: dict[Any, list[int]] = {}

    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)

    rng = random.Random(seed)

    train_indices: list[int] = []
    test_indices: list[int] = []

    for indices in groups.values():
        rng.shuffle(indices)

        test_count = max(1, round(len(indices) * test_fraction))

        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices


labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
train, test = simple_stratified_split(labels)

print("Stratified train:", train)
print("Stratified test:", test)
print("Train labels:", [labels[index] for index in train])
print("Test labels:", [labels[index] for index in test])


# ---------------------------------------------------------------------------
# 26. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)


def add_income_per_visit(row: dict[str, Any]) -> dict[str, Any]:
    visits = row["visits_per_month"]

    if visits <= 0:
        raise ValueError("visits_per_month must be positive.")

    enriched = dict(row)
    enriched["income_per_monthly_visit"] = row["annual_income"] / visits

    return enriched


engineered_row = add_income_per_visit(dataset[1])
print(engineered_row)

"""
Feature engineering transforms raw variables into representations that may
better express the underlying problem.

Examples:
    date -> day_of_week
    date -> month
    income and age -> income_per_age
    text -> token counts
    transaction history -> rolling statistics

Feature engineering must respect the information available at prediction
time.
"""


# ---------------------------------------------------------------------------
# 27. CATEGORICAL ENCODING
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("CATEGORICAL ENCODING")
print("=" * 80)


def one_hot_encode(
    rows: list[dict[str, Any]],
    column: str,
) -> list[dict[str, Any]]:
    categories = sorted({row[column] for row in rows})
    encoded_rows = []

    for row in rows:
        transformed = {
            key: value
            for key, value in row.items()
            if key != column
        }

        for category in categories:
            encoded_name = f"{column}__{category}"
            transformed[encoded_name] = int(row[column] == category)

        encoded_rows.append(transformed)

    return encoded_rows


encoded_membership = one_hot_encode(dataset, "membership")

for row in encoded_membership:
    print(row)


# ---------------------------------------------------------------------------
# 28. ORDINAL ENCODING
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("ORDINAL ENCODING")
print("=" * 80)

priority_order = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}

priorities = ["Low", "High", "Medium", "Low"]
encoded_priorities = [priority_order[value] for value in priorities]

print("Original:", priorities)
print("Encoded:", encoded_priorities)

"""
Ordinal encoding is appropriate when order is meaningful.

It can be misleading for nominal categories such as:
    Delhi = 1
    Mumbai = 2
    Lucknow = 3

because the numerical ordering is artificial.
"""


# ---------------------------------------------------------------------------
# 29. NORMALIZATION AND STANDARDIZATION CONCEPTS
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SCALING NUMERICAL FEATURES")
print("=" * 80)


def min_max_scale(values: list[float]) -> list[float]:
    minimum = min(values)
    maximum = max(values)

    if math.isclose(minimum, maximum):
        return [0.0 for _ in values]

    return [
        (value - minimum) / (maximum - minimum)
        for value in values
    ]


def standardize(values: list[float]) -> list[float]:
    average = mean(values)

    variance = mean([
        (value - average) ** 2
        for value in values
    ])

    standard_deviation = math.sqrt(variance)

    if math.isclose(standard_deviation, 0):
        return [0.0 for _ in values]

    return [
        (value - average) / standard_deviation
        for value in values
    ]


income_values = [41000, 52000, 76000, 91000]

print("Original:", income_values)
print("Min-max:", min_max_scale(income_values))
print("Standardized:", standardize(income_values))

"""
Scaling should be fitted using training data and then applied to validation
and test data. Computing scaling parameters from the complete dataset before
splitting can leak information from evaluation data.
"""


# ---------------------------------------------------------------------------
# 30. OUTLIERS
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("OUTLIERS")
print("=" * 80)


def iqr_bounds(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)

    midpoint = len(ordered) // 2

    if len(ordered) % 2:
        lower_half = ordered[:midpoint]
        upper_half = ordered[midpoint + 1:]
    else:
        lower_half = ordered[:midpoint]
        upper_half = ordered[midpoint:]

    q1 = mean(lower_half)
    q3 = mean(upper_half)

    iqr = q3 - q1

    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


income_with_outlier = [40000, 42000, 45000, 47000, 50000, 1000000]
lower, upper = iqr_bounds(income_with_outlier)

print("Lower bound:", lower)
print("Upper bound:", upper)
print(
    "Potential outliers:",
    [
        value
        for value in income_with_outlier
        if value < lower or value > upper
    ],
)

"""
An outlier is not automatically an error.

A very large income might be:
    a valid executive salary
    a data-entry error
    a different population
    an unusual but important observation

Investigate before deleting.
"""


# ---------------------------------------------------------------------------
# 31. CORRELATION
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("CORRELATION")
print("=" * 80)


def pearson_correlation(
    x_values: list[float],
    y_values: list[float],
) -> float:
    if len(x_values) != len(y_values):
        raise ValueError("Both sequences must have equal length.")

    if len(x_values) < 2:
        raise ValueError("At least two observations are required.")

    x_mean = mean(x_values)
    y_mean = mean(y_values)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    )

    x_variation = sum((x - x_mean) ** 2 for x in x_values)
    y_variation = sum((y - y_mean) ** 2 for y in y_values)

    denominator = math.sqrt(x_variation * y_variation)

    if math.isclose(denominator, 0):
        raise ValueError("Correlation is undefined for constant variables.")

    return numerator / denominator


visits = [1, 2, 4, 7, 9]
purchases = [0, 0, 1, 1, 1]

print("Correlation:", pearson_correlation(visits, purchases))

"""
Correlation measures association under a particular statistical definition.
It does not establish causation.

A third variable can influence both variables, selection effects can distort
relationships, and temporal structure can produce misleading associations.
"""


# ---------------------------------------------------------------------------
# 32. TARGET DISTRIBUTION AND FEATURE DISTRIBUTION
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DISTRIBUTION INSPECTION")
print("=" * 80)


def numerical_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Values cannot be empty.")

    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": mean(values),
        "range": max(values) - min(values),
    }


print(
    "Income summary:",
    numerical_summary([row["annual_income"] for row in dataset]),
)

print(
    "Visits summary:",
    numerical_summary([row["visits_per_month"] for row in dataset]),
)


# ---------------------------------------------------------------------------
# 33. FEATURE SELECTION
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

candidate_features = [
    "age",
    "annual_income",
    "city",
    "membership",
    "visits_per_month",
    "customer_id",
]

excluded_features = {
    "customer_id": "identifier without established predictive meaning",
}

selected_features = [
    feature
    for feature in candidate_features
    if feature not in excluded_features
]

print("Candidate features:", candidate_features)
print("Excluded:", excluded_features)
print("Selected:", selected_features)

"""
Feature selection should consider:
    predictive usefulness
    data availability
    leakage
    stability
    redundancy
    interpretability
    cost
    privacy
    fairness
    production latency
"""


# ---------------------------------------------------------------------------
# 34. TRAINING-TIME VERSUS PREDICTION-TIME AVAILABILITY
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("PREDICTION-TIME AVAILABILITY")
print("=" * 80)


@dataclass
class FeatureAvailability:
    name: str
    available_at_prediction: bool
    reason: str


availability = [
    FeatureAvailability(
        "credit_score",
        True,
        "available during loan application",
    ),
    FeatureAvailability(
        "loan_default_status",
        False,
        "known only after the outcome",
    ),
    FeatureAvailability(
        "annual_income",
        True,
        "provided during application",
    ),
]

for item in availability:
    print(item)


# ---------------------------------------------------------------------------
# 35. DATA CONTRACT
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DATA CONTRACT")
print("=" * 80)


@dataclass
class ColumnContract:
    name: str
    required: bool
    allowed_types: tuple[type, ...]
    nullable: bool


customer_contract = [
    ColumnContract("customer_id", True, (str,), False),
    ColumnContract("age", True, (int,), False),
    ColumnContract("annual_income", True, (int, float), False),
    ColumnContract("city", True, (str,), False),
    ColumnContract("membership", True, (str,), False),
    ColumnContract("visits_per_month", True, (int,), False),
    ColumnContract("purchased", True, (int,), False),
]


def validate_contract(
    row: dict[str, Any],
    contract: list[ColumnContract],
) -> list[str]:
    errors = []

    for specification in contract:
        if specification.name not in row:
            if specification.required:
                errors.append(f"missing required field: {specification.name}")
            continue

        value = row[specification.name]

        if value is None:
            if not specification.nullable:
                errors.append(f"{specification.name} cannot be null")
            continue

        if not isinstance(value, specification.allowed_types):
            errors.append(
                f"{specification.name} has invalid type "
                f"{type(value).__name__}"
            )

    return errors


print("Contract errors:", validate_contract(dataset[0], customer_contract))


# ---------------------------------------------------------------------------
# 36. END-TO-END DATA UNDERSTANDING PIPELINE
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("END-TO-END DATA UNDERSTANDING")
print("=" * 80)


def understand_dataset(
    rows: list[dict[str, Any]],
    feature_columns: list[str],
    target_column: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot understand an empty dataset.")

    columns = list(rows[0].keys())

    schema = {}

    for column in columns:
        values = [row.get(column) for row in rows]
        non_missing = [value for value in values if value is not None]

        schema[column] = {
            "observations": len(values),
            "missing": len(values) - len(non_missing),
            "unique": len(set(non_missing)),
            "role": (
                "target"
                if column == target_column
                else "feature"
                if column in feature_columns
                else "other"
            ),
        }

    return {
        "observations": len(rows),
        "variables": len(columns),
        "columns": columns,
        "schema": schema,
        "features": feature_columns,
        "target": target_column,
        "target_distribution": value_counts(rows, target_column),
    }


understanding_report = understand_dataset(
    dataset,
    feature_names,
    target_name,
)

for key, value in understanding_report.items():
    print(f"{key}: {value}")


# ---------------------------------------------------------------------------
# 37. EDGE CASES
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("EDGE CASES")
print("=" * 80)

edge_cases = {
    "empty_dataset": [],
    "constant_feature": [10, 10, 10, 10],
    "single_class_target": [1, 1, 1, 1],
    "missing_target": [0, None, 1],
    "duplicate_ids": ["C1", "C1", "C2"],
}

for name, values in edge_cases.items():
    print(f"{name}: {values}")

"""
Important edge cases include:

- empty datasets
- one-row datasets
- constant features
- constant targets
- all values missing
- missing target labels
- duplicate observations
- duplicate identifiers
- unexpected categories
- invalid numerical ranges
- extreme outliers
- inconsistent units
- timezone inconsistencies
- future information accidentally included in features
- target values represented with inconsistent encodings
"""


# ---------------------------------------------------------------------------
# 38. COMMON MISTAKES
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("COMMON MISTAKES")
print("=" * 80)

mistakes = [
    "Treating every column as a feature",
    "Using the target as an input feature",
    "Using an arbitrary ID as a numerical predictor",
    "Ignoring the unit of analysis",
    "Assuming every integer is continuous",
    "Assuming every number is quantitative",
    "Encoding nominal categories as ordered integers",
    "Removing all outliers without investigation",
    "Filling every missing value with zero",
    "Calculating preprocessing statistics on the full dataset",
    "Allowing future information into training features",
    "Using accuracy alone on severely imbalanced targets",
    "Confusing correlation with causation",
    "Ignoring production-time feature availability",
]

for number, mistake in enumerate(mistakes, start=1):
    print(f"{number:02}. {mistake}")


# ---------------------------------------------------------------------------
# 39. SIMPLE MODEL-LIKE SCORING EXAMPLE
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FEATURES AS INPUTS TO A PREDICTIVE RULE")
print("=" * 80)


def purchase_score(row: dict[str, Any]) -> float:
    """
    Educational scoring rule.

    This is not a trained machine-learning model. It illustrates the
    conceptual relationship:

        features -> transformation/rule -> prediction

    The target is not supplied to the scoring function.
    """

    score = 0.0

    if row["age"] >= 30:
        score += 1.0

    if row["annual_income"] >= 70000:
        score += 1.0

    if row["membership"] == "Premium":
        score += 1.5

    if row["visits_per_month"] >= 6:
        score += 1.5

    return score


for row in dataset:
    score = purchase_score(row)
    prediction = int(score >= 2.5)

    print(
        row["customer_id"],
        "score=",
        score,
        "predicted_purchase=",
        prediction,
        "actual_target=",
        row["purchased"],
    )


# ---------------------------------------------------------------------------
# 40. CONCEPTUAL FEATURE MATRIX
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FEATURE MATRIX")
print("=" * 80)

numeric_feature_names = [
    "age",
    "annual_income",
    "visits_per_month",
]

feature_matrix = [
    [row[name] for name in numeric_feature_names]
    for row in dataset
]

print("Columns:", numeric_feature_names)
print("Rows:")

for matrix_row in feature_matrix:
    print(matrix_row)

"""
For n observations and p numerical features, X can be represented as an
n x p matrix.

The target y is conceptually separate:

X = [feature_1, feature_2, ..., feature_p]
y = target

The exact data structure depends on the language and library.
"""


# ---------------------------------------------------------------------------
# 41. DATASET QUALITY CHECKLIST
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DATA QUALITY CHECKLIST")
print("=" * 80)

quality_checks = [
    ("Rows represent the correct unit of analysis", True),
    ("Columns have documented meanings", True),
    ("Target is clearly identified", True),
    ("Identifiers are distinguished from features", True),
    ("Missing values have been measured", True),
    ("Duplicates have been investigated", True),
    ("Ranges have been validated", True),
    ("Categories have been standardized", True),
    ("Feature availability matches prediction time", True),
    ("Train/test contamination has been considered", True),
]

for check, status in quality_checks:
    print(f"[{'PASS' if status else 'FAIL'}] {check}")


# ---------------------------------------------------------------------------
# 42. PRODUCTION DATA UNDERSTANDING
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("PRODUCTION CONSIDERATIONS")
print("=" * 80)

production_principles = [
    "Define the prediction unit precisely.",
    "Document every feature and target.",
    "Version the data schema.",
    "Validate incoming data before inference.",
    "Monitor missingness and category drift.",
    "Monitor numerical distribution drift.",
    "Ensure feature computation uses prediction-time information only.",
    "Keep preprocessing consistent between training and inference.",
    "Record assumptions about units, timestamps, and time zones.",
    "Monitor target availability delays.",
]

for principle in production_principles:
    print("-", principle)


# ---------------------------------------------------------------------------
# 43. FINAL CONCEPT MAP
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("CONCEPT MAP")
print("=" * 80)

concept_map = """
Dataset
|
+-- Observations (rows)
|   |
|   +-- Each represents a defined unit of analysis
|
+-- Variables (columns)
    |
    +-- Features
    |   |
    |   +-- Numerical
    |   +-- Categorical
    |   +-- Temporal
    |   +-- Text
    |   +-- Engineered
    |
    +-- Target / Label
    |   |
    |   +-- Binary
    |   +-- Multiclass
    |   +-- Multi-label
    |   +-- Regression
    |   +-- Multi-output
    |
    +-- Identifiers / metadata
        |
        +-- Often excluded from predictive features

Reliable modeling begins with correctly understanding what every row and
column represents, when each value is available, and whether the target has
been separated from the predictors without leakage.
"""

print(concept_map)

print("\n" + "=" * 80)
print("END OF DATA UNDERSTANDING STUDY PROGRAM")
print("=" * 80)
