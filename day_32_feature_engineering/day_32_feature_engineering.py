#!/usr/bin/env python3
"""
Feature Engineering: feature creation, transformation, aggregation, and domain features.

This self-contained example builds a small customer-transaction dataset and demonstrates
a realistic feature-engineering pipeline for machine-learning preparation.

The implementation covers:
- raw feature creation
- numerical transformations
- categorical encoding
- date/time decomposition
- transaction aggregation
- rolling behavioral features
- ratio and interaction features
- domain-specific financial/customer features
- leakage-aware feature construction
- missing-value handling
- validation
- train/test-safe fitting
- feature metadata
- serialization to CSV/JSON
- reproducibility and testing

Only the Python standard library is required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from math import log1p, sqrt
from pathlib import Path
import csv
import json
import random
import statistics
import tempfile
import unittest
from collections import defaultdict
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Customer:
    customer_id: str
    age: int
    region: str
    acquisition_channel: str
    signup_date: date
    annual_income: float
    credit_limit: float


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    customer_id: str
    timestamp: datetime
    amount: float
    category: str
    payment_method: str
    merchant_risk: float


@dataclass
class FeatureSpec:
    name: str
    category: str
    description: str
    source_fields: tuple[str, ...]
    leakage_sensitive: bool = False


# ---------------------------------------------------------------------------
# Reproducible realistic sample data
# ---------------------------------------------------------------------------

def build_sample_customers() -> list[Customer]:
    """Create customer records with fields useful for behavioral features."""
    return [
        Customer(
            "C001", 29, "North", "organic",
            date(2025, 1, 12), 72000, 12000
        ),
        Customer(
            "C002", 41, "South", "referral",
            date(2024, 11, 4), 98000, 20000
        ),
        Customer(
            "C003", 35, "West", "paid_search",
            date(2025, 2, 20), 61000, 9000
        ),
        Customer(
            "C004", 52, "East", "partner",
            date(2024, 8, 15), 145000, 30000
        ),
    ]


def build_sample_transactions() -> list[Transaction]:
    """Create a transaction history with temporal and behavioral variation."""
    base = datetime(2025, 4, 1, 9, 30)

    raw = [
        ("T001", "C001", 0, 120.50, "groceries", "card", 0.10),
        ("T002", "C001", 1, 42.00, "transport", "wallet", 0.05),
        ("T003", "C001", 4, 680.00, "electronics", "card", 0.40),
        ("T004", "C001", 8, 90.00, "dining", "wallet", 0.12),
        ("T005", "C001", 13, 1250.00, "travel", "card", 0.25),

        ("T006", "C002", 0, 250.00, "groceries", "card", 0.08),
        ("T007", "C002", 2, 1100.00, "travel", "card", 0.22),
        ("T008", "C002", 5, 340.00, "utilities", "bank_transfer", 0.04),
        ("T009", "C002", 11, 85.00, "dining", "card", 0.10),

        ("T010", "C003", 1, 55.00, "transport", "wallet", 0.07),
        ("T011", "C003", 3, 230.00, "groceries", "card", 0.10),
        ("T012", "C003", 7, 430.00, "electronics", "card", 0.35),
        ("T013", "C003", 12, 70.00, "dining", "wallet", 0.11),

        ("T014", "C004", 0, 450.00, "utilities", "bank_transfer", 0.03),
        ("T015", "C004", 4, 2100.00, "travel", "card", 0.28),
        ("T016", "C004", 9, 850.00, "electronics", "card", 0.38),
    ]

    return [
        Transaction(
            transaction_id=transaction_id,
            customer_id=customer_id,
            timestamp=base + timedelta(days=offset),
            amount=amount,
            category=category,
            payment_method=payment_method,
            merchant_risk=merchant_risk,
        )
        for (
            transaction_id,
            customer_id,
            offset,
            amount,
            category,
            payment_method,
            merchant_risk,
        ) in raw
    ]


# ---------------------------------------------------------------------------
# Basic feature creation
# ---------------------------------------------------------------------------

def customer_age_band(age: int) -> str:
    """Convert a continuous demographic value into interpretable bands."""
    if age < 25:
        return "under_25"
    if age < 35:
        return "25_34"
    if age < 45:
        return "35_44"
    if age < 55:
        return "45_54"
    return "55_plus"


def days_since_signup(customer: Customer, as_of: date) -> int:
    """Create customer-tenure information relative to a prediction date."""
    value = (as_of - customer.signup_date).days
    if value < 0:
        raise ValueError("Prediction date cannot precede signup date.")
    return value


def create_customer_base_features(
    customers: Iterable[Customer],
    as_of: date,
) -> dict[str, dict[str, Any]]:
    """
    Build features that can be computed without transaction outcomes.

    Keeping this stage separate prevents behavioral aggregates from being
    accidentally mixed with static customer attributes.
    """
    result: dict[str, dict[str, Any]] = {}

    for customer in customers:
        if customer.age <= 0:
            raise ValueError(f"Invalid age for {customer.customer_id}")
        if customer.annual_income <= 0:
            raise ValueError(f"Invalid income for {customer.customer_id}")
        if customer.credit_limit <= 0:
            raise ValueError(f"Invalid credit limit for {customer.customer_id}")

        tenure_days = days_since_signup(customer, as_of)

        result[customer.customer_id] = {
            "customer_id": customer.customer_id,
            "age": customer.age,
            "region": customer.region,
            "acquisition_channel": customer.acquisition_channel,
            "annual_income": customer.annual_income,
            "credit_limit": customer.credit_limit,
            "age_band": customer_age_band(customer.age),
            "log_income": log1p(customer.annual_income),
            "income_per_credit_limit": (
                customer.annual_income / customer.credit_limit
            ),
            "tenure_days": tenure_days,
            "tenure_years": tenure_days / 365.25,
        }

    return result


# ---------------------------------------------------------------------------
# Numerical transformations
# ---------------------------------------------------------------------------

class StandardScaler:
    """
    Small train-only standardization implementation.

    The scaler must be fitted only on training observations. Applying statistics
    learned from the complete dataset would leak information from the test set.
    """

    def __init__(self) -> None:
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.fitted = False

    def fit(self, rows: Iterable[dict[str, Any]], columns: Iterable[str]) -> None:
        rows = list(rows)
        if not rows:
            raise ValueError("Cannot fit scaler on an empty dataset.")

        for column in columns:
            values = [float(row[column]) for row in rows]
            mean = statistics.fmean(values)

            if len(values) > 1:
                std = statistics.stdev(values)
            else:
                std = 0.0

            self.means[column] = mean
            self.stds[column] = std
            self.fitted = True

    def transform(
        self,
        rows: Iterable[dict[str, Any]],
        columns: Iterable[str],
    ) -> list[dict[str, Any]]:
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transformation.")

        output = []
        for original in rows:
            row = dict(original)
            for column in columns:
                denominator = self.stds[column]
                if denominator == 0:
                    row[f"{column}_z"] = 0.0
                else:
                    row[f"{column}_z"] = (
                        float(row[column]) - self.means[column]
                    ) / denominator
            output.append(row)

        return output


def winsorize(values: list[float], lower: float, upper: float) -> list[float]:
    """
    Clip extreme values to specified bounds.

    This is useful when a feature contains legitimate but unusually large
    observations that would dominate a model. Bounds should normally be learned
    from the training set rather than chosen after examining test data.
    """
    if not values:
        return []
    if lower > upper:
        raise ValueError("Lower bound cannot exceed upper bound.")
    return [min(max(value, lower), upper) for value in values]


# ---------------------------------------------------------------------------
# Categorical transformation
# ---------------------------------------------------------------------------

class OneHotEncoder:
    """Small deterministic one-hot encoder for categorical feature creation."""

    def __init__(self) -> None:
        self.categories: dict[str, list[str]] = {}
        self.fitted = False

    def fit(self, rows: Iterable[dict[str, Any]], columns: Iterable[str]) -> None:
        rows = list(rows)

        for column in columns:
            categories = sorted(
                {str(row[column]) for row in rows if row.get(column) is not None}
            )
            if not categories:
                raise ValueError(f"No categories found for {column}")
            self.categories[column] = categories

        self.fitted = True

    def transform(
        self,
        rows: Iterable[dict[str, Any]],
        columns: Iterable[str],
    ) -> list[dict[str, Any]]:
        if not self.fitted:
            raise RuntimeError("Encoder must be fitted before transformation.")

        output = []

        for original in rows:
            row = dict(original)

            for column in columns:
                observed = str(row.get(column, "unknown"))

                for category in self.categories[column]:
                    row[
                        f"{column}__{category}"
                    ] = 1 if observed == category else 0

                # Unknown categories are represented by an all-zero vector
                # rather than causing the pipeline to fail.
                row[f"{column}__unknown"] = int(
                    observed not in self.categories[column]
                )

            output.append(row)

        return output


# ---------------------------------------------------------------------------
# Transaction aggregations
# ---------------------------------------------------------------------------

def aggregate_transactions(
    customers: Iterable[Customer],
    transactions: Iterable[Transaction],
    as_of: datetime,
    window_days: int = 30,
) -> dict[str, dict[str, Any]]:
    """
    Create customer-level behavioral features from transactions observed
    before the prediction timestamp.

    The strict timestamp condition is essential: future transactions must not
    enter features for a prediction made at `as_of`.
    """
    customers = list(customers)
    transactions = list(transactions)

    if window_days <= 0:
        raise ValueError("window_days must be positive.")

    window_start = as_of - timedelta(days=window_days)

    grouped: dict[str, list[Transaction]] = defaultdict(list)

    for transaction in transactions:
        if transaction.timestamp > as_of:
            continue
        if transaction.timestamp < window_start:
            continue
        grouped[transaction.customer_id].append(transaction)

    features: dict[str, dict[str, Any]] = {}

    for customer in customers:
        records = grouped.get(customer.customer_id, [])

        amounts = [record.amount for record in records]
        risky_amounts = [
            record.amount
            for record in records
            if record.merchant_risk >= 0.30
        ]

        total_spend = sum(amounts)
        transaction_count = len(records)
        average_transaction = (
            total_spend / transaction_count if transaction_count else 0.0
        )
        max_transaction = max(amounts, default=0.0)

        category_spend: dict[str, float] = defaultdict(float)
        payment_counts: dict[str, int] = defaultdict(int)

        for record in records:
            category_spend[record.category] += record.amount
            payment_counts[record.payment_method] += 1

        category_entropy = 0.0
        if transaction_count:
            for amount in category_spend.values():
                probability = amount / total_spend if total_spend else 0.0
                if probability > 0:
                    category_entropy -= probability * log1p(
                        probability
                    ) / log1p(1.0)

        features[customer.customer_id] = {
            "transactions_30d": transaction_count,
            "spend_30d": total_spend,
            "average_transaction_30d": average_transaction,
            "max_transaction_30d": max_transaction,
            "risky_transaction_count_30d": len(risky_amounts),
            "risky_spend_30d": sum(risky_amounts),
            "unique_categories_30d": len(category_spend),
            "category_spend_entropy_30d": category_entropy,
            "card_usage_ratio": (
                payment_counts["card"] / transaction_count
                if transaction_count
                else 0.0
            ),
            "wallet_usage_ratio": (
                payment_counts["wallet"] / transaction_count
                if transaction_count
                else 0.0
            ),
            "largest_purchase_share": (
                max_transaction / total_spend if total_spend else 0.0
            ),
        }

    return features


# ---------------------------------------------------------------------------
# Temporal and domain-specific features
# ---------------------------------------------------------------------------

def add_temporal_transaction_features(
    transactions: Iterable[Transaction],
) -> list[dict[str, Any]]:
    """
    Create features directly from transaction timestamps.

    These features preserve calendar information while also creating useful
    behavioral signals such as weekend activity and hour-of-day segments.
    """
    rows = []

    for transaction in transactions:
        hour = transaction.timestamp.hour
        rows.append(
            {
                "transaction_id": transaction.transaction_id,
                "customer_id": transaction.customer_id,
                "amount": transaction.amount,
                "category": transaction.category,
                "payment_method": transaction.payment_method,
                "merchant_risk": transaction.merchant_risk,
                "day_of_week": transaction.timestamp.weekday(),
                "is_weekend": int(transaction.timestamp.weekday() >= 5),
                "month": transaction.timestamp.month,
                "day_of_month": transaction.timestamp.day,
                "hour": hour,
                "is_business_hour": int(9 <= hour < 18),
                "is_night": int(hour < 6 or hour >= 22),
            }
        )

    return rows


def build_domain_features(
    base_features: dict[str, dict[str, Any]],
    aggregate_features: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Build domain features that combine financial capacity with observed
    spending behavior.

    These are domain features because their meaning depends on customer
    financial behavior rather than generic arithmetic alone.
    """
    result = {}

    for customer_id, base in base_features.items():
        aggregate = aggregate_features[customer_id]

        credit_utilization_proxy = (
            aggregate["spend_30d"] / base["credit_limit"]
        )

        spending_to_income_proxy = (
            aggregate["spend_30d"] * 12 / base["annual_income"]
        )

        risk_intensity = (
            aggregate["risky_spend_30d"] / aggregate["spend_30d"]
            if aggregate["spend_30d"]
            else 0.0
        )

        concentration = aggregate["largest_purchase_share"]

        result[customer_id] = {
            "customer_id": customer_id,
            "monthly_spend_to_credit_limit": credit_utilization_proxy,
            "annualized_spend_to_income": spending_to_income_proxy,
            "merchant_risk_intensity": risk_intensity,
            "purchase_concentration": concentration,
            "behavioral_capacity_gap": max(
                0.0,
                base["credit_limit"] - aggregate["spend_30d"],
            ),
            "high_value_purchase_flag": int(
                aggregate["max_transaction_30d"] >= 1000
            ),
            "active_customer_flag": int(
                aggregate["transactions_30d"] >= 3
            ),
        }

    return result


# ---------------------------------------------------------------------------
# Missing-value handling and validation
# ---------------------------------------------------------------------------

def fill_numeric_missing(
    rows: list[dict[str, Any]],
    columns: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Fill missing numeric values using training-derived medians.

    Returning the statistics makes the transformation reproducible for
    production inference.
    """
    columns = list(columns)
    medians: dict[str, float] = {}

    for column in columns:
        values = [
            float(row[column])
            for row in rows
            if row.get(column) is not None
        ]

        if not values:
            raise ValueError(
                f"Cannot calculate a fill value for {column}: no observations."
            )

        medians[column] = statistics.median(values)

    transformed = []

    for original in rows:
        row = dict(original)
        for column in columns:
            if row.get(column) is None:
                row[column] = medians[column]
        transformed.append(row)

    return transformed, medians


def validate_feature_rows(
    rows: Iterable[dict[str, Any]],
    required_columns: Iterable[str],
) -> None:
    """Validate schema and reject invalid feature values."""
    required_columns = list(required_columns)

    for index, row in enumerate(rows):
        missing = [column for column in required_columns if column not in row]
        if missing:
            raise ValueError(
                f"Row {index} is missing required columns: {missing}"
            )

        for key, value in row.items():
            if isinstance(value, float):
                if value != value:
                    raise ValueError(
                        f"NaN detected in row {index}, feature {key}"
                    )


# ---------------------------------------------------------------------------
# Leakage-aware temporal feature construction
# ---------------------------------------------------------------------------

def build_prediction_snapshot(
    customers: list[Customer],
    transactions: list[Transaction],
    prediction_time: datetime,
) -> list[dict[str, Any]]:
    """
    Construct one customer-level feature table for a prediction timestamp.

    Transactions occurring after prediction_time are deliberately excluded.
    This is the central leakage-control rule for temporal behavioral features.
    """
    base = create_customer_base_features(
        customers,
        prediction_time.date(),
    )

    aggregates = aggregate_transactions(
        customers,
        transactions,
        prediction_time,
        window_days=30,
    )

    domain = build_domain_features(base, aggregates)

    result = []

    for customer in customers:
        row = {}
        row.update(base[customer.customer_id])
        row.update(aggregates[customer.customer_id])
        row.update(domain[customer.customer_id])

        # The target is deliberately absent. A production feature builder
        # should not accidentally calculate a future outcome into the inputs.
        result.append(row)

    return result


# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

def build_feature_catalog() -> list[FeatureSpec]:
    """Describe the semantic origin of important generated features."""
    return [
        FeatureSpec(
            "log_income",
            "transformation",
            "Log-transformed annual income to reduce right-skew.",
            ("annual_income",),
        ),
        FeatureSpec(
            "age_band",
            "transformation",
            "Age grouped into interpretable demographic bands.",
            ("age",),
        ),
        FeatureSpec(
            "spend_30d",
            "aggregation",
            "Total transaction amount observed during the trailing 30-day window.",
            ("amount", "timestamp", "customer_id"),
            True,
        ),
        FeatureSpec(
            "average_transaction_30d",
            "aggregation",
            "Mean transaction amount during the trailing 30-day window.",
            ("amount", "timestamp", "customer_id"),
            True,
        ),
        FeatureSpec(
            "monthly_spend_to_credit_limit",
            "domain",
            "Recent monthly spending expressed relative to available credit capacity.",
            ("spend_30d", "credit_limit"),
            True,
        ),
        FeatureSpec(
            "merchant_risk_intensity",
            "domain",
            "Share of recent spending associated with higher-risk merchants.",
            ("risky_spend_30d", "spend_30d"),
            True,
        ),
    ]


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------

def export_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a rectangular feature matrix to CSV."""
    if not rows:
        raise ValueError("Cannot export an empty feature matrix.")

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_json(rows: list[dict[str, Any]], path: Path) -> None:
    """Write feature rows as JSON for downstream inspection."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=str)


def export_feature_catalog(
    catalog: list[FeatureSpec],
    path: Path,
) -> None:
    """Persist semantic metadata alongside the feature matrix."""
    serializable = [asdict(item) for item in catalog]

    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


# ---------------------------------------------------------------------------
# Pipeline demonstration
# ---------------------------------------------------------------------------

def demonstrate_pipeline() -> None:
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 72)

    customers = build_sample_customers()
    transactions = build_sample_transactions()

    prediction_time = datetime(2025, 4, 15, 23, 59)

    print("\nCustomer-level features")
    base = create_customer_base_features(
        customers,
        prediction_time.date(),
    )

    for customer_id, row in base.items():
        print(
            customer_id,
            {
                "age_band": row["age_band"],
                "log_income": round(row["log_income"], 3),
                "tenure_days": row["tenure_days"],
            },
        )

    print("\nBehavioral aggregation features")
    aggregates = aggregate_transactions(
        customers,
        transactions,
        prediction_time,
    )

    for customer_id, row in aggregates.items():
        print(
            customer_id,
            {
                "transactions_30d": row["transactions_30d"],
                "spend_30d": round(row["spend_30d"], 2),
                "average_transaction_30d": round(
                    row["average_transaction_30d"], 2
                ),
                "unique_categories_30d": row["unique_categories_30d"],
            },
        )

    print("\nDomain features")
    domain = build_domain_features(base, aggregates)

    for customer_id, row in domain.items():
        print(
            customer_id,
            {
                "monthly_spend_to_credit_limit": round(
                    row["monthly_spend_to_credit_limit"], 4
                ),
                "annualized_spend_to_income": round(
                    row["annualized_spend_to_income"], 4
                ),
                "merchant_risk_intensity": round(
                    row["merchant_risk_intensity"], 4
                ),
            },
        )

    print("\nCombined prediction snapshot")
    snapshot = build_prediction_snapshot(
        customers,
        transactions,
        prediction_time,
    )

    validate_feature_rows(
        snapshot,
        [
            "customer_id",
            "age",
            "annual_income",
            "spend_30d",
            "transactions_30d",
            "monthly_spend_to_credit_limit",
        ],
    )

    for row in snapshot:
        print(
            row["customer_id"],
            "spend_30d=",
            round(row["spend_30d"], 2),
            "risk_intensity=",
            round(row["merchant_risk_intensity"], 3),
        )

    print("\nTrain-only numerical transformation")
    training_rows = snapshot[:3]
    test_rows = snapshot[3:]

    scaler = StandardScaler()
    scaler.fit(
        training_rows,
        ["annual_income", "spend_30d"],
    )

    transformed_test = scaler.transform(
        test_rows,
        ["annual_income", "spend_30d"],
    )

    for row in transformed_test:
        print(
            row["customer_id"],
            "income_z=",
            round(row["annual_income_z"], 3),
            "spend_z=",
            round(row["spend_30d_z"], 3),
        )

    print("\nCategorical transformation")
    encoder = OneHotEncoder()
    encoder.fit(
        snapshot,
        ["region", "acquisition_channel"],
    )

    encoded = encoder.transform(
        snapshot,
        ["region", "acquisition_channel"],
    )

    for row in encoded[:2]:
        selected = {
            key: value
            for key, value in row.items()
            if key.startswith("region__")
        }
        print(row["customer_id"], selected)

    print("\nTemporal transaction features")
    temporal = add_temporal_transaction_features(transactions)
    for row in temporal[:3]:
        print(
            row["transaction_id"],
            {
                "day_of_week": row["day_of_week"],
                "is_business_hour": row["is_business_hour"],
                "is_night": row["is_night"],
            },
        )

    print("\nFeature catalog")
    for specification in build_feature_catalog():
        print(
            specification.name,
            "->",
            specification.category,
            "|",
            specification.description,
        )

    with tempfile.TemporaryDirectory() as directory:
        output_dir = Path(directory)

        export_csv(snapshot, output_dir / "features.csv")
        export_json(snapshot, output_dir / "features.json")
        export_feature_catalog(
            build_feature_catalog(),
            output_dir / "feature_catalog.json",
        )

        print("\nExported artifacts")
        print((output_dir / "features.csv").name)
        print((output_dir / "features.json").name)
        print((output_dir / "feature_catalog.json").name)


# ---------------------------------------------------------------------------
# Edge cases and tests
# ---------------------------------------------------------------------------

class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.customers = build_sample_customers()
        self.transactions = build_sample_transactions()
        self.prediction_time = datetime(2025, 4, 15, 23, 59)

    def test_future_transactions_are_excluded(self) -> None:
        future_transaction = Transaction(
            "FUTURE",
            "C001",
            self.prediction_time + timedelta(days=2),
            999999.0,
            "electronics",
            "card",
            1.0,
        )

        original = aggregate_transactions(
            self.customers,
            self.transactions,
            self.prediction_time,
        )

        with_future = aggregate_transactions(
            self.customers,
            self.transactions + [future_transaction],
            self.prediction_time,
        )

        self.assertEqual(
            original["C001"]["spend_30d"],
            with_future["C001"]["spend_30d"],
        )

    def test_empty_customer_activity_is_safe(self) -> None:
        customers = [
            Customer(
                "NEW",
                30,
                "North",
                "organic",
                date(2025, 4, 10),
                50000,
                5000,
            )
        ]

        aggregates = aggregate_transactions(
            customers,
            self.transactions,
            self.prediction_time,
        )

        self.assertEqual(aggregates["NEW"]["transactions_30d"], 0)
        self.assertEqual(aggregates["NEW"]["spend_30d"], 0.0)

    def test_scaler_rejects_transform_before_fit(self) -> None:
        scaler = StandardScaler()

        with self.assertRaises(RuntimeError):
            scaler.transform([{"x": 10}], ["x"])

    def test_one_hot_unknown_category(self) -> None:
        encoder = OneHotEncoder()
        encoder.fit(
            [{"region": "North"}, {"region": "South"}],
            ["region"],
        )

        transformed = encoder.transform(
            [{"region": "West"}],
            ["region"],
        )[0]

        self.assertEqual(transformed["region__unknown"], 1)

    def test_invalid_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_transactions(
                self.customers,
                self.transactions,
                self.prediction_time,
                window_days=0,
            )

    def test_feature_snapshot_contains_domain_features(self) -> None:
        snapshot = build_prediction_snapshot(
            self.customers,
            self.transactions,
            self.prediction_time,
        )

        self.assertIn(
            "merchant_risk_intensity",
            snapshot[0],
        )
        self.assertIn(
            "monthly_spend_to_credit_limit",
            snapshot[0],
        )


def run_tests() -> None:
    print("\nRUNNING FEATURE ENGINEERING TESTS")
    print("=" * 72)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        FeatureEngineeringTests
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(42)
    demonstrate_pipeline()
    run_tests()


if __name__ == "__main__":
    main()
