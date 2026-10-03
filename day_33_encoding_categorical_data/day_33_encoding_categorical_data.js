"""
Categorical Data Encoding: Label, One-Hot, Ordinal, and Target Encoding

A self-contained technical learning implementation that progresses from
categorical-data fundamentals to leakage-aware target encoding, including
validation, unseen categories, missing values, train/test separation,
smoothing, cross-validation, and a small model-ready preprocessing workflow.

The script uses only the Python standard library.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from random import Random
from statistics import mean
from typing import Any, Iterable


MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNKNOWN__"


def print_title(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    rows = [[str(value) for value in row] for row in rows]
    headers = [str(header) for header in headers]

    widths = []
    for index, header in enumerate(headers):
        values = [row[index] for row in rows if index < len(row)]
        widths.append(max([len(header)] + [len(value) for value in values]))

    line = " | ".join("-" * width for width in widths)
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print(line)

    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def normalize_category(value: Any) -> str:
    """
    Convert categorical values into stable string keys.

    Missing values are represented explicitly instead of allowing None,
    empty strings, and whitespace-only strings to become separate categories.
    """
    if value is None:
        return MISSING_TOKEN

    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else MISSING_TOKEN

    return str(value)


def validate_categories(values: Iterable[Any]) -> None:
    """
    Reject categories that would be unsafe or ambiguous for this demonstration.

    Real systems may use richer category objects, but deterministic scalar
    categories are easier to serialize, compare, and encode consistently.
    """
    for value in values:
        if isinstance(value, (dict, list, set, tuple)):
            raise TypeError(
                "Categories must be scalar values; complex containers should "
                "be transformed into explicit categorical features first."
            )


@dataclass
class Dataset:
    rows: list[dict[str, Any]]
    target_name: str

    @property
    def target(self) -> list[float]:
        return [float(row[self.target_name]) for row in self.rows]

    def column(self, name: str) -> list[Any]:
        return [row[name] for row in self.rows]


def build_customer_dataset() -> Dataset:
    """
    Create a realistic churn dataset.

    Region is nominal, plan_level is ordinal, acquisition_channel is nominal,
    and customer_segment is nominal. Churn is binary and is intentionally
    correlated with some categories so target encoding has meaningful signal.
    """
    rows = [
        {"region": "North", "plan_level": "Basic", "acquisition_channel": "Organic", "customer_segment": "Student", "churn": 1},
        {"region": "North", "plan_level": "Standard", "acquisition_channel": "Referral", "customer_segment": "Professional", "churn": 0},
        {"region": "South", "plan_level": "Premium", "acquisition_channel": "Partner", "customer_segment": "Enterprise", "churn": 0},
        {"region": "West", "plan_level": "Basic", "acquisition_channel": "Paid Search", "customer_segment": "Student", "churn": 1},
        {"region": "East", "plan_level": "Standard", "acquisition_channel": "Organic", "customer_segment": "Professional", "churn": 0},
        {"region": "South", "plan_level": "Basic", "acquisition_channel": "Paid Search", "customer_segment": "Student", "churn": 1},
        {"region": "West", "plan_level": "Premium", "acquisition_channel": "Referral", "customer_segment": "Enterprise", "churn": 0},
        {"region": "North", "plan_level": "Basic", "acquisition_channel": "Partner", "customer_segment": "Student", "churn": 1},
        {"region": "East", "plan_level": "Premium", "acquisition_channel": "Organic", "customer_segment": "Enterprise", "churn": 0},
        {"region": "South", "plan_level": "Standard", "acquisition_channel": "Referral", "customer_segment": "Professional", "churn": 0},
        {"region": "West", "plan_level": "Basic", "acquisition_channel": "Organic", "customer_segment": "Student", "churn": 1},
        {"region": "East", "plan_level": "Basic", "acquisition_channel": "Paid Search", "customer_segment": "Student", "churn": 1},
        {"region": "North", "plan_level": "Premium", "acquisition_channel": "Partner", "customer_segment": "Enterprise", "churn": 0},
        {"region": "South", "plan_level": "Premium", "acquisition_channel": "Referral", "customer_segment": "Enterprise", "churn": 0},
        {"region": "West", "plan_level": "Standard", "acquisition_channel": "Organic", "customer_segment": "Professional", "churn": 0},
        {"region": "East", "plan_level": "Basic", "acquisition_channel": "Paid Search", "customer_segment": "Student", "churn": 1},
    ]
    return Dataset(rows, "churn")


class LabelEncoder:
    """
    Encode each distinct category as an integer.

    Label encoding is appropriate when the integer is treated as an identifier
    rather than a numerical magnitude. It is generally unsuitable for a
    nominal feature fed directly into a linear model because values such as
    0, 1, and 2 can accidentally imply ordering and distance.
    """

    def __init__(self, handle_unknown: str = "error") -> None:
        if handle_unknown not in {"error", "use_unknown"}:
            raise ValueError("handle_unknown must be 'error' or 'use_unknown'.")

        self.handle_unknown = handle_unknown
        self.mapping: dict[str, int] = {}
        self.inverse_mapping: dict[int, str] = {}

    def fit(self, values: Iterable[Any]) -> "LabelEncoder":
        normalized = [normalize_category(value) for value in values]
        validate_categories(values)

        self.mapping.clear()
        self.inverse_mapping.clear()

        for category in sorted(set(normalized)):
            index = len(self.mapping)
            self.mapping[category] = index
            self.inverse_mapping[index] = category

        if self.handle_unknown == "use_unknown":
            self.mapping.setdefault(UNKNOWN_TOKEN, -1)
            self.inverse_mapping[-1] = UNKNOWN_TOKEN

        return self

    def transform(self, values: Iterable[Any]) -> list[int]:
        if not self.mapping:
            raise RuntimeError("Encoder must be fitted before transform().")

        result = []
        for value in values:
            category = normalize_category(value)
            if category in self.mapping:
                result.append(self.mapping[category])
            elif self.handle_unknown == "use_unknown":
                result.append(-1)
            else:
                raise ValueError(f"Unknown category: {category!r}")

        return result

    def inverse_transform(self, values: Iterable[int]) -> list[str]:
        return [self.inverse_mapping[value] for value in values]


class OneHotEncoder:
    """
    Represent each nominal category with a separate binary feature.

    Example:
        region=North -> [1, 0, 0, 0]
        region=South -> [0, 0, 1, 0]

    The implementation supports dropping one reference category, which can
    prevent perfect multicollinearity when all encoded columns are used in a
    linear regression design matrix with an intercept.
    """

    def __init__(
        self,
        handle_unknown: str = "ignore",
        drop_first: bool = False,
    ) -> None:
        if handle_unknown not in {"error", "ignore"}:
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")

        self.handle_unknown = handle_unknown
        self.drop_first = drop_first
        self.categories: list[str] = []
        self.output_categories: list[str] = []

    def fit(self, values: Iterable[Any]) -> "OneHotEncoder":
        normalized = [normalize_category(value) for value in values]
        validate_categories(values)

        self.categories = sorted(set(normalized))
        self.output_categories = (
            self.categories[1:] if self.drop_first else self.categories.copy()
        )
        return self

    def transform(self, values: Iterable[Any]) -> list[list[int]]:
        if not self.categories:
            raise RuntimeError("Encoder must be fitted before transform().")

        category_set = set(self.categories)
        result = []

        for value in values:
            category = normalize_category(value)

            if category not in category_set:
                if self.handle_unknown == "error":
                    raise ValueError(f"Unknown category: {category!r}")
                result.append([0] * len(self.output_categories))
                continue

            result.append(
                [1 if category == output else 0 for output in self.output_categories]
            )

        return result

    def get_feature_names(self, prefix: str) -> list[str]:
        return [f"{prefix}__{category}" for category in self.output_categories]


class OrdinalEncoder:
    """
    Encode categories according to an explicitly supplied semantic order.

    Unlike label encoding, ordinal encoding is meaningful when the category
    values genuinely have ordered semantics such as Basic < Standard < Premium.
    """

    def __init__(
        self,
        ordered_categories: list[str],
        handle_unknown: str = "error",
    ) -> None:
        if not ordered_categories:
            raise ValueError("At least one ordered category is required.")

        if len(set(ordered_categories)) != len(ordered_categories):
            raise ValueError("ordered_categories must contain unique categories.")

        if handle_unknown not in {"error", "use_unknown"}:
            raise ValueError("handle_unknown must be 'error' or 'use_unknown'.")

        self.ordered_categories = [
            normalize_category(category) for category in ordered_categories
        ]
        self.handle_unknown = handle_unknown
        self.mapping = {
            category: index
            for index, category in enumerate(self.ordered_categories)
        }

    def transform(self, values: Iterable[Any]) -> list[int]:
        result = []

        for value in values:
            category = normalize_category(value)
            if category not in self.mapping:
                if self.handle_unknown == "use_unknown":
                    result.append(-1)
                    continue
                raise ValueError(
                    f"Unknown ordinal category {category!r}. "
                    f"Expected one of {self.ordered_categories}."
                )
            result.append(self.mapping[category])

        return result


class TargetEncoder:
    """
    Mean target encoding with smoothing.

    For each category:

        encoded = (count * category_mean + smoothing * global_mean)
                  / (count + smoothing)

    Smoothing reduces the influence of categories with very few observations.

    Critical rule:
        The encoder must be fitted using training targets only.

    Fitting it on validation/test targets leaks outcome information into the
    feature representation and can produce misleadingly strong evaluation
    results.
    """

    def __init__(
        self,
        smoothing: float = 10.0,
        handle_unknown: str = "global_mean",
    ) -> None:
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative.")

        if handle_unknown != "global_mean":
            raise ValueError("Only global_mean unknown handling is supported.")

        self.smoothing = float(smoothing)
        self.handle_unknown = handle_unknown
        self.global_mean: float | None = None
        self.statistics: dict[str, tuple[int, float]] = {}

    def fit(
        self,
        values: Iterable[Any],
        target: Iterable[float],
    ) -> "TargetEncoder":
        categories = [normalize_category(value) for value in values]
        targets = [float(value) for value in target]

        if len(categories) != len(targets):
            raise ValueError("Feature and target lengths must match.")

        if not targets:
            raise ValueError("Target encoding requires at least one training row.")

        if any(not isinstance(value, (int, float)) for value in targets):
            raise TypeError("Target values must be numeric.")

        self.global_mean = mean(targets)
        grouped_targets: dict[str, list[float]] = defaultdict(list)

        for category, target_value in zip(categories, targets):
            grouped_targets[category].append(target_value)

        self.statistics.clear()

        for category, observations in grouped_targets.items():
            self.statistics[category] = (
                len(observations),
                mean(observations),
            )

        return self

    def transform(self, values: Iterable[Any]) -> list[float]:
        if self.global_mean is None:
            raise RuntimeError("TargetEncoder must be fitted before transform().")

        encoded = []

        for value in values:
            category = normalize_category(value)
            count, category_mean = self.statistics.get(
                category,
                (0, self.global_mean),
            )

            smoothed = (
                count * category_mean + self.smoothing * self.global_mean
            ) / (count + self.smoothing)

            encoded.append(smoothed)

        return encoded

    def fit_transform(
        self,
        values: Iterable[Any],
        target: Iterable[float],
    ) -> list[float]:
        self.fit(values, target)
        return self.transform(values)

    def mapping_table(self) -> list[tuple[str, int, float, float]]:
        if self.global_mean is None:
            raise RuntimeError("Encoder must be fitted first.")

        rows = []
        for category in sorted(self.statistics):
            count, category_mean = self.statistics[category]
            encoded = (
                count * category_mean
                + self.smoothing * self.global_mean
            ) / (count + self.smoothing)
            rows.append((category, count, round(category_mean, 4), round(encoded, 4)))

        return rows


def demonstrate_label_encoding(dataset: Dataset) -> None:
    print_title("Label Encoding: categorical identifiers represented by integers")

    encoder = LabelEncoder(handle_unknown="use_unknown")
    encoder.fit(dataset.column("region"))

    encoded = encoder.transform(dataset.column("region"))
    print("Region mapping:", encoder.mapping)
    print_table(
        ["Region", "Encoded"],
        zip(dataset.column("region"), encoded),
    )

    unseen = encoder.transform(["North", "Central"])
    print("Known and unseen categories:", unseen)

    print(
        "\nImportant distinction: these integers identify categories. "
        "For nominal regions, 0 < 1 < 2 does not mean one region is numerically "
        "larger or closer to another."
    )


def demonstrate_one_hot_encoding(dataset: Dataset) -> None:
    print_title("One-Hot Encoding: independent binary indicators")

    encoder = OneHotEncoder(handle_unknown="ignore", drop_first=False)
    encoder.fit(dataset.column("region"))

    encoded = encoder.transform(["North", "South", "West", "Central"])
    names = encoder.get_feature_names("region")

    print_table(
        names,
        encoded,
    )

    print("\nUnknown category 'Central' becomes all zeros because handle_unknown='ignore'.")

    reference_encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop_first=True,
    )
    reference_encoder.fit(dataset.column("region"))

    print(
        "\nWith drop_first=True, the reference category is removed:",
        reference_encoder.get_feature_names("region"),
    )


def demonstrate_ordinal_encoding(dataset: Dataset) -> None:
    print_title("Ordinal Encoding: preserve real semantic order")

    ordered_levels = ["Basic", "Standard", "Premium"]
    encoder = OrdinalEncoder(
        ordered_categories=ordered_levels,
        handle_unknown="use_unknown",
    )

    values = ["Basic", "Premium", "Standard", "Premium", "Basic"]
    encoded = encoder.transform(values)

    print_table(
        ["Plan level", "Ordinal value"],
        zip(values, encoded),
    )

    print("\nExplicit order:", ordered_levels)
    print(
        "Here the numeric relationship is intentional: Basic < Standard < Premium."
    )

    unknown = encoder.transform(["Enterprise"])
    print("Unknown plan level:", unknown)


def demonstrate_target_encoding(dataset: Dataset) -> None:
    print_title("Target Encoding: category statistics derived from the target")

    encoder = TargetEncoder(smoothing=3.0)
    encoder.fit(
        dataset.column("customer_segment"),
        dataset.target,
    )

    print("Global churn rate:", round(encoder.global_mean or 0.0, 4))
    print_table(
        ["Category", "Count", "Raw mean", "Smoothed encoding"],
        encoder.mapping_table(),
    )

    transformed = encoder.transform(
        ["Student", "Professional", "Enterprise", "New Segment"]
    )

    print_table(
        ["Segment", "Encoded target mean"],
        zip(
            ["Student", "Professional", "Enterprise", "New Segment"],
            [round(value, 4) for value in transformed],
        ),
    )

    print(
        "\nThe unseen category falls back to the global target mean. "
        "This is safer than inventing a category-specific target statistic."
    )


def demonstrate_smoothing() -> None:
    print_title("Target-Encoding Smoothing: controlling small-sample instability")

    categories = [
        "Common",
        "Common",
        "Common",
        "Common",
        "Common",
        "Rare",
    ]
    target = [0, 0, 1, 0, 1, 1]

    global_mean = mean(target)

    raw_statistics: dict[str, list[float]] = defaultdict(list)
    for category, value in zip(categories, target):
        raw_statistics[category].append(value)

    print("Global target mean:", round(global_mean, 4))

    for smoothing in [0.0, 1.0, 5.0, 20.0]:
        encoder = TargetEncoder(smoothing=smoothing)
        encoder.fit(categories, target)
        rare_value = encoder.transform(["Rare"])[0]

        print(
            f"Smoothing={smoothing:>4}: "
            f"Rare raw mean={mean(raw_statistics['Rare']):.4f}, "
            f"encoded={rare_value:.4f}"
        )

    print(
        "\nA rare category has little evidence. Increasing smoothing pulls its "
        "estimate toward the global mean rather than trusting one observation."
    )


def demonstrate_train_test_leakage() -> None:
    print_title("Target Encoding Leakage: why train/test separation matters")

    train_categories = ["A", "A", "B", "B", "C"]
    train_target = [0, 0, 1, 1, 0]

    test_categories = ["A", "B", "C", "D"]
    test_target = [1, 0, 1, 1]

    safe_encoder = TargetEncoder(smoothing=2.0)
    safe_encoder.fit(train_categories, train_target)
    safe_test_values = safe_encoder.transform(test_categories)

    leaky_encoder = TargetEncoder(smoothing=2.0)
    leaky_encoder.fit(
        train_categories + test_categories,
        train_target + test_target,
    )
    leaky_test_values = leaky_encoder.transform(test_categories)

    print_table(
        ["Test category", "Safe encoding", "Leaky encoding"],
        zip(
            test_categories,
            [round(value, 4) for value in safe_test_values],
            [round(value, 4) for value in leaky_test_values],
        ),
    )

    print(
        "\nThe leaky encoder has seen the test outcomes. "
        "Its statistics therefore contain information that would not be "
        "available when making genuine future predictions."
    )


def demonstrate_cross_fitted_target_encoding() -> None:
    print_title("Cross-Fitted Target Encoding: reducing in-sample target leakage")

    categories = [
        "Basic", "Basic", "Basic",
        "Premium", "Premium", "Premium",
        "Standard", "Standard", "Standard",
        "Basic", "Premium", "Standard",
    ]
    target = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0]

    folds = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]

    out_of_fold = [None] * len(categories)

    for validation_indices in folds:
        validation_set = set(validation_indices)
        training_indices = [
            index for index in range(len(categories))
            if index not in validation_set
        ]

        training_categories = [categories[index] for index in training_indices]
        training_target = [target[index] for index in training_indices]

        validation_categories = [
            categories[index] for index in validation_indices
        ]

        encoder = TargetEncoder(smoothing=5.0)
        encoder.fit(training_categories, training_target)

        encoded_validation = encoder.transform(validation_categories)

        for index, encoded_value in zip(validation_indices, encoded_validation):
            out_of_fold[index] = encoded_value

    print_table(
        ["Row", "Category", "Target", "Out-of-fold encoding"],
        [
            (
                index,
                categories[index],
                target[index],
                round(float(out_of_fold[index]), 4),
            )
            for index in range(len(categories))
        ],
    )

    print(
        "\nEach training row receives an encoding calculated without its own "
        "target. This is a practical way to create target-derived training "
        "features while reducing direct row-level target leakage."
    )


def demonstrate_encoding_selection() -> None:
    print_title("Selecting an Encoding According to Category Semantics")

    examples = [
        (
            "Customer region",
            "Nominal",
            "One-hot",
            "Regions have names but no natural magnitude or order.",
        ),
        (
            "Subscription tier",
            "Ordinal",
            "Ordinal",
            "Basic, Standard, and Premium have an explicit business order.",
        ),
        (
            "High-cardinality merchant ID",
            "Nominal/high-cardinality",
            "Target encoding or another supervised representation",
            "One-hot expansion can become extremely wide.",
        ),
        (
            "Class label for a classification target",
            "Target variable",
            "Label encoding",
            "The label is an output representation rather than a feature.",
        ),
    ]

    print_table(
        ["Feature", "Semantic type", "Candidate", "Reason"],
        examples,
    )


def demonstrate_edge_cases() -> None:
    print_title("Edge Cases and Validation")

    label = LabelEncoder(handle_unknown="use_unknown")
    label.fit(["A", None, "B", ""])
    print(
        "Missing/empty categories:",
        label.mapping,
        "encoded:",
        label.transform(["A", None, "", "C"]),
    )

    one_hot = OneHotEncoder(handle_unknown="ignore")
    one_hot.fit(["A", "B"])
    print("Unknown one-hot value:", one_hot.transform(["C"]))

    try:
        OrdinalEncoder(["Low", "Medium", "High"]).transform(["Extreme"])
    except ValueError as error:
        print("Ordinal validation error:", error)

    try:
        TargetEncoder().fit(["A", "B"], [1])
    except ValueError as error:
        print("Target length validation error:", error)

    try:
        TargetEncoder().fit(["A", "B"], [1, "bad"])
    except (TypeError, ValueError) as error:
        print("Target type validation error:", error)


def compute_cardinality(values: Iterable[Any]) -> tuple[int, float]:
    normalized = [normalize_category(value) for value in values]
    if not normalized:
        return 0, 0.0

    unique_count = len(set(normalized))
    ratio = unique_count / len(normalized)
    return unique_count, ratio


def demonstrate_cardinality() -> None:
    print_title("Cardinality and Representation Cost")

    features = {
        "region": ["North", "South", "East", "West"] * 4,
        "plan_level": ["Basic", "Standard", "Premium"] * 6,
        "merchant_id": [f"merchant_{index:03d}" for index in range(24)],
    }

    rows = []
    for name, values in features.items():
        unique_count, ratio = compute_cardinality(values)
        rows.append((name, len(values), unique_count, f"{ratio:.2f}"))

    print_table(
        ["Feature", "Rows", "Unique categories", "Cardinality ratio"],
        rows,
    )

    print(
        "\nOne-hot width grows with the number of categories. "
        "A high-cardinality identifier can therefore create a very wide "
        "feature matrix, while target encoding keeps one numerical column "
        "but introduces supervised-statistical concerns."
    )


def build_model_ready_features(dataset: Dataset) -> list[dict[str, Any]]:
    """
    Construct a compact feature table using different encoders for different
    semantic types.

    This deliberately combines:
      - one-hot for nominal region,
      - ordinal encoding for plan level,
      - one-hot for acquisition channel,
      - target encoding for customer segment.

    The target encoder is fitted on this training dataset only.
    """
    region_encoder = OneHotEncoder(handle_unknown="ignore")
    region_encoder.fit(dataset.column("region"))

    plan_encoder = OrdinalEncoder(
        ["Basic", "Standard", "Premium"],
        handle_unknown="use_unknown",
    )

    channel_encoder = OneHotEncoder(handle_unknown="ignore")
    channel_encoder.fit(dataset.column("acquisition_channel"))

    segment_encoder = TargetEncoder(smoothing=5.0)
    segment_encoder.fit(
        dataset.column("customer_segment"),
        dataset.target,
    )

    regions = region_encoder.transform(dataset.column("region"))
    plans = plan_encoder.transform(dataset.column("plan_level"))
    channels = channel_encoder.transform(
        dataset.column("acquisition_channel")
    )
    segments = segment_encoder.transform(
        dataset.column("customer_segment")
    )

    region_names = region_encoder.get_feature_names("region")
    channel_names = channel_encoder.get_feature_names("channel")

    feature_rows = []

    for index in range(len(dataset.rows)):
        row: dict[str, Any] = {}

        for name, value in zip(region_names, regions[index]):
            row[name] = value

        row["plan_level__ordinal"] = plans[index]

        for name, value in zip(channel_names, channels[index]):
            row[name] = value

        row["customer_segment__target_mean"] = round(segments[index], 6)
        row["churn"] = dataset.rows[index]["churn"]

        feature_rows.append(row)

    return feature_rows


def demonstrate_end_to_end_pipeline(dataset: Dataset) -> None:
    print_title("End-to-End Encoding Pipeline")

    features = build_model_ready_features(dataset)

    columns = [
        key
        for key in features[0]
        if key != "churn"
    ]

    print_table(
        columns + ["churn"],
        [
            [row[column] for column in columns] + [row["churn"]]
            for row in features[:8]
        ],
    )

    print(
        "\nThe resulting representation contains binary indicators for nominal "
        "features, an ordered integer for the subscription tier, and a smoothed "
        "target statistic for the customer segment."
    )


def demonstrate_round_trip_label_encoding(dataset: Dataset) -> None:
    print_title("Label Encoding Round Trip")

    encoder = LabelEncoder(handle_unknown="error")
    original = ["North", "West", "South", "East"]

    encoder.fit(dataset.column("region"))
    encoded = encoder.transform(original)
    restored = encoder.inverse_transform(encoded)

    print("Original:", original)
    print("Encoded :", encoded)
    print("Restored:", restored)

    assert original == restored


def main() -> None:
    dataset = build_customer_dataset()

    print_title("Categorical Data Encoding")
    print(
        "Dataset rows:",
        len(dataset.rows),
        "| Target:",
        dataset.target_name,
    )
    print(
        "The demonstrations distinguish nominal categories, ordered "
        "categories, and supervised target-derived representations."
    )

    demonstrate_label_encoding(dataset)
    demonstrate_one_hot_encoding(dataset)
    demonstrate_ordinal_encoding(dataset)
    demonstrate_target_encoding(dataset)
    demonstrate_smoothing()
    demonstrate_train_test_leakage()
    demonstrate_cross_fitted_target_encoding()
    demonstrate_encoding_selection()
    demonstrate_edge_cases()
    demonstrate_cardinality()
    demonstrate_round_trip_label_encoding(dataset)
    demonstrate_end_to_end_pipeline(dataset)

    print_title("Production Considerations")
    print(
        "Persist the fitted category mappings with the model. "
        "Never rebuild mappings independently during inference."
    )
    print(
        "Fit target encoding only with outcomes available in the training "
        "partition. For model evaluation, generate validation encodings "
        "without using validation outcomes."
    )
    print(
        "Monitor new categories and category-frequency drift because a "
        "previously unseen category may change how an encoder behaves."
    )
    print(
        "Treat categorical values as data rather than trusted configuration. "
        "Validate them before using them as dictionary keys, serialized "
        "feature names, database columns, or downstream identifiers."
    )


if __name__ == "__main__":
    main()
