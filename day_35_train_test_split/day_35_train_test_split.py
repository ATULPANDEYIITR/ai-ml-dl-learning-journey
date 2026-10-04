"""
Train-Test Split: training, validation, testing, leakage, and reproducibility.

Self-contained educational implementation using only the Python standard library.
The script demonstrates:
- deterministic dataset construction
- random train/validation/test splitting
- stratification
- time-aware splitting
- group-aware splitting
- preprocessing fitted only on training data
- data leakage detection
- reproducible experiments
- a small nearest-centroid classifier
- evaluation on untouched test data
- common split failure modes
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import random
import statistics
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Record:
    """A synthetic repository-change record used as an ML dataset row."""
    record_id: int
    lines_changed: float
    files_changed: float
    review_comments: float
    prior_failures: float
    author_experience: float
    label: int
    author_id: str
    timestamp: datetime


def make_dataset(seed: int = 2026, size: int = 180) -> list[Record]:
    """
    Create a realistic synthetic classification dataset.

    The target is whether a proposed repository change requires additional
    engineering intervention. The target itself is generated independently
    from the feature-generation process so that the example does not
    accidentally leak the target into the predictors.
    """
    rng = random.Random(seed)
    authors = [f"developer-{i:02d}" for i in range(12)]
    start = datetime(2026, 1, 1, 9, 0)

    rows: list[Record] = []

    for record_id in range(size):
        author = rng.choice(authors)
        experience = rng.randint(1, 10)
        files = max(1, int(rng.expovariate(1 / 3.0)))
        lines = max(5, int(rng.expovariate(1 / 70.0)))
        comments = max(0, int(rng.expovariate(1 / 2.5)))
        failures = max(0, int(rng.expovariate(1 / 1.3)))

        score = (
            0.025 * lines
            + 0.55 * files
            + 0.75 * comments
            + 0.9 * failures
            - 0.65 * experience
            + rng.gauss(0, 2.5)
        )
        label = int(score > 3.0)

        rows.append(
            Record(
                record_id=record_id,
                lines_changed=float(lines),
                files_changed=float(files),
                review_comments=float(comments),
                prior_failures=float(failures),
                author_experience=float(experience),
                label=label,
                author_id=author,
                timestamp=start + timedelta(hours=8 * record_id),
            )
        )

    return rows


def random_split(
    rows: Sequence[Record],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Record], list[Record], list[Record]]:
    """
    Perform a random three-way split.

    The test set is separated before model selection. The same random seed
    reproduces the exact partition when the input ordering is unchanged.
    """
    if not math.isclose(train_ratio + validation_ratio, 0.85):
        raise ValueError("This demonstration expects 70% train and 15% validation.")

    if len(rows) < 10:
        raise ValueError("Dataset is too small for a meaningful three-way split.")

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)

    train_end = int(len(indices) * train_ratio)
    validation_end = train_end + int(len(indices) * validation_ratio)

    train = [rows[i] for i in indices[:train_end]]
    validation = [rows[i] for i in indices[train_end:validation_end]]
    test = [rows[i] for i in indices[validation_end:]]

    return train, validation, test


def stratified_split(
    rows: Sequence[Record],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Record], list[Record], list[Record]]:
    """
    Split while approximately preserving the target distribution.

    Stratification is useful when classes are imbalanced. It does not solve
    temporal dependence or repeated-entity leakage; those require different
    splitting strategies.
    """
    by_class: dict[int, list[Record]] = {}
    for row in rows:
        by_class.setdefault(row.label, []).append(row)

    rng = random.Random(seed)
    train: list[Record] = []
    validation: list[Record] = []
    test: list[Record] = []

    for label_rows in by_class.values():
        shuffled = label_rows[:]
        rng.shuffle(shuffled)

        train_end = int(len(shuffled) * train_ratio)
        validation_end = train_end + int(len(shuffled) * validation_ratio)

        train.extend(shuffled[:train_end])
        validation.extend(shuffled[train_end:validation_end])
        test.extend(shuffled[validation_end:])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)

    return train, validation, test


def chronological_split(
    rows: Sequence[Record],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[list[Record], list[Record], list[Record]]:
    """
    Time-aware split.

    Future observations never become training observations for a model that
    is supposed to represent historical deployment conditions.
    """
    ordered = sorted(rows, key=lambda row: row.timestamp)
    train_end = int(len(ordered) * train_ratio)
    validation_end = train_end + int(len(ordered) * validation_ratio)

    return (
        ordered[:train_end],
        ordered[train_end:validation_end],
        ordered[validation_end:],
    )


def group_split(
    rows: Sequence[Record],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Record], list[Record], list[Record]]:
    """
    Group-aware split.

    All observations belonging to one author stay in exactly one partition.
    This prevents a model from learning author-specific signals from training
    rows and then being tested on the same people.
    """
    groups: dict[str, list[Record]] = {}
    for row in rows:
        groups.setdefault(row.author_id, []).append(row)

    group_names = list(groups)
    random.Random(seed).shuffle(group_names)

    train_target = len(rows) * train_ratio
    validation_target = len(rows) * validation_ratio

    train: list[Record] = []
    validation: list[Record] = []
    test: list[Record] = []

    for group in group_names:
        bucket = groups[group]
        if len(train) + len(bucket) <= train_target:
            train.extend(bucket)
        elif len(validation) + len(bucket) <= validation_target:
            validation.extend(bucket)
        else:
            test.extend(bucket)

    return train, validation, test


class StandardScaler:
    """
    Training-only feature normalization.

    The critical rule is that mean and standard deviation are learned from
    training rows. Validation and test rows are transformed using those
    already-learned parameters.
    """

    def __init__(self) -> None:
        self.means: list[float] | None = None
        self.stds: list[float] | None = None

    def fit(self, matrix: Sequence[Sequence[float]]) -> None:
        if not matrix:
            raise ValueError("Cannot fit scaler on an empty matrix.")

        width = len(matrix[0])
        if width == 0 or any(len(row) != width for row in matrix):
            raise ValueError("Feature matrix must be rectangular.")

        self.means = [
            statistics.fmean(row[column] for row in matrix)
            for column in range(width)
        ]

        self.stds = []
        for column in range(width):
            values = [row[column] for row in matrix]
            std = statistics.pstdev(values)
            self.stds.append(std if std > 0 else 1.0)

    def transform(self, matrix: Sequence[Sequence[float]]) -> list[list[float]]:
        if self.means is None or self.stds is None:
            raise RuntimeError("Scaler must be fitted before transformation.")

        if any(len(row) != len(self.means) for row in matrix):
            raise ValueError("Feature width does not match fitted scaler.")

        return [
            [
                (value - mean) / std
                for value, mean, std in zip(row, self.means, self.stds)
            ]
            for row in matrix
        ]

    def fit_transform(self, matrix: Sequence[Sequence[float]]) -> list[list[float]]:
        self.fit(matrix)
        return self.transform(matrix)


FEATURES: tuple[str, ...] = (
    "lines_changed",
    "files_changed",
    "review_comments",
    "prior_failures",
    "author_experience",
)


def to_matrix(rows: Sequence[Record]) -> list[list[float]]:
    return [[getattr(row, feature) for feature in FEATURES] for row in rows]


def labels(rows: Sequence[Record]) -> list[int]:
    return [row.label for row in rows]


class NearestCentroidClassifier:
    """
    Small classifier used to make the split mechanics executable.

    Each class is represented by its feature centroid. A new observation is
    assigned to the nearest centroid using squared Euclidean distance.
    """

    def __init__(self) -> None:
        self.centroids: dict[int, list[float]] = {}

    def fit(self, matrix: Sequence[Sequence[float]], target: Sequence[int]) -> None:
        if len(matrix) != len(target) or not matrix:
            raise ValueError("Training features and labels must be non-empty and aligned.")

        grouped: dict[int, list[Sequence[float]]] = {}
        for row, label in zip(matrix, target):
            grouped.setdefault(label, []).append(row)

        self.centroids = {
            label: [
                statistics.fmean(row[column] for row in class_rows)
                for column in range(len(class_rows[0]))
            ]
            for label, class_rows in grouped.items()
        }

    @staticmethod
    def distance(a: Sequence[float], b: Sequence[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def predict(self, matrix: Sequence[Sequence[float]]) -> list[int]:
        if not self.centroids:
            raise RuntimeError("Classifier must be fitted before prediction.")

        return [
            min(
                self.centroids,
                key=lambda label: self.distance(row, self.centroids[label]),
            )
            for row in matrix
        ]


def classification_metrics(
    actual: Sequence[int],
    predicted: Sequence[int],
) -> dict[str, float]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted labels must have equal non-zero length.")

    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))

    accuracy = (tp + tn) / len(actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": float(tp),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
    }


def class_distribution(rows: Sequence[Record]) -> dict[int, float]:
    counts: dict[int, int] = {}
    for row in rows:
        counts[row.label] = counts.get(row.label, 0) + 1
    total = len(rows)
    return {label: count / total for label, count in counts.items()}


def compare_distributions(
    left: Sequence[Record],
    right: Sequence[Record],
) -> float:
    """Return absolute difference in positive-class proportions."""
    left_rate = class_distribution(left).get(1, 0.0)
    right_rate = class_distribution(right).get(1, 0.0)
    return abs(left_rate - right_rate)


def demonstrate_leakage(rows: Sequence[Record]) -> None:
    """
    Show the conceptual difference between correct and contaminated scaling.

    Correct scaling fits preprocessing only on train data. The intentionally
    bad scaler below sees validation and test data before model evaluation,
    allowing information about their distributions to influence training.
    """
    train, validation, test = random_split(rows, seed=11)

    train_matrix = to_matrix(train)
    validation_matrix = to_matrix(validation)
    test_matrix = to_matrix(test)

    correct_scaler = StandardScaler()
    correct_train = correct_scaler.fit_transform(train_matrix)
    correct_validation = correct_scaler.transform(validation_matrix)
    correct_test = correct_scaler.transform(test_matrix)

    leaked_scaler = StandardScaler()
    leaked_scaler.fit(train_matrix + validation_matrix + test_matrix)
    leaked_train = leaked_scaler.transform(train_matrix)

    print("\nLeakage demonstration")
    print(
        "Correct scaler training mean:",
        [round(value, 2) for value in correct_scaler.means or []],
    )
    print(
        "Leaked scaler training mean:",
        [round(value, 2) for value in leaked_scaler.means or []],
    )
    print(
        "The correct pipeline has transformed matrices:",
        len(correct_train),
        len(correct_validation),
        len(correct_test),
    )
    print("The leaked pipeline allowed held-out distributions to influence preprocessing.")
    print("This can make validation/test estimates systematically optimistic.")


def run_reproducibility_check(rows: Sequence[Record]) -> None:
    first = random_split(rows, seed=99)
    second = random_split(rows, seed=99)

    first_ids = tuple(row.record_id for row in first[0])
    second_ids = tuple(row.record_id for row in second[0])

    print("\nReproducibility")
    print("Same seed produces identical training partition:", first_ids == second_ids)

    third = random_split(rows, seed=100)
    third_ids = tuple(row.record_id for row in third[0])
    print("Changing seed changes partition:", first_ids != third_ids)


def train_with_validation(
    train: Sequence[Record],
    validation: Sequence[Record],
) -> tuple[NearestCentroidClassifier, StandardScaler, float]:
    """
    Fit preprocessing and a classifier on training data, then use validation
    performance to assess the current model configuration.

    The test set is intentionally absent from this function. This makes it
    structurally harder to tune the model against the final test set.
    """
    scaler = StandardScaler()
    train_features = scaler.fit_transform(to_matrix(train))
    validation_features = scaler.transform(to_matrix(validation))

    model = NearestCentroidClassifier()
    model.fit(train_features, labels(train))

    predictions = model.predict(validation_features)
    metrics = classification_metrics(labels(validation), predictions)

    return model, scaler, metrics["f1"]


def evaluate_final(
    model: NearestCentroidClassifier,
    scaler: StandardScaler,
    test: Sequence[Record],
) -> dict[str, float]:
    """
    Evaluate once on the untouched test set.

    A production experiment should report this result only after decisions
    about preprocessing, model configuration, and thresholds are finished.
    """
    features = scaler.transform(to_matrix(test))
    predictions = model.predict(features)
    return classification_metrics(labels(test), predictions)


def demonstrate_split_choice(rows: Sequence[Record]) -> None:
    random_train, random_validation, random_test = random_split(rows)
    strat_train, strat_validation, strat_test = stratified_split(rows)
    time_train, time_validation, time_test = chronological_split(rows)
    group_train, group_validation, group_test = group_split(rows)

    print("\nSplit comparison")
    print(
        "Random:",
        len(random_train),
        len(random_validation),
        len(random_test),
        "positive-rate gap train/test =",
        round(compare_distributions(random_train, random_test), 3),
    )
    print(
        "Stratified:",
        len(strat_train),
        len(strat_validation),
        len(strat_test),
        "positive-rate gap train/test =",
        round(compare_distributions(strat_train, strat_test), 3),
    )
    print(
        "Chronological:",
        len(time_train),
        len(time_validation),
        len(time_test),
        "train latest =",
        time_train[-1].timestamp.date(),
        "test earliest =",
        time_test[0].timestamp.date(),
    )

    train_authors = {row.author_id for row in group_train}
    validation_authors = {row.author_id for row in group_validation}
    test_authors = {row.author_id for row in group_test}

    print(
        "Group-aware:",
        len(group_train),
        len(group_validation),
        len(group_test),
        "author overlap =",
        len(
            (train_authors & validation_authors)
            | (train_authors & test_authors)
            | (validation_authors & test_authors)
        ),
    )


def demonstrate_duplicate_detection(rows: Sequence[Record]) -> None:
    """
    Duplicate records can silently defeat a random split.

    If the same underlying observation appears in train and test, the model
    may appear to generalize when it has actually seen the observation already.
    """
    duplicated = list(rows)
    duplicated.append(rows[0])

    signatures: dict[tuple[float, ...], list[int]] = {}
    for row in duplicated:
        signature = (
            row.lines_changed,
            row.files_changed,
            row.review_comments,
            row.prior_failures,
            row.author_experience,
            float(row.label),
        )
        signatures.setdefault(signature, []).append(row.record_id)

    duplicate_groups = [ids for ids in signatures.values() if len(ids) > 1]
    print("\nDuplicate detection")
    print("Duplicate feature/label groups found:", len(duplicate_groups))


def main() -> None:
    print("TRAIN / VALIDATION / TEST WORKFLOW")
    print("=" * 42)

    rows = make_dataset()
    print("Dataset size:", len(rows))
    print("Positive-class rate:", round(class_distribution(rows).get(1, 0.0), 3))

    demonstrate_split_choice(rows)
    run_reproducibility_check(rows)
    demonstrate_duplicate_detection(rows)
    demonstrate_leakage(rows)

    train, validation, test = stratified_split(rows, seed=2026)
    model, scaler, validation_f1 = train_with_validation(train, validation)
    final_metrics = evaluate_final(model, scaler, test)

    print("\nModel selection and final evaluation")
    print("Validation F1:", round(validation_f1, 4))
    print("Test metrics:")
    for name, value in final_metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")

    print("\nOperational rules")
    print("Training data is used to fit model parameters and preprocessing.")
    print("Validation data supports model or configuration decisions.")
    print("Test data estimates final generalization after those decisions.")
    print("Random splitting is inappropriate when time or entity dependence matters.")
    print("Leakage can enter through features, preprocessing, duplicates, targets, or future information.")
    print("A seed makes a random experiment repeatable, but reproducibility also requires stable data and code.")


if __name__ == "__main__":
    main()
