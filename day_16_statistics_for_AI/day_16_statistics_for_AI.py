"""
Statistics for AI
=================
Population, sample, mean, median, mode, variance, and standard deviation.

This standalone study script progresses from elementary descriptive statistics
to AI-oriented statistical analysis. It uses only the Python standard library.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# 1. FUNDAMENTAL TERMINOLOGY
# ---------------------------------------------------------------------------
#
# Population:
#   The complete collection of observations relevant to a study.
#
# Sample:
#   A subset of the population selected for analysis.
#
# Parameter:
#   A numerical property of a population, such as population mean.
#
# Statistic:
#   A numerical property calculated from a sample, such as sample mean.
#
# Example:
#   Population = all students in a university.
#   Sample = 100 students selected from that university.
#
# In AI and machine learning, a dataset is often treated as a sample from a
# broader data-generating process. This distinction matters because a model
# trained on observed data is usually expected to generalize to unseen data.


def population_mean(values: Sequence[float]) -> float:
    """Calculate the arithmetic mean of an entire population."""
    if not values:
        raise ValueError("The population cannot be empty.")
    return sum(values) / len(values)


def sample_mean(values: Sequence[float]) -> float:
    """Calculate the arithmetic mean of a sample."""
    if not values:
        raise ValueError("The sample cannot be empty.")
    return sum(values) / len(values)


# Mean is identical mathematically for a population and a sample.
# The distinction appears primarily when interpreting the result:
# population mean -> parameter
# sample mean     -> statistic


# ---------------------------------------------------------------------------
# 2. MEDIAN
# ---------------------------------------------------------------------------

def median(values: Sequence[float]) -> float:
    """
    Calculate the median.

    For an odd number of observations:
        median = middle observation

    For an even number:
        median = average of the two middle observations
    """
    if not values:
        raise ValueError("Median is undefined for an empty dataset.")

    ordered = sorted(values)
    n = len(ordered)
    middle = n // 2

    if n % 2 == 1:
        return float(ordered[middle])

    return (ordered[middle - 1] + ordered[middle]) / 2.0


# Median is resistant to extreme values.
# Example:
income = [30_000, 32_000, 35_000, 36_000, 2_000_000]
print("Income mean:", population_mean(income))
print("Income median:", median(income))
print(
    "The mean is strongly affected by the extreme income, "
    "while the median is much more stable."
)


# ---------------------------------------------------------------------------
# 3. MODE
# ---------------------------------------------------------------------------

def modes(values: Sequence[float]) -> list[float]:
    """
    Return all modes.

    A mode is an observation with the highest frequency.
    If every value occurs once, there is no unique mode.
    """
    if not values:
        raise ValueError("Mode is undefined for an empty dataset.")

    counts = Counter(values)
    highest_frequency = max(counts.values())

    if highest_frequency == 1:
        return []

    return sorted(
        value for value, frequency in counts.items()
        if frequency == highest_frequency
    )


print("Modes:", modes([1, 2, 2, 3, 3, 4]))


# ---------------------------------------------------------------------------
# 4. RANGE AND DEVIATIONS
# ---------------------------------------------------------------------------

def data_range(values: Sequence[float]) -> float:
    """Difference between maximum and minimum."""
    if not values:
        raise ValueError("Range is undefined for an empty dataset.")
    return max(values) - min(values)


def deviations_from_mean(values: Sequence[float]) -> list[float]:
    """Return x_i - mean for every observation."""
    mean_value = population_mean(values)
    return [value - mean_value for value in values]


scores = [60, 70, 80, 90, 100]
print("Range:", data_range(scores))
print("Deviations:", deviations_from_mean(scores))

# Important property:
# The deviations from the arithmetic mean sum to approximately zero.
print("Sum of deviations:", sum(deviations_from_mean(scores)))


# ---------------------------------------------------------------------------
# 5. POPULATION VARIANCE
# ---------------------------------------------------------------------------

def population_variance(values: Sequence[float]) -> float:
    """
    Calculate population variance.

    Formula:
        σ² = Σ(x_i - μ)² / N

    N is the complete population size.
    """
    if not values:
        raise ValueError("Variance is undefined for an empty dataset.")

    mean_value = population_mean(values)
    squared_deviations = [
        (value - mean_value) ** 2
        for value in values
    ]

    return sum(squared_deviations) / len(values)


def population_standard_deviation(values: Sequence[float]) -> float:
    """Standard deviation is the square root of population variance."""
    return sqrt(population_variance(values))


# ---------------------------------------------------------------------------
# 6. SAMPLE VARIANCE
# ---------------------------------------------------------------------------

def sample_variance(values: Sequence[float]) -> float:
    """
    Calculate unbiased sample variance using Bessel's correction.

    Formula:
        s² = Σ(x_i - x̄)² / (n - 1)

    The n - 1 denominator is used when estimating population variance from
    a sample. This is Bessel's correction.

    A sample containing only one observation cannot provide a conventional
    sample variance estimate.
    """
    if len(values) < 2:
        raise ValueError(
            "At least two observations are required for sample variance."
        )

    mean_value = sample_mean(values)
    squared_deviations = [
        (value - mean_value) ** 2
        for value in values
    ]

    return sum(squared_deviations) / (len(values) - 1)


def sample_standard_deviation(values: Sequence[float]) -> float:
    """Square root of sample variance."""
    return sqrt(sample_variance(values))


dataset = [10, 12, 14, 16, 18]

print("\nBasic descriptive statistics")
print("----------------------------")
print("Data:", dataset)
print("Mean:", population_mean(dataset))
print("Median:", median(dataset))
print("Mode:", modes(dataset))
print("Range:", data_range(dataset))
print("Population variance:", population_variance(dataset))
print("Population standard deviation:", population_standard_deviation(dataset))
print("Sample variance:", sample_variance(dataset))
print("Sample standard deviation:", sample_standard_deviation(dataset))


# ---------------------------------------------------------------------------
# 7. WHY VARIANCE SQUARED DEVIATIONS
# ---------------------------------------------------------------------------
#
# Simply averaging deviations does not measure spread because positive and
# negative deviations cancel.
#
# Squaring:
#   1. removes the sign,
#   2. makes large deviations more influential,
#   3. creates a mathematically useful quantity for many statistical methods.
#
# Standard deviation takes the square root so that the resulting unit matches
# the original measurement.


def demonstrate_cancellation(values: Sequence[float]) -> None:
    mean_value = population_mean(values)
    deviations = [x - mean_value for x in values]
    absolute_deviations = [abs(x) for x in deviations]
    squared_deviations = [x * x for x in deviations]

    print("\nSpread demonstration")
    print("--------------------")
    print("Values:", values)
    print("Mean:", mean_value)
    print("Deviations:", deviations)
    print("Sum of deviations:", sum(deviations))
    print("Mean absolute deviation:", population_mean(absolute_deviations))
    print("Mean squared deviation:", population_mean(squared_deviations))


demonstrate_cancellation([2, 4, 6, 8, 10])


# ---------------------------------------------------------------------------
# 8. A REUSABLE DESCRIPTIVE STATISTICS CLASS
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DescriptiveStatistics:
    """Container for common descriptive statistics."""

    count: int
    mean: float
    median: float
    modes: list[float]
    minimum: float
    maximum: float
    range: float
    population_variance: float
    population_standard_deviation: float
    sample_variance: float | None
    sample_standard_deviation: float | None

    @classmethod
    def from_data(cls, values: Sequence[float]) -> "DescriptiveStatistics":
        if not values:
            raise ValueError("Data cannot be empty.")

        count = len(values)
        sample_var = sample_variance(values) if count >= 2 else None

        return cls(
            count=count,
            mean=population_mean(values),
            median=median(values),
            modes=modes(values),
            minimum=min(values),
            maximum=max(values),
            range=data_range(values),
            population_variance=population_variance(values),
            population_standard_deviation=population_standard_deviation(values),
            sample_variance=sample_var,
            sample_standard_deviation=(
                sqrt(sample_var) if sample_var is not None else None
            ),
        )

    def display(self) -> None:
        for field_name, field_value in self.__dict__.items():
            print(f"{field_name}: {field_value}")


stats = DescriptiveStatistics.from_data(
    [72, 75, 75, 81, 83, 88, 91, 75]
)

print("\nReusable statistics object")
print("--------------------------")
stats.display()


# ---------------------------------------------------------------------------
# 9. EDGE CASES
# ---------------------------------------------------------------------------

def safe_statistics_demo() -> None:
    test_cases = {
        "empty": [],
        "single value": [42],
        "constant": [7, 7, 7, 7],
        "negative values": [-5, -2, 0, 3, 9],
        "decimal values": [1.1, 1.2, 1.3, 1.4],
    }

    for name, values in test_cases.items():
        print(f"\n{name}: {values}")

        if not values:
            print("No descriptive statistics can be calculated.")
            continue

        print("Mean:", population_mean(values))
        print("Median:", median(values))
        print("Mode:", modes(values))
        print("Population variance:", population_variance(values))
        print("Population standard deviation:",
              population_standard_deviation(values))

        if len(values) >= 2:
            print("Sample variance:", sample_variance(values))
        else:
            print("Sample variance: undefined for one observation")


safe_statistics_demo()


# ---------------------------------------------------------------------------
# 10. EFFECT OF OUTLIERS
# ---------------------------------------------------------------------------

normal_data = [48, 49, 50, 51, 52]
outlier_data = [48, 49, 50, 51, 500]

print("\nOutlier analysis")
print("----------------")
for name, values in [
    ("Without outlier", normal_data),
    ("With outlier", outlier_data),
]:
    print(name)
    print("  mean =", population_mean(values))
    print("  median =", median(values))
    print("  standard deviation =",
          population_standard_deviation(values))


# Mean and standard deviation are sensitive to extreme observations.
# Median is generally more robust to isolated extreme observations.
#
# This distinction is important in AI data preprocessing because unusual
# values may represent:
#   - genuine rare events,
#   - measurement errors,
#   - data-entry errors,
#   - fraud,
#   - sensor faults,
#   - legitimate extreme behavior.


# ---------------------------------------------------------------------------
# 11. COMPARING DATASETS WITH MEAN AND STANDARD DEVIATION
# ---------------------------------------------------------------------------

dataset_a = [49, 50, 51, 50, 50]
dataset_b = [20, 35, 50, 65, 80]

print("\nSame mean, different spread")
print("---------------------------")
for name, values in [("A", dataset_a), ("B", dataset_b)]:
    print(
        name,
        "mean =", population_mean(values),
        "standard deviation =",
        population_standard_deviation(values),
    )

# Two datasets can have the same mean but very different variability.
# Mean alone therefore does not describe a distribution completely.


# ---------------------------------------------------------------------------
# 12. NORMALIZATION USING MEAN AND STANDARD DEVIATION
# ---------------------------------------------------------------------------

def z_scores(values: Sequence[float]) -> list[float]:
    """
    Standardize observations using population mean and population standard
    deviation.

    z = (x - mean) / standard_deviation

    A zero standard deviation means every observation is identical, so
    conventional z-score standardization is undefined.
    """
    mean_value = population_mean(values)
    standard_deviation = population_standard_deviation(values)

    if standard_deviation == 0:
        raise ValueError(
            "Z-score standardization is undefined for constant data."
        )

    return [
        (value - mean_value) / standard_deviation
        for value in values
    ]


print("\nZ-score standardization")
print("-----------------------")
values = [10, 20, 30, 40, 50]
standardized = z_scores(values)
print("Original:", values)
print("Z-scores:", standardized)
print("Standardized mean:",
      population_mean(standardized))
print("Standardized standard deviation:",
      population_standard_deviation(standardized))


# ---------------------------------------------------------------------------
# 13. POPULATION VS SAMPLE: A CONCRETE AI EXAMPLE
# ---------------------------------------------------------------------------
#
# Suppose a company has 1,000,000 transactions but only analyzes 10,000.
#
# If the 1,000,000 transactions are the complete target population:
#   population variance uses N.
#
# If the 10,000 transactions are a sample used to estimate the full
# population's variability:
#   sample variance normally uses n - 1.
#
# Confusing these interpretations can produce incorrect statistical claims.


population = [100, 102, 98, 101, 99]
sample = [100, 102, 98]

print("\nPopulation versus sample")
print("------------------------")
print("Population variance:",
      population_variance(population))
print("Sample variance:",
      sample_variance(sample))


# ---------------------------------------------------------------------------
# 14. STREAMING MEAN AND VARIANCE: WELFORD'S ALGORITHM
# ---------------------------------------------------------------------------
#
# For huge AI datasets, storing every value in memory may be undesirable.
# Welford's algorithm computes mean and variance incrementally.
#
# State:
#   count
#   mean
#   M2
#
# Population variance = M2 / n
# Sample variance     = M2 / (n - 1)
#
# This approach is numerically more stable than some naive formulas such as
# E[X²] - E[X]², especially when values are very large and close together.


class OnlineStatistics:
    """Numerically stable streaming mean and variance."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1

        delta = value - self.mean
        self.mean += delta / self.count
        delta_after_mean_update = value - self.mean
        self.m2 += delta * delta_after_mean_update

    @property
    def population_variance(self) -> float:
        if self.count == 0:
            raise ValueError("No observations available.")
        return self.m2 / self.count

    @property
    def sample_variance(self) -> float:
        if self.count < 2:
            raise ValueError(
                "At least two observations are required."
            )
        return self.m2 / (self.count - 1)

    @property
    def population_standard_deviation(self) -> float:
        return sqrt(self.population_variance)

    @property
    def sample_standard_deviation(self) -> float:
        return sqrt(self.sample_variance)


streaming = OnlineStatistics()

for value in [10, 12, 14, 16, 18]:
    streaming.update(value)

print("\nStreaming statistics")
print("--------------------")
print("Count:", streaming.count)
print("Mean:", streaming.mean)
print("Population variance:", streaming.population_variance)
print("Sample variance:", streaming.sample_variance)
print(
    "Population standard deviation:",
    streaming.population_standard_deviation,
)


# ---------------------------------------------------------------------------
# 15. COMBINING TWO GROUPS
# ---------------------------------------------------------------------------
#
# Means cannot always be combined by taking their simple average.
#
# If group A has 10 observations with mean 20 and group B has 90 observations
# with mean 40, the combined mean is weighted by group sizes.
#
# combined_mean = (n1*m1 + n2*m2) / (n1+n2)


def combine_means(
    count_a: int,
    mean_a: float,
    count_b: int,
    mean_b: float,
) -> float:
    if count_a < 0 or count_b < 0:
        raise ValueError("Counts cannot be negative.")

    total_count = count_a + count_b

    if total_count == 0:
        raise ValueError("At least one observation is required.")

    return (
        count_a * mean_a + count_b * mean_b
    ) / total_count


print("\nWeighted mean")
print("------------")
print("Combined mean:",
      combine_means(10, 20, 90, 40))


# ---------------------------------------------------------------------------
# 16. GROUPED AI MODEL METRICS
# ---------------------------------------------------------------------------
#
# Suppose model errors are measured for different user groups.
# Mean error and variance can reveal whether performance is stable.
#
# This is descriptive analysis, not a complete fairness assessment.


model_errors = {
    "group_A": [0.10, 0.12, 0.09, 0.11, 0.10],
    "group_B": [0.10, 0.30, 0.08, 0.25, 0.12],
}

print("\nAI model error variability")
print("--------------------------")

for group, errors in model_errors.items():
    print(
        group,
        "mean_error =",
        population_mean(errors),
        "std_dev =",
        population_standard_deviation(errors),
        "median_error =",
        median(errors),
    )

# A similar mean error does not necessarily imply similar consistency.
# Standard deviation can expose differences in variability.


# ---------------------------------------------------------------------------
# 17. NUMERICAL PRECISION AND FLOATING-POINT CONSIDERATIONS
# ---------------------------------------------------------------------------

decimal_values = [0.1, 0.2, 0.3]

print("\nFloating-point example")
print("----------------------")
print("0.1 + 0.2 =", 0.1 + 0.2)
print("Mean =", population_mean(decimal_values))

# Binary floating-point representation cannot represent many decimal
# fractions exactly. Statistical software therefore works with numerical
# approximations. For ordinary descriptive analysis this is usually acceptable,
# but scientific, financial, and high-precision applications may require
# careful numerical design.


# ---------------------------------------------------------------------------
# 18. COMPREHENSIVE REPORT FUNCTION
# ---------------------------------------------------------------------------

def describe(values: Sequence[float]) -> dict[str, object]:
    """
    Return a compact descriptive-statistics report.

    The function deliberately separates population and sample quantities.
    """
    if not values:
        raise ValueError("Dataset cannot be empty.")

    report: dict[str, object] = {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "range": data_range(values),
        "mean": population_mean(values),
        "median": median(values),
        "modes": modes(values),
        "population_variance": population_variance(values),
        "population_standard_deviation":
            population_standard_deviation(values),
    }

    if len(values) >= 2:
        report["sample_variance"] = sample_variance(values)
        report["sample_standard_deviation"] = sample_standard_deviation(values)
    else:
        report["sample_variance"] = None
        report["sample_standard_deviation"] = None

    return report


print("\nComplete report")
print("---------------")
for key, value in describe([4, 5, 5, 7, 8, 10]).items():
    print(f"{key}: {value}")


# ---------------------------------------------------------------------------
# 19. TESTS
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Basic assertions for correctness."""
    values = [1, 2, 3, 4, 5]

    assert population_mean(values) == 3.0
    assert median(values) == 3.0
    assert modes(values) == []
    assert data_range(values) == 4.0
    assert population_variance(values) == 2.0
    assert sample_variance(values) == 2.5
    assert population_standard_deviation(values) == sqrt(2.0)
    assert sample_standard_deviation(values) == sqrt(2.5)

    assert median([1, 2, 3, 4]) == 2.5
    assert modes([1, 1, 2, 2, 3]) == [1, 2]

    try:
        population_mean([])
    except ValueError:
        pass
    else:
        raise AssertionError("Empty mean should fail.")

    try:
        sample_variance([10])
    except ValueError:
        pass
    else:
        raise AssertionError("One-value sample variance should fail.")

    constant = [7, 7, 7]
    assert population_variance(constant) == 0.0

    print("\nAll built-in tests passed.")


run_tests()


# ---------------------------------------------------------------------------
# 20. KEY INTERPRETATION RULES
# ---------------------------------------------------------------------------
#
# Mean:
#   Measures central tendency but is sensitive to outliers.
#
# Median:
#   Middle ordered observation and usually more robust to extreme values.
#
# Mode:
#   Most frequent observation(s); useful for categorical or repeated data.
#
# Variance:
#   Average squared deviation from the mean for a population, or the
#   n-1-corrected estimator for a sample.
#
# Standard deviation:
#   Square root of variance; expressed in the original unit.
#
# Population:
#   Complete target collection.
#
# Sample:
#   Observed subset used to learn about a broader population.
#
# AI relevance:
#   These statistics are used for exploratory data analysis, feature
#   preprocessing, anomaly investigation, experiment analysis, model
#   monitoring, quality control, and understanding distributions.
#
# Important limitation:
#   Mean, median, mode, variance, and standard deviation do not describe every
#   property of a dataset. Two datasets can share these statistics while
#   having different shapes, skewness, multimodality, tails, or relationships
#   between variables.
#
# A responsible AI workflow therefore treats descriptive statistics as one
# part of data analysis rather than a complete description of a dataset.
