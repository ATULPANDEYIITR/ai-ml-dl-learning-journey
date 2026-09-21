"""
STATISTICAL INFERENCE
Sampling, Confidence Intervals, Hypothesis Testing, and P-values

A comprehensive standalone study and executable demonstration.

This script progresses from basic sampling concepts to confidence intervals,
hypothesis tests, p-values, Type I/II errors, statistical power, effect sizes,
one-sample and two-sample procedures, proportions, paired data, chi-square
testing, correlation, bootstrap inference, permutation testing, multiple
comparisons, and practical interpretation.

Only the Python standard library is used.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Sequence


# ============================================================================
# 1. FOUNDATIONS
# ============================================================================

def mean(values: Sequence[float]) -> float:
    """Arithmetic mean."""
    if not values:
        raise ValueError("At least one value is required.")
    return sum(values) / len(values)


def variance(values: Sequence[float], sample: bool = True) -> float:
    """
    Variance.

    sample=True calculates sample variance using n-1.
    sample=False calculates population variance using n.
    """
    n = len(values)
    if n < 2 and sample:
        raise ValueError("At least two observations are required for sample variance.")

    average = mean(values)
    denominator = n - 1 if sample else n
    return sum((x - average) ** 2 for x in values) / denominator


def standard_deviation(values: Sequence[float], sample: bool = True) -> float:
    """Standard deviation."""
    return math.sqrt(variance(values, sample=sample))


def standard_error(values: Sequence[float]) -> float:
    """
    Standard error of the sample mean.

    SE = s / sqrt(n)

    Standard error measures sampling variability of an estimator, not
    variability of individual observations.
    """
    return standard_deviation(values) / math.sqrt(len(values))


def describe_sample(values: Sequence[float]) -> None:
    print("Sample:", list(values))
    print("n:", len(values))
    print("Mean:", round(mean(values), 4))
    print("Sample variance:", round(variance(values), 4))
    print("Sample standard deviation:", round(standard_deviation(values), 4))
    print("Standard error:", round(standard_error(values), 4))


# ============================================================================
# 2. POPULATIONS, SAMPLES, PARAMETERS, AND STATISTICS
# ============================================================================

def demonstrate_population_and_sample() -> None:
    """
    A population contains all units of interest.

    A sample is a subset observed from that population.

    A parameter describes the population:
        population mean = μ
        population proportion = p
        population variance = σ²

    A statistic describes the sample:
        sample mean = x̄
        sample proportion = p̂
        sample variance = s²
    """
    population = list(range(1, 101))
    sample = random.Random(42).sample(population, 10)

    population_mean = mean(population)
    sample_mean = mean(sample)

    print("\n--- Population and Sample ---")
    print("Population mean:", population_mean)
    print("Sample:", sample)
    print("Sample mean:", round(sample_mean, 4))


# ============================================================================
# 3. SAMPLING METHODS
# ============================================================================

def simple_random_sample(population: Sequence, sample_size: int, seed: int = 42):
    """Every population element has an equal probability of selection."""
    if not 0 < sample_size <= len(population):
        raise ValueError("Invalid sample size.")
    return random.Random(seed).sample(list(population), sample_size)


def systematic_sample(population: Sequence, sample_size: int):
    """
    Basic systematic sampling.

    The population is divided into approximately equal intervals and
    observations are selected at a fixed interval.
    """
    n = len(population)
    if not 0 < sample_size <= n:
        raise ValueError("Invalid sample size.")

    interval = n / sample_size
    indices = [min(n - 1, int(i * interval)) for i in range(sample_size)]
    return [population[index] for index in indices]


def stratified_sample(
    population: Sequence[tuple[str, float]],
    sample_per_stratum: int,
    seed: int = 42,
):
    """
    Stratified sampling.

    Population is divided into meaningful subgroups and observations are
    sampled separately within each subgroup.
    """
    rng = random.Random(seed)
    strata: dict[str, list[float]] = {}

    for group, value in population:
        strata.setdefault(group, []).append(value)

    result = {}
    for group, values in strata.items():
        if sample_per_stratum > len(values):
            raise ValueError(f"Not enough observations in stratum {group}.")
        result[group] = rng.sample(values, sample_per_stratum)

    return result


def cluster_sample(
    population: Sequence[Sequence],
    number_of_clusters: int,
    clusters_to_select: int,
    seed: int = 42,
):
    """
    Cluster sampling.

    Entire naturally occurring groups are selected instead of sampling
    individuals independently from the entire population.
    """
    if clusters_to_select > number_of_clusters:
        raise ValueError("Cannot select more clusters than exist.")

    rng = random.Random(seed)
    selected = rng.sample(range(number_of_clusters), clusters_to_select)

    result = []
    for cluster_index in selected:
        result.extend(population[cluster_index])

    return result


# ============================================================================
# 4. CENTRAL LIMIT THEOREM SIMULATION
# ============================================================================

def demonstrate_central_limit_theorem(
    population: Sequence[float],
    sample_size: int = 30,
    repetitions: int = 5000,
    seed: int = 42,
) -> None:
    """
    Demonstrates the Central Limit Theorem.

    Even when the underlying population is not normally distributed, the
    distribution of sample means becomes approximately normal as sample
    size increases under common regularity conditions.
    """
    rng = random.Random(seed)
    sample_means = []

    for _ in range(repetitions):
        sample = [rng.choice(population) for _ in range(sample_size)]
        sample_means.append(mean(sample))

    theoretical_mean = mean(population)
    empirical_mean = mean(sample_means)
    empirical_sd = standard_deviation(sample_means)

    print("\n--- Central Limit Theorem ---")
    print("Population mean:", round(theoretical_mean, 4))
    print("Mean of simulated sample means:", round(empirical_mean, 4))
    print("Empirical SD of sample means:", round(empirical_sd, 4))
    print(
        "Theoretical standard error:",
        round(standard_deviation(population, sample=False) / math.sqrt(sample_size), 4),
    )


# ============================================================================
# 5. NORMAL DISTRIBUTION UTILITIES
# ============================================================================

def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Probability density of a normal distribution."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    coefficient = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * math.exp(exponent)


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal cumulative distribution function."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def normal_quantile(
    probability: float,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> float:
    """
    Numerically approximates a normal quantile.

    This avoids an external statistics package while providing enough
    precision for educational examples.
    """
    if not 0 < probability < 1:
        raise ValueError("Probability must be strictly between 0 and 1.")

    low = mu - 10 * sigma
    high = mu + 10 * sigma

    for _ in range(120):
        mid = (low + high) / 2
        if normal_cdf(mid, mu, sigma) < probability:
            low = mid
        else:
            high = mid

    return (low + high) / 2


# ============================================================================
# 6. STUDENT'S t DISTRIBUTION
# ============================================================================

def gamma_function(x: float) -> float:
    """Wrapper around the standard library gamma function."""
    return math.gamma(x)


def beta_function(a: float, b: float) -> float:
    """Beta function using the gamma identity."""
    return gamma_function(a) * gamma_function(b) / gamma_function(a + b)


def t_pdf(t: float, degrees_of_freedom: int) -> float:
    """
    Student's t probability density function.
    """
    df = degrees_of_freedom
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive.")

    numerator = gamma_function((df + 1) / 2)
    denominator = math.sqrt(df * math.pi) * gamma_function(df / 2)
    return numerator / denominator * (1 + t * t / df) ** (-(df + 1) / 2)


def integrate(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    steps: int = 10000,
) -> float:
    """Numerical integration using the trapezoidal rule."""
    if steps <= 0:
        raise ValueError("steps must be positive.")

    width = (upper - lower) / steps
    total = 0.5 * (function(lower) + function(upper))

    for i in range(1, steps):
        total += function(lower + i * width)

    return total * width


def t_cdf(t: float, degrees_of_freedom: int) -> float:
    """
    Numerical CDF of Student's t distribution.

    This is intentionally implemented directly so the statistical mechanism
    is visible instead of hiding it behind a library call.
    """
    if t == 0:
        return 0.5

    if t > 0:
        return 0.5 + integrate(
            lambda x: t_pdf(x, degrees_of_freedom),
            0,
            t,
            steps=5000,
        )

    return 1 - t_cdf(-t, degrees_of_freedom)


def t_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Numerically approximates a Student's t quantile."""
    if not 0 < probability < 1:
        raise ValueError("Probability must be between 0 and 1.")

    low = -20.0
    high = 20.0

    for _ in range(80):
        mid = (low + high) / 2
        if t_cdf(mid, degrees_of_freedom) < probability:
            low = mid
        else:
            high = mid

    return (low + high) / 2


# ============================================================================
# 7. CONFIDENCE INTERVALS
# ============================================================================

@dataclass
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str

    def __str__(self) -> str:
        return (
            f"{self.method}: estimate={self.estimate:.4f}, "
            f"interval=({self.lower:.4f}, {self.upper:.4f}), "
            f"confidence={self.confidence_level:.1%}"
        )


def mean_confidence_interval_z(
    sample: Sequence[float],
    confidence_level: float = 0.95,
    population_sd: float | None = None,
) -> ConfidenceInterval:
    """
    Confidence interval for a mean using a z critical value.

    If population_sd is known:
        x̄ ± z* σ / sqrt(n)

    This method is appropriate when the population standard deviation is
    genuinely known or when a normal approximation is justified under the
    intended model.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    n = len(sample)
    if n == 0:
        raise ValueError("Sample cannot be empty.")

    sigma = population_sd
    if sigma is None:
        sigma = standard_deviation(sample)

    alpha = 1 - confidence_level
    z_star = normal_quantile(1 - alpha / 2)

    margin = z_star * sigma / math.sqrt(n)
    estimate = mean(sample)

    return ConfidenceInterval(
        estimate,
        estimate - margin,
        estimate + margin,
        confidence_level,
        "z confidence interval for a mean",
    )


def mean_confidence_interval_t(
    sample: Sequence[float],
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """
    Confidence interval for a mean using Student's t.

        x̄ ± t* s / sqrt(n)

    This accounts for uncertainty in estimating the population standard
    deviation from the sample.
    """
    n = len(sample)
    if n < 2:
        raise ValueError("At least two observations are required.")

    alpha = 1 - confidence_level
    t_star = t_quantile(1 - alpha / 2, n - 1)

    estimate = mean(sample)
    margin = t_star * standard_error(sample)

    return ConfidenceInterval(
        estimate,
        estimate - margin,
        estimate + margin,
        confidence_level,
        "t confidence interval for a mean",
    )


def proportion_confidence_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """
    Wald confidence interval for a population proportion.

    p̂ ± z* sqrt[p̂(1-p̂)/n]

    The Wald interval is simple but can behave poorly for small samples or
    proportions close to 0 or 1. It is included to expose the formula.
    """
    if trials <= 0:
        raise ValueError("Trials must be positive.")
    if not 0 <= successes <= trials:
        raise ValueError("Successes must be between 0 and trials.")

    p_hat = successes / trials
    z_star = normal_quantile(1 - (1 - confidence_level) / 2)
    margin = z_star * math.sqrt(p_hat * (1 - p_hat) / trials)

    return ConfidenceInterval(
        p_hat,
        max(0.0, p_hat - margin),
        min(1.0, p_hat + margin),
        confidence_level,
        "Wald confidence interval for a proportion",
    )


# ============================================================================
# 8. HYPOTHESIS TESTING
# ============================================================================

@dataclass
class HypothesisTestResult:
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    alternative: str
    method: str
    degrees_of_freedom: int | None = None

    def interpretation(self) -> str:
        decision = (
            "Reject H0"
            if self.reject_null
            else "Do not reject H0"
        )

        return (
            f"{self.method}: statistic={self.statistic:.4f}, "
            f"p-value={self.p_value:.6f}, alpha={self.alpha:.3f}. "
            f"Decision: {decision}. "
            f"Alternative: {self.alternative}."
        )


def p_value_from_z(z: float, alternative: str = "two-sided") -> float:
    """Calculate a z-test p-value."""
    alternative = alternative.lower()

    if alternative == "two-sided":
        return 2 * (1 - normal_cdf(abs(z)))
    if alternative == "greater":
        return 1 - normal_cdf(z)
    if alternative == "less":
        return normal_cdf(z)

    raise ValueError("alternative must be two-sided, greater, or less.")


def one_sample_z_test(
    sample: Sequence[float],
    null_mean: float,
    population_sd: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """One-sample z test for a population mean."""
    n = len(sample)
    if n == 0:
        raise ValueError("Sample cannot be empty.")
    if population_sd <= 0:
        raise ValueError("Population SD must be positive.")

    z = (mean(sample) - null_mean) / (population_sd / math.sqrt(n))
    p_value = p_value_from_z(z, alternative)

    return HypothesisTestResult(
        statistic=z,
        p_value=p_value,
        alpha=alpha,
        reject_null=p_value < alpha,
        alternative=alternative,
        method="One-sample z test",
    )


def p_value_from_t(
    t_statistic: float,
    degrees_of_freedom: int,
    alternative: str = "two-sided",
) -> float:
    """Calculate a Student's t-test p-value."""
    cdf = t_cdf(t_statistic, degrees_of_freedom)
    alternative = alternative.lower()

    if alternative == "two-sided":
        return 2 * min(cdf, 1 - cdf)
    if alternative == "greater":
        return 1 - cdf
    if alternative == "less":
        return cdf

    raise ValueError("alternative must be two-sided, greater, or less.")


def one_sample_t_test(
    sample: Sequence[float],
    null_mean: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """
    One-sample t test.

    H0: μ = μ0
    HA: μ != μ0, μ > μ0, or μ < μ0
    """
    n = len(sample)
    if n < 2:
        raise ValueError("At least two observations are required.")

    sample_mean = mean(sample)
    se = standard_error(sample)

    if se == 0:
        if sample_mean == null_mean:
            t_statistic = 0.0
            p_value = 1.0
        else:
            t_statistic = math.inf if sample_mean > null_mean else -math.inf
            p_value = 0.0
    else:
        t_statistic = (sample_mean - null_mean) / se
        p_value = p_value_from_t(
            t_statistic,
            n - 1,
            alternative,
        )

    return HypothesisTestResult(
        statistic=t_statistic,
        p_value=p_value,
        alpha=alpha,
        reject_null=p_value < alpha,
        alternative=alternative,
        method="One-sample t test",
        degrees_of_freedom=n - 1,
    )


# ============================================================================
# 9. TWO-SAMPLE TESTS
# ============================================================================

def pooled_two_sample_t_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """
    Independent two-sample pooled t test.

    Assumes equal population variances.

    Pooled variance:
        sp² = [(n1-1)s1² + (n2-1)s2²] / (n1+n2-2)
    """
    n1, n2 = len(group_a), len(group_b)

    if n1 < 2 or n2 < 2:
        raise ValueError("Both groups require at least two observations.")

    s1_sq = variance(group_a)
    s2_sq = variance(group_b)

    pooled_variance = (
        ((n1 - 1) * s1_sq + (n2 - 1) * s2_sq)
        / (n1 + n2 - 2)
    )

    se = math.sqrt(pooled_variance * (1 / n1 + 1 / n2))
    t_statistic = (mean(group_a) - mean(group_b)) / se
    df = n1 + n2 - 2

    p_value = p_value_from_t(t_statistic, df, alternative)

    return HypothesisTestResult(
        t_statistic,
        p_value,
        alpha,
        p_value < alpha,
        alternative,
        "Independent pooled two-sample t test",
        df,
    )


def welch_two_sample_t_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """
    Welch's independent two-sample t test.

    Unlike the pooled test, Welch's test does not assume equal variances.
    """
    n1, n2 = len(group_a), len(group_b)

    if n1 < 2 or n2 < 2:
        raise ValueError("Both groups require at least two observations.")

    s1_sq = variance(group_a)
    s2_sq = variance(group_b)

    term1 = s1_sq / n1
    term2 = s2_sq / n2

    se = math.sqrt(term1 + term2)
    t_statistic = (mean(group_a) - mean(group_b)) / se

    numerator = (term1 + term2) ** 2
    denominator = (
        term1**2 / (n1 - 1)
        + term2**2 / (n2 - 1)
    )
    df = int(max(1, round(numerator / denominator)))

    p_value = p_value_from_t(t_statistic, df, alternative)

    return HypothesisTestResult(
        t_statistic,
        p_value,
        alpha,
        p_value < alpha,
        alternative,
        "Welch two-sample t test",
        df,
    )


def paired_t_test(
    before: Sequence[float],
    after: Sequence[float],
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """
    Paired t test.

    Converts paired observations into differences and performs a one-sample
    t test on those differences.
    """
    if len(before) != len(after):
        raise ValueError("Paired samples must have equal lengths.")

    differences = [after_value - before_value for before_value, after_value in zip(before, after)]

    result = one_sample_t_test(
        differences,
        null_mean=0.0,
        alternative=alternative,
        alpha=alpha,
    )

    return HypothesisTestResult(
        result.statistic,
        result.p_value,
        result.alpha,
        result.reject_null,
        result.alternative,
        "Paired t test",
        result.degrees_of_freedom,
    )


# ============================================================================
# 10. EFFECT SIZE
# ============================================================================

def cohens_d_independent(
    group_a: Sequence[float],
    group_b: Sequence[float],
) -> float:
    """
    Cohen's d using pooled standard deviation.

    Effect size and statistical significance answer different questions.
    A tiny effect can be statistically significant with a huge sample.
    A meaningful effect can fail to reach significance with a small sample.
    """
    n1, n2 = len(group_a), len(group_b)

    pooled_sd = math.sqrt(
        (
            (n1 - 1) * variance(group_a)
            + (n2 - 1) * variance(group_b)
        )
        / (n1 + n2 - 2)
    )

    if pooled_sd == 0:
        return 0.0

    return (mean(group_a) - mean(group_b)) / pooled_sd


# ============================================================================
# 11. PROPORTION HYPOTHESIS TEST
# ============================================================================

def one_proportion_z_test(
    successes: int,
    trials: int,
    null_proportion: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """
    One-proportion z test.

    Under H0:
        SE = sqrt[p0(1-p0)/n]

    The null proportion, not the observed proportion, is used for the
    standard error because the test distribution is constructed under H0.
    """
    if trials <= 0:
        raise ValueError("Trials must be positive.")
    if not 0 < null_proportion < 1:
        raise ValueError("Null proportion must be strictly between 0 and 1.")

    observed = successes / trials
    se = math.sqrt(null_proportion * (1 - null_proportion) / trials)
    z = (observed - null_proportion) / se
    p_value = p_value_from_z(z, alternative)

    return HypothesisTestResult(
        z,
        p_value,
        alpha,
        p_value < alpha,
        alternative,
        "One-proportion z test",
    )


# ============================================================================
# 12. CHI-SQUARE TEST OF INDEPENDENCE
# ============================================================================

def chi_square_statistic(observed: Sequence[Sequence[int]]) -> float:
    """
    Pearson chi-square statistic.

    X² = Σ (O-E)²/E
    """
    rows = len(observed)
    cols = len(observed[0])

    if rows < 2 or cols < 2:
        raise ValueError("At least a 2x2 table is required.")

    if any(len(row) != cols for row in observed):
        raise ValueError("All rows must have the same number of columns.")

    row_totals = [sum(row) for row in observed]
    column_totals = [
        sum(observed[row][column] for row in range(rows))
        for column in range(cols)
    ]
    total = sum(row_totals)

    statistic = 0.0

    for row in range(rows):
        for column in range(cols):
            expected = row_totals[row] * column_totals[column] / total
            if expected == 0:
                raise ValueError("Expected frequency cannot be zero.")
            statistic += (
                (observed[row][column] - expected) ** 2
                / expected
            )

    return statistic


def chi_square_p_value_2x2(
    observed: Sequence[Sequence[int]],
) -> float:
    """
    Exact chi-square survival probability for df=1.

    For one degree of freedom:
        P(X² >= x) = erfc(sqrt(x/2))
    """
    statistic = chi_square_statistic(observed)
    return math.erfc(math.sqrt(statistic / 2))


# ============================================================================
# 13. CORRELATION
# ============================================================================

def pearson_correlation(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> float:
    """Pearson product-moment correlation coefficient."""
    if len(x_values) != len(y_values):
        raise ValueError("Sequences must have equal length.")
    if len(x_values) < 2:
        raise ValueError("At least two observations are required.")

    x_mean = mean(x_values)
    y_mean = mean(y_values)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    )

    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in x_values)
        * sum((y - y_mean) ** 2 for y in y_values)
    )

    if denominator == 0:
        raise ValueError("Correlation is undefined for a constant variable.")

    return numerator / denominator


# ============================================================================
# 14. BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================================

def bootstrap_confidence_interval(
    sample: Sequence[float],
    statistic_function: Callable[[Sequence[float]], float],
    confidence_level: float = 0.95,
    repetitions: int = 10000,
    seed: int = 42,
) -> ConfidenceInterval:
    """
    Percentile bootstrap confidence interval.

    The empirical sampling distribution is approximated by repeatedly
    sampling with replacement from the observed sample.
    """
    if not sample:
        raise ValueError("Sample cannot be empty.")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    rng = random.Random(seed)
    bootstrap_statistics = []

    for _ in range(repetitions):
        resample = [
            rng.choice(sample)
            for _ in range(len(sample))
        ]
        bootstrap_statistics.append(statistic_function(resample))

    bootstrap_statistics.sort()

    alpha = 1 - confidence_level

    lower_index = int((alpha / 2) * repetitions)
    upper_index = int((1 - alpha / 2) * repetitions) - 1

    lower_index = max(0, min(lower_index, repetitions - 1))
    upper_index = max(0, min(upper_index, repetitions - 1))

    estimate = statistic_function(sample)

    return ConfidenceInterval(
        estimate,
        bootstrap_statistics[lower_index],
        bootstrap_statistics[upper_index],
        confidence_level,
        "Percentile bootstrap confidence interval",
    )


# ============================================================================
# 15. PERMUTATION TEST
# ============================================================================

def permutation_test_difference_in_means(
    group_a: Sequence[float],
    group_b: Sequence[float],
    repetitions: int = 10000,
    seed: int = 42,
) -> float:
    """
    Randomization/permutation test for difference in means.

    Under the null hypothesis that group membership does not affect the
    outcome, labels can be randomly reassigned.
    """
    rng = random.Random(seed)

    observed_difference = mean(group_a) - mean(group_b)
    pooled = list(group_a) + list(group_b)

    extreme_count = 0

    for _ in range(repetitions):
        shuffled = pooled.copy()
        rng.shuffle(shuffled)

        perm_a = shuffled[:len(group_a)]
        perm_b = shuffled[len(group_a):]

        difference = mean(perm_a) - mean(perm_b)

        if abs(difference) >= abs(observed_difference):
            extreme_count += 1

    # +1 correction avoids a zero estimated p-value.
    return (extreme_count + 1) / (repetitions + 1)


# ============================================================================
# 16. TYPE I ERROR, TYPE II ERROR, AND POWER
# ============================================================================

def simulate_type_i_error(
    null_mean: float = 100,
    population_sd: float = 15,
    sample_size: int = 25,
    alpha: float = 0.05,
    repetitions: int = 5000,
    seed: int = 42,
) -> float:
    """
    Under H0, the rejection rate estimates the Type I error probability.

    Type I error:
        Rejecting a true null hypothesis.
    """
    rng = random.Random(seed)
    rejections = 0

    critical_z = normal_quantile(1 - alpha / 2)

    for _ in range(repetitions):
        sample = [
            rng.gauss(null_mean, population_sd)
            for _ in range(sample_size)
        ]

        z = (
            mean(sample) - null_mean
        ) / (population_sd / math.sqrt(sample_size))

        if abs(z) > critical_z:
            rejections += 1

    return rejections / repetitions


def simulate_power(
    true_mean: float,
    null_mean: float,
    population_sd: float,
    sample_size: int,
    alpha: float = 0.05,
    repetitions: int = 5000,
    seed: int = 42,
) -> float:
    """
    Estimates power through simulation.

    Power = P(reject H0 | H1 is true)
         = 1 - beta

    Increasing sample size, effect size, or alpha generally increases power,
    while greater noise generally decreases power.
    """
    rng = random.Random(seed)
    rejections = 0

    critical_z = normal_quantile(1 - alpha / 2)

    for _ in range(repetitions):
        sample = [
            rng.gauss(true_mean, population_sd)
            for _ in range(sample_size)
        ]

        z = (
            mean(sample) - null_mean
        ) / (population_sd / math.sqrt(sample_size))

        if abs(z) > critical_z:
            rejections += 1

    return rejections / repetitions


# ============================================================================
# 17. MULTIPLE TESTING AND BONFERRONI CORRECTION
# ============================================================================

def bonferroni_adjustment(
    p_values: Sequence[float],
    familywise_alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """
    Bonferroni correction.

    Reject H0_i when:
        p_i < alpha / m

    Equivalent adjusted p-value:
        min(m * p_i, 1)
    """
    m = len(p_values)
    if m == 0:
        return []

    adjusted = []

    for p_value in p_values:
        adjusted_p = min(1.0, p_value * m)
        adjusted.append(
            (adjusted_p, adjusted_p < familywise_alpha)
        )

    return adjusted


# ============================================================================
# 18. SAMPLE SIZE FOR A MEAN
# ============================================================================

def approximate_sample_size_for_mean(
    population_sd: float,
    margin_of_error: float,
    confidence_level: float = 0.95,
) -> int:
    """
    Approximate n for a mean when population SD is known/estimated.

        n = (z* sigma / E)^2

    The result is rounded upward because sample size must be an integer.
    """
    if population_sd <= 0 or margin_of_error <= 0:
        raise ValueError("Standard deviation and margin must be positive.")

    z_star = normal_quantile(1 - (1 - confidence_level) / 2)

    n = (z_star * population_sd / margin_of_error) ** 2
    return math.ceil(n)


# ============================================================================
# 19. SAMPLE SIZE FOR A PROPORTION
# ============================================================================

def approximate_sample_size_for_proportion(
    anticipated_proportion: float = 0.5,
    margin_of_error: float = 0.05,
    confidence_level: float = 0.95,
) -> int:
    """
    Approximate sample size for a population proportion.

        n = z² p(1-p) / E²

    p=0.5 maximizes p(1-p), producing the conservative sample size when
    no prior estimate is available.
    """
    if not 0 < anticipated_proportion < 1:
        raise ValueError("Proportion must be strictly between 0 and 1.")
    if margin_of_error <= 0:
        raise ValueError("Margin of error must be positive.")

    z_star = normal_quantile(1 - (1 - confidence_level) / 2)

    n = (
        z_star**2
        * anticipated_proportion
        * (1 - anticipated_proportion)
        / margin_of_error**2
    )

    return math.ceil(n)


# ============================================================================
# 20. PRACTICAL END-TO-END EXAMPLE
# ============================================================================

def practical_product_quality_analysis() -> None:
    """
    A realistic scenario:

    A manufacturer historically reports an average processing time of
    50 minutes. A process improvement is introduced. A random sample is
    collected to determine whether the average processing time has changed.

    The analysis demonstrates:
        - sampling
        - descriptive statistics
        - confidence interval
        - hypothesis testing
        - p-value interpretation
        - effect size
    """
    processing_times = [
        46, 48, 51, 47, 45,
        49, 52, 50, 44, 47,
        46, 48, 45, 49, 47,
        43, 51, 46, 48, 44,
        45, 47, 46, 49, 45,
    ]

    print("\n--- Practical Product Quality Analysis ---")

    describe_sample(processing_times)

    interval = mean_confidence_interval_t(processing_times, 0.95)
    print(interval)

    test = one_sample_t_test(
        processing_times,
        null_mean=50,
        alternative="less",
        alpha=0.05,
    )

    print(test.interpretation())

    print(
        "Effect size relative to 50-minute benchmark:",
        round(
            (mean(processing_times) - 50)
            / standard_deviation(processing_times),
            4,
        ),
    )

    print(
        "Important interpretation:",
        "A p-value below 0.05 would provide evidence against H0 under the"
        " specified model; it would not prove that H0 is impossible.",
    )


# ============================================================================
# 21. COMMON MISINTERPRETATIONS
# ============================================================================

def demonstrate_interpretation_rules() -> None:
    print("\n--- Interpretation Rules ---")

    rules = [
        "A p-value is not the probability that H0 is true.",
        "A 95% confidence interval does not mean there is a 95% probability"
        " that a fixed parameter lies inside this particular interval.",
        "Statistical significance does not automatically imply practical importance.",
        "Failure to reject H0 is not proof that H0 is true.",
        "The null hypothesis is evaluated under an assumed statistical model.",
        "Sampling bias cannot generally be repaired by increasing sample size.",
        "Random sampling and random assignment solve different problems.",
        "A statistically significant correlation does not establish causation.",
        "Multiple testing can inflate the chance of false positives.",
    ]

    for index, rule in enumerate(rules, start=1):
        print(f"{index}. {rule}")


# ============================================================================
# 22. EDGE CASES
# ============================================================================

def demonstrate_edge_cases() -> None:
    print("\n--- Edge Cases ---")

    examples = {
        "Single observation mean": [100],
        "Repeated observations": [5, 5, 5, 5, 5],
        "Small sample": [8, 10, 11],
        "Highly variable sample": [1, 100, 2, 95, 4, 90],
    }

    for name, values in examples.items():
        print(f"\n{name}: {values}")

        try:
            print("Mean:", mean(values))
        except ValueError as error:
            print("Mean error:", error)

        try:
            print("SD:", standard_deviation(values))
        except ValueError as error:
            print("SD error:", error)

        if len(values) >= 2:
            print("t interval:", mean_confidence_interval_t(values))


# ============================================================================
# 23. ASSUMPTIONS AND MODEL CHECKING
# ============================================================================

def discuss_assumptions() -> None:
    print("\n--- Statistical Assumptions ---")

    assumptions = [
        (
            "Independence",
            "Observations should not be dependent in ways ignored by the model."
        ),
        (
            "Random sampling",
            "A probability sample supports generalization to the target population."
        ),
        (
            "Random assignment",
            "Random assignment helps support causal comparisons in experiments."
        ),
        (
            "Normality",
            "Some small-sample procedures rely on approximately normal data or residuals."
        ),
        (
            "Equal variance",
            "The pooled two-sample t test assumes equal variances; Welch's test does not."
        ),
        (
            "Correct model",
            "A mathematically correct calculation is not enough if the statistical model is inappropriate."
        ),
    ]

    for assumption, explanation in assumptions:
        print(f"{assumption}: {explanation}")


# ============================================================================
# 24. REPRODUCIBILITY
# ============================================================================

def reproducibility_demo() -> None:
    """
    Random seeds make simulation results reproducible.

    Reproducibility is important for debugging, teaching, testing, and
    scientific workflows.
    """
    first = simple_random_sample(list(range(100)), 10, seed=123)
    second = simple_random_sample(list(range(100)), 10, seed=123)

    print("\n--- Reproducibility ---")
    print("First:", first)
    print("Second:", second)
    print("Identical:", first == second)


# ============================================================================
# 25. MAIN EDUCATIONAL WORKFLOW
# ============================================================================

def main() -> None:
    print("=" * 78)
    print("STATISTICAL INFERENCE: COMPREHENSIVE PYTHON STUDY")
    print("=" * 78)

    random.seed(42)

    demonstrate_population_and_sample()

    print("\n--- Sampling Methods ---")
    population = list(range(1, 101))
    print(
        "Simple random sample:",
        simple_random_sample(population, 10),
    )
    print(
        "Systematic sample:",
        systematic_sample(population, 10),
    )

    stratified_population = [
        ("A", 10), ("A", 12), ("A", 14), ("A", 16),
        ("B", 20), ("B", 22), ("B", 24), ("B", 26),
    ]
    print(
        "Stratified sample:",
        stratified_sample(stratified_population, 2),
    )

    clusters = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
    print(
        "Cluster sample:",
        cluster_sample(clusters, 4, 2),
    )

    skewed_population = [1] * 80 + [10] * 15 + [100] * 5
    demonstrate_central_limit_theorem(
        skewed_population,
        sample_size=30,
        repetitions=3000,
    )

    print("\n--- Confidence Intervals ---")
    sample = [18, 21, 19, 23, 20, 22, 24, 17, 21, 19]

    print(mean_confidence_interval_t(sample, 0.95))
    print(mean_confidence_interval_z(sample, 0.95, population_sd=3.0))
    print(proportion_confidence_interval(62, 100, 0.95))

    print("\n--- Hypothesis Tests ---")
    print(
        one_sample_z_test(
            sample,
            null_mean=20,
            population_sd=3,
            alternative="two-sided",
        ).interpretation()
    )

    print(
        one_sample_t_test(
            sample,
            null_mean=20,
            alternative="two-sided",
        ).interpretation()
    )

    group_a = [72, 75, 71, 74, 78, 76, 73, 77]
    group_b = [68, 70, 69, 71, 67, 72, 70, 68]

    print(
        welch_two_sample_t_test(
            group_a,
            group_b,
        ).interpretation()
    )

    before = [80, 82, 78, 85, 90, 87, 83, 81]
    after = [76, 79, 75, 81, 85, 84, 80, 78]

    print(
        paired_t_test(
            before,
            after,
            alternative="two-sided",
        ).interpretation()
    )

    print("\n--- Effect Size ---")
    print("Cohen's d:", round(cohens_d_independent(group_a, group_b), 4))

    print("\n--- Proportion Test ---")
    print(
        one_proportion_z_test(
            successes=56,
            trials=100,
            null_proportion=0.50,
        ).interpretation()
    )

    print("\n--- Chi-Square Test ---")
    contingency_table = [
        [30, 20],
        [15, 35],
    ]
    chi_stat = chi_square_statistic(contingency_table)
    chi_p = chi_square_p_value_2x2(contingency_table)
    print("Chi-square statistic:", round(chi_stat, 4))
    print("p-value:", round(chi_p, 6))

    print("\n--- Correlation ---")
    study_hours = [1, 2, 3, 4, 5, 6, 7]
    exam_scores = [55, 59, 63, 67, 70, 76, 81]
    print(
        "Pearson correlation:",
        round(pearson_correlation(study_hours, exam_scores), 4),
    )

    print("\n--- Bootstrap ---")
    bootstrap_sample = [11, 12, 13, 13, 14, 15, 18, 19, 20]
    print(
        bootstrap_confidence_interval(
            bootstrap_sample,
            mean,
            confidence_level=0.95,
            repetitions=3000,
        )
    )

    print("\n--- Permutation Test ---")
    print(
        "Permutation p-value:",
        round(
            permutation_test_difference_in_means(
                group_a,
                group_b,
                repetitions=3000,
            ),
            6,
        ),
    )

    print("\n--- Type I Error Simulation ---")
    print(
        "Estimated rejection rate under H0:",
        round(
            simulate_type_i_error(
                repetitions=2000,
            ),
            4,
        ),
    )

    print("\n--- Power Simulation ---")
    print(
        "Estimated power:",
        round(
            simulate_power(
                true_mean=103,
                null_mean=100,
                population_sd=15,
                sample_size=100,
                repetitions=2000,
            ),
            4,
        ),
    )

    print("\n--- Multiple Testing ---")
    p_values = [0.001, 0.012, 0.021, 0.04, 0.20]
    print("Raw p-values:", p_values)
    print(
        "Bonferroni adjusted:",
        bonferroni_adjustment(p_values),
    )

    print("\n--- Sample Size ---")
    print(
        "Mean sample size:",
        approximate_sample_size_for_mean(
            population_sd=15,
            margin_of_error=2,
        ),
    )
    print(
        "Proportion sample size:",
        approximate_sample_size_for_proportion(
            anticipated_proportion=0.5,
            margin_of_error=0.05,
        ),
    )

    practical_product_quality_analysis()
    demonstrate_interpretation_rules()
    demonstrate_edge_cases()
    discuss_assumptions()
    reproducibility_demo()

    print("\n" + "=" * 78)
    print("END OF STATISTICAL INFERENCE STUDY")
    print("=" * 78)


if __name__ == "__main__":
    main()
