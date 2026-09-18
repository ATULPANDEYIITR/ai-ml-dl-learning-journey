"""
Probability Distributions
=========================

Bernoulli, Binomial, Poisson, Gaussian, Uniform, and Exponential distributions.

This standalone study script moves from elementary probability concepts to
practical statistical modeling, simulation, estimation, numerical methods,
validation, edge cases, and production-oriented considerations.

No third-party packages are required.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


# ============================================================================
# 1. FOUNDATIONS
# ============================================================================

print("=" * 78)
print("PROBABILITY DISTRIBUTIONS")
print("=" * 78)


def section(title: str) -> None:
    print(f"\n{'-' * 78}\n{title}\n{'-' * 78}")


section("1. Probability fundamentals")

print(
    """
A probability is a number between 0 and 1.

    0   -> impossible event
    1   -> certain event
    0.5 -> event with a 50% probability

A random variable assigns numerical values to outcomes.

A discrete random variable takes countable values, such as:
    number of defective products = 0, 1, 2, ...

A continuous random variable can take values on intervals, such as:
    response time = 0.0, 0.01, 0.015, ...

A probability mass function (PMF) gives probabilities to discrete values.

A probability density function (PDF) describes the density of a continuous
random variable. For a continuous variable, probability is obtained from
area under the density over an interval.

A cumulative distribution function (CDF) is:

    F(x) = P(X <= x)

The expectation is the probability-weighted average:

    E[X] = sum(x * P(X=x))

for discrete variables, or the corresponding integral for continuous
variables.

Variance measures squared dispersion around the mean:

    Var(X) = E[(X - E[X])^2]

Standard deviation is:

    SD(X) = sqrt(Var(X))
"""
)


def mean_from_distribution(
    values: Sequence[float], probabilities: Sequence[float]
) -> float:
    if len(values) != len(probabilities):
        raise ValueError("Values and probabilities must have equal lengths.")
    if not probabilities:
        raise ValueError("Distribution cannot be empty.")
    if any(p < 0 for p in probabilities):
        raise ValueError("Probabilities cannot be negative.")
    total = sum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Probabilities must sum to 1.")
    return sum(x * p for x, p in zip(values, probabilities))


def variance_from_distribution(
    values: Sequence[float], probabilities: Sequence[float]
) -> float:
    mu = mean_from_distribution(values, probabilities)
    return sum(p * (x - mu) ** 2 for x, p in zip(values, probabilities))


values = [0, 1, 2]
probabilities = [0.25, 0.50, 0.25]
print("Example discrete mean:", mean_from_distribution(values, probabilities))
print("Example discrete variance:", variance_from_distribution(values, probabilities))


# ============================================================================
# 2. GENERAL NUMERICAL UTILITIES
# ============================================================================

section("2. Numerical utilities and validation")


def validate_probability(p: float, name: str = "p") -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def factorial(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer.")
    if n < 0:
        raise ValueError("Factorial is undefined for negative integers.")
    return math.factorial(n)


def combination(n: int, k: int) -> int:
    if not isinstance(n, int) or not isinstance(k, int):
        raise TypeError("n and k must be integers.")
    if n < 0 or k < 0 or k > n:
        return 0
    return math.comb(n, k)


print("5! =", factorial(5))
print("10 choose 3 =", combination(10, 3))


# ============================================================================
# 3. BERNOULLI DISTRIBUTION
# ============================================================================

section("3. Bernoulli distribution")

print(
    """
Bernoulli distribution models exactly one trial with two outcomes.

Let X ~ Bernoulli(p).

Typically:
    X = 1 for success
    X = 0 for failure

PMF:
    P(X=x) = p^x (1-p)^(1-x), x in {0, 1}

Mean:
    E[X] = p

Variance:
    Var(X) = p(1-p)

The complementary probability is q = 1-p.

Examples:
    pass/fail
    defective/non-defective
    clicked/not clicked
    transaction succeeded/failed
"""
)


@dataclass(frozen=True)
class BernoulliDistribution:
    p: float

    def __post_init__(self) -> None:
        validate_probability(self.p)

    @property
    def q(self) -> float:
        return 1.0 - self.p

    def pmf(self, x: int) -> float:
        if x == 1:
            return self.p
        if x == 0:
            return self.q
        return 0.0

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        if x < 1:
            return self.q
        return 1.0

    @property
    def mean(self) -> float:
        return self.p

    @property
    def variance(self) -> float:
        return self.p * self.q

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(self.variance)

    def sample(self, rng: random.Random | None = None) -> int:
        generator = rng or random
        return int(generator.random() < self.p)


coin = BernoulliDistribution(0.5)
print("Fair coin P(X=1):", coin.pmf(1))
print("Fair coin P(X=0):", coin.pmf(0))
print("Fair coin mean:", coin.mean)
print("Fair coin variance:", coin.variance)

biased_coin = BernoulliDistribution(0.7)
rng = random.Random(42)
print("Ten simulated biased-coin trials:", [biased_coin.sample(rng) for _ in range(10)])


# ============================================================================
# 4. BINOMIAL DISTRIBUTION
# ============================================================================

section("4. Binomial distribution")

print(
    """
Binomial distribution counts successes in n independent Bernoulli trials.

Let X ~ Binomial(n, p).

PMF:
    P(X=k) = C(n,k) p^k (1-p)^(n-k)

where:
    k = 0, 1, ..., n

Mean:
    E[X] = np

Variance:
    Var(X) = np(1-p)

Standard deviation:
    sqrt(np(1-p))

Conditions:
    1. Fixed number of trials
    2. Exactly two outcomes per trial
    3. Same success probability p
    4. Trials are independent

Binomial is not appropriate when the probability changes between trials or
when the trials are meaningfully dependent.
"""
)


@dataclass(frozen=True)
class BinomialDistribution:
    n: int
    p: float

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool):
            raise TypeError("n must be an integer.")
        if self.n < 0:
            raise ValueError("n cannot be negative.")
        validate_probability(self.p)

    def pmf(self, k: int) -> float:
        if not isinstance(k, int):
            return 0.0
        if k < 0 or k > self.n:
            return 0.0
        if self.p == 0.0:
            return 1.0 if k == 0 else 0.0
        if self.p == 1.0:
            return 1.0 if k == self.n else 0.0

        # Using logs helps avoid intermediate overflow for large n.
        log_probability = (
            math.lgamma(self.n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(self.n - k + 1)
            + k * math.log(self.p)
            + (self.n - k) * math.log1p(-self.p)
        )
        return math.exp(log_probability)

    def cdf(self, k: int) -> float:
        if k < 0:
            return 0.0
        if k >= self.n:
            return 1.0
        return sum(self.pmf(i) for i in range(k + 1))

    def probability_at_least(self, k: int) -> float:
        if k <= 0:
            return 1.0
        if k > self.n:
            return 0.0
        return sum(self.pmf(i) for i in range(k, self.n + 1))

    @property
    def mean(self) -> float:
        return self.n * self.p

    @property
    def variance(self) -> float:
        return self.n * self.p * (1.0 - self.p)

    def sample(self, rng: random.Random | None = None) -> int:
        generator = rng or random
        return sum(generator.random() < self.p for _ in range(self.n))


binomial = BinomialDistribution(n=10, p=0.3)
print("Binomial P(X=3):", binomial.pmf(3))
print("Binomial P(X<=3):", binomial.cdf(3))
print("Binomial P(X>=7):", binomial.probability_at_least(7))
print("Binomial mean:", binomial.mean)
print("Binomial variance:", binomial.variance)
print("Binomial sample:", binomial.sample(random.Random(7)))


# ============================================================================
# 5. POISSON DISTRIBUTION
# ============================================================================

section("5. Poisson distribution")

print(
    """
Poisson distribution models counts of events in a fixed interval when the
events occur according to a Poisson process.

Let X ~ Poisson(lambda).

PMF:
    P(X=k) = exp(-lambda) lambda^k / k!

for k = 0, 1, 2, ...

Mean:
    E[X] = lambda

Variance:
    Var(X) = lambda

Typical applications:
    requests arriving at a server per minute
    calls received by a call center per hour
    defects per meter
    accidents at a location per month

The Poisson model is particularly useful when events are approximately
independent and the rate is reasonably stable over the modeled interval.

The parameter lambda is both the expected count and the variance.
"""
)


@dataclass(frozen=True)
class PoissonDistribution:
    rate: float

    def __post_init__(self) -> None:
        if self.rate < 0 or not math.isfinite(self.rate):
            raise ValueError("Poisson rate must be finite and non-negative.")

    def pmf(self, k: int) -> float:
        if not isinstance(k, int) or k < 0:
            return 0.0
        if self.rate == 0:
            return 1.0 if k == 0 else 0.0
        log_probability = -self.rate + k * math.log(self.rate) - math.lgamma(k + 1)
        return math.exp(log_probability)

    def cdf(self, k: int) -> float:
        if k < 0:
            return 0.0
        return sum(self.pmf(i) for i in range(k + 1))

    @property
    def mean(self) -> float:
        return self.rate

    @property
    def variance(self) -> float:
        return self.rate

    def sample(self, rng: random.Random | None = None) -> int:
        """
        Knuth's exact algorithm for moderate rate values.

        For very large rates, production systems normally use a more
        sophisticated Poisson sampler to avoid excessive looping.
        """
        generator = rng or random
        if self.rate == 0:
            return 0

        # Knuth's algorithm becomes inefficient when lambda is very large.
        if self.rate > 30:
            # Normal approximation with a correction is useful for this
            # educational implementation, but is not an exact sampler.
            value = generator.gauss(self.rate, math.sqrt(self.rate))
            return max(0, int(round(value)))

        threshold = math.exp(-self.rate)
        product = 1.0
        count = 0

        while product > threshold:
            count += 1
            product *= generator.random()

        return count - 1


poisson = PoissonDistribution(4.0)
print("Poisson P(X=2):", poisson.pmf(2))
print("Poisson P(X<=2):", poisson.cdf(2))
print("Poisson mean:", poisson.mean)
print("Poisson variance:", poisson.variance)
print("Poisson sample:", poisson.sample(random.Random(11)))


# ============================================================================
# 6. POISSON PROCESS AND EXPONENTIAL INTERARRIVAL TIMES
# ============================================================================

section("6. Relationship between Poisson and Exponential distributions")

print(
    """
A useful distinction:

Poisson distribution:
    counts HOW MANY events occur during an interval.

Exponential distribution:
    models HOW LONG until the next event when the event process has a
    constant rate.

For a Poisson process with rate lambda:

    N(t) ~ Poisson(lambda * t)

and the waiting time T until the next event satisfies:

    T ~ Exponential(lambda)

This relationship is central to queueing, reliability, networking, and
event-arrival models.
"""
)


# ============================================================================
# 7. EXPONENTIAL DISTRIBUTION
# ============================================================================

section("7. Exponential distribution")


@dataclass(frozen=True)
class ExponentialDistribution:
    rate: float

    def __post_init__(self) -> None:
        validate_positive(self.rate, "rate")

    def pdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return self.rate * math.exp(-self.rate * x)

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        # expm1 gives better numerical behavior when rate*x is small.
        return -math.expm1(-self.rate * x)

    def survival(self, x: float) -> float:
        if x < 0:
            return 1.0
        return math.exp(-self.rate * x)

    @property
    def mean(self) -> float:
        return 1.0 / self.rate

    @property
    def variance(self) -> float:
        return 1.0 / (self.rate ** 2)

    @property
    def standard_deviation(self) -> float:
        return 1.0 / self.rate

    def quantile(self, probability: float) -> float:
        validate_probability(probability, "probability")
        if probability == 1.0:
            return math.inf
        if probability == 0.0:
            return 0.0
        return -math.log1p(-probability) / self.rate

    def sample(self, rng: random.Random | None = None) -> float:
        generator = rng or random
        # random() can theoretically return 0, which is safe, but using
        # 1-U ensures the logarithm never receives zero.
        u = generator.random()
        return -math.log1p(-u) / self.rate


exponential = ExponentialDistribution(rate=2.0)
print("Exponential PDF at x=0.5:", exponential.pdf(0.5))
print("Exponential P(T<=0.5):", exponential.cdf(0.5))
print("Exponential P(T>0.5):", exponential.survival(0.5))
print("Exponential mean:", exponential.mean)
print("Exponential median:", exponential.quantile(0.5))
print("Exponential sample:", exponential.sample(random.Random(21)))


# ============================================================================
# 8. MEMORYLESS PROPERTY
# ============================================================================

section("8. Exponential memorylessness")

print(
    """
The exponential distribution has the memoryless property:

    P(T > s+t | T > s) = P(T > t)

The age already spent waiting does not change the distribution of the
remaining waiting time.

For rate lambda:

    P(T > t) = exp(-lambda*t)

This property is special. Many lifetime distributions do not have it.

A common mistake is to assume that every real-world waiting time is
exponential. Real systems may have time-dependent rates, scheduled events,
bursty arrivals, mixtures, or dependencies.
"""
)

rate = 0.25
s = 3.0
t = 2.0
memoryless_left = math.exp(-rate * (s + t)) / math.exp(-rate * s)
memoryless_right = math.exp(-rate * t)
print("Conditional survival:", memoryless_left)
print("Direct survival:", memoryless_right)


# ============================================================================
# 9. UNIFORM DISTRIBUTION
# ============================================================================

section("9. Uniform distribution")

print(
    """
Continuous Uniform(a, b) assigns equal density to every point in [a, b].

PDF:
    f(x) = 1/(b-a), a <= x <= b

CDF:
    0                  x < a
    (x-a)/(b-a)        a <= x <= b
    1                  x > b

Mean:
    (a+b)/2

Variance:
    (b-a)^2 / 12

Uniform distributions are useful for:
    simulation
    randomization
    uncertainty models with equal density
    generating random values

Python's random.uniform is also useful for simulation, but statistical
modeling should distinguish the underlying model from the random-number
generator used to simulate it.
"""
)


@dataclass(frozen=True)
class UniformDistribution:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("Bounds must be finite.")
        if self.upper <= self.lower:
            raise ValueError("upper must be greater than lower.")

    def pdf(self, x: float) -> float:
        if self.lower <= x <= self.upper:
            return 1.0 / (self.upper - self.lower)
        return 0.0

    def cdf(self, x: float) -> float:
        if x < self.lower:
            return 0.0
        if x >= self.upper:
            return 1.0
        return (x - self.lower) / (self.upper - self.lower)

    @property
    def mean(self) -> float:
        return (self.lower + self.upper) / 2.0

    @property
    def variance(self) -> float:
        return (self.upper - self.lower) ** 2 / 12.0

    def sample(self, rng: random.Random | None = None) -> float:
        generator = rng or random
        return generator.uniform(self.lower, self.upper)


uniform = UniformDistribution(10.0, 20.0)
print("Uniform PDF at 15:", uniform.pdf(15))
print("Uniform P(X<=15):", uniform.cdf(15))
print("Uniform mean:", uniform.mean)
print("Uniform variance:", uniform.variance)
print("Uniform samples:", [uniform.sample(random.Random(i)) for i in range(5)])


# ============================================================================
# 10. GAUSSIAN / NORMAL DISTRIBUTION
# ============================================================================

section("10. Gaussian (Normal) distribution")

print(
    """
The Gaussian distribution is also called the Normal distribution.

Let X ~ Normal(mu, sigma^2).

PDF:

             -(x-mu)^2
             ----------
               2sigma²
f(x) = -------------------------- exp(...)
             sigma * sqrt(2pi)

Parameters:
    mu    = mean/location
    sigma = standard deviation/scale, sigma > 0

Mean:
    mu

Variance:
    sigma²

Standardization converts X into a standard normal variable:

    Z = (X-mu)/sigma

The standard normal distribution has:
    mean = 0
    standard deviation = 1

The Normal distribution is central to statistics because sums and averages
of many independent contributions often become approximately normal under
appropriate conditions, as described by the Central Limit Theorem.

The Normal distribution is not appropriate for every dataset. Strong
skewness, hard lower bounds, heavy tails, or discrete counts can require
different models.
"""
)


@dataclass(frozen=True)
class GaussianDistribution:
    mean_value: float
    standard_deviation: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean_value):
            raise ValueError("Mean must be finite.")
        validate_positive(self.standard_deviation, "standard_deviation")

    @property
    def variance(self) -> float:
        return self.standard_deviation ** 2

    def pdf(self, x: float) -> float:
        z = (x - self.mean_value) / self.standard_deviation
        return (
            math.exp(-0.5 * z * z)
            / (self.standard_deviation * math.sqrt(2.0 * math.pi))
        )

    def cdf(self, x: float) -> float:
        z = (x - self.mean_value) / (
            self.standard_deviation * math.sqrt(2.0)
        )
        return 0.5 * (1.0 + math.erf(z))

    def z_score(self, x: float) -> float:
        return (x - self.mean_value) / self.standard_deviation

    def quantile_approx(self, probability: float) -> float:
        """
        Inverse normal CDF using Peter John Acklam's rational approximation.

        This avoids requiring SciPy while providing useful numerical accuracy
        for educational and simulation purposes.
        """
        validate_probability(probability, "probability")

        if probability == 0:
            return -math.inf
        if probability == 1:
            return math.inf

        a = [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
        b = [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]

        low = 0.02425
        high = 1.0 - low

        if probability < low:
            q = math.sqrt(-2.0 * math.log(probability))
            x = (
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
            )
        elif probability <= high:
            q = probability - 0.5
            r = q * q
            x = (
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
            )
        else:
            q = math.sqrt(-2.0 * math.log(1.0 - probability))
            x = -(
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
            )

        return self.mean_value + self.standard_deviation * x

    def sample(self, rng: random.Random | None = None) -> float:
        generator = rng or random
        return generator.gauss(self.mean_value, self.standard_deviation)


normal = GaussianDistribution(100.0, 15.0)
for score in [70, 85, 100, 115, 130]:
    print(
        f"Score={score:3d}, z={normal.z_score(score):6.2f}, "
        f"CDF={normal.cdf(score):.6f}"
    )

print("Normal PDF at mean:", normal.pdf(100))
print("Normal 95th percentile approximation:", normal.quantile_approx(0.95))
print("Normal samples:", [round(normal.sample(random.Random(i)), 2) for i in range(5)])


# ============================================================================
# 11. THE EMPIRICAL RULE
# ============================================================================

section("11. Approximate 68-95-99.7 rule")

print(
    """
For a Normal distribution:

    approximately 68% of observations lie within 1 standard deviation
    approximately 95% lie within 2 standard deviations
    approximately 99.7% lie within 3 standard deviations

These are approximations, not universal rules for arbitrary distributions.
"""

for deviations in [1, 2, 3]:
    probability = normal.cdf(
        normal.mean_value + deviations * normal.standard_deviation
    ) - normal.cdf(
        normal.mean_value - deviations * normal.standard_deviation
    )
    print(f"Within ±{deviations} SD: {probability:.6%}")


# ============================================================================
# 12. SAMPLING AND MONTE CARLO
# ============================================================================

section("12. Simulation and Monte Carlo estimation")


def summarize_sample(sample: Sequence[float]) -> dict[str, float]:
    if not sample:
        raise ValueError("Sample cannot be empty.")
    return {
        "count": float(len(sample)),
        "mean": statistics.fmean(sample),
        "variance_population": statistics.pvariance(sample),
        "standard_deviation_population": statistics.pstdev(sample),
        "minimum": min(sample),
        "maximum": max(sample),
    }


def estimate_probability(
    sampler: Callable[[], float],
    condition: Callable[[float], bool],
    trials: int,
) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    successes = sum(condition(sampler()) for _ in range(trials))
    return successes / trials


rng = random.Random(12345)
sample = [normal.sample(rng) for _ in range(10_000)]
print("Normal simulation statistics:")
for key, value in summarize_sample(sample).items():
    print(f"  {key}: {value:.6f}")

uniform_rng = random.Random(123)
estimated_half = estimate_probability(
    lambda: uniform.sample(uniform_rng),
    lambda x: x <= 15,
    100_000,
)
print("Estimated P(Uniform(10,20) <= 15):", estimated_half)
print("Exact probability:", uniform.cdf(15))


# ============================================================================
# 13. LAW OF LARGE NUMBERS
# ============================================================================

section("13. Law of Large Numbers")

print(
    """
The Law of Large Numbers says that, under suitable conditions, sample
averages tend to approach the expected value as the number of observations
increases.

It does not mean that short samples must resemble the expected value.
Random variation remains possible.
"""
)

bernoulli_rng = random.Random(99)
running_successes = 0
for trial_number in [10, 100, 1_000, 10_000]:
    while running_successes < 0:
        break

# Generate once so every displayed average uses a prefix of the same sample.
bernoulli_observations = [
    int(bernoulli_rng.random() < 0.3) for _ in range(10_000)
]
for count in [10, 100, 1_000, 10_000]:
    average = statistics.fmean(bernoulli_observations[:count])
    print(f"n={count:5d}, observed success rate={average:.5f}, expected=0.30000")


# ============================================================================
# 14. CENTRAL LIMIT THEOREM DEMONSTRATION
# ============================================================================

section("14. Central Limit Theorem demonstration")

print(
    """
The Central Limit Theorem concerns the distribution of properly normalized
sums or sample means of many independent observations under broad conditions.

The original population does not have to be Normal.

Here, samples come from a Uniform(0, 1) population. Individual observations
are uniform, but the distribution of sufficiently large sample means becomes
approximately bell-shaped.
"""
)

clt_rng = random.Random(2024)
sample_means = []
for _ in range(5_000):
    observations = [clt_rng.random() for _ in range(30)]
    sample_means.append(statistics.fmean(observations))

print("Population mean:", 0.5)
print("Observed mean of sample means:", statistics.fmean(sample_means))
print("Observed SD of sample means:", statistics.pstdev(sample_means))
print("Theoretical SD approximately:", math.sqrt((1 / 12) / 30))


# ============================================================================
# 15. APPROXIMATIONS AND RELATIONSHIPS
# ============================================================================

section("15. Distribution relationships and approximations")

print(
    """
Important relationships:

Bernoulli:
    one binary trial.

Binomial:
    sum of n independent Bernoulli variables with common p.

Poisson:
    a count model with rate lambda.
    For large n and small p, Binomial(n,p) can be approximated by
    Poisson(lambda=np).

Exponential:
    waiting time between events in a Poisson process.

Gaussian:
    continuous symmetric model and a major limiting distribution for sums
    and means under Central Limit Theorem conditions.

Uniform:
    equal density across a finite interval.

A Poisson process also implies:
    N(t) ~ Poisson(lambda*t)
    interarrival time ~ Exponential(lambda)
"""
)


def compare_binomial_poisson(
    n: int, p: float, maximum_count: int
) -> list[tuple[int, float, float]]:
    binomial_model = BinomialDistribution(n, p)
    poisson_model = PoissonDistribution(n * p)

    return [
        (k, binomial_model.pmf(k), poisson_model.pmf(k))
        for k in range(maximum_count + 1)
    ]


comparison = compare_binomial_poisson(1_000, 0.004, 12)
print("Binomial vs Poisson approximation:")
print(" k       Binomial       Poisson")
for k, binomial_probability, poisson_probability in comparison:
    print(f"{k:2d}   {binomial_probability:12.8f}   {poisson_probability:12.8f}")


# ============================================================================
# 16. CATEGORICAL DIFFERENCES
# ============================================================================

section("16. Choosing a distribution")

distribution_guide = {
    "Bernoulli": "One binary outcome.",
    "Binomial": "Number of successes in a fixed number of binary trials.",
    "Poisson": "Number of events in a fixed interval or region.",
    "Exponential": "Waiting time to the next event in a constant-rate Poisson process.",
    "Uniform": "Continuous uncertainty with constant density on a bounded interval.",
    "Gaussian": "Continuous, symmetric, bell-shaped variation characterized by mean and SD.",
}

for distribution_name, use_case in distribution_guide.items():
    print(f"{distribution_name:12s}: {use_case}")


# ============================================================================
# 17. EDGE CASES
# ============================================================================

section("17. Important edge cases")

print("Bernoulli p=0:", BernoulliDistribution(0).pmf(1))
print("Bernoulli p=1:", BernoulliDistribution(1).pmf(1))

degenerate_binomial_zero = BinomialDistribution(20, 0)
degenerate_binomial_one = BinomialDistribution(20, 1)
print("Binomial p=0, P(X=0):", degenerate_binomial_zero.pmf(0))
print("Binomial p=1, P(X=20):", degenerate_binomial_one.pmf(20))

zero_poisson = PoissonDistribution(0)
print("Poisson lambda=0, P(X=0):", zero_poisson.pmf(0))

print("Exponential CDF for negative x:", exponential.cdf(-1))
print("Uniform CDF below lower bound:", uniform.cdf(0))
print("Uniform CDF above upper bound:", uniform.cdf(100))
print("Normal CDF far below mean:", normal.cdf(-1_000))


# ============================================================================
# 18. ERROR HANDLING
# ============================================================================

section("18. Validation and common mistakes")

invalid_examples = [
    ("Bernoulli probability", lambda: BernoulliDistribution(1.2)),
    ("Binomial n", lambda: BinomialDistribution(-2, 0.5)),
    ("Poisson rate", lambda: PoissonDistribution(-1)),
    ("Exponential rate", lambda: ExponentialDistribution(0)),
    ("Uniform bounds", lambda: UniformDistribution(5, 5)),
    ("Gaussian SD", lambda: GaussianDistribution(0, -1)),
]

for name, operation in invalid_examples:
    try:
        operation()
    except (TypeError, ValueError) as error:
        print(f"{name}: correctly rejected -> {error}")


print(
    """
Common modeling mistakes include:

1. Treating correlation as independence.
2. Using Binomial when trials have changing probabilities.
3. Using Poisson when the event rate varies substantially and no
   appropriate extension is made.
4. Using Exponential waiting times when the hazard/rate is not constant.
5. Treating a continuous PDF value as a probability.
6. Assuming Normality merely because a dataset has a large sample size.
7. Ignoring truncation, censoring, seasonality, clustering, or dependence.
8. Confusing variance with standard deviation.
9. Using a simulation sample size that is too small for a rare-event estimate.
10. Ignoring numerical underflow or overflow in probability calculations.
"""
)


# ============================================================================
# 19. PDF VALUE IS NOT INTERVAL PROBABILITY
# ============================================================================

section("19. Density versus probability")

print(
    """
For continuous variables:

    P(X = exact_point) = 0

for an ordinary continuous distribution.

A PDF value such as f(15) is a density, not the probability that X equals 15.

For an interval:

    P(a <= X <= b) = integral of f(x) from a to b

For the Uniform(10,20) model:

    P(12 <= X <= 15) = (15-12)/(20-10) = 0.3
"""
)

interval_probability = uniform.cdf(15) - uniform.cdf(12)
print("Computed interval probability:", interval_probability)


# ============================================================================
# 20. CONDITIONAL AND TAIL PROBABILITIES
# ============================================================================

section("20. Tail probabilities")

print(
    """
Tail probabilities are important in anomaly detection, reliability,
quality control, finance, capacity planning, and hypothesis testing.

For a continuous model:

    P(X > x) = 1 - F(x)

For numerical stability, specialized survival functions can be preferable
to subtracting a CDF from 1 when the tail probability is extremely small.
"""
)

x = 130
upper_tail = 1.0 - normal.cdf(x)
print(f"Normal P(X>{x}) using CDF complement: {upper_tail:.10f}")


# ============================================================================
# 21. QUANTILES AND INTERVALS
# ============================================================================

section("21. Quantiles")

print(
    """
A quantile is a value below which a specified proportion of observations
falls.

Examples:
    median = 50th percentile
    first quartile = 25th percentile
    95th percentile = value exceeded by roughly 5% of observations

Quantiles are often easier to interpret operationally than raw density
values.
"""
)

for probability in [0.50, 0.90, 0.95, 0.99]:
    print(
        f"Normal quantile {probability:.0%}: "
        f"{normal.quantile_approx(probability):.3f}"
    )


# ============================================================================
# 22. CONFIDENCE-STYLE SIMULATION INTERVAL
# ============================================================================

section("22. Sampling variability")

print(
    """
A distribution's population parameters are fixed model quantities, while
statistics calculated from a finite sample vary from sample to sample.

For an approximately independent sample with finite variance, the standard
error of the sample mean is approximately:

    SE(mean) = sigma / sqrt(n)

The Central Limit Theorem can justify normal approximations under suitable
conditions.
"""
)

estimated_sigma = statistics.pstdev(sample)
sample_size = len(sample)
standard_error = estimated_sigma / math.sqrt(sample_size)
sample_mean = statistics.fmean(sample)
print("Sample mean:", sample_mean)
print("Estimated standard error:", standard_error)
print(
    "Approximate 95% normal interval:",
    (sample_mean - 1.96 * standard_error,
     sample_mean + 1.96 * standard_error),
)


# ============================================================================
# 23. PRACTICAL CASE STUDY
# ============================================================================

section("23. Practical case study: web service operations")

print(
    """
Suppose a service receives an average of 120 requests per minute.

Possible models:

1. Request count during one minute:
       Poisson(lambda=120)

2. Time between requests:
       Exponential(rate=120 per minute)

3. Whether a request fails:
       Bernoulli(p)

4. Number of failed requests among exactly 500 independent requests:
       Binomial(n=500, p)

5. Response latency:
       It might be approximately Gaussian only after examining the data.
       Real latency often has positive skew and may require another model.

6. A bounded measurement with no preferred value over an interval:
       Uniform(a,b), if that assumption is substantively justified.
"""
)

requests_per_minute = PoissonDistribution(120)
failure_probability = 0.01
failed_requests = BinomialDistribution(500, failure_probability)
interarrival = ExponentialDistribution(120)

print("P(exactly 120 requests):", requests_per_minute.pmf(120))
print("P(at least 130 requests):", 1.0 - requests_per_minute.cdf(129))
print("P(no more than 3 failures in 500):", failed_requests.cdf(3))
print("Expected failures in 500:", failed_requests.mean)
print("Expected interarrival time in minutes:", interarrival.mean)
print("Expected interarrival time in seconds:", interarrival.mean * 60)


# ============================================================================
# 24. HISTOGRAM WITHOUT EXTERNAL PACKAGES
# ============================================================================

section("24. Text histogram")

def text_histogram(
    values: Sequence[float],
    bins: int = 10,
    width: int = 50,
) -> None:
    if not values:
        raise ValueError("values cannot be empty.")
    if bins <= 0:
        raise ValueError("bins must be positive.")

    lower = min(values)
    upper = max(values)

    if math.isclose(lower, upper):
        print(f"{lower:.3f} | {'#' * width}")
        return

    bin_width = (upper - lower) / bins
    counts = [0] * bins

    for value in values:
        index = int((value - lower) / bin_width)
        index = min(index, bins - 1)
        counts[index] += 1

    maximum = max(counts)
    for index, count in enumerate(counts):
        start = lower + index * bin_width
        end = start + bin_width
        bar_length = int(width * count / maximum) if maximum else 0
        print(f"{start:8.3f} - {end:8.3f} | {'#' * bar_length} {count}")


text_histogram(sample[:2_000], bins=12)


# ============================================================================
# 25. PERFORMANCE CONSIDERATIONS
# ============================================================================

section("25. Performance and numerical considerations")

print(
    """
Important implementation considerations:

Factorials:
    Direct factorial values grow extremely quickly. math.lgamma is useful
    when only logarithms of factorials are needed.

Binomial probabilities:
    Computing n! / (k!(n-k)!) directly can overflow for large n.
    Logarithms or recurrence relations are safer.

Poisson probabilities:
    exp(-lambda) * lambda^k / k! can underflow or overflow.
    Log-PMF calculations are more stable.

Normal tails:
    1 - CDF(x) can lose precision when CDF(x) is extremely close to 1.
    Specialized survival functions are preferred in numerical libraries.

Simulation:
    Python loops are convenient for learning but may be slower than
    vectorized numerical implementations for millions of observations.

Randomness:
    A seeded pseudo-random generator gives reproducible experiments.
    It does not make the generator cryptographically secure.
"""
)


# ============================================================================
# 26. RANDOMNESS AND SECURITY
# ============================================================================

section("26. Statistical randomness versus security randomness")

print(
    """
The random module is designed for simulation and general-purpose
pseudo-randomness, not cryptographic security.

For security-sensitive tokens, keys, reset codes, or authentication values,
Python's secrets module should be used instead.

A statistical simulation can be reproducible by setting a seed:

    rng = random.Random(123)

A security token must not depend on a predictable simulation seed.
"""
)

import secrets

print("Example non-sensitive simulation random value:", random.Random(1).random())
print("Example security-oriented random integer:", secrets.randbelow(1_000_000))


# ============================================================================
# 27. PRODUCTION MODELING CHECKLIST
# ============================================================================

section("27. Production-oriented modeling checklist")

print(
    """
Before selecting one of these distributions for a real system, verify:

Data:
    - What exactly is being measured?
    - Is the variable discrete or continuous?
    - Are observations independent?
    - Is the observation window fixed?
    - Are there missing, censored, truncated, or duplicated observations?

Assumptions:
    - Is the event rate stable?
    - Is the success probability stable?
    - Is the Normal symmetry assumption reasonable?
    - Are there structural bounds?
    - Are there multiple populations or mixture components?

Validation:
    - Compare predicted and observed frequencies.
    - Inspect residuals.
    - Check tail behavior.
    - Test sensitivity to parameter changes.
    - Validate on data not used to fit the model.

Operations:
    - Monitor parameter drift.
    - Re-estimate parameters when the process changes.
    - Keep units explicit.
    - Document assumptions.
    - Record random seeds when reproducibility matters.

Risk:
    - Rare-event estimates can require very large simulations.
    - A mathematically convenient model can still be substantively wrong.
    - Tail errors can matter much more than central-region errors.
"""
)


# ============================================================================
# 28. FINAL INTEGRATED EXAMPLE
# ============================================================================

section("28. Integrated probability model")

print(
    """
Scenario:
    A monitoring system observes 60 minutes of operation.

Assumptions for this educational model:
    request rate = 50 per minute
    failure probability per request = 0.02
    response time is modeled as Normal(200 ms, 40 ms)

These assumptions are illustrative. Real data should be used to validate
whether the models are appropriate.
"""
)

minutes = 60
request_rate = 50.0
failure_rate = 0.02
response_time = GaussianDistribution(200.0, 40.0)

expected_requests = request_rate * minutes
expected_failures = expected_requests * failure_rate

print("Expected requests in one hour:", expected_requests)
print("Expected failures in one hour:", expected_failures)

hourly_requests = PoissonDistribution(expected_requests)
print(
    "Probability of at least 3,100 requests:",
    1.0 - hourly_requests.cdf(3_099),
)

request_batch = BinomialDistribution(100, failure_rate)
print("P(0 failures in 100 requests):", request_batch.pmf(0))
print("P(at least 5 failures in 100 requests):", request_batch.probability_at_least(5))

print(
    "P(response time <= 250 ms):",
    response_time.cdf(250),
)

# Demonstrate the connection between count and waiting-time models.
request_process = PoissonDistribution(request_rate)
request_waiting_time = ExponentialDistribution(request_rate)

print("Expected requests in 0.5 minutes:", request_process.mean * 0.5)
print(
    "Expected waiting time between requests, seconds:",
    request_waiting_time.mean * 60,
)


# ============================================================================
# 29. SELF-CHECKING ASSERTIONS
# ============================================================================

section("29. Mathematical consistency checks")

# PMFs should sum approximately to one over a sufficiently complete support.
bernoulli_total = coin.pmf(0) + coin.pmf(1)
binomial_total = sum(binomial.pmf(k) for k in range(binomial.n + 1))
poisson_total = sum(poisson.pmf(k) for k in range(0, 60))

assert math.isclose(bernoulli_total, 1.0, abs_tol=1e-12)
assert math.isclose(binomial_total, 1.0, abs_tol=1e-12)
assert math.isclose(poisson_total, 1.0, abs_tol=1e-10)

assert math.isclose(coin.mean, 0.5)
assert math.isclose(binomial.mean, 3.0)
assert math.isclose(poisson.mean, 4.0)
assert math.isclose(exponential.mean, 0.5)
assert math.isclose(uniform.mean, 15.0)
assert math.isclose(normal.mean_value, 100.0)

print("All consistency assertions passed.")


# ============================================================================
# 30. COMPACT REFERENCE
# ============================================================================

section("30. Compact reference")

reference = [
    ("Bernoulli", "p", "mean=p", "variance=p(1-p)", "binary outcome"),
    ("Binomial", "n,p", "mean=np", "variance=np(1-p)", "success count"),
    ("Poisson", "lambda", "mean=lambda", "variance=lambda", "event count"),
    ("Exponential", "rate=lambda", "mean=1/lambda", "variance=1/lambda^2", "waiting time"),
    ("Uniform", "a,b", "mean=(a+b)/2", "variance=(b-a)^2/12", "bounded continuous"),
    ("Gaussian", "mu,sigma", "mean=mu", "variance=sigma^2", "continuous bell-shaped"),
]

print(f"{'Distribution':12s} | {'Parameters':15s} | {'Mean':18s} | {'Variance':22s} | Use")
print("-" * 100)
for row in reference:
    print(f"{row[0]:12s} | {row[1]:15s} | {row[2]:18s} | {row[3]:22s} | {row[4]}")

print(
    """
End of standalone study script.

Key mathematical map:

    Bernoulli -> one binary trial
    Binomial  -> sum of Bernoulli trials
    Poisson   -> event counts
    Exponential -> waiting times for Poisson events
    Uniform   -> constant density on an interval
    Gaussian  -> symmetric continuous model and important limiting distribution

The correct distribution is determined by the data-generating assumptions,
not by the convenience of the formula.
"""
)
