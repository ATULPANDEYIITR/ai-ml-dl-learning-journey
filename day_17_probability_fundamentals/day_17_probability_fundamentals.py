"""
Probability Fundamentals
=========================

A self-contained study script covering:

- Probability spaces
- Sample spaces and outcomes
- Events and set operations
- Probability axioms
- Counting and finite probability
- Conditional probability
- Bayes' theorem
- Independence and pairwise vs mutual independence
- Law of total probability
- Random experiments and simulations
- Complement, union and intersection rules
- Discrete random variables
- Expectation and variance
- Covariance and correlation
- Common mistakes and edge cases
- Numerical stability and simulation considerations
- A practical reliability case study

The examples use only Python's standard library.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, FrozenSet, Iterable, Sequence, TypeVar


T = TypeVar("T")


# ---------------------------------------------------------------------------
# 1. Fundamental terminology
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def explain_probability_space() -> None:
    """
    A probability space is written as (Omega, F, P).

    Omega:
        The sample space, containing every possible elementary outcome.

    F:
        The event collection. In finite examples this can be represented
        simply as a collection of subsets of Omega.

    P:
        A probability measure assigning each event a number in [0, 1].

    For a finite sample space with equally likely outcomes:
        P(A) = |A| / |Omega|

    Probability is therefore a mathematical measure of uncertainty, not
    merely a percentage attached to an isolated observation.
    """
    sample_space = frozenset({"H", "T"})
    event_heads = frozenset({"H"})

    print("Sample space:", sample_space)
    print("Event A = heads:", event_heads)
    print("P(A) for a fair coin:", len(event_heads) / len(sample_space))


# ---------------------------------------------------------------------------
# 2. A reusable finite probability space
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FiniteProbabilitySpace:
    """
    Represents a finite discrete probability space.

    `probabilities` maps each elementary outcome to its probability.

    Validation enforces the probability axioms:
        P(outcome) >= 0
        sum(P(outcome)) = 1

    A dictionary is used rather than assuming all outcomes are equally likely.
    This allows the same class to model biased coins, loaded dice, medical
    tests, reliability systems, and other non-uniform experiments.
    """

    probabilities: dict[T, float]

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("A probability space must contain an outcome.")

        tolerance = 1e-12

        for outcome, probability in self.probabilities.items():
            if not math.isfinite(probability):
                raise ValueError(f"Probability for {outcome!r} is not finite.")
            if probability < 0:
                raise ValueError(f"Probability for {outcome!r} is negative.")

        total = sum(self.probabilities.values())

        if not math.isclose(total, 1.0, abs_tol=tolerance):
            raise ValueError(
                f"Probabilities must sum to 1. Current total is {total}."
            )

    @property
    def sample_space(self) -> FrozenSet[T]:
        return frozenset(self.probabilities)

    def probability(self, event: Iterable[T]) -> float:
        """
        Calculate P(A) by summing probabilities of elementary outcomes in A.

        An event containing an outcome outside Omega has probability zero for
        that outside outcome, but this method deliberately rejects such input
        because silently accepting a malformed event often hides programming
        mistakes.
        """
        event_set = frozenset(event)

        unknown = event_set - self.sample_space
        if unknown:
            raise ValueError(f"Event contains outcomes outside Omega: {unknown}")

        return sum(self.probabilities[outcome] for outcome in event_set)

    def complement(self, event: Iterable[T]) -> FrozenSet[T]:
        event_set = frozenset(event)

        unknown = event_set - self.sample_space
        if unknown:
            raise ValueError(f"Event contains unknown outcomes: {unknown}")

        return self.sample_space - event_set

    def conditional_probability(
        self,
        event: Iterable[T],
        condition: Iterable[T],
    ) -> float:
        """
        P(A | B) = P(A intersection B) / P(B), provided P(B) > 0.

        Conditional probability changes the effective sample space:
        once B is known to have happened, outcomes outside B are no longer
        possible.
        """
        event_set = frozenset(event)
        condition_set = frozenset(condition)

        p_condition = self.probability(condition_set)

        if math.isclose(p_condition, 0.0, abs_tol=1e-15):
            raise ZeroDivisionError(
                "Conditional probability is undefined when P(B) = 0."
            )

        intersection = event_set & condition_set
        return self.probability(intersection) / p_condition

    def independent(
        self,
        event_a: Iterable[T],
        event_b: Iterable[T],
    ) -> bool:
        """
        A and B are independent when:

            P(A intersection B) = P(A) P(B)

        Equivalently, when P(B) > 0:

            P(A | B) = P(A)

        The product form works even when P(B) = 0.
        """
        a = frozenset(event_a)
        b = frozenset(event_b)

        left = self.probability(a & b)
        right = self.probability(a) * self.probability(b)

        return math.isclose(left, right, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# 3. Set-based event operations
# ---------------------------------------------------------------------------

def event_operations_demo() -> None:
    print_section("Events and set operations")

    space = FiniteProbabilitySpace(
        {
            1: 1 / 6,
            2: 1 / 6,
            3: 1 / 6,
            4: 1 / 6,
            5: 1 / 6,
            6: 1 / 6,
        }
    )

    even = {2, 4, 6}
    greater_than_three = {4, 5, 6}

    intersection = even & greater_than_three
    union = even | greater_than_three
    complement = space.complement(even)

    print("Omega:", sorted(space.sample_space))
    print("A = even:", sorted(even))
    print("B = greater than 3:", sorted(greater_than_three))
    print("A intersection B:", sorted(intersection))
    print("A union B:", sorted(union))
    print("A complement:", sorted(complement))

    print("P(A):", space.probability(even))
    print("P(B):", space.probability(greater_than_three))
    print("P(A intersection B):", space.probability(intersection))
    print("P(A union B):", space.probability(union))
    print("P(A complement):", space.probability(complement))

    # Inclusion-exclusion:
    # P(A union B) = P(A) + P(B) - P(A intersection B)
    lhs = space.probability(union)
    rhs = (
        space.probability(even)
        + space.probability(greater_than_three)
        - space.probability(intersection)
    )
    print("Inclusion-exclusion check:", math.isclose(lhs, rhs))


# ---------------------------------------------------------------------------
# 4. Probability axioms
# ---------------------------------------------------------------------------

def probability_axioms_demo() -> None:
    print_section("Probability axioms")

    space = FiniteProbabilitySpace(
        {
            "red": 0.25,
            "blue": 0.50,
            "green": 0.25,
        }
    )

    empty_event: set[str] = set()
    whole_space = space.sample_space
    red = {"red"}
    blue = {"blue"}

    print("Non-negativity:", space.probability(red) >= 0)
    print("P(empty):", space.probability(empty_event))
    print("P(Omega):", space.probability(whole_space))
    print("P(red):", space.probability(red))
    print("P(blue):", space.probability(blue))

    # For disjoint events, additivity says:
    # P(A union B) = P(A) + P(B)
    print(
        "Finite additivity:",
        math.isclose(
            space.probability(red | blue),
            space.probability(red) + space.probability(blue),
        ),
    )

    # The complement rule follows directly from the axioms:
    # P(A^c) = 1 - P(A)
    print(
        "Complement rule:",
        math.isclose(
            space.probability(space.complement(red)),
            1 - space.probability(red),
        ),
    )


# ---------------------------------------------------------------------------
# 5. Counting and equally likely outcomes
# ---------------------------------------------------------------------------

def combinations(n: int, k: int) -> int:
    if not 0 <= k <= n:
        return 0
    return math.comb(n, k)


def counting_demo() -> None:
    print_section("Counting and finite probability")

    # Two dice produce 36 ordered outcomes.
    outcomes = list(itertools.product(range(1, 7), repeat=2))
    total = len(outcomes)

    sum_at_least_nine = [
        outcome for outcome in outcomes if sum(outcome) >= 9
    ]

    probability = len(sum_at_least_nine) / total

    print("Two-dice outcomes:", total)
    print("Outcomes with sum >= 9:", len(sum_at_least_nine))
    print("P(sum >= 9):", probability)

    # Binomial counting:
    # Exactly 3 heads in 5 fair coin flips:
    # C(5,3) / 2^5.
    ways = combinations(5, 3)
    probability_exactly_three_heads = ways / (2 ** 5)

    print("Ways to choose 3 heads from 5:", ways)
    print("P(exactly 3 heads in 5 flips):", probability_exactly_three_heads)


# ---------------------------------------------------------------------------
# 6. Conditional probability
# ---------------------------------------------------------------------------

def conditional_probability_demo() -> None:
    print_section("Conditional probability")

    # A die is rolled.
    # A = outcome is even
    # B = outcome is greater than 3
    space = FiniteProbabilitySpace({i: 1 / 6 for i in range(1, 7)})

    a = {2, 4, 6}
    b = {4, 5, 6}

    p_a = space.probability(a)
    p_b = space.probability(b)
    p_ab = space.probability(a & b)
    p_a_given_b = space.conditional_probability(a, b)

    print("P(A):", p_a)
    print("P(B):", p_b)
    print("P(A intersection B):", p_ab)
    print("P(A | B):", p_a_given_b)

    # Multiplication rule:
    # P(A intersection B) = P(A | B) P(B)
    print(
        "Multiplication rule:",
        math.isclose(p_ab, p_a_given_b * p_b),
    )


# ---------------------------------------------------------------------------
# 7. Bayes' theorem
# ---------------------------------------------------------------------------

def bayes_demo() -> None:
    print_section("Bayes' theorem")

    """
    Example:

    A disease has prevalence 1%.
    A diagnostic test has:
        sensitivity = P(positive | disease) = 95%
        specificity = P(negative | no disease) = 90%

    Therefore:
        P(positive | no disease) = 10%

    We want:
        P(disease | positive)

    Bayes:
        P(D | +) =
            P(+ | D) P(D)
            -------------------------
            P(+)

    and:

        P(+) =
            P(+ | D)P(D)
            + P(+ | not D)P(not D)
    """

    p_disease = 0.01
    p_positive_given_disease = 0.95
    p_positive_given_no_disease = 0.10

    p_positive = (
        p_positive_given_disease * p_disease
        + p_positive_given_no_disease * (1 - p_disease)
    )

    posterior = (
        p_positive_given_disease * p_disease / p_positive
    )

    print("P(disease):", p_disease)
    print("P(positive):", p_positive)
    print("P(disease | positive):", posterior)

    # The posterior is substantially different from the test sensitivity.
    # This illustrates why base rates matter.
    assert 0 <= posterior <= 1


# ---------------------------------------------------------------------------
# 8. Law of total probability
# ---------------------------------------------------------------------------

def total_probability_demo() -> None:
    print_section("Law of total probability")

    """
    Suppose orders originate from three production facilities.

    Factory probabilities:
        A = 0.50
        B = 0.30
        C = 0.20

    Defect rates:
        P(D | A) = 0.01
        P(D | B) = 0.03
        P(D | C) = 0.05

    Since the factories partition the population:

        P(D) =
            P(D|A)P(A)
          + P(D|B)P(B)
          + P(D|C)P(C)
    """

    factory_probabilities = {
        "A": 0.50,
        "B": 0.30,
        "C": 0.20,
    }

    defect_rates = {
        "A": 0.01,
        "B": 0.03,
        "C": 0.05,
    }

    total_defect_probability = sum(
        factory_probabilities[factory] * defect_rates[factory]
        for factory in factory_probabilities
    )

    print("Overall defect probability:", total_defect_probability)


# ---------------------------------------------------------------------------
# 9. Independence
# ---------------------------------------------------------------------------

def independence_demo() -> None:
    print_section("Independence")

    space = FiniteProbabilitySpace({i: 1 / 6 for i in range(1, 7)})

    even = {2, 4, 6}
    greater_than_three = {4, 5, 6}

    p_a = space.probability(even)
    p_b = space.probability(greater_than_three)
    p_ab = space.probability(even & greater_than_three)

    print("P(A):", p_a)
    print("P(B):", p_b)
    print("P(A intersection B):", p_ab)
    print("P(A)P(B):", p_a * p_b)
    print("Independent?", space.independent(even, greater_than_three))

    # A useful distinction:
    #
    # Mutually exclusive events cannot happen together:
    #     A intersection B = empty
    #
    # If both have positive probability, they cannot be independent.
    #
    # Independence means occurrence of one event does not change the
    # probability of the other.
    red = {1}
    blue = {2}

    print("P(red intersection blue):", space.probability(red & blue))
    print("P(red)P(blue):", space.probability(red) * space.probability(blue))
    print("Mutually exclusive?", not bool(red & blue))
    print("Independent?", space.independent(red, blue))


# ---------------------------------------------------------------------------
# 10. Pairwise versus mutual independence
# ---------------------------------------------------------------------------

def pairwise_not_mutual_demo() -> None:
    print_section("Pairwise independence versus mutual independence")

    """
    Three fair independent-looking events can be constructed from two fair
    bits X and Y.

    Let:
        A = {X = 1}
        B = {Y = 1}
        C = {X XOR Y = 1}

    Every pair is independent, but A, B and C are not mutually independent
    because:

        P(A intersection B intersection C) = 0

    while:

        P(A)P(B)P(C) = 1/8.
    """

    outcomes = {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }

    space = FiniteProbabilitySpace(
        {outcome: 1 / 4 for outcome in outcomes}
    )

    a = {o for o in outcomes if o[0] == 1}
    b = {o for o in outcomes if o[1] == 1}
    c = {o for o in outcomes if (o[0] ^ o[1]) == 1}

    print("P(A):", space.probability(a))
    print("P(B):", space.probability(b))
    print("P(C):", space.probability(c))

    print("A and B independent:", space.independent(a, b))
    print("A and C independent:", space.independent(a, c))
    print("B and C independent:", space.independent(b, c))

    triple_intersection = a & b & c
    p_triple = space.probability(triple_intersection)
    product = (
        space.probability(a)
        * space.probability(b)
        * space.probability(c)
    )

    print("P(A intersection B intersection C):", p_triple)
    print("P(A)P(B)P(C):", product)
    print("Mutually independent:", math.isclose(p_triple, product))


# ---------------------------------------------------------------------------
# 11. Discrete random variables
# ---------------------------------------------------------------------------

def expected_value(
    distribution: dict[T, float],
    value_function: Callable[[T], float] = lambda x: float(x),  # type: ignore[arg-type]
) -> float:
    return sum(
        probability * value_function(outcome)
        for outcome, probability in distribution.items()
    )


def variance(
    distribution: dict[T, float],
    value_function: Callable[[T], float] = lambda x: float(x),  # type: ignore[arg-type]
) -> float:
    mean = expected_value(distribution, value_function)
    return sum(
        probability * (value_function(outcome) - mean) ** 2
        for outcome, probability in distribution.items()
    )


def random_variable_demo() -> None:
    print_section("Discrete random variables")

    die = {i: 1 / 6 for i in range(1, 7)}

    mean = expected_value(die)
    var = variance(die)

    print("E[X]:", mean)
    print("Var(X):", var)
    print("Std(X):", math.sqrt(var))

    # Indicator variables are especially important:
    #
    # I_A = 1 if A happens, 0 otherwise.
    #
    # E[I_A] = P(A).
    event = {4, 5, 6}

    indicator_distribution = {
        outcome: (1.0 if outcome in event else 0.0) / 6
        for outcome in die
    }

    # The dictionary above groups probabilities by elementary outcomes, so
    # compute the indicator expectation directly.
    indicator_expectation = sum(
        die[outcome] * (1 if outcome in event else 0)
        for outcome in die
    )

    print("P(X >= 4):", sum(die[i] for i in event))
    print("E[indicator of X >= 4]:", indicator_expectation)
    print("Indicator distribution representation:", indicator_distribution)


# ---------------------------------------------------------------------------
# 12. Covariance and correlation
# ---------------------------------------------------------------------------

def covariance_demo() -> None:
    print_section("Covariance and correlation")

    """
    For a joint distribution of X and Y:

        Cov(X,Y) = E[(X-E[X])(Y-E[Y])]

    Equivalent form:

        Cov(X,Y) = E[XY] - E[X]E[Y]

    Independence implies zero covariance when the relevant expectations
    exist. The reverse is generally false.
    """

    joint = {
        (0, 0): 0.25,
        (0, 1): 0.25,
        (1, 0): 0.25,
        (1, 1): 0.25,
    }

    mean_x = sum(x * p for (x, _), p in joint.items())
    mean_y = sum(y * p for (_, y), p in joint.items())
    mean_xy = sum(x * y * p for (x, y), p in joint.items())

    covariance = mean_xy - mean_x * mean_y

    variance_x = sum(
        (x - mean_x) ** 2 * p
        for (x, _), p in joint.items()
    )
    variance_y = sum(
        (y - mean_y) ** 2 * p
        for (_, y), p in joint.items()
    )

    correlation = covariance / math.sqrt(variance_x * variance_y)

    print("E[X]:", mean_x)
    print("E[Y]:", mean_y)
    print("Cov(X,Y):", covariance)
    print("Corr(X,Y):", correlation)


# ---------------------------------------------------------------------------
# 13. Simulation and Monte Carlo estimation
# ---------------------------------------------------------------------------

def simulate_coin_flips(
    number_of_flips: int,
    probability_of_heads: float = 0.5,
    seed: int | None = 42,
) -> float:
    """
    Estimate P(heads) through Monte Carlo simulation.

    Simulation produces an estimate rather than the exact probability.

    As sample size grows, random error generally decreases at a rate related
    to 1/sqrt(n). Therefore reducing statistical error by a factor of 10
    can require roughly 100 times as many independent samples.
    """
    if number_of_flips <= 0:
        raise ValueError("number_of_flips must be positive.")

    if not 0 <= probability_of_heads <= 1:
        raise ValueError("probability_of_heads must be between 0 and 1.")

    generator = random.Random(seed)

    heads = sum(
        generator.random() < probability_of_heads
        for _ in range(number_of_flips)
    )

    return heads / number_of_flips


def monte_carlo_demo() -> None:
    print_section("Monte Carlo simulation")

    exact = 0.5

    for sample_size in (100, 1_000, 10_000, 100_000):
        estimate = simulate_coin_flips(sample_size)
        error = abs(estimate - exact)

        print(
            f"n={sample_size:>6}: "
            f"estimate={estimate:.5f}, "
            f"absolute error={error:.5f}"
        )


# ---------------------------------------------------------------------------
# 14. Sampling without replacement
# ---------------------------------------------------------------------------

def without_replacement_demo() -> None:
    print_section("Sampling without replacement")

    """
    A deck contains 52 cards and 13 hearts.

    P(first card is a heart) = 13/52.

    If the first card is known to be a heart:

        P(second heart | first heart) = 12/51.

    The second probability differs because the first draw changes the
    composition of the population.

    This is a key distinction between dependent and independent trials.
    """

    p_first_heart = 13 / 52
    p_second_heart_given_first = 12 / 51

    print("P(first heart):", p_first_heart)
    print(
        "P(second heart | first heart):",
        p_second_heart_given_first,
    )

    # Probability of two hearts:
    p_two_hearts = p_first_heart * p_second_heart_given_first
    print("P(two consecutive hearts):", p_two_hearts)


# ---------------------------------------------------------------------------
# 15. Conditional independence
# ---------------------------------------------------------------------------

def conditional_independence_demo() -> None:
    print_section("Conditional independence")

    """
    Two events can become independent after conditioning on another event.

    A common structural example is:

        Disease -> Test A
        Disease -> Test B

    Given the true disease state, the test outcomes may be modeled as
    conditionally independent even though the tests can be associated in
    the overall population.

    This is one reason conditional probability is fundamental to Bayesian
    networks, diagnosis, classification and probabilistic graphical models.
    """

    # A small numerical illustration.
    p_disease = 0.2
    p_a_given_disease = 0.9
    p_b_given_disease = 0.8

    p_both_given_disease = p_a_given_disease * p_b_given_disease

    print("P(A | D):", p_a_given_disease)
    print("P(B | D):", p_b_given_disease)
    print("P(A and B | D), under conditional independence:",
          p_both_given_disease)


# ---------------------------------------------------------------------------
# 16. Probability bounds
# ---------------------------------------------------------------------------

def probability_bounds_demo() -> None:
    print_section("Useful probability bounds")

    space = FiniteProbabilitySpace({i: 1 / 6 for i in range(1, 7)})

    a = {2, 4, 6}
    b = {4, 5, 6}

    p_a = space.probability(a)
    p_b = space.probability(b)
    p_intersection = space.probability(a & b)
    p_union = space.probability(a | b)

    # Fréchet bounds:
    #
    # max(0, P(A)+P(B)-1) <= P(A intersection B)
    #                         <= min(P(A), P(B))
    lower = max(0.0, p_a + p_b - 1)
    upper = min(p_a, p_b)

    print("P(A intersection B):", p_intersection)
    print("Lower bound:", lower)
    print("Upper bound:", upper)

    # Union bound:
    # P(A union B) <= P(A) + P(B)
    print("Union bound:", p_union <= p_a + p_b)


# ---------------------------------------------------------------------------
# 17. Complement and at-least-one calculations
# ---------------------------------------------------------------------------

def at_least_one_demo() -> None:
    print_section("At least one event")

    """
    For independent events with probability p of success over n trials:

        P(at least one success)
            = 1 - P(no successes)
            = 1 - (1-p)^n

    Computing the complement is often much simpler than adding every
    mutually exclusive case for 1, 2, ..., n successes.
    """

    p = 0.1
    n = 10

    probability_at_least_one = 1 - (1 - p) ** n

    print("P(at least one success):", probability_at_least_one)


# ---------------------------------------------------------------------------
# 18. Reliability system
# ---------------------------------------------------------------------------

@dataclass
class Component:
    name: str
    reliability: float

    def __post_init__(self) -> None:
        if not 0 <= self.reliability <= 1:
            raise ValueError("Reliability must be in [0, 1].")


def series_reliability(components: Sequence[Component]) -> float:
    """
    In a series system every component must work.

    Under an independence assumption:

        R_series = product(R_i)
    """
    reliability = 1.0

    for component in components:
        reliability *= component.reliability

    return reliability


def parallel_reliability(components: Sequence[Component]) -> float:
    """
    In a parallel system at least one component must work.

    Under independence:

        R_parallel = 1 - product(1 - R_i)
    """
    probability_all_fail = 1.0

    for component in components:
        probability_all_fail *= 1 - component.reliability

    return 1 - probability_all_fail


def reliability_case_study() -> None:
    print_section("Reliability case study")

    """
    A service has two independent application servers behind a load balancer.
    Either server can serve the request.

    Then the server subsystem behaves like a parallel system.

    A database is required for every successful request, so the database is
    in series with the server subsystem.
    """

    servers = [
        Component("server-1", 0.98),
        Component("server-2", 0.97),
    ]
    database = Component("database", 0.995)

    server_subsystem = parallel_reliability(servers)
    total_system = server_subsystem * database.reliability

    print("Server subsystem reliability:", server_subsystem)
    print("Database reliability:", database.reliability)
    print("Total system reliability:", total_system)

    # Failure probability is the complement.
    print("Total system failure probability:", 1 - total_system)


# ---------------------------------------------------------------------------
# 19. Validation and common probability mistakes
# ---------------------------------------------------------------------------

def validation_demo() -> None:
    print_section("Validation and common mistakes")

    invalid_examples = [
        {"A": 0.7, "B": 0.5},       # total > 1
        {"A": 0.8, "B": -0.2, "C": 0.4},  # negative probability
        {},                          # empty sample space
    ]

    for index, probabilities in enumerate(invalid_examples, start=1):
        try:
            FiniteProbabilitySpace(probabilities)
        except ValueError as error:
            print(f"Invalid example {index}: correctly rejected -> {error}")

    # Important distinction:
    #
    # "P(A and B) = P(A)P(B)" is not always valid.
    # It is valid when A and B are independent.
    #
    # For dependent events:
    # P(A and B) = P(A|B)P(B).
    #
    # Assuming independence without evidence or a modeling assumption is
    # a common probability error.


# ---------------------------------------------------------------------------
# 20. Floating-point considerations
# ---------------------------------------------------------------------------

def floating_point_demo() -> None:
    print_section("Floating-point probability calculations")

    probabilities = [0.1, 0.2, 0.3, 0.4]
    total = sum(probabilities)

    print("Raw floating-point total:", total)
    print("Exactly equal to 1.0:", total == 1.0)
    print(
        "Numerically close to 1.0:",
        math.isclose(total, 1.0, abs_tol=1e-12),
    )

    """
    Probability formulas are mathematical, but computer arithmetic uses
    finite-precision representations.

    Practical rules:
        - validate with a tolerance;
        - avoid exact equality for floating-point calculations;
        - keep probabilities within [0, 1] after transformations;
        - for very small probabilities, consider log-probabilities in advanced
          statistical systems.
    """


# ---------------------------------------------------------------------------
# 21. Log-probabilities
# ---------------------------------------------------------------------------

def log_probability_demo() -> None:
    print_section("Log-probabilities")

    """
    Multiplying many probabilities can underflow toward zero.

    Instead of:

        P = p1 * p2 * ... * pn

    we can use:

        log(P) = log(p1) + log(p2) + ... + log(pn)

    This is common in probabilistic machine learning and numerical statistics.
    """

    probabilities = [1e-20] * 100

    direct_product = math.prod(probabilities)
    log_product = sum(math.log(p) for p in probabilities)

    print("Direct product:", direct_product)
    print("Log product:", log_product)
    print("Equivalent product from log:", math.exp(log_product))


# ---------------------------------------------------------------------------
# 22. A complete probability query helper
# ---------------------------------------------------------------------------

def probability_table_demo() -> None:
    print_section("Probability table")

    weather_space = FiniteProbabilitySpace(
        {
            "sunny": 0.50,
            "cloudy": 0.30,
            "rainy": 0.20,
        }
    )

    print(f"{'Outcome':<12}{'Probability':>14}")
    print("-" * 26)

    for outcome, probability in weather_space.probabilities.items():
        print(f"{outcome:<12}{probability:>14.4f}")

    outdoor_friendly = {"sunny", "cloudy"}

    print(
        "\nP(outdoor-friendly):",
        weather_space.probability(outdoor_friendly),
    )


# ---------------------------------------------------------------------------
# 23. Mini test suite
# ---------------------------------------------------------------------------

def run_tests() -> None:
    print_section("Assertions and self-tests")

    space = FiniteProbabilitySpace({i: 1 / 6 for i in range(1, 7)})

    assert math.isclose(space.probability(space.sample_space), 1.0)
    assert math.isclose(space.probability(set()), 0.0)

    event = {2, 4, 6}
    assert math.isclose(
        space.probability(space.complement(event)),
        1 - space.probability(event),
    )

    a = {2, 4, 6}
    b = {4, 5, 6}

    assert math.isclose(
        space.probability(a | b),
        space.probability(a)
        + space.probability(b)
        - space.probability(a & b),
    )

    assert math.isclose(
        space.conditional_probability(a, b),
        space.probability(a & b) / space.probability(b),
    )

    assert not space.independent(a, b)

    print("All probability-space tests passed.")


# ---------------------------------------------------------------------------
# 24. Main educational execution
# ---------------------------------------------------------------------------

def main() -> None:
    print_section("Probability fundamentals")

    print(
        """
Probability studies uncertain outcomes using a formal mathematical model.

Core objects:
    sample space  -> Omega, all possible outcomes
    event         -> subset of Omega
    probability   -> numerical measure P(A) in [0, 1]
    conditional probability -> P(A | B)
    independence  -> occurrence of one event does not change the probability
                      of the other event

The script moves from finite probability spaces to conditional probability,
Bayes' theorem, independence, random variables, simulation, numerical issues,
and a reliability case study.
""".strip()
    )

    explain_probability_space()
    probability_axioms_demo()
    event_operations_demo()
    counting_demo()
    conditional_probability_demo()
    bayes_demo()
    total_probability_demo()
    independence_demo()
    pairwise_not_mutual_demo()
    random_variable_demo()
    covariance_demo()
    monte_carlo_demo()
    without_replacement_demo()
    conditional_independence_demo()
    probability_bounds_demo()
    at_least_one_demo()
    reliability_case_study()
    validation_demo()
    floating_point_demo()
    log_probability_demo()
    probability_table_demo()
    run_tests()

    print_section("End of probability fundamentals demonstration")


if __name__ == "__main__":
    main()
