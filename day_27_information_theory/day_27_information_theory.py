"""
Information Theory: Entropy, Cross-Entropy, KL Divergence, and Mutual Information

This standalone study script progresses from the basic ideas of probability and
self-information to entropy, cross-entropy, KL divergence, mutual information,
conditional entropy, joint distributions, numerical stability, applications,
edge cases, and practical machine-learning examples.

No external packages are required.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


# ============================================================================
# 1. FUNDAMENTAL MATHEMATICAL HELPERS
# ============================================================================

EPSILON = 1e-15


def validate_distribution(
    distribution: Dict[str, float],
    *,
    tolerance: float = 1e-9,
    allow_zero: bool = True,
) -> None:
    """Validate a finite discrete probability distribution."""
    if not distribution:
        raise ValueError("A probability distribution cannot be empty.")

    total = 0.0
    for outcome, probability in distribution.items():
        if not math.isfinite(probability):
            raise ValueError(f"Probability for {outcome!r} is not finite.")
        if probability < 0:
            raise ValueError(f"Probability for {outcome!r} is negative.")
        if not allow_zero and probability == 0:
            raise ValueError(f"Probability for {outcome!r} cannot be zero.")
        total += probability

    if abs(total - 1.0) > tolerance:
        raise ValueError(f"Probabilities must sum to 1; received {total}.")


def log2_safe(value: float) -> float:
    """Return log2(value), rejecting non-positive values."""
    if value <= 0:
        raise ValueError("Logarithm requires a positive value.")
    return math.log2(value)


def self_information(probability: float, base: float = 2.0) -> float:
    """
    Information content of an event.

    I(x) = -log_b P(x)

    Base 2 gives bits, base e gives nats.
    """
    if not 0 < probability <= 1:
        raise ValueError("Probability must be in (0, 1].")
    return -math.log(probability, base)


# ============================================================================
# 2. SELF-INFORMATION
# ============================================================================

def demonstrate_self_information() -> None:
    print("\n" + "=" * 78)
    print("1. SELF-INFORMATION")
    print("=" * 78)

    probabilities = [1.0, 0.5, 0.25, 0.1, 0.01]
    for probability in probabilities:
        print(
            f"P={probability:>5}: "
            f"{self_information(probability):.4f} bits"
        )

    print("\nInterpretation:")
    print("- Certain events carry 0 bits of surprise.")
    print("- Rare events carry more information.")
    print("- Independent surprises add because logarithms turn products into sums.")


# ============================================================================
# 3. ENTROPY
# ============================================================================

def entropy(distribution: Dict[str, float], base: float = 2.0) -> float:
    """
    Shannon entropy:

        H(X) = -sum_x p(x) log_b p(x)

    Zero-probability outcomes contribute zero by convention.
    """
    validate_distribution(distribution)

    result = 0.0
    for probability in distribution.values():
        if probability > 0:
            result -= probability * math.log(probability, base)
    return result


def maximum_entropy(number_of_outcomes: int, base: float = 2.0) -> float:
    """Maximum entropy for a uniform distribution."""
    if number_of_outcomes <= 0:
        raise ValueError("Number of outcomes must be positive.")
    return math.log(number_of_outcomes, base)


def demonstrate_entropy() -> None:
    print("\n" + "=" * 78)
    print("2. SHANNON ENTROPY")
    print("=" * 78)

    fair_coin = {"heads": 0.5, "tails": 0.5}
    biased_coin = {"heads": 0.9, "tails": 0.1}
    deterministic = {"heads": 1.0, "tails": 0.0}
    die = {str(i): 1 / 6 for i in range(1, 7)}

    for name, distribution in [
        ("Fair coin", fair_coin),
        ("Biased coin", biased_coin),
        ("Deterministic coin", deterministic),
        ("Fair die", die),
    ]:
        value = entropy(distribution)
        print(f"{name:22}: {value:.6f} bits")

    print(f"\nMaximum entropy of a binary variable: {maximum_entropy(2):.6f} bits")
    print(f"Maximum entropy of a six-outcome variable: {maximum_entropy(6):.6f} bits")

    print(
        "\nEntropy measures expected uncertainty, not the information of one "
        "specific outcome. For n equally likely outcomes, entropy is log2(n)."
    )


# ============================================================================
# 4. CROSS-ENTROPY
# ============================================================================

def cross_entropy(
    true_distribution: Dict[str, float],
    predicted_distribution: Dict[str, float],
    base: float = 2.0,
) -> float:
    """
    Cross-entropy:

        H(P, Q) = -sum_x P(x) log_b Q(x)

    P is the true/reference distribution.
    Q is the model/predicted distribution.

    If P assigns positive probability where Q assigns zero probability,
    cross-entropy is infinite.
    """
    validate_distribution(true_distribution)
    validate_distribution(predicted_distribution)

    result = 0.0
    for outcome, true_probability in true_distribution.items():
        if true_probability == 0:
            continue

        predicted_probability = predicted_distribution.get(outcome, 0.0)

        if predicted_probability == 0:
            return math.inf

        result -= true_probability * math.log(predicted_probability, base)

    return result


def demonstrate_cross_entropy() -> None:
    print("\n" + "=" * 78)
    print("3. CROSS-ENTROPY")
    print("=" * 78)

    true_distribution = {"cat": 0.7, "dog": 0.3}
    good_model = {"cat": 0.68, "dog": 0.32}
    poor_model = {"cat": 0.2, "dog": 0.8}
    impossible_model = {"cat": 1.0, "dog": 0.0}

    true_entropy = entropy(true_distribution)
    print(f"True entropy:                 {true_entropy:.6f} bits")
    print(
        f"Good model cross-entropy:     "
        f"{cross_entropy(true_distribution, good_model):.6f} bits"
    )
    print(
        f"Poor model cross-entropy:     "
        f"{cross_entropy(true_distribution, poor_model):.6f} bits"
    )
    print(
        f"Impossible model cross-entropy:"
        f" {cross_entropy(true_distribution, impossible_model)}"
    )

    print(
        "\nCross-entropy is the expected coding cost when data follow P but "
        "we encode them using Q. It is widely used as a classification loss."
    )


# ============================================================================
# 5. KL DIVERGENCE
# ============================================================================

def kl_divergence(
    p: Dict[str, float],
    q: Dict[str, float],
    base: float = 2.0,
) -> float:
    """
    Kullback-Leibler divergence:

        D_KL(P || Q) = sum_x P(x) log_b(P(x) / Q(x))

    Important:
    - It is non-negative.
    - D_KL(P || Q) is zero iff P and Q agree on their support.
    - It is not symmetric.
    - It is not a distance metric.
    """
    validate_distribution(p)
    validate_distribution(q)

    result = 0.0
    for outcome, p_probability in p.items():
        if p_probability == 0:
            continue

        q_probability = q.get(outcome, 0.0)

        if q_probability == 0:
            return math.inf

        result += p_probability * math.log(
            p_probability / q_probability,
            base,
        )

    return result


def demonstrate_kl() -> None:
    print("\n" + "=" * 78)
    print("4. KL DIVERGENCE")
    print("=" * 78)

    p = {"A": 0.5, "B": 0.3, "C": 0.2}
    q = {"A": 0.4, "B": 0.4, "C": 0.2}
    r = {"A": 0.2, "B": 0.3, "C": 0.5}

    print(f"D_KL(P || Q): {kl_divergence(p, q):.6f} bits")
    print(f"D_KL(Q || P): {kl_divergence(q, p):.6f} bits")
    print(f"D_KL(P || R): {kl_divergence(p, r):.6f} bits")

    print(
        "\nThe first and second values need not match because KL divergence "
        "is directional."
    )


# ============================================================================
# 6. THE FUNDAMENTAL IDENTITY
# ============================================================================

def demonstrate_cross_entropy_identity() -> None:
    print("\n" + "=" * 78)
    print("5. CROSS-ENTROPY AND KL DIVERGENCE RELATIONSHIP")
    print("=" * 78)

    p = {"A": 0.5, "B": 0.3, "C": 0.2}
    q = {"A": 0.4, "B": 0.4, "C": 0.2}

    h_p = entropy(p)
    h_pq = cross_entropy(p, q)
    kl_pq = kl_divergence(p, q)

    print(f"H(P)       = {h_p:.10f}")
    print(f"H(P, Q)    = {h_pq:.10f}")
    print(f"D_KL(P||Q) = {kl_pq:.10f}")
    print(f"H(P)+D_KL  = {h_p + kl_pq:.10f}")

    print("\nIdentity:")
    print("H(P, Q) = H(P) + D_KL(P || Q)")


# ============================================================================
# 7. JOINT DISTRIBUTIONS
# ============================================================================

JointDistribution = Dict[Tuple[str, str], float]


def validate_joint_distribution(joint: JointDistribution) -> None:
    if not joint:
        raise ValueError("Joint distribution cannot be empty.")

    total = sum(joint.values())

    if any(p < 0 for p in joint.values()):
        raise ValueError("Joint probabilities cannot be negative.")

    if abs(total - 1.0) > 1e-9:
        raise ValueError("Joint probabilities must sum to 1.")


def marginal_x(joint: JointDistribution) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for (x, _), probability in joint.items():
        result[x] = result.get(x, 0.0) + probability
    return result


def marginal_y(joint: JointDistribution) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for (_, y), probability in joint.items():
        result[y] = result.get(y, 0.0) + probability
    return result


def conditional_distribution_y_given_x(
    joint: JointDistribution,
    x_value: str,
) -> Dict[str, float]:
    marginal = marginal_x(joint)
    denominator = marginal.get(x_value, 0.0)

    if denominator == 0:
        raise ValueError(f"P(X={x_value}) is zero.")

    result: Dict[str, float] = {}
    for (x, y), probability in joint.items():
        if x == x_value:
            result[y] = probability / denominator

    return result


def conditional_entropy_y_given_x(joint: JointDistribution) -> float:
    """
    H(Y|X) = sum_x P(x) H(Y|X=x)
    """
    validate_joint_distribution(joint)
    px = marginal_x(joint)

    result = 0.0
    for x_value, probability_x in px.items():
        conditional = conditional_distribution_y_given_x(joint, x_value)
        result += probability_x * entropy(conditional)

    return result


# ============================================================================
# 8. MUTUAL INFORMATION
# ============================================================================

def mutual_information(joint: JointDistribution) -> float:
    """
    Mutual information:

        I(X;Y) =
        sum_{x,y} P(x,y) log(P(x,y)/(P(x)P(y)))

    Equivalent identities:
        I(X;Y) = H(X) - H(X|Y)
        I(X;Y) = H(Y) - H(Y|X)
        I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    validate_joint_distribution(joint)

    px = marginal_x(joint)
    py = marginal_y(joint)

    result = 0.0
    for (x, y), probability_xy in joint.items():
        if probability_xy == 0:
            continue

        expected_independent_probability = px[x] * py[y]
        result += probability_xy * math.log(
            probability_xy / expected_independent_probability,
            2,
        )

    return result


def joint_entropy(joint: JointDistribution) -> float:
    return entropy(
        {str(key): probability for key, probability in joint.items()}
    )


def demonstrate_mutual_information() -> None:
    print("\n" + "=" * 78)
    print("6. MUTUAL INFORMATION")
    print("=" * 78)

    # X and Y are perfectly related: Y = X.
    correlated = {
        ("sunny", "sunny"): 0.5,
        ("rainy", "rainy"): 0.5,
    }

    # X and Y are independent.
    independent = {
        ("sunny", "sunny"): 0.25,
        ("sunny", "rainy"): 0.25,
        ("rainy", "sunny"): 0.25,
        ("rainy", "rainy"): 0.25,
    }

    for name, distribution in [
        ("Perfectly correlated", correlated),
        ("Independent", independent),
    ]:
        px = marginal_x(distribution)
        py = marginal_y(distribution)

        print(f"\n{name}")
        print(f"H(X)       = {entropy(px):.6f} bits")
        print(f"H(Y)       = {entropy(py):.6f} bits")
        print(f"H(Y|X)     = {conditional_entropy_y_given_x(distribution):.6f} bits")
        print(f"I(X;Y)     = {mutual_information(distribution):.6f} bits")


# ============================================================================
# 9. ENTROPY IDENTITIES AND INEQUALITIES
# ============================================================================

def demonstrate_identities() -> None:
    print("\n" + "=" * 78)
    print("7. IMPORTANT IDENTITIES")
    print("=" * 78)

    joint = {
        ("A", "0"): 0.30,
        ("A", "1"): 0.20,
        ("B", "0"): 0.10,
        ("B", "1"): 0.40,
    }

    px = marginal_x(joint)
    py = marginal_y(joint)

    hx = entropy(px)
    hy = entropy(py)
    hy_given_x = conditional_entropy_y_given_x(joint)
    mi = mutual_information(joint)

    print(f"H(X)          = {hx:.6f}")
    print(f"H(Y)          = {hy:.6f}")
    print(f"H(Y|X)        = {hy_given_x:.6f}")
    print(f"I(X;Y)        = {mi:.6f}")
    print(f"H(Y)-H(Y|X)  = {hy - hy_given_x:.6f}")

    print("\nKey inequalities:")
    print("0 <= I(X;Y) <= min(H(X), H(Y))")
    print("H(X|Y) <= H(X)")
    print("H(Y|X) <= H(Y)")
    print("D_KL(P||Q) >= 0")


# ============================================================================
# 10. NUMERICAL STABILITY
# ============================================================================

def stable_cross_entropy_from_logits(
    true_class_index: int,
    logits: Sequence[float],
) -> float:
    """
    Compute single-example cross-entropy directly from logits.

    Instead of:
        softmax(logits) -> probability -> log(probability)

    use the stable log-sum-exp expression:
        loss = log(sum(exp(z_i))) - z_true

    This avoids unnecessary overflow/underflow.
    """
    if not logits:
        raise ValueError("Logit sequence cannot be empty.")

    if not 0 <= true_class_index < len(logits):
        raise IndexError("True class index is out of range.")

    maximum = max(logits)

    log_sum_exp = maximum + math.log(
        sum(math.exp(logit - maximum) for logit in logits)
    )

    return log_sum_exp - logits[true_class_index]


def softmax(logits: Sequence[float]) -> List[float]:
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def demonstrate_numerical_stability() -> None:
    print("\n" + "=" * 78)
    print("8. NUMERICAL STABILITY AND LOGITS")
    print("=" * 78)

    logits = [1000.0, 999.0, 998.0]
    probabilities = softmax(logits)
    loss = stable_cross_entropy_from_logits(0, logits)

    print(f"Stable softmax: {probabilities}")
    print(f"Stable cross-entropy for class 0: {loss:.10f}")

    print(
        "\nSubtracting the maximum logit before exponentiation preserves the "
        "softmax result while preventing exponential overflow."
    )


# ============================================================================
# 11. MULTI-CLASS CLASSIFICATION
# ============================================================================

def classification_cross_entropy(
    true_labels: Sequence[int],
    predicted_probabilities: Sequence[Sequence[float]],
) -> float:
    """Mean categorical cross-entropy for one-hot/integer labels."""
    if len(true_labels) != len(predicted_probabilities):
        raise ValueError("Number of labels and predictions must match.")

    if not true_labels:
        raise ValueError("At least one example is required.")

    losses: List[float] = []

    for label, probabilities in zip(true_labels, predicted_probabilities):
        if not probabilities:
            raise ValueError("Each probability vector must be non-empty.")

        if not 0 <= label < len(probabilities):
            raise ValueError("Label is outside the probability vector.")

        total = sum(probabilities)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Each probability vector must sum to 1.")

        probability_of_true_class = probabilities[label]

        if probability_of_true_class <= 0:
            losses.append(math.inf)
        else:
            losses.append(-math.log(probability_of_true_class))

    if math.isinf(sum(losses)):
        return math.inf

    return sum(losses) / len(losses)


def demonstrate_classification_loss() -> None:
    print("\n" + "=" * 78)
    print("9. CLASSIFICATION CROSS-ENTROPY")
    print("=" * 78)

    labels = [0, 2, 1]

    predictions = [
        [0.80, 0.15, 0.05],
        [0.10, 0.20, 0.70],
        [0.20, 0.60, 0.20],
    ]

    loss = classification_cross_entropy(labels, predictions)

    print(f"Mean categorical cross-entropy: {loss:.6f} nats")
    print(
        "A confident correct prediction has low loss; a confident incorrect "
        "prediction has high loss."
    )


# ============================================================================
# 12. BINARY CROSS-ENTROPY
# ============================================================================

def binary_cross_entropy(
    actual_labels: Sequence[int],
    predicted_probabilities: Sequence[float],
) -> float:
    """
    Binary cross-entropy:

        -[y log(p) + (1-y) log(1-p)]

    Clipping is used only to prevent numerical log(0).
    """
    if len(actual_labels) != len(predicted_probabilities):
        raise ValueError("Input lengths must match.")

    if not actual_labels:
        raise ValueError("At least one observation is required.")

    losses = []

    for y, probability in zip(actual_labels, predicted_probabilities):
        if y not in (0, 1):
            raise ValueError("Binary labels must be 0 or 1.")

        if not 0 <= probability <= 1:
            raise ValueError("Predicted probability must be in [0, 1].")

        clipped = min(max(probability, EPSILON), 1 - EPSILON)

        losses.append(
            -(y * math.log(clipped) + (1 - y) * math.log(1 - clipped))
        )

    return sum(losses) / len(losses)


def demonstrate_binary_cross_entropy() -> None:
    print("\n" + "=" * 78)
    print("10. BINARY CROSS-ENTROPY")
    print("=" * 78)

    labels = [1, 0, 1, 1]
    predictions = [0.90, 0.10, 0.80, 0.70]

    print(f"Binary cross-entropy: {binary_cross_entropy(labels, predictions):.6f} nats")


# ============================================================================
# 13. EMPIRICAL ENTROPY FROM DATA
# ============================================================================

def empirical_distribution(values: Iterable[str]) -> Dict[str, float]:
    values = list(values)

    if not values:
        raise ValueError("Cannot estimate a distribution from empty data.")

    counts = Counter(values)
    total = len(values)

    return {
        value: count / total
        for value, count in counts.items()
    }


def demonstrate_empirical_entropy() -> None:
    print("\n" + "=" * 78)
    print("11. EMPIRICAL ENTROPY")
    print("=" * 78)

    observations = [
        "red",
        "red",
        "red",
        "blue",
        "blue",
        "green",
        "green",
        "green",
        "green",
    ]

    distribution = empirical_distribution(observations)

    print("Observed distribution:")
    for category, probability in distribution.items():
        print(f"  {category:>5}: {probability:.6f}")

    print(f"Empirical entropy: {entropy(distribution):.6f} bits")


# ============================================================================
# 14. INFORMATION GAIN
# ============================================================================

def information_gain(
    parent_labels: Sequence[str],
    child_groups: Sequence[Sequence[str]],
) -> float:
    """
    Information gain used in decision-tree learning:

        IG = H(parent) - weighted average child entropy
    """
    if not parent_labels:
        raise ValueError("Parent dataset cannot be empty.")

    if not child_groups:
        raise ValueError("At least one child group is required.")

    parent_distribution = empirical_distribution(parent_labels)
    parent_entropy = entropy(parent_distribution)

    total = len(parent_labels)
    weighted_child_entropy = 0.0

    for child in child_groups:
        if not child:
            continue

        child_distribution = empirical_distribution(child)
        weight = len(child) / total
        weighted_child_entropy += weight * entropy(child_distribution)

    return parent_entropy - weighted_child_entropy


def demonstrate_information_gain() -> None:
    print("\n" + "=" * 78)
    print("12. INFORMATION GAIN")
    print("=" * 78)

    parent = [
        "yes", "yes", "yes", "no", "no", "no", "yes", "no"
    ]

    child_one = ["yes", "yes", "yes", "yes"]
    child_two = ["no", "no", "no", "no"]

    gain = information_gain(parent, [child_one, child_two])

    print(f"Parent entropy: {entropy(empirical_distribution(parent)):.6f} bits")
    print(f"Information gain: {gain:.6f} bits")
    print(
        "A useful split produces child groups that are more predictable than "
        "the parent group."
    )


# ============================================================================
# 15. DISCRETE FEATURE SELECTION USING MUTUAL INFORMATION
# ============================================================================

def mutual_information_from_samples(
    x_values: Sequence[str],
    y_values: Sequence[str],
) -> float:
    if len(x_values) != len(y_values):
        raise ValueError("Sample sequences must have equal lengths.")

    if not x_values:
        raise ValueError("Samples cannot be empty.")

    counts = Counter(zip(x_values, y_values))
    total = len(x_values)

    joint = {
        pair: count / total
        for pair, count in counts.items()
    }

    return mutual_information(joint)


def demonstrate_feature_selection() -> None:
    print("\n" + "=" * 78)
    print("13. MUTUAL INFORMATION FOR FEATURE RELEVANCE")
    print("=" * 78)

    feature = ["A", "A", "B", "B", "A", "B", "A", "B"]
    target = ["0", "0", "1", "1", "0", "1", "0", "1"]

    irrelevant_feature = [
        "X", "Y", "X", "Y", "Y", "X", "X", "Y"
    ]

    print(
        f"MI(relevant-looking feature, target): "
        f"{mutual_information_from_samples(feature, target):.6f} bits"
    )

    print(
        f"MI(independent-looking feature, target): "
        f"{mutual_information_from_samples(irrelevant_feature, target):.6f} bits"
    )


# ============================================================================
# 16. CODING INTERPRETATION
# ============================================================================

def expected_code_length(
    distribution: Dict[str, float],
    code_lengths: Dict[str, float],
) -> float:
    validate_distribution(distribution)

    for symbol in distribution:
        if symbol not in code_lengths:
            raise ValueError(f"Missing code length for {symbol!r}.")
        if code_lengths[symbol] < 0:
            raise ValueError("Code lengths cannot be negative.")

    return sum(
        distribution[symbol] * code_lengths[symbol]
        for symbol in distribution
    )


def demonstrate_coding_interpretation() -> None:
    print("\n" + "=" * 78)
    print("14. ENTROPY AS A CODING LIMIT")
    print("=" * 78)

    distribution = {
        "A": 0.5,
        "B": 0.25,
        "C": 0.125,
        "D": 0.125,
    }

    code_lengths = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 3,
    }

    h = entropy(distribution)
    average_length = expected_code_length(distribution, code_lengths)

    print(f"Entropy:               {h:.6f} bits/symbol")
    print(f"Average code length:   {average_length:.6f} bits/symbol")
    print(
        "Entropy establishes a fundamental lower bound for ideal lossless "
        "average coding under the stated probabilistic model."
    )


# ============================================================================
# 17. DATA PROCESSING AND ESTIMATION EDGE CASES
# ============================================================================

def demonstrate_edge_cases() -> None:
    print("\n" + "=" * 78)
    print("15. EDGE CASES")
    print("=" * 78)

    deterministic = {"one": 1.0, "zero": 0.0}
    print(
        "Entropy with a zero-probability outcome:",
        entropy(deterministic),
    )

    p = {"A": 1.0, "B": 0.0}
    q = {"A": 0.999, "B": 0.001}
    print("KL with compatible support:", kl_divergence(p, q))

    q_impossible = {"A": 0.0, "B": 1.0}
    print("KL with impossible true event:", kl_divergence(p, q_impossible))

    print(
        "Cross-entropy with an impossible predicted true event:",
        cross_entropy(p, q_impossible),
    )


# ============================================================================
# 18. MONTE CARLO INTUITION
# ============================================================================

def sample_from_distribution(
    distribution: Dict[str, float],
    count: int,
) -> List[str]:
    validate_distribution(distribution)

    if count <= 0:
        raise ValueError("Sample count must be positive.")

    outcomes = list(distribution)
    probabilities = [distribution[outcome] for outcome in outcomes]

    return random.choices(outcomes, weights=probabilities, k=count)


def demonstrate_sampling() -> None:
    print("\n" + "=" * 78)
    print("16. SAMPLING AND EMPIRICAL ENTROPY")
    print("=" * 78)

    random.seed(42)

    theoretical = {
        "A": 0.7,
        "B": 0.2,
        "C": 0.1,
    }

    sample = sample_from_distribution(theoretical, 10_000)
    estimated = empirical_distribution(sample)

    print(f"Theoretical entropy: {entropy(theoretical):.6f} bits")
    print(f"Estimated entropy:   {entropy(estimated):.6f} bits")
    print("Finite samples produce estimation error.")


# ============================================================================
# 19. NORMALIZED MUTUAL INFORMATION
# ============================================================================

def normalized_mutual_information(joint: JointDistribution) -> float:
    """
    One common normalization:

        I(X;Y) / sqrt(H(X) H(Y))

    This is useful for comparison but is not the only normalization.
    """
    px = marginal_x(joint)
    py = marginal_y(joint)

    hx = entropy(px)
    hy = entropy(py)

    if hx == 0 or hy == 0:
        return 0.0

    return mutual_information(joint) / math.sqrt(hx * hy)


def demonstrate_normalized_mi() -> None:
    print("\n" + "=" * 78)
    print("17. NORMALIZED MUTUAL INFORMATION")
    print("=" * 78)

    joint = {
        ("A", "A"): 0.4,
        ("A", "B"): 0.1,
        ("B", "A"): 0.1,
        ("B", "B"): 0.4,
    }

    print(f"Mutual information:            {mutual_information(joint):.6f}")
    print(
        f"Normalized mutual information: "
        f"{normalized_mutual_information(joint):.6f}"
    )


# ============================================================================
# 20. PRACTICAL LANGUAGE MODELING EXAMPLE
# ============================================================================

def language_model_perplexity(cross_entropy_nats: float) -> float:
    """
    Perplexity = exp(cross-entropy in nats).

    If cross-entropy is measured in bits:
        perplexity = 2 ** cross_entropy_bits
    """
    if cross_entropy_nats < 0:
        raise ValueError("Cross-entropy cannot be negative.")
    return math.exp(cross_entropy_nats)


def demonstrate_perplexity() -> None:
    print("\n" + "=" * 78)
    print("18. CROSS-ENTROPY AND PERPLEXITY")
    print("=" * 78)

    token_distribution = {
        "the": 0.50,
        "cat": 0.25,
        "sat": 0.25,
    }

    model_distribution = {
        "the": 0.60,
        "cat": 0.20,
        "sat": 0.20,
    }

    h_bits = cross_entropy(token_distribution, model_distribution)
    h_nats = cross_entropy(
        token_distribution,
        model_distribution,
        base=math.e,
    )

    print(f"Cross-entropy: {h_bits:.6f} bits")
    print(f"Cross-entropy: {h_nats:.6f} nats")
    print(f"Perplexity:    {language_model_perplexity(h_nats):.6f}")


# ============================================================================
# 21. COMPARISON TABLE
# ============================================================================

def print_comparison() -> None:
    print("\n" + "=" * 78)
    print("19. CONCEPT COMPARISON")
    print("=" * 78)

    rows = [
        ("Entropy", "One distribution", "Uncertainty", "H(P) >= 0"),
        (
            "Cross-entropy",
            "P and Q",
            "Expected coding/prediction cost",
            ">= H(P)",
        ),
        (
            "KL divergence",
            "P and Q",
            "Mismatch/information loss",
            ">= 0, directional",
        ),
        (
            "Mutual information",
            "Joint P(X,Y)",
            "Dependence/shared information",
            ">= 0, symmetric",
        ),
    ]

    for concept, inputs, interpretation, property in rows:
        print(
            f"{concept:20} | "
            f"{inputs:18} | "
            f"{interpretation:34} | "
            f"{property}"
        )


# ============================================================================
# 22. STUDY CHECKS
# ============================================================================

def run_assertions() -> None:
    """Executable correctness checks for the main mathematical identities."""
    fair_coin = {"H": 0.5, "T": 0.5}

    assert abs(entropy(fair_coin) - 1.0) < 1e-12
    assert abs(kl_divergence(fair_coin, fair_coin)) < 1e-12

    independent = {
        ("0", "0"): 0.25,
        ("0", "1"): 0.25,
        ("1", "0"): 0.25,
        ("1", "1"): 0.25,
    }

    assert abs(mutual_information(independent)) < 1e-12

    correlated = {
        ("0", "0"): 0.5,
        ("1", "1"): 0.5,
    }

    assert abs(mutual_information(correlated) - 1.0) < 1e-12

    p = {"A": 0.5, "B": 0.5}
    q = {"A": 0.25, "B": 0.75}

    identity_error = abs(
        cross_entropy(p, q) -
        (entropy(p) + kl_divergence(p, q))
    )

    assert identity_error < 1e-12


# ============================================================================
# 23. ADVANCED CONCEPTS: CONDITIONAL MUTUAL INFORMATION
# ============================================================================

def conditional_mutual_information(
    joint_xyz: Dict[Tuple[str, str, str], float],
) -> float:
    """
    Conditional mutual information:

        I(X;Y|Z)
        = sum_xyz P(x,y,z)
          log [P(x,y|z) / (P(x|z)P(y|z))]

    It measures dependence between X and Y after conditioning on Z.
    """
    if not joint_xyz:
        raise ValueError("Distribution cannot be empty.")

    total = sum(joint_xyz.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Joint probabilities must sum to 1.")

    pz: Dict[str, float] = {}
    pxz: Dict[Tuple[str, str], float] = {}
    pyz: Dict[Tuple[str, str], float] = {}

    for (x, y, z), probability in joint_xyz.items():
        pz[z] = pz.get(z, 0.0) + probability
        pxz[(x, z)] = pxz.get((x, z), 0.0) + probability
        pyz[(y, z)] = pyz.get((y, z), 0.0) + probability

    result = 0.0

    for (x, y, z), probability_xyz in joint_xyz.items():
        if probability_xyz == 0:
            continue

        probability_xy_given_z = probability_xyz / pz[z]
        probability_x_given_z = pxz[(x, z)] / pz[z]
        probability_y_given_z = pyz[(y, z)] / pz[z]

        result += probability_xyz * math.log(
            probability_xy_given_z /
            (probability_x_given_z * probability_y_given_z),
            2,
        )

    return result


def demonstrate_conditional_mutual_information() -> None:
    print("\n" + "=" * 78)
    print("20. CONDITIONAL MUTUAL INFORMATION")
    print("=" * 78)

    # A simple distribution with Z selecting one of two environments.
    joint = {
        ("0", "0", "A"): 0.25,
        ("0", "1", "A"): 0.00,
        ("1", "0", "A"): 0.00,
        ("1", "1", "A"): 0.25,
        ("0", "0", "B"): 0.125,
        ("0", "1", "B"): 0.125,
        ("1", "0", "B"): 0.125,
        ("1", "1", "B"): 0.125,
    }

    print(
        "I(X;Y|Z): "
        f"{conditional_mutual_information(joint):.6f} bits"
    )


# ============================================================================
# 24. DATA-DRIVEN KL COMPARISON
# ============================================================================

@dataclass
class DistributionComparison:
    reference: Dict[str, float]
    candidate: Dict[str, float]

    @property
    def reference_entropy(self) -> float:
        return entropy(self.reference)

    @property
    def cross_entropy(self) -> float:
        return cross_entropy(self.reference, self.candidate)

    @property
    def kl_divergence(self) -> float:
        return kl_divergence(self.reference, self.candidate)

    def report(self) -> str:
        return (
            f"H(P) = {self.reference_entropy:.6f} bits\n"
            f"H(P,Q) = {self.cross_entropy:.6f} bits\n"
            f"D_KL(P||Q) = {self.kl_divergence:.6f} bits"
        )


def demonstrate_class_based_design() -> None:
    print("\n" + "=" * 78)
    print("21. CLASS-BASED INFORMATION-THEORY ANALYSIS")
    print("=" * 78)

    comparison = DistributionComparison(
        reference={"low": 0.6, "medium": 0.3, "high": 0.1},
        candidate={"low": 0.5, "medium": 0.4, "high": 0.1},
    )

    print(comparison.report())


# ============================================================================
# 25. MAIN STUDY PROGRAM
# ============================================================================

def main() -> None:
    demonstrate_self_information()
    demonstrate_entropy()
    demonstrate_cross_entropy()
    demonstrate_kl()
    demonstrate_cross_entropy_identity()
    demonstrate_mutual_information()
    demonstrate_identities()
    demonstrate_numerical_stability()
    demonstrate_classification_loss()
    demonstrate_binary_cross_entropy()
    demonstrate_empirical_entropy()
    demonstrate_information_gain()
    demonstrate_feature_selection()
    demonstrate_coding_interpretation()
    demonstrate_edge_cases()
    demonstrate_sampling()
    demonstrate_normalized_mi()
    demonstrate_perplexity()
    print_comparison()
    demonstrate_conditional_mutual_information()
    demonstrate_class_based_design()

    run_assertions()

    print("\n" + "=" * 78)
    print("EXECUTABLE CHECKS PASSED")
    print("=" * 78)
    print("Core entropy, cross-entropy, KL-divergence, and mutual-information")
    print("identities passed their numerical validation checks.")


if __name__ == "__main__":
    main()
