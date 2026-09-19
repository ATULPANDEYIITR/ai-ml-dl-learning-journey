"""
Bayes' Theorem: From Conditional Probability to Bayesian Inference

This standalone script teaches Bayes' theorem through executable examples.
It progresses from basic probability concepts to likelihood, priors, posterior
probabilities, diagnostic testing, medical screening, spam detection, Naive
Bayes classification, sequential Bayesian updating, odds form, continuous
distributions, numerical stability, decision thresholds, and practical
implementation concerns.

No third-party packages are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt, pi
from typing import Dict, Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# 1. BASIC PROBABILITY TERMINOLOGY
# ---------------------------------------------------------------------------

def probability(event_count: float, total_count: float) -> float:
    """Compute empirical probability P(A) = favorable outcomes / total."""
    if total_count <= 0:
        raise ValueError("Total count must be positive.")
    if event_count < 0 or event_count > total_count:
        raise ValueError("Event count must be between 0 and total count.")
    return event_count / total_count


def complement(p: float) -> float:
    """P(not A) = 1 - P(A)."""
    validate_probability(p)
    return 1.0 - p


def validate_probability(p: float, name: str = "probability") -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def validate_positive_probability(p: float, name: str = "probability") -> None:
    if not 0.0 < p <= 1.0:
        raise ValueError(f"{name} must be greater than 0 and at most 1.")


def intersection_from_conditional(
    p_a: float,
    p_b_given_a: float,
) -> float:
    """
    Multiplication rule:
        P(A and B) = P(A) P(B | A)
    """
    validate_probability(p_a, "P(A)")
    validate_probability(p_b_given_a, "P(B | A)")
    return p_a * p_b_given_a


# ---------------------------------------------------------------------------
# 2. CONDITIONAL PROBABILITY
# ---------------------------------------------------------------------------

def conditional_probability(
    p_a_and_b: float,
    p_b: float,
) -> float:
    """
    Definition:
        P(A | B) = P(A and B) / P(B)
    """
    validate_probability(p_a_and_b, "P(A and B)")
    validate_positive_probability(p_b, "P(B)")
    if p_a_and_b > p_b:
        raise ValueError("P(A and B) cannot exceed P(B).")
    return p_a_and_b / p_b


def demonstrate_conditional_probability() -> None:
    print("\n=== CONDITIONAL PROBABILITY ===")

    # Suppose 40% of customers are premium customers.
    # Among premium customers, 70% use a particular service.
    p_premium = 0.40
    p_service_given_premium = 0.70

    joint = intersection_from_conditional(
        p_premium,
        p_service_given_premium,
    )

    # P(premium AND service) = 0.40 * 0.70 = 0.28
    print(f"P(Premium) = {p_premium:.2f}")
    print(f"P(Service | Premium) = {p_service_given_premium:.2f}")
    print(f"P(Premium AND Service) = {joint:.2f}")

    # Recover the conditional probability from the joint probability.
    recovered = conditional_probability(joint, p_premium)
    print(f"P(Service | Premium), recovered = {recovered:.2f}")


# ---------------------------------------------------------------------------
# 3. TOTAL PROBABILITY
# ---------------------------------------------------------------------------

def total_probability(
    conditional_probabilities: Sequence[float],
    prior_probabilities: Sequence[float],
) -> float:
    """
    Law of total probability:

        P(B) = sum_i P(B | A_i) P(A_i)

    The A_i events must form a partition of the sample space.
    """
    if len(conditional_probabilities) != len(prior_probabilities):
        raise ValueError("Both sequences must have equal length.")

    for p in conditional_probabilities:
        validate_probability(p, "conditional probability")

    for p in prior_probabilities:
        validate_probability(p, "prior probability")

    if not prior_probabilities:
        raise ValueError("At least one hypothesis is required.")

    if abs(sum(prior_probabilities) - 1.0) > 1e-9:
        raise ValueError("Prior probabilities must sum to 1.")

    return sum(
        likelihood * prior
        for likelihood, prior in zip(
            conditional_probabilities,
            prior_probabilities,
        )
    )


# ---------------------------------------------------------------------------
# 4. BAYES' THEOREM
# ---------------------------------------------------------------------------

def bayes_theorem(
    prior: float,
    likelihood: float,
    evidence_probability: float,
) -> float:
    """
    Bayes' theorem:

        P(H | E) = P(E | H) P(H) / P(E)

    H = hypothesis
    E = evidence

    prior      = P(H)
    likelihood = P(E | H)
    evidence   = P(E)
    posterior  = P(H | E)
    """
    validate_probability(prior, "prior")
    validate_probability(likelihood, "likelihood")
    validate_positive_probability(evidence_probability, "P(E)")

    numerator = likelihood * prior
    posterior = numerator / evidence_probability

    if posterior > 1.0 + 1e-12:
        raise ValueError(
            "Inputs are inconsistent: posterior probability exceeds 1."
        )

    return min(posterior, 1.0)


def bayes_binary(
    prior_h: float,
    likelihood_e_given_h: float,
    likelihood_e_given_not_h: float,
) -> Tuple[float, float]:
    """
    Binary Bayes calculation.

    P(E) = P(E|H)P(H) + P(E|not H)P(not H)

    Then:
        P(H|E) = P(E|H)P(H) / P(E)
    """
    validate_probability(prior_h, "P(H)")
    validate_probability(likelihood_e_given_h, "P(E | H)")
    validate_probability(likelihood_e_given_not_h, "P(E | not H)")

    prior_not_h = 1.0 - prior_h

    evidence = (
        likelihood_e_given_h * prior_h
        + likelihood_e_given_not_h * prior_not_h
    )

    if evidence <= 0.0:
        raise ValueError("Evidence probability is zero.")

    posterior_h = (
        likelihood_e_given_h * prior_h
    ) / evidence

    return posterior_h, evidence


def demonstrate_bayes_basic() -> None:
    print("\n=== BASIC BAYES' THEOREM ===")

    prior = 0.10
    likelihood = 0.80
    false_positive_rate = 0.05

    posterior, evidence = bayes_binary(
        prior,
        likelihood,
        false_positive_rate,
    )

    print(f"Prior P(H) = {prior:.4f}")
    print(f"Likelihood P(E | H) = {likelihood:.4f}")
    print(f"P(E | not H) = {false_positive_rate:.4f}")
    print(f"Evidence P(E) = {evidence:.4f}")
    print(f"Posterior P(H | E) = {posterior:.4f}")
    print(f"Posterior percentage = {posterior * 100:.2f}%")


# ---------------------------------------------------------------------------
# 5. WHY THE DENOMINATOR MATTERS
# ---------------------------------------------------------------------------

def demonstrate_denominator() -> None:
    print("\n=== WHY THE DENOMINATOR MATTERS ===")

    # A positive test is not interpreted using sensitivity alone.
    # The denominator P(positive) includes both true positives and false
    # positives.
    prevalence = 0.01
    sensitivity = 0.99
    false_positive_rate = 0.05

    true_positive_mass = prevalence * sensitivity
    false_positive_mass = (
        (1.0 - prevalence) * false_positive_rate
    )
    positive_probability = true_positive_mass + false_positive_mass

    posterior = true_positive_mass / positive_probability

    print(f"True-positive probability mass = {true_positive_mass:.6f}")
    print(f"False-positive probability mass = {false_positive_mass:.6f}")
    print(f"All-positive probability = {positive_probability:.6f}")
    print(f"P(Disease | Positive) = {posterior:.4%}")


# ---------------------------------------------------------------------------
# 6. FREQUENCY TABLE INTERPRETATION
# ---------------------------------------------------------------------------

def demonstrate_frequency_table() -> None:
    print("\n=== FREQUENCY-TABLE INTERPRETATION ===")

    population = 100_000
    disease_rate = 0.01
    sensitivity = 0.99
    false_positive_rate = 0.05

    diseased = int(population * disease_rate)
    healthy = population - diseased

    true_positives = round(diseased * sensitivity)
    false_negatives = diseased - true_positives
    false_positives = round(healthy * false_positive_rate)
    true_negatives = healthy - false_positives

    positive_tests = true_positives + false_positives

    print(f"Population: {population:,}")
    print(f"Diseased: {diseased:,}")
    print(f"Healthy: {healthy:,}")
    print(f"True positives: {true_positives:,}")
    print(f"False positives: {false_positives:,}")
    print(f"False negatives: {false_negatives:,}")
    print(f"True negatives: {true_negatives:,}")
    print(
        "Probability of disease given positive test: "
        f"{true_positives / positive_tests:.2%}"
    )


# ---------------------------------------------------------------------------
# 7. PRIORS, LIKELIHOODS, EVIDENCE, POSTERIORS
# ---------------------------------------------------------------------------

@dataclass
class BayesianTerms:
    hypothesis: str
    prior: float
    likelihood: float
    unnormalized_posterior: float = 0.0
    posterior: float = 0.0


def posterior_distribution(
    hypotheses: Sequence[str],
    priors: Sequence[float],
    likelihoods: Sequence[float],
) -> List[BayesianTerms]:
    """
    Compute a normalized posterior over mutually exclusive hypotheses.

    posterior_i is proportional to:
        prior_i * P(E | H_i)
    """
    if not (
        len(hypotheses)
        == len(priors)
        == len(likelihoods)
    ):
        raise ValueError("All inputs must have equal length.")

    if not hypotheses:
        raise ValueError("At least one hypothesis is required.")

    if abs(sum(priors) - 1.0) > 1e-9:
        raise ValueError("Priors must sum to 1.")

    for prior in priors:
        validate_probability(prior, "prior")

    for likelihood in likelihoods:
        validate_probability(likelihood, "likelihood")

    terms = [
        BayesianTerms(
            hypothesis=h,
            prior=p,
            likelihood=l,
            unnormalized_posterior=p * l,
        )
        for h, p, l in zip(hypotheses, priors, likelihoods)
    ]

    evidence = sum(
        term.unnormalized_posterior
        for term in terms
    )

    if evidence <= 0.0:
        raise ValueError("Evidence is zero; posterior is undefined.")

    for term in terms:
        term.posterior = term.unnormalized_posterior / evidence

    return terms


def print_posterior_distribution(
    terms: Sequence[BayesianTerms],
) -> None:
    print(
        f"{'Hypothesis':<20}"
        f"{'Prior':>12}"
        f"{'Likelihood':>14}"
        f"{'Posterior':>14}"
    )

    for term in terms:
        print(
            f"{term.hypothesis:<20}"
            f"{term.prior:>12.4f}"
            f"{term.likelihood:>14.4f}"
            f"{term.posterior:>14.4f}"
        )


def demonstrate_multiple_hypotheses() -> None:
    print("\n=== MULTIPLE HYPOTHESES ===")

    hypotheses = [
        "Server A",
        "Server B",
        "Server C",
    ]

    priors = [
        0.50,
        0.30,
        0.20,
    ]

    # Probability of observing the evidence under each server hypothesis.
    likelihoods = [
        0.20,
        0.70,
        0.40,
    ]

    terms = posterior_distribution(
        hypotheses,
        priors,
        likelihoods,
    )

    print_posterior_distribution(terms)


# ---------------------------------------------------------------------------
# 8. ODDS FORM OF BAYES' THEOREM
# ---------------------------------------------------------------------------

def probability_to_odds(p: float) -> float:
    validate_probability(p)
    if p == 1.0:
        return float("inf")
    if p == 0.0:
        return 0.0
    return p / (1.0 - p)


def odds_to_probability(odds: float) -> float:
    if odds < 0:
        raise ValueError("Odds cannot be negative.")
    if odds == float("inf"):
        return 1.0
    return odds / (1.0 + odds)


def bayes_odds(
    prior_probability: float,
    likelihood_ratio: float,
) -> float:
    """
    Odds form:

        Posterior odds = Prior odds * Likelihood ratio

    where:
        LR = P(E|H) / P(E|not H)
    """
    if likelihood_ratio < 0:
        raise ValueError("Likelihood ratio cannot be negative.")

    prior_odds = probability_to_odds(prior_probability)
    return prior_odds * likelihood_ratio


def demonstrate_odds_form() -> None:
    print("\n=== ODDS FORM ===")

    prior = 0.10
    p_e_given_h = 0.80
    p_e_given_not_h = 0.05

    likelihood_ratio = p_e_given_h / p_e_given_not_h
    posterior_odds = bayes_odds(
        prior,
        likelihood_ratio,
    )
    posterior = odds_to_probability(posterior_odds)

    print(f"Prior probability: {prior:.4f}")
    print(f"Prior odds: {probability_to_odds(prior):.4f}")
    print(f"Likelihood ratio: {likelihood_ratio:.4f}")
    print(f"Posterior odds: {posterior_odds:.4f}")
    print(f"Posterior probability: {posterior:.4f}")


# ---------------------------------------------------------------------------
# 9. SEQUENTIAL BAYESIAN UPDATING
# ---------------------------------------------------------------------------

def sequential_update(
    prior: float,
    likelihood_of_positive_evidence: float,
    likelihood_of_negative_hypothesis_evidence: float,
    observations: Iterable[bool],
) -> List[float]:
    """
    Update P(H) repeatedly as independent observations arrive.

    For every observation:
        if positive:
            use P(E|H) and P(E|not H)
        if negative:
            use:
                P(not E|H) = 1 - P(E|H)
                P(not E|not H) = 1 - P(E|not H)

    The independence assumption is important. Reusing correlated evidence
    as if it were independent can produce overconfident posteriors.
    """
    posterior = prior
    history = [posterior]

    for observation in observations:
        if observation:
            h_likelihood = likelihood_of_positive_evidence
            not_h_likelihood = likelihood_of_negative_hypothesis_evidence
        else:
            h_likelihood = 1.0 - likelihood_of_positive_evidence
            not_h_likelihood = 1.0 - likelihood_of_negative_hypothesis_evidence

        posterior, _ = bayes_binary(
            posterior,
            h_likelihood,
            not_h_likelihood,
        )
        history.append(posterior)

    return history


def demonstrate_sequential_updating() -> None:
    print("\n=== SEQUENTIAL BAYESIAN UPDATING ===")

    prior = 0.10
    sensitivity = 0.80
    false_positive_rate = 0.05

    observations = [True, True, False, True]

    history = sequential_update(
        prior,
        sensitivity,
        false_positive_rate,
        observations,
    )

    for index, posterior in enumerate(history):
        print(
            f"After {index} observation(s): "
            f"P(H) = {posterior:.4%}"
        )


# ---------------------------------------------------------------------------
# 10. BAYESIAN DIAGNOSTIC MODEL
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticTest:
    name: str
    sensitivity: float
    specificity: float

    def __post_init__(self) -> None:
        validate_probability(
            self.sensitivity,
            "sensitivity",
        )
        validate_probability(
            self.specificity,
            "specificity",
        )

    @property
    def false_positive_rate(self) -> float:
        return 1.0 - self.specificity

    @property
    def false_negative_rate(self) -> float:
        return 1.0 - self.sensitivity

    def positive_predictive_value(self, prevalence: float) -> float:
        validate_probability(prevalence, "prevalence")

        numerator = self.sensitivity * prevalence
        denominator = (
            self.sensitivity * prevalence
            + self.false_positive_rate * (1.0 - prevalence)
        )

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def negative_predictive_value(self, prevalence: float) -> float:
        validate_probability(prevalence, "prevalence")

        numerator = (
            self.specificity * (1.0 - prevalence)
        )
        denominator = (
            self.specificity * (1.0 - prevalence)
            + self.false_negative_rate * prevalence
        )

        if denominator == 0:
            return 0.0

        return numerator / denominator


def demonstrate_diagnostic_testing() -> None:
    print("\n=== DIAGNOSTIC TESTING ===")

    test = DiagnosticTest(
        name="Screening Test",
        sensitivity=0.95,
        specificity=0.90,
    )

    prevalence = 0.02

    print(f"Test: {test.name}")
    print(f"Sensitivity: {test.sensitivity:.2%}")
    print(f"Specificity: {test.specificity:.2%}")
    print(f"Prevalence: {prevalence:.2%}")
    print(
        f"Positive predictive value: "
        f"{test.positive_predictive_value(prevalence):.2%}"
    )
    print(
        f"Negative predictive value: "
        f"{test.negative_predictive_value(prevalence):.2%}"
    )


# ---------------------------------------------------------------------------
# 11. NAIVE BAYES CLASSIFIER
# ---------------------------------------------------------------------------

class NaiveBayesTextClassifier:
    """
    Small multinomial Naive Bayes classifier.

    It demonstrates the conditional-independence approximation:

        P(class | words) proportional to
        P(class) * product P(word | class)

    Real implementations commonly use logarithms because multiplying many
    probabilities can underflow toward zero.
    """

    def __init__(self) -> None:
        self.class_document_counts: Dict[str, int] = {}
        self.word_counts: Dict[str, Dict[str, int]] = {}
        self.total_word_counts: Dict[str, int] = {}
        self.vocabulary: set[str] = set()
        self.total_documents = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [
            word.strip(".,!?;:()[]{}\"'").lower()
            for word in text.split()
            if word.strip(".,!?;:()[]{}\"'")
        ]

    def fit(
        self,
        documents: Sequence[Tuple[str, str]],
    ) -> None:
        if not documents:
            raise ValueError("Training data cannot be empty.")

        for label, text in documents:
            self.total_documents += 1
            self.class_document_counts[label] = (
                self.class_document_counts.get(label, 0) + 1
            )

            if label not in self.word_counts:
                self.word_counts[label] = {}
                self.total_word_counts[label] = 0

            for word in self.tokenize(text):
                self.vocabulary.add(word)
                self.word_counts[label][word] = (
                    self.word_counts[label].get(word, 0) + 1
                )
                self.total_word_counts[label] += 1

    def log_prior(self, label: str) -> float:
        count = self.class_document_counts[label]
        return log(count / self.total_documents)

    def log_word_probability(
        self,
        word: str,
        label: str,
    ) -> float:
        """
        Laplace smoothing:

            P(word|class) =
            (count(word,class) + 1) /
            (total words in class + vocabulary size)
        """
        vocabulary_size = len(self.vocabulary)

        numerator = (
            self.word_counts[label].get(word, 0) + 1
        )
        denominator = (
            self.total_word_counts[label]
            + vocabulary_size
        )

        return log(numerator / denominator)

    def predict_scores(
        self,
        text: str,
    ) -> Dict[str, float]:
        words = self.tokenize(text)

        scores: Dict[str, float] = {}

        for label in self.class_document_counts:
            score = self.log_prior(label)

            for word in words:
                if word in self.vocabulary:
                    score += self.log_word_probability(
                        word,
                        label,
                    )

            scores[label] = score

        return scores

    def predict(self, text: str) -> str:
        scores = self.predict_scores(text)
        return max(scores, key=scores.get)


def demonstrate_naive_bayes() -> None:
    print("\n=== NAIVE BAYES TEXT CLASSIFICATION ===")

    training_data = [
        ("spam", "free prize win money"),
        ("spam", "claim free money now"),
        ("spam", "win free prize"),
        ("ham", "project meeting tomorrow"),
        ("ham", "please review project report"),
        ("ham", "meeting schedule report"),
    ]

    classifier = NaiveBayesTextClassifier()
    classifier.fit(training_data)

    messages = [
        "free money prize",
        "project meeting",
        "free project",
        "report meeting",
    ]

    for message in messages:
        prediction = classifier.predict(message)
        scores = classifier.predict_scores(message)

        print(f"Message: {message!r}")
        print(f"Prediction: {prediction}")
        print(f"Log scores: {scores}")


# ---------------------------------------------------------------------------
# 12. NUMERICAL STABILITY
# ---------------------------------------------------------------------------

def stable_log_sum_exp(values: Sequence[float]) -> float:
    """
    Stable calculation of log(sum(exp(x_i))).

    Direct exponentiation may overflow for large values.
    Subtracting the maximum keeps exponentials in a manageable range.
    """
    if not values:
        raise ValueError("At least one value is required.")

    maximum = max(values)

    if maximum == float("-inf"):
        return maximum

    return maximum + log(
        sum(exp(value - maximum) for value in values)
    )


def demonstrate_log_space() -> None:
    print("\n=== NUMERICAL STABILITY ===")

    tiny_probability = 1e-300

    direct_product = tiny_probability * tiny_probability
    log_product = log(tiny_probability) + log(tiny_probability)

    print(f"Direct product: {direct_product:.3e}")
    print(f"Log product: {log_product:.3f}")
    print(
        "Logarithms are useful because products become sums and extremely "
        "small probabilities can be represented more safely."
    )

    log_values = [
        -1000.0,
        -1001.0,
        -1005.0,
    ]

    print(
        "Stable log-sum-exp:",
        stable_log_sum_exp(log_values),
    )


# ---------------------------------------------------------------------------
# 13. CONTINUOUS BAYESIAN MODEL
# ---------------------------------------------------------------------------

def normal_pdf(
    x: float,
    mean: float,
    standard_deviation: float,
) -> float:
    """Probability density of a normal distribution."""
    if standard_deviation <= 0:
        raise ValueError("Standard deviation must be positive.")

    coefficient = (
        1.0
        / (standard_deviation * sqrt(2.0 * pi))
    )

    exponent = -0.5 * (
        (x - mean) / standard_deviation
    ) ** 2

    return coefficient * exp(exponent)


def gaussian_hypothesis_posterior(
    observation: float,
    hypotheses: Sequence[str],
    priors: Sequence[float],
    means: Sequence[float],
    standard_deviations: Sequence[float],
) -> List[BayesianTerms]:
    """
    Bayesian inference with continuous observations.

    For continuous variables, a probability density is used in the
    likelihood term rather than assigning a probability to an exact
    real-valued point.
    """
    if not (
        len(hypotheses)
        == len(priors)
        == len(means)
        == len(standard_deviations)
    ):
        raise ValueError("All sequences must have equal length.")

    likelihoods = [
        normal_pdf(observation, mean, std)
        for mean, std in zip(
            means,
            standard_deviations,
        )
    ]

    # posterior_distribution expects likelihood values in [0, 1], while
    # a continuous probability density can exceed 1. Therefore normalize
    # manually using Bayes' proportionality rule.
    unnormalized = [
        prior * likelihood
        for prior, likelihood in zip(priors, likelihoods)
    ]

    evidence = sum(unnormalized)

    if evidence <= 0:
        raise ValueError("Evidence is zero.")

    return [
        BayesianTerms(
            hypothesis=h,
            prior=prior,
            likelihood=likelihood,
            unnormalized_posterior=value,
            posterior=value / evidence,
        )
        for h, prior, likelihood, value in zip(
            hypotheses,
            priors,
            likelihoods,
            unnormalized,
        )
    ]


def demonstrate_continuous_bayes() -> None:
    print("\n=== CONTINUOUS OBSERVATIONS ===")

    terms = gaussian_hypothesis_posterior(
        observation=72.0,
        hypotheses=["Machine A", "Machine B"],
        priors=[0.60, 0.40],
        means=[70.0, 80.0],
        standard_deviations=[3.0, 4.0],
    )

    print_posterior_distribution(terms)


# ---------------------------------------------------------------------------
# 14. BAYESIAN DECISION THEORY
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    action: str
    expected_loss: float


def choose_minimum_expected_loss(
    posterior: Dict[str, float],
    loss_matrix: Dict[str, Dict[str, float]],
) -> Decision:
    """
    Decision theory separates probability from action.

    posterior[state] = probability of a state
    loss_matrix[action][state] = loss when action is chosen and state occurs

    Expected loss(action) =
        sum P(state | evidence) * loss(action, state)
    """
    decisions: List[Decision] = []

    for action, losses in loss_matrix.items():
        expected_loss = sum(
            posterior[state] * losses[state]
            for state in posterior
        )
        decisions.append(
            Decision(
                action=action,
                expected_loss=expected_loss,
            )
        )

    return min(
        decisions,
        key=lambda decision: decision.expected_loss,
    )


def demonstrate_decision_theory() -> None:
    print("\n=== BAYESIAN DECISION THEORY ===")

    posterior = {
        "fraud": 0.30,
        "legitimate": 0.70,
    }

    loss_matrix = {
        "approve": {
            "fraud": 100.0,
            "legitimate": 0.0,
        },
        "review": {
            "fraud": 10.0,
            "legitimate": 2.0,
        },
        "decline": {
            "fraud": 0.0,
            "legitimate": 20.0,
        },
    }

    decision = choose_minimum_expected_loss(
        posterior,
        loss_matrix,
    )

    print(f"Posterior: {posterior}")
    print(f"Selected action by minimum expected loss: {decision.action}")
    print(f"Expected loss: {decision.expected_loss:.2f}")


# ---------------------------------------------------------------------------
# 15. EDGE CASES AND COMMON MISTAKES
# ---------------------------------------------------------------------------

def demonstrate_edge_cases() -> None:
    print("\n=== EDGE CASES ===")

    print(
        "Certain hypothesis:",
        bayes_binary(1.0, 0.8, 0.1)[0],
    )

    print(
        "Impossible hypothesis:",
        bayes_binary(0.0, 0.8, 0.1)[0],
    )

    try:
        bayes_binary(0.5, 0.0, 0.0)
    except ValueError as error:
        print(f"Zero-evidence error: {error}")

    try:
        conditional_probability(0.2, 0.0)
    except ValueError as error:
        print(f"Undefined conditional probability: {error}")

    # Common mistake:
    # P(H|E) is generally NOT equal to P(E|H).
    p_h_given_e = 0.25
    p_e_given_h = 0.80

    print(f"P(H | E) = {p_h_given_e}")
    print(f"P(E | H) = {p_e_given_h}")
    print("Conditional probabilities are directional.")


# ---------------------------------------------------------------------------
# 16. MONTE CARLO VALIDATION
# ---------------------------------------------------------------------------

def monte_carlo_bayes_check(
    trials: int = 200_000,
    seed: int = 42,
) -> float:
    """
    Empirically approximate P(H|E).

    A local deterministic pseudo-random generator is implemented so that
    this example requires no external package.
    """
    # Linear congruential generator.
    state = seed
    modulus = 2**31
    multiplier = 1103515245
    increment = 12345

    def random_unit() -> float:
        nonlocal state
        state = (
            multiplier * state + increment
        ) % modulus
        return state / modulus

    prior_h = 0.10
    p_e_given_h = 0.80
    p_e_given_not_h = 0.05

    evidence_count = 0
    hypothesis_and_evidence_count = 0

    for _ in range(trials):
        h = random_unit() < prior_h

        if h:
            e = random_unit() < p_e_given_h
        else:
            e = random_unit() < p_e_given_not_h

        if e:
            evidence_count += 1
            if h:
                hypothesis_and_evidence_count += 1

    if evidence_count == 0:
        return 0.0

    return (
        hypothesis_and_evidence_count
        / evidence_count
    )


def demonstrate_monte_carlo() -> None:
    print("\n=== MONTE CARLO CHECK ===")

    analytical, _ = bayes_binary(
        prior_h=0.10,
        likelihood_e_given_h=0.80,
        likelihood_e_given_not_h=0.05,
    )

    estimated = monte_carlo_bayes_check()

    print(f"Analytical posterior: {analytical:.4%}")
    print(f"Simulation estimate: {estimated:.4%}")
    print(
        f"Absolute difference: "
        f"{abs(analytical - estimated):.4%}"
    )


# ---------------------------------------------------------------------------
# 17. CALIBRATION AND INTERPRETATION
# ---------------------------------------------------------------------------

def brier_score(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
) -> float:
    """
    Brier score for binary probabilistic predictions:

        mean((p_i - y_i)^2)

    Lower values indicate better probabilistic accuracy.
    """
    if len(probabilities) != len(outcomes):
        raise ValueError("Inputs must have equal length.")

    if not probabilities:
        raise ValueError("Inputs cannot be empty.")

    for p in probabilities:
        validate_probability(p)

    for outcome in outcomes:
        if outcome not in (0, 1):
            raise ValueError("Outcomes must be 0 or 1.")

    return sum(
        (p - y) ** 2
        for p, y in zip(probabilities, outcomes)
    ) / len(probabilities)


def demonstrate_calibration_metric() -> None:
    print("\n=== PROBABILISTIC EVALUATION ===")

    predicted = [0.9, 0.8, 0.2, 0.1]
    actual = [1, 1, 0, 1]

    print(
        f"Brier score: "
        f"{brier_score(predicted, actual):.4f}"
    )


# ---------------------------------------------------------------------------
# 18. COMPLETE STUDY WALKTHROUGH
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("BAYES' THEOREM: CONDITIONAL PROBABILITY AND BAYESIAN INFERENCE")
    print("=" * 78)

    demonstrate_conditional_probability()
    demonstrate_bayes_basic()
    demonstrate_denominator()
    demonstrate_frequency_table()
    demonstrate_multiple_hypotheses()
    demonstrate_odds_form()
    demonstrate_sequential_updating()
    demonstrate_diagnostic_testing()
    demonstrate_naive_bayes()
    demonstrate_log_space()
    demonstrate_continuous_bayes()
    demonstrate_decision_theory()
    demonstrate_edge_cases()
    demonstrate_monte_carlo()
    demonstrate_calibration_metric()

    print("\n=== KEY FORMULAS ===")
    print("Conditional probability:")
    print("  P(A | B) = P(A AND B) / P(B)")
    print()
    print("Multiplication rule:")
    print("  P(A AND B) = P(A) P(B | A)")
    print()
    print("Bayes' theorem:")
    print("  P(H | E) = P(E | H) P(H) / P(E)")
    print()
    print("Law of total probability:")
    print("  P(E) = SUM[P(E | H_i) P(H_i)]")
    print()
    print("Odds form:")
    print("  Posterior odds = Prior odds × Likelihood ratio")
    print()
    print("The central distinction is:")
    print("  Prior      -> belief before observing evidence")
    print("  Likelihood -> compatibility of evidence with a hypothesis")
    print("  Evidence   -> total probability of the observed evidence")
    print("  Posterior  -> updated belief after observing evidence")

    print("\n=== STUDY FILE COMPLETE ===")


if __name__ == "__main__":
    main()
