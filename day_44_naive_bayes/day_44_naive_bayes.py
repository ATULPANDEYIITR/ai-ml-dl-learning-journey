"""
Naive Bayes Classification Laboratory
=====================================

A self-contained Python implementation covering:
- Bayes' theorem and posterior probability
- Conditional independence and the Naive Bayes assumption
- Gaussian Naive Bayes for continuous measurements
- Multinomial Naive Bayes for word counts
- Bernoulli Naive Bayes for binary features
- Log probabilities, smoothing, prediction, evaluation, and model persistence
- Numerical stability, zero-frequency problems, and practical limitations

Requirements: Python 3.10 or later. Standard library only.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def heading(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def bayes_theorem(
    prior: float,
    likelihood: float,
    evidence_probability: float,
) -> float:
    """Calculate P(H|E) = P(E|H)P(H)/P(E)."""
    if not 0.0 <= prior <= 1.0:
        raise ValueError("Prior must lie between zero and one.")
    if not 0.0 <= likelihood <= 1.0:
        raise ValueError("Likelihood must lie between zero and one.")
    if not 0.0 < evidence_probability <= 1.0:
        raise ValueError("Evidence probability must be in (0, 1].")

    posterior = prior * likelihood / evidence_probability
    if posterior > 1.0 + 1e-12:
        raise ValueError("The supplied probabilities are inconsistent.")
    return min(posterior, 1.0)


def normalize_log_probabilities(
    log_scores: dict[str, float],
) -> dict[str, float]:
    """Convert arbitrary finite log scores into normalized probabilities."""
    if not log_scores:
        raise ValueError("At least one class score is required.")

    maximum = max(log_scores.values())
    if maximum == -math.inf:
        raise ValueError("All classes have zero probability.")
    if maximum == math.inf or math.isnan(maximum):
        raise ValueError("Log scores must be finite or negative infinity.")

    exponentials = {
        label: math.exp(score - maximum)
        for label, score in log_scores.items()
    }
    denominator = sum(exponentials.values())
    return {
        label: value / denominator
        for label, value in exponentials.items()
    }


def stratified_train_test_split(
    samples: Sequence[Any],
    labels: Sequence[str],
    test_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[Any], list[str], list[Any], list[str]]:
    """Split each class independently to preserve class representation."""
    if len(samples) != len(labels):
        raise ValueError("Samples and labels must have equal lengths.")
    if not samples:
        raise ValueError("The dataset cannot be empty.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one.")

    rng = random.Random(seed)
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)

    train_indices: list[int] = []
    test_indices: list[int] = []

    for indices in by_class.values():
        rng.shuffle(indices)
        test_count = max(1, round(len(indices) * test_fraction))
        if test_count >= len(indices):
            raise ValueError(
                "Each class needs at least two examples for this split."
            )
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        [samples[i] for i in train_indices],
        [labels[i] for i in train_indices],
        [samples[i] for i in test_indices],
        [labels[i] for i in test_indices],
    )


class GaussianNaiveBayes:
    """
    Gaussian NB models each feature within each class using a normal density.

    The implementation uses population variance and a variance floor to
    prevent division by zero when a feature is constant within a class.
    """

    def __init__(self, variance_smoothing: float = 1e-9) -> None:
        if variance_smoothing <= 0:
            raise ValueError("variance_smoothing must be positive.")
        self.variance_smoothing = variance_smoothing
        self.classes: list[str] = []
        self.log_priors: dict[str, float] = {}
        self.means: dict[str, list[float]] = {}
        self.variances: dict[str, list[float]] = {}
        self.n_features: int | None = None
        self.fitted = False

    def fit(
        self,
        samples: Sequence[Sequence[float]],
        labels: Sequence[str],
    ) -> "GaussianNaiveBayes":
        if not samples or len(samples) != len(labels):
            raise ValueError("Non-empty samples and matching labels are required.")

        width = len(samples[0])
        if width == 0 or any(len(row) != width for row in samples):
            raise ValueError("Every sample must have the same positive width.")

        numeric_samples: list[list[float]] = []
        for row in samples:
            converted = [float(value) for value in row]
            if not all(math.isfinite(value) for value in converted):
                raise ValueError("Features must be finite numbers.")
            numeric_samples.append(converted)

        grouped: dict[str, list[list[float]]] = defaultdict(list)
        for row, label in zip(numeric_samples, labels):
            if not label:
                raise ValueError("Class labels cannot be empty.")
            grouped[label].append(row)

        self.classes = sorted(grouped)
        self.n_features = width
        total = len(numeric_samples)
        self.log_priors = {}
        self.means = {}
        self.variances = {}

        global_variances: list[float] = []
        for column in range(width):
            values = [row[column] for row in numeric_samples]
            mean = sum(values) / total
            variance = sum((value - mean) ** 2 for value in values) / total
            global_variances.append(variance)

        # The floor scales with the dataset's feature magnitudes.
        for label in self.classes:
            rows = grouped[label]
            self.log_priors[label] = math.log(len(rows) / total)
            means: list[float] = []
            variances: list[float] = []

            for column in range(width):
                values = [row[column] for row in rows]
                mean = sum(values) / len(values)
                variance = sum(
                    (value - mean) ** 2 for value in values
                ) / len(values)

                scale = max(global_variances[column], 1.0)
                variance = max(
                    variance,
                    self.variance_smoothing * scale,
                )
                means.append(mean)
                variances.append(variance)

            self.means[label] = means
            self.variances[label] = variances

        self.fitted = True
        return self

    def predict_log_scores(self, sample: Sequence[float]) -> dict[str, float]:
        if not self.fitted or self.n_features is None:
            raise RuntimeError("Fit the classifier before prediction.")
        if len(sample) != self.n_features:
            raise ValueError("The sample has an incorrect feature count.")

        values = [float(value) for value in sample]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Features must be finite numbers.")

        scores: dict[str, float] = {}
        for label in self.classes:
            score = self.log_priors[label]

            for column, value in enumerate(values):
                mean = self.means[label][column]
                variance = self.variances[label][column]
                score -= 0.5 * (
                    math.log(2.0 * math.pi * variance)
                    + (value - mean) ** 2 / variance
                )

            scores[label] = score

        return scores

    def predict_proba(self, sample: Sequence[float]) -> dict[str, float]:
        return normalize_log_probabilities(self.predict_log_scores(sample))

    def predict(self, sample: Sequence[float]) -> str:
        return max(
            self.predict_log_scores(sample),
            key=self.predict_log_scores(sample).get,
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted model.")
        return {
            "model_type": "GaussianNaiveBayes",
            "variance_smoothing": self.variance_smoothing,
            "classes": self.classes,
            "log_priors": self.log_priors,
            "means": self.means,
            "variances": self.variances,
            "n_features": self.n_features,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "GaussianNaiveBayes":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("model_type") != "GaussianNaiveBayes":
            raise ValueError("File does not contain a Gaussian NB model.")

        model = cls(float(data["variance_smoothing"]))
        model.classes = list(data["classes"])
        model.log_priors = {
            key: float(value) for key, value in data["log_priors"].items()
        }
        model.means = {
            key: [float(value) for value in values]
            for key, values in data["means"].items()
        }
        model.variances = {
            key: [float(value) for value in values]
            for key, values in data["variances"].items()
        }
        model.n_features = int(data["n_features"])

        if (
            set(model.classes) != set(model.log_priors)
            or set(model.classes) != set(model.means)
            or set(model.classes) != set(model.variances)
            or any(len(model.means[c]) != model.n_features for c in model.classes)
            or any(len(model.variances[c]) != model.n_features for c in model.classes)
            or any(v <= 0 for c in model.classes for v in model.variances[c])
        ):
            raise ValueError("The saved model has an invalid structure.")

        model.fitted = True
        return model


def tokenize(text: str) -> list[str]:
    """Normalize a small English text corpus without external NLP packages."""
    normalized = "".join(
        character.lower() if character.isalnum() else " "
        for character in text
    )
    return normalized.split()


class MultinomialNaiveBayes:
    """
    Multinomial NB models token counts.

    Laplace smoothing assigns probability mass to vocabulary terms that
    were absent from a class during training.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0 or not math.isfinite(alpha):
            raise ValueError("alpha must be a finite positive number.")
        self.alpha = alpha
        self.classes: list[str] = []
        self.log_priors: dict[str, float] = {}
        self.log_likelihoods: dict[str, dict[str, float]] = {}
        self.vocabulary: set[str] = set()
        self.fitted = False

    def fit(
        self,
        documents: Sequence[str],
        labels: Sequence[str],
    ) -> "MultinomialNaiveBayes":
        if not documents or len(documents) != len(labels):
            raise ValueError("Non-empty documents and matching labels are required.")

        tokenized = [tokenize(document) for document in documents]
        grouped_documents: dict[str, list[list[str]]] = defaultdict(list)
        for document, label in zip(tokenized, labels):
            if not label:
                raise ValueError("Class labels cannot be empty.")
            grouped_documents[label].append(document)

        self.classes = sorted(grouped_documents)
        self.vocabulary = {
            token for document in tokenized for token in document
        }
        if not self.vocabulary:
            raise ValueError("The training corpus contains no tokens.")

        total_documents = len(documents)
        self.log_priors = {}
        self.log_likelihoods = {}

        for label in self.classes:
            class_documents = grouped_documents[label]
            counts = Counter(
                token
                for document in class_documents
                for token in document
            )
            total_tokens = sum(counts.values())
            denominator = total_tokens + self.alpha * len(self.vocabulary)

            self.log_priors[label] = math.log(
                len(class_documents) / total_documents
            )
            self.log_likelihoods[label] = {
                token: math.log(
                    (counts[token] + self.alpha) / denominator
                )
                for token in self.vocabulary
            }

        self.fitted = True
        return self

    def predict_log_scores(self, document: str) -> dict[str, float]:
        if not self.fitted:
            raise RuntimeError("Fit the classifier before prediction.")

        tokens = Counter(tokenize(document))
        scores: dict[str, float] = {}

        for label in self.classes:
            score = self.log_priors[label]
            for token, count in tokens.items():
                # An unseen word is ignored because the model's vocabulary
                # is defined from training data. An <UNK> policy is another
                # valid design, but must be applied consistently.
                if token in self.vocabulary:
                    score += count * self.log_likelihoods[label][token]
            scores[label] = score

        return scores

    def predict_proba(self, document: str) -> dict[str, float]:
        return normalize_log_probabilities(self.predict_log_scores(document))

    def predict(self, document: str) -> str:
        scores = self.predict_log_scores(document)
        return max(scores, key=scores.get)


class BernoulliNaiveBayes:
    """
    Bernoulli NB models whether each vocabulary feature is present.

    Repeated occurrences do not increase a feature's contribution.
    Both present and absent features contribute to the log likelihood.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0 or not math.isfinite(alpha):
            raise ValueError("alpha must be a finite positive number.")
        self.alpha = alpha
        self.classes: list[str] = []
        self.vocabulary: set[str] = set()
        self.log_priors: dict[str, float] = {}
        self.log_present: dict[str, dict[str, float]] = {}
        self.log_absent: dict[str, dict[str, float]] = {}
        self.fitted = False

    def fit(
        self,
        documents: Sequence[str],
        labels: Sequence[str],
    ) -> "BernoulliNaiveBayes":
        if not documents or len(documents) != len(labels):
            raise ValueError("Non-empty documents and matching labels are required.")

        token_sets = [set(tokenize(document)) for document in documents]
        grouped: dict[str, list[set[str]]] = defaultdict(list)

        for token_set, label in zip(token_sets, labels):
            if not label:
                raise ValueError("Class labels cannot be empty.")
            grouped[label].append(token_set)

        self.classes = sorted(grouped)
        self.vocabulary = set().union(*token_sets)
        if not self.vocabulary:
            raise ValueError("The training corpus contains no tokens.")

        total_documents = len(documents)
        self.log_priors = {}
        self.log_present = {}
        self.log_absent = {}

        for label in self.classes:
            class_documents = grouped[label]
            count = len(class_documents)
            self.log_priors[label] = math.log(count / total_documents)
            self.log_present[label] = {}
            self.log_absent[label] = {}

            for token in self.vocabulary:
                present_count = sum(
                    token in document for document in class_documents
                )
                probability = (
                    present_count + self.alpha
                ) / (count + 2 * self.alpha)

                self.log_present[label][token] = math.log(probability)
                self.log_absent[label][token] = math.log1p(-probability)

        self.fitted = True
        return self

    def predict_log_scores(self, document: str) -> dict[str, float]:
        if not self.fitted:
            raise RuntimeError("Fit the classifier before prediction.")

        present = set(tokenize(document)) & self.vocabulary
        scores: dict[str, float] = {}

        for label in self.classes:
            score = self.log_priors[label]

            for token in self.vocabulary:
                # Bernoulli NB explicitly models feature absence as well.
                if token in present:
                    score += self.log_present[label][token]
                else:
                    score += self.log_absent[label][token]

            scores[label] = score

        return scores

    def predict_proba(self, document: str) -> dict[str, float]:
        return normalize_log_probabilities(self.predict_log_scores(document))

    def predict(self, document: str) -> str:
        scores = self.predict_log_scores(document)
        return max(scores, key=scores.get)


def accuracy_score(
    actual: Sequence[str],
    predicted: Sequence[str],
) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Matching, non-empty label sequences are required.")
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def confusion_matrix(
    actual: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted labels must have equal lengths.")

    matrix = {
        true_label: {predicted_label: 0 for predicted_label in labels}
        for true_label in labels
    }

    for true_label, predicted_label in zip(actual, predicted):
        if true_label not in matrix or predicted_label not in matrix:
            raise ValueError("All labels must appear in the supplied label list.")
        matrix[true_label][predicted_label] += 1

    return matrix


def print_probabilities(probabilities: dict[str, float]) -> None:
    for label, probability in sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {label:12s} {probability:.4f}")


def demonstrate_bayes_theorem() -> None:
    heading("Bayes' theorem and posterior probability")

    # A screening test has 1% prevalence, 95% sensitivity, and 90% specificity.
    prevalence = 0.01
    sensitivity = 0.95
    specificity = 0.90
    false_positive_rate = 1.0 - specificity

    evidence_probability = (
        sensitivity * prevalence
        + false_positive_rate * (1.0 - prevalence)
    )
    posterior = bayes_theorem(
        prevalence,
        sensitivity,
        evidence_probability,
    )

    print(f"Prior probability of condition: {prevalence:.3%}")
    print(f"Probability of a positive test: {evidence_probability:.3%}")
    print(f"Probability of condition given positive test: {posterior:.3%}")
    print(
        "A positive test does not imply a 95% posterior probability; "
        "the base rate matters."
    )


def demonstrate_gaussian_nb() -> None:
    heading("Gaussian Naive Bayes: equipment condition classification")

    # Each row contains vibration RMS and operating temperature.
    samples = [
        [1.1, 43.0], [1.3, 45.0], [1.5, 44.0], [1.2, 46.0],
        [1.4, 42.0], [1.6, 47.0], [1.0, 44.0], [1.3, 43.0],
        [4.2, 83.0], [4.5, 87.0], [4.7, 90.0], [4.3, 85.0],
        [4.8, 91.0], [4.1, 82.0], [4.6, 88.0], [4.4, 86.0],
        [1.7, 48.0], [4.0, 80.0], [1.2, 41.0], [4.9, 92.0],
    ]
    labels = ["normal"] * 10 + ["fault"] * 10

    x_train, y_train, x_test, y_test = stratified_train_test_split(
        samples, labels, test_fraction=0.25, seed=17
    )
    model = GaussianNaiveBayes().fit(x_train, y_train)

    predictions = [model.predict(row) for row in x_test]
    print(f"Holdout accuracy: {accuracy_score(y_test, predictions):.3f}")

    new_reading = [4.35, 86.0]
    print(f"Reading: vibration={new_reading[0]}, temperature={new_reading[1]}")
    print("Class probabilities:")
    print_probabilities(model.predict_proba(new_reading))

    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "equipment_model.json"
        model.save(model_path)
        restored = GaussianNaiveBayes.load(model_path)
        print(f"Restored model prediction: {restored.predict(new_reading)}")


def demonstrate_text_models() -> None:
    heading("Multinomial and Bernoulli Naive Bayes: support message routing")

    documents = [
        "server latency timeout connection failed",
        "database connection timeout server error",
        "network server latency high",
        "failed deployment server timeout",
        "invoice payment receipt account billing",
        "payment invoice billing receipt overdue",
        "account payment transaction invoice",
        "billing receipt account payment overdue",
        "reset password account login security",
        "login password account authentication",
        "security reset password access",
        "authentication login access account",
        "server timeout error latency",
        "payment receipt invoice",
        "password reset login",
        "deployment failed timeout",
    ]
    labels = [
        "technical",
        "technical",
        "technical",
        "technical",
        "billing",
        "billing",
        "billing",
        "billing",
        "account",
        "account",
        "account",
        "account",
        "technical",
        "billing",
        "account",
        "technical",
    ]

    x_train, y_train, x_test, y_test = stratified_train_test_split(
        documents, labels, test_fraction=0.25, seed=21
    )

    models = {
        "Multinomial": MultinomialNaiveBayes(alpha=1.0).fit(
            x_train, y_train
        ),
        "Bernoulli": BernoulliNaiveBayes(alpha=1.0).fit(
            x_train, y_train
        ),
    }

    for name, model in models.items():
        predictions = [model.predict(document) for document in x_test]
        print(f"\n{name} NB holdout accuracy: {accuracy_score(y_test, predictions):.3f}")
        print("Confusion matrix, rows=actual and columns=predicted:")
        print(confusion_matrix(y_test, predictions, model.classes))

        message = "server latency timeout after deployment"
        print(f"Message: {message!r}")
        print_probabilities(model.predict_proba(message))

    repeated_word = "payment payment payment invoice"
    print(f"\nRepeated-token message: {repeated_word!r}")
    print("Multinomial probabilities:")
    print_probabilities(models["Multinomial"].predict_proba(repeated_word))
    print("Bernoulli probabilities:")
    print_probabilities(models["Bernoulli"].predict_proba(repeated_word))


class NaiveBayesTests(unittest.TestCase):
    def test_bayes_theorem(self) -> None:
        posterior = bayes_theorem(0.2, 0.5, 0.25)
        self.assertAlmostEqual(posterior, 0.4)

    def test_inconsistent_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            bayes_theorem(0.9, 0.9, 0.1)

    def test_gaussian_prediction_and_constant_feature(self) -> None:
        model = GaussianNaiveBayes().fit(
            [[1.0, 10.0], [1.1, 10.0], [5.0, 10.0], [5.1, 10.0]],
            ["A", "A", "B", "B"],
        )
        self.assertEqual(model.predict([5.05, 10.0]), "B")
        probabilities = model.predict_proba([5.05, 10.0])
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_gaussian_feature_validation(self) -> None:
        model = GaussianNaiveBayes().fit(
            [[0.0], [1.0], [5.0], [6.0]],
            ["A", "A", "B", "B"],
        )
        with self.assertRaises(ValueError):
            model.predict([1.0, 2.0])

    def test_multinomial_unseen_word(self) -> None:
        model = MultinomialNaiveBayes().fit(
            ["server error", "server timeout", "invoice payment", "invoice receipt"],
            ["technical", "technical", "billing", "billing"],
        )
        self.assertIn(model.predict("unknownword"), model.classes)
        self.assertAlmostEqual(
            sum(model.predict_proba("unknownword").values()), 1.0
        )

    def test_bernoulli_absence_changes_score(self) -> None:
        model = BernoulliNaiveBayes().fit(
            ["red round", "red circle", "blue square", "blue angular"],
            ["red_group", "red_group", "blue_group", "blue_group"],
        )
        present_score = model.predict_log_scores("red")
        absent_score = model.predict_log_scores("round")
        self.assertNotEqual(present_score, absent_score)

    def test_smoothing_validation(self) -> None:
        with self.assertRaises(ValueError):
            MultinomialNaiveBayes(alpha=0)

    def test_empty_text_corpus(self) -> None:
        with self.assertRaises(ValueError):
            MultinomialNaiveBayes().fit(["", "   "], ["A", "B"])


def run_tests() -> None:
    heading("Automated verification")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NaiveBayesTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    demonstrate_bayes_theorem()
    demonstrate_gaussian_nb()
    demonstrate_text_models()
    run_tests()


if __name__ == "__main__":
    main()
