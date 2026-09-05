"""
MACHINE LEARNING PARADIGMS
==========================

Topic:
    Supervised Learning
    Unsupervised Learning
    Semi-Supervised Learning
    Self-Supervised Learning
    Reinforcement Learning

Purpose:
    A standalone educational Python script that progresses from absolute
    beginner concepts to practical and advanced implementations.

The examples intentionally use mostly Python's standard library and
NumPy where numerical matrix operations make an implementation clearer.
If NumPy is unavailable, the script still explains the concepts, but the
numerical demonstrations that depend on it will not run.

Install NumPy if needed:
    pip install numpy
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# 1. MACHINE LEARNING FUNDAMENTALS
# =============================================================================

def print_section(title: str) -> None:
    """Print a readable section heading."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_subsection(title: str) -> None:
    """Print a readable subsection heading."""
    print(f"\n--- {title} ---")


def machine_learning_fundamentals() -> None:
    print_section("1. MACHINE LEARNING FUNDAMENTALS")

    print(
        """
Machine learning learns useful relationships from data.

A typical dataset contains:
    X = features or observations
    y = target labels or values

The learning paradigm is determined largely by what information is available
during training and by how the learning signal is constructed.

Five important paradigms demonstrated in this file are:

1. Supervised learning:
       Training data contains input-output pairs (X, y).

2. Unsupervised learning:
       Training data contains inputs X but no explicit target y.

3. Semi-supervised learning:
       A small portion of data is labeled while a larger portion is unlabeled.

4. Self-supervised learning:
       The training target is generated from the data itself.
       No manually supplied external labels are required for the pretext task.

5. Reinforcement learning:
       An agent interacts with an environment and learns from rewards,
       consequences, and accumulated experience.

Important distinction:
    "Unlabeled" does not automatically mean "self-supervised."
    Unsupervised learning generally searches for structure without creating
    a target prediction problem, whereas self-supervised learning constructs
    supervisory signals from the observations themselves.
"""
    )

    example_features = [
        [25, 42000],
        [32, 65000],
        [41, 90000],
        [29, 52000],
    ]

    print("Example feature matrix X:")
    for row in example_features:
        print(" ", row)

    print(
        """
Possible interpretation:
    column 1 = age
    column 2 = annual income

A supervised problem might additionally provide:
    y = whether a customer purchased a product

An unsupervised problem might ask:
    Which customers naturally form similar groups?

A semi-supervised problem might have:
    100 labeled customers + 10,000 unlabeled customers

A self-supervised problem might ask:
    Given part of a sequence, predict a hidden part.

A reinforcement learning problem might be:
    An agent chooses actions in a game and receives rewards.
"""
    )


# =============================================================================
# 2. SUPERVISED LEARNING
# =============================================================================

def supervised_learning_concepts() -> None:
    print_section("2. SUPERVISED LEARNING")

    print(
        """
Supervised learning uses labeled examples.

The fundamental abstraction is:

    f(X) -> y

where:
    X = input/features
    y = target/label
    f = learned function/model

Two major supervised task families are:

Classification:
    Predict a discrete class.
    Examples:
        spam/not-spam
        fraud/not-fraud
        disease class
        image category

Regression:
    Predict a continuous numerical value.
    Examples:
        house price
        demand
        temperature
        revenue

A supervised training workflow commonly contains:

    1. Data collection
    2. Data cleaning
    3. Feature construction
    4. Train/validation/test split
    5. Model selection
    6. Training
    7. Validation and hyperparameter tuning
    8. Final evaluation
    9. Deployment
    10. Monitoring

A central concern is generalization:
    The model should perform well on unseen examples, not merely memorize
    the training set.
"""
    )


# -----------------------------------------------------------------------------
# 2.1 Supervised regression from scratch
# -----------------------------------------------------------------------------

class SimpleLinearRegression:
    """
    Ordinary least squares for one feature.

    Model:
        y_hat = slope * x + intercept

    This implementation demonstrates the mathematical structure rather than
    relying on a machine-learning library.
    """

    def __init__(self) -> None:
        self.slope = 0.0
        self.intercept = 0.0

    def fit(self, x: Sequence[float], y: Sequence[float]) -> None:
        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of samples.")
        if len(x) < 2:
            raise ValueError("At least two observations are required.")

        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        denominator = sum((value - mean_x) ** 2 for value in x)
        if denominator == 0:
            raise ValueError("A constant feature cannot determine a slope.")

        numerator = sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(x, y)
        )

        self.slope = numerator / denominator
        self.intercept = mean_y - self.slope * mean_x

    def predict(self, x: Sequence[float]) -> List[float]:
        return [self.slope * value + self.intercept for value in x]


def mean_squared_error(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Mean squared error for regression."""
    if len(actual) != len(predicted):
        raise ValueError("Inputs must have equal lengths.")
    if not actual:
        raise ValueError("At least one observation is required.")

    return statistics.mean(
        (a - p) ** 2 for a, p in zip(actual, predicted)
    )


def mean_absolute_error(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Mean absolute error for regression."""
    if len(actual) != len(predicted):
        raise ValueError("Inputs must have equal lengths.")
    if not actual:
        raise ValueError("At least one observation is required.")

    return statistics.mean(
        abs(a - p) for a, p in zip(actual, predicted)
    )


def demonstrate_supervised_regression() -> None:
    print_subsection("2.1 Supervised Regression")

    hours_studied = [1, 2, 3, 4, 5, 6]
    exam_scores = [45, 50, 57, 64, 70, 78]

    model = SimpleLinearRegression()
    model.fit(hours_studied, exam_scores)

    predictions = model.predict([2.5, 7])

    print(f"Slope: {model.slope:.3f}")
    print(f"Intercept: {model.intercept:.3f}")
    print(f"Predictions: {[round(value, 2) for value in predictions]}")

    training_predictions = model.predict(hours_studied)
    print(
        "Training MSE:",
        round(mean_squared_error(exam_scores, training_predictions), 3),
    )
    print(
        "Training MAE:",
        round(mean_absolute_error(exam_scores, training_predictions), 3),
    )

    print(
        """
Important regression metrics:

MSE:
    Penalizes large errors strongly because errors are squared.

RMSE:
    sqrt(MSE), expressed in the same units as the target.

MAE:
    Average absolute error. It is generally less sensitive to extreme errors
    than MSE.

R^2:
    Measures variance explained relative to a baseline model.

Metric selection should reflect the business cost of errors.
"""


# -----------------------------------------------------------------------------
# 2.2 Supervised classification from scratch
# -----------------------------------------------------------------------------

class KNearestNeighborsClassifier:
    """
    Small educational implementation of k-nearest neighbors classification.

    KNN is a lazy learner:
        - little explicit training occurs
        - most computation happens during prediction

    It demonstrates:
        - distance calculation
        - neighborhood selection
        - majority voting
        - a hyperparameter k
    """

    def __init__(self, k: int = 3) -> None:
        if k <= 0:
            raise ValueError("k must be positive.")
        self.k = k
        self.training_features: List[Tuple[float, ...]] = []
        self.training_labels: List[str] = []

    @staticmethod
    def euclidean_distance(
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        if len(first) != len(second):
            raise ValueError("Points must have the same dimensionality.")

        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(first, second))
        )

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[str],
    ) -> None:
        if len(features) != len(labels):
            raise ValueError("Features and labels must have equal lengths.")
        if not features:
            raise ValueError("Training data cannot be empty.")

        dimensions = len(features[0])
        if dimensions == 0:
            raise ValueError("Each observation needs at least one feature.")

        for row in features:
            if len(row) != dimensions:
                raise ValueError("All feature vectors must have equal dimensions.")

        self.training_features = [tuple(row) for row in features]
        self.training_labels = list(labels)

    def predict_one(self, observation: Sequence[float]) -> str:
        if not self.training_features:
            raise RuntimeError("The classifier must be fitted first.")

        distances = [
            (
                self.euclidean_distance(observation, training_point),
                label,
            )
            for training_point, label in zip(
                self.training_features,
                self.training_labels,
            )
        ]

        nearest = sorted(distances, key=lambda pair: pair[0])[: self.k]
        votes = Counter(label for _, label in nearest)

        # Deterministic tie-breaking makes the example reproducible.
        return sorted(
            votes.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    def predict(
        self,
        observations: Sequence[Sequence[float]],
    ) -> List[str]:
        return [self.predict_one(row) for row in observations]


def classification_accuracy(
    actual: Sequence[str],
    predicted: Sequence[str],
) -> float:
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted labels must have equal lengths.")
    if not actual:
        raise ValueError("At least one observation is required.")

    correct = sum(a == p for a, p in zip(actual, predicted))
    return correct / len(actual)


def demonstrate_supervised_classification() -> None:
    print_subsection("2.2 Supervised Classification")

    features = [
        [1.0, 1.1],
        [1.2, 0.9],
        [0.8, 1.0],
        [5.0, 5.1],
        [5.2, 4.8],
        [4.9, 5.3],
    ]

    labels = [
        "class_A",
        "class_A",
        "class_A",
        "class_B",
        "class_B",
        "class_B",
    ]

    classifier = KNearestNeighborsClassifier(k=3)
    classifier.fit(features, labels)

    test_points = [
        [1.1, 1.0],
        [5.1, 5.0],
        [3.0, 3.0],
    ]

    predictions = classifier.predict(test_points)

    for point, prediction in zip(test_points, predictions):
        print(f"{point} -> {prediction}")

    training_predictions = classifier.predict(features)
    print(
        "Training accuracy:",
        round(classification_accuracy(labels, training_predictions), 3),
    )

    print(
        """
KNN illustrates an important practical issue: feature scale.

Suppose:
    age ranges from 18 to 80
    income ranges from 20,000 to 10,000,000

Euclidean distance can become dominated by income.

Common remedies include:
    - standardization
    - min-max normalization
    - domain-specific transformations

Scaling must be learned from the training set and then applied consistently
to validation, test, and production data.
"""
    )


# =============================================================================
# 3. CLASSIFICATION METRICS
# =============================================================================

def classification_metrics_demo() -> None:
    print_section("3. CLASSIFICATION METRICS")

    actual = ["positive", "positive", "negative", "negative", "negative"]
    predicted = ["positive", "negative", "negative", "negative", "positive"]

    true_positive = sum(
        a == "positive" and p == "positive"
        for a, p in zip(actual, predicted)
    )
    true_negative = sum(
        a == "negative" and p == "negative"
        for a, p in zip(actual, predicted)
    )
    false_positive = sum(
        a == "negative" and p == "positive"
        for a, p in zip(actual, predicted)
    )
    false_negative = sum(
        a == "positive" and p == "negative"
        for a, p in zip(actual, predicted)
    )

    accuracy = (true_positive + true_negative) / len(actual)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    print("Confusion matrix components:")
    print("TP =", true_positive)
    print("TN =", true_negative)
    print("FP =", false_positive)
    print("FN =", false_negative)

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 score : {f1:.3f}")

    print(
        """
Definitions:

True Positive (TP):
    A positive example correctly classified as positive.

True Negative (TN):
    A negative example correctly classified as negative.

False Positive (FP):
    A negative example incorrectly classified as positive.

False Negative (FN):
    A positive example incorrectly classified as negative.

Precision:
    Of predicted positives, how many were actually positive?

Recall:
    Of actual positives, how many were found?

F1:
    Harmonic mean of precision and recall.

Accuracy can be misleading on imbalanced datasets.

Example:
    If 99% of transactions are legitimate, a model predicting "legitimate"
    for every transaction achieves 99% accuracy while detecting zero fraud.
"""
    )


# =============================================================================
# 4. TRAIN / VALIDATION / TEST SPLITS AND LEAKAGE
# =============================================================================

def train_validation_test_demo() -> None:
    print_section("4. DATA SPLITTING AND DATA LEAKAGE")

    data = list(range(1, 21))
    random.Random(42).shuffle(data)

    train = data[:12]
    validation = data[12:16]
    test = data[16:]

    print("Training set  :", train)
    print("Validation set:", validation)
    print("Test set      :", test)

    print(
        """
Training set:
    Used to estimate model parameters.

Validation set:
    Used for model selection and hyperparameter tuning.

Test set:
    Reserved for final unbiased evaluation.

Data leakage occurs when information unavailable at prediction time
influences model training.

Examples:
    - scaling using the complete dataset before splitting
    - using future information to predict the past
    - including a post-outcome variable as a feature
    - selecting features using the test set

A useful principle is:

    Fit preprocessing on training data only.

Then:
    transform training data using training statistics
    transform validation data using the same statistics
    transform test data using the same statistics
"""
    )


# =============================================================================
# 5. OVERFITTING, UNDERFITTING, BIAS, AND VARIANCE
# =============================================================================

def bias_variance_demo() -> None:
    print_section("5. OVERFITTING, UNDERFITTING, BIAS, AND VARIANCE")

    print(
        """
Underfitting:
    The model is too simple to capture important structure.
    Typical pattern:
        high training error
        high validation error

Overfitting:
    The model captures training-specific noise.
    Typical pattern:
        very low training error
        substantially higher validation error

Good generalization:
    Training and validation performance are both acceptable and reasonably
    consistent.

Bias:
    Error associated with overly restrictive assumptions.

Variance:
    Sensitivity to fluctuations in the training data.

Typical methods for controlling overfitting:
    - regularization
    - simpler models
    - more training data
    - data augmentation
    - feature selection
    - early stopping
    - cross-validation
    - dropout for neural networks
    - pruning for some tree-based models
"""
    )


# =============================================================================
# 6. UNSUPERVISED LEARNING
# =============================================================================

def unsupervised_learning_concepts() -> None:
    print_section("6. UNSUPERVISED LEARNING")

    print(
        """
Unsupervised learning receives observations without externally supplied
target labels.

The goal may be:

    clustering
    dimensionality reduction
    density estimation
    anomaly detection
    discovering latent structure

The central difference from supervised learning is not merely whether the
dataset has columns. The key difference is the absence of an externally
provided target that defines the desired prediction.

Important examples:

Clustering:
    Discover groups of similar observations.

Dimensionality reduction:
    Represent data with fewer dimensions while preserving useful structure.

Anomaly detection:
    Identify observations that differ strongly from expected patterns.

A major challenge is evaluation:
    There may be no ground-truth answer.

Therefore, internal metrics, stability analysis, domain validation, and
downstream usefulness can all matter.
"""
    )


# -----------------------------------------------------------------------------
# 6.1 K-means clustering from scratch
# -----------------------------------------------------------------------------

class KMeans:
    """
    Educational K-means implementation.

    Algorithm:
        1. Initialize k centroids.
        2. Assign every point to its nearest centroid.
        3. Recalculate each centroid as the mean of assigned points.
        4. Repeat until assignments stabilize or max_iterations is reached.

    Objective:
        Minimize within-cluster sum of squared distances.
    """

    def __init__(
        self,
        n_clusters: int,
        max_iterations: int = 100,
        random_state: int = 42,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")

        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids: List[List[float]] = []
        self.labels_: List[int] = []

    @staticmethod
    def _distance(
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))

    @staticmethod
    def _mean_points(points: Sequence[Sequence[float]]) -> List[float]:
        if not points:
            raise ValueError("Cannot calculate a mean of zero points.")

        dimensions = len(points[0])
        return [
            statistics.mean(point[dimension] for point in points)
            for dimension in range(dimensions)
        ]

    def fit(self, data: Sequence[Sequence[float]]) -> "KMeans":
        if len(data) < self.n_clusters:
            raise ValueError("There must be at least k observations.")

        dimensions = len(data[0])
        if dimensions == 0:
            raise ValueError("Observations must have at least one dimension.")

        if any(len(point) != dimensions for point in data):
            raise ValueError("All observations must have the same dimensions.")

        rng = random.Random(self.random_state)
        selected = rng.sample(list(data), self.n_clusters)
        self.centroids = [list(point) for point in selected]

        previous_labels: Optional[List[int]] = None

        for _ in range(self.max_iterations):
            labels = [
                min(
                    range(self.n_clusters),
                    key=lambda cluster_index: self._distance(
                        point,
                        self.centroids[cluster_index],
                    ),
                )
                for point in data
            ]

            if labels == previous_labels:
                break

            previous_labels = labels.copy()

            new_centroids = []
            for cluster_index in range(self.n_clusters):
                members = [
                    point
                    for point, label in zip(data, labels)
                    if label == cluster_index
                ]

                # Empty clusters are possible. Reusing the old centroid is
                # a simple safe strategy for this educational implementation.
                if members:
                    new_centroids.append(self._mean_points(members))
                else:
                    new_centroids.append(self.centroids[cluster_index])

            self.centroids = new_centroids

        self.labels_ = labels
        return self

    def predict(self, data: Sequence[Sequence[float]]) -> List[int]:
        if not self.centroids:
            raise RuntimeError("KMeans must be fitted before prediction.")

        return [
            min(
                range(self.n_clusters),
                key=lambda cluster_index: self._distance(
                    point,
                    self.centroids[cluster_index],
                ),
            )
            for point in data
        ]

    def inertia(self, data: Sequence[Sequence[float]]) -> float:
        """Within-cluster sum of squared distances."""
        labels = self.predict(data)

        return sum(
            self._distance(point, self.centroids[label]) ** 2
            for point, label in zip(data, labels)
        )


def demonstrate_kmeans() -> None:
    print_subsection("6.1 K-means Clustering")

    data = [
        [1.0, 1.2],
        [1.2, 0.8],
        [0.8, 1.1],
        [5.0, 5.2],
        [5.1, 4.8],
        [4.7, 5.0],
    ]

    model = KMeans(n_clusters=2, random_state=42)
    model.fit(data)

    print("Centroids:")
    for centroid in model.centroids:
        print(" ", [round(value, 3) for value in centroid])

    print("Cluster assignments:", model.labels_)
    print("Inertia:", round(model.inertia(data), 3))

    print(
        """
Important K-means assumptions and limitations:

    - k must be specified.
    - Euclidean distance is commonly used.
    - Feature scaling can substantially change results.
    - It tends to work best for compact, roughly spherical clusters.
    - Outliers can influence centroids.
    - Different initialization can produce different solutions.

Choosing k:
    The elbow method, silhouette analysis, stability, and domain knowledge
    are commonly considered.

K-means is not a universal clustering algorithm.
Density-based methods, hierarchical clustering, and probabilistic models can
be more appropriate for different structures.
"""
    )


# =============================================================================
# 7. DIMENSIONALITY REDUCTION
# =============================================================================

def pca_from_scratch_demo() -> None:
    print_section("7. DIMENSIONALITY REDUCTION AND PCA")

    print(
        """
Principal Component Analysis (PCA) transforms correlated variables into a
new coordinate system.

The first principal component captures the greatest possible variance subject
to an orthogonality constraint. Subsequent components capture remaining
variance under additional orthogonality constraints.

Typical applications:
    - visualization
    - compression
    - noise reduction
    - preprocessing
    - exploratory analysis

A conceptual PCA pipeline is:

    center data
        ->
    optionally scale data
        ->
    calculate covariance structure
        ->
    obtain eigenvectors/eigenvalues
        ->
    sort components by explained variance
        ->
    project data into the selected component space

Important distinction:
    PCA is unsupervised. It does not use a target variable.
"""
    )

    try:
        import numpy as np
    except ImportError:
        print("NumPy is unavailable; PCA numerical demonstration skipped.")
        return

    data = np.array(
        [
            [2.0, 2.1, 4.0],
            [3.0, 3.1, 6.0],
            [4.0, 4.2, 8.0],
            [5.0, 5.1, 10.0],
            [6.0, 5.9, 12.0],
        ]
    )

    centered = data - np.mean(data, axis=0)
    covariance = np.cov(centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    print("Explained variance ratio:")
    print(np.round(explained_variance_ratio, 4))

    projected = centered @ eigenvectors[:, :2]

    print("Two-dimensional representation:")
    print(np.round(projected, 3))

    print(
        """
PCA caveat:
    High variance is not necessarily the same thing as high predictive value.

For supervised prediction, a dimension that explains little variance could still
contain information strongly related to the target.

PCA components are linear combinations of original features and can therefore
be difficult to interpret.
"""
    )


# =============================================================================
# 8. SEMI-SUPERVISED LEARNING
# =============================================================================

def semi_supervised_learning_demo() -> None:
    print_section("8. SEMI-SUPERVISED LEARNING")

    print(
        """
Semi-supervised learning combines:

    labeled examples:
        (X_labeled, y_labeled)

    unlabeled examples:
        X_unlabeled

The setting is useful when labels are expensive but raw observations are
abundant.

Examples:
    - medical images requiring expert annotation
    - speech data requiring transcription
    - web pages with only a small set of manually categorized examples
    - industrial sensor data where fault labels are rare

The central assumption varies by method. Common assumptions include:

    Cluster assumption:
        Nearby observations are likely to share a label.

    Smoothness assumption:
        The target function changes smoothly over high-density regions.

    Low-density separation:
        Decision boundaries should preferably pass through low-density regions.

Common methods:
    - pseudo-labeling
    - label propagation
    - label spreading
    - consistency regularization
    - graph-based methods
    - semi-supervised generative models

A critical risk is confirmation bias:
    incorrect pseudo-labels can become training targets and reinforce errors.
"""
    )

    # A small graph-based label propagation example.
    labeled = {
        0: "red",
        4: "blue",
    }

    points = [
        [1.0, 1.0],
        [1.2, 0.9],
        [1.4, 1.1],
        [3.0, 3.1],
        [4.0, 4.1],
    ]

    def distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    labels = dict(labeled)

    # Educational nearest-labeled-neighbor propagation.
    # Real label propagation methods can use graph structures and iterative
    # probability propagation rather than this simple one-pass rule.
    for index, point in enumerate(points):
        if index in labels:
            continue

        nearest_index = min(
            labeled,
            key=lambda labeled_index: distance(
                point,
                points[labeled_index],
            ),
        )
        labels[index] = labels[nearest_index]

    print("Initial labels:", labeled)
    print("Propagated labels:", labels)

    print(
        """
This example intentionally simplifies semi-supervised learning.

It demonstrates the principle:
    unlabeled observations can receive information from nearby labeled
    observations.

In production systems, uncertainty should be measured before accepting
pseudo-labels. A confidence threshold can reduce error propagation.
"""
    )


# =============================================================================
# 9. SELF-SUPERVISED LEARNING
# =============================================================================

def self_supervised_learning_demo() -> None:
    print_section("9. SELF-SUPERVISED LEARNING")

    print(
        """
Self-supervised learning constructs a learning target from the input data.

Instead of manually obtaining:
    sentence -> human-provided label

the system can construct:
    corrupted sentence -> original information

Examples include:

Masked prediction:
    Input:
        "The cat sat on the [MASK]."
    Target:
        "mat"

Next-token prediction:
    Input:
        "Machine learning is"
    Target:
        "useful"

Contrastive learning:
    Two augmented views of the same object are treated as related while
    unrelated objects are pushed apart in representation space.

Autoencoding:
    Corrupted or compressed input -> reconstruction of original input.

The important conceptual distinction is:

    Supervised:
        target comes from an external labeling process.

    Self-supervised:
        target is generated from the observations.

Self-supervised learning is particularly useful for representation learning.
A model can first learn general representations from large unlabeled datasets
and then be adapted to downstream tasks.
"""
    )


# -----------------------------------------------------------------------------
# 9.1 Masked token prediction as a toy self-supervised task
# -----------------------------------------------------------------------------

def create_masked_prediction_examples(
    sentences: Sequence[str],
    mask_token: str = "[MASK]",
) -> List[Tuple[str, str]]:
    """
    Create toy masked-word prediction examples.

    This is deliberately deterministic and simple:
        the middle word is masked.

    It demonstrates the construction of self-generated targets.
    """
    examples = []

    for sentence in sentences:
        words = sentence.split()

        if len(words) < 3:
            continue

        middle = len(words) // 2
        target = words[middle]
        masked_words = words.copy()
        masked_words[middle] = mask_token

        examples.append((" ".join(masked_words), target))

    return examples


def demonstrate_self_supervision() -> None:
    print_subsection("9.1 Constructing Self-Supervised Targets")

    sentences = [
        "the cat drinks milk",
        "the dog chases balls",
        "machine learning uses data",
        "python supports many libraries",
    ]

    examples = create_masked_prediction_examples(sentences)

    for input_text, target in examples:
        print(f"Input : {input_text}")
        print(f"Target: {target}")
        print()

    print(
        """
Notice that no human supplied a separate classification label.

The target was derived directly from the original sequence.

This construction is called a pretext task.

A pretext task should create a learning signal that encourages the model to
learn useful representations.

Important limitation:
    Solving a pretext task successfully does not guarantee useful downstream
    representations. The relationship between pretraining objective and
    downstream task matters.
"""
    )


# =============================================================================
# 10. SUPERVISED VS UNSUPERVISED VS SEMI-SUPERVISED VS SELF-SUPERVISED
# =============================================================================

def paradigm_comparison() -> None:
    print_section("10. COMPARISON OF LEARNING PARADIGMS")

    comparison = [
        (
            "Supervised",
            "Labeled X and y",
            "Predict target",
            "Classification, regression",
        ),
        (
            "Unsupervised",
            "X only",
            "Discover structure",
            "Clustering, PCA, density estimation",
        ),
        (
            "Semi-supervised",
            "Small labeled set + large unlabeled set",
            "Use both sources",
            "Label propagation, pseudo-labeling",
        ),
        (
            "Self-supervised",
            "Raw data",
            "Create target from data",
            "Masked prediction, contrastive learning",
        ),
        (
            "Reinforcement",
            "States, actions, rewards",
            "Learn policy/value",
            "Games, robotics, sequential control",
        ),
    ]

    headers = (
        "Paradigm",
        "Training signal",
        "Typical objective",
        "Examples",
    )

    widths = [18, 31, 27, 38]

    print(
        " | ".join(
            header.ljust(width)
            for header, width in zip(headers, widths)
        )
    )
    print("-" * sum(widths) + "-" * 9)

    for row in comparison:
        print(
            " | ".join(
                value.ljust(width)
                for value, width in zip(row, widths)
            )
        )

    print(
        """
These categories can also be combined.

For example:
    self-supervised pretraining
        ->
    supervised fine-tuning

A modern system can therefore use more than one learning paradigm at
different stages.

Semi-supervised and self-supervised learning are not synonymous:
    Semi-supervised learning explicitly uses a labeled subset.
    Self-supervised learning constructs targets from the data itself.
"""
    )


# =============================================================================
# 11. REINFORCEMENT LEARNING FUNDAMENTALS
# =============================================================================

def reinforcement_learning_concepts() -> None:
    print_section("11. REINFORCEMENT LEARNING")

    print(
        """
Reinforcement learning (RL) models sequential decision making.

Core components:

Agent:
    The decision-maker.

Environment:
    The world in which the agent operates.

State s:
    Information describing the current situation.

Action a:
    A decision available to the agent.

Reward r:
    Numerical feedback associated with an interaction.

Policy pi(a|s):
    A strategy for choosing actions from states.

Transition:
    The environment moves from one state to another after an action.

Episode:
    A sequence from an initial state until a terminal condition.

The agent seeks a policy that maximizes expected cumulative reward.

Discounted return:

    G_t = r_(t+1) + gamma*r_(t+2)
          + gamma^2*r_(t+3) + ...

where:
    gamma in [0, 1]
    controls the importance of future rewards.

Two central learning objects are:

Value function:
    V(s)
    Expected return starting from state s under a policy.

Action-value function:
    Q(s, a)
    Expected return after taking action a in state s and following a policy.

A major RL challenge is the exploration-exploitation trade-off:

Exploration:
    Try uncertain actions to learn more.

Exploitation:
    Choose actions believed to produce high reward.
"""
    )


# =============================================================================
# 12. Q-LEARNING FROM SCRATCH
# =============================================================================

class GridWorld:
    """
    Small deterministic environment for demonstrating Q-learning.

    Grid:
        S = start
        G = goal
        # = obstacle
        . = ordinary cell

    The agent receives:
        +10 for reaching the goal
        -1 for an ordinary step
        -5 for hitting an obstacle or boundary
    """

    ACTIONS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def __init__(self) -> None:
        self.grid = [
            ["S", ".", ".", "."],
            [".", "#", ".", "."],
            [".", "#", ".", "."],
            [".", ".", ".", "G"],
        ]
        self.start = (0, 0)
        self.goal = (3, 3)
        self.position = self.start

    def reset(self) -> Tuple[int, int]:
        self.position = self.start
        return self.position

    def step(
        self,
        action: str,
    ) -> Tuple[Tuple[int, int], float, bool]:
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        row, column = self.position
        delta_row, delta_column = self.ACTIONS[action]

        new_row = row + delta_row
        new_column = column + delta_column

        # Boundary or obstacle collision.
        if (
            new_row < 0
            or new_row >= len(self.grid)
            or new_column < 0
            or new_column >= len(self.grid[0])
            or self.grid[new_row][new_column] == "#"
        ):
            return self.position, -5.0, False

        self.position = (new_row, new_column)

        if self.position == self.goal:
            return self.position, 10.0, True

        return self.position, -1.0, False


class QLearningAgent:
    """
    Tabular Q-learning agent.

    Update rule:

        Q(s,a) <- Q(s,a)
                 + alpha * [
                       r + gamma * max_a' Q(s',a')
                       - Q(s,a)
                   ]

    alpha:
        learning rate

    gamma:
        discount factor

    epsilon:
        probability of exploratory action in epsilon-greedy policy
    """

    def __init__(
        self,
        actions: Sequence[str],
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        minimum_epsilon: float = 0.05,
        random_seed: int = 42,
    ) -> None:
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be in [0, 1].")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1].")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1].")

        self.actions = list(actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.q_table: Dict[Tuple[Tuple[int, int], str], float] = defaultdict(float)
        self.random = random.Random(random_seed)

    def choose_action(
        self,
        state: Tuple[int, int],
        explore: bool = True,
    ) -> str:
        if explore and self.random.random() < self.epsilon:
            return self.random.choice(self.actions)

        values = [
            (action, self.q_table[(state, action)])
            for action in self.actions
        ]

        # Randomized tie-breaking prevents a fixed action from dominating
        # before meaningful values have been learned.
        best_value = max(value for _, value in values)
        best_actions = [
            action
            for action, value in values
            if math.isclose(value, best_value)
        ]

        return self.random.choice(best_actions)

    def update(
        self,
        state: Tuple[int, int],
        action: str,
        reward: float,
        next_state: Tuple[int, int],
        done: bool,
    ) -> None:
        current = self.q_table[(state, action)]

        if done:
            target = reward
        else:
            future = max(
                self.q_table[(next_state, next_action)]
                for next_action in self.actions
            )
            target = reward + self.discount_factor * future

        self.q_table[(state, action)] = (
            current
            + self.learning_rate * (target - current)
        )

    def decay_exploration(self) -> None:
        self.epsilon = max(
            self.minimum_epsilon,
            self.epsilon * self.epsilon_decay,
        )


def train_q_learning(
    episodes: int = 1000,
) -> Tuple[GridWorld, QLearningAgent]:
    environment = GridWorld()
    agent = QLearningAgent(
        actions=list(GridWorld.ACTIONS.keys()),
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        minimum_epsilon=0.05,
    )

    for _ in range(episodes):
        state = environment.reset()

        for _ in range(100):
            action = agent.choose_action(state, explore=True)
            next_state, reward, done = environment.step(action)

            agent.update(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state

            if done:
                break

        agent.decay_exploration()

    return environment, agent


def evaluate_q_learning(
    environment: GridWorld,
    agent: QLearningAgent,
) -> None:
    print_subsection("12.1 Evaluating the Learned Policy")

    state = environment.reset()
    path = [state]
    total_reward = 0.0

    for _ in range(30):
        action = agent.choose_action(state, explore=False)
        state, reward, done = environment.step(action)

        path.append(state)
        total_reward += reward

        if done:
            break

    print("Path:", path)
    print("Total reward:", total_reward)
    print("Reached goal:", state == environment.goal)
    print("Final epsilon:", round(agent.epsilon, 4))


# =============================================================================
# 13. MARKOV DECISION PROCESSES
# =============================================================================

def mdp_demo() -> None:
    print_section("13. MARKOV DECISION PROCESSES")

    print(
        """
A common formal model for reinforcement learning is the Markov Decision
Process (MDP):

    (S, A, P, R, gamma)

S:
    State space.

A:
    Action space.

P:
    Transition dynamics:
        P(s' | s, a)

R:
    Reward function.

gamma:
    Discount factor.

The Markov property means that the relevant future depends on the current
state and action rather than requiring the entire historical sequence.

If the state does not contain enough information to satisfy the Markov
property, the problem may require memory or a richer representation.

Examples:
    - recurrent policies
    - belief states
    - partially observable Markov decision processes (POMDPs)
"""
    )


# =============================================================================
# 14. EXPLORATION STRATEGIES
# =============================================================================

def exploration_demo() -> None:
    print_section("14. EXPLORATION VS EXPLOITATION")

    print(
        """
Epsilon-greedy:
    With probability epsilon:
        choose a random action.
    Otherwise:
        choose the best-known action.

Advantages:
    - extremely simple
    - easy to implement

Limitations:
    - random exploration may be inefficient
    - all non-greedy actions can be treated similarly

Other approaches include:

    Softmax/Boltzmann exploration:
        Sample actions according to their estimated values.

    Upper Confidence Bound:
        Prefer actions with both high estimated value and uncertainty.

    Thompson sampling:
        Sample from a posterior belief about action quality.

Exploration is not merely random behavior.
The design of the exploration mechanism can significantly affect learning
efficiency and safety.
"""
    )


# =============================================================================
# 15. POLICY GRADIENT AND ACTOR-CRITIC CONCEPTS
# =============================================================================

def policy_gradient_concepts() -> None:
    print_section("15. POLICY GRADIENTS AND ACTOR-CRITIC METHODS")

    print(
        """
Value-based methods:
    Learn values such as Q(s,a), then derive a policy.

Policy-based methods:
    Directly optimize a parameterized policy pi_theta(a|s).

Policy gradient methods use an objective based on expected return and adjust
parameters in the direction associated with better outcomes.

A conceptual gradient expression is:

    grad J(theta)
        approximately
    E[grad log pi_theta(a|s) * advantage]

The advantage measures whether an action performed better or worse than a
baseline expectation.

Actor-critic methods combine two roles:

Actor:
    Represents the policy and selects actions.

Critic:
    Estimates value or advantage to guide the actor.

This can reduce variance relative to some pure policy-gradient approaches.

Modern deep RL commonly combines:
    neural-network function approximation
    policy optimization
    value estimation
    experience collection
    exploration strategies
"""
    )


# =============================================================================
# 16. DEEP LEARNING RELATIONSHIP TO THE PARADIGMS
# =============================================================================

def deep_learning_relationship() -> None:
    print_section("16. DEEP LEARNING AND THE FIVE PARADIGMS")

    print(
        """
Deep learning is not itself a learning paradigm in the same sense as the
five categories in this file.

It refers primarily to using multi-layer neural networks to learn
representations and functions.

Deep networks can be trained under different paradigms:

Supervised:
    image -> class
    text -> category
    tabular data -> numerical target

Unsupervised:
    autoencoder reconstruction
    clustering of learned embeddings

Semi-supervised:
    labeled + unlabeled examples

Self-supervised:
    masked prediction
    next-token prediction
    contrastive representation learning

Reinforcement:
    neural policy
    neural value function
    deep Q-learning
    actor-critic algorithms

Therefore:

    "deep learning" describes model architecture/representation capacity.

    "supervised/self-supervised/RL" describes how learning signals are formed.

They are not mutually exclusive categories.
"""
    )


# =============================================================================
# 17. REPRESENTATION LEARNING
# =============================================================================

def representation_learning_demo() -> None:
    print_section("17. REPRESENTATION LEARNING")

    print(
        """
Representation learning attempts to transform raw observations into useful
internal representations.

Raw input:
    image pixels
    audio samples
    tokens
    sensor measurements
    transaction attributes

Representation:
    a numerical vector or structured internal state

Downstream task:
    classification
    retrieval
    ranking
    clustering
    prediction
    decision making

Self-supervised learning is often used to learn representations because
unlabeled data can provide large amounts of training signal.

A useful representation should preserve information relevant to downstream
tasks while discarding irrelevant variation.

This introduces an important trade-off:
    excessive compression can discard useful information;
    insufficient abstraction can preserve noise.
"""
    )

    vectors = {
        "customer_A": [0.92, 0.10, 0.82],
        "customer_B": [0.88, 0.14, 0.79],
        "customer_C": [0.15, 0.90, 0.20],
    }

    def cosine_similarity(
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        numerator = sum(a * b for a, b in zip(first, second))
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))

        if first_norm == 0 or second_norm == 0:
            raise ValueError("Cosine similarity is undefined for zero vectors.")

        return numerator / (first_norm * second_norm)

    print(
        "Cosine similarity A/B:",
        round(cosine_similarity(vectors["customer_A"], vectors["customer_B"]), 3),
    )

    print(
        "Cosine similarity A/C:",
        round(cosine_similarity(vectors["customer_A"], vectors["customer_C"]), 3),
    )


# =============================================================================
# 18. CLASS IMBALANCE
# =============================================================================

def class_imbalance_demo() -> None:
    print_section("18. CLASS IMBALANCE")

    labels = ["normal"] * 95 + ["fraud"] * 5
    majority_baseline = ["normal"] * len(labels)

    accuracy = classification_accuracy(labels, majority_baseline)

    print("Majority-class baseline accuracy:", accuracy)

    print(
        """
A 95% accuracy baseline can still be useless when the rare class is the
important class.

Potential techniques:

Data-level:
    - oversampling
    - undersampling
    - synthetic sampling

Algorithm-level:
    - class-weighted loss
    - cost-sensitive learning
    - focal loss in suitable deep-learning settings

Evaluation:
    - precision
    - recall
    - F1
    - PR-AUC
    - class-specific confusion matrix

Threshold selection:
    The default probability threshold of 0.5 is not universally optimal.

Business costs should determine the acceptable balance between false positives
and false negatives.
"""
    )


# =============================================================================
# 19. CROSS-VALIDATION
# =============================================================================

def k_fold_cross_validation_demo() -> None:
    print_section("19. K-FOLD CROSS-VALIDATION")

    data = list(range(1, 11))
    k = 5

    folds = [data[index::k] for index in range(k)]

    scores = []

    for fold_index in range(k):
        validation = folds[fold_index]
        training = [
            item
            for index, fold in enumerate(folds)
            if index != fold_index
            for item in fold
        ]

        # Toy scoring rule:
        # estimate a "model" from the mean of the training set and measure
        # average absolute error on the validation fold.
        training_mean = statistics.mean(training)
        error = statistics.mean(
            abs(value - training_mean)
            for value in validation
        )
        scores.append(error)

        print(
            f"Fold {fold_index + 1}: "
            f"validation={validation}, "
            f"MAE={error:.3f}"
        )

    print("Mean cross-validation error:", round(statistics.mean(scores), 3))

    print(
        """
K-fold cross-validation repeatedly trains on k-1 partitions and validates
on the remaining partition.

Advantages:
    - more efficient use of limited data
    - estimates performance variability
    - useful for model selection

Variants:
    - stratified k-fold for classification
    - grouped cross-validation when groups must not be split
    - time-series validation for temporal data

Randomly mixing time-dependent data can cause temporal leakage.
"""
    )


# =============================================================================
# 20. HYPERPARAMETERS VS PARAMETERS
# =============================================================================

def parameters_vs_hyperparameters() -> None:
    print_section("20. PARAMETERS VS HYPERPARAMETERS")

    print(
        """
Parameters:
    Learned from training data.

Examples:
    linear regression slope
    neural-network weights
    K-means centroids

Hyperparameters:
    Chosen outside the direct parameter-fitting process.

Examples:
    K in KNN
    number of clusters in K-means
    learning rate
    tree depth
    regularization strength
    neural-network architecture

Hyperparameter search methods include:
    - grid search
    - random search
    - Bayesian optimization
    - population-based methods

Random search can be more efficient than grid search when only a subset of
hyperparameters strongly affects performance.
"""
    )


# =============================================================================
# 21. REGULARIZATION
# =============================================================================

def regularization_demo() -> None:
    print_section("21. REGULARIZATION")

    print(
        """
Regularization discourages undesirable model complexity.

L2 regularization:
    Add a penalty proportional to the squared parameter magnitude.

        loss + lambda * sum(w_i^2)

L1 regularization:
    Add a penalty proportional to absolute parameter magnitude.

        loss + lambda * sum(|w_i|)

L1 can encourage sparse parameters.

L2 generally shrinks parameters without producing as much exact sparsity.

Other forms of regularization include:
    - dropout
    - early stopping
    - data augmentation
    - architectural constraints
    - noise injection

The regularization strength is a hyperparameter.
Too little regularization can permit overfitting.
Too much can cause underfitting.
"""
    )


# =============================================================================
# 22. DATA QUALITY
# =============================================================================

def data_quality_demo() -> None:
    print_section("22. DATA QUALITY AND PREPROCESSING")

    raw_records = [
        {"age": 25, "income": 50000},
        {"age": None, "income": 60000},
        {"age": 40, "income": 70000},
        {"age": 40, "income": 70000},  # duplicate
    ]

    print("Raw records:", raw_records)

    ages = [
        record["age"]
        for record in raw_records
        if record["age"] is not None
    ]

    median_age = statistics.median(ages)

    cleaned = []

    for record in raw_records:
        normalized = record.copy()

        if normalized["age"] is None:
            normalized["age"] = median_age

        cleaned.append(normalized)

    unique_records = list(
        {
            (record["age"], record["income"]): record
            for record in cleaned
        }.values()
    )

    print("After missing-value handling:", cleaned)
    print("After duplicate removal:", unique_records)

    print(
        """
Common data problems:

    missing values
    duplicates
    outliers
    inconsistent units
    incorrect labels
    distribution shifts
    measurement errors
    selection bias

Preprocessing is part of the machine-learning system, not merely a cosmetic
data-cleaning step.

A production pipeline should preserve the exact transformations required to
turn raw production input into model-ready input.
"""
    )


# =============================================================================
# 23. CONCEPT DRIFT AND DISTRIBUTION SHIFT
# =============================================================================

def distribution_shift_demo() -> None:
    print_section("23. DISTRIBUTION SHIFT AND CONCEPT DRIFT")

    print(
        """
Training data distribution:
    P_train(X, y)

Production data distribution:
    P_production(X, y)

If these distributions differ materially, performance may degrade.

Covariate shift:
    P(X) changes while the relationship P(y|X) remains approximately stable.

Label shift:
    P(y) changes.

Concept drift:
    P(y|X) changes.

Examples:
    - consumer behavior changes
    - fraud strategies evolve
    - sensor calibration changes
    - economic conditions change

Production monitoring can include:
    - feature distribution monitoring
    - missing-value rates
    - prediction distribution
    - calibration
    - delayed ground-truth performance
    - data quality checks

A model is not finished merely because offline test performance is high.
"""
    )


# =============================================================================
# 24. CALIBRATION
# =============================================================================

def probability_calibration_demo() -> None:
    print_section("24. PROBABILITY CALIBRATION")

    print(
        """
A classifier can rank examples correctly while producing poorly calibrated
probabilities.

If a model predicts probability 0.8 for many observations, a calibrated model
should have approximately 80% positives among those observations.

Calibration matters when probabilities drive:
    - risk decisions
    - pricing
    - resource allocation
    - medical triage
    - financial decisions

Ranking quality and probability calibration are distinct properties.

Common calibration methods include:
    - Platt scaling
    - isotonic regression

Calibration must be evaluated on appropriate held-out data.
"""
    )


# =============================================================================
# 25. SECURITY AND ADVERSARIAL CONSIDERATIONS
# =============================================================================

def machine_learning_security_demo() -> None:
    print_section("25. MACHINE LEARNING SECURITY")

    print(
        """
Machine-learning systems can have security-specific threats.

Training-data poisoning:
    Attackers manipulate training data to influence model behavior.

Evasion attacks:
    Inputs are deliberately modified to cause incorrect predictions.

Model extraction:
    An attacker attempts to approximate a model through repeated queries.

Membership inference:
    An attacker attempts to determine whether a particular example was part
    of training data.

Data leakage:
    Sensitive information may be memorized or accidentally exposed.

Security controls can include:
    - authenticated data pipelines
    - provenance tracking
    - dataset validation
    - access control
    - rate limiting
    - anomaly detection
    - privacy-aware training
    - model monitoring
    - secure serialization
    - reproducible builds

Never load untrusted serialized model objects blindly.
Some serialization formats can execute arbitrary code during deserialization.
"""
    )


# =============================================================================
# 26. PRIVACY CONSIDERATIONS
# =============================================================================

def privacy_considerations() -> None:
    print_section("26. PRIVACY CONSIDERATIONS")

    print(
        """
Machine-learning pipelines can process sensitive information.

Privacy considerations include:
    - data minimization
    - purpose limitation
    - access controls
    - retention policies
    - anonymization or pseudonymization where appropriate
    - privacy-preserving computation
    - differential privacy in suitable applications

Differential privacy introduces controlled randomness so that the contribution
of an individual record is difficult to infer while aggregate patterns remain
useful.

Privacy is not identical to security:
    Security protects systems and information from unauthorized access.
    Privacy concerns appropriate collection, use, disclosure, and inference
    about individuals.
"""
    )


# =============================================================================
# 27. FAIRNESS AND EVALUATION
# =============================================================================

def fairness_demo() -> None:
    print_section("27. FAIRNESS AND MODEL EVALUATION")

    print(
        """
Performance can differ across population groups.

Useful evaluation questions include:

    Does error rate differ between groups?
    Does false-positive rate differ?
    Does false-negative rate differ?
    Are probabilities similarly calibrated?
    Are important subgroups sufficiently represented?

There is no universally correct single fairness metric.

Different fairness criteria can conflict mathematically and operationally.

Responsible evaluation therefore requires:
    - clearly defined task
    - relevant population definitions
    - appropriate metrics
    - subgroup analysis
    - awareness of data limitations
    - domain and legal context

A high aggregate accuracy can hide serious subgroup failures.
"""
    )


# =============================================================================
# 28. MODEL SELECTION AND BASELINES
# =============================================================================

def baseline_and_model_selection_demo() -> None:
    print_section("28. BASELINES AND MODEL SELECTION")

    print(
        """
Always establish a simple baseline before evaluating complex models.

Examples:

Classification:
    majority-class predictor

Regression:
    mean-target predictor

Time series:
    previous-value predictor

Clustering:
    simple heuristic segmentation

Why baselines matter:
    A sophisticated model is valuable only if it improves the relevant
    objective enough to justify its complexity.

Model selection should consider:
    - predictive performance
    - latency
    - memory
    - interpretability
    - robustness
    - training cost
    - inference cost
    - operational complexity
    - security
    - maintenance burden

The highest offline score is not necessarily the best production model.
"""
    )


# =============================================================================
# 29. EDGE CASES
# =============================================================================

def edge_cases_demo() -> None:
    print_section("29. IMPORTANT EDGE CASES")

    print_subsection("29.1 Regression with Constant Feature")

    model = SimpleLinearRegression()

    try:
        model.fit([1, 1, 1], [10, 20, 30])
    except ValueError as error:
        print("Handled error:", error)

    print_subsection("29.2 Empty Metric Input")

    try:
        mean_squared_error([], [])
    except ValueError as error:
        print("Handled error:", error)

    print_subsection("29.3 KNN Before Fitting")

    classifier = KNearestNeighborsClassifier(k=3)

    try:
        classifier.predict_one([1, 2])
    except RuntimeError as error:
        print("Handled error:", error)

    print_subsection("29.4 K-means with Too Many Clusters")

    try:
        KMeans(n_clusters=10).fit([[1, 2], [3, 4]])
    except ValueError as error:
        print("Handled error:", error)

    print(
        """
Important edge cases in machine learning include:

    - empty datasets
    - constant features
    - missing values
    - infinite numerical values
    - duplicate observations
    - conflicting labels
    - rare classes
    - unseen categories
    - high-dimensional data
    - extremely large feature magnitudes
    - distribution shift
    - invalid timestamps
    - leakage across related records
    - data from future periods entering training
    - feedback loops from deployed predictions

Production systems should validate these conditions explicitly.
"""
    )


# =============================================================================
# 30. PERFORMANCE CONSIDERATIONS
# =============================================================================

def performance_considerations() -> None:
    print_section("30. PERFORMANCE CONSIDERATIONS")

    print(
        """
Machine-learning performance has several dimensions.

Training complexity:
    How expensive is fitting the model?

Inference latency:
    How quickly can one prediction be generated?

Throughput:
    How many predictions can be processed per unit time?

Memory:
    How much RAM/VRAM/storage is required?

Data pipeline cost:
    How expensive is feature extraction and preprocessing?

Examples:

KNN:
    Training is cheap, but prediction can require comparing a new point
    against many training observations.

K-means:
    Repeated distance calculations can become expensive with large datasets.

Deep neural networks:
    Training can require substantial computation and specialized hardware.

Dimensionality:
    More features can increase computational cost and sometimes worsen
    generalization.

Production optimization techniques can include:
    - vectorization
    - batching
    - caching
    - approximate nearest neighbors
    - model quantization
    - pruning
    - distillation
    - efficient feature computation
    - parallel processing

Optimization should be driven by measured bottlenecks rather than assumptions.
"""
    )


# =============================================================================
# 31. REPRODUCIBILITY
# =============================================================================

def reproducibility_demo() -> None:
    print_section("31. REPRODUCIBILITY")

    seed = 42

    first_rng = random.Random(seed)
    second_rng = random.Random(seed)

    first_values = [first_rng.random() for _ in range(5)]
    second_values = [second_rng.random() for _ in range(5)]

    print("First sequence :", first_values)
    print("Second sequence:", second_values)
    print("Identical:", first_values == second_values)

    print(
        """
Reproducibility requires more than setting a random seed.

A robust experiment should record:
    - dataset version
    - feature definitions
    - preprocessing configuration
    - model architecture
    - hyperparameters
    - random seeds
    - software versions
    - hardware where relevant
    - evaluation protocol
    - source-code version

Even with fixed seeds, numerical libraries, hardware, parallelism, and
distributed computation can introduce differences.
"""
    )


# =============================================================================
# 32. TESTING MACHINE-LEARNING COMPONENTS
# =============================================================================

def testing_demo() -> None:
    print_section("32. TESTING MACHINE-LEARNING CODE")

    print(
        """
Machine-learning systems require both conventional software tests and
statistical evaluation.

Unit tests:
    Test individual functions.

Integration tests:
    Test interactions between components.

Data validation tests:
    Verify schema, ranges, missingness, uniqueness, and constraints.

Model tests:
    Verify expected prediction shape, class domain, numerical stability,
    and basic behavioral requirements.

Regression tests:
    Detect unintended changes in model behavior.

Statistical tests:
    Evaluate predictive performance and uncertainty.

A small example is shown below.
"""
    )

    assert mean_squared_error([1, 2, 3], [1, 2, 3]) == 0.0
    assert classification_accuracy(
        ["a", "b", "a"],
        ["a", "b", "a"],
    ) == 1.0

    classifier = KNearestNeighborsClassifier(k=1)
    classifier.fit([[0, 0], [10, 10]], ["near_zero", "near_ten"])

    assert classifier.predict_one([0.1, 0.1]) == "near_zero"
    assert classifier.predict_one([9.9, 10.1]) == "near_ten"

    print("Basic assertions passed.")


# =============================================================================
# 33. MODEL INTERPRETABILITY
# =============================================================================

def interpretability_demo() -> None:
    print_section("33. INTERPRETABILITY AND EXPLAINABILITY")

    print(
        """
Interpretability asks how understandable a model's behavior is.

Simple models:
    linear models
    small decision trees
    rule-based systems

can sometimes be easier to inspect directly.

Complex models:
    large neural networks
    ensembles
    nonlinear systems

may require explanation methods.

Important distinction:
    An explanation of a prediction is not necessarily a complete causal
    explanation of the world.

Feature importance can indicate association with model predictions without
proving that changing the feature would cause the prediction to change.

For high-impact applications, explanation should be evaluated carefully rather
than treated as automatically truthful.
"""
    )


# =============================================================================
# 34. REAL-WORLD PARADIGM MAPPING
# =============================================================================

def real_world_mapping() -> None:
    print_section("34. REAL-WORLD APPLICATION MAPPING")

    examples = {
        "Supervised": [
            "credit-risk prediction",
            "medical image classification",
            "sales forecasting",
            "spam detection",
        ],
        "Unsupervised": [
            "customer segmentation",
            "anomaly discovery",
            "exploratory dimensionality reduction",
            "group discovery",
        ],
        "Semi-supervised": [
            "large image collection with limited expert labels",
            "document classification with few manually labeled documents",
            "industrial inspection with scarce fault labels",
        ],
        "Self-supervised": [
            "masked sequence prediction",
            "representation learning from images",
            "contrastive learning",
            "predicting withheld portions of observations",
        ],
        "Reinforcement": [
            "robotic control",
            "game playing",
            "sequential resource allocation",
            "adaptive decision making",
        ],
    }

    for paradigm, applications in examples.items():
        print(f"\n{paradigm}:")
        for application in applications:
            print(f"  - {application}")

    print(
        """
Choosing a paradigm should begin with the structure of the problem:

    Do we have reliable target labels?
        -> supervised may be appropriate.

    Do we have mostly unlabeled data and want to discover structure?
        -> unsupervised may be appropriate.

    Do we have a small labeled subset plus abundant unlabeled data?
        -> semi-supervised methods may help.

    Can the observations themselves create useful prediction targets?
        -> self-supervised learning may help.

    Is the problem sequential, with actions affecting future states and
    rewards arriving over time?
        -> reinforcement learning may be appropriate.
"""
    )


# =============================================================================
# 35. PRODUCTION MACHINE-LEARNING LIFECYCLE
# =============================================================================

def production_lifecycle() -> None:
    print_section("35. PRODUCTION MACHINE-LEARNING LIFECYCLE")

    lifecycle = [
        "Problem definition",
        "Data acquisition",
        "Data validation",
        "Label strategy or learning-signal design",
        "Feature/representation design",
        "Train/validation/test strategy",
        "Baseline model",
        "Model training",
        "Hyperparameter selection",
        "Evaluation",
        "Fairness and robustness analysis",
        "Security and privacy review",
        "Deployment",
        "Monitoring",
        "Drift detection",
        "Retraining or model replacement",
    ]

    for number, stage in enumerate(lifecycle, start=1):
        print(f"{number:2d}. {stage}")

    print(
        """
Production considerations extend beyond model accuracy.

A production system may need:
    - reproducible data pipelines
    - model versioning
    - feature versioning
    - deployment rollback
    - observability
    - access control
    - audit logging
    - latency budgets
    - cost controls
    - incident response
    - data and model drift monitoring

The deployed model is one component of a larger socio-technical system.
"""
    )


# =============================================================================
# 36. ADVANCED COMBINATION: SELF-SUPERVISED PRETRAINING + SUPERVISED FINETUNING
# =============================================================================

def combined_paradigm_demo() -> None:
    print_section("36. COMBINING LEARNING PARADIGMS")

    print(
        """
A common conceptual architecture is:

    Large unlabeled dataset
            |
            v
    self-supervised pretraining
            |
            v
    learned representation
            |
            v
    small labeled dataset
            |
            v
    supervised fine-tuning
            |
            v
    downstream model

This combines two paradigms rather than treating them as competing choices.

A similar pattern can occur in reinforcement learning:
    representation learning
        +
    policy/value learning

The key design question is which training signal provides useful information
at each stage.
"""
    )

    # Tiny numerical illustration of a representation transformation.
    raw_features = [
        [10, 1000],
        [20, 2000],
        [30, 3000],
    ]

    normalized_features = []

    means = [
        statistics.mean(column)
        for column in zip(*raw_features)
    ]

    standard_deviations = [
        statistics.pstdev(column)
        for column in zip(*raw_features)
    ]

    for row in raw_features:
        normalized_features.append(
            [
                (value - mean) / standard_deviation
                if standard_deviation != 0
                else 0.0
                for value, mean, standard_deviation in zip(
                    row,
                    means,
                    standard_deviations,
                )
            ]
        )

    print("Raw features:")
    for row in raw_features:
        print(" ", row)

    print("Normalized representation:")
    for row in normalized_features:
        print(" ", [round(value, 3) for value in row])


# =============================================================================
# 37. COMMON MISTAKES
# =============================================================================

def common_mistakes() -> None:
    print_section("37. COMMON MISTAKES")

    mistakes = [
        (
            "Using accuracy alone on severe class imbalance",
            "Inspect class-specific metrics and decision costs.",
        ),
        (
            "Tuning on the test set",
            "Reserve the test set for final evaluation.",
        ),
        (
            "Scaling before the data split",
            "Fit scaling parameters on training data only.",
        ),
        (
            "Calling every unlabeled method self-supervised",
            "Check whether an explicit target is constructed from the data.",
        ),
        (
            "Assuming K-means always discovers meaningful groups",
            "Validate cluster stability and domain relevance.",
        ),
        (
            "Assuming high PCA variance means high predictive value",
            "Evaluate downstream task performance.",
        ),
        (
            "Trusting pseudo-labels without uncertainty checks",
            "Use confidence thresholds and monitor confirmation bias.",
        ),
        (
            "Treating reward as the same thing as real-world success",
            "Validate reward design and unintended incentives.",
        ),
        (
            "Ignoring distribution shift",
            "Monitor production data and delayed outcomes.",
        ),
        (
            "Assuming a fixed random seed guarantees complete reproducibility",
            "Track data, software, hardware, configuration, and randomness.",
        ),
    ]

    for mistake, correction in mistakes:
        print(f"\nMistake: {mistake}")
        print(f"Better practice: {correction}")


# =============================================================================
# 38. DECISION FRAMEWORK
# =============================================================================

def paradigm_decision_framework() -> None:
    print_section("38. PARADIGM SELECTION DECISION FRAMEWORK")

    print(
        """
Question 1:
    Is there a reliable target for each training observation?

    YES -> supervised learning is a natural candidate.
    NO  -> continue.

Question 2:
    Is the objective to discover structure without a target?

    YES -> unsupervised learning is a natural candidate.
    NO  -> continue.

Question 3:
    Is there a small labeled subset and a large unlabeled subset?

    YES -> semi-supervised learning may be appropriate.
    NO  -> continue.

Question 4:
    Can useful prediction targets be generated from the observations?

    YES -> self-supervised learning may be appropriate.
    NO  -> continue.

Question 5:
    Does an agent repeatedly act, observe consequences, and optimize
    cumulative rewards?

    YES -> reinforcement learning may be appropriate.

The final choice depends on:
    - data availability
    - task structure
    - cost of labeling
    - feedback timing
    - operational constraints
    - safety requirements
    - evaluation quality
    - computational resources
"""
    )


# =============================================================================
# 39. FINAL INTEGRATED DEMONSTRATION
# =============================================================================

def integrated_demonstration() -> None:
    print_section("39. INTEGRATED PARADIGM DEMONSTRATION")

    print(
        """
Consider an online marketplace.

Raw data:
    customer activity
    product descriptions
    transactions
    clicks
    images
    search queries

Possible solutions:

SUPERVISED:
    Predict whether a customer will purchase a product.

UNSUPERVISED:
    Cluster customers according to behavioral similarity.

SEMI-SUPERVISED:
    Classify products when only a small subset has human categories.

SELF-SUPERVISED:
    Learn product or customer representations by predicting masked or
    withheld information.

REINFORCEMENT:
    Learn sequential recommendation policies where actions affect future
    engagement and rewards.

The same organization can use all five paradigms in one system.

This is why learning paradigms should be understood as different mechanisms
for obtaining learning signals rather than as isolated technologies.
"""
    )


# =============================================================================
# 40. EXECUTION ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Execute the educational curriculum in a deliberate progression.
    """
    random.seed(42)

    machine_learning_fundamentals()

    supervised_learning_concepts()
    demonstrate_supervised_regression()
    demonstrate_supervised_classification()
    classification_metrics_demo()

    train_validation_test_demo()
    bias_variance_demo()

    unsupervised_learning_concepts()
    demonstrate_kmeans()
    pca_from_scratch_demo()

    semi_supervised_learning_demo()

    self_supervised_learning_demo()
    demonstrate_self_supervision()

    paradigm_comparison()

    reinforcement_learning_concepts()
    mdp_demo()
    exploration_demo()

    environment, agent = train_q_learning(episodes=1000)
    evaluate_q_learning(environment, agent)

    policy_gradient_concepts()
    deep_learning_relationship()
    representation_learning_demo()

    class_imbalance_demo()
    k_fold_cross_validation_demo()
    parameters_vs_hyperparameters()
    regularization_demo()
    data_quality_demo()
    distribution_shift_demo()
    probability_calibration_demo()

    machine_learning_security_demo()
    privacy_considerations()
    fairness_demo()

    baseline_and_model_selection_demo()
    edge_cases_demo()
    performance_considerations()
    reproducibility_demo()
    testing_demo()
    interpretability_demo()

    real_world_mapping()
    production_lifecycle()
    combined_paradigm_demo()
    common_mistakes()
    paradigm_decision_framework()
    integrated_demonstration()

    print_section("END OF MACHINE LEARNING PARADIGMS STUDY SCRIPT")

    print(
        """
The script has demonstrated the defining learning signals, algorithms,
metrics, implementation patterns, limitations, and production considerations
associated with supervised, unsupervised, semi-supervised, self-supervised,
and reinforcement learning.
"""
    )


if __name__ == "__main__":
    main()
