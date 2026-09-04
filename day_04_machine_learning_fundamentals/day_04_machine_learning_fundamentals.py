"""
Machine Learning Fundamentals
=============================

A comprehensive, executable study script covering machine learning from
absolute beginner concepts through foundational advanced ideas.

The script demonstrates:

1. What machine learning is
2. Why machine learning exists
3. Traditional programming vs machine learning
4. Learning from data
5. Features, labels, examples, datasets
6. Training, validation, and test data
7. Supervised, unsupervised, and reinforcement learning
8. Regression and classification
9. Rules, parameters, models, and predictions
10. A simple rule-based system
11. Linear regression from scratch
12. Gradient descent from scratch
13. Classification from scratch
14. Nearest-neighbor learning
15. Training and prediction
16. Loss functions
17. Generalization
18. Overfitting and underfitting
19. Bias and variance
20. Data leakage
21. Feature engineering and scaling
22. Evaluation metrics
23. Cross-validation
24. Baselines
25. Decision boundaries
26. Regularization
27. Model complexity
28. Reproducibility
29. Practical ML workflow
30. Edge cases, limitations, and production considerations

The examples intentionally use only Python's standard library so that the
file can run without external dependencies.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence


# ============================================================================
# 1. BASIC IDEAS: WHAT IS MACHINE LEARNING?
# ============================================================================

def section(title: str) -> None:
    """Print a readable section heading."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


section("1. What Is Machine Learning?")

print(
    """
Machine learning is a way of building systems that learn patterns from data
and use those learned patterns to make predictions or decisions.

A traditional program generally follows:

    INPUT DATA + EXPLICIT RULES -> OUTPUT

A machine-learning system generally follows:

    INPUT DATA + EXPECTED OUTPUTS -> LEARNING ALGORITHM -> MODEL

Then:

    NEW INPUT -> TRAINED MODEL -> PREDICTION

The important distinction is that the programmer does not necessarily write
every decision rule explicitly. Instead, the learning algorithm estimates
useful parameters or structures from examples.

Example:

Suppose we want to determine whether an email is spam.

Traditional programming might attempt to define rules such as:

    if "free money" appears:
        spam
    if "winner" appears:
        spam

A machine-learning approach can instead receive many examples:

    email text -> spam
    email text -> not spam
    email text -> spam
    ...

The algorithm uses these examples to construct a model that can estimate
whether a previously unseen email is likely to be spam.
"""
)


# ============================================================================
# 2. WHY MACHINE LEARNING EXISTS
# ============================================================================

section("2. Why Does Machine Learning Exist?")

print(
    """
Machine learning becomes useful when explicit rules are difficult, numerous,
changing, expensive to maintain, or impossible to specify precisely.

Examples include:

- recognizing handwritten digits
- detecting fraudulent transactions
- predicting house prices
- recommending products
- recognizing speech
- estimating demand
- classifying images
- detecting unusual network behavior
- predicting customer churn

The key issue is often not that humans have no understanding of the task.
The issue is that manually converting all useful human knowledge into exact
programming rules can be impractical.

Consider recognizing a cat in an image.

A human can often recognize one immediately, but a programmer would need to
encode enormous numbers of possible variations:

- different poses
- different lighting
- different backgrounds
- different breeds
- partial visibility
- different camera angles
- different sizes
- occlusion

Machine learning attempts to infer useful patterns from examples instead.
"""
)


# ============================================================================
# 3. TRADITIONAL PROGRAMMING VS MACHINE LEARNING
# ============================================================================

section("3. Traditional Programming vs Machine Learning")

print(
    """
Traditional programming:

    Rules + Data -> Output

Machine learning during training:

    Data + Desired Outputs -> Learning Algorithm -> Model

Machine learning during inference:

    New Data + Trained Model -> Prediction

A useful conceptual comparison:

Traditional programming:
    Human discovers rules.
    Human writes rules.
    Computer executes rules.

Machine learning:
    Human supplies data and learning procedure.
    Algorithm estimates patterns.
    Computer uses the learned model.

This does not mean machine learning eliminates programming.

A real machine-learning system still requires programming for:

- data collection
- data validation
- preprocessing
- feature construction
- model training
- evaluation
- deployment
- monitoring
- error handling
- security
- retraining
"""
)


def traditional_house_price_estimator(
    area_sqft: float,
    bedrooms: int,
) -> float:
    """
    A traditional rule-based estimator.

    The coefficients are manually specified rather than learned from data.
    """
    base_price = 500_000
    price_per_sqft = 200
    bedroom_adjustment = 50_000

    return (
        base_price
        + area_sqft * price_per_sqft
        + bedrooms * bedroom_adjustment
    )


print(
    "Traditional rule-based house estimate:",
    traditional_house_price_estimator(1200, 3),
)


# ============================================================================
# 4. LEARNING FROM DATA
# ============================================================================

section("4. Learning From Data")

print(
    """
A machine-learning dataset is commonly represented as examples.

For supervised learning, each example contains:

    features -> target

For example:

    area = 1000
    bedrooms = 2
    age = 10

might correspond to:

    price = 250000

Features are measurable pieces of information used by the model.

The target, label, or response is what we want to predict.

Terminology varies by field:

    observation
    example
    instance
    sample
    record

can refer to an individual data point.

A dataset may be represented mathematically as:

    X = feature matrix
    y = target vector

For n examples and p features:

    X has shape n x p
    y has length n

Example:

    X =
        [1000, 2]
        [1200, 3]
        [1500, 3]

    y =
        [200000]
        [260000]
        [320000]
"""
)


@dataclass
class Example:
    """A simple supervised-learning example."""

    features: list[float]
    target: float


housing_data = [
    Example([1000, 2], 220_000),
    Example([1200, 2], 250_000),
    Example([1400, 3], 290_000),
    Example([1600, 3], 330_000),
    Example([1800, 4], 380_000),
]

print("Number of examples:", len(housing_data))
print("First example:", housing_data[0])


# ============================================================================
# 5. FEATURES AND LABELS
# ============================================================================

section("5. Features and Labels")

print(
    """
A feature is an input variable.

Examples:

    age
    income
    temperature
    transaction_amount
    number_of_rooms

A label is the desired output in supervised learning.

Examples:

    spam / not spam
    house price
    disease class
    customer churn
    demand

A model learns a function:

    f(X) -> y

The model is not necessarily a perfect representation of reality.

It is an approximation that attempts to perform well on relevant data.
"""
)


# ============================================================================
# 6. SUPERVISED, UNSUPERVISED, AND REINFORCEMENT LEARNING
# ============================================================================

section("6. Major Learning Paradigms")

print(
    """
SUPERVISED LEARNING
-------------------

Training data contains target answers.

Example:

    image -> dog
    image -> cat

Common tasks:

    classification
    regression

UNSUPERVISED LEARNING
---------------------

Training data does not provide explicit target answers.

The algorithm attempts to discover structure.

Examples:

    clustering
    dimensionality reduction
    density estimation

REINFORCEMENT LEARNING
----------------------

An agent interacts with an environment.

It receives rewards or penalties and attempts to learn behavior that
maximizes cumulative reward.

Conceptually:

    state -> action -> reward -> new state

These paradigms are related but solve different learning problems.
"""
)


# ============================================================================
# 7. REGRESSION AND CLASSIFICATION
# ============================================================================

section("7. Regression vs Classification")

print(
    """
REGRESSION
----------

Predicts a numerical quantity.

Examples:

    house price = 350000
    temperature = 31.7
    demand = 1842

CLASSIFICATION
--------------

Predicts a category.

Examples:

    spam / not spam
    cat / dog
    fraud / legitimate

Binary classification has two classes.

Multiclass classification has more than two classes.

Multilabel classification allows multiple labels simultaneously.

For example, an image might contain:

    [car, road, person]
"""
)


# ============================================================================
# 8. A SIMPLE LINEAR MODEL
# ============================================================================

section("8. A Model as a Mathematical Function")

print(
    """
A simple linear model with one feature can be written as:

    y_hat = w*x + b

where:

    x     = input feature
    w     = weight / parameter
    b     = bias / intercept
    y_hat = prediction

With multiple features:

    y_hat = w1*x1 + w2*x2 + ... + wp*xp + b

The parameters are not necessarily manually selected. In machine learning,
the learning algorithm estimates them from training data.
"""
)


def linear_predict(weights: Sequence[float], bias: float,
                   features: Sequence[float]) -> float:
    """Compute a linear model prediction."""
    if len(weights) != len(features):
        raise ValueError("Weights and features must have equal length.")

    return sum(w * x for w, x in zip(weights, features)) + bias


weights = [100.0, 50_000.0]
bias = 50_000.0

print(
    "Linear model prediction:",
    linear_predict(weights, bias, [1200, 3]),
)


# ============================================================================
# 9. PARAMETERS VS HYPERPARAMETERS
# ============================================================================

section("9. Parameters and Hyperparameters")

print(
    """
PARAMETERS
----------

Parameters are values learned from training data.

Examples:

    linear regression weights
    neural-network weights
    regression intercept

HYPERPARAMETERS
---------------

Hyperparameters are configuration choices made outside the direct parameter
optimization process.

Examples:

    learning rate
    number of neighbors in k-nearest neighbors
    regularization strength
    maximum tree depth

The distinction matters because parameters are normally estimated from data,
while hyperparameters are selected through experimentation, validation, or
other model-selection procedures.
"""
)


# ============================================================================
# 10. LINEAR REGRESSION FROM SCRATCH
# ============================================================================

section("10. Linear Regression From Scratch")

print(
    """
We now implement a simple one-dimensional linear regression learner.

The model is:

    y_hat = w*x + b

We will minimize mean squared error:

    MSE = (1/n) * sum((y_hat - y)^2)

Gradient descent repeatedly changes w and b in the direction that reduces
the loss.

This demonstrates an important idea:

    DATA -> LOSS -> GRADIENT -> PARAMETER UPDATE -> BETTER MODEL
"""
)


def mean_squared_error(
    predictions: Sequence[float],
    actuals: Sequence[float],
) -> float:
    """Calculate mean squared error."""
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have equal length.")

    if not predictions:
        raise ValueError("At least one observation is required.")

    return sum(
        (prediction - actual) ** 2
        for prediction, actual in zip(predictions, actuals)
    ) / len(predictions)


def train_linear_regression(
    x_values: Sequence[float],
    y_values: Sequence[float],
    learning_rate: float = 0.0001,
    epochs: int = 10_000,
) -> tuple[float, float, list[float]]:
    """
    Train y = w*x + b using gradient descent.

    Returns:
        weight, bias, loss_history
    """
    if len(x_values) != len(y_values):
        raise ValueError("x and y must have equal lengths.")

    if not x_values:
        raise ValueError("Training data cannot be empty.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    weight = 0.0
    bias = 0.0
    loss_history = []

    n = len(x_values)

    for _ in range(epochs):
        predictions = [
            weight * x + bias
            for x in x_values
        ]

        errors = [
            prediction - actual
            for prediction, actual in zip(predictions, y_values)
        ]

        loss = sum(error ** 2 for error in errors) / n
        loss_history.append(loss)

        # Derivatives of MSE with respect to weight and bias.
        gradient_weight = (
            2 / n
        ) * sum(
            error * x
            for error, x in zip(errors, x_values)
        )

        gradient_bias = (
            2 / n
        ) * sum(errors)

        weight -= learning_rate * gradient_weight
        bias -= learning_rate * gradient_bias

    return weight, bias, loss_history


# Small values are used so that the manually implemented gradient descent
# remains numerically easy to inspect.
x_train = [1, 2, 3, 4, 5]
y_train = [3, 5, 7, 9, 11]

learned_weight, learned_bias, losses = train_linear_regression(
    x_train,
    y_train,
    learning_rate=0.01,
    epochs=5_000,
)

print("Learned weight:", round(learned_weight, 4))
print("Learned bias:", round(learned_bias, 4))
print("Final training loss:", round(losses[-1], 6))

new_x = 6
new_prediction = learned_weight * new_x + learned_bias
print("Prediction for x=6:", round(new_prediction, 4))


# ============================================================================
# 11. WHY LOSS FUNCTIONS EXIST
# ============================================================================

section("11. Loss Functions")

print(
    """
A model needs a way to measure how wrong its predictions are.

That measurement is often called a loss function or objective function.

For regression:

    Mean Squared Error
    Mean Absolute Error

are common choices.

For classification:

    binary cross-entropy
    multiclass cross-entropy

are common choices.

Different losses create different optimization behavior.

For example:

MSE strongly penalizes large errors because errors are squared.

MAE grows linearly with absolute error and can therefore be less sensitive
to extreme errors.

The loss used during training is not always identical to the metric used
for final business evaluation.
"""
)


def mean_absolute_error(
    predictions: Sequence[float],
    actuals: Sequence[float],
) -> float:
    """Calculate mean absolute error."""
    if len(predictions) != len(actuals):
        raise ValueError("Length mismatch.")

    if not predictions:
        raise ValueError("At least one observation is required.")

    return sum(
        abs(prediction - actual)
        for prediction, actual in zip(predictions, actuals)
    ) / len(predictions)


predictions = [100, 120, 150]
actuals = [110, 100, 160]

print("MSE:", mean_squared_error(predictions, actuals))
print("MAE:", mean_absolute_error(predictions, actuals))


# ============================================================================
# 12. K-NEAREST NEIGHBORS
# ============================================================================

section("12. Learning by Similarity: K-Nearest Neighbors")

print(
    """
K-nearest neighbors is conceptually simple.

For a new point:

1. Calculate its distance to known training examples.
2. Select the k closest examples.
3. Use their labels to make a prediction.

For classification, the most common class can be selected.

This illustrates a different idea from linear regression.

Linear regression learns a global mathematical relationship.

KNN can make predictions using local similarity.

For two-dimensional Euclidean distance:

    distance = sqrt(
        (x1 - x2)^2 +
        (y1 - y2)^2
    )

A major practical issue is feature scale.

If one feature ranges from 0 to 1 and another from 0 to 1,000, the larger-scale
feature can dominate Euclidean distance.
"""
)


def euclidean_distance(
    point_a: Sequence[float],
    point_b: Sequence[float],
) -> float:
    """Calculate Euclidean distance between equal-length vectors."""
    if len(point_a) != len(point_b):
        raise ValueError("Points must have equal dimensionality.")

    return math.sqrt(
        sum(
            (a - b) ** 2
            for a, b in zip(point_a, point_b)
        )
    )


def knn_classify(
    training_features: Sequence[Sequence[float]],
    training_labels: Sequence[str],
    query: Sequence[float],
    k: int = 3,
) -> str:
    """Classify a query using majority vote among nearest neighbors."""
    if len(training_features) != len(training_labels):
        raise ValueError("Features and labels must have equal length.")

    if not training_features:
        raise ValueError("Training set cannot be empty.")

    if k <= 0:
        raise ValueError("k must be positive.")

    if k > len(training_features):
        raise ValueError("k cannot exceed number of training examples.")

    distances = []

    for features, label in zip(training_features, training_labels):
        distance = euclidean_distance(features, query)
        distances.append((distance, label))

    distances.sort(key=lambda item: item[0])

    nearest_labels = [
        label
        for _, label in distances[:k]
    ]

    counts: dict[str, int] = {}

    for label in nearest_labels:
        counts[label] = counts.get(label, 0) + 1

    # Deterministic tie-breaking: alphabetically smallest class.
    return max(
        sorted(counts),
        key=lambda label: counts[label],
    )


knn_features = [
    [1.0, 1.0],
    [1.2, 0.9],
    [0.8, 1.1],
    [5.0, 5.0],
    [5.2, 4.8],
    [4.9, 5.1],
]

knn_labels = [
    "class_A",
    "class_A",
    "class_A",
    "class_B",
    "class_B",
    "class_B",
]

print(
    "KNN prediction:",
    knn_classify(knn_features, knn_labels, [1.1, 1.0], k=3),
)


# ============================================================================
# 13. CLASSIFICATION FROM SCRATCH WITH LOGISTIC FUNCTION
# ============================================================================

section("13. Binary Classification and Probability")

print(
    """
A binary classifier can estimate a probability:

    P(y = 1 | x)

A common mathematical function for mapping arbitrary real-valued scores into
the interval (0, 1) is the sigmoid function:

    sigmoid(z) = 1 / (1 + e^(-z))

A logistic model uses:

    z = w*x + b

and then:

    probability = sigmoid(z)

A classification threshold can convert probability into a class.

For example:

    probability >= 0.5 -> class 1
    probability < 0.5  -> class 0

The threshold is a decision choice and does not have to be 0.5.
"""
)


def sigmoid(value: float) -> float:
    """
    Numerically stable sigmoid.

    The separate branches avoid unnecessarily calculating exp(-value) for
    very large positive values.
    """
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)

    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def logistic_probability(
    weight: float,
    bias: float,
    feature: float,
) -> float:
    """Return estimated probability for a binary outcome."""
    return sigmoid(weight * feature + bias)


for feature in [-2, -1, 0, 1, 2]:
    probability = logistic_probability(2.0, 0.0, feature)
    print(
        f"x={feature:>2}, probability={probability:.4f}"
    )


# ============================================================================
# 14. TRAINING, VALIDATION, AND TEST DATA
# ============================================================================

section("14. Training, Validation, and Test Sets")

print(
    """
A dataset is often divided into separate portions.

TRAINING SET
------------

Used to estimate model parameters.

VALIDATION SET
--------------

Used for choices such as:

- hyperparameter selection
- model comparison
- threshold selection
- feature decisions

TEST SET
--------

Used for a final estimate of performance on unseen data.

A critical principle:

    Do not repeatedly optimize against the test set.

If the test set repeatedly influences decisions, it gradually becomes part
of the development process and may no longer provide an unbiased estimate
of generalization.

For small datasets, cross-validation can provide a more efficient estimate
than maintaining a large separate validation set.
"""
)


def train_test_split(
    data: Sequence[Example],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Example], list[Example]]:
    """Randomly split examples into training and test portions."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    if len(data) < 2:
        raise ValueError("At least two examples are required.")

    shuffled = list(data)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    test_size = max(1, round(len(shuffled) * test_fraction))

    if test_size >= len(shuffled):
        test_size = len(shuffled) - 1

    return shuffled[test_size:], shuffled[:test_size]


train_set, test_set = train_test_split(housing_data)

print("Training examples:", len(train_set))
print("Test examples:", len(test_set))


# ============================================================================
# 15. GENERALIZATION
# ============================================================================

section("15. Generalization")

print(
    """
The purpose of training is not merely to memorize the training examples.

The important goal is generalization:

    performing well on previously unseen but relevant data.

A model can have very low training error while having poor test error.

This is a central reason machine learning is different from simply storing
examples.

A useful conceptual distinction:

    Training performance = how well the model fits observed data.

    Generalization performance = how well it works on unseen data.
"""
)


# ============================================================================
# 16. OVERFITTING AND UNDERFITTING
# ============================================================================

section("16. Overfitting and Underfitting")

print(
    """
UNDERFITTING
------------

The model is too simple to capture important patterns.

Typical signs:

    high training error
    high validation/test error

Possible causes:

    insufficient model capacity
    poor features
    excessive regularization
    inadequate training

OVERFITTING
-----------

The model fits training data too closely and captures noise or accidental
patterns that do not generalize.

Typical signs:

    very low training error
    substantially higher validation/test error

Possible causes:

    excessive model complexity
    insufficient training data
    noisy features
    weak regularization
    data leakage in some cases

The goal is not to maximize training performance at any cost.
The goal is useful generalization.
"""
)


# ============================================================================
# 17. BIAS AND VARIANCE
# ============================================================================

section("17. Bias and Variance")

print(
    """
Bias refers to systematic error associated with assumptions that are too
restrictive.

Variance refers to sensitivity to fluctuations in the training data.

A very simple model may have:

    high bias
    low variance

A highly flexible model may have:

    low training bias
    high variance

This creates the classical bias-variance trade-off.

The practical objective is not to achieve the lowest possible bias or
variance independently. It is to obtain good expected generalization.

The exact decomposition depends on the loss and statistical assumptions,
so "bias" and "variance" should not be treated as interchangeable synonyms
for underfitting and overfitting.
"""
)


# ============================================================================
# 18. FEATURE SCALING
# ============================================================================

section("18. Feature Scaling")

print(
    """
Feature scaling changes the numerical representation of features.

Two common transformations are:

STANDARDIZATION

    z = (x - mean) / standard_deviation

This produces a feature with mean approximately 0 and standard deviation
approximately 1 when calculated on the same dataset.

MIN-MAX SCALING

    x_scaled = (x - min) / (max - min)

This commonly maps values to [0, 1].

Scaling is particularly important for distance-based methods and many
optimization-based algorithms.

Tree-based algorithms are generally much less sensitive to feature scaling.

A critical implementation rule:

    Fit preprocessing parameters on training data only.

Then apply those learned parameters to validation and test data.

Otherwise information can leak from evaluation data into training.
"""
)


@dataclass
class StandardScaler:
    """A minimal standardization transformer."""

    means: Optional[list[float]] = None
    standard_deviations: Optional[list[float]] = None

    def fit(self, rows: Sequence[Sequence[float]]) -> None:
        if not rows:
            raise ValueError("Cannot fit on empty data.")

        column_count = len(rows[0])

        if column_count == 0:
            raise ValueError("Rows must contain at least one feature.")

        if any(len(row) != column_count for row in rows):
            raise ValueError("All rows must have equal dimensionality.")

        self.means = [
            statistics.mean(row[column] for row in rows)
            for column in range(column_count)
        ]

        self.standard_deviations = []

        for column in range(column_count):
            values = [
                row[column]
                for row in rows
            ]

            standard_deviation = statistics.pstdev(values)

            # A constant feature has no variation. Dividing by zero would be
            # invalid, so represent its standardized value as zero.
            self.standard_deviations.append(
                standard_deviation
                if standard_deviation != 0
                else 1.0
            )

    def transform(
        self,
        rows: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        if self.means is None or self.standard_deviations is None:
            raise RuntimeError("Scaler must be fitted before transform().")

        transformed = []

        for row in rows:
            if len(row) != len(self.means):
                raise ValueError("Feature dimensionality mismatch.")

            transformed.append([
                (value - mean) / standard_deviation
                for value, mean, standard_deviation
                in zip(
                    row,
                    self.means,
                    self.standard_deviations,
                )
            ])

        return transformed

    def fit_transform(
        self,
        rows: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        self.fit(rows)
        return self.transform(rows)


scaler = StandardScaler()

scaled_training_data = scaler.fit_transform([
    [1000, 2],
    [1500, 3],
    [2000, 4],
])

scaled_new_data = scaler.transform([
    [1750, 3],
])

print("Scaled training data:")
for row in scaled_training_data:
    print(row)

print("Scaled new data:")
for row in scaled_new_data:
    print(row)


# ============================================================================
# 19. DATA LEAKAGE
# ============================================================================

section("19. Data Leakage")

print(
    """
Data leakage occurs when information that would not legitimately be available
at prediction time influences model training or evaluation.

Examples:

1. Calculating normalization statistics using the entire dataset before
   splitting into training and test sets.

2. Including future information in a feature used to predict the past.

3. Using a target-derived feature that accidentally reveals the label.

4. Selecting features based on the complete test set.

Leakage can produce deceptively strong evaluation results.

A useful question is:

    "Would this information genuinely be available at the moment the model
     makes the prediction?"

If the answer is no, the feature or processing step is suspect.
"""
)


# ============================================================================
# 20. EVALUATION METRICS
# ============================================================================

section("20. Evaluation Metrics")

print(
    """
Metrics should match the task and the consequences of errors.

REGRESSION METRICS
------------------

MAE:
    average absolute prediction error

MSE:
    average squared prediction error

RMSE:
    square root of MSE

CLASSIFICATION METRICS
----------------------

Accuracy:

    correct predictions / all predictions

Precision:

    true positives / predicted positives

Recall:

    true positives / actual positives

F1 score:

    harmonic mean of precision and recall

Accuracy can be misleading for severe class imbalance.

Example:

If only 1% of transactions are fraudulent, a model that predicts "legitimate"
for every transaction achieves 99% accuracy while detecting zero fraud.
"""
)


def accuracy_score(
    actual: Sequence[int],
    predicted: Sequence[int],
) -> float:
    """Calculate classification accuracy."""
    if len(actual) != len(predicted):
        raise ValueError("Length mismatch.")

    if not actual:
        raise ValueError("No observations.")

    return sum(
        a == p
        for a, p in zip(actual, predicted)
    ) / len(actual)


def binary_confusion_counts(
    actual: Sequence[int],
    predicted: Sequence[int],
) -> tuple[int, int, int, int]:
    """
    Return TP, TN, FP, FN.

    Labels must be binary 0/1.
    """
    if len(actual) != len(predicted):
        raise ValueError("Length mismatch.")

    tp = tn = fp = fn = 0

    for a, p in zip(actual, predicted):
        if a not in (0, 1) or p not in (0, 1):
            raise ValueError("Binary labels must be 0 or 1.")

        if a == 1 and p == 1:
            tp += 1
        elif a == 0 and p == 0:
            tn += 1
        elif a == 0 and p == 1:
            fp += 1
        elif a == 1 and p == 0:
            fn += 1

    return tp, tn, fp, fn


def precision_score(
    actual: Sequence[int],
    predicted: Sequence[int],
) -> float:
    """Calculate binary precision."""
    tp, _, fp, _ = binary_confusion_counts(actual, predicted)
    denominator = tp + fp

    return tp / denominator if denominator else 0.0


def recall_score(
    actual: Sequence[int],
    predicted: Sequence[int],
) -> float:
    """Calculate binary recall."""
    tp, _, _, fn = binary_confusion_counts(actual, predicted)
    denominator = tp + fn

    return tp / denominator if denominator else 0.0


def f1_score(
    actual: Sequence[int],
    predicted: Sequence[int],
) -> float:
    """Calculate binary F1 score."""
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


actual_labels = [1, 1, 1, 0, 0, 0, 1, 0]
predicted_labels = [1, 0, 1, 0, 0, 1, 1, 0]

print("Accuracy:", accuracy_score(actual_labels, predicted_labels))
print("Precision:", precision_score(actual_labels, predicted_labels))
print("Recall:", recall_score(actual_labels, predicted_labels))
print("F1:", f1_score(actual_labels, predicted_labels))


# ============================================================================
# 21. CLASS IMBALANCE
# ============================================================================

section("21. Class Imbalance")

print(
    """
Class imbalance occurs when one class is much more common than another.

Examples:

    fraud detection
    rare disease detection
    equipment failure
    intrusion detection

Accuracy alone may hide poor minority-class performance.

Useful approaches include:

- precision
- recall
- F1
- precision-recall analysis
- confusion matrices
- class weighting
- resampling
- threshold adjustment

The correct choice depends on the cost of false positives and false
negatives.

There is no universally best metric.
"""
)


# ============================================================================
# 22. CONFUSION MATRIX
# ============================================================================

section("22. Confusion Matrix")

print(
    """
For binary classification:

                    Predicted
                  Negative Positive

Actual Negative      TN       FP
Actual Positive      FN       TP

TN = true negative
TP = true positive
FP = false positive
FN = false negative

Different applications care differently about these four outcomes.

Medical screening may prioritize recall.

A system where false alarms are expensive may prioritize precision.

The business or operational consequences should influence metric selection.
"""
)


# ============================================================================
# 23. BASELINE MODELS
# ============================================================================

section("23. Baselines")

print(
    """
A baseline is a simple reference model.

For classification, a baseline might always predict the majority class.

For regression, a baseline might always predict the training-set mean.

A sophisticated model is not automatically useful.

If a complex model barely beats a simple baseline, the additional complexity
may not be justified.

Baselines help answer:

    "Is the machine-learning system actually learning something useful?"
"""
)


def majority_class(labels: Sequence[str]) -> str:
    """Return the most common label."""
    if not labels:
        raise ValueError("labels cannot be empty.")

    counts: dict[str, int] = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    return max(
        sorted(counts),
        key=lambda label: counts[label],
    )


print(
    "Majority-class baseline:",
    majority_class(["A", "A", "A", "B", "B"]),
)


# ============================================================================
# 24. CROSS-VALIDATION
# ============================================================================

section("24. Cross-Validation")

print(
    """
K-fold cross-validation divides data into K folds.

For each iteration:

    one fold -> validation
    remaining folds -> training

This repeats until every fold has served as validation data.

The resulting validation scores can be averaged.

Benefits:

- better use of small datasets
- less dependence on one random train/validation split

Limitations:

- computationally more expensive
- still requires careful preprocessing
- ordinary random folds may be inappropriate for time-series data
- grouped observations may require group-aware splitting

For time-dependent data, training on future data and validating on the past
would create unrealistic evaluation.
"""
)


def k_fold_indices(
    n_samples: int,
    k: int,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """
    Generate deterministic K-fold train/validation indices.

    This basic implementation is appropriate only when observations can
    reasonably be shuffled independently.
    """
    if n_samples <= 1:
        raise ValueError("At least two samples are required.")

    if not 2 <= k <= n_samples:
        raise ValueError("k must be between 2 and n_samples.")

    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)

    folds = [
        indices[i::k]
        for i in range(k)
    ]

    result = []

    for validation_indices in folds:
        validation_set = set(validation_indices)

        training_indices = [
            index
            for index in indices
            if index not in validation_set
        ]

        result.append(
            (training_indices, validation_indices)
        )

    return result


folds = k_fold_indices(10, 5)

for number, (training_indices, validation_indices) in enumerate(
    folds,
    start=1,
):
    print(
        f"Fold {number}: "
        f"train={training_indices}, "
        f"validation={validation_indices}"
    )


# ============================================================================
# 25. REGULARIZATION
# ============================================================================

section("25. Regularization")

print(
    """
Regularization discourages overly complex parameter values or models.

For linear models, two common forms are:

L1 regularization:

    lambda * sum(abs(w))

L2 regularization:

    lambda * sum(w^2)

L1 can encourage exact zero coefficients and therefore can be useful for
sparse representations.

L2 generally shrinks weights toward zero without usually forcing them to
exactly zero.

Regularization modifies the optimization objective.

Conceptually:

    data-fitting loss + complexity penalty

The regularization strength is a hyperparameter.
"""
)


def l1_penalty(weights: Sequence[float], strength: float) -> float:
    """Calculate L1 regularization penalty."""
    if strength < 0:
        raise ValueError("strength cannot be negative.")

    return strength * sum(abs(weight) for weight in weights)


def l2_penalty(weights: Sequence[float], strength: float) -> float:
    """Calculate L2 regularization penalty."""
    if strength < 0:
        raise ValueError("strength cannot be negative.")

    return strength * sum(weight ** 2 for weight in weights)


example_weights = [2.0, -3.0, 0.5]

print("L1 penalty:", l1_penalty(example_weights, 0.1))
print("L2 penalty:", l2_penalty(example_weights, 0.1))


# ============================================================================
# 26. DECISION THRESHOLDS
# ============================================================================

section("26. Probability Thresholds")

print(
    """
A classifier may produce probabilities rather than final labels.

Suppose:

    P(fraud) = 0.30

With threshold 0.50:

    0.30 < 0.50 -> legitimate

With threshold 0.20:

    0.30 >= 0.20 -> fraud

Lowering the threshold generally increases the number of positive predictions.

This can increase recall while potentially decreasing precision.

Threshold selection should reflect the consequences of false positives and
false negatives.
"""
)


def apply_threshold(
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> list[int]:
    """Convert probabilities into binary predictions."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    if any(
        probability < 0 or probability > 1
        for probability in probabilities
    ):
        raise ValueError("Probabilities must be between 0 and 1.")

    return [
        int(probability >= threshold)
        for probability in probabilities
    ]


probabilities = [0.1, 0.25, 0.45, 0.7, 0.95]

print(
    "Threshold 0.5:",
    apply_threshold(probabilities, 0.5),
)

print(
    "Threshold 0.3:",
    apply_threshold(probabilities, 0.3),
)


# ============================================================================
# 27. DATA REPRESENTATION AND MISSING VALUES
# ============================================================================

section("27. Missing Values and Data Quality")

print(
    """
Real-world data is rarely perfect.

Potential problems include:

- missing values
- duplicate records
- inconsistent units
- invalid categories
- impossible values
- measurement errors
- outliers
- stale information
- encoding problems
- incorrect labels

Missing data is not automatically equivalent to zero.

For example:

    missing income != income of 0

Possible strategies include:

- removing observations
- removing features
- mean/median imputation
- model-based imputation
- explicit missing-category representation

The appropriate method depends on why the data is missing and how the model
will use the feature.

Data cleaning is part of machine learning engineering, not merely a cosmetic
preprocessing step.
"""
)


def median_impute(values: Sequence[Optional[float]]) -> list[float]:
    """
    Replace missing values with the median of observed values.

    None represents missing data in this example.
    """
    observed = [
        value
        for value in values
        if value is not None
    ]

    if not observed:
        raise ValueError("Cannot impute when every value is missing.")

    median = statistics.median(observed)

    return [
        median if value is None else value
        for value in values
    ]


missing_values = [10.0, None, 12.0, None, 15.0]

print(
    "Original:",
    missing_values,
)

print(
    "Median-imputed:",
    median_impute(missing_values),
)


# ============================================================================
# 28. OUTLIERS
# ============================================================================

section("28. Outliers")

print(
    """
An outlier is an observation unusually far from other observations under
some definition.

An outlier may represent:

- a genuine rare event
- measurement error
- data-entry error
- fraud
- system failure
- a legitimate extreme case

Blindly removing outliers can destroy useful information.

For example, a fraud-detection system may care precisely about unusual
transactions.

Outlier treatment must therefore be based on domain meaning, not only
statistical distance.
"""
)


# ============================================================================
# 29. CORRELATION IS NOT CAUSATION
# ============================================================================

section("29. Correlation Is Not Causation")

print(
    """
Machine-learning models can exploit statistical associations without
establishing causal relationships.

If two variables are correlated, that does not prove:

    X causes Y

A predictive feature can be useful even when it is not causal.

This distinction matters when models are used for:

- policy decisions
- medical decisions
- financial decisions
- scientific interpretation
- interventions

Prediction and causal inference are related but distinct objectives.
"""
)


# ============================================================================
# 30. TRAINING VS INFERENCE
# ============================================================================

section("30. Training vs Inference")

print(
    """
TRAINING
--------

The model uses data to estimate parameters.

This phase may involve:

    preprocessing
    optimization
    validation
    hyperparameter selection

INFERENCE
---------

The trained model receives new input and produces predictions.

Inference often needs to be:

- fast
- reliable
- reproducible
- resource-efficient
- observable

A production system must ensure that training-time preprocessing and
inference-time preprocessing are consistent.

A mismatch can produce serious prediction errors.
"""
)


# ============================================================================
# 31. DISTRIBUTION SHIFT
# ============================================================================

section("31. Distribution Shift")

print(
    """
Machine-learning performance depends on assumptions about the relationship
between training data and future data.

Distribution shift occurs when the data-generating environment changes.

Examples:

- customer behavior changes
- market conditions change
- sensors are replaced
- fraud tactics evolve
- language changes
- product catalogs change

A model can therefore degrade after deployment even if the software itself
has not changed.

Monitoring should consider both:

    model performance

and:

    input-data behavior
"""
)


# ============================================================================
# 32. DATASET BIAS
# ============================================================================

section("32. Dataset Bias")

print(
    """
A model can only learn from information represented in its data.

Problems can arise from:

- sampling bias
- measurement bias
- label errors
- historical decisions
- underrepresentation
- changing populations

A highly accurate model on a biased dataset can still produce undesirable
real-world outcomes.

The phrase "the model learned from the data" should therefore be interpreted
literally.

The data-generating process influences what the model can learn.
"""
)


# ============================================================================
# 33. FEATURE ENGINEERING
# ============================================================================

section("33. Feature Engineering")

print(
    """
Feature engineering converts raw information into representations that are
useful for learning.

Example:

Raw timestamp:

    2026-09-04 21:30

Potential derived features:

    hour = 21
    day_of_week = Friday
    is_weekend = False

For a transaction:

    amount
    transaction_count_last_24_hours
    average_amount_last_30_days

Good feature engineering can expose useful structure.

It can also create leakage if derived information uses data unavailable at
prediction time.
"""
)


def transaction_features(
    amount: float,
    transaction_count_last_24h: int,
) -> list[float]:
    """Construct two simple transaction features."""
    if amount < 0:
        raise ValueError("Transaction amount cannot be negative.")

    if transaction_count_last_24h < 0:
        raise ValueError("Transaction count cannot be negative.")

    return [
        amount,
        float(transaction_count_last_24h),
    ]


print(
    "Engineered features:",
    transaction_features(1500.0, 4),
)


# ============================================================================
# 34. CATEGORICAL VARIABLES
# ============================================================================

section("34. Categorical Variables")

print(
    """
Categorical variables represent discrete categories.

Examples:

    country = India
    device = mobile
    plan = premium

Many mathematical models require numerical representations.

A common approach is one-hot encoding.

For categories:

    red
    green
    blue

a one-hot representation can be:

    red   -> [1, 0, 0]
    green -> [0, 1, 0]
    blue  -> [0, 0, 1]

Care must be taken to ensure that categories in production are handled
consistently with those seen during training.

Unknown categories are a practical edge case.
"""
)


def one_hot_encode(
    values: Sequence[str],
    categories: Sequence[str],
) -> list[list[int]]:
    """Simple one-hot encoder with explicit unknown-category handling."""
    category_to_index = {
        category: index
        for index, category in enumerate(categories)
    }

    encoded = []

    for value in values:
        row = [0] * len(categories)

        if value in category_to_index:
            row[category_to_index[value]] = 1

        encoded.append(row)

    return encoded


print(
    one_hot_encode(
        ["red", "blue", "unknown"],
        ["red", "green", "blue"],
    )
)


# ============================================================================
# 35. DATA PIPELINE ORDER
# ============================================================================

section("35. A Safe Conceptual ML Pipeline")

print(
    """
A simplified supervised-learning workflow is:

    1. Define the prediction problem.
    2. Define the target.
    3. Collect relevant data.
    4. Validate data quality.
    5. Split data appropriately.
    6. Fit preprocessing on training data.
    7. Transform training data.
    8. Train baseline.
    9. Train candidate models.
   10. Evaluate on validation data.
   11. Select model and hyperparameters.
   12. Perform final evaluation on test data.
   13. Package preprocessing and model together.
   14. Deploy.
   15. Monitor.
   16. Retrain when appropriate.

The exact workflow varies by problem.

Time-series forecasting, grouped observations, recommendation systems,
reinforcement learning, and streaming systems may require substantially
different splitting and evaluation strategies.
"""
)


# ============================================================================
# 36. A COMPLETE MINI ML WORKFLOW
# ============================================================================

section("36. Complete Mini Machine-Learning Workflow")

print(
    """
The following example creates a small synthetic classification problem.

The objective is not statistical sophistication.

The objective is to connect:

    data
    -> split
    -> model
    -> prediction
    -> evaluation

into one coherent workflow.
"""
)


@dataclass
class BinaryExample:
    """One binary classification observation."""

    feature: float
    label: int


def threshold_model(
    feature: float,
    threshold: float,
) -> int:
    """
    A deliberately simple model.

    It predicts class 1 when feature >= threshold.
    """
    return int(feature >= threshold)


classification_data = [
    BinaryExample(1.0, 0),
    BinaryExample(1.5, 0),
    BinaryExample(2.0, 0),
    BinaryExample(2.5, 0),
    BinaryExample(3.0, 1),
    BinaryExample(3.5, 1),
    BinaryExample(4.0, 1),
    BinaryExample(4.5, 1),
]

train_examples, test_examples = train_test_split(
    [
        Example([example.feature], example.label)
        for example in classification_data
    ],
    test_fraction=0.25,
    seed=10,
)

# Estimate a threshold from the training set.
positive_values = [
    example.features[0]
    for example in train_examples
    if example.target == 1
]

negative_values = [
    example.features[0]
    for example in train_examples
    if example.target == 0
]

if positive_values and negative_values:
    threshold = (
        statistics.mean(positive_values)
        + statistics.mean(negative_values)
    ) / 2
else:
    threshold = statistics.mean(
        [example.features[0] for example in train_examples]
    )

test_actual = [
    int(example.target)
    for example in test_examples
]

test_predicted = [
    threshold_model(
        example.features[0],
        threshold,
    )
    for example in test_examples
]

print("Learned threshold:", threshold)
print("Test actual:", test_actual)
print("Test predicted:", test_predicted)
print(
    "Test accuracy:",
    accuracy_score(test_actual, test_predicted),
)


# ============================================================================
# 37. EDGE CASES
# ============================================================================

section("37. Important Edge Cases")

print(
    """
Machine-learning code should explicitly consider edge cases.

Examples include:

- empty datasets
- one-row datasets
- constant features
- missing values
- unknown categories
- division by zero
- invalid labels
- invalid probabilities
- extremely large numerical values
- duplicate observations
- insufficient minority-class examples
- k larger than the number of KNN observations
- time-dependent observations
- unseen categories at inference time

A model that works only on the happy path is not necessarily production
ready.
"""
)


# Demonstrate constant-feature scaling safely.
constant_scaler = StandardScaler()

print(
    "Constant feature scaling:",
    constant_scaler.fit_transform([
        [5.0],
        [5.0],
        [5.0],
    ]),
)


# ============================================================================
# 38. RANDOMNESS AND REPRODUCIBILITY
# ============================================================================

section("38. Reproducibility")

print(
    """
Many machine-learning procedures involve randomness:

- train/test splitting
- initialization
- sampling
- minibatch ordering
- stochastic optimization

A random seed can make an experiment reproducible under controlled
conditions.

Reproducibility does not mean every production prediction is deterministic
in every system.

It means that experimental behavior can be recreated sufficiently to
understand and compare results.
"""
)


def reproducible_random_numbers(seed: int) -> list[float]:
    """Generate deterministic random values for demonstration."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(5)]


print(
    "Run 1:",
    reproducible_random_numbers(42),
)

print(
    "Run 2:",
    reproducible_random_numbers(42),
)


# ============================================================================
# 39. MODEL COMPLEXITY
# ============================================================================

section("39. Model Complexity")

print(
    """
Model complexity describes how flexible a model is in representing
relationships in data.

Examples of increasing flexibility might include:

    simple linear relationship
    polynomial relationship
    decision tree
    ensemble
    neural network

More flexibility can help capture complex patterns.

More flexibility can also increase:

- overfitting risk
- computational requirements
- debugging difficulty
- data requirements
- operational complexity

The best model is not necessarily the most sophisticated model.

A simpler model may be preferable when it provides adequate performance
with lower cost, easier interpretation, or greater reliability.
"""
)


# ============================================================================
# 40. INTERPRETABILITY
# ============================================================================

section("40. Interpretability")

print(
    """
Interpretability asks whether humans can understand why a model produced
a prediction.

A linear model may expose explicit coefficients.

For example:

    y = 3*x1 - 2*x2 + 5

provides a direct mathematical relationship.

More complex models may be harder to interpret.

Interpretability can matter for:

- debugging
- compliance
- scientific analysis
- user trust
- high-impact decisions

Interpretability and predictive performance are not identical objectives.
"""
)


# ============================================================================
# 41. CORRELATION-BASED FEATURE INSPECTION
# ============================================================================

section("41. Simple Correlation Calculation")

print(
    """
Correlation measures statistical association between two numerical
variables.

Pearson correlation is:

    covariance(X,Y) / (std(X) * std(Y))

Its value is generally between -1 and +1.

Values near:

    +1 -> strong positive linear association
     0 -> weak linear association
    -1 -> strong negative linear association

Correlation does not prove causality and can miss nonlinear relationships.
"""
)


def pearson_correlation(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> float:
    """Calculate Pearson correlation."""
    if len(x_values) != len(y_values):
        raise ValueError("Length mismatch.")

    if len(x_values) < 2:
        raise ValueError("At least two observations are required.")

    mean_x = statistics.mean(x_values)
    mean_y = statistics.mean(y_values)

    centered_x = [
        x - mean_x
        for x in x_values
    ]

    centered_y = [
        y - mean_y
        for y in y_values
    ]

    numerator = sum(
        x * y
        for x, y in zip(centered_x, centered_y)
    )

    denominator = math.sqrt(
        sum(x ** 2 for x in centered_x)
        * sum(y ** 2 for y in centered_y)
    )

    if denominator == 0:
        raise ValueError(
            "Correlation is undefined for a constant variable."
        )

    return numerator / denominator


print(
    "Correlation:",
    pearson_correlation(
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
    ),
)


# ============================================================================
# 42. TRAINING ERROR VS TEST ERROR
# ============================================================================

section("42. Training Error and Test Error")

print(
    """
Consider two models:

Model A:
    training error = 10
    test error = 12

Model B:
    training error = 2
    test error = 40

Model B fits the training set much better but generalizes much worse.

This illustrates why evaluation must use data that was not used to fit the
model.

The test set is not simply another training example.

Its value comes from being held out from model-development decisions.
"""
)


# ============================================================================
# 43. HYPERPARAMETER SELECTION
# ============================================================================

section("43. Hyperparameter Selection")

print(
    """
Suppose a model has a hyperparameter:

    k in KNN

Possible choices:

    k = 1
    k = 3
    k = 5
    k = 9

A basic development process is:

    train using training data
    evaluate candidate choices using validation data
    select a reasonable value
    evaluate the selected configuration once on the test set

The test set should not become the tool for repeatedly choosing k.
"""
)


# ============================================================================
# 44. LEARNING CURVES CONCEPT
# ============================================================================

section("44. Learning Curves")

print(
    """
A learning curve examines performance as training-set size changes.

If training with more data consistently improves validation performance,
additional data may be valuable.

If both training and validation performance plateau at poor levels,
changing the model, features, or objective may be more important.

Learning curves can help diagnose:

- high bias
- high variance
- data limitations

They are diagnostic tools, not universal decision rules.
"""
)


# ============================================================================
# 45. ONLINE VS BATCH LEARNING
# ============================================================================

section("45. Batch, Mini-Batch, and Online Learning")

print(
    """
BATCH LEARNING
--------------

Uses a complete training dataset for an update.

MINI-BATCH LEARNING
-------------------

Uses smaller batches of observations.

ONLINE LEARNING
---------------

Updates incrementally as new observations arrive.

Trade-offs involve:

- memory
- computation
- update frequency
- noise
- adaptation speed
- stability

Streaming environments often require careful handling of changing data
distributions.
"""
)


# ============================================================================
# 46. PARAMETRIC VS NONPARAMETRIC METHODS
# ============================================================================

section("46. Parametric vs Nonparametric Learning")

print(
    """
A parametric model assumes a fixed-form model family characterized by a
finite number of parameters.

Example:

    linear regression

    y = w1*x1 + w2*x2 + b

A nonparametric method generally does not assume a fixed finite-dimensional
functional form in the same way.

KNN is a common example.

The distinction is about modeling assumptions and capacity, not simply
whether a model has "parameters" in an informal sense.
"""
)


# ============================================================================
# 47. MACHINE LEARNING AS FUNCTION APPROXIMATION
# ============================================================================

section("47. Machine Learning as Function Approximation")

print(
    """
A useful abstraction is:

    y = f(x)

The real-world relationship is usually unknown.

Machine learning constructs an approximation:

    y_hat = f_hat(x)

Training attempts to make f_hat useful according to a selected objective.

The quality of the result depends on:

- data quality
- representation
- model family
- optimization
- regularization
- evaluation design
- deployment environment
"""
)


# ============================================================================
# 48. PRODUCTION CONSIDERATIONS
# ============================================================================

section("48. Production Machine Learning")

print(
    """
A model is only one component of a production ML system.

A practical system may contain:

    data ingestion
    validation
    preprocessing
    feature generation
    model
    prediction service
    monitoring
    logging
    alerting
    retraining workflow
    model versioning

Important production concerns include:

- latency
- throughput
- memory
- reliability
- reproducibility
- version compatibility
- data drift
- model drift
- rollback
- access control
- auditability
- privacy
- security

A model with excellent offline metrics can still fail operationally.
"""
)


# ============================================================================
# 49. SECURITY CONSIDERATIONS
# ============================================================================

section("49. Security Considerations")

print(
    """
Machine-learning systems can face security problems.

Examples include:

- poisoned training data
- malicious input manipulation
- unauthorized model access
- sensitive information exposure
- compromised dependencies
- insecure model artifacts
- abuse of prediction APIs

Security controls should cover the complete system rather than only the
model algorithm.

Training data should have provenance and validation controls.

Production prediction interfaces should use appropriate authentication,
authorization, rate limiting, logging, and input validation where relevant.
"""
)


# ============================================================================
# 50. PRIVACY CONSIDERATIONS
# ============================================================================

section("50. Privacy Considerations")

print(
    """
Machine-learning data may contain personal or sensitive information.

Important principles include:

- collect only necessary information
- restrict access
- protect stored data
- protect data in transit
- understand retention requirements
- minimize unnecessary copies
- consider whether features can reveal sensitive attributes
- evaluate whether trained models may expose sensitive information

Privacy requirements depend on jurisdiction, application, organization,
and type of data.
"""
)


# ============================================================================
# 51. COMMON BEGINNER MISTAKES
# ============================================================================

section("51. Common Beginner Mistakes")

print(
    """
1. Confusing training accuracy with real-world performance.

2. Evaluating on the same data used for training.

3. Scaling the entire dataset before splitting.

4. Selecting a model only because it is more complex.

5. Using accuracy for heavily imbalanced problems without inspection.

6. Treating correlation as causation.

7. Ignoring missing values.

8. Ignoring data leakage.

9. Using future information in historical prediction tasks.

10. Changing many experimental variables simultaneously.

11. Not keeping track of preprocessing.

12. Forgetting the production data distribution.

13. Assuming more data is always automatically better without checking
    quality and relevance.

14. Treating a model's prediction as certainty.

15. Ignoring the costs of false positives and false negatives.
"""
)


# ============================================================================
# 52. A CONCEPTUAL ML CHECKLIST
# ============================================================================

section("52. Machine-Learning Problem Checklist")

print(
    """
Before training:

    What exactly is being predicted?
    What is the target?
    What information is available at prediction time?
    What constitutes an example?
    Is this regression or classification?
    Is the data independent, grouped, temporal, or sequential?
    How will the dataset be split?
    What baseline should be used?

During development:

    Is preprocessing fitted only on training data?
    Is leakage possible?
    Which loss is being optimized?
    Which metric represents business or scientific success?
    Is the model underfitting or overfitting?
    Are hyperparameters selected using validation data?

Before deployment:

    Is the complete preprocessing pipeline preserved?
    Are input schemas validated?
    Are edge cases handled?
    Are latency and resource requirements acceptable?
    Are security controls present?
    Can predictions be monitored?

After deployment:

    Has the data distribution changed?
    Has performance changed?
    Are labels eventually available for monitoring?
    Does the model require retraining?
    Can the previous model version be restored?
"""
)


# ============================================================================
# 53. END-TO-END CONCEPTUAL DEMONSTRATION
# ============================================================================

section("53. End-to-End Demonstration")

print(
    """
This final demonstration combines the central ideas:

    1. Represent examples as features and labels.
    2. Split data.
    3. Learn a simple parameter.
    4. Predict unseen observations.
    5. Measure performance.

The algorithm below learns the average target value as a deliberately simple
regression baseline.
"""
)


def mean_regression_baseline(
    training_targets: Sequence[float],
) -> Callable[[Sequence[float]], float]:
    """
    Build a constant regression predictor.

    The feature values are ignored. The model predicts the training mean.
    """
    if not training_targets:
        raise ValueError("Training targets cannot be empty.")

    mean_target = statistics.mean(training_targets)

    def predict(_: Sequence[float]) -> float:
        return mean_target

    return predict


regression_training = [
    Example([1.0], 10.0),
    Example([2.0], 20.0),
    Example([3.0], 30.0),
    Example([4.0], 40.0),
]

regression_test = [
    Example([5.0], 50.0),
    Example([6.0], 60.0),
]

baseline_predictor = mean_regression_baseline(
    [
        example.target
        for example in regression_training
    ]
)

baseline_predictions = [
    baseline_predictor(example.features)
    for example in regression_test
]

baseline_actuals = [
    example.target
    for example in regression_test
]

print("Baseline predictions:", baseline_predictions)
print("Actual values:", baseline_actuals)
print(
    "Baseline MAE:",
    mean_absolute_error(
        baseline_predictions,
        baseline_actuals,
    ),
)


# ============================================================================
# 54. FINAL KNOWLEDGE MAP
# ============================================================================

section("54. Machine Learning Fundamentals Knowledge Map")

print(
    """
The foundational chain demonstrated by this script is:

    REAL-WORLD PROBLEM
            |
            v
        DATA
            |
            v
      FEATURES + TARGET
            |
            v
       DATA SPLITTING
            |
            v
      PREPROCESSING
            |
            v
      LEARNING ALGORITHM
            |
            v
          MODEL
            |
            v
        PREDICTION
            |
            v
        EVALUATION
            |
            v
      GENERALIZATION
            |
            v
        DEPLOYMENT
            |
            v
        MONITORING

The central conceptual shift is:

Traditional programming:

    HUMAN-SPECIFIED RULES + DATA -> OUTPUT

Machine learning:

    EXAMPLES + LEARNING PROCEDURE -> LEARNED MODEL

Then:

    NEW DATA + LEARNED MODEL -> PREDICTION

Machine learning therefore combines several disciplines:

    programming
    statistics
    mathematics
    optimization
    data engineering
    software engineering
    domain knowledge
    evaluation
    systems engineering

The algorithm is only one part of the complete machine-learning problem.
"""
)


if __name__ == "__main__":
    print("\nMachine Learning Fundamentals study script completed successfully.")
