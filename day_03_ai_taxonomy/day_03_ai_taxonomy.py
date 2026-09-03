"""
=======================================================================
AI TAXONOMY: FROM SYMBOLIC AI TO MODERN GENERATIVE AI
=======================================================================

TOPIC COVERED
-------------
AI Taxonomy:
    1. Artificial Intelligence
    2. Symbolic AI
    3. Statistical AI
    4. Machine Learning
    5. Deep Learning
    6. Generative AI
    7. Discriminative AI

PURPOSE
-------
This script is an educational, self-contained guide to understanding
the major families of Artificial Intelligence.

It progresses from:
    BASIC CONCEPTS
        ->
    SYMBOLIC AI
        ->
    STATISTICAL AI
        ->
    MACHINE LEARNING
        ->
    DISCRIMINATIVE AI
        ->
    DEEP LEARNING
        ->
    GENERATIVE AI
        ->
    MODERN AI SYSTEMS

IMPORTANT
---------
This file intentionally uses mostly Python standard-library code so
that the concepts can be understood without requiring a large ML
framework.

Where appropriate, simplified mathematical implementations are
provided.

=======================================================================
"""


# =====================================================================
# 0. IMPORTS
# =====================================================================

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from math import exp, log, sqrt
from random import Random
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


# =====================================================================
# 1. WHAT IS ARTIFICIAL INTELLIGENCE?
# =====================================================================

"""
Artificial Intelligence (AI)
----------------------------

Artificial Intelligence is the broad field of computer science concerned
with creating systems that perform tasks that normally require some form
of intelligence.

Examples:

    - reasoning
    - planning
    - perception
    - language understanding
    - prediction
    - classification
    - decision-making
    - learning
    - generation of content

A useful conceptual hierarchy is:

    Artificial Intelligence
    |
    +-- Symbolic AI
    |
    +-- Statistical AI
         |
         +-- Machine Learning
              |
              +-- Discriminative Learning
              |
              +-- Generative Learning
              |
              +-- Deep Learning
                   |
                   +-- Generative Deep Learning
                   |
                   +-- Foundation Models
                        |
                        +-- Large Language Models
                        +-- Multimodal Models
                        +-- Image Generators
                        +-- Speech Models
                        +-- etc.

This hierarchy is conceptual rather than a strict mathematical
classification. Categories can overlap.

For example:

    Deep Learning is a type of Machine Learning.

    Generative AI can use Deep Learning.

    Discriminative models can use Deep Learning.

    Statistical AI is a broader idea that includes many approaches
    based on probability and statistical inference.

    Modern AI systems may combine symbolic reasoning, statistical
    learning, retrieval, planning, and neural networks.
"""


# =====================================================================
# 2. BASIC AI TERMINOLOGY
# =====================================================================

"""
Before studying AI taxonomy, understand these basic terms.

ALGORITHM
---------
A sequence of computational steps used to solve a problem.

MODEL
-----
A mathematical or computational representation that maps inputs to
outputs.

DATA
----
Information used to train, evaluate, or operate an AI system.

FEATURE
-------
An input variable used by a machine-learning model.

LABEL
-----
The target answer associated with a training example in supervised
learning.

TRAINING
--------
The process of learning model parameters from data.

INFERENCE
---------
Using a trained model to produce predictions or outputs.

PARAMETER
---------
A value learned by a model during training.

HYPERPARAMETER
--------------
A configuration selected by the practitioner rather than learned
directly from the training data.

PREDICTION
----------
An output produced by a model.

CLASSIFICATION
--------------
Predicting a category.

Example:

    email -> spam

REGRESSION
----------
Predicting a numerical value.

Example:

    house features -> house price

GENERATION
----------
Producing new content.

Example:

    prompt -> paragraph

REASONING
---------
Using rules, learned representations, probabilities, or other
mechanisms to derive conclusions.

INFERENCE
---------
In AI literature, inference can mean either:
    - applying a trained model, or
    - logically/statistically deriving a conclusion.

Context determines the meaning.
"""


# =====================================================================
# 3. SYMBOLIC AI
# =====================================================================

"""
SYMBOLIC AI
-----------

Symbolic AI is an approach in which knowledge and reasoning are
represented explicitly using symbols, rules, logic, concepts,
relationships, and structured representations.

It is sometimes associated with:

    - Good Old-Fashioned AI (GOFAI)
    - expert systems
    - knowledge representation
    - rule-based systems
    - logic programming
    - automated theorem proving
    - planning systems

Core idea:

    Intelligence can be represented explicitly.

Instead of learning:

    "If temperature is high and engine warning is on,
     then engine may be overheating."

we can explicitly encode the rule.

Example:

    IF temperature > 100
    AND warning_light == True
    THEN diagnosis = "possible overheating"

Symbolic AI therefore emphasizes:

    KNOWLEDGE + RULES + LOGIC + REASONING

rather than primarily:

    DATA + STATISTICAL LEARNING
"""


# ---------------------------------------------------------------------
# 3.1 Simple symbolic rule engine
# ---------------------------------------------------------------------

@dataclass
class Fact:
    subject: str
    predicate: str
    value: Any


@dataclass
class Rule:
    name: str
    condition: Callable[[Dict[Tuple[str, str], Any]], bool]
    conclusion: Callable[[Dict[Tuple[str, str], Any]], Fact]


class SymbolicRuleEngine:
    """
    Very small educational rule-based inference engine.
    """

    def __init__(self) -> None:
        self.facts: Dict[Tuple[str, str], Any] = {}
        self.rules: List[Rule] = []

    def add_fact(self, fact: Fact) -> None:
        self.facts[(fact.subject, fact.predicate)] = fact.value

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def infer(self) -> List[Fact]:
        new_facts: List[Fact] = []

        changed = True

        while changed:
            changed = False

            for rule in self.rules:
                if rule.condition(self.facts):
                    fact = rule.conclusion(self.facts)
                    key = (fact.subject, fact.predicate)

                    if key not in self.facts:
                        self.facts[key] = fact.value
                        new_facts.append(fact)
                        changed = True

        return new_facts


def symbolic_ai_demo() -> None:
    """
    Demonstrates explicit logical reasoning.
    """

    engine = SymbolicRuleEngine()

    engine.add_fact(
        Fact("engine", "temperature", 120)
    )

    engine.add_fact(
        Fact("engine", "warning_light", True)
    )

    engine.add_rule(
        Rule(
            name="Overheating rule",
            condition=lambda facts: (
                facts.get(("engine", "temperature"), 0) > 100
                and facts.get(("engine", "warning_light"), False)
            ),
            conclusion=lambda facts: Fact(
                "engine",
                "diagnosis",
                "possible overheating"
            )
        )
    )

    inferred = engine.infer()

    print("\nSYMBOLIC AI DEMO")
    print("-" * 60)

    for fact in inferred:
        print(
            f"{fact.subject}.{fact.predicate} = {fact.value}"
        )


# =====================================================================
# 4. SYMBOLIC AI: ADVANTAGES AND LIMITATIONS
# =====================================================================

"""
Advantages of Symbolic AI
-------------------------

1. Explicit reasoning
2. Rules can be inspected
3. Knowledge can be manually encoded
4. Often highly interpretable
5. Can work well in constrained domains
6. Logical consistency can be enforced
7. Useful where formal rules are available

Limitations
-----------

1. Knowledge acquisition problem
2. Difficult to manually encode real-world knowledge
3. Brittle when situations differ from predefined rules
4. Difficult to handle ambiguity
5. Difficult to handle noisy data
6. Poor scalability for enormous unstructured datasets
7. Real-world perception is difficult to express with simple rules

Example:

A rule-based system might know:

    IF age >= 18
    THEN adult = True

But real-world concepts can be much harder:

    "Does this photograph contain a dangerous object?"

Writing all possible visual rules manually is impractical.

This limitation contributed to the rise of statistical and
machine-learning approaches.
"""


# =====================================================================
# 5. STATISTICAL AI
# =====================================================================

"""
STATISTICAL AI
--------------

Statistical AI uses probability, statistics, uncertainty modeling,
inference, and data-driven estimation to make predictions or decisions.

Instead of saying:

    "This is definitely class A."

a statistical system may say:

    P(class A | evidence) = 0.82
    P(class B | evidence) = 0.18

This allows AI systems to represent uncertainty.

Core ideas include:

    - probability
    - distributions
    - conditional probability
    - Bayesian inference
    - maximum likelihood
    - statistical estimation
    - hypothesis testing
    - probabilistic graphical models
    - statistical decision theory
"""


# ---------------------------------------------------------------------
# 5.1 Probability
# ---------------------------------------------------------------------

def probability_of_event(
    favorable: int,
    total: int
) -> float:
    """
    Basic probability:

        P(A) = favorable outcomes / total outcomes
    """

    if total <= 0:
        raise ValueError("Total outcomes must be positive.")

    return favorable / total


# ---------------------------------------------------------------------
# 5.2 Conditional probability
# ---------------------------------------------------------------------

def conditional_probability(
    p_a_and_b: float,
    p_b: float
) -> float:
    """
    P(A | B) = P(A and B) / P(B)
    """

    if p_b == 0:
        raise ValueError("P(B) cannot be zero.")

    return p_a_and_b / p_b


# ---------------------------------------------------------------------
# 5.3 Bayes' theorem
# ---------------------------------------------------------------------

def bayes_theorem(
    p_b_given_a: float,
    p_a: float,
    p_b: float
) -> float:
    """
    Bayes' theorem:

        P(A | B) = P(B | A) P(A) / P(B)
    """

    if p_b == 0:
        raise ValueError("P(B) cannot be zero.")

    return (p_b_given_a * p_a) / p_b


def statistical_ai_demo() -> None:
    """
    Simple Bayesian reasoning example.

    Suppose:

        P(spam) = 0.30

        P(word="free" | spam) = 0.80

        P(word="free") = 0.40

    Calculate:

        P(spam | word="free")
    """

    p_spam = 0.30
    p_free_given_spam = 0.80
    p_free = 0.40

    result = bayes_theorem(
        p_free_given_spam,
        p_spam,
        p_free
    )

    print("\nSTATISTICAL AI DEMO")
    print("-" * 60)
    print(
        f"P(spam | 'free') = {result:.3f}"
    )


# =====================================================================
# 6. SYMBOLIC AI VS STATISTICAL AI
# =====================================================================

"""
SYMBOLIC AI
------------

Knowledge:
    Explicit rules

Reasoning:
    Logic

Example:
    IF fever AND cough
    THEN possible_infection

Strength:
    Interpretability

Weakness:
    Difficult to manually encode complex knowledge

STATISTICAL AI
--------------

Knowledge:
    Probabilistic patterns

Reasoning:
    Statistical inference

Example:
    P(infection | symptoms) = 0.87

Strength:
    Handles uncertainty

Weakness:
    Predictions depend heavily on assumptions and data

MODERN AI
---------

Modern AI frequently combines both.

Example:

    Neural model
        +
    Knowledge base
        +
    Rules
        +
    Retrieval
        +
    Probabilistic reasoning

This is one reason modern AI systems are increasingly hybrid.
"""


# =====================================================================
# 7. MACHINE LEARNING
# =====================================================================

"""
MACHINE LEARNING
----------------

Machine Learning (ML) is a subset of AI in which algorithms learn
patterns or relationships from data rather than relying exclusively
on manually programmed rules.

Traditional programming:

    DATA + RULES -> OUTPUT

Machine learning:

    DATA + OUTPUT EXAMPLES -> LEARNED MODEL

Then:

    NEW DATA + MODEL -> PREDICTION

A simplified learning problem can be represented as:

    y_hat = f_theta(x)

where:

    x     = input
    y_hat = predicted output
    f     = model
    theta = learned parameters
"""


# =====================================================================
# 8. TYPES OF MACHINE LEARNING
# =====================================================================

"""
Major learning paradigms:

1. SUPERVISED LEARNING
2. UNSUPERVISED LEARNING
3. SEMI-SUPERVISED LEARNING
4. SELF-SUPERVISED LEARNING
5. REINFORCEMENT LEARNING

---------------------------------------------------------------
8.1 SUPERVISED LEARNING
---------------------------------------------------------------

Training data contains:

    input X
    +
    known target y

Examples:

    image -> cat

    customer features -> churn/no churn

    house features -> price

Tasks:

    classification
    regression

---------------------------------------------------------------
8.2 UNSUPERVISED LEARNING
---------------------------------------------------------------

There is no explicit target label.

The algorithm tries to discover structure.

Examples:

    customer segmentation
    clustering
    dimensionality reduction
    anomaly detection

---------------------------------------------------------------
8.3 SEMI-SUPERVISED LEARNING
---------------------------------------------------------------

Uses:

    small amount of labeled data
    +
    large amount of unlabeled data

Useful when labels are expensive.

---------------------------------------------------------------
8.4 SELF-SUPERVISED LEARNING
---------------------------------------------------------------

The system creates learning targets from the data itself.

Example:

Sentence:

    "The cat sat on the ___."

The model can learn to predict:

    "mat"

No human needs to manually label every example.

Self-supervised learning has been extremely important for modern
foundation models.

---------------------------------------------------------------
8.5 REINFORCEMENT LEARNING
---------------------------------------------------------------

An agent interacts with an environment.

The agent:

    observes state
    chooses action
    receives reward
    updates its strategy

Conceptually:

    STATE -> ACTION -> REWARD -> NEW STATE

Examples:

    robotics
    game playing
    resource allocation
    sequential decision-making
"""


# =====================================================================
# 9. SIMPLE LINEAR REGRESSION
# =====================================================================

"""
Regression predicts a continuous numerical value.

Example:

    hours studied -> exam score

A simple linear model:

    y = wx + b

where:

    w = weight
    b = bias
    x = input
    y = prediction

Training attempts to find values of w and b that minimize an error
function.

A common loss function is Mean Squared Error:

    MSE = (1/n) * SUM((y_i - y_hat_i)^2)
"""


class SimpleLinearRegression:
    """
    Educational implementation of gradient-descent linear regression.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0

    def predict(self, x: float) -> float:
        return self.w * x + self.b

    def fit(
        self,
        X: Sequence[float],
        y: Sequence[float]
    ) -> None:

        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        n = len(X)

        if n == 0:
            raise ValueError("Training data cannot be empty.")

        for _ in range(self.epochs):

            dw = 0.0
            db = 0.0

            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = prediction - yi

                dw += error * xi
                db += error

            dw /= n
            db /= n

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def score_mse(
        self,
        X: Sequence[float],
        y: Sequence[float]
    ) -> float:

        errors = [
            (self.predict(xi) - yi) ** 2
            for xi, yi in zip(X, y)
        ]

        return sum(errors) / len(errors)


def machine_learning_demo() -> None:

    X = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]

    model = SimpleLinearRegression(
        learning_rate=0.01,
        epochs=2000
    )

    model.fit(X, y)

    prediction = model.predict(6)

    print("\nMACHINE LEARNING DEMO")
    print("-" * 60)
    print(f"Learned weight: {model.w:.4f}")
    print(f"Learned bias:   {model.b:.4f}")
    print(f"Prediction for x=6: {prediction:.4f}")


# =====================================================================
# 10. DISCRIMINATIVE AI
# =====================================================================

"""
DISCRIMINATIVE AI
-----------------

Discriminative models learn to distinguish between categories or
predict target values from input features.

For classification, the model estimates:

    P(y | x)

where:

    x = observed input
    y = class/label

The model focuses on the boundary or relationship separating classes.

Examples:

    spam vs non-spam
    fraud vs legitimate
    cat vs dog
    disease vs no disease
    positive vs negative sentiment

Common discriminative models:

    - logistic regression
    - support vector machines
    - decision trees
    - random forests
    - gradient boosting
    - neural-network classifiers
    - many modern classification models
"""


# =====================================================================
# 11. LOGISTIC FUNCTION
# =====================================================================

def sigmoid(z: float) -> float:
    """
    Sigmoid:

        sigmoid(z) = 1 / (1 + e^-z)

    Converts an arbitrary real number into a value between 0 and 1.
    """

    if z >= 0:
        return 1.0 / (1.0 + exp(-z))

    # Numerically safer form for very negative z.
    e = exp(z)
    return e / (1.0 + e)


def discriminative_demo() -> None:

    values = [-5, -2, 0, 2, 5]

    print("\nDISCRIMINATIVE AI DEMO")
    print("-" * 60)

    for value in values:
        print(
            f"sigmoid({value:>2}) = {sigmoid(value):.4f}"
        )


# =====================================================================
# 12. DISCRIMINATIVE CLASSIFIER
# =====================================================================

class SimpleBinaryClassifier:
    """
    Extremely simplified logistic classifier.

    This is intended to demonstrate the concept rather than provide
    production-quality machine learning.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 1000
    ) -> None:

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0

    def predict_probability(self, x: float) -> float:

        return sigmoid(self.w * x + self.b)

    def predict(self, x: float) -> int:

        probability = self.predict_probability(x)

        return int(probability >= 0.5)

    def fit(
        self,
        X: Sequence[float],
        y: Sequence[int]
    ) -> None:

        n = len(X)

        if n == 0:
            raise ValueError("Training data cannot be empty.")

        for _ in range(self.epochs):

            dw = 0.0
            db = 0.0

            for xi, yi in zip(X, y):

                probability = self.predict_probability(xi)

                error = probability - yi

                dw += error * xi
                db += error

            dw /= n
            db /= n

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db


# =====================================================================
# 13. GENERATIVE AI
# =====================================================================

"""
GENERATIVE AI
-------------

Generative AI refers to AI systems capable of generating new content.

Possible outputs include:

    - text
    - images
    - audio
    - video
    - code
    - 3D assets
    - synthetic data
    - structured outputs

Traditional discriminative task:

    Input -> Class

Example:

    photograph -> "cat"

Generative task:

    Prompt -> New photograph/text/audio/etc.

A simplified probabilistic view is:

    model learns a distribution P(x)

or, for conditional generation:

    P(output | input)

For language generation:

    P(token_t | previous tokens)

The model generates one token at a time.

Example:

    "The weather today is"

could be followed by:

    "pleasant"

Then:

    "The weather today is pleasant"

and so on.
"""


# =====================================================================
# 14. GENERATIVE VS DISCRIMINATIVE MODELS
# =====================================================================

"""
DISCRIMINATIVE
--------------

Primary objective:

    Distinguish or predict.

Typical conceptual objective:

    P(y | x)

Example:

    image -> dog

GENRATIVE
---------

Primary objective:

    Model/generate data.

Typical conceptual objective:

    P(x)

or:

    P(x | condition)

Example:

    prompt -> image

Another important distinction:

A generative model can often be used for classification, and a
discriminative model can sometimes produce structured outputs.

Therefore, these categories describe modeling objectives, not always
rigid product categories.
"""


# =====================================================================
# 15. NAIVE BAYES AS A SIMPLE GENERATIVE CLASSIFIER
# =====================================================================

"""
Naive Bayes is useful for understanding the distinction between
generative and discriminative approaches.

A Naive Bayes classifier estimates:

    P(x | y) P(y)

and uses Bayes' theorem to obtain:

    P(y | x)

The "naive" assumption is that features are conditionally independent
given the class.

Despite the simplicity, Naive Bayes can work well for certain text
classification tasks.
"""


class SimpleNaiveBayes:
    """
    Small categorical Naive Bayes demonstration.

    Input:
        documents represented as lists of words

    Output:
        class prediction
    """

    def __init__(self) -> None:

        self.class_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.total_words = Counter()
        self.vocabulary = set()

    def fit(
        self,
        documents: Sequence[Sequence[str]],
        labels: Sequence[str]
    ) -> None:

        for document, label in zip(documents, labels):

            self.class_counts[label] += 1

            for word in document:

                self.word_counts[label][word] += 1
                self.total_words[label] += 1
                self.vocabulary.add(word)

    def predict(self, document: Sequence[str]) -> str:

        total_documents = sum(self.class_counts.values())

        best_label = None
        best_log_probability = float("-inf")

        vocabulary_size = len(self.vocabulary)

        for label, class_count in self.class_counts.items():

            log_probability = log(
                class_count / total_documents
            )

            denominator = (
                self.total_words[label] + vocabulary_size
            )

            for word in document:

                count = self.word_counts[label][word]

                probability = (
                    count + 1
                ) / denominator

                log_probability += log(probability)

            if log_probability > best_log_probability:

                best_log_probability = log_probability
                best_label = label

        if best_label is None:
            raise RuntimeError("Model has not been trained.")

        return best_label


# =====================================================================
# 16. DEEP LEARNING
# =====================================================================

"""
DEEP LEARNING
-------------

Deep Learning is a subfield of Machine Learning that uses neural
networks with multiple layers.

A simplified neural network:

    Input
      |
      v
    Layer 1
      |
      v
    Layer 2
      |
      v
    Layer 3
      |
      v
    Output

Each layer transforms the representation.

Traditional machine learning often depends heavily on manually
designed features.

Deep learning can learn increasingly abstract representations
directly from data.

Example in computer vision:

    pixels
      ->
    edges
      ->
    textures
      ->
    shapes
      ->
    objects
      ->
    classification

Example in language:

    tokens
      ->
    embeddings
      ->
    contextual representations
      ->
    semantic patterns
      ->
    language prediction
"""


# =====================================================================
# 17. NEURAL NETWORK FUNDAMENTALS
# =====================================================================

"""
A basic neuron computes:

    z = w1*x1 + w2*x2 + ... + wn*xn + b

Then applies an activation function:

    a = activation(z)

Common activation functions:

    Sigmoid
    Tanh
    ReLU
    GELU
    Softmax

ReLU:

    ReLU(x) = max(0, x)

Neural networks contain many such computations.

Training typically involves:

    1. Forward pass
    2. Loss calculation
    3. Backpropagation
    4. Parameter update

The optimization process commonly uses gradient descent or variants
such as:

    SGD
    Adam
    AdamW
"""


# ---------------------------------------------------------------------
# 17.1 Activation functions
# ---------------------------------------------------------------------

def relu(x: float) -> float:
    return max(0.0, x)


def tanh_activation(x: float) -> float:
    """
    Hyperbolic tangent.

    Range:

        -1 to +1
    """

    from math import tanh

    return tanh(x)


def softmax(values: Sequence[float]) -> List[float]:

    if not values:
        return []

    max_value = max(values)

    exponentials = [
        exp(value - max_value)
        for value in values
    ]

    total = sum(exponentials)

    return [
        value / total
        for value in exponentials
    ]


# =====================================================================
# 18. A SINGLE ARTIFICIAL NEURON
# =====================================================================

class Neuron:
    """
    Educational single neuron.

    The neuron calculates:

        output = activation(sum(w_i*x_i) + b)
    """

    def __init__(
        self,
        weights: Sequence[float],
        bias: float = 0.0,
        activation: Callable[[float], float] = relu
    ) -> None:

        self.weights = list(weights)
        self.bias = bias
        self.activation = activation

    def forward(
        self,
        inputs: Sequence[float]
    ) -> float:

        if len(inputs) != len(self.weights):
            raise ValueError(
                "Number of inputs must match number of weights."
            )

        weighted_sum = sum(
            weight * value
            for weight, value in zip(
                self.weights,
                inputs
            )
        )

        weighted_sum += self.bias

        return self.activation(weighted_sum)


def neural_network_demo() -> None:

    neuron = Neuron(
        weights=[0.5, 0.8],
        bias=0.2,
        activation=relu
    )

    output = neuron.forward([2.0, 3.0])

    print("\nNEURAL NETWORK DEMO")
    print("-" * 60)
    print(f"Neuron output = {output:.4f}")


# =====================================================================
# 19. DEEP LEARNING ARCHITECTURES
# =====================================================================

"""
Important deep-learning architectures include:

---------------------------------------------------------------
MULTILAYER PERCEPTRON (MLP)
---------------------------------------------------------------

Fully connected neural networks.

Useful for:

    tabular data
    general function approximation

---------------------------------------------------------------
CONVOLUTIONAL NEURAL NETWORK (CNN)
---------------------------------------------------------------

Historically important for:

    image classification
    object detection
    image segmentation

Convolution captures local patterns.

---------------------------------------------------------------
RECURRENT NEURAL NETWORK (RNN)
---------------------------------------------------------------

Designed for sequential information.

Examples:

    time series
    language
    speech

Variants:

    LSTM
    GRU

---------------------------------------------------------------
TRANSFORMER
---------------------------------------------------------------

Uses attention mechanisms to process relationships between elements.

Transformers became foundational to modern:

    language models
    vision models
    multimodal models
    speech models
    generative systems

---------------------------------------------------------------
AUTOENCODER
---------------------------------------------------------------

Learns to encode and reconstruct data.

Structure:

    input
      ->
    encoder
      ->
    latent representation
      ->
    decoder
      ->
    reconstruction

---------------------------------------------------------------
VARIATIONAL AUTOENCODER (VAE)
---------------------------------------------------------------

Probabilistic generative model using a latent-variable framework.

---------------------------------------------------------------
GAN
---------------------------------------------------------------

Generative Adversarial Network.

Contains:

    Generator
    Discriminator

The generator creates synthetic data.

The discriminator attempts to distinguish real from generated data.

---------------------------------------------------------------
DIFFUSION MODELS
---------------------------------------------------------------

Learn to generate data by reversing a gradual corruption/noising
process.

They became particularly influential in image generation.

---------------------------------------------------------------
TRANSFORMER-BASED GENERATIVE MODELS
---------------------------------------------------------------

Commonly used for:

    text generation
    code generation
    multimodal generation
"""


# =====================================================================
# 20. ATTENTION
# =====================================================================

"""
ATTENTION
---------

Attention allows a model to assign different importance to different
parts of an input when computing a representation.

A simplified conceptual formulation is:

    Attention(Q, K, V)
        =
    softmax(QK^T / sqrt(d_k)) V

where:

    Q = queries
    K = keys
    V = values
    d_k = dimensionality of keys

Self-attention allows tokens within the same sequence to interact.

For example:

    "The animal didn't cross the road because it was tired."

The model needs to determine what "it" refers to.

Attention helps the model establish contextual relationships.

Multi-head attention performs several attention operations in
parallel, allowing different representation subspaces to capture
different relationships.
"""


# =====================================================================
# 21. TRANSFORMERS
# =====================================================================

"""
TRANSFORMER
-----------

The Transformer architecture is a neural-network architecture built
around attention mechanisms.

A simplified Transformer block contains:

    input
      |
      v
    self-attention
      |
      v
    residual connection + normalization
      |
      v
    feed-forward network
      |
      v
    residual connection + normalization

Modern Transformer systems often include additional architectural
components and implementation details.

Key concepts:

    tokenization
    embeddings
    positional information
    self-attention
    multi-head attention
    feed-forward layers
    residual connections
    normalization
    autoregressive decoding
    masked attention
"""


# =====================================================================
# 22. TOKENIZATION
# =====================================================================

"""
Language models generally do not process raw text directly.

Text is transformed into tokens.

Example:

    "Artificial intelligence is powerful"

might become tokens such as:

    ["Artificial", " intelligence", " is", " powerful"]

Actual tokenization depends on the tokenizer.

Tokens are mapped to integer IDs.

Example:

    "AI" -> 12345

The model processes token IDs through learned embeddings.

Important:

    Token != necessarily word.

A token may represent:

    - a whole word
    - part of a word
    - punctuation
    - whitespace-associated text
    - special symbols
"""


# =====================================================================
# 23. EMBEDDINGS
# =====================================================================

"""
An embedding maps discrete objects into vectors.

Example:

    "king" -> [0.21, -0.18, 0.73, ...]

    "queen" -> [0.19, -0.12, 0.75, ...]

Embeddings provide continuous numerical representations.

They can represent:

    words
    tokens
    documents
    images
    users
    products
    entities
    concepts

Modern AI systems use embeddings extensively for:

    semantic search
    retrieval
    recommendation
    clustering
    classification
    multimodal representation
"""


# =====================================================================
# 24. GENERATIVE LANGUAGE MODELS
# =====================================================================

"""
A language model estimates probabilities over token sequences.

Conceptually:

    P(x1, x2, ..., xn)

Using the chain rule:

    P(x1, ..., xn)
    =
    P(x1)
    *
    P(x2 | x1)
    *
    P(x3 | x1, x2)
    *
    ...
    *
    P(xn | x1, ..., x(n-1))

An autoregressive language model predicts:

    next token | previous tokens

Example:

    Input:
        "The capital of France is"

Possible probabilities:

    Paris      0.95
    London     0.01
    Berlin     0.005
    ...

Generation repeatedly selects a token and feeds it back into the
model.
"""


# =====================================================================
# 25. TEMPERATURE
# =====================================================================

def temperature_scaled_probabilities(
    logits: Sequence[float],
    temperature: float
) -> List[float]:
    """
    Demonstrates temperature scaling.

    Lower temperature:
        more concentrated probabilities

    Higher temperature:
        flatter probabilities

    In practical language-model generation, temperature is applied to
    logits before softmax.
    """

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    scaled = [
        logit / temperature
        for logit in logits
    ]

    return softmax(scaled)


def temperature_demo() -> None:

    logits = [4.0, 2.0, 1.0]

    print("\nTEMPERATURE DEMO")
    print("-" * 60)

    for temperature in [0.5, 1.0, 2.0]:

        probabilities = (
            temperature_scaled_probabilities(
                logits,
                temperature
            )
        )

        print(
            f"temperature={temperature}: "
            f"{[round(p, 4) for p in probabilities]}"
        )


# =====================================================================
# 26. TOP-K AND TOP-P CONCEPTS
# =====================================================================

"""
TOP-K
-----

Only the K most probable tokens are considered for generation.

Example:

    vocabulary = 50,000 tokens

    top_k = 50

Only the 50 highest-probability tokens are considered.

TOP-P / NUCLEUS SAMPLING
------------------------

Instead of a fixed number of tokens, select the smallest set of
tokens whose cumulative probability reaches p.

Example:

    p = 0.90

The candidate set contains tokens accounting for approximately the
top 90% probability mass.

These techniques can control generation diversity.
"""


# =====================================================================
# 27. DISCRIMINATIVE AI IN DEEP LEARNING
# =====================================================================

"""
Deep learning can be discriminative.

Examples:

    CNN:
        image -> class

    Transformer classifier:
        text -> sentiment

    Neural fraud detector:
        transaction -> fraud probability

The neural network learns a mapping such as:

    f(x) -> y

or:

    P(y | x)

Therefore:

    Deep Learning != Generative AI

Deep Learning is a modeling approach.

Generative AI is a capability/objective.

A deep-learning system can be:

    discriminative
    generative
    both, depending on its components/tasks.
"""


# =====================================================================
# 28. GENERATIVE AI MODEL FAMILIES
# =====================================================================

"""
Major families include:

1. Autoregressive models
2. Variational Autoencoders
3. GANs
4. Diffusion models
5. Flow-based models
6. Energy-based approaches
7. Transformer-based multimodal models

---------------------------------------------------------------
AUTOREGRESSIVE
---------------------------------------------------------------

Generate one element conditioned on previous elements.

Text:

    token 1 -> token 2 -> token 3 -> ...

---------------------------------------------------------------
VAE
---------------------------------------------------------------

Learn latent probability distributions and reconstruct/generate data.

---------------------------------------------------------------
GAN
---------------------------------------------------------------

Generator competes with discriminator.

---------------------------------------------------------------
DIFFUSION
---------------------------------------------------------------

Learn to reverse a noise-corruption process.

---------------------------------------------------------------
FLOW-BASED MODELS
---------------------------------------------------------------

Use invertible transformations to model probability distributions.
"""


# =====================================================================
# 29. FOUNDATION MODELS
# =====================================================================

"""
FOUNDATION MODEL
----------------

A foundation model is a large, broadly trained model that can be
adapted to many downstream tasks.

Examples of capabilities:

    language understanding
    text generation
    summarization
    translation
    coding
    reasoning
    classification
    extraction

Foundation models can be adapted using:

    prompting
    fine-tuning
    instruction tuning
    parameter-efficient fine-tuning
    retrieval augmentation
    tool use
    agents

The term describes the broad reusable role of the model rather than
one specific architecture.
"""


# =====================================================================
# 30. LARGE LANGUAGE MODELS
# =====================================================================

"""
LLM
---

Large Language Model.

An LLM is typically a large neural language model trained on very
large amounts of text and/or related data.

Typical pipeline:

    raw data
        |
        v
    preprocessing
        |
        v
    tokenization
        |
        v
    pretraining
        |
        v
    model
        |
        v
    instruction tuning / alignment
        |
        v
    inference
        |
        v
    generated response

Important concepts:

    context window
    parameters
    tokens
    embeddings
    attention
    inference
    decoding
    fine-tuning
    instruction following
    evaluation
"""


# =====================================================================
# 31. TRAINING VS INFERENCE
# =====================================================================

"""
TRAINING
--------

Training changes model parameters.

Example:

    data -> forward pass -> loss -> backpropagation -> update

INFERENCE
---------

Inference uses already-trained parameters.

Example:

    prompt -> model -> output

Training is generally computationally expensive.

Inference can also be computationally expensive, particularly for
large models, but the computational pattern is different.
"""


# =====================================================================
# 32. LOSS FUNCTIONS
# =====================================================================

"""
A loss function measures how wrong a model's prediction is.

Examples:

    Mean Squared Error
    Cross Entropy
    Binary Cross Entropy
    Contrastive Loss
    Ranking Loss

Regression:

    MSE = average((y - y_hat)^2)

Classification commonly uses cross entropy.

For a classification target y and predicted probability p:

    Binary Cross Entropy
        =
    -[y log(p) + (1-y) log(1-p)]
"""


def binary_cross_entropy(
    y_true: int,
    probability: float
) -> float:

    epsilon = 1e-15

    probability = max(
        epsilon,
        min(1 - epsilon, probability)
    )

    return -(
        y_true * log(probability)
        +
        (1 - y_true) * log(1 - probability)
    )


# =====================================================================
# 33. OVERFITTING AND UNDERFITTING
# =====================================================================

"""
OVERFITTING
-----------

The model learns training data too specifically and performs poorly
on unseen data.

Symptoms:

    training performance -> very good
    validation performance -> poor

Causes:

    model too complex
    too little data
    noisy data
    excessive training

Solutions:

    more data
    regularization
    dropout
    early stopping
    data augmentation
    simpler model


UNDERFITTING
------------

The model is too simple to capture important patterns.

Symptoms:

    training performance -> poor
    validation performance -> poor

Solutions:

    more expressive model
    better features
    better training
    reduced regularization
"""


# =====================================================================
# 34. TRAIN / VALIDATION / TEST SPLIT
# =====================================================================

"""
A standard machine-learning workflow divides data into:

TRAINING SET
------------
Used to learn parameters.

VALIDATION SET
--------------
Used to tune model choices and hyperparameters.

TEST SET
--------
Used for final unbiased evaluation.

Important:

    The test set should not be repeatedly used to make modeling
    decisions.

Otherwise, the test set effectively becomes part of the optimization
process.
"""


# =====================================================================
# 35. EVALUATION METRICS
# =====================================================================

"""
CLASSIFICATION METRICS
----------------------

Accuracy:

    correct / total

Precision:

    TP / (TP + FP)

Recall:

    TP / (TP + FN)

F1:

    2 * precision * recall / (precision + recall)

Specificity:

    TN / (TN + FP)

ROC-AUC:
    Measures ranking/separation performance across thresholds.

PR-AUC:
    Particularly useful in some imbalanced classification problems.


REGRESSION METRICS
------------------

MAE:
    Mean Absolute Error

MSE:
    Mean Squared Error

RMSE:
    Root Mean Squared Error

R²:
    Coefficient of determination


GENERATIVE AI METRICS
---------------------

Evaluation is more difficult.

Possible dimensions:

    factuality
    relevance
    coherence
    instruction following
    toxicity
    safety
    diversity
    groundedness
    citation correctness
    task success
    human preference
"""


# =====================================================================
# 36. CONFUSION MATRIX
# =====================================================================

@dataclass
class ConfusionMatrix:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def accuracy(self) -> float:

        total = (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        )

        return (
            (self.true_positive + self.true_negative)
            / total
        )

    def precision(self) -> float:

        denominator = (
            self.true_positive
            + self.false_positive
        )

        return (
            self.true_positive / denominator
            if denominator
            else 0.0
        )

    def recall(self) -> float:

        denominator = (
            self.true_positive
            + self.false_negative
        )

        return (
            self.true_positive / denominator
            if denominator
            else 0.0
        )

    def f1(self) -> float:

        p = self.precision()
        r = self.recall()

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)


# =====================================================================
# 37. MODEL PARAMETERS VS HYPERPARAMETERS
# =====================================================================

"""
PARAMETERS
----------

Learned during training.

Examples:

    neural-network weights
    neural-network biases

HYPERPARAMETERS
---------------

Selected externally.

Examples:

    learning rate
    batch size
    number of layers
    number of epochs
    regularization strength
    tree depth
    temperature during generation

This distinction is fundamental in machine learning.
"""


# =====================================================================
# 38. REGULARIZATION
# =====================================================================

"""
Regularization discourages models from becoming excessively complex
or relying too heavily on particular patterns.

Common techniques:

    L1 regularization
    L2 regularization
    dropout
    early stopping
    data augmentation

L2 regularization adds a penalty related to squared parameter values.

Conceptually:

    total loss =
        prediction loss
        +
        lambda * parameter penalty
"""


# =====================================================================
# 39. GRADIENT DESCENT
# =====================================================================

"""
Gradient descent is a fundamental optimization method.

Conceptually:

    parameter_new
        =
    parameter_old
        -
    learning_rate * gradient

The gradient indicates how the loss changes with respect to a
parameter.

Learning rate controls the size of updates.

Too large:

    training may become unstable.

Too small:

    training can become very slow.
"""


# =====================================================================
# 40. BACKPROPAGATION
# =====================================================================

"""
Backpropagation computes gradients of a neural network's loss with
respect to its parameters.

Conceptually:

    forward pass
        ->
    compute loss
        ->
    propagate gradients backward
        ->
    update parameters

Backpropagation relies heavily on the chain rule of calculus.

For a composite function:

    y = f(g(x))

the derivative is:

    dy/dx = df/dg * dg/dx

A deep network is essentially a large composition of functions, so
the chain rule allows gradients to be propagated through layers.
"""


# =====================================================================
# 41. SELF-SUPERVISED LEARNING
# =====================================================================

"""
Self-supervised learning is especially important for modern AI.

Instead of humans providing labels:

    data -> automatically constructed learning objective

Examples:

MASKED LANGUAGE MODELING
------------------------

Some tokens are hidden.

Input:

    "The cat sat on the [MASK]."

Target:

    "mat"


NEXT-TOKEN PREDICTION
---------------------

Input:

    "The cat sat on the"

Target:

    next token

CONTRASTIVE LEARNING
--------------------

The system learns representations by bringing related examples
closer and unrelated examples farther apart.
"""


# =====================================================================
# 42. FINE-TUNING
# =====================================================================

"""
Fine-tuning means adapting a pretrained model to a particular task or
domain.

Pipeline:

    pretrained model
          |
          v
    task/domain data
          |
          v
    optimization
          |
          v
    adapted model

Examples:

    general language model
        ->
    legal-domain assistant

    general vision model
        ->
    industrial defect detector

Fine-tuning can be:

    full-parameter
    parameter-efficient
"""


# =====================================================================
# 43. PARAMETER-EFFICIENT FINE-TUNING
# =====================================================================

"""
PEFT
----

Parameter-Efficient Fine-Tuning methods adapt models while updating
only a small subset or low-dimensional transformation of parameters.

Examples include:

    LoRA
    adapters
    prefix tuning

LoRA conceptually represents an update as:

    Delta W = B A

where A and B are much smaller matrices than the original weight
matrix.

Benefits:

    lower memory requirements
    smaller trainable parameter count
    easier storage of task-specific adapters
    potentially lower computational cost
"""


# =====================================================================
# 44. TRANSFER LEARNING
# =====================================================================

"""
Transfer learning means using knowledge learned from one task/domain
to help solve another task/domain.

Example:

    model learns general visual features

then:

    model adapted for medical image classification

The basic idea:

    learn general representation
        ->
    reuse representation
        ->
    specialize
"""


# =====================================================================
# 45. RETRIEVAL-AUGMENTED GENERATION
# =====================================================================

"""
RAG
---

Retrieval-Augmented Generation combines retrieval with generative
models.

Typical architecture:

    User Query
        |
        v
    Query Embedding
        |
        v
    Vector Search
        |
        v
    Relevant Documents
        |
        v
    Context
        |
        v
    Generative Model
        |
        v
    Answer

RAG can help a model use:

    private documents
    current information
    company knowledge
    technical documentation
    databases

RAG does not automatically guarantee factuality. Retrieval quality,
chunking, ranking, context construction, and generation all matter.
"""


# =====================================================================
# 46. AGENTIC AI
# =====================================================================

"""
AGENTIC AI
----------

An AI agent is generally a system that can perceive a goal/environment,
reason or plan, use tools, and take actions.

A simplified architecture:

    Goal
      |
      v
    Planning
      |
      v
    Tool selection
      |
      v
    Tool execution
      |
      v
    Observation
      |
      v
    Evaluation
      |
      +------> Continue
      |
      v
    Final result

Possible tools:

    calculator
    database
    search
    APIs
    code execution
    enterprise systems

Agentic systems can combine:

    LLMs
    tools
    memory
    retrieval
    planning
    workflows
    rules
    human approval
"""


# =====================================================================
# 47. HYBRID AI
# =====================================================================

"""
HYBRID AI
---------

Hybrid AI combines different AI paradigms.

Example:

    Symbolic rules
          +
    Machine learning
          +
    Deep learning
          +
    Retrieval
          +
    Generative model

A financial fraud system might contain:

    rule engine:
        transaction above threshold -> review

    ML classifier:
        fraud probability

    graph model:
        suspicious relationship patterns

    LLM:
        summarize investigation evidence

    human:
        approve/reject action

This demonstrates that AI taxonomy is not simply a set of mutually
exclusive boxes.
"""


# =====================================================================
# 48. KNOWLEDGE GRAPH
# =====================================================================

"""
A knowledge graph represents entities and relationships.

Example:

    [Alice]
       |
       | works_at
       v
    [Company A]
       |
       | located_in
       v
    [City X]

Knowledge graphs are commonly represented as triples:

    subject
    predicate
    object

Example:

    Alice | works_at | Company A

They can support:

    semantic search
    reasoning
    recommendations
    entity resolution
    question answering
    enterprise knowledge systems
"""


# =====================================================================
# 49. EXPLAINABILITY
# =====================================================================

"""
Symbolic systems are often naturally interpretable because their
rules can be inspected directly.

Many statistical and neural models are less transparent.

Important explainability concepts:

    feature importance
    partial dependence
    SHAP
    LIME
    saliency
    attention analysis
    counterfactual explanations

Important warning:

    An explanation technique does not automatically reveal the true
    internal causal reasoning of a model.

Interpretability and explainability should therefore be treated
carefully.
"""


# =====================================================================
# 50. BIAS
# =====================================================================

"""
AI systems can inherit or amplify biases from:

    training data
    labeling processes
    sampling
    measurement
    model design
    deployment environment

Types of bias include:

    sampling bias
    measurement bias
    label bias
    historical bias
    selection bias
    automation bias

Responsible AI requires:

    appropriate data
    evaluation
    documentation
    monitoring
    governance
    human oversight
"""


# =====================================================================
# 51. HALLUCINATION
# =====================================================================

"""
In generative AI, hallucination generally refers to generated content
that is unsupported, fabricated, or incorrect.

Possible causes include:

    probabilistic generation
    incomplete knowledge
    ambiguous prompts
    weak retrieval
    model limitations
    decoding behavior
    conflicting context

Mitigation strategies:

    retrieval
    grounding
    tool use
    structured outputs
    verification
    citation checking
    human review
    domain-specific evaluation
"""


# =====================================================================
# 52. AI SAFETY
# =====================================================================

"""
AI safety includes reducing risks associated with AI systems.

Examples:

    harmful outputs
    privacy violations
    security vulnerabilities
    unreliable automation
    misuse
    model manipulation
    data leakage
    unsafe tool actions

Modern production AI systems often require:

    access controls
    monitoring
    evaluation
    guardrails
    sandboxing
    human approval
    audit logs
"""


# =====================================================================
# 53. AI TAXONOMY AT A GLANCE
# =====================================================================

AI_TAXONOMY = {
    "Artificial Intelligence": {
        "definition": "Broad field of creating systems capable of intelligent behavior.",
        "examples": [
            "reasoning",
            "planning",
            "learning",
            "perception",
            "generation",
        ],
    },

    "Symbolic AI": {
        "definition": "Explicit knowledge representation and logical reasoning.",
        "examples": [
            "expert systems",
            "rule engines",
            "logic programming",
            "knowledge representation",
        ],
    },

    "Statistical AI": {
        "definition": "AI based on probability, statistics, and uncertainty.",
        "examples": [
            "Bayesian inference",
            "probabilistic models",
            "statistical decision systems",
        ],
    },

    "Machine Learning": {
        "definition": "Algorithms that learn patterns from data.",
        "examples": [
            "regression",
            "classification",
            "clustering",
            "reinforcement learning",
        ],
    },

    "Discriminative AI": {
        "definition": "Models focused on predicting or distinguishing outputs.",
        "examples": [
            "spam classifiers",
            "fraud detectors",
            "sentiment classifiers",
        ],
    },

    "Deep Learning": {
        "definition": "Machine learning based on multi-layer neural networks.",
        "examples": [
            "CNN",
            "RNN",
            "Transformer",
            "deep classifiers",
        ],
    },

    "Generative AI": {
        "definition": "AI systems capable of producing new content.",
        "examples": [
            "LLMs",
            "image generators",
            "audio generators",
            "video generators",
            "code generators",
        ],
    },
}


def print_taxonomy() -> None:

    print("\nAI TAXONOMY")
    print("=" * 60)

    for category, information in AI_TAXONOMY.items():

        print(f"\n{category}")
        print("-" * len(category))

        print(
            f"Definition: {information['definition']}"
        )

        print(
            "Examples: "
            + ", ".join(information["examples"])
        )


# =====================================================================
# 54. TAXONOMY RELATIONSHIP
# =====================================================================

"""
A simplified conceptual relationship:

                     ARTIFICIAL INTELLIGENCE
                              |
             +----------------+----------------+
             |                                 |
        SYMBOLIC AI                    STATISTICAL AI
                                               |
                                      +--------+--------+
                                      |                 |
                               MACHINE LEARNING      Other
                                      |
                        +-------------+-------------+
                        |                           |
                DISCRIMINATIVE                 GENERATIVE
                        |                           |
                        +-------------+-------------+
                                      |
                               DEEP LEARNING
                                      |
                    +-----------------+----------------+
                    |                 |                |
                  CNN             RNN/LSTM        TRANSFORMER
                                                        |
                                              +---------+---------+
                                              |                   |
                                             LLM            Multimodal AI

Important:

This is a conceptual map.

The categories overlap.

For example:

    A Transformer can be discriminative.

    A Transformer can be generative.

    A deep model can be trained using self-supervised learning.

    A generative model can be used for classification.

    A production system can combine symbolic and neural approaches.
"""


# =====================================================================
# 55. SIMPLE DECISION GUIDE
# =====================================================================

def choose_ai_approach(problem: str) -> str:
    """
    Educational decision guide.

    This is not a production model-selection algorithm.
    """

    p = problem.lower()

    if any(
        word in p
        for word in [
            "explicit rules",
            "logic",
            "formal reasoning",
            "business rules"
        ]
    ):
        return "Consider Symbolic AI or a hybrid symbolic system."

    if any(
        word in p
        for word in [
            "predict",
            "classify",
            "forecast",
            "score"
        ]
    ):
        return (
            "Consider supervised/discriminative machine learning "
            "or deep learning."
        )

    if any(
        word in p
        for word in [
            "cluster",
            "group",
            "discover structure"
        ]
    ):
        return (
            "Consider unsupervised learning."
        )

    if any(
        word in p
        for word in [
            "generate text",
            "generate image",
            "generate code",
            "create content"
        ]
    ):
        return (
            "Consider Generative AI, often based on deep learning."
        )

    if any(
        word in p
        for word in [
            "uncertainty",
            "probability",
            "risk"
        ]
    ):
        return (
            "Consider statistical/probabilistic AI."
        )

    return (
        "Define the problem, data, objective, constraints, "
        "evaluation metric, and risk before selecting an approach."
    )


# =====================================================================
# 56. COMPLETE COMPARISON
# =====================================================================

def print_comparison() -> None:

    comparison = [
        (
            "Symbolic AI",
            "Rules and symbols",
            "Logic",
            "Explicit knowledge",
            "Expert systems"
        ),

        (
            "Statistical AI",
            "Probability/data",
            "Statistical inference",
            "Uncertainty",
            "Bayesian systems"
        ),

        (
            "Machine Learning",
            "Data",
            "Optimization",
            "Learned patterns",
            "Regression"
        ),

        (
            "Discriminative AI",
            "Input/labels",
            "Prediction",
            "Class boundaries",
            "Spam classifier"
        ),

        (
            "Deep Learning",
            "Large datasets",
            "Gradient optimization",
            "Learned representations",
            "Transformers"
        ),

        (
            "Generative AI",
            "Data distributions",
            "Probabilistic generation",
            "New content",
            "LLMs"
        ),
    ]

    print("\nCOMPARISON")
    print("=" * 100)

    print(
        f"{'Category':<20}"
        f"{'Primary basis':<25}"
        f"{'Core mechanism':<25}"
        f"{'Primary output':<25}"
        f"{'Example'}"
    )

    print("-" * 100)

    for row in comparison:

        print(
            f"{row[0]:<20}"
            f"{row[1]:<25}"
            f"{row[2]:<25}"
            f"{row[3]:<25}"
            f"{row[4]}"
        )


# =====================================================================
# 57. COMMON MISCONCEPTIONS
# =====================================================================

COMMON_MISCONCEPTIONS = {
    "Deep Learning = AI":
        "False. Deep learning is one subset of machine learning, which is itself part of the broader AI field.",

    "Generative AI = all AI":
        "False. Generative AI is one category of AI capability/modeling objective.",

    "Machine Learning always means Deep Learning":
        "False. Decision trees, linear regression, random forests, and many other ML methods are not deep neural networks.",

    "Discriminative AI cannot generate anything":
        "Too simplistic. The discriminative/generative distinction concerns modeling objectives and probability structure.",

    "Symbolic AI is obsolete":
        "False. Symbolic reasoning remains useful in rules, planning, formal verification, knowledge systems, and hybrid AI.",

    "More parameters always means a better model":
        "False. Data quality, architecture, training, evaluation, domain fit, inference strategy, and system design all matter.",

    "Generative AI always knows facts":
        "False. Generative models can produce plausible but incorrect information.",

    "AI is only neural networks":
        "False. AI includes symbolic, probabilistic, evolutionary, optimization-based, neural, and hybrid approaches."
}


# =====================================================================
# 58. PRODUCTION AI ARCHITECTURE
# =====================================================================

"""
A modern enterprise AI system may look like:

    USER
      |
      v
    APPLICATION
      |
      v
    ROUTER / ORCHESTRATOR
      |
      +--------------------+
      |                    |
      v                    v
   RETRIEVAL             TOOLS
      |                    |
      v                    v
 KNOWLEDGE BASE       APIs / DATABASES
      |                    |
      +---------+----------+
                |
                v
          FOUNDATION MODEL
                |
                v
       VALIDATION / GUARDRAILS
                |
                v
        HUMAN APPROVAL
                |
                v
             OUTPUT

This architecture demonstrates that modern AI is usually a system,
not simply one model.
"""


# =====================================================================
# 59. AI SYSTEM VS AI MODEL
# =====================================================================

"""
MODEL
-----

A trained computational artifact.

SYSTEM
------

The complete operational environment around the model.

A production AI system may contain:

    model
    prompt
    retrieval
    database
    APIs
    authentication
    authorization
    monitoring
    evaluation
    guardrails
    user interface
    business rules
    human approval

This distinction is extremely important in enterprise AI.
"""


# =====================================================================
# 60. AI DEVELOPMENT LIFECYCLE
# =====================================================================

"""
A typical AI lifecycle:

    1. Problem definition
    2. Data collection
    3. Data cleaning
    4. Data exploration
    5. Feature/representation design
    6. Model selection
    7. Training
    8. Validation
    9. Evaluation
   10. Deployment
   11. Monitoring
   12. Feedback
   13. Retraining/improvement

For generative AI:

    problem definition
        ->
    model selection
        ->
    prompting / RAG / fine-tuning
        ->
    evaluation
        ->
    deployment
        ->
    monitoring
"""


# =====================================================================
# 61. DATA-CENTRIC AI
# =====================================================================

"""
DATA-CENTRIC AI emphasizes improving:

    data quality
    labeling
    coverage
    consistency
    representativeness
    data pipelines

A sophisticated model trained on poor data can perform poorly.

Important data concepts:

    missing values
    outliers
    duplicates
    label noise
    leakage
    class imbalance
    distribution shift
"""


# =====================================================================
# 62. DATA LEAKAGE
# =====================================================================

"""
Data leakage occurs when information unavailable at prediction time
accidentally enters the training process.

Example:

Predict:

    whether a customer will churn next month.

If the training features include:

    "account_cancelled"

then the model may appear extremely accurate while learning from
information that would not actually be available when making the
prediction.

Leakage can make evaluation misleading.
"""


# =====================================================================
# 63. DISTRIBUTION SHIFT
# =====================================================================

"""
Training data distribution:

    P_train(X, Y)

Production distribution:

    P_production(X, Y)

If these differ significantly, performance can degrade.

Examples:

    changing customer behavior
    new fraud patterns
    new products
    new language usage
    changes in sensors
    changes in regulations

Production AI therefore requires monitoring.
"""


# =====================================================================
# 64. MODEL DRIFT
# =====================================================================

"""
Model drift refers broadly to degradation as real-world conditions
change.

Possible causes:

    data drift
    concept drift
    label drift
    environmental changes

Example:

A fraud model trained on 2025 transaction behavior may perform
differently when attackers change their strategies.
"""


# =====================================================================
# 65. RESPONSIBLE AI
# =====================================================================

"""
Responsible AI generally involves:

    fairness
    transparency
    accountability
    privacy
    security
    safety
    robustness
    reliability
    human oversight

Responsible AI should be considered throughout the lifecycle rather
than added only at the end.
"""


# =====================================================================
# 66. AI GOVERNANCE
# =====================================================================

"""
AI governance establishes organizational processes for managing AI.

Possible components:

    AI inventory
    risk classification
    model approval
    documentation
    data governance
    security review
    privacy review
    evaluation standards
    monitoring
    incident management
    auditability
    human oversight

A mature organization treats AI as both:

    technical infrastructure

and:

    organizational risk.
"""


# =====================================================================
# 67. SYMBOLIC + STATISTICAL + NEURAL HYBRID
# =====================================================================

class HybridAISystem:
    """
    Simplified conceptual hybrid AI system.

    It demonstrates how explicit rules can be combined with a
    statistical prediction.
    """

    def __init__(
        self,
        risk_threshold: float = 0.8
    ) -> None:

        self.risk_threshold = risk_threshold

    def rule_check(
        self,
        transaction_amount: float
    ) -> bool:

        return transaction_amount > 100000

    def statistical_score(
        self,
        transaction_amount: float
    ) -> float:

        # This is only a conceptual score, not a real fraud model.
        return min(
            0.99,
            transaction_amount / 200000
        )

    def decide(
        self,
        transaction_amount: float
    ) -> str:

        rule_triggered = self.rule_check(
            transaction_amount
        )

        probability = self.statistical_score(
            transaction_amount
        )

        if rule_triggered:

            return (
                "MANUAL REVIEW: explicit business rule triggered."
            )

        if probability >= self.risk_threshold:

            return (
                "MANUAL REVIEW: statistical risk is high."
            )

        return "ALLOW: no review condition triggered."


# =====================================================================
# 68. END-TO-END EXAMPLE
# =====================================================================

def end_to_end_example() -> None:

    print("\nEND-TO-END AI TAXONOMY EXAMPLE")
    print("=" * 60)

    problem = (
        "Create an AI system that detects fraudulent transactions."
    )

    print(f"Problem: {problem}")

    print("\nPossible architecture:")

    print("""
    Transaction
        |
        v
    Rule Engine
        |
        v
    Machine Learning Model
        |
        v
    Risk Score
        |
        v
    Human Review
        |
        v
    Final Decision
    """)

    hybrid_system = HybridAISystem()

    amounts = [
        5000,
        50000,
        120000,
        190000
    ]

    for amount in amounts:

        decision = hybrid_system.decide(amount)

        print(
            f"Transaction: {amount:>8,.0f} -> {decision}"
        )


# =====================================================================
# 69. AI TAXONOMY CHEAT SHEET
# =====================================================================

AI_CHEAT_SHEET = """
ARTIFICIAL INTELLIGENCE
    Broad field of intelligent computational systems.

SYMBOLIC AI
    Explicit symbols, rules, logic, and knowledge representation.

STATISTICAL AI
    Probability, statistics, uncertainty, and inference.

MACHINE LEARNING
    Learning patterns from data.

DISCRIMINATIVE AI
    Predicting/distinguishing outcomes, commonly modeling P(y|x).

DEEP LEARNING
    Machine learning using multi-layer neural networks.

GENERATIVE AI
    Producing new content/data, often modeling distributions or
    conditional distributions.

FOUNDATION MODEL
    Broadly pretrained reusable model adapted to many tasks.

LLM
    Large neural language model capable of modeling/generating
    language.

RAG
    Retrieval + generation.

AGENT
    Model/system capable of planning, tool use, observation, and
    action.

HYBRID AI
    Combination of symbolic, statistical, neural, retrieval, and/or
    other approaches.
"""


# =====================================================================
# 70. INTERVIEW QUESTIONS
# =====================================================================

INTERVIEW_QUESTIONS = [
    "What is Artificial Intelligence?",
    "What is Symbolic AI?",
    "What is Statistical AI?",
    "What is Machine Learning?",
    "How is Machine Learning different from traditional programming?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is self-supervised learning?",
    "What is reinforcement learning?",
    "What is discriminative AI?",
    "What is generative AI?",
    "What is the difference between P(y|x) and P(x|y)?",
    "What is Deep Learning?",
    "Why are neural networks called deep?",
    "What is a neuron?",
    "What is an activation function?",
    "What is backpropagation?",
    "What is gradient descent?",
    "What is a Transformer?",
    "What is self-attention?",
    "What is an embedding?",
    "What is tokenization?",
    "What is an LLM?",
    "What is a foundation model?",
    "What is fine-tuning?",
    "What is LoRA?",
    "What is RAG?",
    "What is an AI agent?",
    "What is hybrid AI?",
    "What is overfitting?",
    "What is underfitting?",
    "What is data leakage?",
    "What is distribution shift?",
    "What is model drift?",
    "What is hallucination?",
    "What is responsible AI?",
    "What is AI governance?",
]


# =====================================================================
# 71. PRACTICE QUESTIONS
# =====================================================================

PRACTICE_QUESTIONS = [
    {
        "question":
            "A system uses explicit IF-THEN rules. "
            "Which AI approach is this?",
        "answer":
            "Symbolic AI"
    },

    {
        "question":
            "A system estimates the probability that an email is spam.",
        "answer":
            "Statistical/discriminative machine learning"
    },

    {
        "question":
            "A model predicts whether a transaction is fraudulent.",
        "answer":
            "Discriminative AI"
    },

    {
        "question":
            "A model writes a new paragraph from a prompt.",
        "answer":
            "Generative AI"
    },

    {
        "question":
            "A neural network contains many layers.",
        "answer":
            "Deep Learning"
    },

    {
        "question":
            "A model predicts the next token in a sentence.",
        "answer":
            "Autoregressive generative language modeling"
    },

    {
        "question":
            "A system retrieves company documents before asking an LLM "
            "to answer.",
        "answer":
            "Retrieval-Augmented Generation"
    },
]


def print_practice_questions() -> None:

    print("\nPRACTICE QUESTIONS")
    print("=" * 60)

    for number, item in enumerate(
        PRACTICE_QUESTIONS,
        start=1
    ):

        print(f"{number}. {item['question']}")
        print(f"   Answer: {item['answer']}")


# =====================================================================
# 72. LEARNING ROADMAP
# =====================================================================

LEARNING_ROADMAP = [
    "1. Python programming fundamentals",
    "2. Mathematics for AI",
    "3. Probability and statistics",
    "4. Linear algebra",
    "5. Optimization",
    "6. Classical machine learning",
    "7. Supervised learning",
    "8. Unsupervised learning",
    "9. Model evaluation",
    "10. Neural networks",
    "11. Deep learning",
    "12. CNNs",
    "13. RNNs/LSTMs",
    "14. Attention",
    "15. Transformers",
    "16. Generative AI",
    "17. Large language models",
    "18. Embeddings",
    "19. RAG",
    "20. Fine-tuning",
    "21. Agents",
    "22. AI evaluation",
    "23. AI safety",
    "24. AI governance",
    "25. Production AI/MLOps",
]


# =====================================================================
# 73. MAIN PROGRAM
# =====================================================================

def main() -> None:

    print("=" * 80)
    print("AI TAXONOMY: COMPLETE PYTHON LEARNING GUIDE")
    print("=" * 80)

    print("""
This script covers:

    Artificial Intelligence
    Symbolic AI
    Statistical AI
    Machine Learning
    Discriminative AI
    Deep Learning
    Generative AI
    Foundation Models
    LLMs
    Transformers
    RAG
    AI Agents
    Hybrid AI
    Responsible AI
    AI Governance
""")

    symbolic_ai_demo()

    statistical_ai_demo()

    machine_learning_demo()

    discriminative_demo()

    neural_network_demo()

    temperature_demo()

    print_taxonomy()

    print_comparison()

    print("\nAI DECISION GUIDE")
    print("=" * 60)

    example_problems = [
        "Use explicit rules and formal logic",
        "Predict whether a customer will churn",
        "Generate text from a prompt",
        "Handle uncertainty and probability",
        "Discover groups in customer data",
    ]

    for problem in example_problems:

        print(
            f"\nProblem: {problem}"
        )

        print(
            f"Recommendation: "
            f"{choose_ai_approach(problem)}"
        )

    end_to_end_example()

    print_practice_questions()

    print("\nAI CHEAT SHEET")
    print("=" * 60)
    print(AI_CHEAT_SHEET)

    print("\nLEARNING ROADMAP")
    print("=" * 60)

    for item in LEARNING_ROADMAP:
        print(item)

    print("\nCOMMON MISCONCEPTIONS")
    print("=" * 60)

    for misconception, correction in COMMON_MISCONCEPTIONS.items():

        print(f"\n{misconception}")
        print(f"    {correction}")

    print("\n" + "=" * 80)
    print("END OF AI TAXONOMY GUIDE")
    print("=" * 80)


if __name__ == "__main__":
    main()
