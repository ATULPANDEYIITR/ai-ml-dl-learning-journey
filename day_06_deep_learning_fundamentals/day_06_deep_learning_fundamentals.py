"""
Deep Learning Fundamentals
===========================

A self-contained study and demonstration script covering:

1. What deep learning is
2. Artificial neurons
3. Features, inputs, outputs, targets, predictions
4. Weights and biases
5. Linear combinations
6. Activation functions
7. Single neurons
8. Layers and network architecture
9. Forward propagation
10. Loss functions
11. Gradients and derivatives
12. Backpropagation
13. Gradient descent
14. Learning rate
15. Training loops
16. Multi-layer neural networks
17. Binary classification
18. Regression
19. Multiclass classification
20. Common activation functions
21. Weight initialization
22. Underfitting and overfitting
23. Numerical stability
24. Vectorization
25. A neural network implemented from scratch using NumPy
26. XOR as a nonlinear problem
27. Practical debugging and validation
28. Advanced architectural and optimization concepts
29. Production considerations
30. Common mistakes and important edge cases

The script uses only Python's standard library and NumPy.
NumPy is used for array operations so that matrix-based neural-network
operations can be demonstrated clearly.

Run:
    python deep_learning_fundamentals.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "This script requires NumPy. Install it with: pip install numpy"
    ) from exc


# ============================================================================
# 1. BASIC TERMINOLOGY
# ============================================================================

def section(title: str) -> None:
    """Print a readable section heading."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def explain_basic_terminology() -> None:
    section("1. Basic Deep Learning Terminology")

    terminology = {
        "Deep learning": (
            "A branch of machine learning that uses neural networks with "
            "one or more hidden layers to learn representations from data."
        ),
        "Feature": (
            "An input variable supplied to a model. For a house-price model, "
            "area and number of bedrooms can be features."
        ),
        "Target": (
            "The desired output associated with a training example."
        ),
        "Prediction": (
            "The output produced by the neural network for an input."
        ),
        "Parameter": (
            "A value learned during training, primarily weights and biases."
        ),
        "Hyperparameter": (
            "A configuration chosen by the practitioner, such as learning "
            "rate, number of layers, or number of training epochs."
        ),
        "Neuron": (
            "A computational unit that combines inputs using weights and a "
            "bias and usually applies an activation function."
        ),
        "Layer": (
            "A collection of neurons operating at the same stage of a "
            "network."
        ),
        "Epoch": (
            "One complete pass through the training dataset."
        ),
        "Batch": (
            "A subset of training examples processed together."
        ),
        "Loss": (
            "A numerical measure of how different predictions are from "
            "desired targets."
        ),
    }

    for name, description in terminology.items():
        print(f"{name:20s}: {description}")


# ============================================================================
# 2. THE MATHEMATICAL MODEL OF A NEURON
# ============================================================================

def weighted_sum(
    inputs: Sequence[float],
    weights: Sequence[float],
    bias: float,
) -> float:
    """
    Calculate the pre-activation value:

        z = x1*w1 + x2*w2 + ... + xn*wn + b

    The dot product is the fundamental operation performed by a dense neuron.
    """
    if len(inputs) != len(weights):
        raise ValueError("Inputs and weights must have the same length.")

    return float(np.dot(inputs, weights) + bias)


def demonstrate_single_neuron() -> None:
    section("2. A Single Artificial Neuron")

    inputs = np.array([2.0, 3.0])
    weights = np.array([0.5, -0.25])
    bias = 1.0

    z = weighted_sum(inputs, weights, bias)

    print("Inputs :", inputs)
    print("Weights:", weights)
    print("Bias   :", bias)
    print("z = x·w + b =", z)

    print(
        "\nThe neuron first computes a weighted sum. "
        "The result is called the pre-activation value z."
    )


# ============================================================================
# 3. BIAS
# ============================================================================

def demonstrate_bias() -> None:
    section("3. Understanding the Bias")

    x = 2.0
    w = 3.0

    without_bias = x * w
    with_bias = x * w + 5.0

    print("Input:", x)
    print("Weight:", w)
    print("Without bias:", without_bias)
    print("With bias 5:", with_bias)

    print(
        "\nThe bias shifts the neuron's activation threshold. "
        "It gives the neuron an additional learned degree of freedom."
    )


# ============================================================================
# 4. ACTIVATION FUNCTIONS
# ============================================================================

def step_function(z: float) -> float:
    """Return 1 for non-negative z and 0 otherwise."""
    return 1.0 if z >= 0 else 0.0


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """
    Logistic sigmoid.

        sigmoid(z) = 1 / (1 + exp(-z))

    Clipping is used to avoid overflow for very large magnitudes.
    """
    z_array = np.asarray(z, dtype=float)
    clipped = np.clip(z_array, -500, 500)
    result = 1.0 / (1.0 + np.exp(-clipped))

    if np.ndim(z) == 0:
        return float(result)
    return result


def sigmoid_derivative_from_output(output: np.ndarray | float) -> np.ndarray | float:
    """
    Derivative of sigmoid when its output a = sigmoid(z) is already known.

        da/dz = a(1-a)
    """
    result = np.asarray(output) * (1.0 - np.asarray(output))

    if np.ndim(output) == 0:
        return float(result)
    return result


def tanh(z: np.ndarray | float) -> np.ndarray | float:
    """Hyperbolic tangent activation."""
    result = np.tanh(z)
    if np.ndim(z) == 0:
        return float(result)
    return result


def tanh_derivative_from_output(output: np.ndarray | float) -> np.ndarray | float:
    """Derivative of tanh when its output is already known."""
    result = 1.0 - np.asarray(output) ** 2

    if np.ndim(output) == 0:
        return float(result)
    return result


def relu(z: np.ndarray | float) -> np.ndarray | float:
    """
    Rectified Linear Unit:

        ReLU(z) = max(0, z)
    """
    result = np.maximum(0, np.asarray(z))

    if np.ndim(z) == 0:
        return float(result)
    return result


def relu_derivative(z: np.ndarray | float) -> np.ndarray | float:
    """Derivative of ReLU away from its non-differentiable point at zero."""
    result = (np.asarray(z) > 0).astype(float)

    if np.ndim(z) == 0:
        return float(result)
    return result


def leaky_relu(
    z: np.ndarray | float,
    negative_slope: float = 0.01,
) -> np.ndarray | float:
    """Leaky ReLU keeps a small gradient for negative values."""
    result = np.where(np.asarray(z) > 0, np.asarray(z), negative_slope * np.asarray(z))

    if np.ndim(z) == 0:
        return float(result)
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Convert logits into a probability distribution.

    Numerical stability is obtained by subtracting the largest logit.
    """
    logits = np.asarray(logits, dtype=float)

    if logits.ndim == 1:
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials)

    if logits.ndim == 2:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    raise ValueError("Softmax expects a 1D or 2D array.")


def demonstrate_activation_functions() -> None:
    section("4. Activation Functions")

    values = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

    print("Input values:", values)
    print("Step       :", [step_function(float(x)) for x in values])
    print("Sigmoid    :", np.round(sigmoid(values), 4))
    print("Tanh       :", np.round(tanh(values), 4))
    print("ReLU       :", relu(values))
    print("Leaky ReLU :", np.round(leaky_relu(values), 4))

    logits = np.array([2.0, 1.0, 0.1])
    print("\nLogits:", logits)
    print("Softmax probabilities:", np.round(softmax(logits), 4))
    print("Probability sum:", np.sum(softmax(logits)))

    print(
        """
Activation-function roles:
- Step: historically useful for illustrating threshold logic, but unsuitable
  for ordinary gradient-based neural-network training because it has no useful
  gradient almost everywhere.
- Sigmoid: maps values to (0, 1), useful for binary output probabilities.
- Tanh: maps values to (-1, 1), historically common in hidden layers.
- ReLU: simple and computationally efficient; widely used in hidden layers.
- Leaky ReLU: reduces the risk of permanently inactive ReLU units.
- Softmax: converts a vector of logits into a categorical probability
  distribution.
"""
    )


# ============================================================================
# 5. ONE NEURON WITH ACTIVATION
# ============================================================================

def neuron(
    inputs: Sequence[float],
    weights: Sequence[float],
    bias: float,
    activation: Callable[[float], float],
) -> float:
    """Complete neuron: weighted sum followed by an activation function."""
    z = weighted_sum(inputs, weights, bias)
    return float(activation(z))


def demonstrate_activated_neuron() -> None:
    section("5. Complete Neuron: Weighted Sum + Activation")

    inputs = [1.0, 2.0, -1.0]
    weights = [0.8, -0.4, 0.5]
    bias = 0.2

    z = weighted_sum(inputs, weights, bias)
    output = sigmoid(z)

    print("Inputs:", inputs)
    print("Weights:", weights)
    print("Bias:", bias)
    print("Pre-activation z:", z)
    print("Sigmoid output:", output)


# ============================================================================
# 6. LAYERS
# ============================================================================

def dense_layer(
    inputs: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    activation: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Dense layer for a single example.

    inputs shape:
        (number_of_input_features,)

    weights shape:
        (number_of_neurons, number_of_input_features)

    biases shape:
        (number_of_neurons,)

    Output shape:
        (number_of_neurons,)
    """
    inputs = np.asarray(inputs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    biases = np.asarray(biases, dtype=float)

    z = weights @ inputs + biases
    return np.asarray(activation(z), dtype=float)


def demonstrate_dense_layer() -> None:
    section("6. A Dense Layer")

    inputs = np.array([1.0, 2.0, 3.0])

    weights = np.array(
        [
            [0.1, 0.2, 0.3],
            [-0.5, 0.4, 0.2],
        ]
    )

    biases = np.array([0.5, -0.2])

    output = dense_layer(inputs, weights, biases, relu)

    print("Input shape:", inputs.shape)
    print("Weight shape:", weights.shape)
    print("Bias shape:", biases.shape)
    print("Layer output:", output)
    print("Output shape:", output.shape)

    print(
        "\nA layer containing two neurons receives the same three input "
        "features but has a different weight vector and bias for each neuron."
    )


# ============================================================================
# 7. NETWORK ARCHITECTURE
# ============================================================================

@dataclass
class NetworkArchitecture:
    """Simple description of a fully connected feed-forward network."""

    input_size: int
    hidden_sizes: list[int]
    output_size: int

    def describe(self) -> None:
        sizes = [self.input_size, *self.hidden_sizes, self.output_size]

        print("Network architecture:")
        for index, size in enumerate(sizes):
            if index == 0:
                label = "Input"
            elif index == len(sizes) - 1:
                label = "Output"
            else:
                label = f"Hidden {index}"
            print(f"  {label:10s}: {size} neurons")

        parameter_count = 0
        for previous, current in zip(sizes[:-1], sizes[1:]):
            parameter_count += previous * current
            parameter_count += current

        print("Total trainable parameters:", parameter_count)


def demonstrate_architecture() -> None:
    section("7. Neural Network Architecture")

    architecture = NetworkArchitecture(
        input_size=4,
        hidden_sizes=[8, 6],
        output_size=3,
    )
    architecture.describe()

    print(
        """
A feed-forward network is commonly described by:
input layer -> hidden layer(s) -> output layer.

A network is called "deep" when it contains multiple computational layers.
The precise terminology used for depth varies across contexts, but the
important idea is that successive layers can learn increasingly transformed
representations.
"""
    )


# ============================================================================
# 8. FORWARD PROPAGATION
# ============================================================================

def forward_two_layer_network(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward propagation through:

        input -> ReLU hidden layer -> sigmoid output

    Returns:
        hidden pre-activation z1
        hidden activation a1
        output activation a2
    """
    z1 = w1 @ x + b1
    a1 = relu(z1)

    z2 = w2 @ a1 + b2
    a2 = sigmoid(z2)

    return z1, a1, np.asarray(a2)


def demonstrate_forward_propagation() -> None:
    section("8. Forward Propagation")

    x = np.array([0.5, 1.0])

    w1 = np.array(
        [
            [0.8, -0.2],
            [0.3, 0.5],
            [-0.4, 0.7],
        ]
    )
    b1 = np.array([0.1, 0.0, 0.2])

    w2 = np.array([[0.4, -0.3, 0.9]])
    b2 = np.array([-0.1])

    z1, a1, a2 = forward_two_layer_network(x, w1, b1, w2, b2)

    print("Input x:", x)
    print("Hidden pre-activation z1:", z1)
    print("Hidden activation a1:", a1)
    print("Output probability a2:", a2)

    print(
        "\nForward propagation means computing predictions from inputs by "
        "passing information through the network in the forward direction."
    )


# ============================================================================
# 9. LOSS FUNCTIONS
# ============================================================================

def mean_squared_error(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Mean squared error, commonly used for regression."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)

    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have matching shapes.")

    return float(np.mean((predictions - targets) ** 2))


def binary_cross_entropy(
    predictions: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """
    Binary cross-entropy.

    Predictions are clipped away from exactly 0 and 1 to avoid log(0).
    """
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)

    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have matching shapes.")

    clipped = np.clip(predictions, epsilon, 1.0 - epsilon)

    loss = -np.mean(
        targets * np.log(clipped)
        + (1.0 - targets) * np.log(1.0 - clipped)
    )
    return float(loss)


def categorical_cross_entropy(
    probabilities: np.ndarray,
    target_indices: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Categorical cross-entropy for integer class labels."""
    probabilities = np.asarray(probabilities, dtype=float)
    target_indices = np.asarray(target_indices, dtype=int)

    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array.")

    if len(target_indices) != len(probabilities):
        raise ValueError("One target class is required per example.")

    clipped = np.clip(probabilities, epsilon, 1.0)
    correct_class_probabilities = clipped[
        np.arange(len(target_indices)),
        target_indices,
    ]

    return float(-np.mean(np.log(correct_class_probabilities)))


def demonstrate_losses() -> None:
    section("9. Loss Functions")

    targets = np.array([1.0, 0.0, 1.0])
    predictions = np.array([0.9, 0.2, 0.6])

    print("Targets:", targets)
    print("Predictions:", predictions)
    print("Mean squared error:", mean_squared_error(predictions, targets))
    print(
        "Binary cross-entropy:",
        binary_cross_entropy(predictions, targets),
    )

    class_probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.05, 0.15, 0.8],
        ]
    )
    class_targets = np.array([0, 1, 2])

    print(
        "Categorical cross-entropy:",
        categorical_cross_entropy(class_probabilities, class_targets),
    )

    print(
        """
Loss is the training objective.

The model does not directly "know" that a prediction is wrong. A loss
function converts the difference between predictions and targets into a
scalar value that can be minimized.

Typical pairings:
- Regression -> mean squared error or another regression loss.
- Binary classification -> binary cross-entropy with a sigmoid output.
- Multiclass classification -> categorical cross-entropy with softmax output.
"""
    )


# ============================================================================
# 10. DERIVATIVES AND GRADIENTS
# ============================================================================

def numerical_derivative(
    function: Callable[[float], float],
    x: float,
    epsilon: float = 1e-6,
) -> float:
    """
    Central-difference numerical derivative.

        f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    """
    return (function(x + epsilon) - function(x - epsilon)) / (2.0 * epsilon)


def demonstrate_derivatives() -> None:
    section("10. Derivatives and Gradients")

    x = 3.0

    def square(value: float) -> float:
        return value**2

    numerical = numerical_derivative(square, x)
    analytical = 2.0 * x

    print("Function: f(x) = x²")
    print("x:", x)
    print("Analytical derivative:", analytical)
    print("Numerical derivative :", numerical)

    print(
        """
A derivative describes how a function changes with respect to a variable.

A gradient generalizes this idea to many parameters. A neural network can
have thousands, millions, or billions of parameters. The gradient tells us
how changing each parameter would change the loss locally.

Backpropagation efficiently computes these gradients by applying the chain
rule through the computational graph.
"""
    )


# ============================================================================
# 11. CHAIN RULE
# ============================================================================

def demonstrate_chain_rule() -> None:
    section("11. Chain Rule in a Tiny Network")

    # y = sigmoid(2x + 1)
    x = 0.5
    w = 2.0
    b = 1.0

    z = w * x + b
    y = float(sigmoid(z))

    # dy/dz for sigmoid
    dy_dz = float(sigmoid_derivative_from_output(y))

    # dz/dw = x
    dz_dw = x

    # Therefore:
    # dy/dw = dy/dz * dz/dw
    dy_dw = dy_dz * dz_dw

    print("x =", x)
    print("w =", w)
    print("b =", b)
    print("z =", z)
    print("y =", y)
    print("dy/dz =", dy_dz)
    print("dz/dw =", dz_dw)
    print("dy/dw =", dy_dw)

    print(
        "\nThis is the core pattern used repeatedly during backpropagation."
    )


# ============================================================================
# 12. GRADIENT DESCENT
# ============================================================================

def gradient_descent_demo() -> None:
    section("12. Gradient Descent")

    # Minimize f(w) = (w - 3)^2.
    # The derivative is 2(w - 3).
    weight = 10.0
    learning_rate = 0.1

    print("Initial parameter:", weight)

    for step in range(10):
        gradient = 2.0 * (weight - 3.0)
        weight -= learning_rate * gradient

        print(
            f"Step {step + 1:2d}: "
            f"gradient={gradient:8.4f}, "
            f"parameter={weight:8.4f}"
        )

    print(
        """
The update rule is:

    parameter_new = parameter_old - learning_rate * gradient

The negative sign moves the parameter in the direction that locally reduces
the objective.

Important hyperparameters:
- Learning rate controls update magnitude.
- Number of epochs controls how long training proceeds.
- Batch size controls how much data contributes to each update.
"""
    )


# ============================================================================
# 13. MANUAL BACKPROPAGATION FOR ONE NEURON
# ============================================================================

def train_single_neuron(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.5,
    epochs: int = 2000,
) -> tuple[np.ndarray, float, list[float]]:
    """
    Train a single sigmoid neuron for binary classification.

    This implementation explicitly demonstrates gradient descent.

    For sigmoid + binary cross-entropy:

        dL/dz = prediction - target

    for the per-example loss, which makes the parameter gradients especially
    simple.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    losses: list[float] = []

    for _ in range(epochs):
        logits = x @ weights + bias
        predictions = sigmoid(logits)

        loss = binary_cross_entropy(predictions, y)
        losses.append(loss)

        error = predictions - y

        gradient_weights = (x.T @ error) / len(x)
        gradient_bias = float(np.mean(error))

        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    return weights, bias, losses


def demonstrate_single_neuron_training() -> None:
    section("13. Training a Single Neuron")

    # OR gate is linearly separable.
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0.0, 1.0, 1.0, 1.0])

    weights, bias, losses = train_single_neuron(
        x,
        y,
        learning_rate=0.5,
        epochs=2000,
    )

    probabilities = sigmoid(x @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)

    print("Learned weights:", np.round(weights, 4))
    print("Learned bias:", round(bias, 4))
    print("Final loss:", losses[-1])
    print("Probabilities:", np.round(probabilities, 4))
    print("Predicted classes:", predictions)
    print("Targets:", y.astype(int))


# ============================================================================
# 14. WHY ONE NEURON CANNOT SOLVE XOR
# ============================================================================

def demonstrate_xor_limitation() -> None:
    section("14. Linear Separability and XOR")

    xor_x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    xor_y = np.array([0.0, 1.0, 1.0, 0.0])

    print("XOR inputs:")
    print(xor_x)
    print("XOR targets:", xor_y)

    print(
        """
A single neuron computes:

    activation(w1*x1 + w2*x2 + b)

Before the activation, the decision boundary is a hyperplane. For two
features this is a line.

XOR is not linearly separable. A single linear decision boundary cannot
separate its positive and negative examples.

A hidden layer allows the network to compose multiple nonlinear
transformations, making XOR learnable.
"""
    )


# ============================================================================
# 15. NEURAL NETWORK FROM SCRATCH
# ============================================================================

class NeuralNetwork:
    """
    A small fully connected neural network implemented from scratch.

    Architecture:
        input -> hidden ReLU -> output sigmoid

    Intended for educational use. It deliberately exposes the mathematical
    operations instead of hiding them behind a machine-learning framework.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        seed: int = 42,
    ) -> None:
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError("Layer sizes must be positive.")

        rng = np.random.default_rng(seed)

        # He initialization is appropriate for ReLU hidden units.
        self.w1 = rng.normal(
            0.0,
            math.sqrt(2.0 / input_size),
            size=(hidden_size, input_size),
        )
        self.b1 = np.zeros(hidden_size)

        # Xavier-style scaling is reasonable for a sigmoid output layer.
        self.w2 = rng.normal(
            0.0,
            math.sqrt(1.0 / hidden_size),
            size=(output_size, hidden_size),
        )
        self.b2 = np.zeros(output_size)

    def forward(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform forward propagation.

        x shape:
            (number_of_examples, input_size)
        """
        x = np.asarray(x, dtype=float)

        if x.ndim != 2:
            raise ValueError("Input must be a 2D array.")

        z1 = x @ self.w1.T + self.b1
        a1 = relu(z1)

        z2 = a1 @ self.w2.T + self.b2
        a2 = sigmoid(z2)

        return z1, a1, z2, np.asarray(a2)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return sigmoid probabilities."""
        return self.forward(x)[-1]

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using a configurable threshold."""
        if not 0.0 < threshold < 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        probabilities = self.predict_proba(x)
        return (probabilities >= threshold).astype(int)

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.1,
        epochs: int = 5000,
        verbose_every: int = 500,
    ) -> list[float]:
        """
        Train using full-batch gradient descent and explicit backpropagation.

        This method demonstrates the chain rule directly.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 2:
            raise ValueError("x must be 2D.")

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if y.shape != (len(x), 1):
            raise ValueError("For this network, y must have shape (n, 1).")

        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        if epochs <= 0:
            raise ValueError("Epochs must be positive.")

        losses: list[float] = []

        for epoch in range(1, epochs + 1):
            z1, a1, _z2, a2 = self.forward(x)

            loss = binary_cross_entropy(a2, y)
            losses.append(loss)

            # For sigmoid + binary cross-entropy:
            # dL/dz2 = a2 - y
            dz2 = a2 - y

            # z2 = a1 @ w2.T + b2
            # Therefore:
            # dL/dw2 = dz2.T @ a1 / n
            n = len(x)
            dw2 = (dz2.T @ a1) / n
            db2 = np.mean(dz2, axis=0)

            # Backpropagate into hidden activations.
            da1 = dz2 @ self.w2

            # ReLU derivative is 1 for positive z1 and 0 for negative z1.
            dz1 = da1 * relu_derivative(z1)

            # z1 = x @ w1.T + b1
            dw1 = (dz1.T @ x) / n
            db1 = np.mean(dz1, axis=0)

            # Gradient-descent parameter update.
            self.w2 -= learning_rate * dw2
            self.b2 -= learning_rate * db2
            self.w1 -= learning_rate * dw1
            self.b1 -= learning_rate * db1

            if verbose_every > 0 and (
                epoch == 1 or epoch % verbose_every == 0
            ):
                print(f"Epoch {epoch:5d} | Loss {loss:.6f}")

        return losses


def demonstrate_xor_network() -> None:
    section("15. Multi-Layer Network Learning XOR")

    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0.0, 1.0, 1.0, 0.0])

    network = NeuralNetwork(
        input_size=2,
        hidden_size=8,
        output_size=1,
        seed=7,
    )

    losses = network.train(
        x,
        y,
        learning_rate=0.1,
        epochs=5000,
        verbose_every=1000,
    )

    probabilities = network.predict_proba(x).reshape(-1)
    predictions = network.predict(x).reshape(-1)

    print("\nFinal loss:", losses[-1])
    print("Probabilities:", np.round(probabilities, 4))
    print("Predictions:", predictions)
    print("Targets:", y.astype(int))


# ============================================================================
# 16. BATCH DIMENSIONS
# ============================================================================

def demonstrate_batch_processing() -> None:
    section("16. Batch Processing and Matrix Shapes")

    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [0.0, 4.0, 2.0],
            [3.0, 1.0, 5.0],
        ]
    )

    weights = np.array(
        [
            [0.1, 0.2, 0.3],
            [-0.1, 0.5, 0.4],
        ]
    )

    biases = np.array([0.5, -0.3])

    z = x @ weights.T + biases
    output = relu(z)

    print("Input batch shape :", x.shape)
    print("Weight shape      :", weights.shape)
    print("Bias shape        :", biases.shape)
    print("Output shape      :", output.shape)
    print("\nOutput:")
    print(output)

    print(
        """
If X has shape (batch_size, input_features) and W has shape
(output_units, input_features), then:

    Z = X @ W.T + b

produces shape:

    (batch_size, output_units)

Vectorized operations allow many examples to be processed using optimized
array operations rather than an explicit Python loop over every neuron.
"""
    )


# ============================================================================
# 17. REGRESSION
# ============================================================================

class LinearRegressionNeuron:
    """Single-neuron regression model trained with mean squared error."""

    def __init__(self, input_size: int) -> None:
        self.weights = np.zeros(input_size, dtype=float)
        self.bias = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 1000,
    ) -> list[float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        losses: list[float] = []

        for _ in range(epochs):
            predictions = self.predict(x)
            errors = predictions - y

            loss = mean_squared_error(predictions, y)
            losses.append(loss)

            # MSE derivative:
            # dL/dprediction = 2(prediction - target) / n
            gradient_prediction = 2.0 * errors / len(x)

            gradient_weights = x.T @ gradient_prediction
            gradient_bias = float(np.sum(gradient_prediction))

            self.weights -= learning_rate * gradient_weights
            self.bias -= learning_rate * gradient_bias

        return losses


def demonstrate_regression() -> None:
    section("17. Neural Networks for Regression")

    # Target relationship is approximately y = 2x + 1.
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

    model = LinearRegressionNeuron(input_size=1)
    losses = model.train(
        x,
        y,
        learning_rate=0.01,
        epochs=2000,
    )

    predictions = model.predict(x)

    print("Learned weight:", round(model.weights[0], 4))
    print("Learned bias  :", round(model.bias, 4))
    print("Final loss    :", losses[-1])
    print("Predictions   :", np.round(predictions, 4))


# ============================================================================
# 18. MULTICLASS CLASSIFICATION
# ============================================================================

def demonstrate_multiclass_output() -> None:
    section("18. Multiclass Classification")

    logits = np.array(
        [
            [2.5, 0.3, -1.0],
            [0.1, 2.0, 0.5],
            [-1.0, 0.2, 3.0],
        ]
    )

    probabilities = softmax(logits)
    predicted_classes = np.argmax(probabilities, axis=1)

    print("Logits:")
    print(logits)

    print("\nSoftmax probabilities:")
    print(np.round(probabilities, 4))

    print("\nPredicted class indices:", predicted_classes)
    print("Row probability sums:", np.sum(probabilities, axis=1))

    print(
        """
For K mutually exclusive classes, an output layer commonly contains K
logits. Softmax converts those logits into probabilities that sum to one.

The predicted class is commonly:

    argmax(probabilities)

The model itself produces continuous numerical outputs. A class label is
usually obtained afterward by applying a decision rule such as argmax.
"""
    )


# ============================================================================
# 19. LOGITS AND NUMERICAL STABILITY
# ============================================================================

def naive_softmax(logits: np.ndarray) -> np.ndarray:
    """Intentionally naive implementation used to demonstrate instability."""
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials)


def demonstrate_numerical_stability() -> None:
    section("19. Numerical Stability")

    large_logits = np.array([1000.0, 1001.0, 999.0])

    print("Large logits:", large_logits)

    try:
        print("Naive softmax:", naive_softmax(large_logits))
    except FloatingPointError as exc:
        print("Numerical problem:", exc)

    print("Stable softmax:", softmax(large_logits))

    print(
        """
Exponentials grow extremely quickly. Directly evaluating exp(1000) can
overflow floating-point limits.

The stable softmax transformation subtracts max(logits) before exponentiation.
This does not change the final probability distribution because the same
constant is subtracted from every logit.
"""
    )


# ============================================================================
# 20. WEIGHT INITIALIZATION
# ============================================================================

def demonstrate_initialization() -> None:
    section("20. Weight Initialization")

    rng = np.random.default_rng(123)

    input_size = 100
    output_size = 50

    standard = rng.normal(0.0, 1.0, size=(output_size, input_size))
    xavier = rng.normal(
        0.0,
        math.sqrt(1.0 / input_size),
        size=(output_size, input_size),
    )
    he = rng.normal(
        0.0,
        math.sqrt(2.0 / input_size),
        size=(output_size, input_size),
    )

    print("Standard normal std:", np.std(standard))
    print("Xavier std         :", np.std(xavier))
    print("He std             :", np.std(he))

    print(
        """
Initialization matters because activations and gradients should not
systematically explode or vanish as they pass through many layers.

Common ideas:
- Xavier/Glorot initialization is often associated with sigmoid or tanh
  networks.
- He initialization is especially common with ReLU-like activations.
- Initializing every weight to exactly zero is generally problematic because
  neurons in the same layer can remain symmetric and learn identical
  representations.
- Biases are often initialized to zero, though some architectures use other
  choices.
"""
    )


# ============================================================================
# 21. VANISHING AND EXPLODING GRADIENTS
# ============================================================================

def demonstrate_gradient_pathologies() -> None:
    section("21. Vanishing and Exploding Gradients")

    sigmoid_slopes = sigmoid_derivative_from_output(
        np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    )

    print("Sigmoid derivative at representative outputs:")
    print(np.round(sigmoid_slopes, 6))

    print(
        """
The sigmoid derivative reaches a maximum of 0.25. When sigmoid units are
deeply saturated near 0 or 1, the derivative becomes very small.

During repeated chain-rule multiplication across many layers, small
derivatives can produce vanishing gradients.

Very large derivatives or poorly scaled transformations can cause exploding
gradients.

Mitigation techniques include:
- Appropriate initialization.
- ReLU-family activations.
- Normalization techniques where appropriate.
- Residual/skip connections.
- Careful learning-rate selection.
- Gradient clipping in situations where exploding gradients occur.
"""
    )


# ============================================================================
# 22. OVERFITTING AND UNDERFITTING
# ============================================================================

def demonstrate_fit_concepts() -> None:
    section("22. Underfitting and Overfitting")

    print(
        """
Underfitting:
- Model is too simple to capture important structure.
- Training performance is poor.
- Validation performance is also poor.

Overfitting:
- Model learns training-specific patterns that do not generalize.
- Training loss can become very low.
- Validation loss may stop improving or increase.

Possible responses:
- Underfitting -> increase capacity, improve features, train longer, or
  reduce excessive regularization.
- Overfitting -> obtain more data, reduce model capacity, apply regularization,
  use data augmentation where appropriate, or use early stopping.

Training performance alone is not sufficient to assess generalization.
"""
    )


# ============================================================================
# 23. REGULARIZATION
# ============================================================================

def l2_regularization(
    weights: Iterable[np.ndarray],
    coefficient: float,
) -> float:
    """Compute an L2 penalty over a collection of weight matrices."""
    if coefficient < 0:
        raise ValueError("Regularization coefficient cannot be negative.")

    squared_sum = sum(float(np.sum(matrix**2)) for matrix in weights)
    return coefficient * squared_sum


def demonstrate_regularization() -> None:
    section("23. L2 Regularization")

    weights = [
        np.array([[1.0, -2.0], [0.5, 3.0]]),
        np.array([[0.2, -0.7]]),
    ]

    penalty = l2_regularization(weights, coefficient=0.01)

    print("L2 penalty:", penalty)

    print(
        """
L2 regularization adds a penalty related to squared parameter magnitudes.
Conceptually:

    total_loss = data_loss + lambda * sum(weights²)

It discourages unnecessarily large weights and can improve generalization.

Other regularization mechanisms include dropout, early stopping, data
augmentation, and architectural constraints. Their effects and appropriate
uses differ by problem.
"""
    )


# ============================================================================
# 24. NORMALIZATION AND SCALING
# ============================================================================

def standardize_features(
    x: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features:

        x_scaled = (x - mean) / std

    Training statistics should be computed on the training set and reused
    for validation/test data. A tiny fallback prevents division by zero for
    constant features.
    """
    x = np.asarray(x, dtype=float)

    if mean is None:
        mean = np.mean(x, axis=0)

    if std is None:
        std = np.std(x, axis=0)

    safe_std = np.where(std == 0.0, 1.0, std)
    scaled = (x - mean) / safe_std

    return scaled, np.asarray(mean), np.asarray(safe_std)


def demonstrate_feature_scaling() -> None:
    section("24. Feature Scaling")

    x_train = np.array(
        [
            [1000.0, 2.0],
            [1500.0, 3.0],
            [2000.0, 4.0],
        ]
    )

    scaled, mean, std = standardize_features(x_train)

    print("Original features:")
    print(x_train)

    print("\nTraining mean:", mean)
    print("Training std :", std)

    print("\nStandardized features:")
    print(np.round(scaled, 4))

    print(
        """
Feature scales can strongly affect optimization. If one feature ranges in
the thousands and another ranges between 0 and 1, gradients may become
poorly conditioned.

Common transformations include:
- Standardization: subtract mean and divide by standard deviation.
- Min-max scaling: map values to a specified interval.
- Domain-specific transformations such as logarithms for highly skewed data.

Scaling statistics must be learned from training data only. Computing them
using the entire dataset can cause data leakage.
"""
    )


# ============================================================================
# 25. TRAINING / VALIDATION / TEST SPLITS
# ============================================================================

def demonstrate_data_splits() -> None:
    section("25. Training, Validation, and Test Sets")

    total_examples = 1000

    train_count = int(total_examples * 0.70)
    validation_count = int(total_examples * 0.15)
    test_count = total_examples - train_count - validation_count

    print("Total examples     :", total_examples)
    print("Training examples  :", train_count)
    print("Validation examples:", validation_count)
    print("Test examples      :", test_count)

    print(
        """
Training set:
    Used to learn model parameters.

Validation set:
    Used for model selection and hyperparameter decisions.

Test set:
    Used for final unbiased evaluation after decisions have been made.

The exact split proportions depend on dataset size, task, temporal structure,
and evaluation requirements.

For time-dependent data, random splitting can be inappropriate because it
may allow future information to influence the apparent performance on past
examples.
"""
    )


# ============================================================================
# 26. MINI-BATCH TRAINING
# ============================================================================

def create_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create mini-batches and correctly handle a final partial batch."""
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of examples.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    indices = np.arange(len(x))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    batches: list[tuple[np.ndarray, np.ndarray]] = []

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batches.append((x[batch_indices], y[batch_indices]))

    return batches


def demonstrate_mini_batches() -> None:
    section("26. Mini-Batch Training")

    x = np.arange(10).reshape(10, 1)
    y = x**2

    batches = create_batches(
        x,
        y,
        batch_size=4,
        shuffle=False,
    )

    for index, (batch_x, batch_y) in enumerate(batches, start=1):
        print(
            f"Batch {index}: "
            f"size={len(batch_x)}, "
            f"x={batch_x.reshape(-1).tolist()}, "
            f"y={batch_y.reshape(-1).tolist()}"
        )

    print(
        """
If the dataset size is not divisible by batch size, the final batch can be
smaller. Production training code must handle this case correctly.

Full-batch gradient descent uses the entire dataset for each update.
Mini-batch gradient descent uses a subset.
Stochastic gradient descent commonly uses one example at a time.

Mini-batches are widely used because they provide a practical balance
between noisy updates and efficient hardware utilization.
"""
    )


# ============================================================================
# 27. LEARNING RATE EFFECTS
# ============================================================================

def simulate_learning_rate(
    learning_rate: float,
    initial_value: float = 10.0,
    steps: int = 20,
) -> list[float]:
    """Optimize f(x) = (x - 2)^2 with a specified learning rate."""
    value = initial_value
    history = [value]

    for _ in range(steps):
        gradient = 2.0 * (value - 2.0)
        value -= learning_rate * gradient
        history.append(value)

    return history


def demonstrate_learning_rates() -> None:
    section("27. Learning Rate Behavior")

    for learning_rate in [0.01, 0.1, 0.5, 1.0]:
        history = simulate_learning_rate(learning_rate)
        print(
            f"learning_rate={learning_rate:<4}: "
            f"final_value={history[-1]:.6f}"
        )

    print(
        """
A learning rate that is too small can make training unnecessarily slow.

A learning rate that is too large can:
- Overshoot useful regions.
- Cause oscillation.
- Cause divergence.

There is no universal best learning rate. Its appropriate value depends on
architecture, optimizer, data scaling, loss landscape, batch size, and other
training choices.
"""
    )


# ============================================================================
# 28. OPTIMIZER CONCEPTS
# ============================================================================

@dataclass
class MomentumOptimizer:
    """
    Educational implementation of momentum gradient descent.

        v = beta*v + gradient
        parameter = parameter - learning_rate*v
    """

    learning_rate: float = 0.01
    beta: float = 0.9
    velocity: float = 0.0

    def step(self, parameter: float, gradient: float) -> float:
        self.velocity = self.beta * self.velocity + gradient
        return parameter - self.learning_rate * self.velocity


@dataclass
class AdamScalar:
    """
    Scalar Adam optimizer demonstration.

    This is intentionally scalar to make the moving-average mechanism clear.
    """

    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    first_moment: float = 0.0
    second_moment: float = 0.0
    timestep: int = 0

    def step(self, parameter: float, gradient: float) -> float:
        self.timestep += 1

        self.first_moment = (
            self.beta1 * self.first_moment
            + (1.0 - self.beta1) * gradient
        )
        self.second_moment = (
            self.beta2 * self.second_moment
            + (1.0 - self.beta2) * gradient**2
        )

        corrected_first = self.first_moment / (
            1.0 - self.beta1**self.timestep
        )
        corrected_second = self.second_moment / (
            1.0 - self.beta2**self.timestep
        )

        return parameter - self.learning_rate * corrected_first / (
            math.sqrt(corrected_second) + self.epsilon
        )


def demonstrate_optimizers() -> None:
    section("28. Optimizer Concepts")

    parameter = 10.0
    gradient = 4.0

    momentum = MomentumOptimizer(
        learning_rate=0.1,
        beta=0.9,
    )

    adam = AdamScalar(
        learning_rate=0.1,
    )

    momentum_parameter = parameter
    adam_parameter = parameter

    for step in range(1, 6):
        momentum_parameter = momentum.step(
            momentum_parameter,
            gradient,
        )
        adam_parameter = adam.step(
            adam_parameter,
            gradient,
        )

        print(
            f"Step {step}: "
            f"momentum={momentum_parameter:.5f}, "
            f"adam={adam_parameter:.5f}"
        )

    print(
        """
Gradient descent is the basic optimization idea. Practical neural-network
training commonly uses improved optimizers.

Momentum accumulates a moving direction of recent gradients and can reduce
some forms of oscillation.

Adam maintains first and second moment estimates of gradients and adapts
the effective step for each parameter.

Optimizer choice affects convergence behavior, but it does not eliminate
the need for sound data preparation, architecture design, loss selection,
and evaluation.
"""
    )


# ============================================================================
# 29. DROPOUT CONCEPT
# ============================================================================

def dropout(
    activations: np.ndarray,
    probability: float,
    rng: np.random.Generator,
    training: bool = True,
) -> np.ndarray:
    """
    Educational inverted-dropout implementation.

    During training:
        randomly zero activations with probability p
        scale surviving activations by 1/(1-p)

    During evaluation:
        return activations unchanged.
    """
    activations = np.asarray(activations, dtype=float)

    if not 0.0 <= probability < 1.0:
        raise ValueError("Dropout probability must be in [0, 1).")

    if not training:
        return activations.copy()

    keep_probability = 1.0 - probability
    mask = rng.random(activations.shape) < keep_probability

    return activations * mask / keep_probability


def demonstrate_dropout() -> None:
    section("29. Dropout")

    rng = np.random.default_rng(5)
    activations = np.ones(10)

    training_output = dropout(
        activations,
        probability=0.3,
        rng=rng,
        training=True,
    )

    evaluation_output = dropout(
        activations,
        probability=0.3,
        rng=rng,
        training=False,
    )

    print("Original activations :", activations)
    print("Training output      :", training_output)
    print("Evaluation output    :", evaluation_output)

    print(
        """
Dropout randomly removes units during training. In inverted dropout, the
remaining activations are scaled during training so the expected activation
magnitude remains approximately consistent.

A critical distinction is training mode versus evaluation mode. Applying
dropout during evaluation changes predictions and is usually incorrect for
standard deterministic inference.
"""
    )


# ============================================================================
# 30. EARLY STOPPING CONCEPT
# ============================================================================

def early_stopping_epoch(
    validation_losses: Sequence[float],
    patience: int = 3,
) -> int | None:
    """
    Return the first epoch at which patience is exhausted.

    The returned value is an educational approximation rather than a complete
    production callback.
    """
    if patience < 1:
        raise ValueError("Patience must be at least 1.")

    best = float("inf")
    bad_epochs = 0

    for epoch, loss in enumerate(validation_losses, start=1):
        if loss < best:
            best = loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            return epoch

    return None


def demonstrate_early_stopping() -> None:
    section("30. Early Stopping")

    validation_losses = [
        0.80,
        0.60,
        0.50,
        0.45,
        0.43,
        0.44,
        0.46,
        0.48,
    ]

    stop_epoch = early_stopping_epoch(
        validation_losses,
        patience=2,
    )

    print("Validation losses:", validation_losses)
    print("Approximate stopping epoch:", stop_epoch)

    print(
        """
Early stopping monitors validation performance and stops training when the
model has failed to improve for a configured patience period.

The best checkpoint should generally be retained rather than automatically
assuming the final parameter state is best.
"""
    )


# ============================================================================
# 31. ACCURACY, PRECISION, RECALL
# ============================================================================

def binary_classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate basic binary classification metrics."""
    targets = np.asarray(targets, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    if targets.shape != predictions.shape:
        raise ValueError("Targets and predictions must have matching shapes.")

    true_positive = int(np.sum((targets == 1) & (predictions == 1)))
    true_negative = int(np.sum((targets == 0) & (predictions == 0)))
    false_positive = int(np.sum((targets == 0) & (predictions == 1)))
    false_negative = int(np.sum((targets == 1) & (predictions == 0)))

    total = len(targets)

    accuracy = (
        (true_positive + true_negative) / total
        if total
        else 0.0
    )

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = (
        true_positive / recall_denominator
        if recall_denominator
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "true_positive": float(true_positive),
        "true_negative": float(true_negative),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }


def demonstrate_metrics() -> None:
    section("31. Classification Metrics")

    targets = np.array([1, 1, 1, 0, 0, 0])
    predictions = np.array([1, 1, 0, 0, 1, 0])

    metrics = binary_classification_metrics(
        targets,
        predictions,
    )

    for name, value in metrics.items():
        print(f"{name:16s}: {value:.4f}" if isinstance(value, float) else value)

    print(
        """
Accuracy:
    Fraction of all predictions that are correct.

Precision:
    Of predicted positives, how many are actually positive?

Recall:
    Of actual positives, how many were detected?

These metrics answer different questions. Accuracy can be misleading on
highly imbalanced datasets.
"""
    )


# ============================================================================
# 32. CLASS IMBALANCE
# ============================================================================

def demonstrate_class_imbalance() -> None:
    section("32. Class Imbalance")

    targets = np.array([0] * 95 + [1] * 5)
    always_negative = np.zeros_like(targets)

    metrics = binary_classification_metrics(
        targets,
        always_negative,
    )

    print("Positive examples:", int(np.sum(targets)))
    print("Total examples:", len(targets))
    print("Accuracy of always-negative model:", metrics["accuracy"])
    print("Recall:", metrics["recall"])

    print(
        """
A classifier can achieve high accuracy by predicting the majority class
while completely failing to identify the minority class.

Potential responses include:
- Appropriate metrics.
- Class weighting.
- Resampling.
- Threshold adjustment.
- Better data collection.
- Task-specific loss design.

The correct approach depends on the cost of false positives and false
negatives.
"""
    )


# ============================================================================
# 33. DECISION THRESHOLDS
# ============================================================================

def demonstrate_thresholds() -> None:
    section("33. Decision Thresholds")

    probabilities = np.array([0.2, 0.45, 0.55, 0.8])

    for threshold in [0.3, 0.5, 0.7]:
        predictions = (probabilities >= threshold).astype(int)
        print(
            f"Threshold {threshold:.1f}: "
            f"{predictions.tolist()}"
        )

    print(
        """
The sigmoid output is a probability-like score. The threshold used to turn
that score into a class decision is separate from the model's learned
parameters.

Changing the threshold changes the balance between false positives and false
negatives.
"""
    )


# ============================================================================
# 34. PARAMETER COUNT
# ============================================================================

def parameter_count(layer_sizes: Sequence[int]) -> int:
    """Calculate weights plus biases for consecutive dense layers."""
    if len(layer_sizes) < 2:
        raise ValueError("At least two layer sizes are required.")

    total = 0

    for previous, current in zip(layer_sizes[:-1], layer_sizes[1:]):
        total += previous * current
        total += current

    return total


def demonstrate_parameter_count() -> None:
    section("34. Parameter Counting")

    architecture = [10, 20, 15, 3]

    count = parameter_count(architecture)

    print("Layer sizes:", architecture)
    print("Trainable parameter count:", count)

    print(
        """
For a dense connection between n input units and m output units:

    weights = n * m
    biases  = m
    total   = n*m + m

Large dense layers can produce substantial parameter counts, memory use,
compute requirements, and overfitting risk.
"""
    )


# ============================================================================
# 35. ACTIVATION AND OUTPUT DESIGN
# ============================================================================

def demonstrate_output_design() -> None:
    section("35. Choosing Output Activation and Loss")

    choices = [
        (
            "Regression",
            "Linear output",
            "MSE or another regression loss",
        ),
        (
            "Binary classification",
            "Sigmoid",
            "Binary cross-entropy",
        ),
        (
            "Multiclass, mutually exclusive",
            "Softmax",
            "Categorical cross-entropy",
        ),
        (
            "Multilabel classification",
            "Independent sigmoids",
            "Binary cross-entropy per label",
        ),
    ]

    for task, activation, loss in choices:
        print(f"{task:35s} | {activation:25s} | {loss}")


# ============================================================================
# 36. COMMON MISTAKES
# ============================================================================

def demonstrate_common_mistakes() -> None:
    section("36. Common Mistakes and Edge Cases")

    mistakes = [
        "Using mismatched input and weight dimensions.",
        "Using sigmoid or softmax without considering numerical stability.",
        "Taking log(0) in cross-entropy calculations.",
        "Computing preprocessing statistics using validation/test data.",
        "Evaluating a model using training data only.",
        "Initializing all weights identically and destroying symmetry breaking.",
        "Using a learning rate that is inappropriate for the problem.",
        "Forgetting to distinguish logits from probabilities.",
        "Applying dropout during ordinary inference.",
        "Ignoring class imbalance.",
        "Using accuracy as the only metric.",
        "Failing to switch between training and evaluation behavior.",
        "Allowing NaN or infinite values to propagate through training.",
        "Changing preprocessing between training and production inference.",
    ]

    for number, mistake in enumerate(mistakes, start=1):
        print(f"{number:2d}. {mistake}")

    print(
        """
Shape checking is particularly important in neural networks. A mathematically
valid matrix multiplication can still represent the wrong conceptual
operation, so dimensional correctness alone does not guarantee correctness.

Assertions and explicit validation should be used around data boundaries.
"""
    )


# ============================================================================
# 37. DEBUGGING CHECKS
# ============================================================================

def validate_network(
    network: NeuralNetwork,
    x: np.ndarray,
) -> None:
    """Perform simple numerical sanity checks."""
    probabilities = network.predict_proba(x)

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Network produced NaN or infinite probabilities.")

    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Sigmoid probabilities must lie in [0, 1].")


def demonstrate_debugging() -> None:
    section("37. Neural Network Debugging")

    network = NeuralNetwork(
        input_size=2,
        hidden_size=4,
        seed=1,
    )

    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    validate_network(network, x)

    print("Finite-output check: PASSED")
    print(
        "Probability-range check: PASSED "
        f"({network.predict_proba(x).reshape(-1)})"
    )

    print(
        """
Useful debugging checks:
- Verify tensor/array shapes.
- Verify finite values.
- Inspect activation ranges.
- Inspect loss before and after an update.
- Confirm that loss generally decreases on a tiny known dataset.
- Overfit a very small dataset as a diagnostic.
- Compare analytical gradients with numerical gradients.
- Confirm training and inference preprocessing are identical.
- Check labels and target encoding carefully.
"""
    )


# ============================================================================
# 38. NUMERICAL GRADIENT CHECK
# ============================================================================

def demonstrate_gradient_check() -> None:
    section("38. Numerical Gradient Checking")

    # Tiny scalar function corresponding to a parameterized computation.
    parameter = 1.7

    def loss_function(w: float) -> float:
        prediction = float(sigmoid(w * 2.0))
        target = 1.0
        return binary_cross_entropy(
            np.array([prediction]),
            np.array([target]),
        )

    numerical = numerical_derivative(
        loss_function,
        parameter,
        epsilon=1e-5,
    )

    prediction = float(sigmoid(parameter * 2.0))
    dloss_dprediction = -1.0 / prediction
    dprediction_dz = float(sigmoid_derivative_from_output(prediction))
    dz_dw = 2.0

    analytical = (
        dloss_dprediction
        * dprediction_dz
        * dz_dw
    )

    print("Numerical gradient :", numerical)
    print("Analytical gradient:", analytical)
    print("Absolute difference:", abs(numerical - analytical))

    print(
        """
Gradient checking is useful for debugging a custom implementation.

It compares an analytically derived gradient with a finite-difference
approximation. Numerical gradients are much slower and are not normally
used for full training. They are diagnostic tools.
"""
    )


# ============================================================================
# 39. COMPUTATIONAL GRAPH
# ============================================================================

def demonstrate_computational_graph() -> None:
    section("39. Computational Graph")

    x = 2.0
    w = 3.0
    b = 1.0

    multiplication = x * w
    addition = multiplication + b
    activation = float(sigmoid(addition))

    print("x:", x)
    print("w:", w)
    print("b:", b)
    print("x*w:", multiplication)
    print("x*w+b:", addition)
    print("sigmoid(x*w+b):", activation)

    print(
        """
The sequence can be viewed as a computational graph:

    x ----\
           multiply -> add bias -> activation -> output
    w ----/                  ^
                             |
                             b

Forward propagation evaluates the graph.

Backpropagation traverses the graph in reverse, applying local derivatives
and the chain rule to compute gradients.
"""
    )


# ============================================================================
# 40. DEEPER ARCHITECTURAL CONCEPTS
# ============================================================================

def explain_advanced_architecture() -> None:
    section("40. Advanced Neural-Network Architecture Concepts")

    concepts = {
        "Fully connected / dense": (
            "Each output neuron connects to every input activation from the "
            "previous layer."
        ),
        "Convolutional network": (
            "Uses local receptive fields and shared filters, especially "
            "effective for spatial or grid-like data."
        ),
        "Recurrent network": (
            "Processes sequences with state carried across time steps."
        ),
        "Residual connection": (
            "Provides a shortcut path around one or more transformations, "
            "helping information and gradients propagate through deep models."
        ),
        "Embedding": (
            "Maps discrete identifiers into learned dense vectors."
        ),
        "Attention": (
            "Computes content-dependent interactions among representations."
        ),
        "Normalization": (
            "Transforms intermediate activations according to statistics or "
            "learned transformations, potentially improving optimization."
        ),
    }

    for concept, description in concepts.items():
        print(f"{concept:25s}: {description}")

    print(
        """
The same fundamental components remain present across many architectures:
parameters transform representations, nonlinearities increase expressive
power, forward computation produces outputs, a loss measures an objective,
and gradients drive parameter updates.
"""
    )


# ============================================================================
# 41. REPRESENTATION LEARNING
# ============================================================================

def demonstrate_representation_learning() -> None:
    section("41. Representation Learning")

    print(
        """
Consider raw inputs:

    x -> hidden layer 1 -> hidden layer 2 -> output

The first layer may learn simple useful combinations of features.
Later layers can combine those representations into more abstract patterns.

This is one of the central ideas behind deep learning: useful intermediate
representations can be learned from data instead of manually specifying every
feature transformation.

The quality of these representations depends on:
- Data quantity and quality.
- Architecture.
- Objective function.
- Optimization.
- Regularization.
- Distribution alignment between training and deployment.
"""
    )


# ============================================================================
# 42. TRANSFER LEARNING CONCEPT
# ============================================================================

def explain_transfer_learning() -> None:
    section("42. Transfer Learning")

    print(
        """
Transfer learning uses knowledge learned from one task or dataset as a
starting point for another related task.

A common pattern is:

    pretrained representation -> task-specific output layer

Advantages:
- Less task-specific data may be required.
- Training can be faster.
- Useful representations may already exist.

Important considerations:
- Similarity between source and target domains.
- Whether to freeze or fine-tune earlier layers.
- Learning-rate selection during fine-tuning.
- Risk of negative transfer when the source knowledge is poorly matched.
"""
    )


# ============================================================================
# 43. PRODUCTION CONSIDERATIONS
# ============================================================================

def explain_production_considerations() -> None:
    section("43. Production Considerations")

    considerations = [
        ("Reproducibility", "Record seeds, data versions, preprocessing, architecture, and training configuration."),
        ("Data validation", "Reject malformed, missing, out-of-range, or unexpected inputs where appropriate."),
        ("Numerical stability", "Monitor NaN, infinity, overflow, underflow, and unstable losses."),
        ("Model versioning", "Track the exact parameters and preprocessing used by each deployed model."),
        ("Latency", "Measure end-to-end inference time, not only neural-network computation."),
        ("Memory", "Account for parameters, intermediate activations, batches, and runtime overhead."),
        ("Monitoring", "Monitor input distribution, prediction behavior, errors, and operational metrics."),
        ("Security", "Protect model artifacts, inference interfaces, credentials, and training data."),
        ("Privacy", "Avoid unnecessary collection and exposure of sensitive training or inference data."),
        ("Distribution shift", "Detect changes between the training distribution and production inputs."),
        ("Failure handling", "Define behavior for invalid inputs, unavailable dependencies, and low-confidence predictions."),
    ]

    for topic, explanation in considerations:
        print(f"{topic:20s}: {explanation}")

    print(
        """
A model that performs well in a controlled notebook can still fail in
production because the data pipeline, preprocessing, infrastructure,
monitoring, or operational assumptions differ from the training environment.
"""
    )


# ============================================================================
# 44. SECURITY-SPECIFIC CONSIDERATIONS
# ============================================================================

def explain_security() -> None:
    section("44. Security Considerations")

    print(
        """
Neural-network security is broader than protecting the mathematical model.

Relevant risks include:
- Malicious or corrupted training data.
- Unauthorized access to model artifacts.
- Insecure model-serving endpoints.
- Leakage of sensitive training information.
- Adversarial inputs designed to manipulate predictions.
- Dependency and supply-chain vulnerabilities.
- Inadequate access controls around datasets and checkpoints.
- Logging sensitive input data unnecessarily.

Defensive practices include:
- Authentication and authorization for model services.
- Secure artifact storage.
- Input validation.
- Dependency management.
- Least-privilege access.
- Encryption where appropriate.
- Careful logging and retention policies.
- Monitoring for anomalous behavior.
- Controlled model and dataset versioning.
"""
    )


# ============================================================================
# 45. PERFORMANCE CONSIDERATIONS
# ============================================================================

def explain_performance() -> None:
    section("45. Performance Considerations")

    print(
        """
The main computational operations in dense networks are matrix multiplications
and related tensor operations.

Performance is influenced by:
- Number of parameters.
- Batch size.
- Input dimensions.
- Number of layers.
- Hardware.
- Numerical precision.
- Memory bandwidth.
- Data-loading efficiency.

Vectorization is important because optimized numerical libraries can execute
matrix operations much more efficiently than Python-level loops.

For large models, additional methods can include:
- Mixed or reduced precision.
- Quantization.
- Pruning.
- Knowledge distillation.
- Efficient architectures.
- Hardware acceleration.
- Batch and pipeline optimization.

Performance optimization must preserve acceptable model quality and numerical
correctness.
"""
    )


# ============================================================================
# 46. EDGE CASES
# ============================================================================

def demonstrate_edge_cases() -> None:
    section("46. Important Edge Cases")

    print("Sigmoid of a very large positive value:", sigmoid(1000.0))
    print("Sigmoid of a very large negative value:", sigmoid(-1000.0))

    probabilities = np.array([1.0, 0.0, 0.999999999])
    targets = np.array([1.0, 0.0, 1.0])

    print(
        "Stable binary cross-entropy with boundary probabilities:",
        binary_cross_entropy(probabilities, targets),
    )

    constant_feature = np.array(
        [
            [5.0, 1.0],
            [5.0, 2.0],
            [5.0, 3.0],
        ]
    )

    scaled, mean, std = standardize_features(constant_feature)

    print("\nConstant-feature mean:", mean)
    print("Safe standard deviation:", std)
    print("Scaled constant feature:")
    print(scaled)

    print(
        """
Important edge cases include:
- Very large positive or negative logits.
- Probabilities exactly equal to zero or one.
- Constant input features.
- Empty datasets.
- Single-example batches.
- Final mini-batches smaller than the requested batch size.
- Shape mismatches.
- NaN and infinity.
- Extremely imbalanced classes.
- Thresholds that are unsuitable for the application's error costs.
"""
    )


# ============================================================================
# 47. COMPLETE SMALL TRAINING EXAMPLE
# ============================================================================

def generate_binary_dataset(
    n: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple synthetic binary dataset.

    Class 1 generally corresponds to x1 + x2 > 0.
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(
        0.0,
        1.0,
        size=(n, 2),
    )

    y = (x[:, 0] + x[:, 1] > 0.0).astype(float)

    return x, y


def demonstrate_complete_training_workflow() -> None:
    section("47. Complete Small Training Workflow")

    x, y = generate_binary_dataset(
        n=200,
        seed=10,
    )

    # Shuffle once and create train/test sets.
    rng = np.random.default_rng(10)
    indices = np.arange(len(x))
    rng.shuffle(indices)

    split = int(0.8 * len(x))

    train_indices = indices[:split]
    test_indices = indices[split:]

    x_train = x[train_indices]
    y_train = y[train_indices]

    x_test = x[test_indices]
    y_test = y[test_indices]

    # Learn scaling only from training data.
    x_train_scaled, train_mean, train_std = standardize_features(x_train)
    x_test_scaled, _, _ = standardize_features(
        x_test,
        mean=train_mean,
        std=train_std,
    )

    model = NeuralNetwork(
        input_size=2,
        hidden_size=8,
        seed=21,
    )

    model.train(
        x_train_scaled,
        y_train,
        learning_rate=0.1,
        epochs=1000,
        verbose_every=250,
    )

    probabilities = model.predict_proba(x_test_scaled).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)

    metrics = binary_classification_metrics(
        y_test.astype(int),
        predictions,
    )

    print("\nTest metrics:")
    for key, value in metrics.items():
        print(f"{key:16s}: {value:.4f}")

    print(
        """
This workflow illustrates a minimal end-to-end pattern:

1. Generate or collect data.
2. Split data into development and evaluation sets.
3. Fit preprocessing on training data.
4. Apply the same preprocessing to unseen data.
5. Initialize the model.
6. Train using the training set.
7. Generate predictions on unseen data.
8. Convert probabilities into decisions when required.
9. Evaluate using appropriate metrics.
"""
    )


# ============================================================================
# 48. CONCEPTUAL COMPARISONS
# ============================================================================

def explain_key_comparisons() -> None:
    section("48. Important Comparisons")

    comparisons = [
        (
            "Weight",
            "Scales the contribution of an input.",
            "Learned parameter.",
        ),
        (
            "Bias",
            "Shifts the pre-activation value.",
            "Learned parameter.",
        ),
        (
            "Activation",
            "Transforms a neuron's pre-activation.",
            "Usually nonlinear in hidden layers.",
        ),
        (
            "Loss",
            "Measures prediction error relative to the training objective.",
            "Optimized during training.",
        ),
        (
            "Gradient",
            "Indicates local sensitivity of loss to parameters.",
            "Used by optimizers.",
        ),
        (
            "Learning rate",
            "Controls update magnitude.",
            "Hyperparameter.",
        ),
        (
            "Epoch",
            "One complete dataset pass.",
            "Training-unit concept.",
        ),
        (
            "Batch",
            "Subset used for an update.",
            "Training-unit concept.",
        ),
        (
            "Logit",
            "Unnormalized model score.",
            "Often transformed by sigmoid or softmax.",
        ),
        (
            "Probability",
            "Normalized or bounded model output interpreted probabilistically.",
            "Used for decisions and evaluation.",
        ),
    ]

    print(
        f"{'Concept':15s} | {'Meaning':55s} | {'Role':30s}"
    )
    print("-" * 105)

    for name, meaning, role in comparisons:
        print(f"{name:15s} | {meaning:55s} | {role:30s}")


# ============================================================================
# 49. STUDY CHECKPOINTS
# ============================================================================

def study_checkpoints() -> None:
    section("49. Conceptual Checkpoints")

    questions = [
        "What is the mathematical equation for a neuron before activation?",
        "Why does a neuron need a bias?",
        "Why are nonlinear activation functions important?",
        "What is the difference between a parameter and a hyperparameter?",
        "What happens during forward propagation?",
        "What does a loss function measure?",
        "What does a gradient represent?",
        "How does the chain rule enable backpropagation?",
        "Why does gradient descent subtract the gradient?",
        "Why can one neuron not represent XOR?",
        "Why are ReLU-like activations common in hidden layers?",
        "Why are sigmoid outputs useful for binary classification?",
        "Why does softmax produce probabilities that sum to one?",
        "Why is numerical stability important in softmax and cross-entropy?",
        "Why should preprocessing statistics be learned from training data only?",
        "What is the difference between overfitting and underfitting?",
        "Why can high accuracy be misleading for imbalanced data?",
        "What is the difference between training and inference behavior?",
        "Why does initialization affect deep-network training?",
        "Why is gradient checking useful when implementing backpropagation?",
    ]

    for number, question in enumerate(questions, start=1):
        print(f"{number:2d}. {question}")


# ============================================================================
# 50. MAIN PROGRAM
# ============================================================================

def main() -> None:
    """
    Run all educational demonstrations in a deliberate fundamentals-to-
    advanced sequence.
    """
    np.set_printoptions(precision=4, suppress=True)

    random.seed(42)
    np.random.seed(42)

    explain_basic_terminology()
    demonstrate_single_neuron()
    demonstrate_bias()
    demonstrate_activation_functions()
    demonstrate_activated_neuron()
    demonstrate_dense_layer()
    demonstrate_architecture()
    demonstrate_forward_propagation()
    demonstrate_losses()
    demonstrate_derivatives()
    demonstrate_chain_rule()
    gradient_descent_demo()
    demonstrate_single_neuron_training()
    demonstrate_xor_limitation()
    demonstrate_xor_network()
    demonstrate_batch_processing()
    demonstrate_regression()
    demonstrate_multiclass_output()
    demonstrate_numerical_stability()
    demonstrate_initialization()
    demonstrate_gradient_pathologies()
    demonstrate_fit_concepts()
    demonstrate_regularization()
    demonstrate_feature_scaling()
    demonstrate_data_splits()
    demonstrate_mini_batches()
    demonstrate_learning_rates()
    demonstrate_optimizers()
    demonstrate_dropout()
    demonstrate_early_stopping()
    demonstrate_metrics()
    demonstrate_class_imbalance()
    demonstrate_thresholds()
    demonstrate_parameter_count()
    demonstrate_output_design()
    demonstrate_common_mistakes()
    demonstrate_debugging()
    demonstrate_gradient_check()
    demonstrate_computational_graph()
    explain_advanced_architecture()
    demonstrate_representation_learning()
    explain_transfer_learning()
    explain_production_considerations()
    explain_security()
    explain_performance()
    demonstrate_edge_cases()
    demonstrate_complete_training_workflow()
    explain_key_comparisons()
    study_checkpoints()

    section("End of Deep Learning Fundamentals Demonstration")
    print(
        "The script has demonstrated the mathematical and practical building "
        "blocks of neural networks from individual neurons through training "
        "and production considerations."
    )


if __name__ == "__main__":
    main()
