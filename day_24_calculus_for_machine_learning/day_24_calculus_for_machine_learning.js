/*
 * Calculus for Machine Learning
 *
 * This file complements the Python implementation by demonstrating
 * calculus through JavaScript's numerical, functional, object-oriented,
 * and application-oriented features.
 *
 * Covered:
 *   - Functions and composition
 *   - Limits and continuity
 *   - Numerical derivatives
 *   - Partial derivatives and gradients
 *   - Directional derivatives
 *   - Jacobians and Hessians
 *   - Chain rule
 *   - Taylor approximation
 *   - Loss functions
 *   - Gradient descent
 *   - Logistic regression
 *   - Automatic differentiation with dual numbers
 *   - Softmax and cross-entropy
 *   - A small neural-network forward pass
 *   - Numerical stability
 *   - Edge cases
 *   - Testing
 *
 * Run with:
 *   node calculus_for_ml.js
 */

"use strict";

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

function heading(title) {
    console.log("\n" + "=".repeat(78));
    console.log(title);
    console.log("=".repeat(78));
}

function subsection(title) {
    console.log("\n" + "-".repeat(78));
    console.log(title);
    console.log("-".repeat(78));
}

function assertClose(actual, expected, tolerance = 1e-9) {
    if (Math.abs(actual - expected) > tolerance) {
        throw new Error(
            `Assertion failed: expected ${expected}, received ${actual}`
        );
    }
}

function dotProduct(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have equal dimensions.");
    }

    return a.reduce((sum, value, index) => {
        return sum + value * b[index];
    }, 0);
}

function vectorNorm(vector) {
    return Math.sqrt(dotProduct(vector, vector));
}

function normalize(vector) {
    const norm = vectorNorm(vector);

    if (norm === 0) {
        throw new Error("Cannot normalize a zero vector.");
    }

    return vector.map(value => value / norm);
}


// ---------------------------------------------------------------------------
// 1. Functions
// ---------------------------------------------------------------------------

function demonstrateFunctions() {
    heading("1. Functions");

    subsection("1.1 Basic functions");

    // JavaScript functions are first-class values.
    const quadratic = x => x ** 2 + 2 * x + 1;

    console.log("f(x) = x² + 2x + 1");
    console.log(`f(3) = ${quadratic(3)}`);

    // A machine-learning model can itself be represented as a function.
    const linearModel = (x, weight, bias) => weight * x + bias;

    console.log(`Linear model output = ${linearModel(5, 2, 1)}`);

    subsection("1.2 Function composition");

    const addThree = x => x + 3;
    const square = x => x ** 2;

    // compose(outer, inner) returns outer(inner(x)).
    function compose(outer, inner) {
        return x => outer(inner(x));
    }

    const composed = compose(square, addThree);

    console.log(`(x + 3)² at x=2 = ${composed(2)}`);

    subsection("1.3 Functions as data");

    const functions = {
        linear: x => 2 * x + 1,
        quadratic: x => x ** 2,
        exponential: x => Math.exp(x),
        logarithm: x => Math.log(x),
        sigmoid: x => 1 / (1 + Math.exp(-x))
    };

    for (const [name, functionValue] of Object.entries(functions)) {
        console.log(`${name.padEnd(12)} f(2) = ${functionValue(2).toFixed(6)}`);
    }
}


// ---------------------------------------------------------------------------
// 2. Limits and continuity
// ---------------------------------------------------------------------------

function numericalLimit(func, point, epsilon = 1e-6) {
    return {
        left: func(point - epsilon),
        right: func(point + epsilon)
    };
}

function demonstrateLimitsAndContinuity() {
    heading("2. Limits and Continuity");

    const square = x => x ** 2;

    for (const epsilon of [1e-1, 1e-2, 1e-3, 1e-5]) {
        const result = numericalLimit(square, 2, epsilon);

        console.log(
            `epsilon=${epsilon.toExponential(0)}, ` +
            `left=${result.left.toFixed(10)}, ` +
            `right=${result.right.toFixed(10)}`
        );
    }

    subsection("Removable discontinuity");

    // Algebraically:
    // (x² - 1)/(x - 1) = x + 1 for x != 1.
    // The original expression remains undefined at x=1.
    const removable = x => (x ** 2 - 1) / (x - 1);

    for (const epsilon of [1e-2, 1e-4, 1e-6]) {
        console.log(
            `epsilon=${epsilon.toExponential(0)}, ` +
            `left=${removable(1 - epsilon).toFixed(8)}, ` +
            `right=${removable(1 + epsilon).toFixed(8)}`
        );
    }

    subsection("One-sided limits");

    const absoluteRatio = x => Math.abs(x) / x;

    console.log(`Left limit near 0: ${absoluteRatio(-1e-7)}`);
    console.log(`Right limit near 0: ${absoluteRatio(1e-7)}`);
    console.log("Different one-sided limits imply that the two-sided limit does not exist.");
}


// ---------------------------------------------------------------------------
// 3. Numerical derivatives
// ---------------------------------------------------------------------------

function forwardDifference(func, x, h = 1e-5) {
    if (h <= 0) {
        throw new Error("Step size must be positive.");
    }

    return (func(x + h) - func(x)) / h;
}

function backwardDifference(func, x, h = 1e-5) {
    if (h <= 0) {
        throw new Error("Step size must be positive.");
    }

    return (func(x) - func(x - h)) / h;
}

function centralDifference(func, x, h = 1e-5) {
    if (h <= 0) {
        throw new Error("Step size must be positive.");
    }

    return (func(x + h) - func(x - h)) / (2 * h);
}

function secondDerivative(func, x, h = 1e-4) {
    return (
        func(x + h) -
        2 * func(x) +
        func(x - h)
    ) / (h ** 2);
}

function demonstrateDerivatives() {
    heading("3. Derivatives");

    const functionValue = x => x ** 2;
    const x = 3;

    console.log(`f(x) = x²`);
    console.log(`Exact derivative = ${2 * x}`);
    console.log(
        `Central difference = ${centralDifference(functionValue, x).toFixed(8)}`
    );

    subsection("Finite-difference comparison");

    for (const h of [1e-1, 1e-2, 1e-3, 1e-4]) {
        const forward = forwardDifference(functionValue, x, h);
        const backward = backwardDifference(functionValue, x, h);
        const central = centralDifference(functionValue, x, h);

        console.log(
            `h=${h.toExponential(0)} | ` +
            `forward=${forward.toFixed(8)} | ` +
            `backward=${backward.toFixed(8)} | ` +
            `central=${central.toFixed(8)}`
        );
    }

    subsection("Second derivative");

    const fourthPower = xValue => xValue ** 4;

    console.log(
        `Numerical second derivative of x⁴ at x=2: ` +
        `${secondDerivative(fourthPower, 2).toFixed(6)}`
    );
}


// ---------------------------------------------------------------------------
// 4. Partial derivatives, gradients, directional derivatives
// ---------------------------------------------------------------------------

function partialDerivative(func, point, variableIndex, h = 1e-5) {
    if (variableIndex < 0 || variableIndex >= point.length) {
        throw new Error("Invalid variable index.");
    }

    const plus = [...point];
    const minus = [...point];

    plus[variableIndex] += h;
    minus[variableIndex] -= h;

    return (func(plus) - func(minus)) / (2 * h);
}

function gradient(func, point, h = 1e-5) {
    return point.map((_, index) => {
        return partialDerivative(func, point, index, h);
    });
}

function directionalDerivative(func, point, direction, h = 1e-5) {
    const unitDirection = normalize(direction);

    const plus = point.map(
        (value, index) => value + h * unitDirection[index]
    );

    const minus = point.map(
        (value, index) => value - h * unitDirection[index]
    );

    return (func(plus) - func(minus)) / (2 * h);
}

function demonstrateMultivariableCalculus() {
    heading("4. Multivariable Calculus");

    // f(x,y) = x² + 3xy + y²
    const functionValue = point => {
        const [x, y] = point;
        return x ** 2 + 3 * x * y + y ** 2;
    };

    const point = [2, 4];

    console.log(`Point = [${point.join(", ")}]`);

    const calculatedGradient = gradient(functionValue, point);

    console.log(
        "Numerical gradient =",
        calculatedGradient.map(value => value.toFixed(8))
    );

    console.log(
        "Analytical gradient = [10, 14]"
    );

    const direction = [1, 1];

    console.log(
        `Directional derivative = ` +
        `${directionalDerivative(functionValue, point, direction).toFixed(8)}`
    );
}


// ---------------------------------------------------------------------------
// 5. Jacobian and Hessian
// ---------------------------------------------------------------------------

function jacobian(functions, point, h = 1e-5) {
    return functions.map(func => gradient(func, point, h));
}

function hessian(func, point, h = 1e-4) {
    const dimension = point.length;
    const matrix = Array.from(
        { length: dimension },
        () => Array(dimension).fill(0)
    );

    for (let i = 0; i < dimension; i++) {
        for (let j = 0; j < dimension; j++) {
            const pp = [...point];
            const pm = [...point];
            const mp = [...point];
            const mm = [...point];

            pp[i] += h;
            pp[j] += h;

            pm[i] += h;
            pm[j] -= h;

            mp[i] -= h;
            mp[j] += h;

            mm[i] -= h;
            mm[j] -= h;

            matrix[i][j] = (
                func(pp) -
                func(pm) -
                func(mp) +
                func(mm)
            ) / (4 * h * h);
        }
    }

    return matrix;
}

function demonstrateJacobianAndHessian() {
    heading("5. Jacobians and Hessians");

    const point = [2, 3];

    const outputOne = p => p[0] ** 2 + p[1];
    const outputTwo = p => p[0] * p[1];

    const matrix = jacobian(
        [outputOne, outputTwo],
        point
    );

    console.log("Jacobian:");

    for (const row of matrix) {
        console.log(row.map(value => value.toFixed(6)));
    }

    const scalarFunction = p => {
        const [x, y] = p;
        return x ** 2 + 3 * y ** 2 + x * y;
    };

    const secondOrderMatrix = hessian(
        scalarFunction,
        point
    );

    console.log("Hessian:");

    for (const row of secondOrderMatrix) {
        console.log(row.map(value => value.toFixed(6)));
    }
}


// ---------------------------------------------------------------------------
// 6. Chain rule and Taylor approximation
// ---------------------------------------------------------------------------

function demonstrateChainRuleAndTaylor() {
    heading("6. Chain Rule and Taylor Approximation");

    const x = 1.2;

    // y = sin(x²)
    // dy/dx = cos(x²) * 2x
    const analytical = Math.cos(x ** 2) * 2 * x;
    const numerical = centralDifference(
        value => Math.sin(value ** 2),
        x
    );

    console.log(`Chain-rule derivative = ${analytical.toFixed(10)}`);
    console.log(`Numerical derivative = ${numerical.toFixed(10)}`);

    subsection("First-order Taylor approximation");

    const center = 0;
    const target = 0.5;

    const firstOrder =
        Math.exp(center) +
        Math.exp(center) * (target - center);

    const secondOrder =
        Math.exp(center) +
        Math.exp(center) * (target - center) +
        0.5 * Math.exp(center) * (target - center) ** 2;

    console.log(`Exact exp(0.5) = ${Math.exp(target).toFixed(10)}`);
    console.log(`First-order approximation = ${firstOrder.toFixed(10)}`);
    console.log(`Second-order approximation = ${secondOrder.toFixed(10)}`);
}


// ---------------------------------------------------------------------------
// 7. Loss functions
// ---------------------------------------------------------------------------

function meanSquaredError(predictions, targets) {
    if (predictions.length === 0) {
        throw new Error("Predictions cannot be empty.");
    }

    if (predictions.length !== targets.length) {
        throw new Error("Predictions and targets must have equal length.");
    }

    return predictions.reduce((sum, prediction, index) => {
        return sum + (prediction - targets[index]) ** 2;
    }, 0) / predictions.length;
}

function mseGradient(predictions, targets) {
    const n = predictions.length;

    return predictions.map((prediction, index) => {
        return 2 * (prediction - targets[index]) / n;
    });
}

function sigmoid(value) {
    // This piecewise implementation avoids unnecessary overflow.
    if (value >= 0) {
        const exponent = Math.exp(-value);
        return 1 / (1 + exponent);
    }

    const exponent = Math.exp(value);
    return exponent / (1 + exponent);
}

function binaryCrossEntropy(probabilities, targets) {
    if (probabilities.length !== targets.length) {
        throw new Error("Probability and target arrays must match.");
    }

    const epsilon = 1e-15;

    return probabilities.reduce((sum, probability, index) => {
        const target = targets[index];

        if (target !== 0 && target !== 1) {
            throw new Error("Binary targets must be 0 or 1.");
        }

        const clipped = Math.min(
            Math.max(probability, epsilon),
            1 - epsilon
        );

        return sum - (
            target * Math.log(clipped) +
            (1 - target) * Math.log(1 - clipped)
        );
    }, 0) / probabilities.length;
}

function demonstrateLossFunctions() {
    heading("7. Loss Functions");

    const predictions = [2.5, 3.5, 4.2];
    const targets = [3.0, 3.0, 5.0];

    console.log(
        `MSE = ${meanSquaredError(predictions, targets).toFixed(8)}`
    );

    console.log(
        "MSE gradient =",
        mseGradient(predictions, targets)
            .map(value => value.toFixed(8))
    );

    const probabilities = [0.9, 0.2, 0.8, 0.1];
    const binaryTargets = [1, 0, 1, 0];

    console.log(
        `Binary cross-entropy = ` +
        `${binaryCrossEntropy(probabilities, binaryTargets).toFixed(8)}`
    );
}


// ---------------------------------------------------------------------------
// 8. Gradient descent
// ---------------------------------------------------------------------------

function gradientDescent1D(
    func,
    derivative,
    initialValue,
    learningRate,
    iterations
) {
    if (learningRate <= 0) {
        throw new Error("Learning rate must be positive.");
    }

    let value = initialValue;
    const history = [value];

    for (let iteration = 0; iteration < iterations; iteration++) {
        value -= learningRate * derivative(value);
        history.push(value);
    }

    return { value, history };
}

function demonstrateGradientDescent() {
    heading("8. Gradient Descent");

    // f(x) = (x - 4)²
    // f'(x) = 2(x - 4)
    const func = x => (x - 4) ** 2;
    const derivative = x => 2 * (x - 4);

    const result = gradientDescent1D(
        func,
        derivative,
        0,
        0.1,
        30
    );

    console.log(`Initial x = ${result.history[0]}`);
    console.log(`Final x = ${result.value.toFixed(10)}`);
    console.log(`Final f(x) = ${func(result.value).toFixed(10)}`);

    subsection("Learning-rate comparison");

    for (const learningRate of [0.05, 0.2, 0.5, 1.0, 1.1]) {
        const run = gradientDescent1D(
            func,
            derivative,
            0,
            learningRate,
            20
        );

        console.log(
            `rate=${learningRate.toFixed(2)}, ` +
            `x=${run.value.toFixed(6)}, ` +
            `f(x)=${func(run.value).toFixed(6)}`
        );
    }
}


// ---------------------------------------------------------------------------
// 9. Logistic regression
// ---------------------------------------------------------------------------

class LogisticRegression {
    constructor(featureCount) {
        if (!Number.isInteger(featureCount) || featureCount <= 0) {
            throw new Error("Feature count must be a positive integer.");
        }

        this.weights = Array(featureCount).fill(0);
        this.bias = 0;
    }

    logit(features) {
        if (features.length !== this.weights.length) {
            throw new Error("Feature dimension mismatch.");
        }

        return dotProduct(this.weights, features) + this.bias;
    }

    probability(features) {
        return sigmoid(this.logit(features));
    }

    predict(features, threshold = 0.5) {
        if (!(threshold > 0 && threshold < 1)) {
            throw new Error("Threshold must be between 0 and 1.");
        }

        return this.probability(features) >= threshold ? 1 : 0;
    }
}

function logisticLossAndGradients(model, features, targets) {
    if (features.length === 0) {
        throw new Error("Training data cannot be empty.");
    }

    if (features.length !== targets.length) {
        throw new Error("Features and targets must have equal lengths.");
    }

    const gradients = Array(model.weights.length).fill(0);
    let biasGradient = 0;
    let totalLoss = 0;

    const epsilon = 1e-15;

    for (let rowIndex = 0; rowIndex < features.length; rowIndex++) {
        const row = features[rowIndex];
        const target = targets[rowIndex];

        if (row.length !== model.weights.length) {
            throw new Error("Inconsistent feature dimensions.");
        }

        if (target !== 0 && target !== 1) {
            throw new Error("Targets must be 0 or 1.");
        }

        const probability = model.probability(row);

        const clipped = Math.min(
            Math.max(probability, epsilon),
            1 - epsilon
        );

        totalLoss -= (
            target * Math.log(clipped) +
            (1 - target) * Math.log(1 - clipped)
        );

        // For sigmoid + binary cross-entropy:
        // dL/dz = prediction - target.
        const error = probability - target;

        for (let index = 0; index < gradients.length; index++) {
            gradients[index] += error * row[index];
        }

        biasGradient += error;
    }

    const count = features.length;

    return {
        loss: totalLoss / count,
        gradients: gradients.map(value => value / count),
        biasGradient: biasGradient / count
    };
}

function demonstrateLogisticRegression() {
    heading("9. Logistic Regression");

    const features = [
        [0.0, 0.0],
        [0.2, 0.1],
        [0.1, 0.3],
        [0.3, 0.2],
        [1.0, 1.0],
        [1.2, 0.9],
        [0.8, 1.1],
        [1.1, 1.3]
    ];

    const targets = [0, 0, 0, 0, 1, 1, 1, 1];

    const model = new LogisticRegression(2);
    const learningRate = 0.8;

    for (let epoch = 1; epoch <= 1000; epoch++) {
        const result = logisticLossAndGradients(
            model,
            features,
            targets
        );

        model.weights = model.weights.map(
            (weight, index) =>
                weight - learningRate * result.gradients[index]
        );

        model.bias -= learningRate * result.biasGradient;

        if ([1, 2, 10, 100, 500, 1000].includes(epoch)) {
            console.log(
                `epoch=${epoch}, ` +
                `loss=${result.loss.toFixed(8)}, ` +
                `weights=[${model.weights.map(v => v.toFixed(4))}], ` +
                `bias=${model.bias.toFixed(4)}`
            );
        }
    }

    subsection("Predictions");

    features.forEach((row, index) => {
        console.log(
            `x=${JSON.stringify(row)}, ` +
            `target=${targets[index]}, ` +
            `probability=${model.probability(row).toFixed(4)}, ` +
            `prediction=${model.predict(row)}`
        );
    });
}


// ---------------------------------------------------------------------------
// 10. Forward-mode automatic differentiation
// ---------------------------------------------------------------------------

class DualNumber {
    /*
     * A dual number stores both a value and its derivative.
     *
     * If:
     *     x = value + derivative * ε
     * and ε² = 0,
     *
     * arithmetic operations naturally propagate first derivatives.
     */
    constructor(value, derivative = 0) {
        this.value = value;
        this.derivative = derivative;
    }

    add(other) {
        const right = toDual(other);

        return new DualNumber(
            this.value + right.value,
            this.derivative + right.derivative
        );
    }

    subtract(other) {
        const right = toDual(other);

        return new DualNumber(
            this.value - right.value,
            this.derivative - right.derivative
        );
    }

    multiply(other) {
        const right = toDual(other);

        return new DualNumber(
            this.value * right.value,
            this.derivative * right.value +
            this.value * right.derivative
        );
    }

    divide(other) {
        const right = toDual(other);

        if (right.value === 0) {
            throw new Error("Division by zero.");
        }

        return new DualNumber(
            this.value / right.value,
            (
                this.derivative * right.value -
                this.value * right.derivative
            ) / (right.value ** 2)
        );
    }

    power(exponent) {
        const value = this.value ** exponent;

        const derivative =
            exponent === 0
                ? 0
                : exponent *
                  this.value ** (exponent - 1) *
                  this.derivative;

        return new DualNumber(value, derivative);
    }
}

function toDual(value) {
    return value instanceof DualNumber
        ? value
        : new DualNumber(value, 0);
}

function dualSin(value) {
    return new DualNumber(
        Math.sin(value.value),
        Math.cos(value.value) * value.derivative
    );
}

function dualExp(value) {
    const exponential = Math.exp(value.value);

    return new DualNumber(
        exponential,
        exponential * value.derivative
    );
}

function demonstrateAutomaticDifferentiation() {
    heading("10. Forward-Mode Automatic Differentiation");

    const x = new DualNumber(1.2, 1);

    // f(x) = sin(x²) + exp(x)
    const result = dualSin(
        x.power(2)
    ).add(
        dualExp(x)
    );

    const exact =
        2 * x.value * Math.cos(x.value ** 2) +
        Math.exp(x.value);

    console.log(`Function value = ${result.value.toFixed(10)}`);
    console.log(`AD derivative = ${result.derivative.toFixed(10)}`);
    console.log(`Exact derivative = ${exact.toFixed(10)}`);
}


// ---------------------------------------------------------------------------
// 11. Softmax
// ---------------------------------------------------------------------------

function softmax(logits) {
    if (logits.length === 0) {
        throw new Error("Logits cannot be empty.");
    }

    // Subtracting max(logits) prevents unnecessarily large exponentials.
    const maximum = Math.max(...logits);

    const exponentials = logits.map(
        logit => Math.exp(logit - maximum)
    );

    const denominator = exponentials.reduce(
        (sum, value) => sum + value,
        0
    );

    return exponentials.map(
        value => value / denominator
    );
}

function crossEntropyForClass(probabilities, targetIndex) {
    if (
        targetIndex < 0 ||
        targetIndex >= probabilities.length
    ) {
        throw new Error("Target index is outside the probability vector.");
    }

    const epsilon = 1e-15;

    const probability = Math.min(
        Math.max(probabilities[targetIndex], epsilon),
        1 - epsilon
    );

    return -Math.log(probability);
}

function demonstrateSoftmax() {
    heading("11. Softmax and Cross-Entropy");

    const logits = [2.0, 1.0, 0.1];
    const probabilities = softmax(logits);

    console.log(
        "Probabilities:",
        probabilities.map(value => value.toFixed(8))
    );

    console.log(
        `Probability sum = ${
            probabilities.reduce((a, b) => a + b, 0).toFixed(8)
        }`
    );

    console.log(
        `Loss for class 0 = ${
            crossEntropyForClass(probabilities, 0).toFixed(8)
        }`
    );
}


// ---------------------------------------------------------------------------
// 12. Small neural-network forward pass
// ---------------------------------------------------------------------------

class TinyNeuralNetwork {
    constructor() {
        this.hiddenWeights = [
            [0.5, -0.3],
            [0.8, 0.2]
        ];

        this.hiddenBiases = [0.1, -0.1];
        this.outputWeights = [0.7, -0.4];
        this.outputBias = 0.05;
    }

    forward(inputs) {
        if (inputs.length !== 2) {
            throw new Error("Expected two input features.");
        }

        const hidden = this.hiddenWeights.map(
            (weights, index) => {
                const z =
                    dotProduct(weights, inputs) +
                    this.hiddenBiases[index];

                return sigmoid(z);
            }
        );

        const outputLogit =
            dotProduct(this.outputWeights, hidden) +
            this.outputBias;

        return {
            hidden,
            probability: sigmoid(outputLogit)
        };
    }
}

function demonstrateNeuralNetwork() {
    heading("12. Tiny Neural Network");

    const network = new TinyNeuralNetwork();
    const result = network.forward([1, 2]);

    console.log(
        "Hidden activations:",
        result.hidden.map(value => value.toFixed(8))
    );

    console.log(
        `Output probability = ${result.probability.toFixed(8)}`
    );

    console.log(
        "Backpropagation would differentiate the output loss through "
        + "the output layer, activation functions, and hidden layer."
    );
}


// ---------------------------------------------------------------------------
// 13. Browser-oriented example
// ---------------------------------------------------------------------------

function demonstrateBrowserIntegration() {
    heading("13. Browser Integration Pattern");

    /*
     * JavaScript is especially useful when calculus calculations are connected
     * to interactive interfaces. The following function is safe to define in
     * Node.js because it does not assume that document exists.
     */
    function attachGradientDemo(element) {
        if (
            typeof document === "undefined" ||
            !element
        ) {
            return false;
        }

        const display = document.createElement("div");

        display.textContent =
            "For f(x)=x², the gradient is 2x.";

        element.appendChild(display);

        return true;
    }

    console.log(
        `Browser DOM available: ${typeof document !== "undefined"}`
    );

    console.log(
        "A browser application can connect numerical calculus functions "
        + "to sliders, charts, forms, and interactive model visualizations."
    );

    // Keep the function referenced so static analysis sees the intended API.
    void attachGradientDemo;
}


// ---------------------------------------------------------------------------
// 14. Numerical stability and edge cases
// ---------------------------------------------------------------------------

function demonstrateEdgeCases() {
    heading("14. Numerical Stability and Edge Cases");

    subsection("Sigmoid saturation");

    for (const value of [-1000, -50, 0, 50, 1000]) {
        console.log(
            `sigmoid(${value}) = ${sigmoid(value)}`
        );
    }

    subsection("Softmax with large logits");

    const largeLogits = [1000, 1001, 999];

    console.log(
        "Stable softmax:",
        softmax(largeLogits)
            .map(value => value.toFixed(8))
    );

    subsection("Invalid direction");

    try {
        normalize([0, 0]);
    } catch (error) {
        console.log(`Caught expected error: ${error.message}`);
    }

    subsection("Invalid learning rate");

    try {
        gradientDescent1D(
            x => x ** 2,
            x => 2 * x,
            1,
            -0.1,
            10
        );
    } catch (error) {
        console.log(`Caught expected error: ${error.message}`);
    }
}


// ---------------------------------------------------------------------------
// 15. Tests
// ---------------------------------------------------------------------------

function runTests() {
    heading("15. Tests");

    assertClose(
        (x => x ** 2)(3),
        9
    );

    assertClose(
        centralDifference(x => x ** 3, 2, 1e-6),
        12,
        1e-5
    );

    const calculatedGradient = gradient(
        point => point[0] ** 2 + 3 * point[1] ** 2,
        [2, 4]
    );

    assertClose(calculatedGradient[0], 4, 1e-5);
    assertClose(calculatedGradient[1], 24, 1e-5);

    assertClose(sigmoid(0), 0.5);

    const probabilities = softmax([1, 2, 3]);

    assertClose(
        probabilities.reduce((a, b) => a + b, 0),
        1,
        1e-12
    );

    assertClose(
        meanSquaredError([1, 2], [1, 4]),
        2
    );

    const dualInput = new DualNumber(2, 1);
    const dualOutput = dualInput.power(3);

    assertClose(dualOutput.value, 8);
    assertClose(dualOutput.derivative, 12);

    console.log("All tests passed.");
}


// ---------------------------------------------------------------------------
// Main execution
// ---------------------------------------------------------------------------

function main() {
    demonstrateFunctions();
    demonstrateLimitsAndContinuity();
    demonstrateDerivatives();
    demonstrateMultivariableCalculus();
    demonstrateJacobianAndHessian();
    demonstrateChainRuleAndTaylor();
    demonstrateLossFunctions();
    demonstrateGradientDescent();
    demonstrateLogisticRegression();
    demonstrateAutomaticDifferentiation();
    demonstrateSoftmax();
    demonstrateNeuralNetwork();
    demonstrateBrowserIntegration();
    demonstrateEdgeCases();
    runTests();
}

main();
