/*
 * OPTIMIZATION MATHEMATICS
 * ========================
 *
 * A self-contained JavaScript study implementation covering:
 *
 * - Objective functions and decision variables
 * - Local and global minima
 * - Derivatives and numerical differentiation
 * - Convexity and non-convexity
 * - Gradient descent
 * - Backtracking line search
 * - Multivariable optimization
 * - Hessian matrices
 * - Box constraints and projected gradient descent
 * - Lagrange multipliers
 * - Penalty methods
 * - KKT concepts
 * - Least-squares optimization
 * - Logistic regression
 * - Numerical stability
 * - Global-search heuristics
 *
 * Run with:
 *
 *     node optimization_mathematics.js
 *
 * No external npm packages are required.
 */

"use strict";

// ============================================================================
// 1. BASIC OBJECTIVE FUNCTIONS
// ============================================================================

function square(x) {
    return (x - 3) ** 2 + 2;
}

function doubleWell(x) {
    return x ** 4 - 4 * x ** 2 + 4;
}

// Numerical differentiation is useful for understanding calculus-based
// optimization, although analytical derivatives are usually preferable in
// performance-sensitive production code.
function numericalDerivative(fn, x, step = 1e-6) {
    if (step <= 0) {
        throw new Error("step must be positive");
    }

    return (fn(x + step) - fn(x - step)) / (2 * step);
}

function numericalSecondDerivative(fn, x, step = 1e-4) {
    if (step <= 0) {
        throw new Error("step must be positive");
    }

    return (
        fn(x + step)
        - 2 * fn(x)
        + fn(x - step)
    ) / (step ** 2);
}

function classifyOneDimensionalPoint(fn, x) {
    const second = numericalSecondDerivative(fn, x);

    if (second > 1e-5) {
        return "local minimum";
    }

    if (second < -1e-5) {
        return "local maximum";
    }

    return "inconclusive or degenerate";
}

// ============================================================================
// 2. VECTOR OPERATIONS
// ============================================================================

function assertSameLength(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vector dimensions do not match");
    }
}

function vectorAdd(a, b) {
    assertSameLength(a, b);
    return a.map((value, index) => value + b[index]);
}

function vectorSubtract(a, b) {
    assertSameLength(a, b);
    return a.map((value, index) => value - b[index]);
}

function scalarMultiply(scalar, vector) {
    return vector.map(value => scalar * value);
}

function dot(a, b) {
    assertSameLength(a, b);
    return a.reduce(
        (sum, value, index) => sum + value * b[index],
        0
    );
}

function vectorNorm(vector) {
    return Math.sqrt(dot(vector, vector));
}

function distance(a, b) {
    return vectorNorm(vectorSubtract(a, b));
}

// ============================================================================
// 3. NUMERICAL GRADIENT AND HESSIAN
// ============================================================================

function numericalGradient(fn, point, step = 1e-6) {
    const gradient = [];

    for (let i = 0; i < point.length; i++) {
        const plus = [...point];
        const minus = [...point];

        plus[i] += step;
        minus[i] -= step;

        gradient.push(
            (fn(plus) - fn(minus)) / (2 * step)
        );
    }

    return gradient;
}

function numericalHessian(fn, point, step = 1e-4) {
    const n = point.length;
    const hessian = Array.from(
        { length: n },
        () => Array(n).fill(0)
    );

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (i === j) {
                const plus = [...point];
                const minus = [...point];

                plus[i] += step;
                minus[i] -= step;

                hessian[i][j] = (
                    fn(plus)
                    - 2 * fn(point)
                    + fn(minus)
                ) / (step ** 2);
            } else {
                const pp = [...point];
                const pm = [...point];
                const mp = [...point];
                const mm = [...point];

                pp[i] += step;
                pp[j] += step;

                pm[i] += step;
                pm[j] -= step;

                mp[i] -= step;
                mp[j] += step;

                mm[i] -= step;
                mm[j] -= step;

                hessian[i][j] = (
                    fn(pp)
                    - fn(pm)
                    - fn(mp)
                    + fn(mm)
                ) / (4 * step ** 2);
            }
        }
    }

    return hessian;
}

// ============================================================================
// 4. GRADIENT DESCENT
// ============================================================================

function gradientDescent(
    fn,
    gradientFn,
    start,
    {
        learningRate = 0.05,
        iterations = 1000,
        tolerance = 1e-8
    } = {}
) {
    if (learningRate <= 0) {
        throw new Error("learningRate must be positive");
    }

    let point = [...start];
    const history = [];

    for (let iteration = 0; iteration < iterations; iteration++) {
        const gradient = gradientFn(point);
        const gradientSize = vectorNorm(gradient);

        history.push({
            iteration,
            point: [...point],
            value: fn(point),
            gradientNorm: gradientSize
        });

        if (gradientSize <= tolerance) {
            break;
        }

        const nextPoint = vectorSubtract(
            point,
            scalarMultiply(learningRate, gradient)
        );

        if (distance(nextPoint, point) <= tolerance) {
            point = nextPoint;
            break;
        }

        point = nextPoint;
    }

    return {
        solution: point,
        history
    };
}

// ============================================================================
// 5. QUADRATIC OBJECTIVE
// ============================================================================

class QuadraticFunction {
    /*
     * Represents:
     *
     * f(x) = 1/2 x^T Q x + c^T x + constant
     *
     * When Q is symmetric positive definite, the objective is strictly
     * convex and therefore has at most one minimizer.
     */

    constructor(Q, c, constant = 0) {
        this.Q = Q.map(row => [...row]);
        this.c = [...c];
        this.constant = constant;

        const n = Q.length;

        if (n === 0) {
            throw new Error("Q cannot be empty");
        }

        if (Q.some(row => row.length !== n)) {
            throw new Error("Q must be square");
        }

        if (c.length !== n) {
            throw new Error("c dimension must match Q");
        }
    }

    evaluate(x) {
        const Qx = this.Q.map(row => dot(row, x));

        return (
            0.5 * dot(x, Qx)
            + dot(this.c, x)
            + this.constant
        );
    }

    gradient(x) {
        const Qx = this.Q.map(row => dot(row, x));
        return vectorAdd(Qx, this.c);
    }

    hessian() {
        return this.Q.map(row => [...row]);
    }
}

// ============================================================================
// 6. BACKTRACKING LINE SEARCH
// ============================================================================

function backtrackingLineSearch(
    fn,
    gradientFn,
    point,
    direction,
    {
        initialStep = 1,
        reduction = 0.5,
        armijoConstant = 1e-4,
        minimumStep = 1e-12
    } = {}
) {
    if (!(reduction > 0 && reduction < 1)) {
        throw new Error("reduction must be between 0 and 1");
    }

    let step = initialStep;

    const currentValue = fn(point);
    const gradient = gradientFn(point);
    const directionalDerivative = dot(
        gradient,
        direction
    );

    while (step >= minimumStep) {
        const candidate = vectorAdd(
            point,
            scalarMultiply(step, direction)
        );

        const candidateValue = fn(candidate);

        // Armijo sufficient-decrease condition.
        if (
            candidateValue
            <= currentValue
            + armijoConstant
            * step
            * directionalDerivative
        ) {
            return step;
        }

        step *= reduction;
    }

    return minimumStep;
}

function gradientDescentWithLineSearch(
    fn,
    gradientFn,
    start,
    {
        iterations = 500,
        tolerance = 1e-8
    } = {}
) {
    let point = [...start];
    const history = [];

    for (let iteration = 0; iteration < iterations; iteration++) {
        const gradient = gradientFn(point);

        history.push({
            iteration,
            point: [...point],
            value: fn(point),
            gradientNorm: vectorNorm(gradient)
        });

        if (vectorNorm(gradient) <= tolerance) {
            break;
        }

        const direction = scalarMultiply(
            -1,
            gradient
        );

        const step = backtrackingLineSearch(
            fn,
            gradientFn,
            point,
            direction
        );

        const nextPoint = vectorAdd(
            point,
            scalarMultiply(step, direction)
        );

        if (distance(nextPoint, point) <= tolerance) {
            point = nextPoint;
            break;
        }

        point = nextPoint;
    }

    return {
        solution: point,
        history
    };
}

// ============================================================================
// 7. PROJECTION FOR BOX CONSTRAINTS
// ============================================================================

function projectToBox(point, lower, upper) {
    assertSameLength(point, lower);
    assertSameLength(point, upper);

    return point.map((value, index) => {
        if (lower[index] > upper[index]) {
            throw new Error("Lower bound exceeds upper bound");
        }

        return Math.max(
            lower[index],
            Math.min(value, upper[index])
        );
    });
}

function projectedGradientDescent(
    fn,
    gradientFn,
    start,
    lower,
    upper,
    {
        learningRate = 0.05,
        iterations = 1000,
        tolerance = 1e-8
    } = {}
) {
    let point = projectToBox(
        start,
        lower,
        upper
    );

    const history = [];

    for (let iteration = 0; iteration < iterations; iteration++) {
        const gradient = gradientFn(point);

        history.push({
            iteration,
            point: [...point],
            value: fn(point)
        });

        const candidate = vectorSubtract(
            point,
            scalarMultiply(learningRate, gradient)
        );

        const projected = projectToBox(
            candidate,
            lower,
            upper
        );

        if (distance(projected, point) <= tolerance) {
            point = projected;
            break;
        }

        point = projected;
    }

    return {
        solution: point,
        history
    };
}

// ============================================================================
// 8. LEAST-SQUARES REGRESSION
// ============================================================================

function solveTwoByTwo(A, b) {
    const determinant =
        A[0][0] * A[1][1]
        - A[0][1] * A[1][0];

    if (Math.abs(determinant) < 1e-12) {
        throw new Error("Matrix is singular or nearly singular");
    }

    return [
        (
            b[0] * A[1][1]
            - A[0][1] * b[1]
        ) / determinant,

        (
            A[0][0] * b[1]
            - b[0] * A[1][0]
        ) / determinant
    ];
}

function leastSquaresFit(xValues, yValues) {
    if (xValues.length !== yValues.length) {
        throw new Error("x and y must have equal lengths");
    }

    if (xValues.length < 2) {
        throw new Error("At least two observations are required");
    }

    let sumX = 0;
    let sumY = 0;
    let sumXX = 0;
    let sumXY = 0;

    for (let i = 0; i < xValues.length; i++) {
        sumX += xValues[i];
        sumY += yValues[i];
        sumXX += xValues[i] ** 2;
        sumXY += xValues[i] * yValues[i];
    }

    const n = xValues.length;

    const matrix = [
        [n, sumX],
        [sumX, sumXX]
    ];

    const vector = [
        sumY,
        sumXY
    ];

    return solveTwoByTwo(matrix, vector);
}

// ============================================================================
// 9. NUMERICALLY STABLE SIGMOID AND LOGISTIC REGRESSION
// ============================================================================

function sigmoid(value) {
    // Avoid direct exp(-value) for very large negative values.
    if (value >= 0) {
        const z = Math.exp(-value);
        return 1 / (1 + z);
    }

    const z = Math.exp(value);
    return z / (1 + z);
}

function logisticLossAndGradient(
    features,
    labels,
    weights,
    regularization = 0
) {
    if (features.length !== labels.length) {
        throw new Error("features and labels must match");
    }

    if (features.length === 0) {
        throw new Error("Training data is empty");
    }

    const gradient = Array(weights.length).fill(0);
    let loss = 0;

    for (let i = 0; i < features.length; i++) {
        const row = features[i];
        const label = labels[i];

        if (row.length !== weights.length) {
            throw new Error("Feature dimension mismatch");
        }

        if (label !== 0 && label !== 1) {
            throw new Error("Binary labels must be 0 or 1");
        }

        const score = dot(row, weights);

        // Stable softplus:
        // log(1 + exp(score)) = max(score,0) + log(1+exp(-|score|))
        const softplus =
            Math.max(score, 0)
            + Math.log1p(Math.exp(-Math.abs(score)));

        loss += softplus - label * score;

        const probability = sigmoid(score);
        const error = probability - label;

        for (let j = 0; j < weights.length; j++) {
            gradient[j] += error * row[j];
        }
    }

    loss /= features.length;

    for (let j = 0; j < gradient.length; j++) {
        gradient[j] /= features.length;
    }

    if (regularization > 0) {
        loss += (
            regularization
            * dot(weights, weights)
            / 2
        );

        for (let j = 0; j < weights.length; j++) {
            gradient[j] += regularization * weights[j];
        }
    }

    return {
        loss,
        gradient
    };
}

function trainLogisticRegression(
    features,
    labels,
    {
        learningRate = 0.5,
        iterations = 1000,
        regularization = 0.01
    } = {}
) {
    const weights = Array(features[0].length).fill(0);
    const losses = [];

    for (let iteration = 0; iteration < iterations; iteration++) {
        const result = logisticLossAndGradient(
            features,
            labels,
            weights,
            regularization
        );

        losses.push(result.loss);

        for (let j = 0; j < weights.length; j++) {
            weights[j] -= (
                learningRate
                * result.gradient[j]
            );
        }
    }

    return {
        weights,
        losses
    };
}

function logisticPredict(features, weights) {
    return sigmoid(dot(features, weights)) >= 0.5
        ? 1
        : 0;
}

// ============================================================================
// 10. LAGRANGE MULTIPLIER EXAMPLE
// ============================================================================

function lagrangeEqualityExample() {
    /*
     * Minimize:
     *
     *     x^2 + y^2
     *
     * subject to:
     *
     *     x + y = 10
     *
     * The Lagrangian is:
     *
     *     L = x^2 + y^2 + lambda(x+y-10)
     *
     * Stationarity gives:
     *
     *     2x + lambda = 0
     *     2y + lambda = 0
     *
     * Hence x = y = 5.
     */

    return {
        point: [5, 5],
        multiplier: -10
    };
}

// ============================================================================
// 11. PENALTY METHOD
// ============================================================================

function equalityPenalty(point, penaltyWeight) {
    const [x, y] = point;
    const violation = x + y - 10;

    return (
        x ** 2
        + y ** 2
        + penaltyWeight * violation ** 2
    );
}

function equalityPenaltyGradient(point, penaltyWeight) {
    const [x, y] = point;
    const violation = x + y - 10;

    return [
        2 * x + 2 * penaltyWeight * violation,
        2 * y + 2 * penaltyWeight * violation
    ];
}

// ============================================================================
// 12. ASYNCHRONOUS OPTIMIZATION DEMONSTRATION
// ============================================================================

function delayedValue(value, milliseconds) {
    return new Promise(resolve => {
        setTimeout(() => resolve(value), milliseconds);
    });
}

async function asynchronousObjectiveEvaluation(points) {
    /*
     * Real applications may evaluate expensive objective functions through
     * services, simulations, workers, files, or databases. Promise-based
     * concurrency can be useful when evaluations are independent.
     */

    const evaluations = await Promise.all(
        points.map(async point => {
            const value = await delayedValue(
                square(point),
                10
            );

            return {
                point,
                value
            };
        })
    );

    return evaluations;
}

// ============================================================================
// 13. GLOBAL SEARCH HEURISTIC
// ============================================================================

function simulatedAnnealing(
    fn,
    start,
    lower,
    upper,
    {
        initialTemperature = 10,
        coolingRate = 0.995,
        iterations = 5000,
        stepScale = 1,
        random = Math.random
    } = {}
) {
    let current = Math.max(
        lower,
        Math.min(start, upper)
    );

    let currentValue = fn(current);

    let best = current;
    let bestValue = currentValue;

    let temperature = initialTemperature;

    // Box-Muller normal random variable.
    function gaussian() {
        let u = 0;
        let v = 0;

        while (u === 0) {
            u = random();
        }

        while (v === 0) {
            v = random();
        }

        return Math.sqrt(-2 * Math.log(u))
            * Math.cos(2 * Math.PI * v);
    }

    for (let i = 0; i < iterations; i++) {
        let candidate = current
            + gaussian() * stepScale;

        candidate = Math.max(
            lower,
            Math.min(candidate, upper)
        );

        const candidateValue = fn(candidate);
        const difference =
            candidateValue - currentValue;

        const acceptanceProbability =
            Math.exp(
                -difference
                / Math.max(temperature, 1e-12)
            );

        if (
            difference <= 0
            || random() < acceptanceProbability
        ) {
            current = candidate;
            currentValue = candidateValue;
        }

        if (currentValue < bestValue) {
            best = current;
            bestValue = currentValue;
        }

        temperature *= coolingRate;
    }

    return {
        point: best,
        value: bestValue
    };
}

// ============================================================================
// 14. EXAMPLES
// ============================================================================

function printSection(title) {
    console.log("\n" + "=".repeat(78));
    console.log(title);
    console.log("=".repeat(78));
}

async function main() {
    console.log("OPTIMIZATION MATHEMATICS");
    console.log("Executable JavaScript study implementation");

    printSection("1. OBJECTIVE FUNCTION");

    console.log("f(0) =", square(0));
    console.log("f(3) =", square(3));
    console.log("Classification at x=3:",
        classifyOneDimensionalPoint(square, 3)
    );

    printSection("2. NON-CONVEX OBJECTIVE");

    for (const point of [
        -Math.sqrt(2),
        0,
        Math.sqrt(2)
    ]) {
        console.log(
            `x=${point.toFixed(5)}, ` +
            `f(x)=${doubleWell(point).toFixed(5)}, ` +
            `type=${classifyOneDimensionalPoint(doubleWell, point)}`
        );
    }

    printSection("3. GRADIENT DESCENT");

    const gradientResult = gradientDescent(
        square,
        x => [numericalDerivative(square, x[0])],
        [-10],
        {
            learningRate: 0.2,
            iterations: 200
        }
    );

    console.log("Solution:", gradientResult.solution);
    console.log("Iterations:",
        gradientResult.history.length
    );

    printSection("4. MULTIVARIABLE QUADRATIC");

    const quadratic = new QuadraticFunction(
        [
            [4, 1],
            [1, 2]
        ],
        [-8, -6]
    );

    const quadraticResult = gradientDescent(
        point => quadratic.evaluate(point),
        point => quadratic.gradient(point),
        [0, 0],
        {
            learningRate: 0.15,
            iterations: 1000
        }
    );

    console.log(
        "Gradient descent solution:",
        quadraticResult.solution
    );

    console.log(
        "Objective:",
        quadratic.evaluate(quadraticResult.solution)
    );

    console.log(
        "Hessian:",
        quadratic.hessian()
    );

    printSection("5. BACKTRACKING LINE SEARCH");

    const anisotropic = point =>
        20 * point[0] ** 2
        + point[1] ** 2
        + 3 * point[0];

    const anisotropicGradient = point => [
        40 * point[0] + 3,
        2 * point[1]
    ];

    const lineSearchResult =
        gradientDescentWithLineSearch(
            anisotropic,
            anisotropicGradient,
            [10, 10]
        );

    console.log(
        "Solution:",
        lineSearchResult.solution
    );

    console.log(
        "Objective:",
        anisotropic(lineSearchResult.solution)
    );

    printSection("6. BOX-CONSTRAINED OPTIMIZATION");

    const boundedFunction = point =>
        (point[0] - 5) ** 2
        + (point[1] + 3) ** 2;

    const boundedGradient = point => [
        2 * (point[0] - 5),
        2 * (point[1] + 3)
    ];

    const boundedResult =
        projectedGradientDescent(
            boundedFunction,
            boundedGradient,
            [0, 0],
            [0, -1],
            [4, 2],
            {
                learningRate: 0.1
            }
        );

    console.log(
        "Constrained solution:",
        boundedResult.solution
    );

    printSection("7. LAGRANGE MULTIPLIERS");

    console.log(lagrangeEqualityExample());

    printSection("8. PENALTY METHOD");

    for (const penalty of [1, 10, 100, 1000]) {
        const result = gradientDescent(
            point => equalityPenalty(
                point,
                penalty
            ),
            point => equalityPenaltyGradient(
                point,
                penalty
            ),
            [0, 0],
            {
                learningRate: 0.01,
                iterations: 5000
            }
        );

        const [x, y] = result.solution;

        console.log({
            penalty,
            solution: result.solution,
            constraintViolation: x + y - 10
        });
    }

    printSection("9. LEAST-SQUARES OPTIMIZATION");

    const regression = leastSquaresFit(
        [1, 2, 3, 4, 5],
        [2.2, 4.1, 6.2, 7.9, 10.1]
    );

    console.log(
        "Intercept:",
        regression[0]
    );

    console.log(
        "Slope:",
        regression[1]
    );

    printSection("10. LOGISTIC REGRESSION");

    const features = [
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5]
    ];

    const labels = [
        0, 0, 0, 1, 1, 1
    ];

    const logisticResult =
        trainLogisticRegression(
            features,
            labels,
            {
                learningRate: 0.5,
                iterations: 1000,
                regularization: 0.01
            }
        );

    console.log(
        "Weights:",
        logisticResult.weights
    );

    console.log(
        "Initial loss:",
        logisticResult.losses[0]
    );

    console.log(
        "Final loss:",
        logisticResult.losses.at(-1)
    );

    console.log(
        "Predictions:",
        features.map(row =>
            logisticPredict(
                row,
                logisticResult.weights
            )
        )
    );

    printSection("11. ASYNCHRONOUS OBJECTIVE EVALUATION");

    const asynchronousResults =
        await asynchronousObjectiveEvaluation(
            [0, 1, 2, 3]
        );

    console.log(
        asynchronousResults
    );

    printSection("12. SIMULATED ANNEALING");

    const annealingResult =
        simulatedAnnealing(
            doubleWell,
            0,
            -3,
            3,
            {
                initialTemperature: 10,
                coolingRate: 0.995,
                iterations: 5000,
                stepScale: 1
            }
        );

    console.log(
        annealingResult
    );

    printSection("13. NUMERICAL LESSONS");

    console.log(
        "Gradient descent depends strongly on step size."
    );

    console.log(
        "Convexity converts local optimality into global optimality."
    );

    console.log(
        "Non-convex objectives can have multiple local minima."
    );

    console.log(
        "Constraints require feasibility to be treated explicitly."
    );

    console.log(
        "Line search can adapt step sizes instead of relying on one fixed value."
    );

    console.log(
        "Regularization changes the optimization objective by adding a penalty."
    );

    console.log(
        "Floating-point arithmetic means numerical equality should normally "
        + "be checked using tolerances."
    );

    printSection("14. BASIC VALIDATION");

    if (Math.abs(gradientResult.solution[0] - 3) > 1e-4) {
        throw new Error("Gradient descent test failed");
    }

    if (Math.abs(regression[1] - 2) > 0.2) {
        throw new Error("Regression test failed");
    }

    console.log("Validation checks passed.");
}

main().catch(error => {
    console.error("Optimization study failed:", error);
    process.exitCode = 1;
});
