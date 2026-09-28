/*
MATHEMATICAL REVISION
Probability, Statistics, Linear Algebra, Calculus, and Optimization

This file is self-contained and executable with a modern JavaScript runtime
such as Node.js. It progresses from elementary numerical ideas to integrated
computational examples.
*/

"use strict";

// ============================================================================
// 1. BASIC NUMERICAL FOUNDATIONS
// ============================================================================

function section(title) {
    console.log("\n" + "=".repeat(78));
    console.log(title);
    console.log("=".repeat(78));
}

function subsection(title) {
    console.log("\n" + "-".repeat(60));
    console.log(title);
    console.log("-".repeat(60));
}

section("MATHEMATICAL REVISION");

console.log(`
Topics:
1. Mathematical foundations
2. Probability
3. Statistics
4. Linear algebra
5. Calculus
6. Optimization
7. Numerical methods
8. Integrated applications
`);

subsection("1. Arithmetic and numerical precision");

const a = 12;
const b = 5;

console.log("Addition:", a + b);
console.log("Subtraction:", a - b);
console.log("Multiplication:", a * b);
console.log("Division:", a / b);
console.log("Remainder:", a % b);
console.log("Power:", a ** 2);
console.log("Square root:", Math.sqrt(a));
console.log("Natural logarithm:", Math.log(a));
console.log("Logarithm base 10:", Math.log10(a));
console.log("Exponential:", Math.exp(2));

// JavaScript Number uses IEEE-754 double-precision floating point.
const floatingResult = 0.1 + 0.2;
console.log("0.1 + 0.2 =", floatingResult);
console.log(
    "Approximately equal to 0.3:",
    Math.abs(floatingResult - 0.3) < Number.EPSILON * 2
);


// ============================================================================
// 2. FUNCTIONS
// ============================================================================

subsection("2. Functions");

function linearFunction(x, slope, intercept) {
    return slope * x + intercept;
}

function quadratic(x, a, b, c) {
    return a * x * x + b * x + c;
}

console.log("2(3)+1 =", linearFunction(3, 2, 1));
console.log("q(2) for x²-3x+2 =", quadratic(2, 1, -3, 2));


// ============================================================================
// 3. PROBABILITY
// ============================================================================

section("PROBABILITY");

subsection("3.1 Classical probability");

function eventProbability(eventCount, totalCount) {
    if (totalCount <= 0) {
        throw new Error("Total count must be positive.");
    }

    if (eventCount < 0 || eventCount > totalCount) {
        throw new Error("Invalid event count.");
    }

    return eventCount / totalCount;
}

console.log(
    "Probability of an even die roll:",
    eventProbability(3, 6)
);


subsection("3.2 Factorials, combinations, and permutations");

function factorial(n) {
    if (!Number.isInteger(n) || n < 0) {
        throw new Error("Factorial requires a non-negative integer.");
    }

    let result = 1;

    for (let i = 2; i <= n; i++) {
        result *= i;
    }

    return result;
}

function combination(n, r) {
    if (
        !Number.isInteger(n) ||
        !Number.isInteger(r) ||
        n < 0 ||
        r < 0 ||
        r > n
    ) {
        throw new Error("Invalid combination parameters.");
    }

    // Use the smaller side of the binomial coefficient to reduce work.
    r = Math.min(r, n - r);

    let result = 1;

    for (let i = 1; i <= r; i++) {
        result = result * (n - r + i) / i;
    }

    return result;
}

function permutation(n, r) {
    if (
        !Number.isInteger(n) ||
        !Number.isInteger(r) ||
        n < 0 ||
        r < 0 ||
        r > n
    ) {
        throw new Error("Invalid permutation parameters.");
    }

    let result = 1;

    for (let i = 0; i < r; i++) {
        result *= n - i;
    }

    return result;
}

console.log("5! =", factorial(5));
console.log("5C2 =", combination(5, 2));
console.log("5P2 =", permutation(5, 2));


subsection("3.3 Binomial distribution");

function binomialProbability(n, k, p) {
    if (
        !Number.isInteger(n) ||
        !Number.isInteger(k) ||
        n < 0 ||
        k < 0 ||
        k > n
    ) {
        throw new Error("Invalid n or k.");
    }

    if (p < 0 || p > 1) {
        throw new Error("p must lie between 0 and 1.");
    }

    return combination(n, k) *
        Math.pow(p, k) *
        Math.pow(1 - p, n - k);
}

console.log(
    "P(X=3) for Binomial(10, 0.4):",
    binomialProbability(10, 3, 0.4)
);

console.log("Binomial mean:", 10 * 0.4);
console.log("Binomial variance:", 10 * 0.4 * 0.6);


subsection("3.4 Bayes' theorem");

const priorDisease = 0.01;
const positiveGivenDisease = 0.95;
const positiveGivenNoDisease = 0.05;
const priorNoDisease = 1 - priorDisease;

const positiveProbability =
    positiveGivenDisease * priorDisease +
    positiveGivenNoDisease * priorNoDisease;

const diseaseGivenPositive =
    positiveGivenDisease * priorDisease / positiveProbability;

console.log("P(positive) =", positiveProbability);
console.log("P(disease | positive) =", diseaseGivenPositive);


subsection("3.5 Expected value");

function expectedValue(values, probabilities) {
    if (values.length !== probabilities.length) {
        throw new Error("Values and probabilities must have equal length.");
    }

    const probabilitySum =
        probabilities.reduce((sum, p) => sum + p, 0);

    if (Math.abs(probabilitySum - 1) > 1e-12) {
        throw new Error("Probabilities must sum to 1.");
    }

    return values.reduce(
        (total, value, index) =>
            total + value * probabilities[index],
        0
    );
}

const dieValues = [1, 2, 3, 4, 5, 6];
const dieProbabilities = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];

console.log(
    "Expected fair-die value:",
    expectedValue(dieValues, dieProbabilities)
);


// ============================================================================
// 4. RANDOMNESS AND MONTE CARLO
// ============================================================================

section("MONTE CARLO SIMULATION");

function monteCarloPi(samples, seed = 12345) {
    if (samples <= 0) {
        throw new Error("Samples must be positive.");
    }

    // A simple deterministic linear-congruential generator is used so that
    // the educational example can be reproducible without external packages.
    let state = seed >>> 0;

    function random() {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    }

    let inside = 0;

    for (let i = 0; i < samples; i++) {
        const x = 2 * random() - 1;
        const y = 2 * random() - 1;

        if (x * x + y * y <= 1) {
            inside++;
        }
    }

    return 4 * inside / samples;
}

console.log("Estimated pi:", monteCarloPi(100000));


// ============================================================================
// 5. STATISTICS
// ============================================================================

section("STATISTICS");

const sample = [12, 15, 15, 18, 20, 22, 22, 24, 30];

subsection("5.1 Descriptive statistics");

function mean(values) {
    if (values.length === 0) {
        throw new Error("Cannot calculate mean of empty data.");
    }

    return values.reduce((sum, value) => sum + value, 0) /
        values.length;
}

function median(values) {
    if (values.length === 0) {
        throw new Error("Cannot calculate median of empty data.");
    }

    const sorted = [...values].sort((x, y) => x - y);
    const middle = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 0) {
        return (sorted[middle - 1] + sorted[middle]) / 2;
    }

    return sorted[middle];
}

function variance(values, sampleVariance = false) {
    if (values.length < (sampleVariance ? 2 : 1)) {
        throw new Error("Not enough observations.");
    }

    const average = mean(values);

    const squaredDeviationSum = values.reduce(
        (sum, value) => sum + (value - average) ** 2,
        0
    );

    return squaredDeviationSum /
        (values.length - (sampleVariance ? 1 : 0));
}

function standardDeviation(values, sampleVariance = false) {
    return Math.sqrt(variance(values, sampleVariance));
}

console.log("Mean:", mean(sample));
console.log("Median:", median(sample));
console.log("Population variance:", variance(sample));
console.log("Population standard deviation:", standardDeviation(sample));
console.log("Sample variance:", variance(sample, true));
console.log("Sample standard deviation:", standardDeviation(sample, true));


function percentile(values, percentage) {
    if (values.length === 0) {
        throw new Error("Data cannot be empty.");
    }

    if (percentage < 0 || percentage > 100) {
        throw new Error("Percentage must be between 0 and 100.");
    }

    const sorted = [...values].sort((x, y) => x - y);
    const position = (sorted.length - 1) * percentage / 100;

    const lower = Math.floor(position);
    const upper = Math.ceil(position);

    if (lower === upper) {
        return sorted[lower];
    }

    const fraction = position - lower;

    return sorted[lower] +
        fraction * (sorted[upper] - sorted[lower]);
}

console.log("25th percentile:", percentile(sample, 25));
console.log("75th percentile:", percentile(sample, 75));


subsection("5.2 Covariance and correlation");

function covariance(xValues, yValues) {
    if (
        xValues.length !== yValues.length ||
        xValues.length < 2
    ) {
        throw new Error("Samples must have equal length and at least 2 values.");
    }

    const xMean = mean(xValues);
    const yMean = mean(yValues);

    const numerator = xValues.reduce(
        (sum, x, index) =>
            sum + (x - xMean) * (yValues[index] - yMean),
        0
    );

    return numerator / (xValues.length - 1);
}

function correlation(xValues, yValues) {
    const xStandardDeviation = standardDeviation(xValues, true);
    const yStandardDeviation = standardDeviation(yValues, true);

    if (
        xStandardDeviation === 0 ||
        yStandardDeviation === 0
    ) {
        throw new Error("Correlation undefined for zero variance.");
    }

    return covariance(xValues, yValues) /
        (xStandardDeviation * yStandardDeviation);
}

const studyHours = [1, 2, 3, 4, 5, 6];
const examScores = [45, 50, 58, 65, 73, 82];

console.log("Covariance:", covariance(studyHours, examScores));
console.log("Correlation:", correlation(studyHours, examScores));


// ============================================================================
// 6. LINEAR REGRESSION
// ============================================================================

subsection("6. Linear regression");

class LinearRegressionModel {
    constructor(slope, intercept) {
        this.slope = slope;
        this.intercept = intercept;
    }

    predict(x) {
        return this.intercept + this.slope * x;
    }
}

function fitLinearRegression(xValues, yValues) {
    if (
        xValues.length !== yValues.length ||
        xValues.length < 2
    ) {
        throw new Error("Need equal-length data with at least two points.");
    }

    const xMean = mean(xValues);
    const yMean = mean(yValues);

    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < xValues.length; i++) {
        numerator +=
            (xValues[i] - xMean) *
            (yValues[i] - yMean);

        denominator +=
            (xValues[i] - xMean) ** 2;
    }

    if (denominator === 0) {
        throw new Error("Slope is undefined.");
    }

    const slope = numerator / denominator;
    const intercept = yMean - slope * xMean;

    return new LinearRegressionModel(slope, intercept);
}

const regression = fitLinearRegression(studyHours, examScores);

console.log("Slope:", regression.slope);
console.log("Intercept:", regression.intercept);
console.log("Prediction for 7 hours:", regression.predict(7));


// ============================================================================
// 7. LINEAR ALGEBRA
// ============================================================================

section("LINEAR ALGEBRA");

subsection("7.1 Vectors");

function vectorAdd(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have equal dimensions.");
    }

    return a.map((value, index) => value + b[index]);
}

function vectorSubtract(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have equal dimensions.");
    }

    return a.map((value, index) => value - b[index]);
}

function scalarMultiply(scalar, vector) {
    return vector.map(value => scalar * value);
}

function dotProduct(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have equal dimensions.");
    }

    return a.reduce(
        (sum, value, index) => sum + value * b[index],
        0
    );
}

function vectorNorm(vector) {
    return Math.sqrt(dotProduct(vector, vector));
}

function cosineSimilarity(a, b) {
    const denominator = vectorNorm(a) * vectorNorm(b);

    if (denominator === 0) {
        throw new Error("Cosine similarity is undefined for zero vectors.");
    }

    return dotProduct(a, b) / denominator;
}

const vectorA = [1, 2, 3];
const vectorB = [4, 5, 6];

console.log("A+B:", vectorAdd(vectorA, vectorB));
console.log("A-B:", vectorSubtract(vectorA, vectorB));
console.log("2A:", scalarMultiply(2, vectorA));
console.log("A·B:", dotProduct(vectorA, vectorB));
console.log("||A||:", vectorNorm(vectorA));
console.log("Cosine similarity:", cosineSimilarity(vectorA, vectorB));


subsection("7.2 Matrices");

function validateMatrix(matrix) {
    if (
        !Array.isArray(matrix) ||
        matrix.length === 0 ||
        matrix[0].length === 0
    ) {
        throw new Error("Matrix cannot be empty.");
    }

    const columns = matrix[0].length;

    for (const row of matrix) {
        if (row.length !== columns) {
            throw new Error("Matrix rows must have equal length.");
        }
    }

    return {
        rows: matrix.length,
        columns
    };
}

function matrixAdd(A, B) {
    const dimensionsA = validateMatrix(A);
    const dimensionsB = validateMatrix(B);

    if (
        dimensionsA.rows !== dimensionsB.rows ||
        dimensionsA.columns !== dimensionsB.columns
    ) {
        throw new Error("Matrices must have equal dimensions.");
    }

    return A.map((row, i) =>
        row.map((value, j) => value + B[i][j])
    );
}

function matrixMultiply(A, B) {
    const dimensionsA = validateMatrix(A);
    const dimensionsB = validateMatrix(B);

    if (dimensionsA.columns !== dimensionsB.rows) {
        throw new Error("Matrix dimensions are incompatible.");
    }

    const result = Array.from(
        { length: dimensionsA.rows },
        () => Array(dimensionsB.columns).fill(0)
    );

    for (let i = 0; i < dimensionsA.rows; i++) {
        for (let j = 0; j < dimensionsB.columns; j++) {
            for (let k = 0; k < dimensionsA.columns; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

function transpose(matrix) {
    const { rows, columns } = validateMatrix(matrix);

    return Array.from(
        { length: columns },
        (_, column) =>
            Array.from(
                { length: rows },
                (_, row) => matrix[row][column]
            )
    );
}

const A = [
    [1, 2],
    [3, 4]
];

const B = [
    [5, 6],
    [7, 8]
];

console.log("A+B:", matrixAdd(A, B));
console.log("AB:", matrixMultiply(A, B));
console.log("Aᵀ:", transpose(A));


subsection("7.3 Determinant");

function determinant(matrix) {
    const { rows, columns } = validateMatrix(matrix);

    if (rows !== columns) {
        throw new Error("Determinant requires a square matrix.");
    }

    if (rows === 1) {
        return matrix[0][0];
    }

    if (rows === 2) {
        return (
            matrix[0][0] * matrix[1][1] -
            matrix[0][1] * matrix[1][0]
        );
    }

    let result = 0;

    for (let column = 0; column < columns; column++) {
        const minor = matrix
            .slice(1)
            .map(row =>
                row.filter((_, index) => index !== column)
            );

        const sign = column % 2 === 0 ? 1 : -1;

        result +=
            sign *
            matrix[0][column] *
            determinant(minor);
    }

    return result;
}

console.log("det(A):", determinant(A));


// ============================================================================
// 8. SOLVING LINEAR SYSTEMS
// ============================================================================

subsection("8. Solving Ax=b with Gaussian elimination");

function solveLinearSystem(matrix, constants, tolerance = 1e-12) {
    const { rows, columns } = validateMatrix(matrix);

    if (rows !== columns) {
        throw new Error("Coefficient matrix must be square.");
    }

    if (constants.length !== rows) {
        throw new Error("Invalid constants vector.");
    }

    const augmented = matrix.map((row, index) => [
        ...row.map(Number),
        Number(constants[index])
    ]);

    for (let pivot = 0; pivot < rows; pivot++) {
        let pivotRow = pivot;

        for (let row = pivot + 1; row < rows; row++) {
            if (
                Math.abs(augmented[row][pivot]) >
                Math.abs(augmented[pivotRow][pivot])
            ) {
                pivotRow = row;
            }
        }

        if (Math.abs(augmented[pivotRow][pivot]) < tolerance) {
            throw new Error("Matrix is singular or nearly singular.");
        }

        [augmented[pivot], augmented[pivotRow]] =
            [augmented[pivotRow], augmented[pivot]];

        const pivotValue = augmented[pivot][pivot];

        for (let column = pivot; column <= rows; column++) {
            augmented[pivot][column] /= pivotValue;
        }

        for (let row = 0; row < rows; row++) {
            if (row === pivot) {
                continue;
            }

            const factor = augmented[row][pivot];

            for (let column = pivot; column <= rows; column++) {
                augmented[row][column] -=
                    factor * augmented[pivot][column];
            }
        }
    }

    return augmented.map(row => row[rows]);
}

console.log(
    "Solution:",
    solveLinearSystem(
        [
            [2, 1],
            [1, -1]
        ],
        [7, 1]
    )
);


// ============================================================================
// 9. CALCULUS
// ============================================================================

section("CALCULUS");

subsection("9.1 Numerical limits");

function numericalLimit(func, point, direction = 1, steps = 8) {
    if (direction !== 1 && direction !== -1) {
        throw new Error("Direction must be +1 or -1.");
    }

    let result;

    for (let exponent = 1; exponent <= steps; exponent++) {
        const h = 10 ** (-exponent);
        result = func(point + direction * h);
    }

    return result;
}

console.log(
    "lim(x→1)x²:",
    numericalLimit(x => x * x, 1)
);


subsection("9.2 Numerical differentiation");

function forwardDifference(func, x, h = 1e-5) {
    if (h <= 0) {
        throw new Error("h must be positive.");
    }

    return (func(x + h) - func(x)) / h;
}

function centralDifference(func, x, h = 1e-5) {
    if (h <= 0) {
        throw new Error("h must be positive.");
    }

    return (func(x + h) - func(x - h)) / (2 * h);
}

const calculusFunction = x => x ** 3 - 2 * x + 1;

console.log(
    "Forward derivative:",
    forwardDifference(calculusFunction, 2)
);

console.log(
    "Central derivative:",
    centralDifference(calculusFunction, 2)
);

console.log("Exact derivative:", 10);


subsection("9.3 Numerical integration");

function trapezoidalIntegral(func, lower, upper, intervals = 1000) {
    if (intervals <= 0) {
        throw new Error("Intervals must be positive.");
    }

    const h = (upper - lower) / intervals;

    let total =
        0.5 * (func(lower) + func(upper));

    for (let i = 1; i < intervals; i++) {
        total += func(lower + i * h);
    }

    return total * h;
}

function simpsonIntegral(func, lower, upper, intervals = 1000) {
    if (intervals <= 0 || intervals % 2 !== 0) {
        throw new Error("Simpson's rule needs a positive even interval count.");
    }

    const h = (upper - lower) / intervals;

    let total = func(lower) + func(upper);

    for (let i = 1; i < intervals; i++) {
        const coefficient = i % 2 === 0 ? 2 : 4;
        total += coefficient * func(lower + i * h);
    }

    return total * h / 3;
}

const square = x => x * x;

console.log(
    "Trapezoidal integral:",
    trapezoidalIntegral(square, 0, 3)
);

console.log(
    "Simpson integral:",
    simpsonIntegral(square, 0, 3)
);


// ============================================================================
// 10. TAYLOR SERIES
// ============================================================================

subsection("10. Taylor series");

function exponentialTaylor(x, terms = 20) {
    if (terms <= 0) {
        throw new Error("Terms must be positive.");
    }

    let total = 1;
    let term = 1;

    for (let n = 1; n < terms; n++) {
        term *= x / n;
        total += term;
    }

    return total;
}

console.log("Taylor approximation e²:", exponentialTaylor(2));
console.log("Math.exp(2):", Math.exp(2));


// ============================================================================
// 11. OPTIMIZATION
// ============================================================================

section("OPTIMIZATION");

subsection("11.1 Numerical gradient");

function numericalGradient(func, point, h = 1e-6) {
    if (h <= 0) {
        throw new Error("h must be positive.");
    }

    return point.map((_, index) => {
        const plus = [...point];
        const minus = [...point];

        plus[index] += h;
        minus[index] -= h;

        return (
            func(plus) - func(minus)
        ) / (2 * h);
    });
}

function quadraticObjective(point) {
    const [x, y] = point;

    return (x - 3) ** 2 +
        2 * (y + 1) ** 2;
}

console.log(
    "Gradient at [0,0]:",
    numericalGradient(quadraticObjective, [0, 0])
);


subsection("11.2 Gradient descent");

function vectorNorm(vector) {
    return Math.sqrt(
        vector.reduce((sum, value) => sum + value * value, 0)
    );
}

function gradientDescent(
    func,
    initialPoint,
    learningRate = 0.1,
    iterations = 100,
    tolerance = 1e-8
) {
    if (learningRate <= 0) {
        throw new Error("Learning rate must be positive.");
    }

    let point = [...initialPoint];
    const history = [func(point)];

    for (let iteration = 0; iteration < iterations; iteration++) {
        const gradient = numericalGradient(func, point);

        const newPoint = point.map(
            (coordinate, index) =>
                coordinate - learningRate * gradient[index]
        );

        history.push(func(newPoint));

        const movement = vectorNorm(
            newPoint.map(
                (value, index) => value - point[index]
            )
        );

        point = newPoint;

        if (movement < tolerance) {
            break;
        }
    }

    return {
        point,
        history
    };
}

const optimizationResult = gradientDescent(
    quadraticObjective,
    [0, 0],
    0.1,
    200
);

console.log("Estimated minimum:", optimizationResult.point);
console.log(
    "Objective value:",
    quadraticObjective(optimizationResult.point)
);


// ============================================================================
// 12. NEWTON'S METHOD
// ============================================================================

subsection("12. Newton's method");

function newtonMethod(
    func,
    derivative,
    initialValue,
    iterations = 20,
    tolerance = 1e-10
) {
    let x = initialValue;

    for (let i = 0; i < iterations; i++) {
        const derivativeValue = derivative(x);

        if (Math.abs(derivativeValue) < 1e-15) {
            throw new Error("Derivative is too close to zero.");
        }

        const next = x - func(x) / derivativeValue;

        if (Math.abs(next - x) < tolerance) {
            return next;
        }

        x = next;
    }

    return x;
}

const squareRootTwo = newtonMethod(
    x => x * x - 2,
    x => 2 * x,
    1
);

console.log("sqrt(2) using Newton:", squareRootTwo);


// ============================================================================
// 13. GOLDEN-SECTION SEARCH
// ============================================================================

subsection("13. Golden-section minimization");

function goldenSectionMinimize(
    func,
    lower,
    upper,
    tolerance = 1e-10
) {
    if (lower >= upper) {
        throw new Error("Invalid interval.");
    }

    const phi = (1 + Math.sqrt(5)) / 2;

    let c = upper - (upper - lower) / phi;
    let d = lower + (upper - lower) / phi;

    while (Math.abs(upper - lower) > tolerance) {
        if (func(c) < func(d)) {
            upper = d;
        } else {
            lower = c;
        }

        c = upper - (upper - lower) / phi;
        d = lower + (upper - lower) / phi;
    }

    return (lower + upper) / 2;
}

const minimum = goldenSectionMinimize(
    x => (x - 4) ** 2 + 2,
    -10,
    10
);

console.log("Minimum:", minimum);


// ============================================================================
// 14. NORMAL DISTRIBUTION
// ============================================================================

section("NORMAL DISTRIBUTION");

function normalPDF(x, meanValue = 0, standardDeviationValue = 1) {
    if (standardDeviationValue <= 0) {
        throw new Error("Standard deviation must be positive.");
    }

    const coefficient =
        1 /
        (standardDeviationValue * Math.sqrt(2 * Math.PI));

    const exponent =
        -0.5 *
        ((x - meanValue) / standardDeviationValue) ** 2;

    return coefficient * Math.exp(exponent);
}

function normalCDF(x, meanValue = 0, standardDeviationValue = 1) {
    if (standardDeviationValue <= 0) {
        throw new Error("Standard deviation must be positive.");
    }

    /*
     * JavaScript does not standardize an erf function across all older
     * runtimes, so a numerical approximation is implemented here.
     */
    const z =
        (x - meanValue) /
        (standardDeviationValue * Math.sqrt(2));

    const sign = z < 0 ? -1 : 1;
    const absoluteZ = Math.abs(z);

    const t = 1 / (1 + 0.3275911 * absoluteZ);

    const polynomial =
        1 -
        (
            (
                (
                    (
                        1.061405429 * t
                        - 1.453152027
                    ) * t
                    + 1.421413741
                ) * t
                - 0.284496736
            ) * t
            + 0.254829592
        ) * t *
        Math.exp(-absoluteZ * absoluteZ);

    const erfValue = sign * polynomial;

    return 0.5 * (1 + erfValue);
}

console.log("Normal PDF at 0:", normalPDF(0));
console.log("Normal CDF at 0:", normalCDF(0));
console.log(
    "Approximate P(-1 <= Z <= 1):",
    normalCDF(1) - normalCDF(-1)
);


// ============================================================================
// 15. LINEAR REGRESSION WITH GRADIENT DESCENT
// ============================================================================

section("STATISTICS + OPTIMIZATION");

function linearRegressionGradientDescent(
    xValues,
    yValues,
    learningRate = 0.01,
    iterations = 5000
) {
    if (
        xValues.length !== yValues.length ||
        xValues.length === 0
    ) {
        throw new Error("Invalid training data.");
    }

    let intercept = 0;
    let slope = 0;

    const losses = [];
    const n = xValues.length;

    for (let iteration = 0; iteration < iterations; iteration++) {
        const predictions = xValues.map(
            x => intercept + slope * x
        );

        const errors = predictions.map(
            (prediction, index) =>
                prediction - yValues[index]
        );

        const loss =
            errors.reduce(
                (sum, error) => sum + error ** 2,
                0
            ) / n;

        losses.push(loss);

        const gradientIntercept =
            2 * errors.reduce(
                (sum, error) => sum + error,
                0
            ) / n;

        const gradientSlope =
            2 * errors.reduce(
                (sum, error, index) =>
                    sum + error * xValues[index],
                0
            ) / n;

        intercept -= learningRate * gradientIntercept;
        slope -= learningRate * gradientSlope;
    }

    return {
        intercept,
        slope,
        losses
    };
}

const gradientRegression =
    linearRegressionGradientDescent(
        studyHours,
        examScores
    );

console.log(
    "Gradient-descent intercept:",
    gradientRegression.intercept
);

console.log(
    "Gradient-descent slope:",
    gradientRegression.slope
);

console.log(
    "Initial loss:",
    gradientRegression.losses[0]
);

console.log(
    "Final loss:",
    gradientRegression.losses[
        gradientRegression.losses.length - 1
    ]
);


// ============================================================================
// 16. CONSTRAINED OPTIMIZATION
// ============================================================================

section("CONSTRAINED OPTIMIZATION");

class Allocation {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    get profit() {
        return 40 * this.x + 30 * this.y;
    }

    get feasible() {
        return (
            this.x >= 0 &&
            this.y >= 0 &&
            2 * this.x + this.y <= 100 &&
            this.x + 2 * this.y <= 80
        );
    }
}

const candidates = [
    new Allocation(0, 0),
    new Allocation(50, 0),
    new Allocation(0, 40),
    new Allocation(40, 20)
];

const feasibleCandidates =
    candidates.filter(candidate => candidate.feasible);

for (const candidate of feasibleCandidates) {
    console.log({
        x: candidate.x,
        y: candidate.y,
        profit: candidate.profit
    });
}

const bestAllocation =
    feasibleCandidates.reduce(
        (best, candidate) =>
            candidate.profit > best.profit
                ? candidate
                : best
    );

console.log("Highest-profit feasible candidate:", bestAllocation);


// ============================================================================
// 17. DISCRETE OPTIMIZATION: KNAPSACK
// ============================================================================

section("DISCRETE OPTIMIZATION");

function zeroOneKnapsack(weights, values, capacity) {
    if (weights.length !== values.length) {
        throw new Error("Weights and values must have equal length.");
    }

    if (capacity < 0) {
        throw new Error("Capacity cannot be negative.");
    }

    if (weights.some(weight => weight <= 0)) {
        throw new Error("Weights must be positive.");
    }

    const itemCount = weights.length;

    const dp = Array.from(
        { length: itemCount + 1 },
        () => Array(capacity + 1).fill(0)
    );

    for (let i = 1; i <= itemCount; i++) {
        const weight = weights[i - 1];
        const value = values[i - 1];

        for (let currentCapacity = 0;
             currentCapacity <= capacity;
             currentCapacity++) {

            dp[i][currentCapacity] =
                dp[i - 1][currentCapacity];

            if (weight <= currentCapacity) {
                dp[i][currentCapacity] =
                    Math.max(
                        dp[i][currentCapacity],
                        dp[i - 1][currentCapacity - weight] + value
                    );
            }
        }
    }

    const selected = [];
    let currentCapacity = capacity;

    for (let i = itemCount; i >= 1; i--) {
        if (
            dp[i][currentCapacity] !==
            dp[i - 1][currentCapacity]
        ) {
            selected.push(i - 1);
            currentCapacity -= weights[i - 1];
        }
    }

    selected.reverse();

    return {
        maximumValue: dp[itemCount][capacity],
        selectedItems: selected
    };
}

console.log(
    zeroOneKnapsack(
        [2, 3, 4, 5],
        [3, 4, 5, 8],
        8
    )
);


// ============================================================================
// 18. NUMERICAL ERROR
// ============================================================================

section("NUMERICAL ERROR");

function absoluteError(approximation, exact) {
    return Math.abs(approximation - exact);
}

function relativeError(approximation, exact) {
    if (exact === 0) {
        return approximation === 0
            ? 0
            : Infinity;
    }

    return Math.abs(approximation - exact) /
        Math.abs(exact);
}

const approximation = 3.14159;

console.log(
    "Absolute error:",
    absoluteError(approximation, Math.PI)
);

console.log(
    "Relative error:",
    relativeError(approximation, Math.PI)
);


// ============================================================================
// 19. NUMERICAL STABILITY
// ============================================================================

subsection("Stable algebraic reformulation");

function directDifference(x) {
    return Math.sqrt(x + 1) - Math.sqrt(x);
}

function stableDifference(x) {
    return 1 /
        (Math.sqrt(x + 1) + Math.sqrt(x));
}

const largeX = 1e16;

console.log(
    "Direct:",
    directDifference(largeX)
);

console.log(
    "Stable:",
    stableDifference(largeX)
);


// ============================================================================
// 20. PRESENT VALUE AND NPV
// ============================================================================

section("INTEGRATED FINANCIAL APPLICATION");

function presentValue(cashFlow, discountRate, period) {
    if (discountRate <= -1) {
        throw new Error("Discount rate must exceed -100%.");
    }

    if (period < 0) {
        throw new Error("Period cannot be negative.");
    }

    return cashFlow /
        Math.pow(1 + discountRate, period);
}

function netPresentValue(cashFlows, discountRate) {
    return cashFlows.reduce(
        (total, cashFlow, period) =>
            total +
            presentValue(cashFlow, discountRate, period),
        0
    );
}

const cashFlows = [-1000, 250, 300, 400, 500];

console.log(
    "NPV at 10%:",
    netPresentValue(cashFlows, 0.10)
);


// ============================================================================
// 21. BISECTION ROOT FINDING
// ============================================================================

section("ROOT FINDING");

function bisectionRoot(
    func,
    lower,
    upper,
    tolerance = 1e-10,
    iterations = 200
) {
    let lowerValue = func(lower);
    let upperValue = func(upper);

    if (lowerValue === 0) {
        return lower;
    }

    if (upperValue === 0) {
        return upper;
    }

    if (lowerValue * upperValue > 0) {
        throw new Error(
            "Function must have opposite signs at the bounds."
        );
    }

    for (let iteration = 0; iteration < iterations; iteration++) {
        const midpoint = (lower + upper) / 2;
        const midpointValue = func(midpoint);

        if (
            Math.abs(midpointValue) < tolerance ||
            Math.abs(upper - lower) < tolerance
        ) {
            return midpoint;
        }

        if (lowerValue * midpointValue < 0) {
            upper = midpoint;
            upperValue = midpointValue;
        } else {
            lower = midpoint;
            lowerValue = midpointValue;
        }
    }

    return (lower + upper) / 2;
}

const irr = bisectionRoot(
    rate => netPresentValue(cashFlows, rate),
    0,
    1
);

console.log("Approximate IRR:", irr);
console.log("NPV at IRR:", netPresentValue(cashFlows, irr));


// ============================================================================
// 22. VALIDATION
// ============================================================================

section("VALIDATION");

if (quadratic(2, 1, -3, 2) !== 0) {
    throw new Error("Quadratic validation failed.");
}

if (
    Math.abs(
        expectedValue(dieValues, dieProbabilities) - 3.5
    ) > 1e-12
) {
    throw new Error("Expected-value validation failed.");
}

if (Math.abs(determinant([[1, 2], [3, 4]]) + 2) > 1e-12) {
    throw new Error("Determinant validation failed.");
}

const solution = solveLinearSystem(
    [
        [2, 1],
        [1, -1]
    ],
    [7, 1]
);

if (
    Math.abs(solution[0] - 8 / 3) > 1e-10 ||
    Math.abs(solution[1] - 5 / 3) > 1e-10
) {
    throw new Error("Linear-system validation failed.");
}

if (
    Math.abs(
        simpsonIntegral(x => x * x, 0, 3) - 9
    ) > 1e-8
) {
    throw new Error("Integration validation failed.");
}

if (
    Math.abs(
        squareRootTwo - Math.sqrt(2)
    ) > 1e-8
) {
    throw new Error("Newton validation failed.");
}

console.log("All validation checks passed.");


// ============================================================================
// 23. REFERENCE FORMULAS
// ============================================================================

section("CORE FORMULAS");

console.log(`
PROBABILITY
P(Aᶜ) = 1 - P(A)
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
P(A|B) = P(A ∩ B) / P(B)
P(A ∩ B) = P(A|B)P(B)

Binomial:
P(X=k) = C(n,k)p^k(1-p)^(n-k)
E[X] = np
Var(X) = np(1-p)

STATISTICS
Mean = Σxᵢ/n
Sample variance = Σ(xᵢ-x̄)²/(n-1)
z = (x-μ)/σ
Correlation = covariance/(sₓsᵧ)

LINEAR ALGEBRA
Ax=b
(AB)ᵢⱼ = ΣAᵢₖBₖⱼ
det([[a,b],[c,d]]) = ad-bc

CALCULUS
f'(x) = lim(h→0)[f(x+h)-f(x)]/h
∫ₐᵇf(x)dx = F(b)-F(a)

OPTIMIZATION
xₖ₊₁ = xₖ - α∇f(xₖ)

Newton:
xₖ₊₁ = xₖ - f(xₖ)/f'(xₖ)

For unconstrained differentiable optimization, a stationary point
satisfies ∇f(x*) = 0.
`);

console.log("\nMathematical revision demonstrations completed.");
