"use strict";

/*
 * Kernel Methods: a Node.js implementation focused on similarity spaces.
 *
 * This file deliberately uses JavaScript's strengths:
 * - immutable-style domain objects,
 * - Maps for label aggregation,
 * - asynchronous event-driven evaluation,
 * - validation at API boundaries,
 * - structured policy-style result objects.
 *
 * No external packages are required.
 */

class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = "ValidationError";
    }
}

function validateVector(values, name = "vector") {
    if (!Array.isArray(values) || values.length === 0) {
        throw new ValidationError(`${name} must be a non-empty array`);
    }

    const vector = values.map(Number);

    if (vector.some((value) => !Number.isFinite(value))) {
        throw new ValidationError(`${name} contains a non-finite value`);
    }

    return Object.freeze(vector);
}

function assertSameDimension(a, b) {
    if (a.length !== b.length) {
        throw new ValidationError(
            `dimension mismatch: ${a.length} != ${b.length}`
        );
    }
}

function dot(a, b) {
    assertSameDimension(a, b);

    let result = 0;

    for (let i = 0; i < a.length; i += 1) {
        result += a[i] * b[i];
    }

    return result;
}

function squaredDistance(a, b) {
    assertSameDimension(a, b);

    let result = 0;

    for (let i = 0; i < a.length; i += 1) {
        const difference = a[i] - b[i];
        result += difference * difference;
    }

    return result;
}

function linearKernel(a, b) {
    return dot(a, b);
}

function polynomialKernel(
    a,
    b,
    { degree = 3, gamma = 1 / a.length, coefficient = 1 } = {}
) {
    assertSameDimension(a, b);

    if (!Number.isInteger(degree) || degree < 1) {
        throw new ValidationError("degree must be a positive integer");
    }

    if (!(gamma > 0) || !Number.isFinite(gamma)) {
        throw new ValidationError("gamma must be finite and positive");
    }

    if (!(coefficient >= 0) || !Number.isFinite(coefficient)) {
        throw new ValidationError(
            "coefficient must be finite and non-negative"
        );
    }

    return Math.pow(
        gamma * dot(a, b) + coefficient,
        degree
    );
}

function rbfKernel(
    a,
    b,
    gamma = 1 / a.length
) {
    assertSameDimension(a, b);

    if (!(gamma > 0) || !Number.isFinite(gamma)) {
        throw new ValidationError("gamma must be finite and positive");
    }

    return Math.exp(-gamma * squaredDistance(a, b));
}

function gramMatrix(samples, kernel) {
    if (!Array.isArray(samples) || samples.length === 0) {
        throw new ValidationError("samples must not be empty");
    }

    const normalized = samples.map((sample, index) =>
        validateVector(sample, `samples[${index}]`)
    );

    const dimension = normalized[0].length;

    if (normalized.some((sample) => sample.length !== dimension)) {
        throw new ValidationError(
            "all samples must have the same dimension"
        );
    }

    const matrix = normalized.map(() =>
        Array(normalized.length).fill(0)
    );

    for (let i = 0; i < normalized.length; i += 1) {
        for (let j = i; j < normalized.length; j += 1) {
            const value = kernel(normalized[i], normalized[j]);
            matrix[i][j] = value;
            matrix[j][i] = value;
        }
    }

    return matrix;
}

/*
 * A lightweight positive-semidefinite test.
 *
 * For a valid kernel, every finite Gram matrix should be PSD.
 * The test uses a Cholesky-style decomposition and tolerates
 * tiny negative values introduced by floating-point arithmetic.
 */
function isPositiveSemidefinite(matrix, tolerance = 1e-9) {
    const n = matrix.length;

    if (n === 0) {
        return true;
    }

    if (matrix.some((row) => row.length !== n)) {
        throw new ValidationError("matrix must be square");
    }

    const lower = Array.from(
        { length: n },
        () => Array(n).fill(0)
    );

    for (let i = 0; i < n; i += 1) {
        for (let j = 0; j <= i; j += 1) {
            let value = matrix[i][j];

            for (let k = 0; k < j; k += 1) {
                value -= lower[i][k] * lower[j][k];
            }

            if (i === j) {
                if (value < -tolerance) {
                    return false;
                }

                lower[i][j] = Math.sqrt(Math.max(0, value));
            } else {
                const pivot = lower[j][j];

                if (Math.abs(pivot) <= tolerance) {
                    if (Math.abs(value) > tolerance) {
                        return false;
                    }

                    lower[i][j] = 0;
                } else {
                    lower[i][j] = value / pivot;
                }
            }
        }
    }

    return true;
}

function solveLinearSystem(matrix, values) {
    const n = matrix.length;

    if (
        n === 0 ||
        values.length !== n ||
        matrix.some((row) => row.length !== n)
    ) {
        throw new ValidationError("invalid linear system");
    }

    const augmented = matrix.map((row, index) => [
        ...row,
        values[index]
    ]);

    for (let column = 0; column < n; column += 1) {
        let pivotRow = column;

        for (let row = column + 1; row < n; row += 1) {
            if (
                Math.abs(augmented[row][column]) >
                Math.abs(augmented[pivotRow][column])
            ) {
                pivotRow = row;
            }
        }

        if (Math.abs(augmented[pivotRow][column]) < 1e-12) {
            throw new Error(
                "singular or numerically unstable system"
            );
        }

        [augmented[column], augmented[pivotRow]] = [
            augmented[pivotRow],
            augmented[column]
        ];

        for (let row = column + 1; row < n; row += 1) {
            const factor =
                augmented[row][column] /
                augmented[column][column];

            for (let k = column; k <= n; k += 1) {
                augmented[row][k] -=
                    factor * augmented[column][k];
            }
        }
    }

    const solution = Array(n).fill(0);

    for (let row = n - 1; row >= 0; row -= 1) {
        let remainder = augmented[row][n];

        for (let column = row + 1; column < n; column += 1) {
            remainder -=
                augmented[row][column] * solution[column];
        }

        solution[row] =
            remainder / augmented[row][row];
    }

    return solution;
}

class KernelSimilarityClassifier {
    constructor(samples, labels, kernel) {
        if (samples.length !== labels.length) {
            throw new ValidationError(
                "samples and labels must have equal length"
            );
        }

        if (samples.length === 0) {
            throw new ValidationError("training data is empty");
        }

        this.samples = samples.map((sample, index) =>
            validateVector(sample, `samples[${index}]`)
        );

        this.labels = [...labels];
        this.kernel = kernel;
    }

    predict(query) {
        const normalizedQuery =
            validateVector(query, "query");

        const scores = new Map();

        for (let i = 0; i < this.samples.length; i += 1) {
            const similarity = this.kernel(
                this.samples[i],
                normalizedQuery
            );

            const label = this.labels[i];

            scores.set(
                label,
                (scores.get(label) ?? 0) + similarity
            );
        }

        let bestLabel;
        let bestScore = -Infinity;

        for (const [label, score] of scores.entries()) {
            if (score > bestScore) {
                bestScore = score;
                bestLabel = label;
            }
        }

        return {
            label: bestLabel,
            similarityScore: bestScore
        };
    }
}

class KernelRidgeModel {
    constructor(kernel, regularization = 0.1) {
        if (!(regularization > 0)) {
            throw new ValidationError(
                "regularization must be positive"
            );
        }

        this.kernel = kernel;
        this.regularization = regularization;
        this.samples = [];
        this.coefficients = [];
    }

    fit(samples, targets) {
        if (samples.length !== targets.length) {
            throw new ValidationError(
                "samples and targets must have equal length"
            );
        }

        this.samples = samples.map((sample, index) =>
            validateVector(sample, `samples[${index}]`)
        );

        const kernelMatrix = gramMatrix(
            this.samples,
            this.kernel
        );

        const regularized = kernelMatrix.map(
            (row, i) =>
                row.map(
                    (value, j) =>
                        value +
                        (i === j
                            ? this.regularization
                            : 0)
                )
        );

        this.coefficients = solveLinearSystem(
            regularized,
            targets.map(Number)
        );
    }

    predict(query) {
        if (this.coefficients.length === 0) {
            throw new Error("model has not been fitted");
        }

        const normalizedQuery =
            validateVector(query, "query");

        return this.samples.reduce(
            (prediction, sample, index) =>
                prediction +
                this.coefficients[index] *
                    this.kernel(sample, normalizedQuery),
            0
        );
    }
}

/*
 * Event-driven evaluation keeps model execution separate from
 * presentation. This resembles a service that emits evaluation
 * events as individual kernel computations complete.
 */
function evaluateAsync(name, task) {
    return new Promise((resolve, reject) => {
        setImmediate(() => {
            try {
                const result = task();

                console.log(`[${name}] completed`);
                resolve(result);
            } catch (error) {
                console.error(
                    `[${name}] failed: ${error.message}`
                );
                reject(error);
            }
        });
    });
}

async function main() {
    const samples = [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
        [-0.8, -1.1],
        [-1.1, 0.8],
        [0.9, -1.2],
        [1.1, 1.0]
    ];

    const labels = [0, 1, 1, 0, 0, 1, 1, 0];

    await evaluateAsync("RBF Gram matrix", () => {
        const matrix = gramMatrix(
            samples,
            (a, b) => rbfKernel(a, b, 2)
        );

        console.table(matrix);
        console.log(
            "PSD:",
            isPositiveSemidefinite(matrix)
        );
    });

    await evaluateAsync("Polynomial Gram matrix", () => {
        const matrix = gramMatrix(
            samples,
            (a, b) =>
                polynomialKernel(a, b, {
                    degree: 2,
                    gamma: 0.5,
                    coefficient: 1
                })
        );

        console.log(
            "Polynomial kernel PSD:",
            isPositiveSemidefinite(matrix)
        );
    });

    await evaluateAsync("Kernel classifier", () => {
        const classifier =
            new KernelSimilarityClassifier(
                samples,
                labels,
                (a, b) => rbfKernel(a, b, 2)
            );

        const queries = [
            [-0.9, -0.9],
            [-0.9, 0.9],
            [0.9, -0.9],
            [0.9, 0.9]
        ];

        for (const query of queries) {
            console.log(
                query,
                "=>",
                classifier.predict(query)
            );
        }
    });

    await evaluateAsync("Kernel ridge", () => {
        const regressionSamples = [
            [-2],
            [-1],
            [0],
            [1],
            [2]
        ];

        const targets = [4, 1, 0, 1, 4];

        const model = new KernelRidgeModel(
            (a, b) => rbfKernel(a, b, 0.7),
            0.05
        );

        model.fit(regressionSamples, targets);

        for (const value of [-1.5, -0.5, 0.5, 1.5]) {
            console.log(
                `x=${value}`,
                "=>",
                model.predict([value]).toFixed(4)
            );
        }
    });

    try {
        polynomialKernel([1, 2], [1], { degree: 2 });
    } catch (error) {
        console.log(
            "Expected validation failure:",
            error.message
        );
    }

    console.log("Kernel method evaluation completed.");
}

main().catch((error) => {
    console.error("Fatal error:", error.message);
    process.exitCode = 1;
});
