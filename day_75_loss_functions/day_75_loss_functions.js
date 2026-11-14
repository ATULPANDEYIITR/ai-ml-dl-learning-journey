"use strict";

/*
 * Loss Functions: Event-driven evaluation and training telemetry.
 *
 * Features:
 * - MSE and MAE regression metrics
 * - Binary and categorical cross-entropy
 * - Stable logit-based calculations
 * - Asynchronous batch evaluation
 * - EventEmitter-based metric notifications
 * - Threshold-based evaluation policies
 * - Gradient calculations
 * - Input validation and failure handling
 *
 * Runtime: Node.js 18+
 */

const { EventEmitter } = require("node:events");

const EPSILON = 1e-12;

class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = "ValidationError";
    }
}

class LossEvaluationError extends Error {
    constructor(message) {
        super(message);
        this.name = "LossEvaluationError";
    }
}

const Reduction = Object.freeze({
    NONE: "none",
    SUM: "sum",
    MEAN: "mean"
});

function validateArray(actual, predicted) {
    if (!Array.isArray(actual) || !Array.isArray(predicted)) {
        throw new ValidationError("Inputs must be arrays.");
    }

    if (actual.length === 0 || actual.length !== predicted.length) {
        throw new ValidationError(
            "Inputs must be nonempty and have equal lengths."
        );
    }
}

function validateFinite(values) {
    if (!values.every(
        value => typeof value === "number" && Number.isFinite(value)
    )) {
        throw new ValidationError("All values must be finite numbers.");
    }
}

function reduceLoss(name, values, reduction = Reduction.MEAN) {
    if (!Object.values(Reduction).includes(reduction)) {
        throw new ValidationError("Invalid reduction strategy.");
    }

    if (values.length === 0) {
        throw new ValidationError("Cannot reduce empty losses.");
    }

    let result;

    switch (reduction) {
        case Reduction.NONE:
            result = [...values];
            break;

        case Reduction.SUM:
            result = values.reduce((sum, value) => sum + value, 0);
            break;

        case Reduction.MEAN:
            result = values.reduce((sum, value) => sum + value, 0)
                / values.length;
            break;
    }

    if (typeof result === "number" && !Number.isFinite(result)) {
        throw new LossEvaluationError("Loss overflow detected.");
    }

    return Object.freeze({
        name,
        values: Object.freeze([...values]),
        reduction,
        result
    });
}

function mse(actual, predicted, reduction = Reduction.MEAN) {
    validateArray(actual, predicted);
    validateFinite(actual);
    validateFinite(predicted);

    return reduceLoss(
        "MSE",
        actual.map((value, index) => {
            const error = value - predicted[index];
            return error * error;
        }),
        reduction
    );
}

function mae(actual, predicted, reduction = Reduction.MEAN) {
    validateArray(actual, predicted);
    validateFinite(actual);
    validateFinite(predicted);

    return reduceLoss(
        "MAE",
        actual.map((value, index) =>
            Math.abs(value - predicted[index])
        ),
        reduction
    );
}

function validateBinary(actual, probabilities) {
    validateArray(actual, probabilities);

    actual.forEach(label => {
        if (label !== 0 && label !== 1) {
            throw new ValidationError(
                "Binary labels must be 0 or 1."
            );
        }
    });

    probabilities.forEach(probability => {
        if (
            typeof probability !== "number" ||
            !Number.isFinite(probability) ||
            probability < 0 ||
            probability > 1
        ) {
            throw new ValidationError(
                "Invalid binary probability."
            );
        }
    });
}

function binaryCrossEntropy(
    actual,
    probabilities,
    reduction = Reduction.MEAN
) {
    validateBinary(actual, probabilities);

    const values = actual.map((label, index) => {
        const p = Math.min(
            Math.max(probabilities[index], EPSILON),
            1 - EPSILON
        );

        return -(
            label * Math.log(p) +
            (1 - label) * Math.log1p(-p)
        );
    });

    return reduceLoss("Binary Cross-Entropy", values, reduction);
}

function sigmoid(value) {
    if (!Number.isFinite(value)) {
        throw new ValidationError("Logit must be finite.");
    }

    if (value >= 0) {
        return 1 / (1 + Math.exp(-value));
    }

    const exp = Math.exp(value);
    return exp / (1 + exp);
}

function softplus(value) {
    return Math.max(value, 0) +
        Math.log1p(Math.exp(-Math.abs(value)));
}

function binaryCrossEntropyWithLogits(
    actual,
    logits,
    reduction = Reduction.MEAN
) {
    validateArray(actual, logits);

    actual.forEach(label => {
        if (label !== 0 && label !== 1) {
            throw new ValidationError("Invalid binary label.");
        }
    });

    validateFinite(logits);

    const values = logits.map((logit, index) =>
        Math.max(logit, 0) -
        logit * actual[index] +
        softplus(-Math.abs(logit))
    );

    return reduceLoss(
        "Binary Cross-Entropy with Logits",
        values,
        reduction
    );
}

function validateMulticlass(actual, predictions) {
    validateArray(actual, predictions);

    const classes = predictions[0].length;

    if (!Number.isInteger(classes) || classes < 2) {
        throw new ValidationError("At least two classes are required.");
    }

    actual.forEach((label, index) => {
        const row = predictions[index];

        if (!Number.isInteger(label) || label < 0 || label >= classes) {
            throw new ValidationError("Invalid class label.");
        }

        if (!Array.isArray(row) || row.length !== classes) {
            throw new ValidationError("Invalid probability dimensions.");
        }

        validateFinite(row);

        if (row.some(p => p < 0 || p > 1)) {
            throw new ValidationError("Invalid class probability.");
        }

        const sum = row.reduce((total, p) => total + p, 0);

        if (Math.abs(sum - 1) > 1e-8) {
            throw new ValidationError(
                "Probabilities must sum to one."
            );
        }
    });

    return classes;
}

function categoricalCrossEntropy(
    actual,
    predictions,
    reduction = Reduction.MEAN
) {
    validateMulticlass(actual, predictions);

    const values = predictions.map((row, index) =>
        -Math.log(Math.max(row[actual[index]], EPSILON))
    );

    return reduceLoss(
        "Categorical Cross-Entropy",
        values,
        reduction
    );
}

function softmax(logits) {
    if (!Array.isArray(logits) || logits.length < 2) {
        throw new ValidationError("At least two logits are required.");
    }

    validateFinite(logits);

    const maximum = Math.max(...logits);

    const exponentials = logits.map(value =>
        Math.exp(value - maximum)
    );

    const denominator = exponentials.reduce(
        (sum, value) => sum + value,
        0
    );

    return exponentials.map(value => value / denominator);
}

function categoricalCrossEntropyWithLogits(
    actual,
    logits,
    reduction = Reduction.MEAN
) {
    validateArray(actual, logits);

    const classes = logits[0].length;

    if (!Number.isInteger(classes) || classes < 2) {
        throw new ValidationError("Invalid number of classes.");
    }

    const values = logits.map((row, index) => {
        if (!Array.isArray(row) || row.length !== classes) {
            throw new ValidationError("Inconsistent logit dimensions.");
        }

        if (
            !Number.isInteger(actual[index]) ||
            actual[index] < 0 ||
            actual[index] >= classes
        ) {
            throw new ValidationError("Invalid target class.");
        }

        validateFinite(row);

        const maximum = Math.max(...row);

        const logSumExp = maximum + Math.log(
            row.reduce(
                (sum, value) => sum + Math.exp(value - maximum),
                0
            )
        );

        return logSumExp - row[actual[index]];
    });

    return reduceLoss(
        "Categorical Cross-Entropy with Logits",
        values,
        reduction
    );
}

function mseGradient(actual, predicted) {
    validateArray(actual, predicted);
    validateFinite(actual);
    validateFinite(predicted);

    return actual.map(
        (value, index) =>
            2 * (predicted[index] - value) / actual.length
    );
}

function binaryLogitGradient(actual, logits) {
    validateArray(actual, logits);

    actual.forEach(label => {
        if (label !== 0 && label !== 1) {
            throw new ValidationError("Invalid binary label.");
        }
    });

    validateFinite(logits);

    return logits.map(
        (value, index) =>
            (sigmoid(value) - actual[index]) / actual.length
    );
}

/*
 * LossMonitor separates calculation from operational policy.
 * Subscribers receive immutable evaluation events.
 */
class LossMonitor extends EventEmitter {
    constructor(policy = {}) {
        super();

        this.policy = Object.freeze({
            maxRegressionLoss: policy.maxRegressionLoss ?? 1.0,
            maxClassificationLoss: policy.maxClassificationLoss ?? 0.5
        });

        for (const threshold of Object.values(this.policy)) {
            if (!Number.isFinite(threshold) || threshold < 0) {
                throw new ValidationError("Invalid policy threshold.");
            }
        }

        this.history = [];
        this.sequence = 0;
    }

    evaluate(name, result) {
        if (!result || typeof result.result !== "number") {
            throw new ValidationError(
                "A scalar loss result is required."
            );
        }

        if (!Number.isFinite(result.result)) {
            throw new LossEvaluationError("Invalid loss result.");
        }

        const regression = name === "MSE" || name === "MAE";

        const threshold = regression
            ? this.policy.maxRegressionLoss
            : this.policy.maxClassificationLoss;

        const event = Object.freeze({
            id: ++this.sequence,
            name,
            loss: result.result,
            threshold,
            accepted: result.result <= threshold,
            timestamp: new Date().toISOString()
        });

        this.history.push(event);

        this.emit("evaluation", event);

        if (!event.accepted) {
            this.emit("thresholdExceeded", event);
        }

        return event;
    }

    getReport() {
        const grouped = new Map();

        for (const event of this.history) {
            if (!grouped.has(event.name)) {
                grouped.set(event.name, []);
            }

            grouped.get(event.name).push(event.loss);
        }

        return Object.fromEntries(
            [...grouped.entries()].map(([name, values]) => [
                name,
                {
                    count: values.length,
                    mean: values.reduce((a, b) => a + b, 0)
                        / values.length,
                    maximum: Math.max(...values)
                }
            ])
        );
    }
}

/*
 * Asynchronous batch evaluation.
 * A failure in one batch is isolated and reported.
 */
async function evaluateBatches(batches, evaluator, concurrency = 2) {
    if (!Array.isArray(batches) || batches.length === 0) {
        throw new ValidationError("Batches cannot be empty.");
    }

    if (!Number.isInteger(concurrency) || concurrency < 1) {
        throw new ValidationError("Concurrency must be positive.");
    }

    const results = new Array(batches.length);
    let nextIndex = 0;

    async function worker() {
        while (true) {
            const index = nextIndex++;

            if (index >= batches.length) {
                return;
            }

            try {
                results[index] = {
                    index,
                    success: true,
                    value: await evaluator(batches[index], index)
                };
            } catch (error) {
                results[index] = {
                    index,
                    success: false,
                    error: error.message
                };
            }
        }
    }

    const workers = Array.from(
        { length: Math.min(concurrency, batches.length) },
        () => worker()
    );

    await Promise.all(workers);

    return results;
}

async function main() {
    const monitor = new LossMonitor({
        maxRegressionLoss: 1.0,
        maxClassificationLoss: 0.5
    });

    monitor.on("evaluation", event => {
        console.log(
            `[EVENT ${event.id}] ${event.name}: ${event.loss.toFixed(6)}`
        );
    });

    monitor.on("thresholdExceeded", event => {
        console.warn(
            `[POLICY WARNING] ${event.name} exceeded its threshold.`
        );
    });

    const regression = mse(
        [3, -0.5, 2, 7],
        [2.5, 0, 2, 8]
    );

    monitor.evaluate("MSE", regression);

    const absolute = mae(
        [3, -0.5, 2, 7],
        [2.5, 0, 2, 8]
    );

    monitor.evaluate("MAE", absolute);

    const binary = binaryCrossEntropy(
        [1, 0, 1, 0],
        [0.95, 0.10, 0.80, 0.20]
    );

    monitor.evaluate("Binary Cross-Entropy", binary);

    const multiclass = categoricalCrossEntropy(
        [0, 2],
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7]
        ]
    );

    monitor.evaluate("Categorical Cross-Entropy", multiclass);

    const extreme = binaryCrossEntropyWithLogits(
        [1, 0],
        [1000, -1000]
    );

    console.log("Extreme logit loss:", extreme.result);

    console.log("MSE gradient:", mseGradient(
        [1, 2],
        [2, 4]
    ));

    console.log("Binary logit gradient:", binaryLogitGradient(
        [1, 0],
        [3, -2]
    ));

    const batches = [
        { y: [1, 2], p: [1.1, 2.2] },
        { y: [3, 4], p: [2.8, 4.1] },
        { y: [5], p: [5.5] }
    ];

    const batchResults = await evaluateBatches(
        batches,
        async batch => {
            // Simulates an asynchronous evaluation boundary.
            return mse(batch.y, batch.p).result;
        },
        2
    );

    console.log("\nBatch processing:");

    for (const result of batchResults) {
        console.log(result);
    }

    console.log("\nEvaluation report:");
    console.log(monitor.getReport());
}

if (require.main === module) {
    main().catch(error => {
        console.error("Execution failed:", error.message);
        process.exitCode = 1;
    });
}

module.exports = {
    mse,
    mae,
    binaryCrossEntropy,
    binaryCrossEntropyWithLogits,
    categoricalCrossEntropy,
    categoricalCrossEntropyWithLogits,
    mseGradient,
    binaryLogitGradient,
    LossMonitor,
    evaluateBatches
};
