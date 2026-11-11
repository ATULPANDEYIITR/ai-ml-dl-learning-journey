"use strict";

/*
 * Perceptron Learning Rule and Linear Separability
 *
 * Node.js 18+
 *
 * Demonstrates:
 * - Online perceptron learning.
 * - Binary step activation.
 * - Linear decision boundaries.
 * - Asynchronous training.
 * - Event-driven training observations.
 * - Epoch-based convergence.
 * - Linear separability experiments.
 * - Evaluation and confusion matrices.
 * - Feature normalization.
 * - Model persistence.
 * - Input validation and numerical safeguards.
 */

const fs = require("node:fs/promises");
const EventEmitter = require("node:events");
const { setTimeout: delay } = require("node:timers/promises");

class ValidationError extends Error {}
class TrainingError extends Error {}

const DEFAULT_CONFIG = Object.freeze({
    learningRate: 0.1,
    maxEpochs: 100,
    tolerance: 0,
    shuffle: false,
    seed: 42,
    yieldEvery: 5
});

function validateDataset(features, labels) {
    if (!Array.isArray(features) || !Array.isArray(labels)) {
        throw new ValidationError("Features and labels must be arrays.");
    }

    if (features.length === 0 || features.length !== labels.length) {
        throw new ValidationError("Dataset sizes must match and be non-zero.");
    }

    if (!Array.isArray(features[0]) || features[0].length === 0) {
        throw new ValidationError("Samples must contain features.");
    }

    const dimensions = features[0].length;

    features.forEach((sample, index) => {
        if (!Array.isArray(sample) || sample.length !== dimensions) {
            throw new ValidationError(
                `Invalid feature dimensions at sample ${index}.`
            );
        }

        sample.forEach(value => {
            if (typeof value !== "number" || !Number.isFinite(value)) {
                throw new ValidationError("Features must be finite numbers.");
            }
        });
    });

    if (!labels.every(label => label === 0 || label === 1)) {
        throw new ValidationError("Labels must be 0 or 1.");
    }

    if (new Set(labels).size !== 2) {
        throw new ValidationError("Both classes must be represented.");
    }

    return dimensions;
}

function validateConfig(config) {
    const merged = { ...DEFAULT_CONFIG, ...config };

    if (!Number.isFinite(merged.learningRate) ||
        merged.learningRate <= 0) {
        throw new ValidationError("Learning rate must be positive.");
    }

    if (!Number.isInteger(merged.maxEpochs) ||
        merged.maxEpochs < 1) {
        throw new ValidationError("maxEpochs must be a positive integer.");
    }

    if (!Number.isFinite(merged.tolerance) ||
        merged.tolerance < 0) {
        throw new ValidationError("Tolerance must be non-negative.");
    }

    if (!Number.isInteger(merged.yieldEvery) ||
        merged.yieldEvery < 1) {
        throw new ValidationError("yieldEvery must be positive.");
    }

    return Object.freeze(merged);
}

function activation(value) {
    return value >= 0 ? 1 : 0;
}

// Deterministic pseudo-random generator for reproducible shuffling.
function createRandom(seed) {
    let state = seed >>> 0;

    return function random() {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function shuffleArray(array, random) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

class FeatureScaler {
    constructor() {
        this.means = null;
        this.scales = null;
    }

    fit(features) {
        if (!Array.isArray(features) || features.length === 0) {
            throw new ValidationError("Scaler requires non-empty features.");
        }

        const dimensions = features[0].length;

        if (dimensions === 0 ||
            features.some(row => row.length !== dimensions)) {
            throw new ValidationError("Inconsistent feature dimensions.");
        }

        this.means = Array.from({ length: dimensions }, (_, j) =>
            features.reduce((sum, row) => sum + row[j], 0) / features.length
        );

        this.scales = Array.from({ length: dimensions }, (_, j) => {
            const variance = features.reduce(
                (sum, row) => sum + (row[j] - this.means[j]) ** 2,
                0
            ) / features.length;

            const standardDeviation = Math.sqrt(variance);

            return standardDeviation > 1e-12 ? standardDeviation : 1;
        });

        return this;
    }

    transform(features) {
        if (!this.means || !this.scales) {
            throw new TrainingError("Scaler has not been fitted.");
        }

        return features.map(row => {
            if (row.length !== this.means.length) {
                throw new ValidationError("Feature dimension mismatch.");
            }

            return row.map(
                (value, j) => (value - this.means[j]) / this.scales[j]
            );
        });
    }

    fitTransform(features) {
        return this.fit(features).transform(features);
    }
}

class Perceptron extends EventEmitter {
    constructor(config = {}) {
        super();

        this.config = validateConfig(config);
        this.weights = [];
        this.bias = 0;
        this.dimensions = 0;
        this.converged = false;
        this.epochsTrained = 0;
        this.history = [];
        this.training = false;
    }

    netInput(sample) {
        if (sample.length !== this.dimensions) {
            throw new ValidationError("Feature dimension mismatch.");
        }

        let result = this.bias;

        for (let i = 0; i < this.dimensions; i++) {
            result += this.weights[i] * sample[i];
        }

        return result;
    }

    predict(sample) {
        return activation(this.netInput(sample));
    }

    predictBatch(features) {
        return features.map(sample => this.predict(sample));
    }

    async train(features, labels, options = {}) {
        if (this.training) {
            throw new TrainingError("Training is already running.");
        }

        this.dimensions = validateDataset(features, labels);

        const signal = options.signal;

        this.weights = new Array(this.dimensions).fill(0);
        this.bias = 0;
        this.converged = false;
        this.epochsTrained = 0;
        this.history = [];
        this.training = true;

        const samples = features.map((x, i) => ({
            x: [...x],
            target: labels[i]
        }));

        const random = createRandom(this.config.seed);

        try {
            for (
                let epoch = 1;
                epoch <= this.config.maxEpochs;
                epoch++
            ) {
                if (signal?.aborted) {
                    throw new TrainingError("Training was cancelled.");
                }

                if (this.config.shuffle) {
                    shuffleArray(samples, random);
                }

                let mistakes = 0;

                for (const sample of samples) {
                    const prediction = this.predict(sample.x);
                    const error = sample.target - prediction;

                    if (error !== 0) {
                        mistakes++;

                        // Weight and bias are updated together.
                        for (let j = 0; j < this.dimensions; j++) {
                            this.weights[j] +=
                                this.config.learningRate *
                                error *
                                sample.x[j];
                        }

                        this.bias += this.config.learningRate * error;
                    }
                }

                this.epochsTrained = epoch;

                const snapshot = Object.freeze({
                    epoch,
                    mistakes,
                    weights: [...this.weights],
                    bias: this.bias
                });

                this.history.push(snapshot);
                this.emit("epoch", snapshot);

                if (mistakes === 0) {
                    this.converged = true;
                    this.emit("converged", snapshot);
                    break;
                }

                if (epoch % this.config.yieldEvery === 0) {
                    await delay(0);
                }
            }

            this.emit("finished", {
                converged: this.converged,
                epochs: this.epochsTrained
            });

            return this;
        } catch (error) {
            this.emit("trainingError", error);
            throw error;
        } finally {
            this.training = false;
        }
    }

    evaluate(features, labels) {
        validateDataset(features, labels);

        let tp = 0;
        let tn = 0;
        let fp = 0;
        let fn = 0;

        features.forEach((sample, index) => {
            const prediction = this.predict(sample);
            const actual = labels[index];

            if (prediction === 1 && actual === 1) tp++;
            else if (prediction === 0 && actual === 0) tn++;
            else if (prediction === 1) fp++;
            else fn++;
        });

        const correct = tp + tn;

        return {
            accuracy: correct / labels.length,
            correct,
            incorrect: labels.length - correct,
            confusionMatrix: { tp, tn, fp, fn }
        };
    }

    decisionBoundary2D() {
        if (this.dimensions !== 2) {
            throw new ValidationError(
                "Decision boundary requires exactly two features."
            );
        }

        const [w1, w2] = this.weights;

        if (Math.abs(w2) < 1e-12) {
            return {
                type: "vertical",
                x: Math.abs(w1) < 1e-12 ? null : -this.bias / w1
            };
        }

        return {
            type: "linear",
            slope: -w1 / w2,
            intercept: -this.bias / w2
        };
    }

    toJSON() {
        if (this.weights.length === 0) {
            throw new TrainingError("Cannot serialize an untrained model.");
        }

        return {
            version: 1,
            config: this.config,
            weights: [...this.weights],
            bias: this.bias,
            dimensions: this.dimensions,
            converged: this.converged,
            epochsTrained: this.epochsTrained
        };
    }

    async save(filePath) {
        // Use a temporary file so interrupted writes do not corrupt the target.
        const temporary = `${filePath}.tmp`;

        await fs.writeFile(
            temporary,
            JSON.stringify(this.toJSON(), null, 2),
            { encoding: "utf8", mode: 0o600 }
        );

        await fs.rename(temporary, filePath);
    }

    static async load(filePath) {
        const raw = await fs.readFile(filePath, "utf8");
        const data = JSON.parse(raw);

        if (
            data.version !== 1 ||
            !Array.isArray(data.weights) ||
            !Number.isInteger(data.dimensions) ||
            data.dimensions < 1 ||
            data.weights.length !== data.dimensions ||
            !data.weights.every(Number.isFinite) ||
            !Number.isFinite(data.bias)
        ) {
            throw new ValidationError("Invalid model file.");
        }

        const model = new Perceptron(data.config);

        model.weights = [...data.weights];
        model.bias = data.bias;
        model.dimensions = data.dimensions;
        model.converged = Boolean(data.converged);
        model.epochsTrained = data.epochsTrained;

        return model;
    }
}

async function demonstrateAND() {
    const features = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ];

    const labels = [0, 0, 0, 1];

    const model = new Perceptron({
        learningRate: 0.1,
        maxEpochs: 100,
        shuffle: false
    });

    model.on("epoch", result => {
        console.log(
            `Epoch ${result.epoch}: mistakes=${result.mistakes}, ` +
            `weights=${result.weights}, bias=${result.bias.toFixed(2)}`
        );
    });

    model.on("converged", () => {
        console.log("AND classifier converged.");
    });

    await model.train(features, labels);

    console.log("AND predictions:", model.predictBatch(features));
    console.log("Evaluation:", model.evaluate(features, labels));
    console.log("Decision boundary:", model.decisionBoundary2D());

    if (!model.converged) {
        throw new TrainingError("AND classifier failed to converge.");
    }

    return model;
}

async function demonstrateXOR() {
    const features = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ];

    const labels = [0, 1, 1, 0];

    const model = new Perceptron({
        learningRate: 0.1,
        maxEpochs: 100
    });

    await model.train(features, labels);

    console.log("\nXOR experiment");
    console.log("Converged:", model.converged);
    console.log("Predictions:", model.predictBatch(features));
    console.log("Final epoch:", model.history.at(-1));

    if (model.converged) {
        throw new Error("Unexpected XOR convergence.");
    }
}

async function runTests() {
    const model = await demonstrateAND();

    const predictions = model.predictBatch([
        [0, 0], [0, 1], [1, 0], [1, 1]
    ]);

    if (predictions.join(",") !== "0,0,0,1") {
        throw new Error("AND classification test failed.");
    }

    const scaler = new FeatureScaler();

    const normalized = scaler.fitTransform([
        [10, 100],
        [20, 200],
        [30, 300]
    ]);

    if (normalized.length !== 3) {
        throw new Error("Scaler test failed.");
    }

    try {
        validateDataset([[1, 2]], [4]);
        throw new Error("Invalid label was accepted.");
    } catch (error) {
        if (!(error instanceof ValidationError)) throw error;
    }

    await model.save("perceptron-model.json");

    const restored = await Perceptron.load("perceptron-model.json");

    if (restored.predict([1, 1]) !== 1) {
        throw new Error("Persistence test failed.");
    }

    await fs.unlink("perceptron-model.json");

    await demonstrateXOR();

    console.log("\nAll JavaScript tests passed.");
}

if (require.main === module) {
    runTests().catch(error => {
        console.error("Execution failed:", error.message);
        process.exitCode = 1;
    });
}

module.exports = {
    Perceptron,
    FeatureScaler,
    ValidationError,
    TrainingError,
    validateDataset
};
