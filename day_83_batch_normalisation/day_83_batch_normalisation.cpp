"use strict";

/*
 * Batch Normalization Engine
 *
 * Node.js implementation featuring:
 * - Event-driven batch processing.
 * - Training and inference state separation.
 * - Feature-wise statistics.
 * - Running-statistic updates.
 * - Adaptive affine parameters.
 * - Asynchronous inference queue.
 * - Model-state validation.
 * - Gradient calculation.
 */

const { EventEmitter } = require("node:events");
const fs = require("node:fs/promises");

class NormalizationError extends Error {}

class BatchValidationError extends NormalizationError {}

class NumericalValidationError extends NormalizationError {}

const ExecutionMode = Object.freeze({
    TRAINING: "training",
    INFERENCE: "inference"
});

function validateNumber(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new NumericalValidationError(
            `${name} must be a finite number`
        );
    }
}

class BatchStatistics {
    constructor(mean, variance, count) {
        this.mean = Object.freeze([...mean]);
        this.variance = Object.freeze([...variance]);
        this.count = count;

        Object.freeze(this);
    }
}

class NormalizationEngine extends EventEmitter {
    #features;
    #epsilon;
    #momentum;

    #gamma;
    #beta;

    #runningMean;
    #runningVariance;

    #batchesSeen = 0;
    #inferenceQueue = Promise.resolve();

    constructor({
        features,
        epsilon = 1e-5,
        momentum = 0.9
    }) {
        super();

        if (!Number.isInteger(features) || features <= 0) {
            throw new RangeError("features must be a positive integer");
        }

        validateNumber(epsilon, "epsilon");
        validateNumber(momentum, "momentum");

        if (epsilon <= 0 || momentum < 0 || momentum >= 1) {
            throw new RangeError("Invalid normalization configuration");
        }

        this.#features = features;
        this.#epsilon = epsilon;
        this.#momentum = momentum;

        this.#gamma = Array(features).fill(1);
        this.#beta = Array(features).fill(0);

        this.#runningMean = Array(features).fill(0);
        this.#runningVariance = Array(features).fill(1);
    }

    get state() {
        return Object.freeze({
            features: this.#features,
            batchesSeen: this.#batchesSeen,
            gamma: Object.freeze([...this.#gamma]),
            beta: Object.freeze([...this.#beta]),
            runningMean: Object.freeze([...this.#runningMean]),
            runningVariance: Object.freeze([...this.#runningVariance])
        });
    }

    #validateBatch(batch) {
        if (!Array.isArray(batch) || batch.length === 0) {
            throw new BatchValidationError("Batch cannot be empty");
        }

        for (const [index, row] of batch.entries()) {
            if (!Array.isArray(row) || row.length !== this.#features) {
                throw new BatchValidationError(
                    `Invalid feature count at row ${index}`
                );
            }

            row.forEach((value, feature) => {
                validateNumber(value, `row ${index}, feature ${feature}`);
            });
        }
    }

    #statistics(batch) {
        const count = batch.length;
        const mean = Array(this.#features).fill(0);
        const variance = Array(this.#features).fill(0);

        // Welford-style accumulation avoids unnecessary intermediate
        // arrays and reduces numerical cancellation.
        const m2 = Array(this.#features).fill(0);

        for (let i = 0; i < count; i++) {
            const row = batch[i];

            for (let j = 0; j < this.#features; j++) {
                const delta = row[j] - mean[j];

                mean[j] += delta / (i + 1);

                const deltaAfter = row[j] - mean[j];

                m2[j] += delta * deltaAfter;
            }
        }

        for (let j = 0; j < this.#features; j++) {
            variance[j] = Math.max(0, m2[j] / count);
        }

        return new BatchStatistics(mean, variance, count);
    }

    #commitStatistics(statistics) {
        const nextMean = this.#runningMean.map(
            (old, j) =>
                this.#momentum * old +
                (1 - this.#momentum) * statistics.mean[j]
        );

        const nextVariance = this.#runningVariance.map(
            (old, j) =>
                this.#momentum * old +
                (1 - this.#momentum) * statistics.variance[j]
        );

        // Commit together after all calculations have succeeded.
        this.#runningMean = nextMean;
        this.#runningVariance = nextVariance;
        this.#batchesSeen++;

        this.emit("statisticsUpdated", {
            batchesSeen: this.#batchesSeen,
            statistics
        });
    }

    forward(batch, mode = ExecutionMode.TRAINING) {
        this.#validateBatch(batch);

        if (!Object.values(ExecutionMode).includes(mode)) {
            throw new BatchValidationError("Unsupported execution mode");
        }

        let mean;
        let variance;

        if (mode === ExecutionMode.TRAINING) {
            const statistics = this.#statistics(batch);

            mean = statistics.mean;
            variance = statistics.variance;

            this.#commitStatistics(statistics);
        } else {
            if (this.#batchesSeen === 0) {
                throw new NormalizationError(
                    "Inference statistics have not been initialized"
                );
            }

            mean = this.#runningMean;
            variance = this.#runningVariance;
        }

        const inverseStd = variance.map(
            value => 1 / Math.sqrt(value + this.#epsilon)
        );

        const normalized = [];
        const output = [];

        for (const row of batch) {
            const normalizedRow = [];
            const outputRow = [];

            for (let j = 0; j < this.#features; j++) {
                const value = (row[j] - mean[j]) * inverseStd[j];

                const result = this.#gamma[j] * value + this.#beta[j];

                validateNumber(result, "normalization result");

                normalizedRow.push(value);
                outputRow.push(result);
            }

            normalized.push(normalizedRow);
            output.push(outputRow);
        }

        const result = Object.freeze({
            mode,
            output,
            cache: {
                normalized,
                inverseStd,
                sampleCount: batch.length
            }
        });

        this.emit("forwardCompleted", {
            mode,
            sampleCount: batch.length
        });

        return result;
    }

    backward(upstreamGradient, cache) {
        if (!cache || !Array.isArray(cache.normalized)) {
            throw new BatchValidationError("Invalid forward cache");
        }

        const count = cache.sampleCount;

        if (upstreamGradient.length !== count) {
            throw new BatchValidationError("Gradient batch mismatch");
        }

        this.#validateBatch(upstreamGradient);

        const dGamma = Array(this.#features).fill(0);
        const dBeta = Array(this.#features).fill(0);

        for (let j = 0; j < this.#features; j++) {
            for (let i = 0; i < count; i++) {
                const gradient = upstreamGradient[i][j];

                dGamma[j] += gradient * cache.normalized[i][j];
                dBeta[j] += gradient;
            }
        }

        const dInput = Array.from(
            { length: count },
            () => Array(this.#features).fill(0)
        );

        for (let j = 0; j < this.#features; j++) {
            for (let i = 0; i < count; i++) {
                dInput[i][j] =
                    this.#gamma[j] *
                    cache.inverseStd[j] /
                    count *
                    (
                        count * upstreamGradient[i][j]
                        - dBeta[j]
                        - cache.normalized[i][j] * dGamma[j]
                    );
            }
        }

        return { dInput, dGamma, dBeta };
    }

    updateParameters({ dGamma, dBeta, learningRate }) {
        validateNumber(learningRate, "learningRate");

        if (learningRate <= 0) {
            throw new RangeError("Learning rate must be positive");
        }

        if (
            !Array.isArray(dGamma) ||
            !Array.isArray(dBeta) ||
            dGamma.length !== this.#features ||
            dBeta.length !== this.#features
        ) {
            throw new BatchValidationError("Invalid parameter gradients");
        }

        for (let j = 0; j < this.#features; j++) {
            validateNumber(dGamma[j], "dGamma");
            validateNumber(dBeta[j], "dBeta");
        }

        const nextGamma = this.#gamma.map(
            (value, j) => value - learningRate * dGamma[j]
        );

        const nextBeta = this.#beta.map(
            (value, j) => value - learningRate * dBeta[j]
        );

        this.#gamma = nextGamma;
        this.#beta = nextBeta;

        this.emit("parametersUpdated", {
            learningRate,
            gamma: [...this.#gamma],
            beta: [...this.#beta]
        });
    }

    // Queue inference requests to preserve deterministic processing order.
    inferAsync(batch) {
        const task = this.#inferenceQueue.then(
            () => this.forward(batch, ExecutionMode.INFERENCE)
        );

        this.#inferenceQueue = task.catch(() => undefined);

        return task;
    }

    async save(path) {
        const data = {
            version: 1,
            configuration: {
                features: this.#features,
                epsilon: this.#epsilon,
                momentum: this.#momentum
            },
            state: this.state
        };

        await fs.writeFile(
            path,
            JSON.stringify(data, null, 2),
            { encoding: "utf8", mode: 0o600 }
        );

        this.emit("modelSaved", { path });
    }

    static async load(path) {
        const raw = await fs.readFile(path, "utf8");
        const data = JSON.parse(raw);

        if (data.version !== 1 || !data.configuration || !data.state) {
            throw new NormalizationError("Unsupported model format");
        }

        const engine = new NormalizationEngine(data.configuration);
        const state = data.state;

        for (const key of [
            "gamma",
            "beta",
            "runningMean",
            "runningVariance"
        ]) {
            if (
                !Array.isArray(state[key]) ||
                state[key].length !== engine.#features
            ) {
                throw new BatchValidationError(`Invalid state field: ${key}`);
            }

            state[key].forEach(value => validateNumber(value, key));
        }

        if (state.runningVariance.some(value => value < 0)) {
            throw new NumericalValidationError("Negative running variance");
        }

        if (
            !Number.isInteger(state.batchesSeen) ||
            state.batchesSeen < 0
        ) {
            throw new BatchValidationError("Invalid batch counter");
        }

        engine.#gamma = [...state.gamma];
        engine.#beta = [...state.beta];
        engine.#runningMean = [...state.runningMean];
        engine.#runningVariance = [...state.runningVariance];
        engine.#batchesSeen = state.batchesSeen;

        return engine;
    }
}

async function main() {
    const engine = new NormalizationEngine({
        features: 3,
        epsilon: 1e-5,
        momentum: 0.9
    });

    engine.on("statisticsUpdated", event => {
        console.log(
            "Statistics updated for batch:",
            event.batchesSeen
        );
    });

    engine.on("parametersUpdated", () => {
        console.log("Affine parameters updated");
    });

    const trainingBatch = [
        [2, 4, 6],
        [4, 8, 10],
        [6, 12, 14],
        [8, 16, 18]
    ];

    console.log("\nTRAINING");

    const training = engine.forward(
        trainingBatch,
        ExecutionMode.TRAINING
    );

    console.log("Normalized output:", training.output);

    const gradients = engine.backward(
        trainingBatch.map(() => [1, 1, 1]),
        training.cache
    );

    console.log("Input gradients:", gradients.dInput);

    engine.updateParameters({
        dGamma: gradients.dGamma,
        dBeta: gradients.dBeta,
        learningRate: 0.01
    });

    console.log("\nINFERENCE");

    const inference = await engine.inferAsync([
        [5, 10, 12],
        [7, 13, 15]
    ]);

    console.log("Inference output:", inference.output);

    console.log("\nPERSISTENCE");

    await engine.save("normalization-model.json");

    const restored = await NormalizationEngine.load(
        "normalization-model.json"
    );

    console.log("Restored state:", restored.state);

    console.log("\nVALIDATION");

    try {
        engine.forward([[1, 2]], ExecutionMode.TRAINING);
    } catch (error) {
        console.log("Rejected invalid batch:", error.message);
    }

    try {
        engine.forward([[1, NaN, 3]], ExecutionMode.TRAINING);
    } catch (error) {
        console.log("Rejected non-finite input:", error.message);
    }
}

main().catch(error => {
    console.error("Normalization engine failure:", error.message);
    process.exitCode = 1;
});
