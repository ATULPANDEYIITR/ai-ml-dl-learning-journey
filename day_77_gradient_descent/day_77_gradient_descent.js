"use strict";

/*
 * Gradient Descent Optimization Engine
 *
 * Implements batch, stochastic and mini-batch gradient descent
 * for linear regression using Node.js standard-library features.
 *
 * Features:
 * - Event-driven training notifications
 * - Asynchronous training interface
 * - Mini-batch construction
 * - Analytical gradient calculation
 * - Gradient clipping
 * - L2 regularization
 * - Convergence monitoring
 * - Evaluation and CSV reporting
 * - Deterministic random number generation
 * - Input validation and numerical safeguards
 *
 * Runtime: Node.js 18+
 */

const fs = require("node:fs/promises");
const { EventEmitter } = require("node:events");

const Optimizer = Object.freeze({
    BATCH: "batch",
    STOCHASTIC: "stochastic",
    MINI_BATCH: "mini_batch"
});

class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = "ValidationError";
    }
}

class NumericalError extends Error {
    constructor(message) {
        super(message);
        this.name = "NumericalError";
    }
}

function assertFinite(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new ValidationError(`${name} must be finite.`);
    }
}

function validateConfig(config) {
    if (!Object.values(Optimizer).includes(config.optimizer)) {
        throw new ValidationError("Unsupported optimizer.");
    }

    assertFinite(config.learningRate, "Learning rate");
    assertFinite(config.regularization, "Regularization");
    assertFinite(config.gradientClip, "Gradient clipping");

    if (config.learningRate <= 0) {
        throw new ValidationError("Learning rate must be positive.");
    }

    if (config.regularization < 0) {
        throw new ValidationError("Regularization cannot be negative.");
    }

    if (config.gradientClip <= 0) {
        throw new ValidationError("Gradient clipping must be positive.");
    }

    if (!Number.isInteger(config.epochs) || config.epochs <= 0) {
        throw new ValidationError("Epochs must be a positive integer.");
    }

    if (!Number.isInteger(config.batchSize) || config.batchSize <= 0) {
        throw new ValidationError("Batch size must be positive.");
    }

    if (!Number.isInteger(config.patience) || config.patience <= 0) {
        throw new ValidationError("Patience must be positive.");
    }
}

class RandomGenerator {
    constructor(seed = 42) {
        this.state = seed >>> 0;
    }

    next() {
        // Mulberry32: reproducible pseudo-random sequence.
        let value = this.state += 0x6D2B79F5;

        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);

        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    }

    normal() {
        // Box-Muller transformation.
        const u1 = Math.max(this.next(), Number.EPSILON);
        const u2 = this.next();

        return Math.sqrt(-2 * Math.log(u1)) *
            Math.cos(2 * Math.PI * u2);
    }

    shuffle(values) {
        const result = [...values];

        for (let i = result.length - 1; i > 0; i--) {
            const j = Math.floor(this.next() * (i + 1));
            [result[i], result[j]] = [result[j], result[i]];
        }

        return result;
    }
}

class Dataset {
    static generate(count, noise, seed = 42) {
        if (!Number.isInteger(count) || count < 2) {
            throw new ValidationError("Dataset needs at least two samples.");
        }

        assertFinite(noise, "Noise");

        if (noise < 0) {
            throw new ValidationError("Noise cannot be negative.");
        }

        const random = new RandomGenerator(seed);
        const samples = [];

        for (let i = 0; i < count; i++) {
            const features = [
                random.next() * 10 - 5,
                random.next() * 10 - 5
            ];

            const target =
                2.5 * features[0] -
                1.7 * features[1] +
                4 +
                noise * random.normal();

            samples.push(Object.freeze({
                features: Object.freeze(features),
                target
            }));
        }

        return Object.freeze(samples);
    }
}

class LinearRegression {
    constructor(dimensions) {
        if (!Number.isInteger(dimensions) || dimensions <= 0) {
            throw new ValidationError("Invalid model dimensions.");
        }

        this.weights = new Float64Array(dimensions);
        this.bias = 0;
    }

    predict(features) {
        if (features.length !== this.weights.length) {
            throw new ValidationError("Feature dimension mismatch.");
        }

        let prediction = this.bias;

        for (let i = 0; i < features.length; i++) {
            assertFinite(features[i], `Feature ${i}`);
            prediction += this.weights[i] * features[i];
        }

        return prediction;
    }

    loss(samples, regularization = 0) {
        if (samples.length === 0) {
            throw new ValidationError("Cannot calculate empty dataset loss.");
        }

        let squaredError = 0;

        for (const sample of samples) {
            const error = this.predict(sample.features) - sample.target;
            squaredError += error * error;
        }

        let penalty = 0;

        for (const weight of this.weights) {
            penalty += weight * weight;
        }

        return squaredError / (2 * samples.length) +
            regularization * penalty / 2;
    }

    gradients(samples, regularization) {
        if (samples.length === 0) {
            throw new ValidationError("Cannot calculate empty gradients.");
        }

        const weightGradients = new Float64Array(this.weights.length);
        let biasGradient = 0;

        for (const sample of samples) {
            const error = this.predict(sample.features) - sample.target;

            for (let i = 0; i < this.weights.length; i++) {
                weightGradients[i] += error * sample.features[i];
            }

            biasGradient += error;
        }

        for (let i = 0; i < weightGradients.length; i++) {
            weightGradients[i] =
                weightGradients[i] / samples.length +
                regularization * this.weights[i];
        }

        biasGradient /= samples.length;

        return { weightGradients, biasGradient };
    }

    update(gradients, learningRate, clippingThreshold) {
        let squaredNorm = gradients.biasGradient ** 2;

        for (const gradient of gradients.weightGradients) {
            squaredNorm += gradient * gradient;
        }

        const norm = Math.sqrt(squaredNorm);
        const scale = Math.min(
            1,
            clippingThreshold / Math.max(norm, 1e-15)
        );

        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] -=
                learningRate * gradients.weightGradients[i] * scale;
        }

        this.bias -= learningRate * gradients.biasGradient * scale;

        return norm;
    }
}

class OptimizerStrategy {
    static createBatches(samples, optimizer, batchSize, random) {
        const indices = Array.from(
            { length: samples.length },
            (_, index) => index
        );

        const shuffled = random.shuffle(indices);

        let size;

        switch (optimizer) {
            case Optimizer.BATCH:
                size = samples.length;
                break;

            case Optimizer.STOCHASTIC:
                size = 1;
                break;

            case Optimizer.MINI_BATCH:
                size = batchSize;
                break;

            default:
                throw new ValidationError("Unknown optimization strategy.");
        }

        const batches = [];

        for (let start = 0; start < shuffled.length; start += size) {
            batches.push(
                shuffled.slice(start, start + size)
                    .map(index => samples[index])
            );
        }

        return batches;
    }
}

class TrainingEngine extends EventEmitter {
    constructor(config) {
        super();

        validateConfig(config);

        this.config = Object.freeze({ ...config });
        this.random = new RandomGenerator(config.seed ?? 42);
    }

    async train(samples) {
        if (!Array.isArray(samples) || samples.length === 0) {
            throw new ValidationError("Training samples are required.");
        }

        const dimensions = samples[0].features.length;

        if (dimensions === 0) {
            throw new ValidationError("Features cannot be empty.");
        }

        for (const sample of samples) {
            if (!Array.isArray(sample.features) ||
                sample.features.length !== dimensions) {
                throw new ValidationError("Inconsistent feature dimensions.");
            }

            for (const value of sample.features) {
                assertFinite(value, "Feature");
            }

            assertFinite(sample.target, "Target");
        }

        const model = new LinearRegression(dimensions);
        const history = [];

        let bestLoss = Infinity;
        let staleEpochs = 0;
        let converged = false;

        for (let epoch = 1; epoch <= this.config.epochs; epoch++) {
            const batches = OptimizerStrategy.createBatches(
                samples,
                this.config.optimizer,
                this.config.batchSize,
                this.random
            );

            let totalGradientNorm = 0;

            for (const batch of batches) {
                const gradients = model.gradients(
                    batch,
                    this.config.regularization
                );

                totalGradientNorm += model.update(
                    gradients,
                    this.config.learningRate,
                    this.config.gradientClip
                );
            }

            const loss = model.loss(
                samples,
                this.config.regularization
            );

            if (!Number.isFinite(loss)) {
                throw new NumericalError(
                    `Non-finite loss at epoch ${epoch}.`
                );
            }

            const metrics = Object.freeze({
                epoch,
                loss,
                gradientNorm: totalGradientNorm / batches.length,
                batches: batches.length
            });

            history.push(metrics);

            // Emit events without coupling optimization to presentation.
            this.emit("epoch", metrics);

            if (bestLoss - loss > (this.config.tolerance ?? 1e-10)) {
                bestLoss = loss;
                staleEpochs = 0;
            } else {
                staleEpochs++;
            }

            if (staleEpochs >= this.config.patience) {
                converged = true;
                break;
            }

            // Yield to the Node.js event loop.
            if (epoch % 10 === 0) {
                await new Promise(resolve => setImmediate(resolve));
            }
        }

        const result = {
            optimizer: this.config.optimizer,
            weights: Array.from(model.weights),
            bias: model.bias,
            history,
            epochsCompleted: history.length,
            converged,
            predict: features => model.predict(features)
        };

        this.emit("complete", {
            optimizer: result.optimizer,
            epochsCompleted: result.epochsCompleted,
            finalLoss: history.at(-1).loss
        });

        return Object.freeze(result);
    }
}

function evaluate(model, samples) {
    if (!samples.length) {
        throw new ValidationError("Evaluation dataset cannot be empty.");
    }

    const actual = samples.map(sample => sample.target);
    const predictions = samples.map(
        sample => model.predict(sample.features)
    );

    const mean = actual.reduce((sum, value) => sum + value, 0) /
        actual.length;

    const residual = actual.reduce(
        (sum, value, index) =>
            sum + (value - predictions[index]) ** 2,
        0
    );

    const total = actual.reduce(
        (sum, value) => sum + (value - mean) ** 2,
        0
    );

    return Object.freeze({
        mse: residual / actual.length,
        rSquared: total === 0
            ? (residual === 0 ? 1 : 0)
            : 1 - residual / total
    });
}

async function exportCSV(results, filename) {
    const rows = [
        "optimizer,epoch,loss,gradient_norm,batches"
    ];

    for (const result of results) {
        for (const item of result.history) {
            rows.push([
                result.optimizer,
                item.epoch,
                item.loss.toFixed(10),
                item.gradientNorm.toFixed(10),
                item.batches
            ].join(","));
        }
    }

    // Write a temporary file before replacing the destination.
    const temporary = `${filename}.tmp`;

    await fs.writeFile(
        temporary,
        rows.join("\n") + "\n",
        { encoding: "utf8", flag: "w" }
    );

    await fs.rename(temporary, filename);
}

async function main() {
    console.log("GRADIENT DESCENT OPTIMIZATION ENGINE");
    console.log("=".repeat(65));

    const training = Dataset.generate(240, 0.5, 42);
    const testing = Dataset.generate(80, 0.5, 99);

    const configurations = [
        {
            optimizer: Optimizer.BATCH,
            learningRate: 0.01,
            epochs: 150,
            batchSize: 16,
            regularization: 0.001,
            gradientClip: 10,
            patience: 25,
            seed: 42
        },
        {
            optimizer: Optimizer.STOCHASTIC,
            learningRate: 0.003,
            epochs: 150,
            batchSize: 1,
            regularization: 0.001,
            gradientClip: 10,
            patience: 25,
            seed: 42
        },
        {
            optimizer: Optimizer.MINI_BATCH,
            learningRate: 0.01,
            epochs: 150,
            batchSize: 16,
            regularization: 0.001,
            gradientClip: 10,
            patience: 25,
            seed: 42
        }
    ];

    const results = [];

    for (const config of configurations) {
        const engine = new TrainingEngine(config);

        engine.on("epoch", metrics => {
            if (metrics.epoch === 1 || metrics.epoch % 50 === 0) {
                console.log(
                    `[${config.optimizer}] Epoch ${metrics.epoch}: ` +
                    `loss=${metrics.loss.toFixed(6)}`
                );
            }
        });

        engine.on("complete", event => {
            console.log(
                `Completed ${event.optimizer}: ` +
                `${event.epochsCompleted} epochs`
            );
        });

        const result = await engine.train(training);
        const metrics = evaluate(result, testing);

        results.push(result);

        console.log("\nModel:", result.optimizer);
        console.log("Weights:", result.weights.map(v => v.toFixed(4)));
        console.log("Bias:", result.bias.toFixed(4));
        console.log("Test MSE:", metrics.mse.toFixed(6));
        console.log("R-squared:", metrics.rSquared.toFixed(6));
        console.log("Converged:", result.converged);
    }

    await exportCSV(results, "gradient_descent_results.csv");

    console.log("\nCSV report written successfully.");

    try {
        new TrainingEngine({
            optimizer: Optimizer.BATCH,
            learningRate: -0.1,
            epochs: 100,
            batchSize: 16,
            regularization: 0,
            gradientClip: 10,
            patience: 10
        });
    } catch (error) {
        if (!(error instanceof ValidationError)) {
            throw error;
        }

        console.log("Invalid configuration correctly rejected.");
    }
}

main().catch(error => {
    console.error("Training failed:", error.message);
    process.exitCode = 1;
});
