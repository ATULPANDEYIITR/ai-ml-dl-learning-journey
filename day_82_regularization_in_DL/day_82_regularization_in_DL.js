"use strict";

/*
 * Regularization in Deep Learning
 *
 * Event-driven training experiment controller.
 * Implements:
 * 1. Inverted dropout masks.
 * 2. Decoupled weight decay.
 * 3. Early stopping with best-checkpoint restoration.
 * 4. Label-preserving data augmentation.
 * 5. Experiment lifecycle and policy evaluation.
 *
 * Runtime: Node.js 18+
 */

const { EventEmitter } = require("node:events");
const fs = require("node:fs/promises");

class ValidationError extends Error {}

class SeededRandom {
    constructor(seed = 42) {
        this.state = seed >>> 0;
    }

    next() {
        this.state = (1664525 * this.state + 1013904223) >>> 0;
        return this.state / 4294967296;
    }

    normal() {
        const u1 = Math.max(this.next(), 1e-12);
        const u2 = this.next();

        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    shuffle(array) {
        const result = [...array];

        for (let i = result.length - 1; i > 0; i--) {
            const j = Math.floor(this.next() * (i + 1));
            [result[i], result[j]] = [result[j], result[i]];
        }

        return result;
    }
}

class TrainingPolicy {
    constructor({
        learningRate = 0.03,
        dropoutRate = 0.2,
        weightDecay = 0.001,
        patience = 12,
        minDelta = 0.00001,
        augmentationNoise = 0.04,
        maxEpochs = 100
    } = {}) {
        if (!(learningRate > 0)) {
            throw new ValidationError("Learning rate must be positive.");
        }

        if (dropoutRate < 0 || dropoutRate >= 1) {
            throw new ValidationError("Invalid dropout probability.");
        }

        if (weightDecay < 0 || patience < 1 || maxEpochs < 1) {
            throw new ValidationError("Invalid training policy.");
        }

        if (augmentationNoise < 0 || minDelta < 0) {
            throw new ValidationError("Invalid augmentation or threshold.");
        }

        this.learningRate = learningRate;
        this.dropoutRate = dropoutRate;
        this.weightDecay = weightDecay;
        this.patience = patience;
        this.minDelta = minDelta;
        this.augmentationNoise = augmentationNoise;
        this.maxEpochs = maxEpochs;

        Object.freeze(this);
    }
}

class RegularizationEngine {
    constructor(policy, seed = 42) {
        this.policy = policy;
        this.rng = new SeededRandom(seed);
    }

    createDropoutMask(size, training = true) {
        if (!Number.isInteger(size) || size <= 0) {
            throw new ValidationError("Invalid activation size.");
        }

        if (!training || this.policy.dropoutRate === 0) {
            return Array(size).fill(1);
        }

        const keepProbability = 1 - this.policy.dropoutRate;
        const scale = 1 / keepProbability;

        return Array.from(
            { length: size },
            () => this.rng.next() < keepProbability ? scale : 0
        );
    }

    applyDropout(activations, training = true) {
        if (!Array.isArray(activations) ||
            !activations.every(Number.isFinite)) {
            throw new ValidationError("Activations must be finite numbers.");
        }

        const mask = this.createDropoutMask(
            activations.length,
            training
        );

        return activations.map((value, i) => value * mask[i]);
    }

    augment(features) {
        if (!Array.isArray(features) ||
            !features.every(Number.isFinite)) {
            throw new ValidationError("Invalid input features.");
        }

        return features.map(
            value => value + this.rng.normal() *
                this.policy.augmentationNoise
        );
    }

    applyWeightDecay(parameters, learningRate) {
        if (!Array.isArray(parameters) ||
            !parameters.every(Number.isFinite)) {
            throw new ValidationError("Invalid model parameters.");
        }

        if (!(learningRate > 0)) {
            throw new ValidationError("Invalid learning rate.");
        }

        // Decoupled weight decay, as used by AdamW-style optimization.
        const factor = 1 - learningRate * this.policy.weightDecay;

        if (factor < 0) {
            throw new ValidationError("Weight decay factor is negative.");
        }

        return parameters.map(value => value * factor);
    }
}

class EarlyStopping {
    constructor(patience, minDelta = 0) {
        if (!Number.isInteger(patience) || patience < 1) {
            throw new ValidationError("Patience must be positive.");
        }

        this.patience = patience;
        this.minDelta = minDelta;
        this.bestLoss = Infinity;
        this.bestEpoch = 0;
        this.staleEpochs = 0;
        this.checkpoint = null;
        this.stopped = false;
    }

    observe(epoch, validationLoss, modelState) {
        if (!Number.isFinite(validationLoss) || validationLoss < 0) {
            throw new ValidationError("Validation loss must be finite and nonnegative.");
        }

        if (this.stopped) {
            throw new Error("Early stopping has already triggered.");
        }

        if (validationLoss < this.bestLoss - this.minDelta) {
            this.bestLoss = validationLoss;
            this.bestEpoch = epoch;
            this.checkpoint = structuredClone(modelState);
            this.staleEpochs = 0;
        } else {
            this.staleEpochs++;
        }

        this.stopped = this.staleEpochs >= this.patience;

        return {
            improved: this.bestEpoch === epoch,
            stop: this.stopped
        };
    }

    restore() {
        if (this.checkpoint === null) {
            throw new Error("No valid checkpoint exists.");
        }

        return structuredClone(this.checkpoint);
    }
}

class Experiment extends EventEmitter {
    constructor(name, policy, seed = 42) {
        super();

        if (!name || typeof name !== "string") {
            throw new ValidationError("Experiment name is required.");
        }

        this.name = name;
        this.policy = policy;
        this.engine = new RegularizationEngine(policy, seed);
        this.stopping = new EarlyStopping(
            policy.patience,
            policy.minDelta
        );

        this.state = "CREATED";
        this.history = [];
    }

    transition(nextState) {
        const allowed = {
            CREATED: ["RUNNING", "FAILED"],
            RUNNING: ["COMPLETED", "STOPPED", "FAILED"],
            COMPLETED: [],
            STOPPED: [],
            FAILED: []
        };

        if (!allowed[this.state].includes(nextState)) {
            throw new Error(
                `Invalid state transition: ${this.state} -> ${nextState}`
            );
        }

        this.state = nextState;
        this.emit("stateChanged", {
            experiment: this.name,
            state: nextState
        });
    }

    async run(validationLossProvider) {
        if (typeof validationLossProvider !== "function") {
            throw new ValidationError("A validation loss provider is required.");
        }

        this.transition("RUNNING");

        // A small parameter vector demonstrates regularization operations.
        let parameters = [0.8, -0.5, 0.3, 0.9];

        try {
            for (let epoch = 1; epoch <= this.policy.maxEpochs; epoch++) {
                const augmented = this.engine.augment([0.5, 0.8, -0.2]);
                const dropped = this.engine.applyDropout(augmented, true);

                // Simulated optimization update.
                parameters = parameters.map(
                    (weight, i) => weight -
                        this.policy.learningRate *
                        (weight * 0.02 + dropped[i % dropped.length] * 0.001)
                );

                parameters = this.engine.applyWeightDecay(
                    parameters,
                    this.policy.learningRate
                );

                const validationLoss = await validationLossProvider(epoch);

                const result = this.stopping.observe(
                    epoch,
                    validationLoss,
                    parameters
                );

                const record = Object.freeze({
                    epoch,
                    validationLoss,
                    improved: result.improved
                });

                this.history.push(record);
                this.emit("epochCompleted", record);

                if (result.stop) {
                    this.transition("STOPPED");
                    break;
                }

                if (epoch === this.policy.maxEpochs) {
                    this.transition("COMPLETED");
                }

                // Yield to the event loop for asynchronous monitoring.
                await new Promise(resolve => setImmediate(resolve));
            }

            parameters = this.stopping.restore();

            const result = {
                name: this.name,
                state: this.state,
                bestEpoch: this.stopping.bestEpoch,
                bestValidationLoss: this.stopping.bestLoss,
                restoredParameters: parameters,
                history: this.history
            };

            this.emit("experimentCompleted", result);
            return result;

        } catch (error) {
            if (this.state === "RUNNING") {
                this.transition("FAILED");
            }

            this.emit("experimentFailed", error);
            throw error;
        }
    }
}

function createValidationSequence(bestLoss, deteriorationStart, deteriorationRate) {
    return async epoch => {
        const noise = Math.sin(epoch * 1.7) * 0.001;

        if (epoch <= deteriorationStart) {
            return bestLoss + 1 / (epoch + 2) + noise;
        }

        return (
            bestLoss +
            deteriorationRate * (epoch - deteriorationStart) +
            noise
        );
    };
}

async function saveReport(report, filename = "regularization-report.json") {
    // Exclusive creation avoids silently overwriting an existing report.
    const data = JSON.stringify(report, null, 2);

    await fs.writeFile(filename, data, {
        encoding: "utf8",
        flag: "wx",
        mode: 0o600
    });
}

async function main() {
    const policies = new Map([
        ["Dropout", new TrainingPolicy({
            dropoutRate: 0.3,
            weightDecay: 0,
            augmentationNoise: 0,
            patience: 8
        })],

        ["Weight Decay", new TrainingPolicy({
            dropoutRate: 0,
            weightDecay: 0.02,
            augmentationNoise: 0,
            patience: 8
        })],

        ["Combined", new TrainingPolicy({
            dropoutRate: 0.2,
            weightDecay: 0.01,
            augmentationNoise: 0.05,
            patience: 8
        })]
    ]);

    const reports = [];

    for (const [name, policy] of policies) {
        const experiment = new Experiment(name, policy);

        experiment.on("stateChanged", event => {
            console.log(`[${event.experiment}] ${event.state}`);
        });

        experiment.on("epochCompleted", event => {
            if (event.epoch % 10 === 0) {
                console.log(
                    `Epoch ${event.epoch}: ${event.validationLoss.toFixed(5)}`
                );
            }
        });

        const report = await experiment.run(
            createValidationSequence(0.15, 12, 0.006)
        );

        reports.push(report);
    }

    reports.sort(
        (a, b) => a.bestValidationLoss - b.bestValidationLoss
    );

    console.log("\nExperiment ranking:");

    for (const report of reports) {
        console.log(
            `${report.name}: loss=${report.bestValidationLoss.toFixed(5)}, ` +
            `bestEpoch=${report.bestEpoch}, state=${report.state}`
        );
    }

    try {
        await saveReport(reports);
        console.log("Experiment report saved.");
    } catch (error) {
        if (error.code === "EEXIST") {
            console.error("Report already exists. No file was overwritten.");
        } else {
            throw error;
        }
    }
}

if (require.main === module) {
    main().catch(error => {
        console.error("Training experiment failed:", error.message);
        process.exitCode = 1;
    });
}

module.exports = {
    TrainingPolicy,
    RegularizationEngine,
    EarlyStopping,
    Experiment,
    SeededRandom
};
