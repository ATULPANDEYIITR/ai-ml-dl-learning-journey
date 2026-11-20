"use strict";

/*
 * Deep Learning Training Engine
 *
 * Node.js implementation of mini-batch gradient descent with:
 * - Event-driven epoch monitoring
 * - Adaptive learning-rate scheduling
 * - Validation and early stopping
 * - Atomic checkpoint persistence
 * - Checkpoint recovery
 * - Training cancellation and failure handling
 *
 * Run: node deep_learning_training.js
 */

const fs = require("node:fs/promises");
const path = require("node:path");
const { randomUUID } = require("node:crypto");
const { EventEmitter } = require("node:events");

class TrainingError extends Error {}
class ConfigurationError extends TrainingError {}
class CheckpointError extends TrainingError {}

const State = Object.freeze({
    INITIALIZED: "initialized",
    RUNNING: "running",
    COMPLETED: "completed",
    EARLY_STOPPED: "early_stopped",
    CANCELLED: "cancelled",
    FAILED: "failed"
});

function assertFinite(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new ConfigurationError(`${name} must be finite.`);
    }
}

function validateConfig(config) {
    if (!Number.isInteger(config.epochs) || config.epochs < 1) {
        throw new ConfigurationError("Epochs must be a positive integer.");
    }

    if (!Number.isInteger(config.batchSize) || config.batchSize < 1) {
        throw new ConfigurationError("Batch size must be positive.");
    }

    assertFinite(config.learningRate, "Learning rate");
    assertFinite(config.minLearningRate, "Minimum learning rate");
    assertFinite(config.decay, "Decay");
    assertFinite(config.minDelta, "Minimum delta");

    if (config.learningRate <= 0 || config.minLearningRate <= 0) {
        throw new ConfigurationError("Learning rates must be positive.");
    }

    if (config.decay <= 0 || config.decay > 1) {
        throw new ConfigurationError("Decay must be in (0, 1].");
    }

    if (!Number.isInteger(config.patience) || config.patience < 1) {
        throw new ConfigurationError("Patience must be positive.");
    }

    if (!Number.isInteger(config.checkpointEvery) ||
        config.checkpointEvery < 1) {
        throw new ConfigurationError("Invalid checkpoint interval.");
    }

    if (typeof config.checkpointDirectory !== "string" ||
        config.checkpointDirectory.length === 0) {
        throw new ConfigurationError("Checkpoint directory is required.");
    }
}

class Dataset {
    constructor(features, targets) {
        if (!Array.isArray(features) || !Array.isArray(targets) ||
            features.length === 0 || features.length !== targets.length) {
            throw new ConfigurationError("Invalid dataset dimensions.");
        }

        const dimension = features[0].length;

        if (!Number.isInteger(dimension) || dimension < 1) {
            throw new ConfigurationError("At least one feature is required.");
        }

        features.forEach((row, index) => {
            if (!Array.isArray(row) || row.length !== dimension) {
                throw new ConfigurationError(`Invalid feature row ${index}.`);
            }

            row.forEach((value) => assertFinite(value, "Feature"));
            assertFinite(targets[index], "Target");
        });

        this.features = features.map(row => [...row]);
        this.targets = [...targets];
        this.featureCount = dimension;
    }

    get size() {
        return this.targets.length;
    }
}

class LinearModel {
    constructor(featureCount, seed = 42) {
        if (!Number.isInteger(featureCount) || featureCount < 1) {
            throw new ConfigurationError("Invalid feature count.");
        }

        let state = seed >>> 0;

        const random = () => {
            state = (1664525 * state + 1013904223) >>> 0;
            return state / 4294967296;
        };

        this.weights = Array.from(
            { length: featureCount },
            () => (random() - 0.5) * 0.2
        );

        this.bias = 0;
    }

    predict(features) {
        if (features.length !== this.weights.length) {
            throw new TrainingError("Model feature dimension mismatch.");
        }

        return features.reduce(
            (sum, value, index) => sum + value * this.weights[index],
            this.bias
        );
    }

    clone() {
        const copy = Object.create(LinearModel.prototype);
        copy.weights = [...this.weights];
        copy.bias = this.bias;
        return copy;
    }

    toJSON() {
        return {
            weights: this.weights,
            bias: this.bias
        };
    }
}

class LearningRateScheduler {
    constructor(initialRate, decay, minimum) {
        this.initialRate = initialRate;
        this.decay = decay;
        this.minimum = minimum;
    }

    at(epoch) {
        return Math.max(
            this.minimum,
            this.initialRate * Math.pow(this.decay, epoch - 1)
        );
    }
}

class CheckpointStore {
    constructor(directory) {
        this.directory = path.resolve(directory);
    }

    async initialize() {
        await fs.mkdir(this.directory, { recursive: true });
    }

    async save(checkpoint) {
        const filename = `epoch_${String(checkpoint.epoch).padStart(5, "0")}.json`;
        const destination = path.join(this.directory, filename);
        const temporary = path.join(
            this.directory,
            `.${randomUUID()}.tmp`
        );

        try {
            await fs.writeFile(
                temporary,
                JSON.stringify(checkpoint),
                { encoding: "utf8", flag: "wx", mode: 0o600 }
            );

            await fs.rename(temporary, destination);
            return destination;
        } catch (error) {
            await fs.rm(temporary, { force: true }).catch(() => {});
            throw new CheckpointError(`Checkpoint write failed: ${error.message}`);
        }
    }

    async load(filename) {
        const resolved = path.resolve(filename);

        if (path.dirname(resolved) !== this.directory) {
            throw new CheckpointError("Checkpoint path is outside storage directory.");
        }

        try {
            const content = await fs.readFile(resolved, "utf8");
            const data = JSON.parse(content);

            if (data.version !== 1 ||
                !Number.isInteger(data.epoch) ||
                data.epoch < 0 ||
                !Array.isArray(data.model?.weights) ||
                data.model.weights.length === 0 ||
                !data.model.weights.every(Number.isFinite) ||
                !Number.isFinite(data.model.bias) ||
                !Number.isFinite(data.learningRate) ||
                data.learningRate <= 0 ||
                !Number.isFinite(data.bestLoss)) {
                throw new CheckpointError("Checkpoint integrity validation failed.");
            }

            return data;
        } catch (error) {
            if (error instanceof CheckpointError) throw error;
            throw new CheckpointError(`Checkpoint recovery failed: ${error.message}`);
        }
    }
}

class TrainingEngine extends EventEmitter {
    constructor(config, trainingData, validationData, model) {
        super();

        validateConfig(config);

        if (trainingData.featureCount !== validationData.featureCount ||
            trainingData.featureCount !== model.weights.length) {
            throw new ConfigurationError("Feature dimensions do not match.");
        }

        this.config = Object.freeze({ ...config });
        this.trainingData = trainingData;
        this.validationData = validationData;
        this.model = model;

        this.scheduler = new LearningRateScheduler(
            config.learningRate,
            config.decay,
            config.minLearningRate
        );

        this.store = new CheckpointStore(config.checkpointDirectory);

        this.state = State.INITIALIZED;
        this.history = [];
        this.bestLoss = Infinity;
        this.bestEpoch = 0;
        this.bestModel = null;
        this.staleEpochs = 0;
        this.cancelRequested = false;
        this.startEpoch = 1;
    }

    loss(dataset) {
        let total = 0;

        for (let i = 0; i < dataset.size; i++) {
            const error =
                this.model.predict(dataset.features[i]) -
                dataset.targets[i];

            total += error * error;
        }

        return total / dataset.size;
    }

    trainBatch(indices, learningRate) {
        const gradients = new Array(this.model.weights.length).fill(0);
        let biasGradient = 0;

        for (const index of indices) {
            const features = this.trainingData.features[index];
            const error =
                this.model.predict(features) -
                this.trainingData.targets[index];

            features.forEach((value, j) => {
                gradients[j] += 2 * error * value / indices.length;
            });

            biasGradient += 2 * error / indices.length;
        }

        this.model.weights = this.model.weights.map(
            (weight, j) => weight - learningRate * gradients[j]
        );

        this.model.bias -= learningRate * biasGradient;

        if (!this.model.weights.every(Number.isFinite) ||
            !Number.isFinite(this.model.bias)) {
            throw new TrainingError("Numerical instability detected.");
        }
    }

    async saveCheckpoint(epoch, learningRate) {
        return this.store.save({
            version: 1,
            epoch,
            learningRate,
            bestLoss: this.bestLoss,
            bestEpoch: this.bestEpoch,
            model: this.model.toJSON()
        });
    }

    async restore(filename) {
        if (this.state !== State.INITIALIZED) {
            throw new TrainingError("Restore is only allowed before training.");
        }

        const checkpoint = await this.store.load(filename);

        if (checkpoint.model.weights.length !== this.model.weights.length) {
            throw new CheckpointError("Checkpoint feature count mismatch.");
        }

        this.model.weights = [...checkpoint.model.weights];
        this.model.bias = checkpoint.model.bias;
        this.bestLoss = checkpoint.bestLoss;
        this.bestEpoch = checkpoint.bestEpoch;
        this.bestModel = this.model.clone();
        this.startEpoch = checkpoint.epoch + 1;

        this.emit("restored", { epoch: checkpoint.epoch });
    }

    cancel() {
        if (this.state === State.RUNNING) {
            this.cancelRequested = true;
            this.emit("cancellationRequested");
        }
    }

    async train() {
        if (this.state !== State.INITIALIZED) {
            throw new TrainingError("Training can only start once.");
        }

        this.state = State.RUNNING;
        await this.store.initialize();

        try {
            for (let epoch = this.startEpoch;
                 epoch <= this.config.epochs;
                 epoch++) {

                if (this.cancelRequested) {
                    this.state = State.CANCELLED;
                    break;
                }

                const learningRate = this.scheduler.at(epoch);
                const indices = Array.from(
                    { length: this.trainingData.size },
                    (_, i) => i
                );

                // Fisher-Yates shuffle gives every sample an equal opportunity
                // to appear at different batch positions.
                for (let i = indices.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [indices[i], indices[j]] = [indices[j], indices[i]];
                }

                let batches = 0;

                for (let start = 0;
                     start < indices.length;
                     start += this.config.batchSize) {

                    if (this.cancelRequested) break;

                    this.trainBatch(
                        indices.slice(start, start + this.config.batchSize),
                        learningRate
                    );

                    batches++;
                }

                if (this.cancelRequested) {
                    this.state = State.CANCELLED;
                    break;
                }

                const trainingLoss = this.loss(this.trainingData);
                const validationLoss = this.loss(this.validationData);

                const improved =
                    validationLoss < this.bestLoss - this.config.minDelta;

                if (improved) {
                    this.bestLoss = validationLoss;
                    this.bestEpoch = epoch;
                    this.bestModel = this.model.clone();
                    this.staleEpochs = 0;
                } else {
                    this.staleEpochs++;
                }

                const metrics = Object.freeze({
                    epoch,
                    batches,
                    learningRate,
                    trainingLoss,
                    validationLoss
                });

                this.history.push(metrics);

                this.emit("epochCompleted", metrics);

                if (epoch % this.config.checkpointEvery === 0 || improved) {
                    const checkpointPath =
                        await this.saveCheckpoint(epoch, learningRate);

                    this.emit("checkpointSaved", {
                        epoch,
                        path: checkpointPath
                    });
                }

                if (this.staleEpochs >= this.config.patience) {
                    this.state = State.EARLY_STOPPED;
                    break;
                }
            }

            if (this.bestModel) {
                this.model = this.bestModel.clone();
            }

            if (this.state === State.RUNNING) {
                this.state = State.COMPLETED;
            }

            this.emit("trainingCompleted", {
                state: this.state,
                bestEpoch: this.bestEpoch,
                bestLoss: this.bestLoss
            });

            return this.history;
        } catch (error) {
            this.state = State.FAILED;
            this.emit("trainingFailed", error);
            throw error;
        }
    }
}

function generateDataset(size, seed = 42) {
    let state = seed >>> 0;

    const random = () => {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    };

    const features = [];
    const targets = [];

    for (let i = 0; i < size; i++) {
        const x1 = random() * 4 - 2;
        const x2 = random() * 4 - 2;

        features.push([x1, x2]);
        targets.push(
            3 * x1 - 2 * x2 + 1.5 + (random() - 0.5) * 0.1
        );
    }

    const split = Math.floor(size * 0.8);

    return {
        training: new Dataset(features.slice(0, split), targets.slice(0, split)),
        validation: new Dataset(features.slice(split), targets.slice(split))
    };
}

async function main() {
    const config = {
        epochs: 40,
        batchSize: 16,
        learningRate: 0.05,
        minLearningRate: 0.00001,
        decay: 0.98,
        patience: 8,
        minDelta: 1e-8,
        checkpointEvery: 5,
        checkpointDirectory: "./js_checkpoints"
    };

    const datasets = generateDataset(200);

    const model = new LinearModel(
        datasets.training.featureCount,
        42
    );

    const engine = new TrainingEngine(
        config,
        datasets.training,
        datasets.validation,
        model
    );

    engine.on("epochCompleted", metrics => {
        console.log(
            `Epoch ${metrics.epoch.toString().padStart(3, "0")} | ` +
            `batches=${metrics.batches} | ` +
            `lr=${metrics.learningRate.toFixed(6)} | ` +
            `train=${metrics.trainingLoss.toFixed(6)} | ` +
            `validation=${metrics.validationLoss.toFixed(6)}`
        );
    });

    engine.on("checkpointSaved", event => {
        console.log(`Checkpoint saved: ${event.path}`);
    });

    engine.on("trainingCompleted", event => {
        console.log("\nTraining state:", event.state);
        console.log("Best epoch:", event.bestEpoch);
        console.log("Best validation loss:", event.bestLoss.toFixed(6));
    });

    await engine.train();

    console.log("Learned weights:", engine.model.weights);
    console.log("Learned bias:", engine.model.bias);
}

if (require.main === module) {
    main().catch(error => {
        console.error("Training failed:", error.message);
        process.exitCode = 1;
    });
}

module.exports = {
    Dataset,
    LinearModel,
    LearningRateScheduler,
    CheckpointStore,
    TrainingEngine,
    validateConfig,
    generateDataset
};
