"use strict";

/*
 * TRANSFER LEARNING
 * Event-driven transfer-learning experiment manager.
 *
 * Covers:
 * - Pretrained model registration
 * - Feature extraction
 * - Fine-tuning
 * - Layer freezing
 * - Dataset compatibility
 * - Learning-rate policies
 * - Experiment lifecycle
 * - Evaluation events
 * - Model promotion and deployment validation
 *
 * Runtime: Node.js 18+
 * Dependencies: None
 */

const { EventEmitter } = require("node:events");
const { randomUUID } = require("node:crypto");

// -----------------------------------------------------
// Domain definitions
// -----------------------------------------------------

const Strategy = Object.freeze({
    FEATURE_EXTRACTION: "FEATURE_EXTRACTION",
    FINE_TUNING: "FINE_TUNING"
});

const ExperimentState = Object.freeze({
    CREATED: "CREATED",
    VALIDATED: "VALIDATED",
    TRAINING: "TRAINING",
    EVALUATED: "EVALUATED",
    APPROVED: "APPROVED",
    REJECTED: "REJECTED",
    DEPLOYED: "DEPLOYED",
    FAILED: "FAILED"
});

class TransferLearningError extends Error {
    constructor(message) {
        super(message);
        this.name = "TransferLearningError";
    }
}

function requireValue(condition, message) {
    if (!condition) {
        throw new TransferLearningError(message);
    }
}

function validatePositiveNumber(value, name) {
    requireValue(
        Number.isFinite(value) && value > 0,
        `${name} must be a positive finite number.`
    );
}

// -----------------------------------------------------
// Dataset metadata
// -----------------------------------------------------

class DatasetDescriptor {
    constructor({
        name,
        samples,
        inputShape,
        classes,
        domain,
        version
    }) {
        requireValue(
            typeof name === "string" && name.trim().length > 0,
            "Dataset name is required."
        );

        requireValue(
            Number.isInteger(samples) && samples > 0,
            "Dataset must contain positive sample count."
        );

        requireValue(
            Array.isArray(inputShape) &&
            inputShape.length > 0 &&
            inputShape.every(
                value => Number.isInteger(value) && value > 0
            ),
            "Invalid dataset input shape."
        );

        requireValue(
            Number.isInteger(classes) && classes >= 2,
            "Classification requires at least two classes."
        );

        requireValue(
            typeof domain === "string" && domain.length > 0,
            "Dataset domain is required."
        );

        this.name = name;
        this.samples = samples;
        this.inputShape = Object.freeze([...inputShape]);
        this.classes = classes;
        this.domain = domain;
        this.version = version;
        Object.freeze(this);
    }
}

// -----------------------------------------------------
// Pretrained model
// -----------------------------------------------------

class PretrainedModel {
    constructor({
        id,
        name,
        architecture,
        inputShape,
        layers,
        sourceDomain,
        weightsVersion,
        trusted
    }) {
        requireValue(
            Array.isArray(layers) && layers.length > 0,
            "A model requires at least one layer."
        );

        this.id = id || randomUUID();
        this.name = name;
        this.architecture = architecture;
        this.inputShape = Object.freeze([...inputShape]);
        this.sourceDomain = sourceDomain;
        this.weightsVersion = weightsVersion;
        this.trusted = Boolean(trusted);

        this.layers = layers.map((layer, index) => {
            requireValue(
                Number.isInteger(layer.parameters) &&
                layer.parameters > 0,
                `Invalid parameter count in layer ${index}.`
            );

            return Object.freeze({
                name: layer.name,
                type: layer.type,
                parameters: layer.parameters,
                outputShape: Object.freeze([...layer.outputShape]),
                trainable: false
            });
        });

        Object.freeze(this.layers);
    }

    get parameterCount() {
        return this.layers.reduce(
            (sum, layer) => sum + layer.parameters,
            0
        );
    }
}

// -----------------------------------------------------
// Transfer policy
// -----------------------------------------------------

class TransferPolicy {
    constructor({
        strategy,
        frozenLayers,
        learningRate,
        headLearningRate,
        minimumSamples,
        maximumLearningRate
    }) {
        requireValue(
            Object.values(Strategy).includes(strategy),
            "Unknown transfer strategy."
        );

        validatePositiveNumber(learningRate, "Learning rate");
        validatePositiveNumber(headLearningRate, "Head learning rate");

        requireValue(
            Number.isInteger(frozenLayers) && frozenLayers >= 0,
            "Frozen layer count must be nonnegative."
        );

        requireValue(
            Number.isInteger(minimumSamples) && minimumSamples > 0,
            "Minimum sample count must be positive."
        );

        validatePositiveNumber(
            maximumLearningRate,
            "Maximum learning rate"
        );

        requireValue(
            learningRate <= maximumLearningRate &&
            headLearningRate <= maximumLearningRate,
            "Learning rate exceeds policy maximum."
        );

        if (strategy === Strategy.FEATURE_EXTRACTION) {
            requireValue(
                frozenLayers > 0,
                "Feature extraction requires a frozen backbone."
            );
        }

        this.strategy = strategy;
        this.frozenLayers = frozenLayers;
        this.learningRate = learningRate;
        this.headLearningRate = headLearningRate;
        this.minimumSamples = minimumSamples;
        this.maximumLearningRate = maximumLearningRate;

        Object.freeze(this);
    }

    resolveTrainableLayers(model) {
        const boundary = Math.min(
            this.frozenLayers,
            model.layers.length
        );

        return model.layers.map((layer, index) => ({
            name: layer.name,
            trainable:
                this.strategy === Strategy.FINE_TUNING &&
                index >= boundary
        }));
    }
}

// -----------------------------------------------------
// Experiment aggregate
// -----------------------------------------------------

class TransferExperiment {
    constructor(model, dataset, policy) {
        this.id = randomUUID();
        this.model = model;
        this.dataset = dataset;
        this.policy = policy;

        this.state = ExperimentState.CREATED;
        this.history = [];
        this.metrics = null;
        this.failureReason = null;

        this.transition(ExperimentState.CREATED);
    }

    transition(nextState) {
        const allowed = {
            CREATED: ["VALIDATED", "FAILED"],
            VALIDATED: ["TRAINING", "FAILED"],
            TRAINING: ["EVALUATED", "FAILED"],
            EVALUATED: ["APPROVED", "REJECTED", "FAILED"],
            APPROVED: ["DEPLOYED", "FAILED"],
            REJECTED: [],
            DEPLOYED: [],
            FAILED: []
        };

        requireValue(
            allowed[this.state].includes(nextState),
            `Invalid transition: ${this.state} -> ${nextState}`
        );

        this.state = nextState;

        this.history.push({
            state: nextState,
            timestamp: new Date().toISOString()
        });
    }

    fail(error) {
        this.failureReason = error.message;

        if (
            ![
                ExperimentState.FAILED,
                ExperimentState.REJECTED,
                ExperimentState.DEPLOYED
            ].includes(this.state)
        ) {
            this.transition(ExperimentState.FAILED);
        }
    }
}

// -----------------------------------------------------
// Transfer-learning service
// -----------------------------------------------------

class TransferLearningService extends EventEmitter {

    constructor() {
        super();
        this.models = new Map();
        this.experiments = new Map();
    }

    registerModel(model) {
        requireValue(
            model instanceof PretrainedModel,
            "Invalid pretrained model."
        );

        requireValue(
            model.trusted,
            "Untrusted pretrained model cannot be registered."
        );

        requireValue(
            !this.models.has(model.id),
            "Duplicate model identifier."
        );

        this.models.set(model.id, model);

        this.emit("model:registered", {
            modelId: model.id,
            modelName: model.name
        });

        return model.id;
    }

    validateCompatibility(model, dataset, policy) {
        requireValue(
            model.inputShape.length === dataset.inputShape.length &&
            model.inputShape.every(
                (dimension, index) =>
                    dimension === dataset.inputShape[index]
            ),
            "Source model and target dataset input shapes differ."
        );

        requireValue(
            dataset.samples >= policy.minimumSamples,
            "Target dataset has insufficient samples."
        );

        requireValue(
            policy.frozenLayers <= model.layers.length,
            "Frozen layer count exceeds model depth."
        );

        if (policy.strategy === Strategy.FEATURE_EXTRACTION) {
            requireValue(
                policy.frozenLayers === model.layers.length,
                "Feature extraction must freeze the entire backbone."
            );
        }

        return {
            compatible: true,
            backboneLayers: model.layers.length,
            targetClasses: dataset.classes,
            trainableLayers: policy.resolveTrainableLayers(model)
        };
    }

    createExperiment(modelId, dataset, policy) {
        const model = this.models.get(modelId);

        requireValue(model, "Pretrained model not found.");

        const experiment = new TransferExperiment(
            model,
            dataset,
            policy
        );

        this.experiments.set(experiment.id, experiment);

        this.emit("experiment:created", {
            experimentId: experiment.id
        });

        return experiment;
    }

    async execute(experiment, trainer) {
        requireValue(
            this.experiments.get(experiment.id) === experiment,
            "Experiment is not registered."
        );

        try {
            const compatibility = this.validateCompatibility(
                experiment.model,
                experiment.dataset,
                experiment.policy
            );

            experiment.transition(ExperimentState.VALIDATED);

            this.emit("experiment:validated", {
                experimentId: experiment.id,
                compatibility
            });

            experiment.transition(ExperimentState.TRAINING);

            this.emit("training:started", {
                experimentId: experiment.id,
                strategy: experiment.policy.strategy
            });

            // The trainer is injected so the workflow can use
            // different numerical backends without changing policy.
            const metrics = await trainer({
                model: experiment.model,
                dataset: experiment.dataset,
                policy: experiment.policy,
                signal: AbortSignal.timeout(30000)
            });

            requireValue(
                Number.isFinite(metrics.accuracy) &&
                metrics.accuracy >= 0 &&
                metrics.accuracy <= 1,
                "Invalid accuracy metric."
            );

            requireValue(
                Number.isFinite(metrics.loss) && metrics.loss >= 0,
                "Invalid loss metric."
            );

            experiment.metrics = Object.freeze({
                accuracy: metrics.accuracy,
                loss: metrics.loss,
                evaluatedSamples: metrics.evaluatedSamples
            });

            experiment.transition(ExperimentState.EVALUATED);

            this.emit("training:completed", {
                experimentId: experiment.id,
                metrics: experiment.metrics
            });

            return experiment.metrics;

        } catch (error) {
            experiment.fail(error);

            this.emit("experiment:failed", {
                experimentId: experiment.id,
                reason: error.message
            });

            throw error;
        }
    }

    approve(experiment, requirements) {
        requireValue(
            experiment.state === ExperimentState.EVALUATED,
            "Only evaluated experiments can be approved."
        );

        const valid = (
            experiment.metrics.accuracy >= requirements.minimumAccuracy &&
            experiment.metrics.loss <= requirements.maximumLoss &&
            experiment.metrics.evaluatedSamples >=
                experiment.dataset.samples * requirements.minimumCoverage
        );

        if (!valid) {
            experiment.transition(ExperimentState.REJECTED);

            this.emit("experiment:rejected", {
                experimentId: experiment.id
            });

            return false;
        }

        experiment.transition(ExperimentState.APPROVED);

        this.emit("experiment:approved", {
            experimentId: experiment.id
        });

        return true;
    }

    deploy(experiment) {
        requireValue(
            experiment.state === ExperimentState.APPROVED,
            "Deployment requires an approved experiment."
        );

        experiment.transition(ExperimentState.DEPLOYED);

        this.emit("model:deployed", {
            experimentId: experiment.id,
            modelId: experiment.model.id,
            datasetVersion: experiment.dataset.version
        });
    }

    report(experiment) {
        return {
            id: experiment.id,
            model: experiment.model.name,
            strategy: experiment.policy.strategy,
            sourceDomain: experiment.model.sourceDomain,
            targetDomain: experiment.dataset.domain,
            state: experiment.state,
            metrics: experiment.metrics,
            history: [...experiment.history]
        };
    }
}

// -----------------------------------------------------
// Example asynchronous training backend
// -----------------------------------------------------

async function exampleTrainer({ model, dataset, policy, signal }) {
    await new Promise((resolve, reject) => {
        const timeout = setTimeout(resolve, 100);

        signal.addEventListener("abort", () => {
            clearTimeout(timeout);
            reject(new Error("Training operation timed out."));
        }, { once: true });
    });

    // Deterministic demonstration metrics.
    // A production trainer would return measured validation metrics.
    const baseAccuracy = policy.strategy === Strategy.FINE_TUNING
        ? 0.91
        : 0.84;

    const domainPenalty =
        model.sourceDomain === dataset.domain ? 0 : 0.015;

    return {
        accuracy: Math.max(0, baseAccuracy - domainPenalty),
        loss: policy.strategy === Strategy.FINE_TUNING ? 0.19 : 0.31,
        evaluatedSamples: dataset.samples
    };
}

// -----------------------------------------------------
// Demonstration
// -----------------------------------------------------

async function main() {
    const service = new TransferLearningService();

    service.on("model:registered", event => {
        console.log("Registered:", event.modelName);
    });

    service.on("training:started", event => {
        console.log("Training started:", event.experimentId);
    });

    service.on("training:completed", event => {
        console.log("Evaluation metrics:", event.metrics);
    });

    service.on("model:deployed", event => {
        console.log("Deployment completed:", event.modelId);
    });

    const pretrainedModel = new PretrainedModel({
        name: "VisionBackbone-V1",
        architecture: "Convolutional Neural Network",
        inputShape: [224, 224, 3],
        sourceDomain: "General Image Classification",
        weightsVersion: "1.0.0",
        trusted: true,
        layers: [
            {
                name: "conv_block_1",
                type: "CONVOLUTION",
                parameters: 12000,
                outputShape: [112, 112, 32]
            },
            {
                name: "conv_block_2",
                type: "CONVOLUTION",
                parameters: 48000,
                outputShape: [56, 56, 64]
            },
            {
                name: "feature_block",
                type: "FEATURE_EXTRACTION",
                parameters: 96000,
                outputShape: [512]
            }
        ]
    });

    const modelId = service.registerModel(pretrainedModel);

    const dataset = new DatasetDescriptor({
        name: "IndustrialDefectImages",
        samples: 5000,
        inputShape: [224, 224, 3],
        classes: 4,
        domain: "Industrial Inspection",
        version: "2026.01"
    });

    const policy = new TransferPolicy({
        strategy: Strategy.FINE_TUNING,
        frozenLayers: 1,
        learningRate: 0.0001,
        headLearningRate: 0.001,
        minimumSamples: 500,
        maximumLearningRate: 0.01
    });

    const experiment = service.createExperiment(
        modelId,
        dataset,
        policy
    );

    await service.execute(experiment, exampleTrainer);

    const approved = service.approve(experiment, {
        minimumAccuracy: 0.85,
        maximumLoss: 0.25,
        minimumCoverage: 0.95
    });

    if (approved) {
        service.deploy(experiment);
    }

    console.log("\nEXPERIMENT REPORT");
    console.log(JSON.stringify(service.report(experiment), null, 2));
}

// -----------------------------------------------------
// Execution and failure handling
// -----------------------------------------------------

main().catch(error => {
    console.error("Transfer-learning workflow failed:", error.message);
    process.exitCode = 1;
});
