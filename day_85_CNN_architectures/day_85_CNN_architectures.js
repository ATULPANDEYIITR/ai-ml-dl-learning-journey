"use strict";

/*
 * CNN Architecture Engineering Registry
 *
 * Runtime: Node.js 18+
 *
 * Execute:
 *   node cnn_architectures.js
 *
 * The registry models architectural composition, tensor dimensions,
 * computational estimates, validation and experiment selection.
 *
 * It intentionally uses JavaScript's event-driven architecture rather
 * than duplicating the neural-network tensor implementation.
 */

const { EventEmitter } = require("node:events");
const crypto = require("node:crypto");
const assert = require("node:assert/strict");

const ArchitectureType = Object.freeze({
    LENET: "LENET",
    ALEXNET: "ALEXNET",
    VGG16: "VGG16",
    INCEPTION: "INCEPTION"
});

const Activation = Object.freeze({
    RELU: "relu",
    TANH: "tanh",
    SOFTMAX: "softmax"
});

class ArchitectureError extends Error {
    constructor(message) {
        super(message);
        this.name = "ArchitectureError";
    }
}

class TensorShape {
    constructor(channels, height, width) {
        for (const [name, value] of Object.entries({
            channels,
            height,
            width
        })) {
            if (!Number.isSafeInteger(value) || value <= 0) {
                throw new ArchitectureError(
                    `${name} must be a positive integer`
                );
            }
        }

        this.channels = channels;
        this.height = height;
        this.width = width;
        Object.freeze(this);
    }

    toString() {
        return `${this.channels}x${this.height}x${this.width}`;
    }
}

function convolutionOutput(input, kernel, stride = 1, padding = 0) {
    const output = Math.floor(
        (input + 2 * padding - kernel) / stride
    ) + 1;

    if (output <= 0) {
        throw new ArchitectureError(
            "Convolution produces an invalid output dimension"
        );
    }

    return output;
}

function convolutionParameters(inputChannels, outputChannels, kernel) {
    return (
        inputChannels * outputChannels * kernel * kernel +
        outputChannels
    );
}

function denseParameters(inputFeatures, outputFeatures) {
    return inputFeatures * outputFeatures + outputFeatures;
}

class Layer {
    constructor({
        name,
        type,
        inputChannels,
        outputChannels,
        kernel = 1,
        stride = 1,
        padding = 0,
        activation = Activation.RELU
    }) {
        if (!name || !type) {
            throw new ArchitectureError("Layer name and type are required");
        }

        this.name = name;
        this.type = type;
        this.inputChannels = inputChannels;
        this.outputChannels = outputChannels;
        this.kernel = kernel;
        this.stride = stride;
        this.padding = padding;
        this.activation = activation;
        Object.freeze(this);
    }
}

class ArchitectureDefinition {
    constructor({
        type,
        name,
        year,
        minimumInput,
        channels,
        parametersDescription,
        innovations,
        layers
    }) {
        this.type = type;
        this.name = name;
        this.year = year;
        this.minimumInput = minimumInput;
        this.channels = channels;
        this.parametersDescription = parametersDescription;
        this.innovations = Object.freeze([...innovations]);
        this.layers = Object.freeze([...layers]);
        Object.freeze(this);
    }
}

function makeDefinitions() {
    return new Map([
        [
            ArchitectureType.LENET,
            new ArchitectureDefinition({
                type: ArchitectureType.LENET,
                name: "LeNet-5",
                year: 1998,
                minimumInput: 32,
                channels: 1,
                parametersDescription: "Convolution and subsampling",
                innovations: [
                    "Small convolutional feature extractors",
                    "Average pooling",
                    "Tanh activations"
                ],
                layers: [
                    new Layer({
                        name: "conv1",
                        type: "conv",
                        inputChannels: 1,
                        outputChannels: 6,
                        kernel: 5
                    }),
                    new Layer({
                        name: "conv2",
                        type: "conv",
                        inputChannels: 6,
                        outputChannels: 16,
                        kernel: 5
                    }),
                    new Layer({
                        name: "conv3",
                        type: "conv",
                        inputChannels: 16,
                        outputChannels: 120,
                        kernel: 5
                    })
                ]
            })
        ],
        [
            ArchitectureType.ALEXNET,
            new ArchitectureDefinition({
                type: ArchitectureType.ALEXNET,
                name: "AlexNet",
                year: 2012,
                minimumInput: 224,
                channels: 3,
                parametersDescription: "Large dense classifier",
                innovations: [
                    "ReLU activation",
                    "Dropout",
                    "GPU-based training",
                    "Data augmentation"
                ],
                layers: [
                    new Layer({
                        name: "conv1",
                        type: "conv",
                        inputChannels: 3,
                        outputChannels: 64,
                        kernel: 11,
                        stride: 4,
                        padding: 2
                    }),
                    new Layer({
                        name: "conv2",
                        type: "conv",
                        inputChannels: 64,
                        outputChannels: 192,
                        kernel: 5,
                        padding: 2
                    }),
                    new Layer({
                        name: "conv3",
                        type: "conv",
                        inputChannels: 192,
                        outputChannels: 384,
                        kernel: 3,
                        padding: 1
                    }),
                    new Layer({
                        name: "conv4",
                        type: "conv",
                        inputChannels: 384,
                        outputChannels: 256,
                        kernel: 3,
                        padding: 1
                    }),
                    new Layer({
                        name: "conv5",
                        type: "conv",
                        inputChannels: 256,
                        outputChannels: 256,
                        kernel: 3,
                        padding: 1
                    })
                ]
            })
        ],
        [
            ArchitectureType.VGG16,
            new ArchitectureDefinition({
                type: ArchitectureType.VGG16,
                name: "VGG16",
                year: 2014,
                minimumInput: 224,
                channels: 3,
                parametersDescription: "Thirteen convolutional layers",
                innovations: [
                    "Repeated 3x3 filters",
                    "Five pooling stages",
                    "Uniform network design"
                ],
                layers: [
                    new Layer({
                        name: "block1_conv1",
                        type: "conv",
                        inputChannels: 3,
                        outputChannels: 64,
                        kernel: 3,
                        padding: 1
                    }),
                    new Layer({
                        name: "block1_conv2",
                        type: "conv",
                        inputChannels: 64,
                        outputChannels: 64,
                        kernel: 3,
                        padding: 1
                    }),
                    new Layer({
                        name: "block2_conv1",
                        type: "conv",
                        inputChannels: 64,
                        outputChannels: 128,
                        kernel: 3,
                        padding: 1
                    }),
                    new Layer({
                        name: "block2_conv2",
                        type: "conv",
                        inputChannels: 128,
                        outputChannels: 128,
                        kernel: 3,
                        padding: 1
                    })
                ]
            })
        ],
        [
            ArchitectureType.INCEPTION,
            new ArchitectureDefinition({
                type: ArchitectureType.INCEPTION,
                name: "Inception v1",
                year: 2014,
                minimumInput: 224,
                channels: 3,
                parametersDescription: "Parallel convolution branches",
                innovations: [
                    "1x1 channel reduction",
                    "Parallel 1x1, 3x3 and 5x5 branches",
                    "Pooling projection",
                    "Feature concatenation"
                ],
                layers: [
                    new Layer({
                        name: "branch1",
                        type: "conv",
                        inputChannels: 192,
                        outputChannels: 64,
                        kernel: 1
                    }),
                    new Layer({
                        name: "branch2_reduce",
                        type: "conv",
                        inputChannels: 192,
                        outputChannels: 96,
                        kernel: 1
                    }),
                    new Layer({
                        name: "branch2_conv",
                        type: "conv",
                        inputChannels: 96,
                        outputChannels: 128,
                        kernel: 3,
                        padding: 1
                    }),
                    new Layer({
                        name: "branch3_reduce",
                        type: "conv",
                        inputChannels: 192,
                        outputChannels: 16,
                        kernel: 1
                    }),
                    new Layer({
                        name: "branch3_conv",
                        type: "conv",
                        inputChannels: 16,
                        outputChannels: 32,
                        kernel: 5,
                        padding: 2
                    }),
                    new Layer({
                        name: "branch4_projection",
                        type: "conv",
                        inputChannels: 192,
                        outputChannels: 32,
                        kernel: 1
                    })
                ]
            })
        ]
    ]);
}

class ArchitectureRegistry {
    constructor() {
        this.definitions = makeDefinitions();
    }

    get(type) {
        const definition = this.definitions.get(type);

        if (!definition) {
            throw new ArchitectureError(
                `Unknown architecture: ${type}`
            );
        }

        return definition;
    }

    list() {
        return [...this.definitions.values()];
    }
}

class ArchitectureAnalyzer {
    static analyze(definition, input, classes = 10) {
        if (!(input instanceof TensorShape)) {
            throw new ArchitectureError("Invalid input tensor");
        }

        if (input.height < definition.minimumInput ||
            input.width < definition.minimumInput) {
            throw new ArchitectureError(
                `${definition.name} requires dimensions of at least ` +
                `${definition.minimumInput}`
            );
        }

        if (input.channels !== definition.channels) {
            throw new ArchitectureError(
                `${definition.name} expects ${definition.channels} channels`
            );
        }

        if (!Number.isSafeInteger(classes) || classes < 2) {
            throw new ArchitectureError("At least two classes are required");
        }

        let totalParameters = 0;
        let currentChannels = input.channels;
        const layerReport = [];

        for (const layer of definition.layers) {
            if (layer.inputChannels !== currentChannels) {
                throw new ArchitectureError(
                    `Channel mismatch at ${layer.name}: ` +
                    `${currentChannels} != ${layer.inputChannels}`
                );
            }

            totalParameters += convolutionParameters(
                layer.inputChannels,
                layer.outputChannels,
                layer.kernel
            );

            currentChannels = layer.outputChannels;

            layerReport.push({
                name: layer.name,
                type: layer.type,
                parameters: convolutionParameters(
                    layer.inputChannels,
                    layer.outputChannels,
                    layer.kernel
                )
            });
        }

        return Object.freeze({
            architecture: definition.name,
            input: input.toString(),
            classes,
            estimatedConvolutionParameters: totalParameters,
            layers: layerReport,
            innovations: definition.innovations
        });
    }
}

class Experiment {
    constructor({
        architecture,
        inputSize,
        channels,
        classes,
        batchSize,
        learningRate
    }) {
        this.id = crypto.randomUUID();
        this.architecture = architecture;
        this.inputSize = inputSize;
        this.channels = channels;
        this.classes = classes;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
        this.state = "CREATED";
        this.createdAt = new Date().toISOString();
    }

    transition(nextState) {
        const transitions = {
            CREATED: ["VALIDATED", "FAILED"],
            VALIDATED: ["RUNNING", "FAILED"],
            RUNNING: ["COMPLETED", "FAILED"],
            COMPLETED: [],
            FAILED: []
        };

        if (!transitions[this.state].includes(nextState)) {
            throw new ArchitectureError(
                `Invalid transition ${this.state} -> ${nextState}`
            );
        }

        this.state = nextState;
    }
}

class ExperimentService extends EventEmitter {
    constructor(registry) {
        super();
        this.registry = registry;
        this.experiments = new Map();
    }

    validate(experiment) {
        const definition = this.registry.get(experiment.architecture);

        if (!Number.isSafeInteger(experiment.inputSize) ||
            experiment.inputSize < definition.minimumInput) {
            throw new ArchitectureError("Invalid input dimensions");
        }

        if (experiment.channels !== definition.channels) {
            throw new ArchitectureError("Input channel mismatch");
        }

        if (!Number.isSafeInteger(experiment.classes) ||
            experiment.classes < 2) {
            throw new ArchitectureError("Invalid class count");
        }

        if (!Number.isSafeInteger(experiment.batchSize) ||
            experiment.batchSize < 1 ||
            experiment.batchSize > 1024) {
            throw new ArchitectureError("Invalid batch size");
        }

        if (!Number.isFinite(experiment.learningRate) ||
            experiment.learningRate <= 0 ||
            experiment.learningRate > 1) {
            throw new ArchitectureError("Invalid learning rate");
        }

        return true;
    }

    async submit(configuration) {
        const experiment = new Experiment(configuration);

        this.experiments.set(experiment.id, experiment);

        try {
            this.validate(experiment);
            experiment.transition("VALIDATED");

            this.emit("validated", {
                id: experiment.id,
                architecture: experiment.architecture
            });

            experiment.transition("RUNNING");

            this.emit("started", experiment.id);

            // Represents asynchronous experiment orchestration.
            await Promise.resolve();

            const definition = this.registry.get(
                experiment.architecture
            );

            const report = ArchitectureAnalyzer.analyze(
                definition,
                new TensorShape(
                    experiment.channels,
                    experiment.inputSize,
                    experiment.inputSize
                ),
                experiment.classes
            );

            experiment.transition("COMPLETED");

            this.emit("completed", {
                id: experiment.id,
                report
            });

            return {
                experiment,
                report
            };
        } catch (error) {
            if (!["COMPLETED", "FAILED"].includes(experiment.state)) {
                experiment.transition("FAILED");
            }

            this.emit("failed", {
                id: experiment.id,
                message: error.message
            });

            throw error;
        }
    }

    get(id) {
        return this.experiments.get(id) || null;
    }
}

async function main() {
    const registry = new ArchitectureRegistry();
    const service = new ExperimentService(registry);

    service.on("validated", event => {
        console.log("VALIDATED:", event);
    });

    service.on("started", id => {
        console.log("STARTED:", id);
    });

    service.on("completed", event => {
        console.log("COMPLETED:", event.id);
    });

    service.on("failed", event => {
        console.error("FAILED:", event);
    });

    console.log("\nCNN ARCHITECTURE REGISTRY");
    console.log("=========================");

    for (const definition of registry.list()) {
        console.log(
            `${definition.name} (${definition.year}) | ` +
            definition.parametersDescription
        );
    }

    const configurations = [
        {
            architecture: ArchitectureType.LENET,
            inputSize: 32,
            channels: 1,
            classes: 10,
            batchSize: 32,
            learningRate: 0.01
        },
        {
            architecture: ArchitectureType.ALEXNET,
            inputSize: 224,
            channels: 3,
            classes: 1000,
            batchSize: 16,
            learningRate: 0.001
        },
        {
            architecture: ArchitectureType.VGG16,
            inputSize: 224,
            channels: 3,
            classes: 1000,
            batchSize: 8,
            learningRate: 0.0001
        },
        {
            architecture: ArchitectureType.INCEPTION,
            inputSize: 224,
            channels: 3,
            classes: 1000,
            batchSize: 16,
            learningRate: 0.001
        }
    ];

    for (const configuration of configurations) {
        const result = await service.submit(configuration);

        console.log(
            `${result.report.architecture}: ` +
            `${result.report.estimatedConvolutionParameters.toLocaleString()} ` +
            "estimated convolution parameters"
        );
    }

    assert.equal(
        service.get("missing-experiment"),
        null
    );

    try {
        await service.submit({
            architecture: ArchitectureType.LENET,
            inputSize: 16,
            channels: 1,
            classes: 10,
            batchSize: 8,
            learningRate: 0.01
        });
    } catch (error) {
        assert.ok(error instanceof ArchitectureError);
        console.log("\nInvalid configuration rejected:", error.message);
    }

    console.log("\nAll architecture workflow checks completed.");
}

main().catch(error => {
    console.error("Application failure:", error);
    process.exitCode = 1;
});
