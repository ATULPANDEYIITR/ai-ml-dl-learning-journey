"use strict";

/**
 * Modern CNN Architecture Evaluation Engine.
 *
 * Node.js 18+
 *
 * Demonstrates:
 * - ResNet residual graph evaluation
 * - DenseNet feature concatenation
 * - EfficientNet compound scaling
 * - MBConv and squeeze-and-excitation
 * - Asynchronous experiment execution
 * - Event-driven experiment lifecycle
 * - Architecture validation and resource policies
 * - Concurrent experiment scheduling
 *
 * Run: node modern_cnn.js
 */

const fs = require("node:fs/promises");
const { EventEmitter } = require("node:events");

class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = "ValidationError";
    }
}

class ResourceLimitError extends Error {
    constructor(message) {
        super(message);
        this.name = "ResourceLimitError";
    }
}

const TYPES = Object.freeze({
    CONV: "CONV",
    DEPTHWISE: "DEPTHWISE",
    RESIDUAL: "RESIDUAL",
    CONCAT: "CONCAT",
    SE: "SE",
    POOL: "POOL",
    DENSE: "DENSE",
    NORMALIZATION: "NORMALIZATION"
});

function validatePositiveInteger(value, name) {
    if (!Number.isSafeInteger(value) || value <= 0) {
        throw new ValidationError(`${name} must be a positive integer.`);
    }
}

function tensor(channels, height, width) {
    validatePositiveInteger(channels, "channels");
    validatePositiveInteger(height, "height");
    validatePositiveInteger(width, "width");

    return Object.freeze({ channels, height, width });
}

function sameShape(a, b) {
    return (
        a.channels === b.channels &&
        a.height === b.height &&
        a.width === b.width
    );
}

function convolutionShape(input, kernel, stride = 1, padding = 0) {
    const height = Math.floor(
        (input.height + 2 * padding - kernel) / stride
    ) + 1;

    const width = Math.floor(
        (input.width + 2 * padding - kernel) / stride
    ) + 1;

    if (height <= 0 || width <= 0) {
        throw new ValidationError("Invalid convolution output shape.");
    }

    return tensor(input.channels, height, width);
}

class CNNGraph {
    constructor(name, family, input) {
        if (!name || !family) {
            throw new ValidationError("Architecture identity is required.");
        }

        this.name = name;
        this.family = family;
        this.input = input;
        this.nodes = [];
        this.current = input;
    }

    addNode(node) {
        if (node.input !== this.current) {
            throw new ValidationError(
                `Invalid graph connection at ${node.name}.`
            );
        }

        if (node.parameters < 0 || node.flops < 0) {
            throw new ValidationError("Negative computational cost.");
        }

        this.nodes.push(Object.freeze(node));
        this.current = node.output;
    }

    convolution(name, outputChannels, kernel, stride = 1, groups = 1) {
        validatePositiveInteger(outputChannels, "outputChannels");
        validatePositiveInteger(kernel, "kernel");
        validatePositiveInteger(stride, "stride");
        validatePositiveInteger(groups, "groups");

        const input = this.current;

        if (
            input.channels % groups !== 0 ||
            outputChannels % groups !== 0
        ) {
            throw new ValidationError("Invalid grouped convolution.");
        }

        const padding = Math.floor(kernel / 2);
        const dimensions = convolutionShape(
            input, kernel, stride, padding
        );

        const output = tensor(
            outputChannels,
            dimensions.height,
            dimensions.width
        );

        const parameters =
            outputChannels *
            (input.channels / groups) *
            kernel * kernel;

        const flops =
            2 * output.elements *
            (input.channels / groups) *
            kernel * kernel;

        this.addNode({
            name,
            type: groups === input.channels
                ? TYPES.DEPTHWISE
                : TYPES.CONV,
            input,
            output,
            parameters,
            flops
        });
    }

    normalize(name) {
        const input = this.current;
        const parameters = 2 * input.channels;

        this.addNode({
            name,
            type: TYPES.NORMALIZATION,
            input,
            output: input,
            parameters,
            flops: 4 * input.channels * input.height * input.width
        });
    }

    pool(name, kernel = 2, stride = 2) {
        const input = this.current;
        const dimensions = convolutionShape(
            input, kernel, stride, 0
        );
        const output = tensor(
            input.channels,
            dimensions.height,
            dimensions.width
        );

        this.addNode({
            name,
            type: TYPES.POOL,
            input,
            output,
            parameters: 0,
            flops: output.channels *
                output.height * output.width * kernel * kernel
        });
    }

    activation(name) {
        const input = this.current;
        this.addNode({
            name,
            type: "ACTIVATION",
            input,
            output: input,
            parameters: 0,
            flops: input.channels * input.height * input.width
        });
    }

    residual(name, shortcut) {
        const input = this.current;

        if (!sameShape(input, shortcut)) {
            throw new ValidationError(
                `${name}: residual operands have different dimensions.`
            );
        }

        this.addNode({
            name,
            type: TYPES.RESIDUAL,
            input,
            output: input,
            parameters: 0,
            flops: input.channels * input.height * input.width
        });
    }

    concatenate(name, previous, generated) {
        if (
            previous.height !== generated.height ||
            previous.width !== generated.width
        ) {
            throw new ValidationError(
                "Dense concatenation requires matching spatial dimensions."
            );
        }

        const output = tensor(
            previous.channels + generated.channels,
            previous.height,
            previous.width
        );

        this.addNode({
            name,
            type: TYPES.CONCAT,
            input: previous,
            output,
            parameters: 0,
            flops: output.channels * output.height * output.width
        });
    }

    globalAveragePooling(name) {
        const input = this.current;
        const output = tensor(input.channels, 1, 1);

        this.addNode({
            name,
            type: TYPES.POOL,
            input,
            output,
            parameters: 0,
            flops: input.channels * input.height * input.width
        });
    }

    classifier(name, classes) {
        validatePositiveInteger(classes, "classes");

        const input = this.current;
        const output = tensor(classes, 1, 1);
        const parameters = input.channels * classes + classes;

        this.addNode({
            name,
            type: TYPES.DENSE,
            input,
            output,
            parameters,
            flops: 2 * input.channels * classes
        });
    }

    squeezeExcitation(name, reduction = 4) {
        const input = this.current;
        const reduced = Math.max(
            1, Math.floor(input.channels / reduction)
        );

        const parameters =
            input.channels * reduced + reduced +
            reduced * input.channels + input.channels;

        this.addNode({
            name,
            type: TYPES.SE,
            input,
            output: input,
            parameters,
            flops: 4 * input.channels * reduced +
                input.channels * input.height * input.width
        });
    }

    get totals() {
        return {
            parameters: this.nodes.reduce(
                (sum, node) => sum + node.parameters, 0
            ),
            flops: this.nodes.reduce(
                (sum, node) => sum + node.flops, 0
            ),
            layers: this.nodes.length,
            output: this.current
        };
    }

    serialize() {
        return {
            name: this.name,
            family: this.family,
            input: this.input,
            totals: this.totals,
            nodes: this.nodes
        };
    }
}

function resnetBasicBlock(graph, name, channels, stride = 1) {
    const shortcut = graph.current;

    graph.convolution(`${name}.conv1`, channels, 3, stride);
    graph.normalize(`${name}.bn1`);
    graph.activation(`${name}.relu1`);

    graph.convolution(`${name}.conv2`, channels, 3);
    graph.normalize(`${name}.bn2`);

    if (
        shortcut.channels !== graph.current.channels ||
        shortcut.height !== graph.current.height ||
        shortcut.width !== graph.current.width
    ) {
        const projectionInput = shortcut;
        const projection = tensor(
            graph.current.channels,
            graph.current.height,
            graph.current.width
        );

        const parameters =
            projectionInput.channels * projection.channels;

        graph.addNode({
            name: `${name}.projection`,
            type: TYPES.CONV,
            input: graph.current,
            output: projection,
            parameters,
            flops: 2 * parameters * projection.height * projection.width
        });
    }

    graph.residual(`${name}.add`, shortcut);
    graph.activation(`${name}.output`);
}

function buildResNet18(size = 224, classes = 1000) {
    const graph = new CNNGraph(
        "ResNet-18",
        "Residual Network",
        tensor(3, size, size)
    );

    graph.convolution("stem", 64, 7, 2);
    graph.normalize("stem.bn");
    graph.activation("stem.relu");
    graph.pool("stem.maxpool", 3, 2);

    const stages = [64, 128, 256, 512];

    stages.forEach((channels, stage) => {
        for (let block = 0; block < 2; block++) {
            resnetBasicBlock(
                graph,
                `stage${stage + 1}.block${block + 1}`,
                channels,
                stage > 0 && block === 0 ? 2 : 1
            );
        }
    });

    graph.globalAveragePooling("global_average");
    graph.classifier("classification_head", classes);

    return graph;
}

function buildDenseNet121(size = 224, classes = 1000) {
    const graph = new CNNGraph(
        "DenseNet-121",
        "Dense Connectivity",
        tensor(3, size, size)
    );

    graph.convolution("stem", 64, 7, 2);
    graph.normalize("stem.bn");
    graph.pool("stem.pool", 3, 2);

    const blockSizes = [6, 12, 24, 16];
    const growthRate = 32;

    blockSizes.forEach((count, blockIndex) => {
        for (let index = 0; index < count; index++) {
            const previous = graph.current;

            graph.normalize(`b${blockIndex}.l${index}.bn1`);
            graph.convolution(
                `b${blockIndex}.l${index}.bottleneck`,
                growthRate * 4,
                1
            );

            graph.normalize(`b${blockIndex}.l${index}.bn2`);
            graph.convolution(
                `b${blockIndex}.l${index}.growth`,
                growthRate,
                3
            );

            const generated = graph.current;

            graph.concatenate(
                `b${blockIndex}.l${index}.concat`,
                previous,
                generated
            );
        }

        if (blockIndex < 3) {
            const compressed = Math.max(
                1, Math.floor(graph.current.channels * 0.5)
            );

            graph.convolution(
                `transition${blockIndex}.conv`,
                compressed,
                1
            );

            graph.pool(
                `transition${blockIndex}.pool`,
                2,
                2
            );
        }
    });

    graph.globalAveragePooling("global_average");
    graph.classifier("classification_head", classes);

    return graph;
}

const EFFICIENTNET = Object.freeze({
    B0: { width: 1.0, depth: 1.0, resolution: 224 },
    B1: { width: 1.0, depth: 1.1, resolution: 240 },
    B2: { width: 1.1, depth: 1.2, resolution: 260 },
    B3: { width: 1.2, depth: 1.4, resolution: 300 },
    B4: { width: 1.4, depth: 1.8, resolution: 380 }
});

function roundChannels(value) {
    const divisor = 8;
    const rounded = Math.max(
        divisor,
        Math.floor((value + divisor / 2) / divisor) * divisor
    );

    return rounded < 0.9 * value ? rounded + divisor : rounded;
}

function buildEfficientNet(variant = "B0", classes = 1000) {
    const scale = EFFICIENTNET[variant];

    if (!scale) {
        throw new ValidationError("Unsupported EfficientNet variant.");
    }

    const graph = new CNNGraph(
        `EfficientNet-${variant}`,
        "Compound-Scaled CNN",
        tensor(3, scale.resolution, scale.resolution)
    );

    graph.convolution(
        "stem",
        roundChannels(32 * scale.width),
        3,
        2
    );

    graph.normalize("stem.bn");

    const stages = [
        [1, 16, 1, 3, 1],
        [6, 24, 2, 3, 2],
        [6, 40, 2, 5, 2],
        [6, 80, 3, 3, 2],
        [6, 112, 3, 5, 1],
        [6, 192, 4, 5, 2],
        [6, 320, 1, 3, 1]
    ];

    stages.forEach(
        ([expansion, baseChannels, repeats, kernel, stride], index) => {
            const channels = roundChannels(baseChannels * scale.width);
            const count = Math.ceil(repeats * scale.depth);

            for (let block = 0; block < count; block++) {
                const shortcut = graph.current;
                const expanded = shortcut.channels * expansion;

                if (expansion !== 1) {
                    graph.convolution(
                        `s${index}.b${block}.expand`,
                        expanded,
                        1
                    );
                }

                graph.convolution(
                    `s${index}.b${block}.depthwise`,
                    expanded,
                    kernel,
                    block === 0 ? stride : 1,
                    expanded
                );

                graph.normalize(`s${index}.b${block}.bn`);
                graph.squeezeExcitation(`s${index}.b${block}.se`);

                graph.convolution(
                    `s${index}.b${block}.project`,
                    channels,
                    1
                );

                if (
                    stride === 1 &&
                    block > 0 &&
                    shortcut.channels === channels &&
                    shortcut.height === graph.current.height &&
                    shortcut.width === graph.current.width
                ) {
                    graph.residual(
                        `s${index}.b${block}.skip`,
                        shortcut
                    );
                }
            }
        }
    );

    graph.convolution(
        "head",
        roundChannels(1280 * scale.width),
        1
    );

    graph.globalAveragePooling("global_average");
    graph.classifier("classification_head", classes);

    return graph;
}

class Experiment extends EventEmitter {
    constructor(id, architecture, limits) {
        super();

        if (!id || !architecture) {
            throw new ValidationError("Experiment identity is required.");
        }

        this.id = id;
        this.architecture = architecture;
        this.limits = limits;
        this.state = "CREATED";
        this.result = null;
        this.error = null;
    }

    transition(next) {
        const allowed = {
            CREATED: ["VALIDATING", "CANCELLED"],
            VALIDATING: ["RUNNING", "FAILED", "CANCELLED"],
            RUNNING: ["COMPLETED", "FAILED", "CANCELLED"],
            COMPLETED: [],
            FAILED: [],
            CANCELLED: []
        };

        if (!allowed[this.state].includes(next)) {
            throw new ValidationError(
                `Illegal transition ${this.state} -> ${next}`
            );
        }

        const previous = this.state;
        this.state = next;
        this.emit("stateChanged", {
            id: this.id,
            previous,
            current: next
        });
    }

    validate() {
        const totals = this.architecture.totals;

        if (totals.parameters > this.limits.maxParameters) {
            throw new ResourceLimitError("Parameter budget exceeded.");
        }

        if (totals.flops > this.limits.maxFlops) {
            throw new ResourceLimitError("FLOP budget exceeded.");
        }

        if (totals.layers === 0) {
            throw new ValidationError("Empty architecture.");
        }

        return totals;
    }

    async execute() {
        try {
            this.transition("VALIDATING");
            const totals = this.validate();

            this.transition("RUNNING");

            // Simulates asynchronous experiment orchestration.
            await new Promise(resolve => setImmediate(resolve));

            this.result = {
                experimentId: this.id,
                architecture: this.architecture.name,
                parameters: totals.parameters,
                flops: totals.flops,
                layers: totals.layers,
                output: totals.output
            };

            this.transition("COMPLETED");
            this.emit("completed", this.result);

            return this.result;
        } catch (error) {
            this.error = error.message;

            if (
                this.state === "VALIDATING" ||
                this.state === "RUNNING"
            ) {
                this.transition("FAILED");
            }

            this.emit("failed", {
                id: this.id,
                error: error.message
            });

            throw error;
        }
    }
}

class ExperimentScheduler {
    constructor(concurrency = 2) {
        validatePositiveInteger(concurrency, "concurrency");
        this.concurrency = concurrency;
        this.queue = [];
        this.running = 0;
    }

    submit(experiment) {
        return new Promise((resolve, reject) => {
            this.queue.push({ experiment, resolve, reject });
            this.process();
        });
    }

    process() {
        while (
            this.running < this.concurrency &&
            this.queue.length > 0
        ) {
            const job = this.queue.shift();
            this.running++;

            job.experiment.execute()
                .then(job.resolve)
                .catch(job.reject)
                .finally(() => {
                    this.running--;
                    this.process();
                });
        }
    }
}

async function main() {
    const architectures = [
        buildResNet18(),
        buildDenseNet121(),
        buildEfficientNet("B0"),
        buildEfficientNet("B2")
    ];

    const scheduler = new ExperimentScheduler(2);
    const events = [];

    const jobs = architectures.map((architecture, index) => {
        const experiment = new Experiment(
            `EXP-${index + 1}`,
            architecture,
            {
                maxParameters: 100_000_000,
                maxFlops: 100_000_000_000
            }
        );

        experiment.on("stateChanged", event => {
            events.push(event);
        });

        experiment.on("failed", event => {
            process.stderr.write(
                `Experiment ${event.id} failed: ${event.error}\n`
            );
        });

        return scheduler.submit(experiment);
    });

    const results = await Promise.allSettled(jobs);

    const report = {
        generatedAt: new Date().toISOString(),
        successful: results
            .filter(result => result.status === "fulfilled")
            .map(result => result.value),
        failures: results
            .filter(result => result.status === "rejected")
            .map(result => result.reason.message),
        lifecycleEvents: events
    };

    await fs.writeFile(
        "modern_cnn_experiments.json",
        JSON.stringify(report, null, 2),
        { encoding: "utf8", mode: 0o600 }
    );

    console.table(
        report.successful.map(result => ({
            architecture: result.architecture,
            parameters: result.parameters,
            GFLOPs: (result.flops / 1e9).toFixed(3),
            layers: result.layers
        }))
    );

    console.log("Report saved to modern_cnn_experiments.json");
}

function runTests() {
    const shape = convolutionShape(
        tensor(3, 224, 224), 7, 2, 3
    );

    if (shape.height !== 112 || shape.width !== 112) {
        throw new Error("Convolution shape test failed.");
    }

    const residual = buildResNet18(64, 10);
    const dense = buildDenseNet121(64, 10);
    const efficient = buildEfficientNet("B0", 10);

    for (const model of [residual, dense, efficient]) {
        if (model.totals.parameters <= 0) {
            throw new Error("Invalid parameter count.");
        }

        if (model.totals.flops <= 0) {
            throw new Error("Invalid FLOP count.");
        }
    }

    try {
        buildEfficientNet("B99");
        throw new Error("Invalid variant was accepted.");
    } catch (error) {
        if (!(error instanceof ValidationError)) {
            throw error;
        }
    }

    console.log("All JavaScript tests passed.");
}

if (require.main === module) {
    runTests();
    main().catch(error => {
        console.error("Execution error:", error.message);
        process.exitCode = 1;
    });
}

module.exports = {
    CNNGraph,
    Experiment,
    ExperimentScheduler,
    buildResNet18,
    buildDenseNet121,
    buildEfficientNet
};
