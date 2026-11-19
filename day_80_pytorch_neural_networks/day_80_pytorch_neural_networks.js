"use strict";

/*
 * Neural Networks: Modules, Layers, Forward Propagation,
 * Backpropagation, and Training Loops.
 *
 * Runtime: Node.js 18+
 * Dependencies: Node.js standard library only.
 *
 * Implements a trainable multilayer perceptron using
 * explicit matrix operations and event-driven training.
 */

const fs = require("node:fs/promises");
const path = require("node:path");
const { EventEmitter } = require("node:events");

// ---------------------------------------------------------
// 1. Validation and numerical utilities
// ---------------------------------------------------------

function assertNumber(value, name) {
    if (!Number.isFinite(value)) {
        throw new TypeError(`${name} must be finite.`);
    }
}

function validateVector(vector, expectedLength, name) {
    if (!Array.isArray(vector) || vector.length !== expectedLength) {
        throw new TypeError(`${name} has an invalid dimension.`);
    }

    vector.forEach((value, index) =>
        assertNumber(value, `${name}[${index}]`)
    );
}

function zeros(size) {
    return Array(size).fill(0);
}

function dot(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vector dimensions do not match.");
    }

    return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function argmax(values) {
    return values.reduce(
        (best, value, index) =>
            value > values[best] ? index : best,
        0
    );
}

function softmax(logits) {
    const maximum = Math.max(...logits);

    const exponentials = logits.map(
        value => Math.exp(value - maximum)
    );

    const denominator = exponentials.reduce(
        (sum, value) => sum + value,
        0
    );

    if (!Number.isFinite(denominator) || denominator <= 0) {
        throw new Error("Softmax normalization failed.");
    }

    return exponentials.map(value => value / denominator);
}

// ---------------------------------------------------------
// 2. Deterministic random number generator
// ---------------------------------------------------------

class RandomGenerator {
    constructor(seed = 42) {
        this.state = seed >>> 0;
    }

    next() {
        this.state = (
            Math.imul(1664525, this.state) + 1013904223
        ) >>> 0;

        return this.state / 4294967296;
    }

    normal() {
        const u1 = Math.max(this.next(), Number.EPSILON);
        const u2 = this.next();

        return Math.sqrt(-2 * Math.log(u1)) *
            Math.cos(2 * Math.PI * u2);
    }
}

// ---------------------------------------------------------
// 3. Layer abstraction
// ---------------------------------------------------------

class LinearLayer {
    constructor(inputSize, outputSize, random) {
        if (
            !Number.isInteger(inputSize) ||
            !Number.isInteger(outputSize) ||
            inputSize <= 0 ||
            outputSize <= 0
        ) {
            throw new RangeError("Invalid layer dimensions.");
        }

        this.inputSize = inputSize;
        this.outputSize = outputSize;

        // He initialization for ReLU-based networks.
        const scale = Math.sqrt(2 / inputSize);

        this.weights = Array.from(
            { length: outputSize },
            () => Array.from(
                { length: inputSize },
                () => random.normal() * scale
            )
        );

        this.biases = zeros(outputSize);

        this.lastInput = null;
        this.lastOutput = null;
    }

    forward(input) {
        validateVector(input, this.inputSize, "Layer input");

        this.lastInput = [...input];

        this.lastOutput = this.weights.map(
            (row, index) => dot(row, input) + this.biases[index]
        );

        return [...this.lastOutput];
    }

    backward(outputGradient) {
        validateVector(
            outputGradient,
            this.outputSize,
            "Output gradient"
        );

        if (this.lastInput === null) {
            throw new Error("Forward must run before backward.");
        }

        const inputGradient = zeros(this.inputSize);

        const weightGradients = this.weights.map(
            () => zeros(this.inputSize)
        );

        const biasGradients = zeros(this.outputSize);

        for (let output = 0; output < this.outputSize; output++) {
            const gradient = outputGradient[output];

            biasGradients[output] = gradient;

            for (let input = 0; input < this.inputSize; input++) {
                weightGradients[output][input] =
                    gradient * this.lastInput[input];

                inputGradient[input] +=
                    this.weights[output][input] * gradient;
            }
        }

        return {
            inputGradient,
            weightGradients,
            biasGradients
        };
    }

    update(gradients, learningRate) {
        for (let output = 0; output < this.outputSize; output++) {
            this.biases[output] -=
                learningRate * gradients.biasGradients[output];

            for (let input = 0; input < this.inputSize; input++) {
                this.weights[output][input] -=
                    learningRate *
                    gradients.weightGradients[output][input];
            }
        }
    }

    exportState() {
        return {
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            weights: this.weights,
            biases: this.biases
        };
    }
}

// ---------------------------------------------------------
// 4. Activation module
// ---------------------------------------------------------

class ReLULayer {
    forward(input) {
        this.mask = input.map(value => value > 0);

        return input.map(value => Math.max(0, value));
    }

    backward(gradient) {
        if (!this.mask) {
            throw new Error("ReLU forward pass is missing.");
        }

        return gradient.map(
            (value, index) => this.mask[index] ? value : 0
        );
    }
}

// ---------------------------------------------------------
// 5. Network module
// ---------------------------------------------------------

class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize, seed = 42) {
        const random = new RandomGenerator(seed);

        this.layers = [
            new LinearLayer(inputSize, hiddenSize, random),
            new ReLULayer(),
            new LinearLayer(hiddenSize, hiddenSize, random),
            new ReLULayer(),
            new LinearLayer(hiddenSize, outputSize, random)
        ];

        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    forward(input) {
        let output = input;

        for (const layer of this.layers) {
            output = layer.forward(output);
        }

        return output;
    }

    backward(gradient) {
        let current = gradient;

        for (let index = this.layers.length - 1; index >= 0; index--) {
            const layer = this.layers[index];

            if (layer instanceof LinearLayer) {
                const result = layer.backward(current);

                layer.pendingGradients = result;

                current = result.inputGradient;
            } else {
                current = layer.backward(current);
            }
        }

        return current;
    }

    update(learningRate) {
        for (const layer of this.layers) {
            if (layer instanceof LinearLayer) {
                if (!layer.pendingGradients) {
                    throw new Error("Missing layer gradients.");
                }

                layer.update(
                    layer.pendingGradients,
                    learningRate
                );

                layer.pendingGradients = null;
            }
        }
    }

    predict(input) {
        return argmax(softmax(this.forward(input)));
    }

    exportState() {
        return this.layers
            .filter(layer => layer instanceof LinearLayer)
            .map(layer => layer.exportState());
    }
}

// ---------------------------------------------------------
// 6. Loss function
// ---------------------------------------------------------

function crossEntropyLoss(logits, target) {
    if (
        !Number.isInteger(target) ||
        target < 0 ||
        target >= logits.length
    ) {
        throw new RangeError("Invalid classification target.");
    }

    const probabilities = softmax(logits);

    const loss = -Math.log(
        Math.max(probabilities[target], Number.MIN_VALUE)
    );

    const gradient = probabilities.map(
        (probability, index) =>
            probability - (index === target ? 1 : 0)
    );

    return { loss, gradient };
}

// ---------------------------------------------------------
// 7. Dataset generation
// ---------------------------------------------------------

function generateDataset(samplesPerClass = 300, seed = 42) {
    const random = new RandomGenerator(seed);
    const dataset = [];

    for (let i = 0; i < samplesPerClass; i++) {
        dataset.push({
            x: [
                random.normal() * 0.65 - 1,
                random.normal() * 0.65 - 1
            ],
            y: 0
        });

        dataset.push({
            x: [
                random.normal() * 0.65 + 1,
                random.normal() * 0.65 + 1
            ],
            y: 1
        });
    }

    return dataset;
}

function shuffle(dataset, random) {
    for (let i = dataset.length - 1; i > 0; i--) {
        const j = Math.floor(random.next() * (i + 1));

        [dataset[i], dataset[j]] = [dataset[j], dataset[i]];
    }

    return dataset;
}

// ---------------------------------------------------------
// 8. Event-driven training engine
// ---------------------------------------------------------

class TrainingEngine extends EventEmitter {
    constructor(model, options = {}) {
        super();

        this.model = model;

        this.learningRate = options.learningRate ?? 0.01;
        this.epochs = options.epochs ?? 100;
        this.batchSize = options.batchSize ?? 16;
        this.gradientClip = options.gradientClip ?? 5;

        this.validateOptions();
    }

    validateOptions() {
        for (const [name, value] of Object.entries({
            learningRate: this.learningRate,
            epochs: this.epochs,
            batchSize: this.batchSize,
            gradientClip: this.gradientClip
        })) {
            if (!Number.isFinite(value) || value <= 0) {
                throw new RangeError(`${name} must be positive.`);
            }
        }

        if (
            !Number.isInteger(this.epochs) ||
            !Number.isInteger(this.batchSize)
        ) {
            throw new TypeError("Epochs and batch size must be integers.");
        }
    }

    trainSample(sample) {
        const logits = this.model.forward(sample.x);

        const result = crossEntropyLoss(logits, sample.y);

        const norm = Math.sqrt(
            dot(result.gradient, result.gradient)
        );

        if (norm > this.gradientClip) {
            result.gradient = result.gradient.map(
                value => value * this.gradientClip / norm
            );
        }

        this.model.backward(result.gradient);
        this.model.update(this.learningRate);

        return {
            loss: result.loss,
            correct: argmax(logits) === sample.y
        };
    }

    async train(dataset) {
        if (!Array.isArray(dataset) || dataset.length === 0) {
            throw new Error("Training dataset cannot be empty.");
        }

        for (const sample of dataset) {
            validateVector(sample.x, this.model.inputSize, "Sample");
        }

        const random = new RandomGenerator(99);
        const history = [];

        for (let epoch = 1; epoch <= this.epochs; epoch++) {
            shuffle(dataset, random);

            let totalLoss = 0;
            let correct = 0;

            for (
                let start = 0;
                start < dataset.length;
                start += this.batchSize
            ) {
                const batch = dataset.slice(
                    start,
                    start + this.batchSize
                );

                // Each sample updates parameters immediately.
                // This is stochastic training with small batches.
                for (const sample of batch) {
                    const result = this.trainSample(sample);

                    totalLoss += result.loss;

                    if (result.correct) {
                        correct++;
                    }
                }
            }

            const metrics = {
                epoch,
                loss: totalLoss / dataset.length,
                accuracy: correct / dataset.length
            };

            history.push(metrics);

            this.emit("epoch", metrics);

            await new Promise(resolve => setImmediate(resolve));
        }

        this.emit("completed", history);

        return history;
    }
}

// ---------------------------------------------------------
// 9. Validation dataset evaluation
// ---------------------------------------------------------

function evaluate(model, dataset) {
    if (!dataset.length) {
        throw new Error("Validation dataset cannot be empty.");
    }

    let loss = 0;
    let correct = 0;

    for (const sample of dataset) {
        const logits = model.forward(sample.x);

        loss += crossEntropyLoss(logits, sample.y).loss;

        if (argmax(logits) === sample.y) {
            correct++;
        }
    }

    return {
        loss: loss / dataset.length,
        accuracy: correct / dataset.length
    };
}

// ---------------------------------------------------------
// 10. Checkpoint persistence
// ---------------------------------------------------------

async function saveCheckpoint(model, filename) {
    const absolutePath = path.resolve(filename);

    const payload = JSON.stringify({
        formatVersion: 1,
        architecture: {
            inputSize: model.inputSize,
            outputSize: model.outputSize
        },
        layers: model.exportState()
    });

    const temporaryPath = `${absolutePath}.tmp`;

    try {
        await fs.writeFile(
            temporaryPath,
            payload,
            { encoding: "utf8", flag: "w" }
        );

        await fs.rename(temporaryPath, absolutePath);

    } catch (error) {
        await fs.rm(temporaryPath, { force: true }).catch(() => {});
        throw error;
    }

    return absolutePath;
}

// ---------------------------------------------------------
// 11. Main execution
// ---------------------------------------------------------

async function main() {
    const dataset = generateDataset();

    const split = Math.floor(dataset.length * 0.8);

    const trainingData = dataset.slice(0, split);
    const validationData = dataset.slice(split);

    const model = new NeuralNetwork(2, 32, 2, 42);

    const trainer = new TrainingEngine(model, {
        learningRate: 0.01,
        epochs: 100,
        batchSize: 16,
        gradientClip: 5
    });

    trainer.on("epoch", metrics => {
        if (
            metrics.epoch === 1 ||
            metrics.epoch % 10 === 0
        ) {
            console.log(
                `Epoch ${metrics.epoch}: ` +
                `Loss=${metrics.loss.toFixed(4)}, ` +
                `Accuracy=${metrics.accuracy.toFixed(4)}`
            );
        }
    });

    trainer.on("completed", () => {
        console.log("Training completed.");
    });

    await trainer.train(trainingData);

    const metrics = evaluate(model, validationData);

    console.log("\nVALIDATION");
    console.log(metrics);

    console.log("\nINFERENCE");

    for (const input of [
        [-1.2, -0.8],
        [1.1, 0.9],
        [0.0, 0.0]
    ]) {
        console.log({
            input,
            predictedClass: model.predict(input)
        });
    }

    const checkpoint = await saveCheckpoint(
        model,
        "neural_network_js_checkpoint.json"
    );

    console.log(`Checkpoint saved: ${checkpoint}`);

    if (metrics.accuracy < 0.65) {
        throw new Error("Validation accuracy is below the minimum threshold.");
    }
}

main().catch(error => {
    console.error("Execution failed:", error.message);
    process.exitCode = 1;
});
